#!/usr/bin/env python3
"""Worker-side cell executor for the BO-tuning harness.

Runs one (method, bench, seed, config) cell through `scripts/bench_run.py` and writes a
classified result JSON (status ok | diverged | infra_failure | structurally_excluded).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BENCH_RUN = _REPO_ROOT / "scripts" / "bench_run.py"

# alm_bolton cells accept only projector-knob overrides; ALM hparams come from the frozen winner.
_BOLTON_PROJ_FLOAT_KEYS = {"proj_delta", "proj_tol", "proj_lambda_min",
                           "proj_prescale_floor", "eps_active"}
_BOLTON_PROJ_INT_KEYS = {"proj_max_iters"}
_BOLTON_PROJ_KEYS = _BOLTON_PROJ_FLOAT_KEYS | _BOLTON_PROJ_INT_KEYS

_METHODS = {
    "pal_loggap",
    "alm",
    "alm_bolton",
    "dc3",
    "fsnet",
    "enforce_orig",
    "enforce_v4",
    "snarenet",
}

# DC3's completion does not apply when n_eq > dim, so this pair is refused without running.
_STRUCTURALLY_EXCLUDED: set[tuple[str, str]] = {("dc3", "s5_overdetermined")}

_STDERR_TAIL_LINES = 60


class CellExecutorError(RuntimeError):
    """Raised for caller/usage errors (bad method name, etc.)."""


def is_structurally_excluded(method: str, bench: str) -> bool:
    return (method, bench) in _STRUCTURALLY_EXCLUDED


def _git_sha(repo: Path) -> str:
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def resolve_pal_file(python_bin: str, repo_root: Path = _REPO_ROOT) -> str:
    """Resolve `pal.__file__` exactly the way the subprocess would.

    Mirrors `scripts/bench_run.py`'s `sys.path.insert(0, _REPO_ROOT)`.
    """
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repo_root)!r}); "
        "import pal; "
        "print(pal.__file__)"
    )
    out = subprocess.check_output(
        [python_bin, "-c", code],
        cwd=str(repo_root),
        stderr=subprocess.STDOUT,
    )
    return out.decode().strip()


def parse_set_overrides(pairs: list[str]) -> dict[str, str]:
    """Parse `KEY=VALUE` strings into a dict, last-wins on duplicate keys."""
    resolved: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise CellExecutorError(f"--set expects KEY=VALUE, got: {pair!r}")
        key, _, val = pair.partition("=")
        resolved[key.strip()] = val.strip()
    return resolved


def build_command(
    *,
    python_bin: str,
    method: str,
    bench: str,
    seed: int,
    trial_dir: Path,
    set_overrides: dict[str, str],
    extra_args: list[str] | None = None,
    bench_run_path: Path = _BENCH_RUN,
) -> list[str]:
    """Build the exact argv for the training subprocess.

    The logical `lr` knob maps to `--set learning_rate=...` for snarenet (whose CLI
    ignores `--lr`) and to `--lr` for all other methods.
    """
    if method not in _METHODS:
        raise CellExecutorError(f"unknown method {method!r}; supported: {sorted(_METHODS)}")

    cmd = [
        python_bin,
        str(bench_run_path),
        "run",
        "--method", method,
        "--benchmarks", bench,
        "--seeds", str(seed),
        "--runs-root", str(trial_dir),
    ]

    generic_set_pairs: list[str] = []
    for key, val in set_overrides.items():
        if key == "lr":
            if method == "snarenet":
                generic_set_pairs.append(f"learning_rate={val}")
            else:
                cmd += ["--lr", str(val)]
        else:
            generic_set_pairs.append(f"{key}={val}")
    for pair in generic_set_pairs:
        cmd += ["--set", pair]

    if extra_args:
        cmd += list(extra_args)
    return cmd


def _snapshot_subdirs(trial_dir: Path) -> set[str]:
    if not trial_dir.exists():
        return set()
    return {p.name for p in trial_dir.iterdir() if p.is_dir()}


def _tail_lines(text: str, n: int = _STDERR_TAIL_LINES) -> str:
    lines = text.splitlines()
    return "\n".join(lines[-n:])


def _has_nonfinite(obj: Any) -> bool:
    """Recursively scan a JSON-decoded structure for NaN/inf leaves."""
    if isinstance(obj, bool):
        return False
    if isinstance(obj, (int, float)):
        return isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj))
    if isinstance(obj, dict):
        return any(_has_nonfinite(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_has_nonfinite(v) for v in obj)
    return False


# Deterministic divergences are scientific results (status "diverged"), not infra failures.
# Each signature is an exact phrase raised by a pal solver; a bare "nan" is never matched.
_DIVERGENCE_SIGNATURES: tuple[str, ...] = (
    "non-finite loss",
    "non-finite dc3 loss",
    "non-finite gradient",
    "non-finite residual",
    "non-finite newton step",
    "non-finite step",
    "newton diverged",
    "completiondivergederror",
    "loss=nan",
    # Deterministic linear-algebra breakdown (e.g. eigh on ill-conditioned Jacobians).
    "linalgerror",
    # enforce_v4 raises this when n_constraints > n_outputs (singular Gram matrix).
    "too many constraints",
)


def _divergence_signature(text: str | None) -> str | None:
    """Return the first stderr/error line matching a known divergence signature,
    or None. Case-insensitive; only the curated `_DIVERGENCE_SIGNATURES`."""
    if not text:
        return None
    low = text.lower()
    if not any(sig in low for sig in _DIVERGENCE_SIGNATURES):
        return None
    for line in text.splitlines():
        line_low = line.lower()
        if any(sig in line_low for sig in _DIVERGENCE_SIGNATURES):
            return line.strip()
    return None


def _failed_final_divergence(final_payload: Any) -> str | None:
    """If a `{"status":"failed"}` final.json's error is a divergence signature,
    return the matched line; else None."""
    if isinstance(final_payload, dict) and final_payload.get("status") == "failed":
        return _divergence_signature(str(final_payload.get("error") or ""))
    return None


def _read_run_final(run_dir: Path | None) -> dict[str, Any] | None:
    """Best-effort read of `<run_dir>/final.json` (None if absent/malformed).
    Used to inspect a crash-written failed-final even when rc != 0."""
    if run_dir is None:
        return None
    fjp = run_dir / "final.json"
    if not fjp.exists():
        return None
    try:
        data = json.loads(fjp.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, allow_nan=True)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


# alm_bolton predict-only cells: evaluate a frozen ALM checkpoint with candidate projector knobs.
def resolve_frozen_run_dir(frozen_alm_dir: str | Path, bench: str, seed: int) -> Path:
    """Deterministic frozen ALM run dir for a (bench, seed) cell.

    Layout contract (the controller lays the winner artifacts out this way):
        <frozen_alm_dir>/<bench>_seed<seed>/{config.json, model.pt, final.json}
    """
    return Path(frozen_alm_dir) / f"{bench}_seed{seed}"


def build_bolton_hparams(
    frozen_hparams: dict[str, Any], set_overrides: dict[str, str]
) -> dict[str, Any]:
    """Frozen ALM winner hparams + typed projector knob overrides.

    Only projector knobs (`proj_*`, `eps_active`) may be overridden; any other
    logical key is a caller bug (bolton's ALM-side hparams ARE the frozen
    winner's, by construction) and raises.
    """
    bolton = dict(frozen_hparams)
    for key, raw in set_overrides.items():
        if key in _BOLTON_PROJ_INT_KEYS:
            bolton[key] = int(float(raw))
        elif key in _BOLTON_PROJ_FLOAT_KEYS:
            bolton[key] = float(raw)
        else:
            raise CellExecutorError(
                f"alm_bolton predict-only cell got non-projector override "
                f"{key!r}={raw!r}; only {sorted(_BOLTON_PROJ_KEYS)} may be tuned "
                "(all ALM-side training hparams come from the frozen winner)."
            )
    return bolton


def assert_bolton_matches_frozen_alm(
    bolton_hparams: dict[str, Any],
    frozen_hparams: dict[str, Any],
    *,
    bench: str,
    seed: int,
) -> None:
    """PRELAUNCH ASSERT: every ALM-side field of the resolved bolton config
    equals the frozen ALM winner config. Hard-fail otherwise.
    """
    mismatches = []
    for key, frozen_val in frozen_hparams.items():
        if key in _BOLTON_PROJ_KEYS:
            continue  # projector knob, not an ALM-side training field
        bolton_val = bolton_hparams.get(key, _MISSING)
        if bolton_val != frozen_val:
            mismatches.append((key, frozen_val, bolton_val))
    if "grad_clip" not in frozen_hparams:
        raise CellExecutorError(
            f"frozen ALM winner ({bench} seed{seed}) config.json hparams has no "
            "'grad_clip', cannot verify ALM-side training parity; refusing."
        )
    if mismatches:
        detail = "; ".join(
            f"{k}: frozen={fv!r} bolton={bv!r}" for k, fv, bv in mismatches
        )
        raise CellExecutorError(
            f"alm_bolton prelaunch assert FAILED for {bench} seed{seed}: bolton "
            f"ALM-side config diverges from the frozen ALM winner ({detail}). "
            "Bolton training hyperparams must equal the ALM winner by "
            "construction (composition claim). Fix the frozen artifacts / config "
            "sync (e.g. grad_clip) before launching."
        )


_MISSING = object()


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def stage_bolton_run_dir(
    frozen_run: Path,
    staged_dir: Path,
    bolton_hparams: dict[str, Any],
    frozen_config: dict[str, Any],
) -> None:
    """Materialize a pal run dir that `pal eval` consumes as an alm_bolton run.

    model.pt is hardlinked. final.json is copied because `pal eval` overwrites it in place.
    """
    staged_dir.mkdir(parents=True, exist_ok=True)
    new_config = dict(frozen_config)
    new_config["method"] = "alm_bolton"
    new_config["hparams"] = bolton_hparams
    new_config["bolton_frozen_from"] = str(frozen_run)
    new_config["override_source_method"] = frozen_config.get("method")
    with (staged_dir / "config.json").open("w") as f:
        json.dump(new_config, f, indent=2, sort_keys=True)
    _link_or_copy(frozen_run / "model.pt", staged_dir / "model.pt")
    shutil.copy2(frozen_run / "final.json", staged_dir / "final.json")


def run_predict_cell(
    *,
    bench: str,
    seed: int,
    trial_dir: Path,
    frozen_alm_dir: str | Path,
    set_overrides: dict[str, str],
    timeout_s: float,
    attempt: int,
    python_bin: str,
    result_path: Path,
    repo_root: Path,
    git_sha: str,
) -> dict[str, Any]:
    """alm_bolton predict-only cell: load frozen ALM checkpoint, predict
    with candidate projector knobs, emit a training-cell-compatible result.json.
    """
    method = "alm_bolton"
    trial_dir = Path(trial_dir)
    trial_dir.mkdir(parents=True, exist_ok=True)
    frozen_run = resolve_frozen_run_dir(frozen_alm_dir, bench, seed)

    def _infra(reason: str, *, run_dir: Path | None = None,
               returncode: int | None = None, stderr_tail: str | None = None,
               wall: float = 0.0) -> dict[str, Any]:
        payload = {
            "status": "infra_failure",
            "reason": reason,
            "fingerprint": {
                "git_sha": git_sha, "pal_file": None,
                "method": method, "bench": bench, "seed": seed,
                "requested_overrides": dict(set_overrides),
                "command": None, "attempt": attempt, "wall_time_s": wall,
                "run_dir": str(run_dir) if run_dir else None,
                "frozen_run_dir": str(frozen_run),
            },
            "returncode": returncode, "stderr_tail": stderr_tail, "final": None,
        }
        _atomic_write_json(result_path, payload)
        return payload

    cfg_path = frozen_run / "config.json"
    if not cfg_path.exists() or not (frozen_run / "model.pt").exists() \
            or not (frozen_run / "final.json").exists():
        return _infra(
            f"frozen ALM artifacts incomplete under {frozen_run} (need "
            "config.json + model.pt + final.json)")
    try:
        frozen_config = json.loads(cfg_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return _infra(f"malformed frozen config.json: {type(e).__name__}: {e}")
    frozen_hparams = frozen_config.get("hparams") or {}
    if frozen_config.get("benchmark_id") != bench or int(frozen_config.get("seed", -1)) != int(seed):
        return _infra(
            f"frozen run identity mismatch: config=({frozen_config.get('benchmark_id')},"
            f"{frozen_config.get('seed')}) expected=({bench},{seed})")

    try:
        bolton_hparams = build_bolton_hparams(frozen_hparams, set_overrides)
        assert_bolton_matches_frozen_alm(
            bolton_hparams, frozen_hparams, bench=bench, seed=seed)
    except CellExecutorError:
        raise  # prelaunch invariant violation -> hard error, never a score

    # Stage a run dir and run `pal eval` (no training).
    staged_dir = trial_dir / "predict_run"
    stage_bolton_run_dir(frozen_run, staged_dir, bolton_hparams, frozen_config)

    try:
        pal_file = resolve_pal_file(python_bin, repo_root=repo_root)
    except Exception as e:  # noqa: BLE001
        pal_file = f"<unresolved: {type(e).__name__}: {e}>"

    cmd = [
        python_bin, str(_BENCH_RUN), "eval",
        "--run-id", staged_dir.name,
        "--runs-root", str(staged_dir.parent),
        "--device", "cpu",
    ]
    start = time.monotonic()
    timed_out = False
    stderr = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(cmd, cwd=str(repo_root), capture_output=True,
                              text=True, timeout=timeout_s)
        returncode, stderr = proc.returncode, proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stderr = e.stderr or ""
    wall = time.monotonic() - start

    if timed_out:
        return _infra(f"pal eval timed out after {timeout_s}s", run_dir=staged_dir,
                      stderr_tail=_tail_lines(stderr), wall=wall)
    if returncode != 0:
        return _infra(f"pal eval exited with returncode={returncode}",
                      run_dir=staged_dir, returncode=returncode,
                      stderr_tail=_tail_lines(stderr), wall=wall)

    final_json_path = staged_dir / "final.json"
    if not final_json_path.exists():
        return _infra("pal eval exited 0 but wrote no final.json", run_dir=staged_dir,
                      returncode=returncode, stderr_tail=_tail_lines(stderr), wall=wall)
    try:
        final_payload = json.loads(final_json_path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        return _infra(f"malformed eval final.json: {type(e).__name__}: {e}",
                      run_dir=staged_dir, returncode=returncode, wall=wall)

    if _has_nonfinite(final_payload):
        status, reason = "diverged", "eval final.json metrics contain NaN/inf"
    else:
        status, reason = "ok", None

    payload = {
        "status": status,
        "reason": reason,
        "fingerprint": {
            "git_sha": git_sha, "pal_file": pal_file,
            "method": method, "bench": bench, "seed": seed,
            "requested_overrides": dict(set_overrides),
            "command": cmd, "attempt": attempt, "wall_time_s": wall,
            "run_dir": str(staged_dir),
            "frozen_run_dir": str(frozen_run),
        },
        "returncode": returncode,
        "stderr_tail": None,
        "final": final_payload,
    }
    _atomic_write_json(result_path, payload)
    return payload


def run_cell(
    *,
    method: str,
    bench: str,
    seed: int,
    trial_dir: Path,
    set_overrides: dict[str, str] | None = None,
    timeout_s: float = 3600.0,
    attempt: int = 1,
    extra_args: list[str] | None = None,
    python_bin: str = sys.executable,
    result_path: Path | None = None,
    repo_root: Path = _REPO_ROOT,
    frozen_alm_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one cell and write (+ return) its classified result JSON.

    `result_path` defaults to `<trial_dir>/result.json`.

    When `frozen_alm_dir` is set (alm_bolton), the cell predicts from the frozen
    ALM winner checkpoint instead of training (see `run_predict_cell`).
    """
    trial_dir = Path(trial_dir)
    set_overrides = dict(set_overrides or {})
    result_path = Path(result_path) if result_path is not None else trial_dir / "result.json"

    git_sha = _git_sha(repo_root)

    if frozen_alm_dir is not None:
        if method != "alm_bolton":
            raise CellExecutorError(
                f"--frozen-alm-dir is only valid for method=alm_bolton, got {method!r}")
        return run_predict_cell(
            bench=bench, seed=seed, trial_dir=trial_dir,
            frozen_alm_dir=frozen_alm_dir, set_overrides=set_overrides,
            timeout_s=timeout_s, attempt=attempt, python_bin=python_bin,
            result_path=result_path, repo_root=repo_root, git_sha=git_sha,
        )

    if is_structurally_excluded(method, bench):
        payload = {
            "status": "structurally_excluded",
            "reason": (
                f"({method}, {bench}) is structurally excluded by cell policy "
                "(DC3's completion mechanism is inapplicable when n_eq > dim; "
                "see tests/test_solvers_smoke.py::test_dc3_smoke_s5_overdetermined). "
                "Executor refused without invoking the subprocess."
            ),
            "fingerprint": {
                "git_sha": git_sha,
                "pal_file": None,
                "method": method,
                "bench": bench,
                "seed": seed,
                "requested_overrides": set_overrides,
                "command": None,
                "attempt": attempt,
                "wall_time_s": 0.0,
                "run_dir": None,
            },
            "returncode": None,
            "stderr_tail": None,
            "final": None,
        }
        _atomic_write_json(result_path, payload)
        return payload

    if method not in _METHODS:
        raise CellExecutorError(f"unknown method {method!r}; supported: {sorted(_METHODS)}")

    trial_dir.mkdir(parents=True, exist_ok=True)
    pre_existing = _snapshot_subdirs(trial_dir)

    cmd = build_command(
        python_bin=python_bin,
        method=method,
        bench=bench,
        seed=seed,
        trial_dir=trial_dir,
        set_overrides=set_overrides,
        extra_args=extra_args,
    )

    try:
        pal_file = resolve_pal_file(python_bin, repo_root=repo_root)
    except Exception as e:  # noqa: BLE001 - record as infra failure, don't crash the harness
        pal_file = f"<unresolved: {type(e).__name__}: {e}>"

    start = time.monotonic()
    timed_out = False
    stderr = ""
    returncode: int | None = None
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(repo_root),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        returncode = proc.returncode
        stderr = proc.stderr
    except subprocess.TimeoutExpired as e:
        timed_out = True
        stderr = e.stderr or ""
    wall_time_s = time.monotonic() - start

    post_existing = _snapshot_subdirs(trial_dir)
    new_dirs = sorted(post_existing - pre_existing)
    run_dir: Path | None = None
    ambiguous_note = None
    if len(new_dirs) == 1:
        run_dir = trial_dir / new_dirs[0]
    elif len(new_dirs) > 1:
        # Should not happen; pick the newest dir and flag it.
        run_dir = max((trial_dir / d for d in new_dirs), key=lambda p: p.stat().st_mtime)
        ambiguous_note = f"multiple new run dirs created: {new_dirs}; picked {run_dir.name}"

    fingerprint = {
        "git_sha": git_sha,
        "pal_file": pal_file,
        "method": method,
        "bench": bench,
        "seed": seed,
        "requested_overrides": set_overrides,
        "command": cmd,
        "attempt": attempt,
        "wall_time_s": wall_time_s,
        "run_dir": str(run_dir) if run_dir is not None else None,
    }

    status: str
    reason: str | None = None
    stderr_tail: str | None = None
    final_payload: dict[str, Any] | None = None

    if timed_out:
        status = "infra_failure"
        reason = f"subprocess timed out after {timeout_s}s"
        stderr_tail = _tail_lines(stderr)
    elif returncode != 0:
        stderr_tail = _tail_lines(stderr)
        # A deterministic solver divergence counts as "diverged"; any other crash is infra.
        sig = _divergence_signature(stderr) or _failed_final_divergence(
            _read_run_final(run_dir))
        if sig is not None:
            status = "diverged"
            reason = f"training diverged (non-finite loss): {sig}"
            final_payload = None
        else:
            status = "infra_failure"
            reason = f"subprocess exited with returncode={returncode}"
    elif run_dir is None:
        status = "infra_failure"
        reason = "subprocess exited 0 but no run directory was created under trial_dir"
        stderr_tail = _tail_lines(stderr)
    else:
        final_json_path = run_dir / "final.json"
        if not final_json_path.exists():
            status = "infra_failure"
            reason = f"missing final.json under {run_dir}"
            stderr_tail = _tail_lines(stderr)
        else:
            try:
                with final_json_path.open() as f:
                    final_payload = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                status = "infra_failure"
                reason = f"malformed final.json under {run_dir}: {type(e).__name__}: {e}"
                stderr_tail = _tail_lines(stderr)
                final_payload = None
            else:
                if isinstance(final_payload, dict) and final_payload.get("status") == "failed":
                    # Same rule for a failed final.json: divergence -> diverged, else infra.
                    sig = _failed_final_divergence(final_payload)
                    if sig is not None:
                        status = "diverged"
                        reason = f"training diverged (non-finite loss): {sig}"
                        final_payload = None
                    else:
                        status = "infra_failure"
                        reason = f"run crashed: {final_payload.get('error')}"
                        stderr_tail = _tail_lines(stderr)
                elif _has_nonfinite(final_payload):
                    status = "diverged"
                    reason = "final.json metrics contain NaN/inf"
                else:
                    status = "ok"

    if ambiguous_note:
        reason = f"{reason}; {ambiguous_note}" if reason else ambiguous_note

    payload = {
        "status": status,
        "reason": reason,
        "fingerprint": fingerprint,
        "returncode": returncode,
        "stderr_tail": stderr_tail,
        "final": final_payload,
    }
    _atomic_write_json(result_path, payload)
    return payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--method", required=True, choices=sorted(_METHODS))
    p.add_argument("--bench", required=True, help="benchmark id, e.g. s1_sphere_track")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--trial-dir", required=True, type=Path)
    p.add_argument(
        "--set", action="append", default=[], dest="set_overrides", metavar="KEY=VALUE",
        help="logical hparam override, repeatable. KEY='lr' is special-cased "
             "(mapped to --set learning_rate=VALUE for snarenet, --lr VALUE "
             "otherwise); every other KEY is passed through as a raw "
             "'pal run --set KEY=VALUE' dataclass-field override.",
    )
    p.add_argument("--timeout-s", type=float, default=3600.0)
    p.add_argument("--attempt", type=int, default=1)
    p.add_argument("--result-path", type=Path, default=None)
    p.add_argument("--python-bin", default=sys.executable)
    p.add_argument(
        "--frozen-alm-dir", default=None,
        help="alm_bolton second-stage predict-only: root holding the frozen ALM "
             "winner run dirs (<root>/<bench>_seed<seed>/{config.json,model.pt,"
             "final.json}). When set (method must be alm_bolton), the cell loads "
             "the frozen checkpoint and predicts with the trial's projector "
             "knobs instead of training.",
    )
    p.add_argument(
        "--runner-arg", action="append", default=[], dest="extra_args",
        help="raw passthrough arg appended verbatim to the `pal run` "
             "invocation (repeatable), for flags with no dedicated knob here "
             "(e.g. --runner-arg --n-eval --runner-arg 4).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    set_overrides = parse_set_overrides(args.set_overrides)
    result = run_cell(
        method=args.method,
        bench=args.bench,
        seed=args.seed,
        trial_dir=args.trial_dir,
        set_overrides=set_overrides,
        timeout_s=args.timeout_s,
        attempt=args.attempt,
        extra_args=args.extra_args,
        python_bin=args.python_bin,
        result_path=args.result_path,
        frozen_alm_dir=args.frozen_alm_dir,
    )
    print(json.dumps({"status": result["status"], "reason": result["reason"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
