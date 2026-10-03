"""Unit tests and one slow integration test for scripts/bo/cell_executor.py.

Unit tests monkeypatch `subprocess.run`, so no real training happens.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.bo import cell_executor as ce


def test_snarenet_lr_maps_to_set_learning_rate(tmp_path: Path) -> None:
    cmd = ce.build_command(
        python_bin=sys.executable,
        method="snarenet",
        bench="s1_sphere_track",
        seed=0,
        trial_dir=tmp_path,
        set_overrides={"lr": "0.05", "epochs": "3"},
    )
    assert "--lr" not in cmd
    assert "--set" in cmd
    set_values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--set"]
    assert "learning_rate=0.05" in set_values
    assert "epochs=3" in set_values


@pytest.mark.parametrize("method", ["pal_loggap", "alm", "alm_bolton", "dc3", "fsnet", "enforce_orig", "enforce_v4"])
def test_other_methods_lr_uses_typed_flag(tmp_path: Path, method: str) -> None:
    cmd = ce.build_command(
        python_bin=sys.executable,
        method=method,
        bench="s1_sphere_track",
        seed=0,
        trial_dir=tmp_path,
        set_overrides={"lr": "1e-4"},
    )
    assert "--lr" in cmd
    assert cmd[cmd.index("--lr") + 1] == "1e-4"
    set_values = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "--set"]
    assert not any(v.startswith("lr=") for v in set_values)


def test_build_command_unknown_method_raises(tmp_path: Path) -> None:
    with pytest.raises(ce.CellExecutorError):
        ce.build_command(
            python_bin=sys.executable,
            method="not_a_method",
            bench="s1_sphere_track",
            seed=0,
            trial_dir=tmp_path,
            set_overrides={},
        )


def test_parse_set_overrides() -> None:
    assert ce.parse_set_overrides(["lr=1e-4", "epochs=5"]) == {"lr": "1e-4", "epochs": "5"}
    with pytest.raises(ce.CellExecutorError):
        ce.parse_set_overrides(["not-a-pair"])


def test_dc3_s5_is_structurally_excluded() -> None:
    assert ce.is_structurally_excluded("dc3", "s5_overdetermined")
    assert not ce.is_structurally_excluded("alm", "s5_overdetermined")
    assert not ce.is_structurally_excluded("dc3", "s1_sphere_track")


def test_dc3_s5_refused_without_invoking_subprocess(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _must_not_run(*args, **kwargs):
        raise AssertionError("subprocess.run must not be called for a structurally excluded cell")

    monkeypatch.setattr(ce.subprocess, "run", _must_not_run)

    trial_dir = tmp_path / "dc3" / "trial_0" / "s5_overdetermined_seed0"
    result = ce.run_cell(
        method="dc3",
        bench="s5_overdetermined",
        seed=0,
        trial_dir=trial_dir,
        set_overrides={"lr": "1e-4"},
    )
    assert result["status"] == "structurally_excluded"
    assert result["fingerprint"]["run_dir"] is None
    assert result["fingerprint"]["method"] == "dc3"
    assert result["fingerprint"]["bench"] == "s5_overdetermined"

    result_path = trial_dir / "result.json"
    assert result_path.exists()
    on_disk = json.loads(result_path.read_text())
    assert on_disk["status"] == "structurally_excluded"


def _extract_runs_root(cmd: list[str]) -> Path:
    return Path(cmd[cmd.index("--runs-root") + 1])


def _fake_run_factory(
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
    make_run_dir: bool = True,
    final_json: dict | str | None = None,
):
    """Fake `subprocess.run` mimicking `pal run`: optionally makes a run dir and final.json."""

    def _fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        if make_run_dir:
            runs_root = _extract_runs_root(cmd)
            runs_root.mkdir(parents=True, exist_ok=True)
            run_dir = runs_root / "20260101T000000Z_method_bench_seed0_deadbeef"
            run_dir.mkdir(parents=True, exist_ok=True)
            if final_json is not None:
                text_payload = (
                    final_json if isinstance(final_json, str) else json.dumps(final_json)
                )
                (run_dir / "final.json").write_text(text_payload)
        return subprocess.CompletedProcess(args=cmd, returncode=returncode, stdout=stdout, stderr=stderr)

    return _fake_run


def _base_run_cell_kwargs(trial_dir: Path) -> dict:
    return dict(
        method="alm",
        bench="s1_sphere_track",
        seed=0,
        trial_dir=trial_dir,
        set_overrides={"lr": "1e-4", "epochs": "3"},
        python_bin=sys.executable,
    )


def test_classify_ok(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = {"obj_mean_raw": 1.23, "feasibility_post": 100.0, "viol_max_post": 0.0}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(returncode=0, final_json=final))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "ok"
    assert result["reason"] is None
    assert result["final"] == final
    assert result["fingerprint"]["run_dir"] is not None


def test_classify_diverged_on_nan(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = {"obj_mean_raw": float("nan"), "feasibility_post": 100.0}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(returncode=0, final_json=final))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "diverged"
    assert "NaN" in result["reason"] or "inf" in result["reason"]


def test_classify_diverged_on_inf_nested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    final = {"per_constraint_viol_max_post": [0.0, float("inf")]}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(returncode=0, final_json=final))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "diverged"


def test_classify_infra_missing_final_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ce.subprocess, "run", _fake_run_factory(returncode=0, make_run_dir=True, final_json=None)
    )
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"
    assert "missing final.json" in result["reason"]


def test_classify_infra_malformed_final_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ce.subprocess,
        "run",
        _fake_run_factory(returncode=0, make_run_dir=True, final_json="{not valid json"),
    )
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"
    assert "malformed final.json" in result["reason"]


def test_classify_infra_nonzero_returncode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ce.subprocess,
        "run",
        _fake_run_factory(returncode=1, stderr="Traceback...\nRuntimeError: boom", make_run_dir=False),
    )
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"
    assert result["returncode"] == 1
    assert "boom" in result["stderr_tail"]
    assert result["fingerprint"]["run_dir"] is None


def test_classify_infra_caught_crash_final_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed final.json plus nonzero exit (cli.py's except-handler) is infra, also at rc=0."""
    final = {"status": "failed", "error": "ValueError: something exploded"}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(returncode=0, final_json=final))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"
    assert "something exploded" in result["reason"]


# A crash or failed final with a divergence signature is 'diverged', never infra.
_DC3_NAN_STDERR = (
    "Traceback (most recent call last):\n"
    '  File "/x/pal/baselines/dc3/solver.py", line 521, in train\n'
    "    raise RuntimeError(\n"
    "RuntimeError: epoch 1: non-finite DC3 loss (loss=nan)\n"
)


def test_classify_diverged_rc_nonzero_dc3_signature(tmp_path: Path, monkeypatch) -> None:
    # cli.py writes a failed final.json and re-raises (rc=1).
    failed_final = {"status": "failed", "error": "epoch 1: non-finite DC3 loss (loss=nan)"}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=1, stderr=_DC3_NAN_STDERR, make_run_dir=True, final_json=failed_final))
    result = ce.run_cell(**{**_base_run_cell_kwargs(tmp_path / "trial"), "method": "dc3"})
    assert result["status"] == "diverged"
    assert "training diverged" in result["reason"]
    assert "non-finite DC3 loss" in result["reason"]
    assert result["final"] is None  # crash wrote no real metrics


def test_classify_diverged_rc_nonzero_signature_only_in_failed_final(tmp_path, monkeypatch) -> None:
    # No traceback on stderr, but the crash-written failed-final error matches.
    failed_final = {"status": "failed", "error": "epoch 3: non-finite loss (obj=1.2e+30, pen=nan)"}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=1, stderr="", make_run_dir=True, final_json=failed_final))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "diverged"
    assert "non-finite loss" in result["reason"]
    assert result["final"] is None


def test_classify_diverged_rc_nonzero_newton_completion(tmp_path, monkeypatch) -> None:
    stderr = ("pal.baselines.dc3._completion.CompletionDivergedError: "
              "Newton diverged at iter 7: non-finite residual\n")
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=1, stderr=stderr, make_run_dir=False))
    result = ce.run_cell(**{**_base_run_cell_kwargs(tmp_path / "trial"), "method": "dc3"})
    assert result["status"] == "diverged"
    assert result["final"] is None


def test_classify_infra_rc_nonzero_unrelated_stderr(tmp_path, monkeypatch) -> None:
    # A genuine infra crash (OOM), not a divergence signature, stays infra.
    stderr = "Traceback...\ntorch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ...\n"
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=1, stderr=stderr, make_run_dir=False))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"
    assert "returncode=1" in result["reason"]


def test_classify_infra_rc_nonzero_bare_nan_not_matched(tmp_path, monkeypatch) -> None:
    # A bare 'nan' in an unrelated tensor repr must not be read as divergence.
    stderr = "AssertionError: expected tensor([1.0, nan, 3.0]) to be sorted\n"
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=1, stderr=stderr, make_run_dir=False))
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"))
    assert result["status"] == "infra_failure"


def test_classify_diverged_rc0_failed_final_nonfinite_error(tmp_path, monkeypatch) -> None:
    failed_final = {"status": "failed",
                    "error": "epoch 1: non-finite gradient in param 'net.0.weight'"}
    monkeypatch.setattr(ce.subprocess, "run", _fake_run_factory(
        returncode=0, final_json=failed_final))
    result = ce.run_cell(**{**_base_run_cell_kwargs(tmp_path / "trial"), "method": "pal_loggap"})
    assert result["status"] == "diverged"
    assert result["final"] is None


def test_divergence_signature_matcher() -> None:
    assert ce._divergence_signature("RuntimeError: epoch 1: non-finite DC3 loss (loss=nan)")
    assert ce._divergence_signature("epoch 2: non-finite loss (obj=1e30, pen=nan)")
    assert ce._divergence_signature("epoch 1: non-finite gradient in param 'w'")
    assert ce._divergence_signature("Newton diverged at iter 3: non-finite residual")
    assert ce._divergence_signature("CompletionDivergedError: boom")
    # Deterministic linear-algebra breakdown (enforce AdaNP eigh).
    assert ce._divergence_signature(
        "torch._C._LinAlgError: linalg.eigh: (Batch element 11): The algorithm "
        "failed to converge because the input matrix is ill-conditioned")
    assert ce._divergence_signature("tensor([1.0, nan])") is None
    assert ce._divergence_signature("CUDA out of memory") is None
    assert ce._divergence_signature("") is None
    assert ce._divergence_signature(None) is None
    line = ce._divergence_signature(_DC3_NAN_STDERR)
    assert "non-finite DC3 loss" in line


def test_classify_infra_timeout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _fake_run(cmd, cwd=None, capture_output=True, text=True, timeout=None):
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=timeout, output="partial-out", stderr="partial-err")

    monkeypatch.setattr(ce.subprocess, "run", _fake_run)
    result = ce.run_cell(**_base_run_cell_kwargs(tmp_path / "trial"), timeout_s=5.0)
    assert result["status"] == "infra_failure"
    assert "timed out" in result["reason"]
    assert "partial-err" in result["stderr_tail"]


def test_atomic_write_json_no_leftover_tmp(tmp_path: Path) -> None:
    path = tmp_path / "sub" / "result.json"
    ce._atomic_write_json(path, {"a": 1, "b": [1, 2, 3]})
    assert path.exists()
    assert json.loads(path.read_text()) == {"a": 1, "b": [1, 2, 3]}
    assert not path.with_suffix(".json.tmp").exists()

    ce._atomic_write_json(path, {"a": 2})
    assert json.loads(path.read_text()) == {"a": 2}
    assert not path.with_suffix(".json.tmp").exists()


def test_run_cell_writes_result_json_at_default_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ce.subprocess, "run", _fake_run_factory(returncode=0, final_json={"obj_mean_raw": 0.0})
    )
    trial_dir = tmp_path / "trial"
    result = ce.run_cell(**_base_run_cell_kwargs(trial_dir))
    default_path = trial_dir / "result.json"
    assert default_path.exists()
    assert json.loads(default_path.read_text())["status"] == result["status"] == "ok"


@pytest.mark.slow
def test_integration_alm_s1_cell_ok(tmp_path: Path) -> None:
    trial_dir = tmp_path / "alm" / "trial_0" / "s1_sphere_track_seed123"
    result = ce.run_cell(
        method="alm",
        bench="s1_sphere_track",
        seed=123,
        trial_dir=trial_dir,
        set_overrides={"lr": "1e-4", "epochs": "5"},
        timeout_s=120.0,
        python_bin=sys.executable,
    )
    assert result["status"] == "ok", result
    assert result["returncode"] == 0
    fp = result["fingerprint"]
    assert fp["git_sha"] and fp["git_sha"] != "unknown"
    assert fp["pal_file"].startswith(str(ce._REPO_ROOT))
    assert fp["run_dir"] is not None
    assert (Path(fp["run_dir"]) / "final.json").exists()
    assert result["final"] is not None
    assert "obj_mean_raw" in result["final"]
    assert (trial_dir / "result.json").exists()
