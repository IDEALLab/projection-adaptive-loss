#!/usr/bin/env python3
"""Low-RPC Slurm array lane for BO-tuning cells.

Public API: ``submit_cells``, ``poll_states``, ``retry_failed`` and ``register_manifest``
(to re-associate a manifest with a submitted job after a restart).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import uuid
from collections.abc import Iterable, Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CELL_EXECUTOR = _REPO_ROOT / "scripts" / "bo" / "cell_executor.py"
_TERMINAL_INFRA_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "OOM",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
_ACTIVE_STATES = {
    "COMPLETING",
    "CONFIGURING",
    "PENDING",
    "RUNNING",
    "SUSPENDED",
}

RESOURCE_PRESETS: dict[str, dict[str, Any]] = {
    "cpu_lane": {
        "cpus_per_task": 2,
        "mem_per_cpu_mb": 4096,
        "time": "04:00:00",
    },
    "gpu_lane": {
        "cpus_per_task": 4,
        "mem_per_cpu_mb": 4096,
        "time": "04:00:00",
        "gpus": "<GPU_TYPE>:1",
        "exclude": "gpu-node-046",  # site-specific bad-node exclusion
    },
}

# job id -> manifest path.  No state is written outside the campaign root.
_SUBMISSIONS: dict[str, Path] = {}


class SlurmLaneError(RuntimeError):
    """A malformed cell/resource request or scheduler command failure."""


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _parse_duration(value: str) -> int:
    """Return Slurm duration in seconds (supports D-HH:MM:SS and HH:MM:SS)."""
    day_split = value.split("-", 1)
    if len(day_split) == 2:
        days = int(day_split[0])
        clock = day_split[1]
    else:
        days = 0
        clock = day_split[0]
    pieces = [int(piece) for piece in clock.split(":")]
    if len(pieces) == 3:
        hours, minutes, seconds = pieces
    elif len(pieces) == 2:
        hours, minutes, seconds = 0, *pieces
    elif len(pieces) == 1:
        hours, minutes, seconds = 0, pieces[0], 0
    else:
        raise ValueError(value)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(value)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def _normalise_resources(resources: str | Mapping[str, Any] | None) -> dict[str, Any]:
    if resources is None:
        merged = deepcopy(RESOURCE_PRESETS["cpu_lane"])
    elif isinstance(resources, str):
        try:
            merged = deepcopy(RESOURCE_PRESETS[resources])
        except KeyError as exc:
            raise SlurmLaneError(f"unknown resource preset: {resources!r}") from exc
    else:
        preset = str(resources.get("preset", "cpu_lane"))
        try:
            merged = deepcopy(RESOURCE_PRESETS[preset])
        except KeyError as exc:
            raise SlurmLaneError(f"unknown resource preset: {preset!r}") from exc
        merged.update({key: value for key, value in resources.items() if key != "preset"})

    try:
        mem = int(merged["mem_per_cpu_mb"])
        cpus = int(merged["cpus_per_task"])
        seconds = _parse_duration(str(merged["time"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise SlurmLaneError(f"invalid Slurm resources: {merged}") from exc
    if mem < 1 or mem > 4096:
        raise SlurmLaneError("--mem-per-cpu must be between 1 and 4096 MB")
    if cpus < 2 or cpus > 4:
        raise SlurmLaneError("--cpus-per-task must be between 2 and 4")
    if seconds > 24 * 3600:
        # fsnet s5 cells need ~4.6h at 4 threads.
        raise SlurmLaneError("--time must not exceed 24:00:00")
    merged["mem_per_cpu_mb"] = mem
    merged["cpus_per_task"] = cpus
    merged["time"] = str(merged["time"])
    merged["python_bin"] = str(merged.get("python_bin", sys.executable))
    return merged


def _normalise_overrides(cell: Mapping[str, Any]) -> dict[str, str]:
    raw = cell.get("set_overrides", cell.get("overrides", cell.get("sets", {})))
    if isinstance(raw, Mapping):
        return {str(key): str(value) for key, value in raw.items()}
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        resolved: dict[str, str] = {}
        for item in raw:
            key, separator, value = str(item).partition("=")
            if not separator:
                raise SlurmLaneError(f"cell override must be KEY=VALUE: {item!r}")
            resolved[key.strip()] = value.strip()
        return resolved
    raise SlurmLaneError("cell overrides must be a mapping or sequence of KEY=VALUE strings")


def _normalise_cells(
    cells: Iterable[Mapping[str, Any]],
    campaign_root: Path,
) -> list[dict[str, Any]]:
    normalised: list[dict[str, Any]] = []
    for array_index, cell in enumerate(cells):
        try:
            method = str(cell["method"])
            bench = str(cell["bench"])
            seed = int(cell["seed"])
            trial_dir = Path(cell["trial_dir"]).expanduser().resolve()
        except (KeyError, TypeError, ValueError) as exc:
            raise SlurmLaneError(f"invalid cell at index {array_index}: {cell}") from exc
        if not trial_dir.is_relative_to(campaign_root):
            raise SlurmLaneError(f"trial_dir must be under campaign_root: {trial_dir}")
        entry = {
            "array_index": array_index,
            "source_index": int(cell.get("source_index", array_index)),
            "method": method,
            "bench": bench,
            "seed": seed,
            "set_overrides": _normalise_overrides(cell),
            "trial_dir": str(trial_dir),
            "timeout_s": float(cell.get("timeout_s", 3600.0)),
            "attempt": int(cell.get("attempt", 1)),
        }
        # Only present for alm_bolton predict-only cells.
        frozen_alm_dir = cell.get("frozen_alm_dir")
        if frozen_alm_dir:
            entry["frozen_alm_dir"] = str(frozen_alm_dir)
        normalised.append(entry)
    if not normalised:
        raise SlurmLaneError("cannot submit an empty cell array")
    return normalised


def _artifact_stem() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + uuid.uuid4().hex[:10]


def _sbatch_directives(resources: Mapping[str, Any], campaign_root: Path) -> list[str]:
    logs = campaign_root / "slurm" / "logs"
    lines = [
        f"#SBATCH --cpus-per-task={resources['cpus_per_task']}",
        f"#SBATCH --mem-per-cpu={resources['mem_per_cpu_mb']}M",
        f"#SBATCH --time={resources['time']}",
        f"#SBATCH --output={logs}/%A_%a.out",
        f"#SBATCH --error={logs}/%A_%a.err",
    ]
    option_map = {
        "account": "account",
        "partition": "partition",
        "gpus": "gpus",
        "exclude": "exclude",
    }
    for key, flag in option_map.items():
        if resources.get(key):
            lines.append(f"#SBATCH --{flag}={resources[key]}")
    return lines


def _render_sbatch(
    manifest_path: Path,
    campaign_root: Path,
    resources: Mapping[str, Any],
) -> str:
    directives = "\n".join(_sbatch_directives(resources, campaign_root))
    return f"""#!/usr/bin/env bash
{directives}
set -euo pipefail
exec {shlex.quote(str(resources["python_bin"]))} {shlex.quote(str(Path(__file__).resolve()))} run-task {shlex.quote(str(manifest_path))}
"""


def _sbatch_argv(
    script_path: Path,
    task_count: int,
    resources: Mapping[str, Any],
    campaign_root: Path,
) -> list[str]:
    logs = campaign_root / "slurm" / "logs"
    argv = [
        "sbatch",
        "--parsable",
        f"--array=0-{task_count - 1}",
        f"--cpus-per-task={resources['cpus_per_task']}",
        f"--mem-per-cpu={resources['mem_per_cpu_mb']}M",
        f"--time={resources['time']}",
        f"--output={logs}/%A_%a.out",
        f"--error={logs}/%A_%a.err",
    ]
    for key, flag in (
        ("account", "account"),
        ("partition", "partition"),
        ("gpus", "gpus"),
        ("exclude", "exclude"),
    ):
        if resources.get(key):
            argv.append(f"--{flag}={resources[key]}")
    argv.append(str(script_path))
    return argv


def _submit_manifest(payload: dict[str, Any]) -> str:
    campaign_root = Path(payload["campaign_root"])
    artifact_dir = campaign_root / "slurm"
    (artifact_dir / "logs").mkdir(parents=True, exist_ok=True)
    stem = _artifact_stem()
    manifest_path = artifact_dir / f"manifest-{stem}.json"
    script_path = artifact_dir / f"array-{stem}.sbatch"
    payload["manifest_path"] = str(manifest_path)
    payload["sbatch_script"] = str(script_path)
    _atomic_json(manifest_path, payload)
    script_path.write_text(
        _render_sbatch(manifest_path, campaign_root, payload["resources"])
    )
    script_path.chmod(0o750)

    completed = subprocess.run(
        _sbatch_argv(script_path, len(payload["cells"]), payload["resources"], campaign_root),
        check=True,
        capture_output=True,
        text=True,
    )
    jobid = completed.stdout.strip().split(";", 1)[0]
    if not jobid or not jobid.split("_", 1)[0].isdigit():
        raise SlurmLaneError(f"could not parse sbatch job id from: {completed.stdout!r}")
    payload["job_id"] = jobid
    _atomic_json(manifest_path, payload)
    _SUBMISSIONS[jobid] = manifest_path
    return jobid


def submit_cells(
    cells: Iterable[Mapping[str, Any]],
    campaign_root: str | Path,
    sha: str,
    resources: str | Mapping[str, Any] | None = None,
) -> str:
    """Write an immutable-SHA manifest and submit exactly one job array."""
    root = Path(campaign_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    resource_spec = _normalise_resources(resources)
    payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "campaign_root": str(root),
        "git_sha": str(sha),
        "resources": resource_spec,
        "cells": _normalise_cells(cells, root),
    }
    return _submit_manifest(payload)


def register_manifest(jobid: str, manifest: str | Path) -> None:
    """Associate an existing submission with its manifest after controller restart."""
    _SUBMISSIONS[str(jobid)] = Path(manifest).expanduser().resolve()


def manifest_for_job(jobid: str) -> Path:
    try:
        return _SUBMISSIONS[str(jobid)]
    except KeyError as exc:
        raise SlurmLaneError(
            f"no manifest registered for job {jobid}; call register_manifest()"
        ) from exc


def _load_manifest(manifest: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(manifest, Mapping):
        return deepcopy(dict(manifest))
    with Path(manifest).open() as handle:
        return json.load(handle)


def _scheduler_rows(argv: list[str]) -> dict[int, str]:
    try:
        completed = subprocess.run(
            argv,
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {}
    rows: dict[int, str] = {}
    for line in completed.stdout.splitlines():
        raw_id, separator, raw_state = line.strip().partition("|")
        if not separator:
            continue
        # Ignore allocation/step rows; accept "123_4" and "123_4.batch".
        task_token = raw_id.split(".", 1)[0].rsplit("_", 1)
        if len(task_token) != 2 or not task_token[1].isdigit():
            continue
        state = raw_state.strip().split()[0].split("+", 1)[0].upper()
        rows[int(task_token[1])] = state
    return rows


def _result_status(cell: Mapping[str, Any]) -> str | None:
    """Status recorded in the cell's ``result.json``, or None if there is no
    result for the cell's *current* attempt.
    """
    result_path = Path(str(cell["trial_dir"])) / "result.json"
    if not result_path.exists():
        return None
    try:
        with result_path.open() as handle:
            payload = json.load(handle)
        status = payload.get("status")
        attempt = int((payload.get("fingerprint") or {}).get("attempt", 1))
    except (OSError, json.JSONDecodeError, AttributeError, TypeError, ValueError):
        return "infra_failure"
    if attempt < int(cell.get("attempt", 1)):
        return None  # stale result from a previous attempt
    if status in {"ok", "diverged", "infra_failure", "structurally_excluded"}:
        return str(status)
    return "infra_failure"


def poll_states(jobid: str) -> dict[int, str]:
    """Return manifest-array-index -> result/Slurm classification."""
    manifest = _load_manifest(manifest_for_job(jobid))
    # -r expands pending array ranges (e.g. 123_[0-4]) into one row per task.
    squeue = _scheduler_rows(
        ["squeue", "-h", "-r", "-j", str(jobid), "-o", "%i|%T"]
    )
    sacct = _scheduler_rows(
        [
            "sacct",
            "-n",
            "-j",
            str(jobid),
            "--format=JobIDRaw,State",
            "--parsable2",
        ]
    )
    states: dict[int, str] = {}
    for cell in manifest["cells"]:
        index = int(cell["array_index"])
        result = _result_status(cell)
        if result is not None:
            states[index] = result
            continue
        scheduler_state = squeue.get(index, sacct.get(index))
        if scheduler_state in _TERMINAL_INFRA_STATES or scheduler_state is None:
            states[index] = "infra_failure"
        elif scheduler_state in _ACTIVE_STATES:
            states[index] = scheduler_state.lower()
        elif scheduler_state == "COMPLETED":
            # A completed task without its durable output is infrastructure failure.
            states[index] = "infra_failure"
        else:
            states[index] = scheduler_state.lower()
    return states


def retry_failed(
    manifest: str | Path | Mapping[str, Any],
    jobid: str,
    max_attempts: int,
) -> str | None:
    """Submit one compact array of retryable infra failures, or return None."""
    payload = _load_manifest(manifest)
    register_manifest(jobid, payload["manifest_path"])
    states = poll_states(jobid)
    retry_cells: list[dict[str, Any]] = []
    for cell in payload["cells"]:
        index = int(cell["array_index"])
        attempt = int(cell.get("attempt", 1))
        if states.get(index) != "infra_failure" or attempt >= max_attempts:
            continue
        retried = deepcopy(cell)
        retried["attempt"] = attempt + 1
        retried["array_index"] = len(retry_cells)
        retry_cells.append(retried)
    if not retry_cells:
        return None
    retry_payload = {
        "schema_version": 1,
        "created_at_unix": time.time(),
        "campaign_root": payload["campaign_root"],
        "git_sha": payload["git_sha"],
        "resources": payload["resources"],
        "retry_of_job_id": str(jobid),
        "cells": retry_cells,
    }
    return _submit_manifest(retry_payload)


def _run_task(manifest_path: Path) -> None:
    manifest = _load_manifest(manifest_path)
    try:
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
        cell = manifest["cells"][task_id]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise SlurmLaneError("invalid or missing SLURM_ARRAY_TASK_ID") from exc

    try:
        actual_sha = subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise SlurmLaneError("could not resolve checkout git SHA") from exc
    if actual_sha != manifest["git_sha"]:
        raise SlurmLaneError(
            f"checkout SHA {actual_sha} does not match pinned SHA {manifest['git_sha']}"
        )

    argv = [
        str(manifest["resources"]["python_bin"]),
        str(_CELL_EXECUTOR),
        "--method",
        str(cell["method"]),
        "--bench",
        str(cell["bench"]),
        "--seed",
        str(cell["seed"]),
        "--trial-dir",
        str(cell["trial_dir"]),
        "--timeout-s",
        str(cell["timeout_s"]),
        "--attempt",
        str(cell["attempt"]),
        "--python-bin",
        str(manifest["resources"]["python_bin"]),
    ]
    frozen_alm_dir = cell.get("frozen_alm_dir")
    if frozen_alm_dir:
        argv.extend(["--frozen-alm-dir", str(frozen_alm_dir)])
    for key, value in cell["set_overrides"].items():
        argv.extend(["--set", f"{key}={value}"])
    os.execv(argv[0], argv)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    task = subparsers.add_parser("run-task")
    task.add_argument("manifest", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "run-task":
        _run_task(args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
