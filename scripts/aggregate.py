"""Walk runs/, emit runs.parquet (one row per run, failed runs included).

Usage:
    python scripts/aggregate.py                       # default: ./runs -> ./runs.parquet
    python scripts/aggregate.py --runs-root /scratch/pal/runs --out runs.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import polars as pl

_RUN_DIR_RE = re.compile(
    r"^(?P<ts>\d{8}T\d{6}Z)_"
    r"(?P<method>[a-z_]+)_"
    r"(?P<bench>.+?)_seed(?P<seed>\d+)"
    r"(?:_r(?P<shard_r>\d+)_of(?P<shard_R>\d+))?"
    r"_[0-9a-f]+$"
)


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def _parse_run_dir(name: str) -> dict | None:
    m = _RUN_DIR_RE.match(name)
    if not m:
        return None
    return {
        "run_dir_ts": m["ts"],
        "run_dir_method": m["method"],
        "run_dir_bench": m["bench"],
        "run_dir_seed": int(m["seed"]),
        "run_dir_restart_shard_r": int(m["shard_r"]) if m["shard_r"] is not None else None,
        "run_dir_restart_shard_R": int(m["shard_R"]) if m["shard_R"] is not None else None,
    }


def _row_for_run(run_dir: Path) -> dict:
    """One row per run. Missing files yield null metric columns."""
    parsed = _parse_run_dir(run_dir.name) or {}
    config = _read_json(run_dir / "config.json") or {}
    final = _read_json(run_dir / "final.json")
    status = _read_json(run_dir / "status.json") or {}
    spec = config.get("benchmark_spec") or {}
    hparams = config.get("hparams") or {}

    row: dict = {
        "run_id": run_dir.name,
        "run_path": str(run_dir),
        # Fall back to the dirname if config.json was never written.
        "method": config.get("method") or parsed.get("run_dir_method"),
        "benchmark_id": config.get("benchmark_id") or parsed.get("run_dir_bench"),
        "family": spec.get("family"),
        "variant": spec.get("variant"),
        "seed": config.get("seed") if config.get("seed") is not None else parsed.get("run_dir_seed"),
        "device": config.get("device"),
        "hostname": config.get("hostname"),
        "pal_git_sha": config.get("pal_git_sha"),
        "schema_version": config.get("schema_version"),
        "wall_start": config.get("wall_start"),
        "n_eval_effective": config.get("n_eval_effective"),
        "eval_queries_fingerprint": config.get("eval_queries_fingerprint"),
        "ablation_name": (hparams.get("ablation_name") or None),
        "scenario_label": (hparams.get("scenario_label") or None),
        "slurm_job_id": config.get("slurm_job_id"),
        "slurm_array_job_id": config.get("slurm_array_job_id"),
        "slurm_array_task_id": config.get("slurm_array_task_id"),
        "shard_spec": config.get("shard_spec"),
        # Null for non-sharded runs.
        "restart_shard_r": parsed.get("run_dir_restart_shard_r"),
        "restart_shard_R": parsed.get("run_dir_restart_shard_R"),
        "status": status.get("status"),
        "error": status.get("error"),
        "duration_s": status.get("duration_s"),
        "finished_at": status.get("finished_at"),
        "obj_mean_raw": (final or {}).get("obj_mean_raw"),
        "obj_mean_post": (final or {}).get("obj_mean_post"),
        "viol_max_raw": (final or {}).get("viol_max_raw"),
        "viol_max_post": (final or {}).get("viol_max_post"),
        "feasibility_raw": (final or {}).get("feasibility_raw"),
        "feasibility_post": (final or {}).get("feasibility_post"),
        "train_wall_time_s": (final or {}).get("train_wall_time_s"),
        "predict_wall_time_s": (final or {}).get("predict_wall_time_s"),
        "n_queries": (final or {}).get("n_queries"),
        "n_restarts": (final or {}).get("n_restarts"),
        "tolerance": (final or {}).get("tolerance"),
        "train_fwd_calls": (final or {}).get("train__fwd_calls"),
        "train_bwd_calls": (final or {}).get("train__bwd_calls"),
        "train_fwd_samples": (final or {}).get("train__fwd_samples"),
        "train_bwd_samples": (final or {}).get("train__bwd_samples"),
        "train_opt_steps": (final or {}).get("train__opt_steps"),
        # Populated only when jacobian_mode="sample".
        "train_measurement_fwd_calls":
            (final or {}).get("train__measurement_fwd_calls"),
        "train_measurement_bwd_calls":
            (final or {}).get("train__measurement_bwd_calls"),
        "train_measurement_fwd_samples":
            (final or {}).get("train__measurement_fwd_samples"),
        "train_measurement_bwd_samples":
            (final or {}).get("train__measurement_bwd_samples"),
        "train_measurement_opt_steps":
            (final or {}).get("train__measurement_opt_steps"),
        # Null for solvers without an inference repair loop.
        "inf_iters_median": (final or {}).get("inf_iters_median"),
        "inf_iters_p90": (final or {}).get("inf_iters_p90"),
        "inf_iters_max": (final or {}).get("inf_iters_max"),
        "inf_iters_n_converged": (final or {}).get("inf_iters_n_converged"),
        "inf_iters_max_allowed": (final or {}).get("inf_iters_max_allowed"),
    }
    return row


def aggregate(runs_root: Path) -> pl.DataFrame:
    rows = [
        _row_for_run(d)
        for d in sorted(runs_root.iterdir())
        if d.is_dir()
    ]
    if not rows:
        return pl.DataFrame({
            k: pl.Series(name=k, values=[], dtype=pl.Utf8)
            for k in ("run_id", "method", "benchmark_id", "status")
        })
    return pl.from_dicts(rows, infer_schema_length=None)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="aggregate pal runs into runs.parquet")
    p.add_argument("--runs-root", default="runs", help="root containing run dirs (default: runs)")
    p.add_argument("--out", default="runs.parquet", help="output parquet path (default: runs.parquet)")
    args = p.parse_args(argv)

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"[aggregate] runs root not found: {runs_root}", file=sys.stderr)
        return 1

    df = aggregate(runs_root)
    out = Path(args.out)
    df.write_parquet(out)

    n_total = df.height
    n_ok = df.filter(pl.col("status") == "ok").height if n_total else 0
    n_failed = df.filter(pl.col("status") == "failed").height if n_total else 0
    n_other = n_total - n_ok - n_failed
    print(
        f"[aggregate] wrote {out} "
        f"({n_total} runs: {n_ok} ok, {n_failed} failed, {n_other} unknown)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
