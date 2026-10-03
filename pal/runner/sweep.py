"""`pal sweep`, plan + execute a cartesian (method, bench, seed) sweep.

Subcommands: `plan` writes per-device jobs files and prints the `sbatch`
commands, `run-row` dispatches one row to `pal run` (skipping completed runs),
and `aggregate` writes `results.md`.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pal.benchmarks import list_all as list_all_benchmarks
from pal.benchmarks.registry import _REGISTRY
from pal.sweep import Tier, load_tier_registry, resolve_tier

_ALL_METHODS = ("pal_loggap", "alm", "alm_bolton", "enforce_orig", "enforce_v4", "dc3", "fsnet", "snarenet", "slsqp", "ipopt")
_GPU_TIERS: tuple[Tier, ...] = ("S", "M", "L")
# "cheap" CPU rows go to a short partition, "std" (mid + expensive) to the default one.
_CPU_TIERS: tuple[str, ...] = ("cheap", "std")


def _cpu_tier(bench_id: str) -> str:
    """Map registry cost -> CPU array tier."""
    return "cheap" if _REGISTRY[bench_id].cost == "cheap" else "std"


def _timestamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def _partition_benches(bench_ids: list[str]) -> tuple[list[str], list[str]]:
    """Split benches into (cpu, gpu) by `registry.device`."""
    cpu, gpu = [], []
    for bid in bench_ids:
        entry = _REGISTRY.get(bid)
        if entry is None:
            raise KeyError(f"unknown bench id '{bid}'")
        (gpu if entry.device == "gpu" else cpu).append(bid)
    return cpu, gpu


def _heavy_first(bench_ids: list[str]) -> list[str]:
    """Stable ordering by (cost desc, name). Heavy benches run first so
    failures surface early in an array."""
    order = {"expensive": 0, "mid": 1, "cheap": 2}
    return sorted(bench_ids, key=lambda b: (order.get(_REGISTRY[b].cost, 9), b))


def _enumerate_rows(
    methods: list[str],
    bench_ids: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    """Cartesian product, bench-major so each SLURM slot stays on the same
    bench as long as possible (minimizes heavy-import churn for engineering
    benches)."""
    rows = []
    for bid in bench_ids:
        for method in methods:
            for seed in seeds:
                rows.append({"method": method, "bench_id": bid, "seed": seed})
    return rows


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _existing_final(run_dir: Path) -> dict[str, Any] | None:
    """Return parsed final.json if present, else None."""
    fp = run_dir / "final.json"
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text())
    except Exception:
        return None


def _find_prior_run(
    runs_root: Path, method: str, bench_id: str, seed: int
) -> Path | None:
    """Look for a prior run dir matching the (method, bench, seed) triple.

    Run dir names are stamped `<ts>_<method>_<bench_flat>_seed<seed>_<hash>`.
    We match on method + bench + seed, ignoring timestamp/hash. Returns the
    first match (or None). Bench id slashes are flattened to underscores.
    """
    if not runs_root.exists():
        return None
    bench_flat = bench_id.replace("/", "_")
    needle = f"_{method}_{bench_flat}_seed{seed}_"
    for d in runs_root.iterdir():
        if d.is_dir() and needle in d.name:
            return d
    return None


def _cmd_plan(args: argparse.Namespace) -> int:
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    for m in methods:
        if m not in _ALL_METHODS:
            raise SystemExit(f"unknown method '{m}'; expected one of {_ALL_METHODS}")

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    all_ids = list_all_benchmarks()
    skip = set(x.strip() for x in (args.skip_benches or "").split(",") if x.strip())
    bench_ids = [b for b in all_ids if b not in skip]
    bench_ids = _heavy_first(bench_ids)

    cpu_benches, gpu_benches = _partition_benches(bench_ids)
    cpu_rows = _enumerate_rows(methods, cpu_benches, seeds)
    gpu_rows = _enumerate_rows(methods, gpu_benches, seeds)

    cpu_rows_by_tier: dict[str, list[dict[str, Any]]] = {t: [] for t in _CPU_TIERS}
    for r in cpu_rows:
        tier = _cpu_tier(r["bench_id"])
        cpu_rows_by_tier[tier].append({**r, "cpu_tier": tier})

    # S/M GPU rows are submitted here; L rows need a larger cluster.
    tier_registry = load_tier_registry(args.gpu_presets)
    gpu_rows_by_tier: dict[Tier, list[dict[str, Any]]] = {t: [] for t in _GPU_TIERS}
    for r in gpu_rows:
        tier = resolve_tier(r["bench_id"], r["method"], tier_registry)
        gpu_rows_by_tier[tier].append({**r, "gpu_tier": tier})

    pal_root = Path(__file__).resolve().parents[2]
    sweep_name = args.name or f"sweep_{_timestamp()}"
    sweep_dir = (pal_root / "runs" / sweep_name).resolve()
    (sweep_dir / "runs").mkdir(parents=True, exist_ok=True)

    cpu_paths: dict[str, Path] = {}
    for tier in _CPU_TIERS:
        path = sweep_dir / f"jobs_cpu_{tier}.jsonl"
        _write_jsonl(path, cpu_rows_by_tier[tier])
        cpu_paths[tier] = path

    gpu_paths: dict[Tier, Path] = {}
    for tier in _GPU_TIERS:
        path = sweep_dir / f"jobs_gpu_{tier.lower()}.jsonl"
        _write_jsonl(path, gpu_rows_by_tier[tier])
        gpu_paths[tier] = path

    manifest = {
        "schema_version": 3,
        "created_at": _timestamp(),
        "methods": methods,
        "seeds": seeds,
        "bench_ids": bench_ids,
        "bench_partition": {"cpu": cpu_benches, "gpu": gpu_benches},
        "skip_benches": sorted(skip),
        "n_rows_cpu": len(cpu_rows),
        "n_rows_gpu": len(gpu_rows),
        "n_rows_cpu_by_tier": {t: len(cpu_rows_by_tier[t]) for t in _CPU_TIERS},
        "n_rows_gpu_by_tier": {t: len(gpu_rows_by_tier[t]) for t in _GPU_TIERS},
        "cpu_jobs_files": {t: cpu_paths[t].name for t in _CPU_TIERS},
        "gpu_jobs_files": {t: gpu_paths[t].name for t in _GPU_TIERS},
        "cpu_pool": args.cpu_pool,
        "gpu_pool": args.gpu_pool,
        "wandb": bool(args.wandb),
        "wandb_project": args.wandb_project,
        "wandb_entity": args.wandb_entity,
        "retry_failed": bool(args.retry_failed),
        "sweep_dir": str(sweep_dir),
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[plan] sweep dir: {sweep_dir}")
    for tier in _CPU_TIERS:
        n = len(cpu_rows_by_tier[tier])
        print(f"[plan] cpu/{tier} rows: {n:4d}  ({cpu_paths[tier].name})")
    for tier in _GPU_TIERS:
        n = len(gpu_rows_by_tier[tier])
        print(f"[plan] gpu/{tier} rows: {n:4d}  ({gpu_paths[tier].name})")
    if gpu_rows_by_tier["L"]:
        print(
            f"[plan] WARNING: {len(gpu_rows_by_tier['L'])} L-tier rows "
            f"(>=40 GB VRAM) are written to {gpu_paths['L'].name} but will not "
            "be submitted by launch_sweep.sh, run them on a larger cluster."
        )
    print()
    print("# Launch commands (run these from the cluster login node):")
    print()
    cpu_sbatch = args.cpu_sbatch or "slurm/run_row_cpu.sbatch"
    gpu_sbatch = args.gpu_sbatch or "slurm/run_row_gpu.sbatch"
    agg_sbatch = args.aggregate_sbatch or "slurm/aggregate.sbatch"
    cpu_tier_flags: dict[str, str] = {
        "cheap": "--partition=<PARTITION> --time=04:00:00",
        "std": "",
    }
    for tier in _CPU_TIERS:
        rows = cpu_rows_by_tier[tier]
        if not rows:
            print(f"CPU_{tier.upper()}_JID=")
            continue
        flags = cpu_tier_flags[tier]
        print(
            f"CPU_{tier.upper()}_JID=$(sbatch --parsable "
            f"--array=0-{len(rows) - 1}%{args.cpu_pool} "
            + (flags + " " if flags else "")
            + f"{cpu_sbatch} {cpu_paths[tier]})"
        )
    tier_flags: dict[Tier, str] = {
        "S": "--gpus=1",
        "M": "--gpus=1 --gres=<GPU_MEM>",
    }
    for tier in ("S", "M"):
        rows = gpu_rows_by_tier[tier]
        if not rows:
            print(f"GPU_{tier}_JID=")
            continue
        print(
            f"GPU_{tier}_JID=$(sbatch --parsable "
            f"--array=0-{len(rows) - 1}%{args.gpu_pool} "
            f"{tier_flags[tier]} "
            f"{gpu_sbatch} {gpu_paths[tier]})"
        )
    print(
        'DEPS=$(echo "$CPU_CHEAP_JID:$CPU_STD_JID:$GPU_S_JID:$GPU_M_JID" '
        "| sed 's/^://; s/:$//; s/::/:/g')\n"
        f"sbatch --dependency=afterany:$DEPS {agg_sbatch} {sweep_dir}"
    )
    return 0


def _cmd_run_row(args: argparse.Namespace) -> int:
    jobs_path = Path(args.jobs_file).resolve()
    rows = _read_jsonl(jobs_path)
    row_idx = int(args.row)
    if row_idx < 0 or row_idx >= len(rows):
        raise SystemExit(f"row {row_idx} out of range (have {len(rows)} rows)")
    row = rows[row_idx]
    method = row["method"]
    bench_id = row["bench_id"]
    seed = int(row["seed"])

    sweep_dir = jobs_path.parent
    runs_root = sweep_dir / "runs"

    prior = _find_prior_run(runs_root, method, bench_id, seed)
    if prior is not None:
        final = _existing_final(prior)
        if final is not None:
            status = str(final.get("status", "")).lower()
            # Skip rows already attempted unless --retry-failed.
            completed_ok = status == "ok" or "feasibility_post" in final
            failed = status == "failed"
            if completed_ok:
                print(f"[run-row {row_idx}] SKIP (prior ok): {prior.name}")
                return 0
            if failed and not args.retry_failed:
                err = final.get("error", "")
                print(
                    f"[run-row {row_idx}] SKIP (prior failed, "
                    f"pass --retry-failed to retry): {prior.name}  "
                    f"error={err!r:.200}"
                )
                return 0

    cmd = [
        sys.executable, "-m", "pal.runner.cli", "run",
        "--method", method,
        "--benchmarks", bench_id,
        "--seeds", str(seed),
        "--device", args.device,
        "--runs-root", str(runs_root),
    ]
    if args.wandb:
        cmd += [
            "--wandb",
            "--wandb-project", args.wandb_project,
        ]
        if args.wandb_entity:
            cmd += ["--wandb-entity", args.wandb_entity]
    if args.viz_train_every > 0:
        cmd += ["--viz-train-every", str(args.viz_train_every)]
    if args.viz_final:
        cmd += ["--viz-final"]
        if args.viz_n > 1:
            cmd += ["--viz-n", str(args.viz_n)]

    print(f"[run-row {row_idx}] {method} / {bench_id} / seed={seed}")
    print(f"[run-row {row_idx}] cmd: {' '.join(cmd)}")
    return subprocess.call(cmd)


def _cmd_aggregate(args: argparse.Namespace) -> int:
    sweep_dir = Path(args.sweep_dir).resolve()
    if not sweep_dir.exists():
        raise SystemExit(f"sweep dir not found: {sweep_dir}")

    manifest_path = sweep_dir / "manifest.json"
    manifest = (
        json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    )
    methods = manifest.get("methods", list(_ALL_METHODS))
    bench_ids = manifest.get("bench_ids", list_all_benchmarks())
    seeds = manifest.get("seeds", [])

    runs_root = sweep_dir / "runs"
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(runs_root.iterdir()) if runs_root.exists() else []:
        if not run_dir.is_dir():
            continue
        final = _existing_final(run_dir)
        if final is None:
            continue
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            continue
        cfg = json.loads(cfg_path.read_text())
        rows.append({
            "run_dir": run_dir.name,
            "method": cfg.get("method"),
            "bench_id": cfg.get("benchmark_id"),
            "seed": int(cfg.get("seed", -1)),
            "final": final,
        })

    results_md = _format_results_md(
        rows=rows, methods=methods, bench_ids=bench_ids, seeds=seeds,
    )
    out_path = sweep_dir / "results.md"
    out_path.write_text(results_md)
    print(f"[aggregate] wrote {out_path}  ({len(rows)} runs read)")
    return 0


def _fmt_pct(x: float) -> str:
    """Feasibility in [0, 1] as a rounded integer percentage, e.g. 0.78 -> '78%'."""
    return f"{round(x * 100)}%"


def _fmt_obj(x: float) -> str:
    """Fixed-point with 2 decimals, e.g. 2.133e-02 -> '0.02', 3.199e+08 -> '319900000.00'.

    Values that round to zero (positive or negative) are normalised to
    '0.00' so tables don't show '-0.00'.
    """
    if round(x, 2) == 0:
        return "0.00"
    return f"{x:.2f}"


def _fmt_sci(x: float) -> str:
    """Compact scientific, 1 decimal, no +0 padding. 14000 -> '1.4e4', 21 -> '2.1e1',
    0.0056 -> '5.6e-3', 0 -> '0'."""
    if x == 0:
        return "0"
    mantissa, exp = f"{x:.1e}".split("e")
    return f"{mantissa}e{int(exp)}"


def _format_results_md(
    rows: list[dict[str, Any]],
    methods: list[str],
    bench_ids: list[str],
    seeds: list[int],
) -> str:
    idx: dict[tuple[str, str, int], dict[str, Any]] = {}
    for r in rows:
        key = (r["method"], r["bench_id"], r["seed"])
        idx[key] = r["final"]

    header = "| bench | " + " | ".join(methods) + " |"
    sep = "|" + "---|" * (1 + len(methods))

    out: list[str] = []
    out.append("# Sweep results\n")
    out.append(f"Methods: {', '.join(methods)}\n")
    out.append(f"Seeds: {seeds}\n")
    out.append(f"Benches: {len(bench_ids)}\n\n")

    out.append("## Headline table (feasibility_post % / obj_mean_post, mean over seeds)\n\n")
    out.append(header + "\n" + sep + "\n")
    for bid in bench_ids:
        cells = [bid]
        for m in methods:
            values = [idx.get((m, bid, s)) for s in seeds]
            feas = [v.get("feasibility_post") for v in values if v is not None]
            feas = [x for x in feas if x is not None]
            obj = [v.get("obj_mean_post") for v in values if v is not None]
            obj = [x for x in obj if x is not None]
            if not feas:
                cells.append("x")
                continue
            mean_feas = sum(feas) / len(feas)
            mean_obj = sum(obj) / len(obj) if obj else float("nan")
            cells.append(f"{_fmt_pct(mean_feas)} / {_fmt_obj(mean_obj)}")
        out.append("| " + " | ".join(cells) + " |\n")

    out.append("\n## Train forward samples (mean over seeds)\n\n")
    out.append(header + "\n" + sep + "\n")
    for bid in bench_ids:
        cells = [bid]
        for m in methods:
            values = [idx.get((m, bid, s)) for s in seeds]
            samples = [v.get("train__fwd_samples", 0) or 0 for v in values if v is not None]
            if not samples:
                cells.append("x")
                continue
            cells.append(_fmt_sci(sum(samples) / len(samples)))
        out.append("| " + " | ".join(cells) + " |\n")

    out.append("\n## Train wall time (mean seconds over seeds)\n\n")
    out.append(header + "\n" + sep + "\n")
    for bid in bench_ids:
        cells = [bid]
        for m in methods:
            values = [idx.get((m, bid, s)) for s in seeds]
            wall = [v.get("train_wall_time_s", 0.0) or 0.0 for v in values if v is not None]
            if not wall:
                cells.append("x")
                continue
            cells.append(f"{sum(wall) / len(wall):.1f}s")
        out.append("| " + " | ".join(cells) + " |\n")

    out.append("\n## Full phase-level metrics\n\n")
    out.append("| method | bench | seed | train_fwd | train_bwd | periodic_eval_fwd | final_eval_fwd | train_s |\n")
    out.append("|---|---|---|---|---|---|---|---|\n")
    for bid in bench_ids:
        for m in methods:
            for s in seeds:
                v = idx.get((m, bid, s))
                if v is None:
                    out.append(f"| {m} | {bid} | {s} | x | x | x | x | x |\n")
                    continue
                out.append(
                    f"| {m} | {bid} | {s} | "
                    f"{v.get('train__fwd_samples', 0)} | "
                    f"{v.get('train__bwd_samples', 0)} | "
                    f"{v.get('periodic_eval__fwd_samples', 0)} | "
                    f"{v.get('final_eval__fwd_samples', 0)} | "
                    f"{v.get('train_wall_time_s', 0.0):.1f} |\n"
                )
    return "".join(out)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="pal sweep", description="PAL benchmark sweep")
    sub = p.add_subparsers(dest="verb", required=True)

    plan = sub.add_parser("plan", help="enumerate rows + write jobs files")
    plan.add_argument("--methods", default=",".join(_ALL_METHODS))
    plan.add_argument("--seeds", default="0")
    plan.add_argument(
        "--skip-benches", default="",
        help="comma-separated bench ids to exclude (e.g. e2/urban_wind)",
    )
    plan.add_argument("--cpu-pool", type=int, default=8, help="SLURM array %% cap for cpu jobs")
    plan.add_argument("--gpu-pool", type=int, default=4, help="SLURM array %% cap for gpu jobs")
    plan.add_argument("--name", default=None, help="override sweep dir name (default: sweep_<ts>)")
    plan.add_argument("--wandb", action="store_true")
    plan.add_argument("--wandb-project", default="pal")
    plan.add_argument("--wandb-entity", default=None)
    plan.add_argument("--retry-failed", action="store_true",
                      help="when run-row sees a failed prior run, retry instead of skipping")
    plan.add_argument("--cpu-sbatch", default=None, help="override path to cpu run_row.sbatch")
    plan.add_argument("--gpu-sbatch", default=None, help="override path to gpu run_row.sbatch")
    plan.add_argument("--aggregate-sbatch", default=None, help="override path to aggregate.sbatch")
    plan.add_argument(
        "--gpu-presets", default=None,
        help="override path to pal/sweep/gpu_presets.yaml (tier registry)",
    )
    plan.set_defaults(func=_cmd_plan)

    rr = sub.add_parser("run-row", help="execute one row from a jobs file")
    rr.add_argument("--jobs-file", required=True)
    rr.add_argument("--row", type=int, required=True)
    rr.add_argument("--device", default="auto")
    rr.add_argument("--wandb", action="store_true")
    rr.add_argument("--wandb-project", default="pal")
    rr.add_argument("--wandb-entity", default=None)
    rr.add_argument("--retry-failed", action="store_true")
    rr.add_argument("--viz-train-every", type=int, default=0)
    rr.add_argument("--viz-final", action="store_true")
    rr.add_argument("--viz-n", type=int, default=1)
    rr.set_defaults(func=_cmd_run_row)

    agg = sub.add_parser("aggregate", help="scan runs/ + write results.md")
    agg.add_argument("--sweep-dir", required=True)
    agg.set_defaults(func=_cmd_aggregate)

    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(list(sys.argv[1:] if argv is None else argv))
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
