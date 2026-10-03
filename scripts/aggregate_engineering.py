"""Build the per-bench gap-to-IPOPT parquet for one engineering benchmark.

`gap_rel = (obj_method - obj_ipopt) / |obj_ipopt|` against the best IPOPT obj
for the bench, so negative means the method beat IPOPT.

Usage:
    python scripts/aggregate_engineering.py --bench e3/acopf_ieee57
    python scripts/aggregate_engineering.py --bench e1/bwb \\
        --runs-root /scratch/pal/runs \\
        --out results/engineering/e1_bwb_gap.parquet
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from datetime import date
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate import aggregate as scan_runs  # noqa: E402

from pal.baselines.ipopt.shard_merge import (  # noqa: E402
    aggregate_post_metrics,
    merge_per_query,
)


def _safe_bench(bench: str) -> str:
    return bench.replace("/", "_")


def _read_metrics_mem_stats(run_path: Path) -> tuple[float, float]:
    """Return (mean, std) of `repair_peak_mem_bytes` in `metrics.jsonl`, NaN if absent."""
    p = run_path / "metrics.jsonl"
    if not p.exists():
        return math.nan, math.nan

    vals: list[float] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        v = row.get("repair_peak_mem_bytes")
        if v is None:
            continue
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(f):
            continue
        vals.append(f)

    if not vals:
        return math.nan, math.nan
    if len(vals) == 1:
        return vals[0], math.nan
    return statistics.fmean(vals), statistics.stdev(vals)


def _read_per_query_paper_metrics(run_path: Path) -> tuple[float, float]:
    """Return (mean(max_eq_post), mean(max_ineq_post)) from `eval_rows.parquet`, NaN if absent."""
    p = run_path / "eval_rows.parquet"
    if not p.exists():
        return math.nan, math.nan
    df = pl.read_parquet(p)
    if df.height == 0:
        return math.nan, math.nan
    eq = (
        float(df.get_column("max_eq_post").mean())
        if "max_eq_post" in df.columns else math.nan
    )
    ineq = (
        float(df.get_column("max_ineq_post").mean())
        if "max_ineq_post" in df.columns else math.nan
    )
    return eq, ineq


def _merge_ipopt_restart_shards(
    df: pl.DataFrame,
    runs_root: Path,
    *,
    allow_partial_shards: bool = False,
) -> pl.DataFrame:
    """Merge IPOPT restart shards per (benchmark_id, seed) into one row over all restarts.

    An incomplete shard set raises unless `allow_partial_shards=True`.
    """
    if "restart_shard_r" not in df.columns:
        return df

    sharded = df.filter(
        (pl.col("method") == "ipopt") & pl.col("restart_shard_r").is_not_null()
    )
    if sharded.height == 0:
        return df

    untouched = df.filter(
        ~((pl.col("method") == "ipopt") & pl.col("restart_shard_r").is_not_null())
    )

    merged_rows: list[dict] = []
    for (bench_id, seed), group in sharded.group_by(["benchmark_id", "seed"]):
        Rs = set(group.get_column("restart_shard_R").to_list())
        if len(Rs) != 1:
            raise ValueError(
                f"IPOPT shards for ({bench_id}, seed={seed}) carry inconsistent "
                f"restart_shard_R values {Rs}, mixed fan-out factors in the "
                "same runs_root. Move stale runs out of the way and re-aggregate."
            )
        R = next(iter(Rs))
        partial = group.height < R
        if partial and not allow_partial_shards:
            raise ValueError(
                f"IPOPT shard set for ({bench_id}, seed={seed}) is incomplete: "
                f"only {group.height}/{R} shards present. Refusing to merge a "
                "partial pool into a row labeled 'IPOPT @ multi_start=N'. "
                "Resubmit the missing shards, or pass --allow-partial-shards "
                "to merge the partial set (the row will be flagged "
                "partial_shards=True)."
            )
        if partial:
            print(
                f"[aggregate-eng] WARNING: only {group.height}/{R} IPOPT shards "
                f"present for ({bench_id}, seed={seed}); merging the partial set "
                "(--allow-partial-shards in effect).",
                file=sys.stderr,
            )

        shard_per_queries = []
        run_paths = group.get_column("run_path").to_list()
        for rp in run_paths:
            final = json.loads((Path(rp) / "final.json").read_text())
            pq = final.get("per_query")
            if not pq:
                raise ValueError(
                    f"IPOPT shard {rp} missing per_query in final.json, "
                    "per-query records are required to merge shards."
                )
            shard_per_queries.append(pq)
        merged = merge_per_query(shard_per_queries)
        post = aggregate_post_metrics(merged)

        # Latest shard is the template, post-eval metrics come from the merged pool.
        latest = (
            group.sort("wall_start", descending=True).head(1).to_dicts()[0]
        )
        latest["obj_mean_post"] = post["obj_mean_post"]
        latest["viol_max_post"] = post["viol_max_post"]
        latest["feasibility_post"] = post["feasibility_post"]
        latest["n_restarts"] = sum(len(spq[0].get("restarts", [])) for spq in shard_per_queries)
        latest["restart_shard_r"] = None
        latest["restart_shard_R"] = R
        latest["partial_shards"] = partial
        merged_rows.append(latest)

    if not merged_rows:
        return untouched
    return pl.concat([untouched, pl.from_dicts(merged_rows)], how="diagonal_relaxed")


def build_gap(
    runs_root: Path,
    bench: str,
    *,
    allow_partial_shards: bool = False,
) -> pl.DataFrame:
    df = scan_runs(runs_root)
    if df.height == 0:
        return df
    # Merge before filtering so the merged row carries metrics from the full pool.
    df = _merge_ipopt_restart_shards(
        df, runs_root, allow_partial_shards=allow_partial_shards
    )
    df = df.filter(
        (pl.col("status") == "ok")
        & (pl.col("benchmark_id") == bench)
        & pl.col("obj_mean_post").is_not_null()
    )
    if df.height == 0:
        return df

    df = df.sort("wall_start", descending=True).unique(
        subset=["method", "benchmark_id", "seed"], keep="first"
    )

    mem_means: list[float] = []
    mem_stds: list[float] = []
    max_eq_means: list[float] = []
    max_ineq_means: list[float] = []
    for run_path in df.get_column("run_path").to_list():
        m, s = _read_metrics_mem_stats(Path(run_path))
        mem_means.append(m)
        mem_stds.append(s)
        eq_m, ineq_m = _read_per_query_paper_metrics(Path(run_path))
        max_eq_means.append(eq_m)
        max_ineq_means.append(ineq_m)
    df = df.with_columns(
        pl.Series("repair_peak_mem_bytes_mean", mem_means),
        pl.Series("repair_peak_mem_bytes_std", mem_stds),
        pl.Series("max_eq_post_mean", max_eq_means),
        pl.Series("max_ineq_post_mean", max_ineq_means),
    )

    # Best IPOPT obj is broadcast to every seed: the eval point is deterministic,
    # IPOPT seeds only rekey the multistart RNG.
    ipopt = (
        df.filter(pl.col("method") == "ipopt")
        .group_by("benchmark_id")
        .agg(obj_ipopt=pl.col("obj_mean_post").min())
        .rename({"benchmark_id": "bench"})
    )

    out = df.select(
        bench=pl.col("benchmark_id"),
        method=pl.col("method"),
        seed=pl.col("seed"),
        obj_post=pl.col("obj_mean_post"),
        feas_post=pl.col("feasibility_post"),
        viol_max_post=pl.col("viol_max_post"),
        max_eq_post_mean=pl.col("max_eq_post_mean"),
        max_ineq_post_mean=pl.col("max_ineq_post_mean"),
        repair_peak_mem_bytes_mean=pl.col("repair_peak_mem_bytes_mean"),
        repair_peak_mem_bytes_std=pl.col("repair_peak_mem_bytes_std"),
    ).join(ipopt, on=["bench"], how="left")

    out = out.with_columns(
        gap_rel=pl.when(pl.col("obj_ipopt").is_null())
        .then(None)
        .otherwise(
            (pl.col("obj_post") - pl.col("obj_ipopt"))
            / pl.col("obj_ipopt").abs()
        ),
    )
    return out.sort(["method", "seed"])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="aggregate runs into a per-bench gap-to-IPOPT parquet"
    )
    p.add_argument(
        "--bench", required=True,
        help="single benchmark id (e.g. e3/acopf_ieee57). 'all', comma-lists, "
             "and family prefixes are rejected, render one table at a time.",
    )
    p.add_argument("--runs-root", default="runs", help="run-dir root (default: runs)")
    p.add_argument(
        "--out", default=None,
        help="output parquet path (default: "
             "results/<today>/engineering/<safe_bench>_gap.parquet)",
    )
    p.add_argument(
        "--allow-partial-shards", action="store_true",
        help="merge IPOPT restart-shard sets even when k<R shards are present. "
             "Default behavior fails loud so the parquet doesn't ship "
             "'IPOPT @ multi_start=N' derived from a fraction of restarts. "
             "Synthesized rows from a partial set get partial_shards=True.",
    )
    args = p.parse_args(argv)

    bench = args.bench.strip()
    if "," in bench or bench.lower() == "all":
        print(
            f"[aggregate-eng] --bench must be a single id, got {bench!r}; "
            "render one bench at a time.",
            file=sys.stderr,
        )
        return 2
    if "/" not in bench and re.fullmatch(r"e\d+", bench):
        # Engineering benches are e<N>/<variant>, synthetic ones have no slash.
        print(
            f"[aggregate-eng] --bench {bench!r} looks like a family prefix; "
            "supply the full id including the variant (e.g. e3/acopf_ieee57).",
            file=sys.stderr,
        )
        return 2

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        print(f"[aggregate-eng] runs root not found: {runs_root}", file=sys.stderr)
        return 1

    today = date.today().isoformat()
    safe = _safe_bench(bench)
    out = (
        Path(args.out)
        if args.out
        else Path("results") / today / "engineering" / f"{safe}_gap.parquet"
    )

    df = build_gap(runs_root, bench, allow_partial_shards=args.allow_partial_shards)
    if df.height == 0:
        print(
            f"[aggregate-eng] no ok runs for bench={bench!r} under {runs_root}; "
            "writing empty parquet.",
            file=sys.stderr,
        )

    n_ipopt = df.filter(pl.col("method") == "ipopt").height if df.height else 0
    if df.height and n_ipopt == 0:
        print(
            f"[aggregate-eng] WARNING: no IPOPT runs for {bench!r}; gap_rel "
            "will be null for every row.",
            file=sys.stderr,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(out)

    methods = sorted(df.get_column("method").unique().to_list()) if df.height else []
    print(
        f"[aggregate-eng] {out} ({df.height} rows, {len(methods)} methods: "
        f"{', '.join(methods) or '-'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
