"""Render a method x benchmark results table (markdown) from runs.parquet.

Usage: python scripts/render_results.py --runs-parquet runs.parquet --out results/table.md
"""

from __future__ import annotations

import argparse
import platform
import sys
from datetime import date
from pathlib import Path

import polars as pl

# CPU placeholder: latency is only comparable within one host, so leave a fill-in slot.
_CPU_MODEL_PLACEHOLDER = "<CPU model, fill in for the benchmark host>"


def _cpu_note() -> str:
    hint = platform.processor() or platform.machine()
    if hint:
        return f"{_CPU_MODEL_PLACEHOLDER} (render host reports: {hint})"
    return _CPU_MODEL_PLACEHOLDER


def _format_obj(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:  # NaN guard (single seed)
        return f"{mean:+.2f}"
    return f"{mean:+.2f} +/- {std:.2f}"


def _format_viol(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean:.2f}"
    return f"{mean:.2f} +/- {std:.2f}"


def _format_feas(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean * 100:.0f}%"
    return f"{mean * 100:.0f}% +/- {std * 100:.0f}%"


def _format_secs(x: float | None) -> str:
    """Seconds with 3 significant figures (sub-second walls would collapse under `.2f`)."""
    if x is None or x != x:
        return "-"
    return f"{x:.3g}"


def _format_latency_ms(predict_mean: float | None, n: float | None) -> str:
    """End-to-end amortized latency per query, in ms = predict_wall / n * 1e3."""
    if predict_mean is None or predict_mean != predict_mean or not n or n != n:
        return "-"
    return f"{predict_mean / float(n) * 1e3:.3g}"


def _format_inf_iters(median_mean: float | None, max_mean: float | None) -> str:
    """`{median} ({max})` rounded to nearest int."""
    if median_mean is None or median_mean != median_mean:
        return "-"
    median_str = f"{median_mean:.0f}"
    if max_mean is None or max_mean != max_mean:
        return median_str
    return f"{median_str} ({max_mean:.0f})"


def render(runs_parquet: Path, out_path: Path, methods_filter: list[str] | None = None) -> None:
    df = pl.read_parquet(runs_parquet)

    # Drop partial runs (status missing/failed or no final.json, obj null).
    df = df.filter(
        (pl.col("status") == "ok") & pl.col("obj_mean_post").is_not_null()
    )
    if methods_filter:
        df = df.filter(pl.col("method").is_in(methods_filter))

    if df.height == 0:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("# Results\n\n_No runs with final eval found in `runs.parquet`._\n")
        print(f"[render] {out_path} (empty: no usable runs)")
        return

    df = (
        df.sort("wall_start", descending=True)
        .unique(subset=["method", "benchmark_id", "seed"], keep="first")
    )

    # Aggregate over seeds; `inf_iters_*` columns may be missing in older parquets.
    schema_cols = set(df.columns)
    inf_iters_aggs: dict[str, pl.Expr] = {}
    if "inf_iters_median" in schema_cols:
        inf_iters_aggs["inf_iters_median_mean"] = pl.col("inf_iters_median").mean()
    if "inf_iters_max" in schema_cols:
        inf_iters_aggs["inf_iters_max_mean"] = pl.col("inf_iters_max").mean()
    # Timing columns, guarded the same way.
    timing_aggs: dict[str, pl.Expr] = {}
    has_predict_wall = "predict_wall_time_s" in schema_cols
    has_n_queries = "n_queries" in schema_cols
    if has_predict_wall:
        timing_aggs["predict_wall_mean"] = pl.col("predict_wall_time_s").mean()
    if has_n_queries:
        # Constant per benchmark; max ignores stray nulls.
        timing_aggs["n_queries_agg"] = pl.col("n_queries").max()
    grouped = (
        df.group_by(["method", "benchmark_id"])
        .agg(
            n_seeds=pl.len(),
            obj_post_mean=pl.col("obj_mean_post").mean(),
            obj_post_std=pl.col("obj_mean_post").std(),
            viol_post_mean=pl.col("viol_max_post").mean(),
            viol_post_std=pl.col("viol_max_post").std(),
            feas_post_mean=pl.col("feasibility_post").mean(),
            feas_post_std=pl.col("feasibility_post").std(),
            train_wall_mean=pl.col("train_wall_time_s").mean(),
            **inf_iters_aggs,
            **timing_aggs,
        )
        .sort(["benchmark_id", "method"])
    )

    methods = sorted(df.get_column("method").unique().to_list())
    benches = sorted(df.get_column("benchmark_id").unique().to_list())

    today = date.today().isoformat()
    sha_note = ""
    if "pal_git_sha" in df.columns:
        pal_git_sha = df.get_column("pal_git_sha").drop_nulls().head(1).to_list()
        if pal_git_sha:
            sha_note = f" pal_git_sha: `{pal_git_sha[0][:8]}`."

    lines = [
        f"# Results: {today}",
        "",
        f"Aggregated from `runs.parquet` ({df.height} runs across "
        f"{len(methods)} methods x {len(benches)} benchmarks).{sha_note}",
        "",
        "## Post-projection objective (mean +/- std across seeds)",
        "",
    ]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for m in methods:
            cell = grouped.filter(
                (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_obj(r["obj_post_mean"], r["obj_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Post-projection feasibility (% of eval queries with `max_viol < tolerance`)",
        "",
    ]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for m in methods:
            cell = grouped.filter(
                (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_feas(r["feas_post_mean"], r["feas_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Post-projection violation (mean +/- std across seeds)",
        "",
    ]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for m in methods:
            cell = grouped.filter(
                (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_viol(r["viol_post_mean"], r["viol_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Mean training wall time (seconds across seeds)",
        "",
    ]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for m in methods:
            cell = grouped.filter(
                (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
            )
            if cell.height == 0:
                row.append("-")
            else:
                row.append(f"{cell.row(0, named=True)['train_wall_mean']:.2f}")
        lines.append("| " + " | ".join(row) + " |")

    if has_predict_wall:
        lines += [
            "",
            "## Mean prediction wall time (seconds across seeds)",
            "",
        ]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for m in methods:
                cell = grouped.filter(
                    (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    row.append(_format_secs(cell.row(0, named=True)["predict_wall_mean"]))
            lines.append("| " + " | ".join(row) + " |")

    if has_predict_wall and has_n_queries:
        # n is constant across methods on a bench.
        n_by_bench: dict[str, int] = {}
        for bench in benches:
            cell = grouped.filter(pl.col("benchmark_id") == bench).filter(
                pl.col("n_queries_agg").is_not_null()
            )
            if cell.height:
                n_by_bench[bench] = int(cell.get_column("n_queries_agg").max())
        n_note = ", ".join(f"{b}={n}" for b, n in sorted(n_by_bench.items())) or "n/a"
        lines += [
            "",
            "## Amortized per-query latency (ms/query, end-to-end incl. model load)",
            "",
            "_End-to-end amortized latency per query = `predict_wall / n_queries`, "
            "where `n_queries` is the eval batch size n. This is the full "
            "`predict()` cost including model load: `predict()` rebuilds the model "
            "inside the timed region for several solvers (known and accepted; timers "
            "unchanged). Comparable only within one host, "
            f"CPU: {_cpu_note()}. Eval batch size n per benchmark: {n_note}._",
            "",
        ]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for m in methods:
                cell = grouped.filter(
                    (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    r = cell.row(0, named=True)
                    row.append(_format_latency_ms(r["predict_wall_mean"], r.get("n_queries_agg")))
            lines.append("| " + " | ".join(row) + " |")

    if inf_iters_aggs:
        lines += [
            "",
            "## Inference projection steps to feasibility (median (max), batch-wide early exit)",
            "",
            "_Repair-loop step count under each method's own batch-wide convergence "
            "criterion. `-` for solvers without an inference repair loop (alm, ipopt, "
            "slsqp) or for runs predating the metric._",
            "",
        ]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for m in methods:
                cell = grouped.filter(
                    (pl.col("method") == m) & (pl.col("benchmark_id") == bench)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    r = cell.row(0, named=True)
                    row.append(_format_inf_iters(
                        r.get("inf_iters_median_mean"),
                        r.get("inf_iters_max_mean"),
                    ))
            lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "---",
        "",
        f"_Generated by `scripts/render_results.py` from `{runs_parquet.name}`._",
        "",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(
        f"[render] {out_path} ({df.height} runs, {len(methods)} methods x {len(benches)} benchmarks)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="render method x benchmark results table")
    p.add_argument("--runs-parquet", default="runs.parquet")
    p.add_argument("--out", default=None, help="default: results/<today>.md")
    p.add_argument("--methods", default=None,
                   help="comma-separated method subset (default: all methods in parquet)")
    args = p.parse_args(argv)

    runs_path = Path(args.runs_parquet)
    if not runs_path.exists():
        print(f"[render] runs.parquet not found at {runs_path}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else Path("results") / f"{date.today().isoformat()}.md"
    methods_filter = (
        [m.strip() for m in args.methods.split(",") if m.strip()] if args.methods else None
    )
    render(runs_path, out_path, methods_filter=methods_filter)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
