"""Export markdown, CSV and LaTeX summary tables from runs.parquet.

Usage:
    python scripts/export_paper_tables.py \
        --runs-parquet /path/to/runs.parquet \
        --out-dir /path/to/results_bundle
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import polars as pl

# Latency is host-specific and the render host may differ from the benchmark host.
_CPU_MODEL_PLACEHOLDER = "<CPU model, fill in for the benchmark host>"


def _cpu_note() -> str:
    hint = platform.processor() or platform.machine()
    if hint:
        return f"{_CPU_MODEL_PLACEHOLDER} (render host reports: {hint})"
    return _CPU_MODEL_PLACEHOLDER


def _format_secs(x: float | None) -> str:
    if x is None or x != x:
        return "-"
    return f"{x:.3g}"


def _format_latency_ms(predict_mean: float | None, n: float | None) -> str:
    if predict_mean is None or predict_mean != predict_mean or not n or n != n:
        return "-"
    return f"{predict_mean / float(n) * 1e3:.3g}"


def _format_obj(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean:+.2f}"
    return f"{mean:+.2f} +/- {std:.2f}"


def _format_viol(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean:.3g}"
    return f"{mean:.3g} +/- {std:.2g}"


def _format_feas(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean * 100:.0f}%"
    return f"{mean * 100:.0f}% +/- {std * 100:.0f}%"


def _format_gap(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "-"
    if std is None or std != std:
        return f"{mean:.3g}"
    return f"{mean:.3g} +/- {std:.2g}"


def _format_obj_tex(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or std != std:
        return f"${mean:+.2f}$"
    return f"${mean:+.2f} \\pm {std:.2f}$"


def _format_viol_tex(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or std != std:
        return f"${mean:.3g}$"
    return f"${mean:.3g} \\pm {std:.2g}$"


def _format_feas_tex(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or std != std:
        return f"${mean * 100:.0f}\\%$"
    return f"${mean * 100:.0f}\\% \\pm {std * 100:.0f}\\%$"


def _format_gap_tex(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    if std is None or std != std:
        return f"${mean:.3g}$"
    return f"${mean:.3g} \\pm {std:.2g}$"


def _escape_tex(text: str) -> str:
    mapping = {
        "\\": "\\textbackslash{}",
        "_": "\\_",
        "&": "\\&",
        "%": "\\%",
        "#": "\\#",
        "{": "\\{",
        "}": "\\}",
    }
    return "".join(mapping.get(ch, ch) for ch in text)


def _load_clean_runs(runs_parquet: Path) -> pl.DataFrame:
    df = pl.read_parquet(runs_parquet)
    df = df.filter((pl.col("status") == "ok") & pl.col("obj_mean_post").is_not_null())
    if df.height == 0:
        return df
    return (
        df.sort("wall_start", descending=True)
        .unique(subset=["method", "benchmark_id", "seed"], keep="first")
        .sort(["benchmark_id", "method", "seed"])
    )


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    if df.height == 0:
        return pl.DataFrame(
            {
                "benchmark_id": [],
                "method": [],
                "n_seeds": [],
                "obj_post_mean": [],
                "obj_post_std": [],
                "viol_post_mean": [],
                "viol_post_std": [],
                "feas_post_mean": [],
                "feas_post_std": [],
                "train_wall_mean": [],
                "predict_wall_mean": [],
                "n_queries_agg": [],
            }
        )

    provenance = (
        {"pal_git_sha": pl.col("pal_git_sha").drop_nulls().first()}
        if "pal_git_sha" in df.columns else {}
    )
    return (
        df.group_by(["benchmark_id", "method"])
        .agg(
            n_seeds=pl.len(),
            obj_post_mean=pl.col("obj_mean_post").mean(),
            obj_post_std=pl.col("obj_mean_post").std(),
            viol_post_mean=pl.col("viol_max_post").mean(),
            viol_post_std=pl.col("viol_max_post").std(),
            feas_post_mean=pl.col("feasibility_post").mean(),
            feas_post_std=pl.col("feasibility_post").std(),
            train_wall_mean=pl.col("train_wall_time_s").mean(),
            predict_wall_mean=pl.col("predict_wall_time_s").mean(),
            n_queries_agg=pl.col("n_queries").max(),
            **provenance,
            device=pl.col("device").drop_nulls().first(),
            family=pl.col("family").drop_nulls().first(),
        )
        .sort(["benchmark_id", "method"])
    )


def _render_markdown(agg: pl.DataFrame, seed_df: pl.DataFrame, runs_parquet: Path) -> str:
    if agg.height == 0:
        return "# Paper Results\n\n_No successful runs with final eval were found._\n"

    methods = sorted(agg.get_column("method").unique().to_list())
    benches = sorted(agg.get_column("benchmark_id").unique().to_list())
    sha = (
        agg.get_column("pal_git_sha").drop_nulls().head(1).to_list()
        if "pal_git_sha" in agg.columns else []
    )

    lines = [
        "# Paper Results",
        "",
        f"Source: `{runs_parquet}`",
        f"Runs used: {seed_df.height} deduplicated successful runs",
        f"Benchmarks: {len(benches)}",
        f"Methods: {len(methods)}",
        *([f"pal_git_sha: `{sha[0][:8]}`"] if sha else []),
        "",
        "Paper-note: post-projection objective, feasibility, and violation come from",
        "`final.json` aggregates. The feasible-only median-gap section appears when",
        "retained runs also have `eval_rows.parquet` and an `f*` reference file is supplied.",
        "",
        "## Post-projection objective",
        "",
    ]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for method in methods:
            cell = agg.filter(
                (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_obj(r["obj_post_mean"], r["obj_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Post-projection feasibility", ""]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for method in methods:
            cell = agg.filter(
                (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_feas(r["feas_post_mean"], r["feas_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    lines += ["", "## Post-projection violation", ""]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for method in methods:
            cell = agg.filter(
                (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
            )
            if cell.height == 0:
                row.append("-")
            else:
                r = cell.row(0, named=True)
                row.append(_format_viol(r["viol_post_mean"], r["viol_post_std"]))
        lines.append("| " + " | ".join(row) + " |")

    has_predict_wall = "predict_wall_mean" in agg.columns
    has_n_queries = "n_queries_agg" in agg.columns

    if "train_wall_mean" in agg.columns:
        lines += ["", "## Mean training wall time (seconds across seeds)", ""]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for method in methods:
                cell = agg.filter(
                    (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    row.append(_format_secs(cell.row(0, named=True)["train_wall_mean"]))
            lines.append("| " + " | ".join(row) + " |")

    if has_predict_wall:
        lines += ["", "## Mean prediction wall time (seconds across seeds)", ""]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for method in methods:
                cell = agg.filter(
                    (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    row.append(_format_secs(cell.row(0, named=True)["predict_wall_mean"]))
            lines.append("| " + " | ".join(row) + " |")

    if has_predict_wall and has_n_queries:
        n_by_bench: dict[str, int] = {}
        for bench in benches:
            cell = agg.filter(pl.col("benchmark_id") == bench).filter(
                pl.col("n_queries_agg").is_not_null()
            )
            if cell.height:
                n_by_bench[bench] = int(cell.get_column("n_queries_agg").max())
        n_note = ", ".join(f"{b}={n}" for b, n in sorted(n_by_bench.items())) or "n/a"
        lines += [
            "",
            "## Amortized per-query latency (ms/query, end-to-end incl. model load)",
            "",
            "End-to-end amortized latency per query = `predict_wall / n_queries`, "
            "where `n_queries` is the eval batch size n. Includes model load: "
            "`predict()` rebuilds the model inside the timed region for several "
            "solvers (known and accepted; timers unchanged). Comparable only within "
            f"one host, CPU: {_cpu_note()}. Eval batch size n per benchmark: {n_note}.",
            "",
        ]
        lines.append("| benchmark | " + " | ".join(methods) + " |")
        lines.append("|" + "---|" * (len(methods) + 1))
        for bench in benches:
            row = [bench]
            for method in methods:
                cell = agg.filter(
                    (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
                )
                if cell.height == 0:
                    row.append("-")
                else:
                    r = cell.row(0, named=True)
                    row.append(_format_latency_ms(r["predict_wall_mean"], r.get("n_queries_agg")))
            lines.append("| " + " | ".join(row) + " |")

    lines += [
        "",
        "## Files",
        "",
        "- `paper_results_summary.csv`: aggregated method x benchmark table",
        "- `paper_results_seed_level.csv`: one row per retained run/seed",
        "- `latex/*.tex`: paper-friendly tabular fragments from the same retained runs",
        "- `paper_results_gap_*.csv`: emitted when `eval_rows.parquet` + `f*` references are available",
        "",
    ]
    return "\n".join(lines) + "\n"


def _metric_columns(metric: str) -> tuple[str, str, str]:
    if metric == "objective":
        return "obj_post_mean", "obj_post_std", "Post-projection objective"
    if metric == "feasibility":
        return "feas_post_mean", "feas_post_std", "Post-projection feasibility"
    if metric == "violation":
        return "viol_post_mean", "viol_post_std", "Post-projection violation"
    if metric == "median_gap":
        return "gap_post_mean", "gap_post_std", "Feasible-only median optimality gap"
    raise ValueError(f"unsupported metric: {metric}")


def _format_tex(metric: str, mean: float | None, std: float | None) -> str:
    if metric == "objective":
        return _format_obj_tex(mean, std)
    if metric == "feasibility":
        return _format_feas_tex(mean, std)
    if metric == "violation":
        return _format_viol_tex(mean, std)
    if metric == "median_gap":
        return _format_gap_tex(mean, std)
    raise ValueError(f"unsupported metric: {metric}")


def _render_latex_table(agg: pl.DataFrame, metric: str, title: str | None = None) -> str:
    mean_col, std_col, default_title = _metric_columns(metric)
    if agg.height == 0:
        return "% No successful runs available.\n"

    methods = sorted(agg.get_column("method").unique().to_list())
    benches = sorted(agg.get_column("benchmark_id").unique().to_list())
    heading = title or default_title

    lines = [
        f"% {heading}",
        "\\begin{tabular}{l" + "c" * len(methods) + "}",
        "\\toprule",
        "Benchmark & " + " & ".join(_escape_tex(m) for m in methods) + " \\\\",
        "\\midrule",
    ]
    for bench in benches:
        row = [_escape_tex(bench)]
        for method in methods:
            cell = agg.filter(
                (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
            )
            if cell.height == 0:
                row.append("--")
            else:
                rec = cell.row(0, named=True)
                row.append(_format_tex(metric, rec[mean_col], rec[std_col]))
        lines.append(" & ".join(row) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    return "\n".join(lines)


def _family_subset(agg: pl.DataFrame, family_prefixes: tuple[str, ...]) -> pl.DataFrame:
    if agg.height == 0:
        return agg
    return agg.filter(pl.col("family").is_in(family_prefixes))


def _default_f_star_path() -> Path:
    return Path(__file__).resolve().parents[1] / "paper_tables" / "f_star.json"


F_STAR_REL_GAP_MIN_ABS = 1e-8


def _load_f_star(path: Path | None) -> dict[str, float]:
    if path is None or not path.exists():
        return {}
    raw = json.loads(path.read_text())
    out: dict[str, float] = {}
    for bench_id, payload in raw.items():
        if isinstance(payload, dict):
            value = payload.get("f_star")
            feasible = payload.get("feasible", True)
            if feasible is False or value is None:
                continue
            out[str(bench_id)] = float(value)
        else:
            out[str(bench_id)] = float(payload)
    return out


def _load_eval_rows(seed_df: pl.DataFrame) -> tuple[pl.DataFrame | None, list[str]]:
    if seed_df.height == 0 or "run_path" not in seed_df.columns:
        return None, []
    frames: list[pl.DataFrame] = []
    warnings: list[str] = []
    for row in seed_df.select(["run_id", "run_path"]).iter_rows(named=True):
        path = Path(row["run_path"]) / "eval_rows.parquet"
        if not path.exists():
            warnings.append(f"missing eval_rows.parquet for {row['run_id']}")
            continue
        frames.append(pl.read_parquet(path))
    if not frames:
        return None, warnings
    return pl.concat(frames, how="diagonal_relaxed"), warnings


def _build_gap_tables(
    seed_df: pl.DataFrame,
    *,
    f_star: dict[str, float],
) -> tuple[pl.DataFrame | None, pl.DataFrame | None, list[str]]:
    eval_rows, warnings = _load_eval_rows(seed_df)
    if eval_rows is None or eval_rows.height == 0 or not f_star:
        return None, None, warnings

    f_star_df = pl.DataFrame(
        {
            "benchmark_id": list(f_star.keys()),
            "f_star": list(f_star.values()),
        }
    )
    rows = eval_rows.join(f_star_df, on="benchmark_id", how="inner").with_columns(
        pl.col("f_star").abs().alias("f_star_abs"),
    )
    near_zero_benches = (
        rows.filter(pl.col("f_star_abs") <= F_STAR_REL_GAP_MIN_ABS)
        .select("benchmark_id")
        .unique()
        .sort("benchmark_id")
        .get_column("benchmark_id")
        .to_list()
    )
    for bench_id in near_zero_benches:
        warnings.append(
            f"skipped median-gap for {bench_id}: |f*| <= {F_STAR_REL_GAP_MIN_ABS:g} makes relative gap ill-posed"
        )
    rows = (
        rows
        .filter(pl.col("feasible_post"))
        .filter(pl.col("f_star_abs") > F_STAR_REL_GAP_MIN_ABS)
        .with_columns(
            ((pl.col("obj_post") - pl.col("f_star")) / pl.col("f_star_abs")).alias("optimality_gap_post")
        )
    )
    if rows.height == 0:
        return None, None, warnings

    seed_gap = (
        rows.group_by(["run_id", "method", "benchmark_id", "seed"])
        .agg(
            median_gap_post=pl.col("optimality_gap_post").median(),
            n_feasible_queries=pl.len(),
        )
        .sort(["benchmark_id", "method", "seed"])
    )
    gap_summary = (
        seed_gap.group_by(["benchmark_id", "method"])
        .agg(
            n_seeds=pl.len(),
            gap_post_mean=pl.col("median_gap_post").mean(),
            gap_post_std=pl.col("median_gap_post").std(),
        )
        .sort(["benchmark_id", "method"])
    )
    return seed_gap, gap_summary, warnings


def _append_metric_markdown(lines: list[str], agg: pl.DataFrame, metric: str, heading: str) -> None:
    if agg.height == 0:
        return
    methods = sorted(agg.get_column("method").unique().to_list())
    benches = sorted(agg.get_column("benchmark_id").unique().to_list())
    mean_col, std_col, _ = _metric_columns(metric)
    lines += ["", f"## {heading}", ""]
    lines.append("| benchmark | " + " | ".join(methods) + " |")
    lines.append("|" + "---|" * (len(methods) + 1))
    for bench in benches:
        row = [bench]
        for method in methods:
            cell = agg.filter(
                (pl.col("benchmark_id") == bench) & (pl.col("method") == method)
            )
            if cell.height == 0:
                row.append("-")
            else:
                rec = cell.row(0, named=True)
                if metric == "median_gap":
                    row.append(_format_gap(rec[mean_col], rec[std_col]))
                else:
                    raise ValueError(f"unexpected metric for markdown append: {metric}")
        lines.append("| " + " | ".join(row) + " |")


def _remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="export paper-facing summary tables from runs.parquet")
    p.add_argument("--runs-parquet", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument(
        "--f-star",
        default=None,
        help="optional path to paper_tables/f_star.json (default: repo-local paper_tables/f_star.json if present)",
    )
    args = p.parse_args(argv)

    runs_parquet = Path(args.runs_parquet)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seed_df = _load_clean_runs(runs_parquet)
    agg = _aggregate(seed_df)
    f_star_path = Path(args.f_star) if args.f_star else _default_f_star_path()
    f_star = _load_f_star(f_star_path if f_star_path.exists() else None)
    gap_seed_df, gap_summary, gap_warnings = _build_gap_tables(seed_df, f_star=f_star)

    seed_csv = out_dir / "paper_results_seed_level.csv"
    summary_csv = out_dir / "paper_results_summary.csv"
    summary_md = out_dir / "paper_results.md"
    gap_seed_csv = out_dir / "paper_results_gap_seed_level.csv"
    gap_summary_csv = out_dir / "paper_results_gap_summary.csv"
    latex_dir = out_dir / "latex"
    latex_dir.mkdir(parents=True, exist_ok=True)
    gap_latex_names = (
        "median_gap_all.tex",
        "median_gap_synthetic.tex",
        "median_gap_engineering.tex",
    )

    seed_df.write_csv(seed_csv)
    agg.write_csv(summary_csv)
    markdown = _render_markdown(agg, seed_df, runs_parquet).rstrip()
    markdown_lines = markdown.splitlines()
    if gap_summary is not None and gap_summary.height > 0:
        _append_metric_markdown(
            markdown_lines,
            gap_summary,
            metric="median_gap",
            heading="Feasible-only median optimality gap (mean +/- std across seeds)",
        )
    if f_star:
        markdown_lines += ["", "## f* References", ""]
        markdown_lines.append(f"Loaded from `{f_star_path}`.")
    if gap_warnings:
        markdown_lines += ["", "## Warnings", ""]
        for warning in gap_warnings:
            markdown_lines.append(f"- {warning}")
    summary_md.write_text("\n".join(markdown_lines) + "\n")

    synthetic = _family_subset(agg, tuple(f"b{i:02d}" for i in range(1, 12)))
    engineering = (
        agg.filter(pl.col("family").str.starts_with("b1"))
        if agg.height else agg
    )
    synthetic_gap = None if gap_summary is None else _family_subset(
        gap_summary.join(
            agg.select(["benchmark_id", "family"]).unique(),
            on="benchmark_id",
            how="left",
        ),
        tuple(f"b{i:02d}" for i in range(1, 12)),
    )
    engineering_gap = (
        None if gap_summary is None else gap_summary.join(
            agg.select(["benchmark_id", "family"]).unique(),
            on="benchmark_id",
            how="left",
        ).filter(pl.col("family").str.starts_with("b1"))
    )
    latex_outputs = {
        "objective_all.tex": _render_latex_table(agg, metric="objective"),
        "feasibility_all.tex": _render_latex_table(agg, metric="feasibility"),
        "violation_all.tex": _render_latex_table(agg, metric="violation"),
        "feasibility_synthetic.tex": _render_latex_table(
            synthetic, metric="feasibility", title="Synthetic feasibility"
        ),
        "objective_engineering.tex": _render_latex_table(
            engineering, metric="objective", title="Engineering objective"
        ),
        "feasibility_engineering.tex": _render_latex_table(
            engineering, metric="feasibility", title="Engineering feasibility"
        ),
        "violation_engineering.tex": _render_latex_table(
            engineering, metric="violation", title="Engineering violation"
        ),
    }
    if gap_summary is not None and gap_summary.height > 0:
        gap_with_family = gap_summary.join(
            agg.select(["benchmark_id", "family"]).unique(),
            on="benchmark_id",
            how="left",
        )
        latex_outputs.update(
            {
                "median_gap_all.tex": _render_latex_table(
                    gap_summary, metric="median_gap", title="Feasible-only median optimality gap"
                ),
                "median_gap_synthetic.tex": _render_latex_table(
                    synthetic_gap if synthetic_gap is not None else gap_with_family.head(0),
                    metric="median_gap",
                    title="Synthetic feasible-only median gap",
                ),
                "median_gap_engineering.tex": _render_latex_table(
                    engineering_gap if engineering_gap is not None else gap_with_family.head(0),
                    metric="median_gap",
                    title="Engineering feasible-only median gap",
                ),
            }
        )
    else:
        _remove_if_exists(gap_seed_csv)
        _remove_if_exists(gap_summary_csv)
        for name in gap_latex_names:
            _remove_if_exists(latex_dir / name)
    for name, text in latex_outputs.items():
        (latex_dir / name).write_text(text)
    if gap_seed_df is not None and gap_summary is not None:
        gap_seed_df.write_csv(gap_seed_csv)
        gap_summary.write_csv(gap_summary_csv)

    print(f"[paper] wrote {summary_md}")
    print(f"[paper] wrote {summary_csv}")
    print(f"[paper] wrote {seed_csv}")
    if gap_seed_df is not None and gap_summary is not None:
        print(f"[paper] wrote {gap_summary_csv}")
        print(f"[paper] wrote {gap_seed_csv}")
    for name in sorted(latex_outputs):
        print(f"[paper] wrote {latex_dir / name}")
    if f_star:
        print(f"[paper] used f* refs from {f_star_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
