"""Render one engineering benchmark's table (md + tex) from `<bench>_gap.parquet`.

Cross-seed mean +/- std per method. `max_eq`/`max_ineq` are per-run means of per-instance max
violation, as in DC3 Table 3. The IPOPT row has gap 0 by construction.

Usage: python scripts/render_engineering.py --bench e3/acopf_ieee57 [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import math
import sys
from datetime import date
from pathlib import Path

import polars as pl

# Canonical row order; methods missing from a bench's parquet are skipped.
METHODS_ORDER: tuple[str, ...] = (
    "pal_loggap",
    "alm",
    "alm_bolton",
    "dc3",
    "enforce_orig",
    "fsnet",
    "snarenet",
    "ipopt",
)

_GIB = 1024 ** 3


def _safe_bench(bench: str) -> str:
    return bench.replace("/", "_")


def _is_nan(x: float | None) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


def _fmt_signed(mean: float | None, std: float | None) -> str:
    if _is_nan(mean):
        return "-"
    sci = abs(mean) >= 100 or (mean != 0 and abs(mean) < 1e-2)
    spec = "+.3e" if sci else "+.4f"
    if _is_nan(std):
        return format(mean, spec)
    spec_std = ".3e" if sci else ".4f"
    return f"{format(mean, spec)} +/- {format(std, spec_std)}"


def _fmt_unsigned(mean: float | None, std: float | None) -> str:
    if _is_nan(mean):
        return "-"
    if _is_nan(std):
        return f"{mean:.3e}"
    return f"{mean:.3e} +/- {std:.3e}"


def _fmt_obj(mean: float | None, std: float | None) -> str:
    """Like _fmt_signed but without a forced sign."""
    if _is_nan(mean):
        return "-"
    sci = abs(mean) >= 100 or (mean != 0 and abs(mean) < 1e-2)
    spec = ".3e" if sci else ".4f"
    if _is_nan(std):
        return format(mean, spec)
    return f"{format(mean, spec)} +/- {format(std, spec)}"


def _fmt_pct(mean: float | None, std: float | None) -> str:
    if _is_nan(mean):
        return "-"
    if _is_nan(std):
        return f"{mean * 100:.0f}%"
    return f"{mean * 100:.0f}% +/- {std * 100:.0f}%"


def _fmt_paper(mean: float | None, std: float | None) -> str:
    """`mean (std)` as in DC3 Table 3: 3 decimals for |x| in [1e-2, 100), else scientific."""
    if _is_nan(mean):
        return "-"
    sci = abs(mean) >= 100 or (mean != 0 and abs(mean) < 1e-2)
    spec_mean = ".3e" if sci else ".3f"
    if _is_nan(std):
        return format(mean, spec_mean)
    spec_std = ".3e" if sci else ".3f"
    return f"{format(mean, spec_mean)} ({format(std, spec_std)})"


def _fmt_gib(mean: float | None, std: float | None) -> str:
    if _is_nan(mean):
        return "-"
    if _is_nan(std):
        return f"{mean / _GIB:.2f} GiB"
    return f"{mean / _GIB:.2f} +/- {std / _GIB:.2f} GiB"


def _aggregate(df: pl.DataFrame) -> pl.DataFrame:
    # Per-run scalars (mean over the eval pool), aggregated as mean +/- std across seeds.
    has_eq = "max_eq_post_mean" in df.columns
    has_ineq = "max_ineq_post_mean" in df.columns
    aggs = [
        pl.len().alias("n_seeds"),
        pl.col("obj_post").mean().alias("obj_mean"),
        pl.col("obj_post").std().alias("obj_std"),
        pl.col("gap_rel").mean().alias("gap_mean"),
        pl.col("gap_rel").std().alias("gap_std"),
        pl.col("feas_post").mean().alias("feas_mean"),
        pl.col("feas_post").std().alias("feas_std"),
        pl.col("viol_max_post").mean().alias("viol_mean"),
        pl.col("viol_max_post").std().alias("viol_std"),
        pl.col("repair_peak_mem_bytes_mean").mean().alias("mem_mean"),
        pl.col("repair_peak_mem_bytes_mean").std().alias("mem_std"),
    ]
    if has_eq:
        aggs += [
            pl.col("max_eq_post_mean").mean().alias("max_eq_mean"),
            pl.col("max_eq_post_mean").std().alias("max_eq_std"),
        ]
    if has_ineq:
        aggs += [
            pl.col("max_ineq_post_mean").mean().alias("max_ineq_mean"),
            pl.col("max_ineq_post_mean").std().alias("max_ineq_std"),
        ]
    return df.group_by("method").agg(*aggs)


def _ordered_methods(present: list[str]) -> list[str]:
    seen = set(present)
    ordered = [m for m in METHODS_ORDER if m in seen]
    extras = sorted(seen - set(ordered))
    return ordered + extras


def _row_lookup(grouped: pl.DataFrame, method: str) -> dict | None:
    cell = grouped.filter(pl.col("method") == method)
    if cell.height == 0:
        return None
    return cell.row(0, named=True)


def render(parquet_path: Path, out_md: Path, out_tex: Path) -> None:
    df = pl.read_parquet(parquet_path)
    if df.height == 0:
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("# Engineering: empty\n\n_No rows in gap parquet._\n")
        out_tex.write_text("% empty gap parquet\n")
        print(f"[render-eng] {out_md} + {out_tex} (empty)")
        return

    bench = df.get_column("bench").unique().to_list()[0]
    today = date.today().isoformat()
    grouped = _aggregate(df)
    methods = _ordered_methods(df.get_column("method").unique().to_list())

    md = [
        f"# Engineering: `{bench}`, {today}",
        "",
        f"Aggregated from `{parquet_path.name}` "
        f"({df.height} rows, {len(methods)} methods).",
        "",
        "| method | obj (raw) | gap to IPOPT | feasibility | post-viol max | max eq (paper) | max ineq (paper) | repair peak mem | n seeds |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for m in methods:
        r = _row_lookup(grouped, m)
        if r is None:
            md.append(f"| `{m}` | - | - | - | - | - | - | - | - |")
            continue
        obj_cell = _fmt_obj(r["obj_mean"], r["obj_std"])
        if m == "ipopt":
            gap_cell = "0 (ref)"
            mem_cell = "-"
        else:
            gap_cell = _fmt_signed(r["gap_mean"], r["gap_std"])
            mem_cell = _fmt_gib(r["mem_mean"], r["mem_std"])
        max_eq_cell = _fmt_paper(r.get("max_eq_mean"), r.get("max_eq_std"))
        max_ineq_cell = _fmt_paper(r.get("max_ineq_mean"), r.get("max_ineq_std"))
        md.append(
            "| `{m}` | {o} | {g} | {f} | {v} | {eq} | {ineq} | {mem} | {n} |".format(
                m=m,
                o=obj_cell,
                g=gap_cell,
                f=_fmt_pct(r["feas_mean"], r["feas_std"]),
                v=_fmt_unsigned(r["viol_mean"], r["viol_std"]),
                eq=max_eq_cell,
                ineq=max_ineq_cell,
                mem=mem_cell,
                n=int(r["n_seeds"]),
            )
        )
    md += ["", "---", "",
           f"_Generated by `scripts/render_engineering.py` from `{parquet_path.name}`._",
           ""]

    tex = [
        f"% Engineering: {bench}, {today}",
        "\\begin{tabular}{lrrrrrrrr}",
        "\\toprule",
        "method & obj (raw) & gap to IPOPT & feasibility & post-viol max & max eq (paper) & max ineq (paper) & repair peak mem & n seeds \\\\",
        "\\midrule",
    ]
    for m in methods:
        r = _row_lookup(grouped, m)
        if r is None:
            tex.append(f"\\texttt{{{m}}} & --- & --- & --- & --- & --- & --- & --- & --- \\\\")
            continue
        obj_cell = _fmt_obj(r["obj_mean"], r["obj_std"])
        if m == "ipopt":
            gap_cell = "0 (ref)"
            mem_cell = "-"
        else:
            gap_cell = _fmt_signed(r["gap_mean"], r["gap_std"])
            mem_cell = _fmt_gib(r["mem_mean"], r["mem_std"])
        feas_cell = _fmt_pct(r["feas_mean"], r["feas_std"])
        viol_cell = _fmt_unsigned(r["viol_mean"], r["viol_std"])
        max_eq_cell = _fmt_paper(r.get("max_eq_mean"), r.get("max_eq_std"))
        max_ineq_cell = _fmt_paper(r.get("max_ineq_mean"), r.get("max_ineq_std"))
        n_cell = str(int(r["n_seeds"]))
        cells = [obj_cell, gap_cell, feas_cell, viol_cell, max_eq_cell, max_ineq_cell, mem_cell, n_cell]
        cells = [c.replace("+/-", r"$\pm$").replace("%", r"\%") for c in cells]
        tex.append(f"\\texttt{{{m}}} & " + " & ".join(cells) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", ""]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(md))
    out_tex.write_text("\n".join(tex))
    print(f"[render-eng] {out_md} + {out_tex} ({len(methods)} methods)")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="render one engineering bench's table")
    p.add_argument(
        "--bench", required=True,
        help="single benchmark id (e.g. e3/acopf_ieee57). 'all', comma-lists, "
             "and family prefixes are rejected, render one table at a time.",
    )
    p.add_argument(
        "--gap-parquet", default=None,
        help="path to <bench>_gap.parquet (default: "
             "results/<today>/engineering/<safe_bench>_gap.parquet)",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="default: results/<today>/engineering/",
    )
    args = p.parse_args(argv)

    bench = args.bench.strip()
    if "," in bench or bench.lower() == "all":
        print(
            f"[render-eng] --bench must be a single id, got {bench!r}; "
            "render one bench at a time.",
            file=sys.stderr,
        )
        return 2

    today = date.today().isoformat()
    safe = _safe_bench(bench)
    out_dir = (
        Path(args.out_dir)
        if args.out_dir
        else Path("results") / today / "engineering"
    )
    parquet = (
        Path(args.gap_parquet)
        if args.gap_parquet
        else out_dir / f"{safe}_gap.parquet"
    )
    if not parquet.exists():
        print(f"[render-eng] gap parquet not found: {parquet}", file=sys.stderr)
        return 1

    out_md = out_dir / f"{safe}.md"
    out_tex = out_dir / f"{safe}.tex"
    render(parquet, out_md, out_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
