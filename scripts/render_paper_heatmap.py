"""Render the stacked heatmap cost table (benches x methods x 4 metrics) from `runs.parquet`.

Usage: python scripts/render_paper_heatmap.py --runs-parquet runs.parquet --out table.tex
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from pal.eval.table1_metrics import (
    CellMetrics,
    aggregate_cells,
    method_feas_mean,
    method_obj_mean,
    run_rows_from_records,
)

METHOD_ORDER: list[tuple[str, str]] = [
    ("alm", r"\textbf{ALM}"),
    ("alm_bolton", r"\makecell{\textbf{ALM+}\\\textbf{Bolt-On}}"),
    ("enforce_orig", r"\textbf{ENFORCE}"),
    ("dc3", r"\textbf{DC3}"),
    ("fsnet", r"\textbf{FSNet}"),
    ("snarenet", r"\textbf{SnareNet}"),
    ("pal_loggap", r"\makecell{\textbf{PAL}\\{}[Ours]}"),
]

_MARK_STRUCT = r"$^{\S}$"       # structurally inapplicable (excluded from Mean).
_MARK_DIVERGED = r"$^{\ddag}$"  # fully diverged; worst-cased into the Mean.

BENCH_ORDER: list[tuple[str, str]] = [
    ("s1_sphere_track", "s1"),
    ("s2_active_set_switch", "s2"),
    ("s3_illcond_tube", "s3"),
    ("s4_qv_coupling", "s4"),
    ("s5_overdetermined", "s5"),
    ("s6_redundant_ineq", "s6"),
]


@dataclass
class Cell:
    """One (method, bench, metric) cell aggregated across seeds; `goodness` feeds `\\cc{}`."""

    mean: float | None
    std: float | None
    goodness: int | None
    text: str
    is_best: bool = False
    n_failures: int = 0
    annotation: str = ""  # superscript marker appended after the cell content.


def _format_cell(
    text: str,
    goodness: int | None,
    is_best: bool,
    n_failures: int = 0,
    annotation: str = "",
) -> str:
    """Render one tabular cell: optional `\\cc{N}` + content (best-bolded) + annotation."""
    if goodness is None:
        return "---" + annotation  # no color for missing / diverged cells
    inner = rf"\best{{{text}}}" if is_best else text
    if n_failures == 1:
        inner = inner + r"$^{\dag}$"
    return rf"\cc{{{goodness}}} {inner}{annotation}"


def _safe_div(a: float | None, b: float | None) -> float | None:
    """Return a/b if both finite and b != 0; else None."""
    if a is None or b is None:
        return None
    if not (math.isfinite(a) and math.isfinite(b)) or b == 0:
        return None
    return a / b


def _per_run_nfe(row: dict[str, Any]) -> tuple[float | None, float | None]:
    """Compute (nfe_fwd, nfe_bwd) as calls per opt step, i.e. passes per element per epoch."""
    fwd_calls = row.get("train_fwd_calls")
    bwd_calls = row.get("train_bwd_calls")
    opt_steps = row.get("train_opt_steps")
    return _safe_div(fwd_calls, opt_steps), _safe_div(bwd_calls, opt_steps)


def _aggregate_seeds(values: list[float]) -> tuple[float | None, float | None]:
    """Mean +/- sample std across seeds. None if no values."""
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return None, None
    n = len(finite)
    mean = sum(finite) / n
    if n < 2:
        return mean, 0.0
    var = sum((v - mean) ** 2 for v in finite) / (n - 1)
    return mean, math.sqrt(var)


def _dedupe(df: pl.DataFrame) -> pl.DataFrame:
    """Keep latest run per (method, bench, seed) by `wall_start`."""
    return (
        df.sort("wall_start", descending=True)
        .unique(subset=["method", "benchmark_id", "seed"], keep="first")
    )


def _gather(
    df: pl.DataFrame,
) -> dict[tuple[str, str], dict[str, tuple[float | None, float | None]]]:
    """Group runs by (method, bench), aggregate over seeds.

    Returns {(method, bench): {"feas": (m,s), "obj": (m,s),
                                "nfe_fwd": (m,s), "nfe_bwd": (m,s)}}.
    """
    rows = df.to_dicts()
    by_pair: dict[tuple[str, str], dict[str, list[float]]] = {}
    for row in rows:
        key = (row["method"], row["benchmark_id"])
        slot = by_pair.setdefault(key, {"feas": [], "obj": [], "nfe_fwd": [], "nfe_bwd": []})
        if row.get("feasibility_post") is not None:
            slot["feas"].append(float(row["feasibility_post"]))
        if row.get("obj_mean_post") is not None:
            slot["obj"].append(float(row["obj_mean_post"]))
        nfe_fwd, nfe_bwd = _per_run_nfe(row)
        if nfe_fwd is not None:
            slot["nfe_fwd"].append(nfe_fwd)
        if nfe_bwd is not None:
            slot["nfe_bwd"].append(nfe_bwd)

    return {
        key: {metric: _aggregate_seeds(values) for metric, values in metrics.items()}
        for key, metrics in by_pair.items()
    }


def _count_by_pair(df: pl.DataFrame) -> dict[tuple[str, str], int]:
    """Count unique seeds per (method, bench)."""
    if df.is_empty():
        return {}
    counts = df.group_by(["method", "benchmark_id"]).agg(
        pl.col("seed").n_unique().alias("n")
    )
    return {
        (row["method"], row["benchmark_id"]): int(row["n"])
        for row in counts.to_dicts()
    }


def _build_cells(
    agg: dict[tuple[str, str], dict[str, tuple[float | None, float | None]]],
    methods: list[str],
    benches: list[str],
    attempts: dict[tuple[str, str], int] | None = None,
    successes: dict[tuple[str, str], int] | None = None,
) -> dict[str, dict[tuple[str, str], Cell]]:
    """For each metric, build the (method, bench) -> Cell map."""
    feas: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    obj: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    nfe_fwd: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    nfe_bwd: dict[tuple[str, str], tuple[float | None, float | None]] = {}
    for method in methods:
        for bench in benches:
            metrics = agg.get((method, bench), {})
            feas[(method, bench)] = metrics.get("feas", (None, None))
            obj[(method, bench)] = metrics.get("obj", (None, None))
            nfe_fwd[(method, bench)] = metrics.get("nfe_fwd", (None, None))
            nfe_bwd[(method, bench)] = metrics.get("nfe_bwd", (None, None))

    obj_means = [m for (m, _s) in obj.values() if m is not None]
    # Cap the red end of the obj scale so one blown-up value doesn't squash the rest.
    obj_max_block = min(max(obj_means), OBJ_MAX_CAP) if obj_means else OBJ_MAX_CAP

    # Fixed log-scale NFE anchors (1 = green, 50 = red) keep re-renders comparable.
    nfe_max_block = NFE_MAX
    nfe_floor = NFE_FLOOR

    attempts = attempts or {}
    successes = successes or {}

    feas_cells = _materialize(
        feas, methods, benches, fmt=_fmt_feas, goodness_fn=_goodness_feas,
        attempts=attempts, successes=successes,
    )
    obj_cells = _materialize(
        obj, methods, benches, fmt=_fmt_obj,
        goodness_fn=lambda m: _goodness_obj(m, obj_max_block, OBJ_FLOOR),
        attempts=attempts, successes=successes,
    )
    nfe_fwd_cells = _materialize(
        nfe_fwd, methods, benches, fmt=_fmt_nfe,
        goodness_fn=lambda m: _goodness_nfe(m, nfe_max_block, nfe_floor),
        attempts=attempts, successes=successes,
    )
    nfe_bwd_cells = _materialize(
        nfe_bwd, methods, benches, fmt=_fmt_nfe,
        goodness_fn=lambda m: _goodness_nfe(m, nfe_max_block, nfe_floor),
        attempts=attempts, successes=successes,
    )

    # Only the per-method Mean row gets bold-best; the grid stays unbolded.

    cells_by_metric = {
        "feas": feas_cells,
        "obj": obj_cells,
        "nfe_fwd": nfe_fwd_cells,
        "nfe_bwd": nfe_bwd_cells,
    }

    # Per-method mean across benches; ties in formatted text all bold.
    means_by_metric = {
        "feas": _build_mean_row(
            feas_cells, methods, benches,
            fmt=_fmt_mean_feas, goodness_fn=_goodness_feas, lower_is_better=False,
        ),
        "obj": _build_mean_row(
            obj_cells, methods, benches,
            fmt=_fmt_mean_obj,
            goodness_fn=lambda m: _goodness_obj(m, obj_max_block, OBJ_FLOOR),
            lower_is_better=True,
        ),
        "nfe_fwd": _build_mean_row(
            nfe_fwd_cells, methods, benches,
            fmt=_fmt_mean_nfe,
            goodness_fn=lambda m: _goodness_nfe(m, nfe_max_block, nfe_floor),
            lower_is_better=True,
        ),
        "nfe_bwd": _build_mean_row(
            nfe_bwd_cells, methods, benches,
            fmt=_fmt_mean_nfe,
            goodness_fn=lambda m: _goodness_nfe(m, nfe_max_block, nfe_floor),
            lower_is_better=True,
        ),
    }

    return cells_by_metric, means_by_metric


def _build_mean_row(
    cells: dict[tuple[str, str], Cell],
    methods: list[str],
    benches: list[str],
    fmt,
    goodness_fn,
    lower_is_better: bool,
) -> dict[str, Cell]:
    """Per-method mean across benches; `---` if any (method, bench) cell is missing."""
    out: dict[str, Cell] = {}
    for method in methods:
        any_missing = any(
            cells[(method, bench)].mean is None for bench in benches
        )
        if any_missing:
            out[method] = Cell(None, None, None, "---")
            continue
        vals = [cells[(method, bench)].mean for bench in benches]
        m = sum(vals) / len(vals)
        out[method] = Cell(m, None, goodness_fn(m), fmt(m))

    valid = [(method, c) for method, c in out.items() if c.mean is not None]
    if valid:
        if lower_is_better:
            best_mean = min(c.mean for (_m, c) in valid)
        else:
            best_mean = max(c.mean for (_m, c) in valid)
        best_text = next(c.text for (_m, c) in valid if c.mean == best_mean)
        for _m, c in valid:
            if c.text == best_text:
                c.is_best = True
    return out


def _materialize(
    table: dict[tuple[str, str], tuple[float | None, float | None]],
    methods: list[str],
    benches: list[str],
    fmt,
    goodness_fn,
    attempts: dict[tuple[str, str], int] | None = None,
    successes: dict[tuple[str, str], int] | None = None,
) -> dict[tuple[str, str], Cell]:
    attempts = attempts or {}
    successes = successes or {}
    out: dict[tuple[str, str], Cell] = {}
    for method in methods:
        for bench in benches:
            mean, std = table[(method, bench)]
            n_attempted = attempts.get((method, bench), 0)
            n_ok = successes.get((method, bench), 0)
            n_failures = max(0, n_attempted - n_ok)
            # >=2 failures: emit an em dash regardless of whether we have a mean.
            if mean is None or n_failures >= 2:
                out[(method, bench)] = Cell(
                    None, None, None, "---", n_failures=n_failures
                )
                continue
            text = fmt(mean, std if std is not None else 0.0)
            goodness = goodness_fn(mean)
            out[(method, bench)] = Cell(
                mean, std, goodness, text, n_failures=n_failures
            )
    return out


def _mark_best(
    cells: dict[tuple[str, str], Cell],
    methods: list[str],
    benches: list[str],
    lower_is_better: bool,
) -> None:
    """Set Cell.is_best on every cell whose formatted text equals the row's best."""
    for bench in benches:
        row = [(method, cells[(method, bench)]) for method in methods]
        candidates = [(m, c) for (m, c) in row if c.mean is not None]
        if not candidates:
            continue
        if lower_is_better:
            best_mean = min(c.mean for (_m, c) in candidates)
        else:
            best_mean = max(c.mean for (_m, c) in candidates)
        best_text = next(c.text for (_m, c) in candidates if c.mean == best_mean)
        for _m, c in candidates:
            if c.text == best_text:
                c.is_best = True


def _fmt_feas(m: float, s: float) -> str:
    """`int+/-int` (Feas % is m in [0,1] -> percent integer)."""
    return rf"${int(round(m * 100))}\pm{int(round(s * 100))}$"


def _round_int_when_large(m: float, s: float) -> str | None:
    """If mean >= 100 or std >= 10, round both to integer ($m\\pm s$). Else None."""
    if m >= 100 or s >= 10:
        return rf"${int(round(m))}\pm{int(round(s))}$"
    return None


def _fmt_obj(m: float, s: float) -> str:
    rounded = _round_int_when_large(m, s)
    if rounded is not None:
        return rounded
    return rf"${m:.2f}\pm{s:.2f}$"


def _fmt_nfe(m: float, s: float) -> str:
    rounded = _round_int_when_large(m, s)
    if rounded is not None:
        return rounded
    return rf"${m:.1f}\pm{s:.1f}$"


# Mean-row formatters: single value, no std.
def _fmt_mean_feas(m: float) -> str:
    return rf"${int(round(m * 100))}$"


def _fmt_mean_obj(m: float) -> str:
    if m >= 100:
        return rf"${int(round(m))}$"
    return rf"${m:.2f}$"


def _fmt_mean_nfe(m: float) -> str:
    if m >= 100:
        return rf"${int(round(m))}$"
    return rf"${m:.1f}$"


FEAS_POWER = 5.0       # 100%->100, 95%->77, 90%->59, 85%->44, 80%->33. Higher = more red.
OBJ_FLOOR = 1e-3       # m <= floor -> full green (matches feasibility tolerance scale).
OBJ_MAX_CAP = 2.0      # cap the red anchor so a single huge obj doesn't squash the rest.
OBJ_FAIL_THRESHOLD = 100.0  # runs with obj_mean_post above this are demoted to "failed"
NFE_FLOOR = 1.0        # PAL theoretical min: 1 fwd + 1 bwd per batch element per epoch.
NFE_MAX = 50.0         # red anchor; >=50 NFE clips to full red.


def _goodness_feas(m: float) -> int:
    """Power curve: 1.0->100, 0.9->59, 0.8->33, 0.5->3. Pushes <100% toward red."""
    g = (max(0.0, min(1.0, m)) ** FEAS_POWER) * 100
    return max(0, min(100, int(round(g))))


def _goodness_obj(m: float, obj_max: float, obj_floor: float) -> int:
    """Log scale on the rounded display value, so a cell printing `0.00` is fully green."""
    m_disp = round(m, 2)  # _fmt_obj uses {m:.2f}; match its rounding.
    if obj_max <= obj_floor:
        return 100
    if m_disp <= obj_floor:
        return 100
    if m_disp >= obj_max:
        return 0
    g = (math.log(obj_max) - math.log(m_disp)) / (math.log(obj_max) - math.log(obj_floor)) * 100
    return max(0, min(100, int(round(g))))


def _goodness_nfe(m: float, nfe_max: float, nfe_floor: float) -> int:
    """Log scale: m <= floor -> 100 (green), m >= nfe_max -> 0 (red)."""
    if nfe_max <= nfe_floor:
        return 100
    if m <= nfe_floor:
        return 100
    if m >= nfe_max:
        return 0
    g = (math.log(nfe_max) - math.log(m)) / (math.log(nfe_max) - math.log(nfe_floor)) * 100
    return max(0, min(100, int(round(g))))


PREAMBLE_MACROS = r"""% Macros for the PAL stacked-heatmap table. Mirrors mock at
% pal_stacked_table_mock.tex, keep `\cc` / `\best` / `goodc` / `badc`
% / colspec `M` in sync if the mock changes.
% midc = pastel yellow chosen to share R with badc and G with goodc, so the
% green->yellow->red ramp interpolates naturally without muddy mid-tones.
\definecolor{goodc}{HTML}{A8E6A6}
\definecolor{midc}{HTML}{F5E6A6}
\definecolor{badc}{HTML}{F5B9B5}
\newcommand{\cc}[1]{%
  \ifnum#1<50
    \cellcolor{badc!\the\numexpr(50-#1)*2\relax!midc}%
  \else
    \cellcolor{goodc!\the\numexpr(#1-50)*2\relax!midc}%
  \fi
}
% \boldmath inside \textbf is required so the math content (`$100\pm0$`)
% actually renders bold, plain \textbf{$100$} leaves the math weight unchanged.
\newcommand{\best}[1]{\textbf{\boldmath #1}}
\newcolumntype{M}{>{\centering\arraybackslash}p{1.55cm}}
"""


def _emit_table(
    cells_by_metric: dict[str, dict[tuple[str, str], Cell]],
    means_by_metric: dict[str, dict[str, Cell]],
    methods: list[str],
    method_labels: list[str],
    benches: list[str],
    bench_labels: list[str],
    caption: str,
    label: str,
) -> str:
    """Build the `\\begin{table}...\\end{table}` block."""
    lines: list[str] = []
    n_methods = len(methods)
    method_cols = f" *{{{n_methods}}}{{M}}"

    # Caption legend: diverged runs are scored worst-case and folded into the Mean.
    caption_full = (
        rf"\TODO{{{caption}}} "
        r"A superscript ``$k/N$'' marks a cell with $k$ of $N$ seeds diverged; "
        r"such seeds are scored worst-case (feasibility 0, objective at the "
        r"per-benchmark ceiling $C_b$) and folded into the seed mean. "
        r"``---''$^{\ddag}$ marks a fully diverged cell (all seeds); it is still "
        r"scored worst-case in the \textit{Mean}. ``---''$^{\S}$ marks a "
        r"structurally inapplicable cell (dc3\,$\times$\,s5, $n_{eq}>\dim$), "
        r"excluded from the \textit{Mean}. A \textit{Mean} superscript "
        r"$^{\ddag}$/$^{\S}$ flags that the row imputes diverged seeds / drops a "
        r"structural cell. "
        r"Cells with mean $\geq 100$ or std $\geq 10$ are rounded to integers."
    )

    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\footnotesize")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\renewcommand{\arraystretch}{1.15}")
    lines.append(rf"\caption{{{caption_full}}}")
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\begin{tabular}{@{}ll" + method_cols + r"@{}}")
    lines.append(r"\toprule")
    header_methods = " & ".join(method_labels)
    lines.append(r" & \textbf{Bench.}")
    lines.append(rf"  & {header_methods} \\")
    lines.append(r"\midrule")

    metric_rows = [
        ("feas", r"Feas.\ \%"),
        ("obj", r"Obj."),
        ("nfe_fwd", r"NFE\textsubscript{fwd}"),
        ("nfe_bwd", r"NFE\textsubscript{bwd}"),
    ]
    n_label_cols = 2  # rotated-metric-label col + bench-label col
    cmidrule_span = f"{n_label_cols}-{n_label_cols + len(methods)}"
    for slab_idx, (metric_key, metric_label) in enumerate(metric_rows):
        cells = cells_by_metric[metric_key]
        means = means_by_metric[metric_key]
        # Multirow spans the benches plus the Mean row.
        n_rows = len(benches) + 1
        for i, (bench, bench_label) in enumerate(zip(benches, bench_labels, strict=False)):
            row_cells = []
            for method in methods:
                cell = cells[(method, bench)]
                row_cells.append(
                    _format_cell(
                        cell.text, cell.goodness, cell.is_best,
                        cell.n_failures, cell.annotation,
                    )
                )
            cell_str = " & ".join(row_cells)
            if i == 0:
                row_prefix = (
                    rf"\multirow{{{n_rows}}}{{*}}"
                    rf"{{\rotatebox[origin=c]{{90}}{{{metric_label}}}}}"
                )
                lines.append(rf"{row_prefix}")
                lines.append(rf"  & {bench_label} & {cell_str} \\")
            else:
                lines.append(rf"  & {bench_label} & {cell_str} \\")
            # Thin cmidrule separates bench rows from the Mean row (skips col 1).
        lines.append(rf"\cmidrule(lr){{{cmidrule_span}}}")
        mean_row_cells = [
            _format_cell(c.text, c.goodness, c.is_best, c.n_failures, c.annotation)
            for c in (means[m] for m in methods)
        ]
        lines.append(rf"  & \textit{{Mean}} & {' & '.join(mean_row_cells)} \\")
        if slab_idx < len(metric_rows) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines) + "\n"


def _emit_standalone(table_block: str, preamble_macros: str) -> str:
    """Wrap the table block in a self-compiling LaTeX document with a no-op `\\TODO`."""
    return (
        r"""\documentclass[11pt]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs}
\usepackage[table]{xcolor}
\usepackage{multirow}
\usepackage{array}
\usepackage{makecell}
\usepackage{graphicx}
\providecommand{\TODO}[1]{\textbf{TODO:} #1}

"""
        + preamble_macros
        + "\n\\begin{document}\n\n"
        + table_block
        + "\n\\end{document}\n"
    )


def _div_annotation(mc: CellMetrics) -> str:
    """Superscript flag for a data-carrying cell: "k/N" if k seeds diverged."""
    if mc.n_diverged > 0:
        return rf"$^{{{mc.n_diverged}/{mc.n_attempted}}}$"
    return ""


def _dash_cell(mc: CellMetrics) -> Cell | None:
    """Return an em dash Cell (with the right marker) if the cell carries no data, else None."""
    if mc.is_structural:
        return Cell(None, None, None, "---", annotation=_MARK_STRUCT)
    if mc.n_attempted == 0 or mc.is_fully_diverged:
        ann = _MARK_DIVERGED if mc.is_fully_diverged else ""
        return Cell(None, None, None, "---", annotation=ann)
    return None


def _module_metric_cells(
    mcells: dict[tuple[str, str], CellMetrics],
    methods: list[str],
    benches: list[str],
    *,
    pick,
    fmt,
    goodness_fn,
) -> dict[tuple[str, str], Cell]:
    """Build feas/obj grid Cells from module metrics. ``pick`` -> (mean, std)."""
    out: dict[tuple[str, str], Cell] = {}
    for method in methods:
        for bench in benches:
            mc = mcells[(method, bench)]
            dash = _dash_cell(mc)
            if dash is not None:
                out[(method, bench)] = dash
                continue
            mean, std = pick(mc)
            if mean is None:
                out[(method, bench)] = Cell(None, None, None, "---")
                continue
            text = fmt(mean, std if std is not None else 0.0)
            out[(method, bench)] = Cell(
                mean, std, goodness_fn(mean), text, annotation=_div_annotation(mc)
            )
    return out


def _nfe_metric_cells(
    mcells: dict[tuple[str, str], CellMetrics],
    nfe_agg: dict[tuple[str, str], dict[str, tuple[float | None, float | None]]],
    methods: list[str],
    benches: list[str],
    *,
    key: str,
    goodness_fn,
) -> dict[tuple[str, str], Cell]:
    """Build NFE grid Cells; diverged/structural cells render an em dash (no imputation)."""
    out: dict[tuple[str, str], Cell] = {}
    for method in methods:
        for bench in benches:
            mc = mcells[(method, bench)]
            dash = _dash_cell(mc)
            if dash is not None:
                out[(method, bench)] = dash
                continue
            mean, std = nfe_agg.get((method, bench), {}).get(key, (None, None))
            if mean is None:
                out[(method, bench)] = Cell(None, None, None, "---")
                continue
            text = _fmt_nfe(mean, std if std is not None else 0.0)
            out[(method, bench)] = Cell(
                mean, std, goodness_fn(mean), text, annotation=_div_annotation(mc)
            )
    return out


def _method_mean_marker(
    mcells: dict[tuple[str, str], CellMetrics], method: str, benches: list[str]
) -> str:
    """Mean-row superscript: `\\ddag` if any seed was worst-cased, `\\S` if a cell was dropped."""
    has_struct = any(mcells[(method, b)].is_structural for b in benches)
    has_div = any(
        mcells[(method, b)].n_diverged > 0 and not mcells[(method, b)].is_structural
        for b in benches
    )
    marks = (r"\ddag" if has_div else "") + (r"\S" if has_struct else "")
    return rf"$^{{{marks}}}$" if marks else ""


def _mark_mean_best(row: dict[str, Cell], *, lower_is_better: bool) -> None:
    """Bold the best Mean cell across methods (ties bold all, via string match)."""
    valid = [(m, c) for m, c in row.items() if c.mean is not None]
    if not valid:
        return
    best_mean = (min if lower_is_better else max)(c.mean for (_m, c) in valid)
    best_text = next(c.text for (_m, c) in valid if c.mean == best_mean)
    for _m, c in valid:
        if c.text == best_text:
            c.is_best = True


def _feasobj_mean_row(
    mcells: dict[tuple[str, str], CellMetrics],
    methods: list[str],
    benches: list[str],
    *,
    value_fn,
    fmt,
    goodness_fn,
    lower_is_better: bool,
) -> dict[str, Cell]:
    """Per-method Mean row for feas/obj: worst-case imputed, structural dropped."""
    out: dict[str, Cell] = {}
    for method in methods:
        val = value_fn(mcells, method, benches)
        out[method] = Cell(
            val, None, goodness_fn(val), fmt(val),
            annotation=_method_mean_marker(mcells, method, benches),
        )
    _mark_mean_best(out, lower_is_better=lower_is_better)
    return out


def _nfe_mean_row(
    mcells: dict[tuple[str, str], CellMetrics],
    nfe_cells: dict[tuple[str, str], Cell],
    methods: list[str],
    benches: list[str],
    *,
    goodness_fn,
) -> dict[str, Cell]:
    """Per-method NFE Mean over the benches that produced NFE (no imputation)."""
    out: dict[str, Cell] = {}
    for method in methods:
        vals = [
            nfe_cells[(method, b)].mean
            for b in benches
            if nfe_cells[(method, b)].mean is not None
        ]
        if not vals:
            out[method] = Cell(None, None, None, "---")
            continue
        val = sum(vals) / len(vals)
        out[method] = Cell(
            val, None, goodness_fn(val), _fmt_mean_nfe(val),
            annotation=_method_mean_marker(mcells, method, benches),
        )
    _mark_mean_best(out, lower_is_better=True)
    return out


def render(
    runs_parquet: Path,
    out_path: Path,
    standalone_path: Path | None,
    preamble_path: Path | None,
    methods: list[tuple[str, str]] = METHOD_ORDER,
    benches: list[tuple[str, str]] = BENCH_ORDER,
    caption: str = "methods sweep caption.",
    label: str = "tab:methods-sweep-synthetic",
) -> None:
    df = pl.read_parquet(runs_parquet)

    method_keys = [m for (m, _) in methods]
    bench_keys = [b for (b, _) in benches]
    method_labels = [lbl for (_, lbl) in methods]
    bench_labels = [lbl for (_, lbl) in benches]

    df = df.filter(
        pl.col("method").is_in(method_keys)
        & pl.col("benchmark_id").is_in(bench_keys)
    )
    # Feasibility/objective and divergence status come from pal.eval.table1_metrics.
    mcells = aggregate_cells(
        run_rows_from_records(df.to_dicts()), method_keys, bench_keys
    )
    # NFE is cost telemetry: ok runs only, deduped, same obj-blowup demotion.
    df_ok = _dedupe(
        df.filter(
            (pl.col("status") == "ok")
            & (
                pl.col("obj_mean_post").is_null()
                | (pl.col("obj_mean_post") <= OBJ_FAIL_THRESHOLD)
            )
        )
    )
    nfe_agg = _gather(df_ok)

    # Obj color anchor: max shown obj across the block, capped.
    obj_means = [
        mcells[(m, b)].obj_disp_mean
        for m in method_keys
        for b in bench_keys
        if _dash_cell(mcells[(m, b)]) is None
        and mcells[(m, b)].obj_disp_mean is not None
    ]
    obj_max_block = min(max(obj_means), OBJ_MAX_CAP) if obj_means else OBJ_MAX_CAP

    feas_cells = _module_metric_cells(
        mcells, method_keys, bench_keys,
        pick=lambda mc: (mc.feas_disp_mean, mc.feas_disp_std),
        fmt=_fmt_feas, goodness_fn=_goodness_feas,
    )
    obj_cells = _module_metric_cells(
        mcells, method_keys, bench_keys,
        pick=lambda mc: (mc.obj_disp_mean, mc.obj_disp_std),
        fmt=_fmt_obj,
        goodness_fn=lambda m: _goodness_obj(m, obj_max_block, OBJ_FLOOR),
    )
    nfe_goodness = lambda m: _goodness_nfe(m, NFE_MAX, NFE_FLOOR)  # noqa: E731
    nfe_fwd_cells = _nfe_metric_cells(
        mcells, nfe_agg, method_keys, bench_keys, key="nfe_fwd", goodness_fn=nfe_goodness
    )
    nfe_bwd_cells = _nfe_metric_cells(
        mcells, nfe_agg, method_keys, bench_keys, key="nfe_bwd", goodness_fn=nfe_goodness
    )

    cells_by_metric = {
        "feas": feas_cells,
        "obj": obj_cells,
        "nfe_fwd": nfe_fwd_cells,
        "nfe_bwd": nfe_bwd_cells,
    }
    means_by_metric = {
        "feas": _feasobj_mean_row(
            mcells, method_keys, bench_keys,
            value_fn=method_feas_mean, fmt=_fmt_mean_feas,
            goodness_fn=_goodness_feas, lower_is_better=False,
        ),
        "obj": _feasobj_mean_row(
            mcells, method_keys, bench_keys,
            value_fn=method_obj_mean, fmt=_fmt_mean_obj,
            goodness_fn=lambda m: _goodness_obj(m, obj_max_block, OBJ_FLOOR),
            lower_is_better=True,
        ),
        "nfe_fwd": _nfe_mean_row(
            mcells, nfe_fwd_cells, method_keys, bench_keys, goodness_fn=nfe_goodness
        ),
        "nfe_bwd": _nfe_mean_row(
            mcells, nfe_bwd_cells, method_keys, bench_keys, goodness_fn=nfe_goodness
        ),
    }

    table_block = _emit_table(
        cells_by_metric,
        means_by_metric,
        method_keys,
        method_labels,
        bench_keys,
        bench_labels,
        caption=caption,
        label=label,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_block)

    if preamble_path is not None:
        preamble_path.parent.mkdir(parents=True, exist_ok=True)
        preamble_path.write_text(PREAMBLE_MACROS)

    if standalone_path is not None:
        standalone_path.parent.mkdir(parents=True, exist_ok=True)
        standalone_path.write_text(_emit_standalone(table_block, PREAMBLE_MACROS))

    n_filled = sum(1 for c in cells_by_metric["feas"].values() if c.mean is not None)
    n_total = len(method_keys) * len(bench_keys)
    print(
        f"[render_paper_heatmap] wrote {out_path} ({n_filled}/{n_total} cells "
        f"filled across {len(method_keys)} methods x {len(bench_keys)} benches)"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="render the paper's stacked heatmap cost table"
    )
    p.add_argument("--runs-parquet", default="runs.parquet")
    p.add_argument("--out", default="results/methods_sweep_synthetic.tex")
    p.add_argument(
        "--standalone",
        default=None,
        help="optional path for a self-compiling .tex (preamble + macros + table)",
    )
    p.add_argument(
        "--preamble",
        default=None,
        help="optional path to write the preamble macros (\\cc / \\best) for "
        "\\input{}-ing into the paper main",
    )
    p.add_argument("--caption", default="methods sweep caption.")
    p.add_argument("--label", default="tab:methods-sweep-synthetic")
    args = p.parse_args(argv)

    runs_parquet = Path(args.runs_parquet)
    if not runs_parquet.exists():
        print(f"[render_paper_heatmap] runs parquet not found: {runs_parquet}", file=sys.stderr)
        return 1

    render(
        runs_parquet,
        out_path=Path(args.out),
        standalone_path=Path(args.standalone) if args.standalone else None,
        preamble_path=Path(args.preamble) if args.preamble else None,
        caption=args.caption,
        label=args.label,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
