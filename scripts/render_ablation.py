"""Render an ablation table from runs.parquet + an ablation YAML (mean +/- std over seeds).

Usage: python scripts/render_ablation.py --ablation <yaml> --format latex --out <tex>
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import polars as pl
import yaml


def _load_ablation_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        ab = yaml.safe_load(f)
    required = ("ablation_name", "method", "benchmarks", "seeds", "anchor",
                "base_hparams", "scenarios")
    missing = [k for k in required if k not in ab]
    if missing:
        raise SystemExit(f"{path}: missing required keys: {missing}")
    return ab


def _effective_hparams(base: dict, overrides: dict | None) -> dict:
    out = dict(base)
    if overrides:
        out.update(overrides)
    return out


def _load_ablation_runs(
    runs_parquet: Path,
    ab: dict[str, Any],
) -> pl.DataFrame:
    """Filter and dedupe runs for this ablation, keeping both ok and failed runs."""
    df = pl.read_parquet(runs_parquet)
    df = df.filter(
        pl.col("status").is_in(["ok", "failed"])
        & (pl.col("method") == ab["method"])
        & pl.col("benchmark_id").is_in(ab["benchmarks"])
        & pl.col("seed").is_in([int(s) for s in ab["seeds"]])
        & (pl.col("ablation_name") == ab["ablation_name"])
        & pl.col("scenario_label").is_in([s["label"] for s in ab["scenarios"]])
    )
    df = (
        df.sort("wall_start", descending=True)
        .unique(subset=["scenario_label", "benchmark_id", "seed"], keep="first")
    )
    return df


def _aggregate_per_scenario(
    df: pl.DataFrame, ab: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Return {scenario_label: {metric: mean/std, n, n_failed}}.

    Per seed, aggregate across benchmarks, then take mean +/- std over seeds. Seeds with a
    failed bench are dropped from the aggregate but counted in `n_failed`.
    """
    n_benchmarks = len(ab["benchmarks"])
    seeds = [int(s) for s in ab["seeds"]]
    out: dict[str, dict[str, float]] = {}
    for scn in ab["scenarios"]:
        label = scn["label"]
        sub = df.filter(pl.col("scenario_label") == label)
        ok_sub = sub.filter(pl.col("status") == "ok")
        n_failed = sub.filter(pl.col("status") == "failed").height
        per_seed_obj_mean: list[float] = []
        per_seed_obj_max: list[float] = []
        per_seed_feas_mean: list[float] = []
        per_seed_feas_min: list[float] = []
        per_seed_violraw_mean: list[float] = []
        seed_rows_ok = 0
        for seed in seeds:
            seed_sub = ok_sub.filter(pl.col("seed") == seed)
            if seed_sub.height != n_benchmarks:
                # Partial seed: skipped here, the `n` column shows coverage.
                continue
            obj = seed_sub.get_column("obj_mean_post").to_list()
            feas = seed_sub.get_column("feasibility_post").to_list()
            violraw = seed_sub.get_column("viol_max_raw").to_list()
            per_seed_obj_mean.append(sum(obj) / len(obj))
            per_seed_obj_max.append(max(obj))
            per_seed_feas_mean.append(sum(feas) / len(feas))
            per_seed_feas_min.append(min(feas))
            per_seed_violraw_mean.append(sum(violraw) / len(violraw))
            seed_rows_ok += 1

        def _mean_std(vs: list[float]) -> tuple[float | None, float | None]:
            if not vs:
                return None, None
            m = sum(vs) / len(vs)
            if len(vs) == 1:
                return m, None
            var = sum((v - m) ** 2 for v in vs) / (len(vs) - 1)
            return m, var ** 0.5

        m_obj_mean, s_obj_mean = _mean_std(per_seed_obj_mean)
        m_obj_max, s_obj_max = _mean_std(per_seed_obj_max)
        m_feas_mean, s_feas_mean = _mean_std(per_seed_feas_mean)
        m_feas_min, s_feas_min = _mean_std(per_seed_feas_min)
        m_violraw, s_violraw = _mean_std(per_seed_violraw_mean)
        out[label] = {
            "obj_mean_m": m_obj_mean, "obj_mean_s": s_obj_mean,
            "obj_max_m":  m_obj_max,  "obj_max_s":  s_obj_max,
            "feas_mean_m": m_feas_mean, "feas_mean_s": s_feas_mean,
            "feas_min_m":  m_feas_min,  "feas_min_s":  s_feas_min,
            "violraw_m": m_violraw, "violraw_s": s_violraw,
            "n": seed_rows_ok,
            "n_failed": n_failed,
        }
    return out


def _varying_hparam_keys(ab: dict[str, Any]) -> list[str]:
    """Union of override keys, ordered by first appearance in scenarios."""
    seen: list[str] = []
    for scn in ab["scenarios"]:
        for k in (scn.get("overrides") or {}).keys():
            if k not in seen:
                seen.append(k)
    return seen


def _fmt_value(
    v: Any,
    fmt: str | None = None,
    value_map: dict | None = None,
) -> str:
    """Format a hparam cell value.

    `value_map` maps raw values to display strings and wins over `fmt`. `fmt="latex_sci"`
    gives "$a \\times 10^{n}$"; None/"auto" gives %g, or %.0e for very small/large values.
    """
    if value_map is not None and v in value_map:
        return str(value_map[v])
    if isinstance(v, bool):
        return "True" if v else "False"
    if isinstance(v, int) and not isinstance(v, bool):
        return str(v)
    if isinstance(v, float):
        if fmt == "latex_sci":
            if v == 0:
                return "0"
            s = f"{v:.1e}"              # "1.0e-03"
            m_str, exp_str = s.split("e")
            mantissa = float(m_str)
            exp = int(exp_str)
            if abs(mantissa - 1.0) < 1e-9:
                return f"$10^{{{exp}}}$"
            m_fmt = f"{mantissa:g}"
            return f"${m_fmt} \\times 10^{{{exp}}}$"
        if v == 0:
            return "0"
        av = abs(v)
        if av < 1e-2 or av >= 1e4:
            return f"{v:.0e}"
        return f"{v:g}"
    return str(v)


def _fmt_cell(m: float | None, s: float | None, fmt: str) -> str:
    if m is None:
        return "-"
    if s is None:
        return fmt.format(m)
    return f"{fmt.format(m)} $\\pm$ {fmt.format(s)}"


def _fmt_feas_cell(m: float | None, s: float | None) -> str:
    """Feasibility cell: integer percent."""
    if m is None:
        return "-"
    m_pct = m * 100
    if s is None:
        return f"{m_pct:.0f}\\%"
    s_pct = s * 100
    return f"{m_pct:.0f} $\\pm$ {s_pct:.0f}\\%"


def _render_latex(
    ab: dict[str, Any],
    varying: list[str],
    base_hp: dict,
    agg: dict[str, dict[str, float]],
    single_bench: bool,
) -> str:
    """booktabs tabular without caption/label; bold marks the lowest obj/violation cells."""
    display = ab.get("display") or {}
    display_labels = display.get("display_labels") or {}
    formats = display.get("format") or {}
    value_maps = display.get("value_map") or {}

    if single_bench:
        metric_cols = ["objective", "feasibility", "$\\|c(\\hat y)\\|_\\infty$"]
        bold_idxs = {0, 2}
    else:
        metric_cols = [
            "obj (mean)", "obj (max)", "feas (min)",
            "$\\|c(\\hat y)\\|_\\infty$",
        ]
        bold_idxs = {0, 1, 3}

    n_hparam_cols = len(varying)
    n_metric_cols = len(metric_cols)
    col_spec = "c" * n_hparam_cols + r"@{\quad}" + "c" * n_metric_cols

    lines: list[str] = []
    # Wrap in a group so \footnotesize / \tabcolsep changes don't leak.
    lines.append("{\\footnotesize")
    lines.append("\\setlength{\\tabcolsep}{3pt}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    # Header row: display_labels win; fall back to raw key with escaped underscores.
    header_cells = (
        [display_labels.get(k, k.replace("_", "\\_")) for k in varying]
        + metric_cols
    )
    lines.append(" & ".join(header_cells) + " \\\\")
    lines.append("\\midrule")

    anchor_label = ab["anchor"]
    scenario_order = [anchor_label] + [
        s["label"] for s in ab["scenarios"] if s["label"] != anchor_label
    ]
    scn_by_label = {s["label"]: s for s in ab["scenarios"]}

    # Pass 1: build per-scenario cells without bolding.
    rows_data: list[dict[str, Any]] = []
    for label in scenario_order:
        scn = scn_by_label[label]
        a = agg.get(label, {})
        is_anchor = (label == anchor_label)
        overrides = scn.get("overrides") or {}
        n_failed = a.get("n_failed", 0)

        hparam_cells: list[str] = []
        for key in varying:
            fmt = formats.get(key)
            vmap = value_maps.get(key)
            if is_anchor:
                hparam_cells.append(_fmt_value(base_hp.get(key, ""), fmt, vmap))
            else:
                hparam_cells.append(
                    _fmt_value(overrides[key], fmt, vmap) if key in overrides else ""
                )

        if n_failed >= 2:
            metric_cells = ["-"] * n_metric_cols
            metric_means: list[float | None] = [None] * n_metric_cols
        else:
            if single_bench:
                metric_cells = [
                    _fmt_cell(a.get("obj_mean_m"), a.get("obj_mean_s"), "{:.2f}"),
                    _fmt_feas_cell(a.get("feas_mean_m"), a.get("feas_mean_s")),
                    _fmt_cell(a.get("violraw_m"), a.get("violraw_s"), "{:.2f}"),
                ]
                metric_means = [
                    a.get("obj_mean_m"),
                    a.get("feas_mean_m"),
                    a.get("violraw_m"),
                ]
            else:
                metric_cells = [
                    _fmt_cell(a.get("obj_mean_m"), a.get("obj_mean_s"), "{:.2f}"),
                    _fmt_cell(a.get("obj_max_m"),  a.get("obj_max_s"),  "{:.2f}"),
                    _fmt_feas_cell(a.get("feas_min_m"), a.get("feas_min_s")),
                    _fmt_cell(a.get("violraw_m"), a.get("violraw_s"), "{:.2f}"),
                ]
                metric_means = [
                    a.get("obj_mean_m"),
                    a.get("obj_max_m"),
                    a.get("feas_min_m"),
                    a.get("violraw_m"),
                ]
            if n_failed == 1:
                # Dagger on the first metric cell; the caption explains it.
                metric_cells[0] = metric_cells[0] + r"$^{\dag}$"

        rows_data.append({
            "label": label,
            "is_anchor": is_anchor,
            "hparam_cells": hparam_cells,
            "metric_cells": metric_cells,
            "metric_means": metric_means,
            "n_failed": n_failed,
        })

    # Pass 2: bold cells whose text matches the column's lowest value; skip n_failed >= 2.
    for col_idx in bold_idxs:
        candidates = [
            (i, r["metric_means"][col_idx], r["metric_cells"][col_idx])
            for i, r in enumerate(rows_data)
            if r["metric_means"][col_idx] is not None
        ]
        if not candidates:
            continue
        best_mean = min(m for (_i, m, _t) in candidates)
        best_text = next(t for (_i, m, t) in candidates if m == best_mean)
        dagger_suffix = r"$^{\dag}$"
        for i, _m, t in candidates:
            if t != best_text:
                continue
            if t.endswith(dagger_suffix):
                # Preserve dagger outside \best so it stays a superscript.
                core = t[: -len(dagger_suffix)]
                rows_data[i]["metric_cells"][col_idx] = rf"\best{{{core}}}" + dagger_suffix
            else:
                rows_data[i]["metric_cells"][col_idx] = rf"\best{{{t}}}"

    for r in rows_data:
        line = " & ".join(r["hparam_cells"] + r["metric_cells"]) + " \\\\"
        lines.append(line)
        if r["is_anchor"]:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("}")  # close \footnotesize / \tabcolsep group
    return "\n".join(lines) + "\n"


def _render_markdown(
    ab: dict[str, Any],
    varying: list[str],
    base_hp: dict,
    agg: dict[str, dict[str, float]],
    single_bench: bool,
) -> str:
    def _cell_md(m, s, fmt="{:.2f}"):
        if m is None:
            return "-"
        if s is None:
            return fmt.format(m)
        return f"{fmt.format(m)} +/- {fmt.format(s)}"

    def _feas_md(m, s):
        if m is None:
            return "-"
        mp = m * 100
        if s is None:
            return f"{mp:.0f}%"
        return f"{mp:.0f} +/- {s * 100:.0f}%"

    display = ab.get("display") or {}
    display_labels = display.get("display_labels") or {}
    formats = display.get("format") or {}
    value_maps = display.get("value_map") or {}

    # Markdown keeps the scenario column (no bolding) but drops feas-mean and n.
    if single_bench:
        metric_cols = ["objective", "feasibility", "$\\|c(\\hat y)\\|_\\infty$"]
    else:
        metric_cols = ["obj (mean)", "obj (max)", "feas (min)",
                       "$\\|c(\\hat y)\\|_\\infty$"]

    header = (
        ["scenario"]
        + [display_labels.get(k, k) for k in varying]
        + metric_cols
    )
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")

    anchor_label = ab["anchor"]
    scenario_order = [anchor_label] + [
        s["label"] for s in ab["scenarios"] if s["label"] != anchor_label
    ]
    scn_by_label = {s["label"]: s for s in ab["scenarios"]}

    n_metric_cols = len(metric_cols)
    any_dagger = False
    any_dash = False
    for label in scenario_order:
        scn = scn_by_label[label]
        a = agg.get(label, {})
        is_anchor = (label == anchor_label)
        overrides = scn.get("overrides") or {}
        n_failed = a.get("n_failed", 0)
        row = [label]
        for key in varying:
            fmt = formats.get(key)
            vmap = value_maps.get(key)
            if is_anchor:
                row.append(_fmt_value(base_hp.get(key, ""), fmt, vmap))
            else:
                row.append(_fmt_value(overrides[key], fmt, vmap) if key in overrides else "")
        if n_failed >= 2:
            any_dash = True
            row.extend(["-"] * n_metric_cols)
            lines.append("| " + " | ".join(row) + " |")
            continue
        if single_bench:
            metric_cells = [
                _cell_md(a.get("obj_mean_m"), a.get("obj_mean_s")),
                _feas_md(a.get("feas_mean_m"), a.get("feas_mean_s")),
                _cell_md(a.get("violraw_m"), a.get("violraw_s"), "{:.2f}"),
            ]
        else:
            metric_cells = [
                _cell_md(a.get("obj_mean_m"), a.get("obj_mean_s")),
                _cell_md(a.get("obj_max_m"),  a.get("obj_max_s")),
                _feas_md(a.get("feas_min_m"),  a.get("feas_min_s")),
                _cell_md(a.get("violraw_m"), a.get("violraw_s"), "{:.2f}"),
            ]
        if n_failed == 1:
            any_dagger = True
            metric_cells[0] = metric_cells[0] + " *"
        row.extend(metric_cells)
        lines.append("| " + " | ".join(row) + " |")
    if any_dagger or any_dash:
        notes = []
        if any_dagger:
            notes.append("* = 1 seed failure (partial-seed mean)")
        if any_dash:
            notes.append("\"-\" = >=2 seed failures")
        lines.append("")
        lines.append("_" + "; ".join(notes) + "._")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="render an ablation table")
    p.add_argument("--ablation", required=True, type=Path)
    p.add_argument("--runs-parquet", default="runs.parquet", type=Path)
    p.add_argument("--format", choices=("md", "latex"), default="md")
    p.add_argument("--out", type=Path, default=None,
                   help="output path (default: stdout)")
    args = p.parse_args(argv)

    ab = _load_ablation_yaml(args.ablation)
    if not args.runs_parquet.exists():
        raise SystemExit(f"runs.parquet not found at {args.runs_parquet}; "
                         f"run scripts/aggregate.py first")

    df = _load_ablation_runs(args.runs_parquet, ab)
    if df.height == 0:
        print(f"[render_ablation] no matching runs for {ab['ablation_name']!r}",
              file=sys.stderr)
        return 1

    agg = _aggregate_per_scenario(df, ab)
    varying = _varying_hparam_keys(ab)
    base_hp = ab["base_hparams"]
    single_bench = (len(ab["benchmarks"]) == 1)

    if args.format == "latex":
        text = _render_latex(ab, varying, base_hp, agg, single_bench)
    else:
        text = _render_markdown(ab, varying, base_hp, agg, single_bench)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"[render_ablation] wrote {args.out}  "
              f"({len(ab['scenarios'])} scenarios, {df.height} runs)")
    else:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
