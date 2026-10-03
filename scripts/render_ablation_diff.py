"""Render the cell-by-cell delta between two ablation runs, with std added in quadrature.

Usage: python scripts/render_ablation_diff.py --ablation <yaml> --old-parquet A --new-parquet B
"""

from __future__ import annotations

import argparse
import math

import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from render_ablation import (  # noqa: E402
    _aggregate_per_scenario,
    _load_ablation_runs,
)

# YAML scenario_label -> table row ID, in table order. No-op overrides are omitted.
PAPER_LETTERS: dict[str, str] = {
    "full_pal":                    "Base",
    "abl_delta_0":                 "A",
    "abl_delta_1e-4":              "B",
    "abl_delta_1e-2":              "C",
    "abl_proj_lambdamin_0_delta_0":"D",
    "abl_rate_1e-3":               "E",
    "abl_rate_1e-1":               "F",
    "abl_rate_1e-1_decades_0.1":   "G",
    "abl_decades_0.2":             "H",
    "abl_decades_5.0":             "J",
    "abl_tauD_1e-2":               "K",
    "abl_lr_1e-3":                 "L",
    "abl_obj_yproj_tauD_tight":    "M",
    "abl_obj_yproj_tauD_loose":    "N",
    "abl_obj_yhat_tauD_loose":     "O",
    "abl_no_constraint_loss":      "P",
    "abl_no_disp_loss":            "Q",
    "abl_no_either_loss":          "R",
    "abl_monotone_mult":           "S",
}


def _delta(m_new: float | None, m_old: float | None) -> float | None:
    if m_new is None or m_old is None:
        return None
    return m_new - m_old


def _prop_std(s_new: float | None, s_old: float | None) -> float | None:
    if s_new is None and s_old is None:
        return None
    a = s_new or 0.0
    b = s_old or 0.0
    return math.sqrt(a * a + b * b)


def _value_plus_delta_cell(
    m_n, s_n, m_o, s_o, fmt: str = "{:.2f}", dfmt: str = "{:+.2f}",
) -> str:
    """Render `new_mean +/- new_std (Delta_mean +/- prop_std)` for a numeric metric."""
    if m_n is None:
        return "-"
    new = fmt.format(m_n) if s_n is None else f"{fmt.format(m_n)} +/- {fmt.format(s_n)}"
    dm = _delta(m_n, m_o)
    if dm is None:
        return new
    ds = _prop_std(s_n, s_o)
    if ds is None:
        return f"{new} ({dfmt.format(dm)})"
    return f"{new} ({dfmt.format(dm)} +/- {abs(ds):.2f})"


def _value_plus_delta_feas_cell(m_n, s_n, m_o, s_o) -> str:
    """Render `pct +/- pct% (+/-Delta pct +/- prop_pct%)` for a feasibility cell."""
    if m_n is None:
        return "-"
    pct = m_n * 100
    new = f"{pct:.0f}%" if s_n is None else f"{pct:.0f} +/- {s_n * 100:.0f}%"
    dm = _delta(m_n, m_o)
    if dm is None:
        return new
    ds = _prop_std(s_n, s_o)
    dm_pct = dm * 100
    if ds is None:
        return f"{new} ({dm_pct:+.0f}%)"
    return f"{new} ({dm_pct:+.0f} +/- {abs(ds) * 100:.0f}%)"


def render(
    ab: dict[str, Any],
    agg_new: dict[str, dict[str, float]],
    agg_old: dict[str, dict[str, float]],
) -> str:
    metric_cols = [
        "obj (mean) (Delta)",
        "obj (max) (Delta)",
        "feas (min) (Delta)",
        "||c(y_hat)||_inf (Delta)",
    ]
    header = ["ID"] + metric_cols
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("|" + "|".join("---" for _ in header) + "|")

    # Iterate in table order, dropping YAML scenarios without a row ID.
    scn_by_label = {s["label"]: s for s in ab["scenarios"]}

    for label, letter in PAPER_LETTERS.items():
        if label not in scn_by_label:
            # Fail loudly if the YAML lost a mapped scenario.
            raise KeyError(
                f"PAPER_LETTERS maps {label!r} -> {letter!r} but the ablation "
                f"YAML has no such scenario."
            )

        an = agg_new.get(label, {})
        ao = agg_old.get(label, {})
        dashed = (an.get("n_failed", 0) >= 2) or (ao.get("n_failed", 0) >= 2)

        row = [f"\\texttt{{{letter}}}"]
        if dashed:
            row.extend(["-"] * len(metric_cols))
            lines.append("| " + " | ".join(row) + " |")
            continue

        row.append(_value_plus_delta_cell(an.get("obj_mean_m"), an.get("obj_mean_s"),
                                          ao.get("obj_mean_m"), ao.get("obj_mean_s")))
        row.append(_value_plus_delta_cell(an.get("obj_max_m"), an.get("obj_max_s"),
                                          ao.get("obj_max_m"), ao.get("obj_max_s")))
        row.append(_value_plus_delta_feas_cell(an.get("feas_min_m"), an.get("feas_min_s"),
                                               ao.get("feas_min_m"), ao.get("feas_min_s")))
        row.append(_value_plus_delta_cell(an.get("violraw_m"), an.get("violraw_s"),
                                          ao.get("violraw_m"), ao.get("violraw_s")))
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append(
        "Per cell: `new_mean +/- new_std (Delta +/- propagated_std)`, where Delta = new "
        "- old and propagated_std = sqrt(s_new^2 + s_old^2). Row IDs "
        "(\\texttt{Base}, \\texttt{A}, ...) match the paper's ablation "
        "table; the YAML's `abl_tauC_1e-4` is a no-op override on synthetics "
        "and is not included in the paper table."
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation", required=True, type=Path)
    ap.add_argument("--old-parquet", required=True, type=Path)
    ap.add_argument("--new-parquet", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    ab = yaml.safe_load(args.ablation.read_text())

    df_old = _load_ablation_runs(args.old_parquet, ab)
    df_new = _load_ablation_runs(args.new_parquet, ab)

    agg_old = _aggregate_per_scenario(df_old, ab)
    agg_new = _aggregate_per_scenario(df_new, ab)

    out_text = render(ab, agg_new, agg_old)
    args.out.write_text(out_text)
    n_scn = len(ab["scenarios"])
    print(
        f"[render_ablation_diff] wrote {args.out}  "
        f"({n_scn} scenarios; old={args.old_parquet}, new={args.new_parquet})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
