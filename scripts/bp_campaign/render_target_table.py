"""Render the breaking-point target table (4 metric blocks x 8 kappa) from
`analyze_curvature.py --campaign --out-json` output.

Usage: python scripts/bp_campaign/render_target_table.py campaign_results.json > target_table.md
"""
from __future__ import annotations

import json
import sys

KAPPA = {"k0": "kappa=0", "k4": "kappa=1", "k6": "kappa=10", "k8": "kappa=1e2", "k10": "kappa=1e3",
         "k11": "kappa=1e4", "k12": "kappa=1e5", "k13": "kappa=1e6"}
ARMS = ["pal_loggap[tau=1e-4]", "pal_loggap[tau=1e-2]", "pal_loggap[tau=1]", "alm",
        "alm_bolton", "dc3", "enforce_v4", "fsnet", "snarenet"]
BLOCKS = [
    ("feas", "Feasibility (fraction of 512 eval queries, mean+/-std over seeds; "
             "(n=..)=completed seeds<10; PENDING=no completed seed)", "{:.3f}"),
    ("gap", "Mean optimality gap, feasible-only (mean+/-std over runs with >=1 feasible "
            "query; (k/nf)=runs contributing)", "{:+.2e}"),
    ("grad_share", "Gradient ratio ||grad L_con||/||grad L_tot||, last-5-logged mean "
                   "(*=borrowed from source alm run; v4 = post-switch phase)", "{:.3f}"),
    ("contraction", "Repair contraction c_post/c_pre at last logged epoch "
                           "(NA=method logs neither / structural)", "{:.2e}"),
]


def cell_text(cell: dict | None, metric: str, fmt: str) -> str:
    """Format one (arm, variant, metric) cell."""
    if cell is None or cell["n_completed"] == 0:
        return "PENDING"
    m = cell["metrics"].get(metric)
    if m is None or m.get("n", 0) == 0 or m.get("mean") is None:
        return "NA"
    n, n_runs = m["n"], m.get("n_runs", cell["n_completed"])
    txt = fmt.format(m["mean"])
    if n > 1 and m.get("std") is not None:
        txt += "+/-" + fmt.format(m["std"])
    tags = []
    if metric == "gap" and n < n_runs:
        tags.append(f"{n}/{n_runs}f")
    if cell["n_completed"] < cell["n_expected"]:
        tags.append(f"n={cell['n_completed']}")
    if metric == "grad_share" and cell.get("grad_share_shared"):
        tags.append("*")
    return txt + (f" ({', '.join(tags)})" if tags else "")


def main(path: str) -> None:
    """Print the markdown table for one campaign json."""
    d = json.load(open(path))
    cells = {(c["arm"], c["variant"]): c for c in d["cells"]}
    arms = [a for a in ARMS if a in d["arms"]] + sorted(set(d["arms"]) - set(ARMS))
    for metric, title, fmt in BLOCKS:
        print(f"### {title}\n")
        print("| arm | " + " | ".join(KAPPA.values()) + " |")
        print("|---|" + "---|" * len(KAPPA))
        for arm in arms:
            row = [cell_text(cells.get((arm, k)), metric, fmt) for k in KAPPA]
            print(f"| {arm} | " + " | ".join(row) + " |")
        print()


if __name__ == "__main__":
    main(sys.argv[1])
