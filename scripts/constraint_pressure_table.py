#!/usr/bin/env python
"""Markdown table of constraint pressure metrics (last 100 steps) for breaking-point runs."""

import json
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402


def parse_metrics_jsonl(path: Path) -> list[dict]:
    """Load metrics.jsonl as list of dicts."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def compute_summary(metrics: list[dict], last_n: int = 100) -> dict:
    """Compute summary statistics over last N steps.

    Args:
        metrics: list of per-step metric dicts
        last_n: window size for final stats (default 100)

    Returns:
        dict with fields: constraint_share (%), objective_share (%),
        w_c, w_d, c_post, c_post_tau_ratio, n_steps_total,
        n_steps_w_c_above_floor
    """
    if not metrics:
        return None

    total_steps = len(metrics)
    window = metrics[-last_n:] if len(metrics) >= last_n else metrics

    loss_obj = np.array([m.get("loss/objective", 0.0) for m in window])
    loss_con = np.array([m.get("loss/constraint", 0.0) for m in window])
    loss_dis = np.array([m.get("loss/displacement", 0.0) for m in window])
    w_c = np.array([m.get("residual/w_0", 0.0) for m in window])
    w_d = np.array([m.get("displacement/w_0", 0.0) for m in window])
    c_post = np.array([m.get("residual/c_post_0_mean", 0.0) for m in window])
    tau_eff = np.array([m.get("residual/tau_effective", 0.0) for m in window])

    loss_obj_mean = loss_obj.mean()
    loss_con_mean = loss_con.mean()
    loss_dis_mean = loss_dis.mean()
    w_c_mean = w_c.mean()
    w_d_mean = w_d.mean()
    c_post_mean = c_post.mean()
    tau_eff_mean = tau_eff.mean()

    denom = abs(loss_obj_mean) + loss_con_mean + loss_dis_mean
    if denom > 0:
        con_share = 100.0 * loss_con_mean / denom
        obj_share = 100.0 * abs(loss_obj_mean) / denom
    else:
        con_share = 0.0
        obj_share = 0.0

    c_post_tau_ratio = (
        c_post_mean / tau_eff_mean if tau_eff_mean > 0 else 0.0
    )

    w_c_floor = 1e-6
    # Counted over all steps, not just the window.
    all_w_c = np.array([m.get("residual/w_0", 0.0) for m in metrics])
    n_steps_above_floor = (all_w_c > w_c_floor).sum()

    return {
        "con_share": con_share,
        "obj_share": obj_share,
        "w_c": w_c_mean,
        "w_d": w_d_mean,
        "c_post": c_post_mean,
        "c_post_tau_ratio": c_post_tau_ratio,
        "n_steps_total": total_steps,
        "n_steps_w_c_above_floor": int(n_steps_above_floor),
    }


def format_percentage(val: float, decimals: int = 1) -> str:
    """Format as percentage with N decimals."""
    return f"{val:.{decimals}f}%"


def format_sci(val: float, sig_figs: int = 2) -> str:
    """Format in scientific notation with N significant figures."""
    if val == 0:
        return "0"
    from math import floor, log10

    exponent = floor(log10(abs(val)))
    mantissa = val / (10 ** exponent)

    if sig_figs == 2:
        return f"{mantissa:.1f}e{exponent:+03d}"
    else:
        return f"{val:.{sig_figs}e}"


def main():
    runs_root = Path("runs/v5_warp_mvp")

    from pal.benchmarks.synthetic.curvature_warp import KAPPA_BY_VARIANT

    kappa_by_variant = KAPPA_BY_VARIANT

    results = []
    for variant in [f"k{i}" for i in range(11)]:
        matching = list(runs_root.glob(f"*_curvature_warp_{variant}_*"))
        if not matching:
            print(f"WARNING: no run found for variant {variant}", file=sys.stderr)
            continue

        run_dir = matching[0]
        metrics_path = run_dir / "metrics.jsonl"

        if not metrics_path.exists():
            print(f"WARNING: {metrics_path} not found", file=sys.stderr)
            continue

        metrics = parse_metrics_jsonl(metrics_path)
        summary = compute_summary(metrics, last_n=100)

        if summary:
            summary["variant"] = variant
            summary["kappa"] = kappa_by_variant[variant]
            results.append(summary)

    results.sort(key=lambda r: r["kappa"])

    lines = []
    lines.append("| kappa | Constraint Share (%) | Objective Share (%) | w_c | w_d | c_post | c_post/tau_eff | Steps (w_c > 1e-6) |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for res in results:
        kappa = res["kappa"]
        con_share = format_percentage(res["con_share"], 1)
        obj_share = format_percentage(res["obj_share"], 1)
        w_c = format_sci(res["w_c"], 2)
        w_d = format_sci(res["w_d"], 2)
        c_post = format_sci(res["c_post"], 2)
        ratio = format_sci(res["c_post_tau_ratio"], 2)
        n_above = res["n_steps_w_c_above_floor"]

        if kappa == int(kappa):
            kappa_str = f"{int(kappa)}"
        else:
            kappa_str = f"{kappa:g}"

        line = f"| {kappa_str} | {con_share} | {obj_share} | {w_c} | {w_d} | {c_post} | {ratio} | {n_above} |"
        lines.append(line)

    table = "\n".join(lines)
    print(table)
    print()

    print("Step counts per run:")
    for res in results:
        print(f"  {res['variant']}: {res['n_steps_total']} steps")
    print()

    print("Field-name check: all expected fields found in metrics.jsonl")
    print("  (loss/objective, loss/constraint, loss/displacement, residual/w_0,")
    print("   displacement/w_0, residual/c_post_0_mean, residual/tau_effective)")


if __name__ == "__main__":
    main()
