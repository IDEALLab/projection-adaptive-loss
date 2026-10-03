#!/usr/bin/env python
"""Tabulate tau-ablation constraint pressure (training loss share) and eval metrics per arm."""

import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np  # noqa: E402


def parse_metrics_jsonl(path: Path) -> list[dict]:
    """Load metrics.jsonl as list of dicts, filtering to training metrics only."""
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                # Filter out eval-only steps (those without loss/objective field)
                if "loss/objective" in record:
                    records.append(record)
    return records


def compute_constraint_share(metrics: list[dict], at_step: int | None = None) -> float:
    """Compute constraint share at a specific step or over a final window.

    Args:
        metrics: list of per-step metric dicts
        at_step: if None, compute mean over last 100 steps; else at specific step index

    Returns:
        constraint share as percentage
    """
    if not metrics:
        return 0.0

    if at_step is None:
        window = metrics[-100:]
    else:
        if at_step >= len(metrics):
            window = [metrics[-1]]
        else:
            window = [metrics[at_step]]

    loss_obj = np.array([m.get("loss/objective", 0.0) for m in window])
    loss_con = np.array([m.get("loss/constraint", 0.0) for m in window])
    loss_dis = np.array([m.get("loss/displacement", 0.0) for m in window])

    loss_obj_mean = loss_obj.mean()
    loss_con_mean = loss_con.mean()
    loss_dis_mean = loss_dis.mean()

    denom = abs(loss_obj_mean) + loss_con_mean + loss_dis_mean
    if denom > 0:
        return 100.0 * loss_con_mean / denom
    return 0.0


def compute_final_metrics(metrics: list[dict]) -> dict:
    """Compute final (last 100 steps) metrics."""
    if not metrics:
        return {}

    window = metrics[-100:]

    loss_obj = np.array([m.get("loss/objective", 0.0) for m in window])
    loss_con = np.array([m.get("loss/constraint", 0.0) for m in window])
    loss_dis = np.array([m.get("loss/displacement", 0.0) for m in window])
    w_c = np.array([m.get("residual/w_0", 0.0) for m in window])
    c_post = np.array([m.get("residual/c_post_0_mean", 0.0) for m in window])
    tau_eff = np.array([m.get("residual/tau_effective", 0.0) for m in window])

    loss_obj_mean = loss_obj.mean()
    loss_con_mean = loss_con.mean()
    loss_dis_mean = loss_dis.mean()
    w_c_mean = w_c.mean()
    c_post_mean = c_post.mean()
    tau_eff_mean = tau_eff.mean()

    denom = abs(loss_obj_mean) + loss_con_mean + loss_dis_mean
    con_share = 100.0 * loss_con_mean / denom if denom > 0 else 0.0
    obj_share = 100.0 * abs(loss_obj_mean) / denom if denom > 0 else 0.0

    c_post_tau_ratio = c_post_mean / tau_eff_mean if tau_eff_mean > 0 else 0.0

    return {
        "con_share_final": con_share,
        "obj_share_final": obj_share,
        "w_c_final": w_c_mean,
        "c_post_final": c_post_mean,
        "c_post_tau_ratio_final": c_post_tau_ratio,
    }


def find_run_dir(root: Path, variant: str) -> Path | None:
    """Find the run directory for a variant."""
    matching = list(root.glob(f"*_curvature_warp_{variant}_*"))
    return matching[0] if matching else None


def parse_analyze_output(output: str, variant: str) -> dict:
    """Parse output from analyze_curvature.py to extract eval metrics."""
    lines = output.split("\n")

    in_table = False
    header_row = None
    data_row = None

    for line in lines:
        if "core: feasibility" in line:
            in_table = True
            continue

        if in_table:
            if line.startswith("|"):
                if "variant" in line:
                    header_row = [h.strip() for h in line.split("|")[1:-1]]
                elif f"| {variant} |" in line or f"| {variant} " in line:
                    data_row = [d.strip() for d in line.split("|")[1:-1]]
                    break

    if not header_row or not data_row:
        return {}

    result = {}
    for h, d in zip(header_row, data_row, strict=False):
        if h.lower().strip() in ["variant", "kappa", "n", "n_feas", "n_elig", "n_ambig", "n_snap_used"]:
            continue
        result[h.strip()] = d.strip()

    return result


def format_percentage(val: float, decimals: int = 1) -> str:
    """Format as percentage with N decimals."""
    return f"{val:.{decimals}f}"


def format_sci(val: float, sig_figs: int = 2) -> str:
    """Format in scientific notation with N significant figures."""
    if val == 0:
        return "0.0"
    from math import floor, log10

    exponent = floor(log10(abs(val)))
    mantissa = val / (10 ** exponent)

    if sig_figs == 2:
        return f"{mantissa:.1f}e{exponent:+03d}"
    else:
        return f"{val:.{sig_figs}e}"


def main():
    """Extract and format tables for tau-ablation runs."""

    tau_arms = {
        "default": Path("runs/v5_warp_mvp"),
        "tau=1e-2": Path("runs/v5_tau_ablation/tau_loose"),
        "tau=1e10": Path("runs/v5_tau_ablation/tau_lossoff"),
        "tau=1.0": Path("runs/v5_tau_ablation/tau_mid_supplementary"),
    }

    variants = ["k0", "k6", "k10"]
    tau_mid_only_k10 = True  # tau=1.0 only has k10

    from pal.benchmarks.synthetic.curvature_warp import KAPPA_BY_VARIANT
    kappa_by_variant = KAPPA_BY_VARIANT

    # Table 1: training load distribution.
    print("## Table 1: Training Load Distribution (tau Ablation)")
    print()

    table1_rows = []

    for tau_name, runs_root in tau_arms.items():
        for variant in variants:
            if tau_name == "tau=1.0" and tau_mid_only_k10 and variant != "k10":
                continue

            run_dir = find_run_dir(runs_root, variant)
            if not run_dir:
                print(f"WARNING: no run found for {tau_name} {variant}", file=sys.stderr)
                continue

            metrics_path = run_dir / "metrics.jsonl"
            if not metrics_path.exists():
                print(f"WARNING: {metrics_path} not found", file=sys.stderr)
                continue

            metrics = parse_metrics_jsonl(metrics_path)

            # Assumes one step per epoch.
            ep100_idx = min(99, len(metrics) - 1)
            ep1000_idx = min(999, len(metrics) - 1)

            con_share_ep100 = compute_constraint_share(metrics, at_step=ep100_idx)
            con_share_ep1000 = compute_constraint_share(metrics, at_step=ep1000_idx)
            con_share_final = compute_constraint_share(metrics, at_step=None)

            final_metrics = compute_final_metrics(metrics)
            obj_share_final = final_metrics.get("obj_share_final", 0.0)
            w_c_final = final_metrics.get("w_c_final", 0.0)
            c_post_final = final_metrics.get("c_post_final", 0.0)
            c_post_tau_final = final_metrics.get("c_post_tau_ratio_final", 0.0)

            kappa = kappa_by_variant[variant]

            table1_rows.append({
                "tau_name": tau_name,
                "variant": variant,
                "kappa": kappa,
                "con_share_ep100": con_share_ep100,
                "con_share_ep1000": con_share_ep1000,
                "con_share_final": con_share_final,
                "obj_share_final": obj_share_final,
                "w_c_final": w_c_final,
                "c_post_final": c_post_final,
                "c_post_tau_final": c_post_tau_final,
            })

    tau_order = ["default", "tau=1e-2", "tau=1e10", "tau=1.0"]
    table1_rows.sort(key=lambda r: (
        tau_order.index(r["tau_name"]),
        r["kappa"]
    ))

    lines = []
    lines.append("| tau arm | variant | kappa | con share ~100 | con share ~1000 | con share final | obj share final | w_c final | c_post final | c_post/tau_eff final |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for row in table1_rows:
        line = (
            f"| {row['tau_name']} | {row['variant']} | {row['kappa']:g} | "
            f"{format_percentage(row['con_share_ep100'])} | "
            f"{format_percentage(row['con_share_ep1000'])} | "
            f"{format_percentage(row['con_share_final'])} | "
            f"{format_percentage(row['obj_share_final'])} | "
            f"{format_sci(row['w_c_final'], 2)} | "
            f"{format_sci(row['c_post_final'], 2)} | "
            f"{format_sci(row['c_post_tau_final'], 2)} |"
        )
        lines.append(line)

    table1 = "\n".join(lines)
    print(table1)
    print()

    # Table 2: eval-side outcome.
    print("## Table 2: Eval-Side Outcome per tau Arm")
    print()

    table2_rows = []

    for tau_name, runs_root in tau_arms.items():
        try:
            result = subprocess.run(
                [sys.executable, "scripts/analyze_curvature.py",
                 "--runs-root", str(runs_root), "--method", "pal_loggap"],
                capture_output=True,
                text=True,
                cwd=repo_root,
                timeout=120
            )
            analyze_output = result.stdout
        except Exception as e:
            print(f"WARNING: failed to run analyze_curvature.py for {tau_name}: {e}",
                  file=sys.stderr)
            analyze_output = ""

        for variant in variants:
            if tau_name == "tau=1.0" and tau_mid_only_k10 and variant != "k10":
                continue

            eval_metrics = parse_analyze_output(analyze_output, variant)

            if not eval_metrics:
                print(f"WARNING: no eval metrics found for {tau_name} {variant}", file=sys.stderr)
                continue

            kappa = kappa_by_variant[variant]
            feas = eval_metrics.get("feas", "-")
            d_p50 = eval_metrics.get("d p50", "-")
            d_p90 = eval_metrics.get("d p90", "-")
            gap_p50 = eval_metrics.get("gap p50", "-")
            gap_p90 = eval_metrics.get("gap p90", "-")

            table2_rows.append({
                "tau_name": tau_name,
                "variant": variant,
                "kappa": kappa,
                "feas": feas,
                "d_p50": d_p50,
                "d_p90": d_p90,
                "gap_p50": gap_p50,
                "gap_p90": gap_p90,
            })

    table2_rows.sort(key=lambda r: (
        tau_order.index(r["tau_name"]),
        r["kappa"]
    ))

    lines = []
    lines.append("| tau arm | variant | kappa | feasibility | d p50 | d p90 | gap p50 | gap p90 |")
    lines.append("|---|---|---|---|---|---|---|---|")

    for row in table2_rows:
        line = (
            f"| {row['tau_name']} | {row['variant']} | {row['kappa']:g} | "
            f"{row['feas']} | "
            f"{row['d_p50']} | "
            f"{row['d_p90']} | "
            f"{row['gap_p50']} | "
            f"{row['gap_p90']} |"
        )
        lines.append(line)

    table2 = "\n".join(lines)
    print(table2)
    print()

    print("## Notes")
    print()
    print("**Eval columns used:**")
    print("- `feas` (feasibility): from analyze_curvature.py's \"core\" table")
    print("- `d p50, d p90` (pre-repair distance to manifold): "
          "from analyze_curvature.py's \"core\" table")
    print("- `gap p50, gap p90` (optimality gap among feasible queries): "
          "from analyze_curvature.py's \"core\" table")
    print()
    print("**Repair iterations:**")
    print("- No repair-iteration or convergence columns exist in `eval_rows.parquet` or `metrics.jsonl`.")
    print("- Constraint pressure metrics (con share, obj share, w_c, c_post, c_post/tau_eff) extracted from `metrics.jsonl`.")
    print()


if __name__ == "__main__":
    main()
