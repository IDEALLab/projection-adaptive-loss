#!/usr/bin/env python3
"""Aggregate the 3-method x 6-bench x 10-seed repair-step ablation.

Prints the results table if all 180 runs are ok, else a coverage report (exit 1).

Usage:
    python scripts/x86_repair_ablation/aggregate.py --campaign-root <path>
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

# Must match launch.sbatch's METHODS / BENCHES / seed range.
METHODS = ["pal_loggap", "pal_sqp", "pal_ip"]
BENCHES = [
    "s1_sphere_track",
    "s2_active_set_switch",
    "s3_illcond_tube",
    "s4_qv_coupling",
    "s5_overdetermined",
    "s6_redundant_ineq",
]
SEEDS = list(range(10))


def expected_combos() -> set[tuple[str, str, int]]:
    return {(m, b, s) for m in METHODS for b in BENCHES for s in SEEDS}


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def sum_metric(metrics_path: Path, key: str) -> float:
    """Sum a presence-gated counter key over every row of metrics.jsonl."""
    if not metrics_path.exists():
        return 0.0
    total = 0.0
    with metrics_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            total += row.get(key, 0) or 0
    return total


def discover_runs(campaign_root: Path) -> dict[tuple[str, str, int], list[Path]]:
    """Map each (method, bench, seed) combo to the run dirs found for it."""
    found: dict[tuple[str, str, int], list[Path]] = {}
    if not campaign_root.exists():
        return found
    for run_dir in sorted(campaign_root.iterdir()):
        if not run_dir.is_dir():
            continue
        config = load_json(run_dir / "config.json")
        if config is None:
            continue
        method = config.get("method")
        bench = config.get("benchmark_id")
        seed = config.get("seed")
        if method is None or bench is None or seed is None:
            continue
        combo = (method, bench, int(seed))
        found.setdefault(combo, []).append(run_dir)
    return found


def fmt(mean: float, std: float, sci: bool = True) -> str:
    if sci:
        return f"{mean:.3e} +/- {std:.1e}"
    return f"{mean:.4f} +/- {std:.4f}"


def build_table(runs: dict[tuple[str, str, int], Path]) -> str:
    """runs: combo -> the single verified-ok run dir for that combo."""
    lines = []
    lines.append(
        "| bench | "
        + " | ".join(
            f"{m} obj_post | {m} viol_post | {m} feas | {m} inf_iters(mean)"
            for m in METHODS
        )
        + " |"
    )
    lines.append("|" + "---|" * (1 + 4 * len(METHODS)))

    for bench in BENCHES:
        row_cells = [bench]
        for method in METHODS:
            obj_post, viol_post, feas, inf_iters = [], [], [], []
            for seed in SEEDS:
                run_dir = runs[(method, bench, seed)]
                final = load_json(run_dir / "final.json") or {}
                obj_post.append(final.get("obj_mean_post", float("nan")))
                viol_post.append(final.get("viol_max_post", float("nan")))
                feas.append(final.get("feasibility_post", float("nan")))
                inf_iters.append(final.get("inf_iters_median", float("nan")))
            row_cells.append(
                fmt(statistics.mean(obj_post), statistics.stdev(obj_post))
            )
            row_cells.append(
                fmt(statistics.mean(viol_post), statistics.stdev(viol_post))
            )
            row_cells.append(
                fmt(statistics.mean(feas), statistics.stdev(feas), sci=False)
            )
            row_cells.append(f"{statistics.fmean(inf_iters):.2f}")
        lines.append("| " + " | ".join(row_cells) + " |")

    lines.append("")
    lines.append(
        "inf_iters(mean): mean over the 10 seeds of each run's "
        "`inf_iters_median` (per-sample iteration counts are not persisted, "
        "so the per-run statistic remains the median)."
    )
    return "\n".join(lines)


def build_telemetry_appendix(runs: dict[tuple[str, str, int], Path]) -> str:
    lines = ["", "### Telemetry appendix (totals over all 60 runs per method)", ""]

    sqp_qp_failures = 0.0
    sqp_clarabel_rescues = 0.0
    sqp_partition_mismatches = 0.0
    for bench in BENCHES:
        for seed in SEEDS:
            run_dir = runs[("pal_sqp", bench, seed)]
            mpath = run_dir / "metrics.jsonl"
            sqp_qp_failures += sum_metric(mpath, "sqp/qp_failures")
            sqp_clarabel_rescues += sum_metric(mpath, "sqp/clarabel_rescues")
            sqp_partition_mismatches += sum_metric(mpath, "sqp/partition_mismatches")
    lines.append(
        f"- pal_sqp: qp_failures={sqp_qp_failures:.0f}, "
        f"clarabel_rescues={sqp_clarabel_rescues:.0f}, "
        f"partition_mismatches={sqp_partition_mismatches:.0f}"
    )

    ip_solve_failures = 0.0
    ip_pinv_fallbacks = 0.0
    for bench in BENCHES:
        for seed in SEEDS:
            run_dir = runs[("pal_ip", bench, seed)]
            mpath = run_dir / "metrics.jsonl"
            ip_solve_failures += sum_metric(mpath, "ip/solve_failures")
            ip_pinv_fallbacks += sum_metric(mpath, "ip/pinv_fallbacks")
    lines.append(
        f"- pal_ip: solve_failures={ip_solve_failures:.0f}, "
        f"pinv_fallbacks={ip_pinv_fallbacks:.0f}"
    )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign-root",
        type=Path,
        required=True,
        help="the --runs-root directory shared by every launch.sbatch task "
        "(the same path 180 run dirs were auto-created under).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional path to also write the markdown report to.",
    )
    args = parser.parse_args()

    expected = expected_combos()
    found_map = discover_runs(args.campaign_root)

    missing: list[tuple[str, str, int]] = []
    duplicates: list[tuple[tuple[str, str, int], list[Path]]] = []
    failed: list[tuple[tuple[str, str, int], Path, str]] = []
    ok_runs: dict[tuple[str, str, int], Path] = {}

    for combo in sorted(expected):
        dirs = found_map.get(combo, [])
        if not dirs:
            missing.append(combo)
            continue
        if len(dirs) > 1:
            duplicates.append((combo, dirs))
            continue
        run_dir = dirs[0]
        status = load_json(run_dir / "status.json")
        status_val = (status or {}).get("status")
        if status_val != "ok":
            failed.append((combo, run_dir, str(status_val)))
            continue
        ok_runs[combo] = run_dir

    report_lines = []
    report_lines.append("# PAL repair-step ablation: aggregation report")
    report_lines.append("")
    report_lines.append(f"campaign root: {args.campaign_root}")
    report_lines.append(
        f"expected combos: {len(expected)} | found ok: {len(ok_runs)} | "
        f"missing: {len(missing)} | duplicate: {len(duplicates)} | "
        f"failed/non-ok: {len(failed)}"
    )
    report_lines.append("")

    if missing:
        report_lines.append(f"## Missing ({len(missing)})")
        for m, b, s in missing:
            report_lines.append(f"- {m} / {b} / seed{s}")
        report_lines.append("")

    if duplicates:
        report_lines.append(f"## Duplicate run dirs ({len(duplicates)} combos)")
        for (m, b, s), dirs in duplicates:
            report_lines.append(f"- {m} / {b} / seed{s}: {[str(d) for d in dirs]}")
        report_lines.append("")

    if failed:
        report_lines.append(f"## Failed / non-ok status ({len(failed)})")
        for (m, b, s), run_dir, status_val in failed:
            report_lines.append(f"- {m} / {b} / seed{s}: status={status_val} ({run_dir})")
        report_lines.append("")

    if ok_runs:
        present_summary = ", ".join(
            f"{m}/{b}/seed{s}" for (m, b, s) in sorted(ok_runs)
        )
        report_lines.append(f"## Present ok runs ({len(ok_runs)})")
        report_lines.append(present_summary)
        report_lines.append("")

    incomplete = bool(missing or duplicates or failed)

    if incomplete:
        report_lines.append(
            "STATUS: INCOMPLETE, refusing to compute the results table. "
            f"{len(missing) + len(duplicates) + len(failed)} of "
            f"{len(expected)} combos are not a single verified-ok run."
        )
        text = "\n".join(report_lines)
        print(text)
        if args.out is not None:
            args.out.write_text(text + "\n")
        return 1

    report_lines.append("STATUS: COMPLETE, all 180 combos present, unique, status=ok.")
    report_lines.append("")
    report_lines.append("## Results (mean +/- std over 10 seeds)")
    report_lines.append("")
    report_lines.append(build_table(ok_runs))
    report_lines.append(build_telemetry_appendix(ok_runs))

    text = "\n".join(report_lines)
    print(text)
    if args.out is not None:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
