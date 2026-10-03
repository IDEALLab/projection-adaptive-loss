"""Aggregate per-seed memory results into a method x bench summary."""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path


def _classify_failure(log_path: Path | None) -> str:
    if log_path is None or not log_path.exists():
        return "fail"
    try:
        text = log_path.read_text()
    except Exception:
        return "fail"
    if "OutOfMemoryError" in text or "CUDA out of memory" in text:
        return "OOM"
    if "_verify_batch_size" in text or "Expected more than 1 value" in text:
        return "BN-BS=1"
    if "input matrix is singular" in text or "_LinAlgError" in text:
        return "singular"
    return "fail"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("results_csv", type=Path)
    p.add_argument(
        "--logs-dir", type=Path, default=None,
        help="Directory containing per-combo logs (defaults to parent of results_csv).",
    )
    args = p.parse_args()

    logs_dir = args.logs_dir or args.results_csv.parent

    rows = list(csv.DictReader(args.results_csv.open()))
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["bench"])].append(r)

    print(f"# Aggregate from {args.results_csv}")
    print(f"# {len(rows)} rows, {len(groups)} (method x bench) groups\n")
    print(
        f"{'method':<14} {'bench':<22} {'n':>3} {'n_ok':>5}  "
        f"{'peak (mean +/- std)':<26}  fail tags"
    )
    print("-" * 90)

    for (method, bench), values in sorted(groups.items()):
        n = len(values)
        ok = [v for v in values if int(v["rc"]) == 0]
        n_ok = len(ok)
        peaks_ok_gb = [int(v["peak_gpu_mb"]) / 1024 for v in ok]

        if n_ok > 0:
            mean = statistics.fmean(peaks_ok_gb)
            std = statistics.stdev(peaks_ok_gb) if n_ok > 1 else 0.0
            peak_str = f"{mean:.2f} +/- {std:.2f} GB"
        else:
            peaks_all_gb = [int(v["peak_gpu_mb"]) / 1024 for v in values]
            peak_str = f"all fail; max partial {max(peaks_all_gb):.2f} GB"

        fail_tags = []
        for v in values:
            if int(v["rc"]) == 0:
                continue
            bench_safe = bench.replace("/", "_")
            log = logs_dir / f"_combo_{method}_{bench_safe}_seed{v['seed']}.log"
            fail_tags.append(_classify_failure(log))
        tag_summary = ",".join(sorted(set(fail_tags))) if fail_tags else "-"

        print(
            f"{method:<14} {bench:<22} {n:>3} {n_ok:>5}  "
            f"{peak_str:<26}  {tag_summary}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
