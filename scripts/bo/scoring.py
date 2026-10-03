#!/usr/bin/env python3
"""Map a trial's per-cell ``result.json`` files to records and score them with the metric module.

Enforces the exact cell set and the config fingerprint before scoring. No metric math here.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pal.eval.table1_constants import TABLE1_BENCHES
from pal.eval.table1_metrics import (
    BOObjective,
    aggregate_cells,
    applicable_benches,
    compute_bo_objective,
    run_rows_from_records,
)

# result.json statuses that carry legitimate scientific data.
_SCOREABLE = {"ok", "diverged"}


class ScoringError(RuntimeError):
    """Cell-set or fingerprint invariant violation (hard error, never a score)."""


def expected_cell_keys(
    method: str, seeds: Sequence[int], benches: Sequence[str] = TABLE1_BENCHES
) -> set[tuple[str, int]]:
    """The (bench, seed) cells a trial must produce (structural exclusions
    dropped: dc3 x s5 is never generated)."""
    return {(b, int(s)) for b in applicable_benches(method, benches) for s in seeds}


def read_result(path: str | Path) -> dict[str, Any] | None:
    """Load a cell ``result.json``; None if absent/malformed."""
    path = Path(path)
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def result_status(result: Mapping[str, Any] | None) -> str | None:
    if result is None:
        return None
    return result.get("status")


def record_from_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Shape one cell ``result.json`` into a metric-module record.

    Non-"ok" statuses (diverged/infra) keep their status so the seed is counted
    as *attempted* but not clean -> the metric imputes worst-case for it.
    """
    fp = result.get("fingerprint", {}) or {}
    final = result.get("final") or {}
    return {
        "method": fp.get("method"),
        "benchmark_id": fp.get("bench"),
        "seed": fp.get("seed"),
        "status": result.get("status"),
        "obj_mean_post": final.get("obj_mean_post"),
        "feasibility_post": final.get("feasibility_post"),
        "viol_max_post": final.get("viol_max_post"),
        "n_queries": final.get("n_queries"),
        "tolerance": final.get("tolerance"),
        "wall_start": None,  # one row per (method,bench,seed) in a trial; dedupe moot
    }


@dataclass(frozen=True)
class TrialScore:
    objective: BOObjective
    records: list[dict[str, Any]]
    per_bench: dict[str, dict[str, float | None]]  # bench -> {feas_bo, obj_bo, l2_b, l3_b}
    worst: bool  # True if NaN/blowup S forced the worst tuple


def _worst_objective(n_benches: int, n_seeds: int) -> BOObjective:
    """Worst legitimate tuple (fully infeasible). Used when S is non-finite."""
    return BOObjective(
        l1=0.0, l2=1.0, l3=1.0, scalar=float("-inf"),
        n_applicable_benches=n_benches, n_seeds=n_seeds, n_queries=0,
        n_diverged_cells=n_benches, n_diverged_seeds=n_benches * n_seeds, diverged=True,
    )


def score_trial(
    method: str,
    expected_overrides: Mapping[str, str],
    seeds: Sequence[int],
    results: Mapping[tuple[str, int], Mapping[str, Any]],
    benches: Sequence[str] = TABLE1_BENCHES,
) -> TrialScore:
    """Validate + score one trial. ``results`` maps (bench, seed) -> result.json.

    Raises ``ScoringError`` on any cell-set or fingerprint mismatch.
    """
    expected = expected_cell_keys(method, seeds, benches)
    got = set(results.keys())
    if got != expected:
        raise ScoringError(
            f"cell set mismatch for {method}: "
            f"missing={sorted(expected - got)} extra={sorted(got - expected)}"
        )

    exp_ov = {str(k): str(v) for k, v in expected_overrides.items()}
    for (bench, seed), result in results.items():
        status = result.get("status")
        if status not in _SCOREABLE:
            raise ScoringError(
                f"cell {method}/{bench}/seed{seed} not scoreable (status={status!r}); "
                "infra failures must be retried/abandoned before scoring"
            )
        fp = result.get("fingerprint", {}) or {}
        req = {str(k): str(v) for k, v in (fp.get("requested_overrides") or {}).items()}
        if req != exp_ov:
            raise ScoringError(
                f"config fingerprint mismatch for {method}/{bench}/seed{seed}: "
                f"result={req} expected={exp_ov}"
            )
        if fp.get("method") != method or fp.get("bench") != bench or int(fp.get("seed")) != int(seed):
            raise ScoringError(
                f"cell identity mismatch: key=({method},{bench},{seed}) "
                f"fingerprint=({fp.get('method')},{fp.get('bench')},{fp.get('seed')})"
            )

    records = [record_from_result(results[k]) for k in sorted(results)]
    rows = run_rows_from_records(records)
    cells = aggregate_cells(rows, methods=[method], benches=list(benches))
    obj = compute_bo_objective(cells, method, benches=list(benches))

    per_bench: dict[str, dict[str, float | None]] = {}
    for b in applicable_benches(method, benches):
        cm = cells[(method, b)]
        per_bench[b] = {
            "feas_bo": cm.feas_bo,
            "obj_bo": cm.obj_bo,
            "l2_b": cm.l2_b,
            "l3_b": cm.l3_b,
        }

    worst = not math.isfinite(obj.scalar)
    if worst:
        obj = _worst_objective(obj.n_applicable_benches, obj.n_seeds)
    return TrialScore(objective=obj, records=records, per_bench=per_bench, worst=worst)
