"""Merge IPOPT restart-sharded runs back into a single (method, bench, seed) view.

Unions per-query restart records across shards and re-applies Deb dominance.
"""

from __future__ import annotations

import math
from typing import Any


def _select_best_record(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Deb dominance over JSON-safe restart records (``None`` counts as inf)."""
    inf = float("inf")
    feasible = [r for r in records if r["feasible"]]
    if feasible:
        return min(
            feasible,
            key=lambda r: r["obj"] if r["obj"] is not None else inf,
        )
    return min(
        records,
        key=lambda r: (
            r["max_violation"] if r["max_violation"] is not None else inf,
            r["obj"] if r["obj"] is not None else inf,
        ),
    )


def merge_per_query(
    shard_per_queries: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Union restart records across sibling shards and re-pick winners.

    `shard_per_queries[k][i]` is shard k's diag dict for query i (the
    `extras["per_query"]` payload from one run's `final.json`).

    Returns a single per_query list (length n_queries) carrying:
      - `obj`, `max_violation`, `feasible`, `status` from the global winner
      - `n_feasible_restarts` summed across shards
      - `restarts` = unioned restart records (sorted by restart_idx)
      - `winner_restart_idx` = the global winner's restart_idx
    """
    if not shard_per_queries:
        return []
    n_queries = len(shard_per_queries[0])
    for k, spq in enumerate(shard_per_queries):
        if len(spq) != n_queries:
            raise ValueError(
                f"shard {k} has {len(spq)} per_query entries, expected {n_queries}"
            )

    merged: list[dict[str, Any]] = []
    for qi in range(n_queries):
        union: list[dict[str, Any]] = []
        for spq in shard_per_queries:
            union.extend(spq[qi].get("restarts", []))
        if not union:
            raise ValueError(
                f"query {qi} has no restart records across {len(shard_per_queries)} shards"
            )
        union.sort(key=lambda r: r["restart_idx"])
        idxs = [r["restart_idx"] for r in union]
        if len(set(idxs)) != len(idxs):
            raise ValueError(
                f"query {qi} has duplicate restart_idx values across shards: {idxs}"
            )
        winner = _select_best_record(union)
        merged.append(
            {
                "obj": winner["obj"],
                "max_violation": winner["max_violation"],
                "feasible": winner["feasible"],
                "status": winner["status"],
                "n_feasible_restarts": sum(1 for r in union if r["feasible"]),
                "restarts": union,
                "winner_restart_idx": winner["restart_idx"],
            }
        )
    return merged


def aggregate_post_metrics(merged_per_query: list[dict[str, Any]]) -> dict[str, float]:
    """Re-derive flat `final.json` post metrics from merged per-query winners (post == raw)."""
    if not merged_per_query:
        return {
            "obj_mean_post": math.nan,
            "viol_max_post": math.nan,
            "feasibility_post": math.nan,
        }
    objs = [m["obj"] for m in merged_per_query if m["obj"] is not None]
    viols = [m["max_violation"] for m in merged_per_query if m["max_violation"] is not None]
    feas = [bool(m["feasible"]) for m in merged_per_query]
    obj_mean = sum(objs) / len(objs) if objs else math.nan
    viol_max = max(viols) if viols else math.nan
    feas_frac = sum(feas) / len(feas) if feas else math.nan
    return {
        "obj_mean_post": obj_mean,
        "viol_max_post": viol_max,
        "feasibility_post": feas_frac,
    }
