"""Tests for the IPOPT restart-shard merge helper (pure data, no cyipopt or torch)."""

from __future__ import annotations

import math

import pytest

from pal.baselines.ipopt.shard_merge import (
    _select_best_record,
    aggregate_post_metrics,
    merge_per_query,
)


def _record(idx: int, obj: float, viol: float, feasible: bool, status: int = 0) -> dict:
    return {
        "restart_idx": idx,
        "x": [0.0, 0.0],
        "obj": obj,
        "max_violation": viol,
        "feasible": feasible,
        "status": status,
    }


def test_select_best_prefers_feasible_over_infeasible() -> None:
    records = [
        _record(0, obj=1.0, viol=0.05, feasible=False),
        _record(1, obj=10.0, viol=0.0, feasible=True),
    ]
    winner = _select_best_record(records)
    assert winner["restart_idx"] == 1


def test_select_best_among_feasible_picks_lowest_obj() -> None:
    records = [
        _record(0, obj=5.0, viol=0.0, feasible=True),
        _record(1, obj=2.0, viol=0.0, feasible=True),
        _record(2, obj=3.0, viol=0.0, feasible=True),
    ]
    assert _select_best_record(records)["restart_idx"] == 1


def test_select_best_all_infeasible_picks_lowest_violation() -> None:
    records = [
        _record(0, obj=1.0, viol=0.5, feasible=False),
        _record(1, obj=10.0, viol=0.05, feasible=False),
    ]
    assert _select_best_record(records)["restart_idx"] == 1


def test_select_best_handles_none_obj() -> None:
    """Non-finite objectives are stored as None (clamped), sorting must not raise."""
    records = [
        _record(0, obj=None, viol=0.5, feasible=False),
        _record(1, obj=1.0, viol=0.05, feasible=False),
    ]
    assert _select_best_record(records)["restart_idx"] == 1


def test_merge_per_query_unions_restarts_across_shards() -> None:
    shard_0 = [
        {"restarts": [_record(0, 1.0, 0.0, True), _record(2, 5.0, 0.0, True)]},
    ]
    shard_1 = [
        {"restarts": [_record(1, 0.5, 0.0, True), _record(3, 2.0, 0.0, True)]},
    ]
    merged = merge_per_query([shard_0, shard_1])
    assert len(merged) == 1
    restart_idxs = [r["restart_idx"] for r in merged[0]["restarts"]]
    assert restart_idxs == [0, 1, 2, 3]
    # Global winner: feasible with lowest obj -> restart_idx=1, obj=0.5
    assert merged[0]["winner_restart_idx"] == 1
    assert merged[0]["obj"] == 0.5
    assert merged[0]["n_feasible_restarts"] == 4


def test_merge_per_query_rejects_duplicate_restart_idx() -> None:
    shard_0 = [{"restarts": [_record(0, 1.0, 0.0, True)]}]
    shard_1 = [{"restarts": [_record(0, 0.5, 0.0, True)]}]  # same restart_idx
    with pytest.raises(ValueError, match="duplicate restart_idx"):
        merge_per_query([shard_0, shard_1])


def test_merge_per_query_rejects_inconsistent_query_count() -> None:
    shard_0 = [
        {"restarts": [_record(0, 1.0, 0.0, True)]},
        {"restarts": [_record(0, 2.0, 0.0, True)]},
    ]
    shard_1 = [{"restarts": [_record(1, 0.5, 0.0, True)]}]
    with pytest.raises(ValueError, match="per_query entries"):
        merge_per_query([shard_0, shard_1])


def test_merge_preserves_per_query_independence() -> None:
    """Restart unions are PER-QUERY, query 0's restarts must not contaminate query 1."""
    shard_0 = [
        {"restarts": [_record(0, 1.0, 0.0, True)]},  # q0
        {"restarts": [_record(0, 100.0, 0.0, True)]},  # q1
    ]
    shard_1 = [
        {"restarts": [_record(1, 0.5, 0.0, True)]},  # q0
        {"restarts": [_record(1, 50.0, 0.0, True)]},  # q1
    ]
    merged = merge_per_query([shard_0, shard_1])
    assert merged[0]["obj"] == 0.5
    assert merged[1]["obj"] == 50.0


def test_aggregate_post_metrics_recomputes_from_merged() -> None:
    merged = [
        {"obj": 1.0, "max_violation": 0.0, "feasible": True, "status": 0,
         "n_feasible_restarts": 1, "restarts": [], "winner_restart_idx": 0},
        {"obj": 3.0, "max_violation": 0.05, "feasible": False, "status": 0,
         "n_feasible_restarts": 0, "restarts": [], "winner_restart_idx": 0},
        {"obj": 2.0, "max_violation": 0.0, "feasible": True, "status": 0,
         "n_feasible_restarts": 1, "restarts": [], "winner_restart_idx": 0},
    ]
    post = aggregate_post_metrics(merged)
    assert post["obj_mean_post"] == pytest.approx(2.0)  # mean(1, 3, 2)
    assert post["viol_max_post"] == pytest.approx(0.05)
    assert post["feasibility_post"] == pytest.approx(2 / 3)


def test_aggregate_post_metrics_handles_empty() -> None:
    post = aggregate_post_metrics([])
    assert math.isnan(post["obj_mean_post"])
    assert math.isnan(post["viol_max_post"])
    assert math.isnan(post["feasibility_post"])
