"""Tests for pal.eval.frozen_points."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pal.benchmarks.base import BenchmarkSpec
from pal.eval.frozen_points import FrozenPoints, load_frozen_points


def _write(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "fp.json"
    p.write_text(json.dumps(payload))
    return p


def _spec(zeta_dim: int, condition_dim: int, bench_id: str = "b00_test") -> BenchmarkSpec:
    return BenchmarkSpec(
        id=bench_id,
        family=bench_id.split("/", 1)[0],
        variant=None,
        dim=zeta_dim,
        n_eq=0,
        n_ineq=0,
        constraint_names=(),
        constraint_types=(),
        output_bounds=(torch.zeros(zeta_dim), torch.ones(zeta_dim)),
        condition_dim=condition_dim,
        zeta_dim=zeta_dim,
        tolerance=1e-4,
        cost="cheap",
        recommended_device="cpu",
    )


def test_load_basic_payload(tmp_path):
    path = _write(tmp_path, {
        "benchmark": "b00_test",
        "frozen_at": "2026-04-30",
        "rationale": "smoke",
        "points": [
            {"zeta": [0.1, 0.2], "condition": [1.0]},
            {"zeta": [0.3, 0.4], "condition": [2.0]},
        ],
        "ipopt": {"max_iter": 500, "tol": 1e-6},
    })
    fp = load_frozen_points(path)
    assert isinstance(fp, FrozenPoints)
    assert fp.benchmark == "b00_test"
    assert fp.frozen_at == "2026-04-30"
    assert len(fp) == 2
    assert fp.zeta.shape == (2, 2)
    assert fp.conditions.shape == (2, 1)
    assert fp.ipopt == {"max_iter": 500, "tol": 1e-6}


def test_to_query_validates_dims(tmp_path):
    path = _write(tmp_path, {
        "benchmark": "b00_test",
        "frozen_at": "2026-04-30",
        "rationale": "",
        "points": [{"zeta": [0.1, 0.2], "condition": [1.0]}],
        "ipopt": {},
    })
    fp = load_frozen_points(path)

    q = fp.to_query(_spec(zeta_dim=2, condition_dim=1))
    assert q.zeta.shape == (1, 2)
    assert q.conditions.shape == (1, 1)

    with pytest.raises(ValueError, match="zeta dim mismatch"):
        fp.to_query(_spec(zeta_dim=3, condition_dim=1))

    with pytest.raises(ValueError, match="conditions dim mismatch"):
        fp.to_query(_spec(zeta_dim=2, condition_dim=2))


def test_to_query_rejects_wrong_benchmark(tmp_path):
    path = _write(tmp_path, {
        "benchmark": "b00_test",
        "points": [{"zeta": [0.0], "condition": []}],
        "ipopt": {},
    })
    fp = load_frozen_points(path)
    with pytest.raises(ValueError, match="benchmark="):
        fp.to_query(_spec(zeta_dim=1, condition_dim=0, bench_id="b99_other"))


def test_family_match_accepted(tmp_path):
    """`benchmark` field can also match the family prefix (e.g. "e1")."""
    path = _write(tmp_path, {
        "benchmark": "e3",
        "points": [{"zeta": [0.0], "condition": [1.0]}],
        "ipopt": {},
    })
    fp = load_frozen_points(path)
    spec = _spec(zeta_dim=1, condition_dim=1, bench_id="e3/acopf_ieee30")
    q = fp.to_query(spec)
    assert q.zeta.shape == (1, 1)


def test_empty_points_allowed(tmp_path):
    """A JSON with an empty points list loads without error."""
    path = _write(tmp_path, {
        "benchmark": "e1",
        "frozen_at": "2026-04-30",
        "rationale": "stub",
        "points": [],
        "ipopt": {"max_iter": 500},
    })
    fp = load_frozen_points(path)
    assert len(fp) == 0
    assert fp.ipopt == {"max_iter": 500}


def test_inconsistent_point_lengths_rejected(tmp_path):
    path = _write(tmp_path, {
        "benchmark": "b00_test",
        "points": [
            {"zeta": [0.1, 0.2], "condition": [1.0]},
            {"zeta": [0.3], "condition": [2.0]},
        ],
        "ipopt": {},
    })
    with pytest.raises(ValueError, match="point 1 zeta"):
        load_frozen_points(path)
