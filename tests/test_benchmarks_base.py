"""BenchmarkSpec + Query unit tests."""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pal.artifacts.core import ArtifactRef
from pal.benchmarks.base import Benchmark, BenchmarkSpec, Query


def _minimal_spec(**overrides) -> BenchmarkSpec:
    defaults = dict(
        id="rosenbrock_eq",
        family="rosenbrock_eq",
        variant=None,
        dim=2,
        n_eq=1,
        n_ineq=0,
        constraint_names=["h1"],
        constraint_types=["eq"],
        output_bounds=(torch.tensor([-5.0, -5.0]), torch.tensor([5.0, 5.0])),
        condition_dim=0,
        zeta_dim=4,
        tolerance=1e-4,
        cost="cheap",
        recommended_device="cpu",
    )
    defaults.update(overrides)
    return BenchmarkSpec(**defaults)


def test_spec_to_json_dict_round_trips_through_json() -> None:
    import json

    spec = _minimal_spec(
        artifacts=[
            ArtifactRef(
                name="dummy",
                hf_filename="rosenbrock_eq/dummy.pt",
                hf_revision="abc123",
                local_subdir="rosenbrock_eq",
            )
        ],
        notes="tests",
    )
    d = spec.to_json_dict()
    serialized = json.dumps(d)
    roundtripped = json.loads(serialized)

    assert roundtripped["id"] == "rosenbrock_eq"
    assert roundtripped["output_bounds_lo"] == [-5.0, -5.0]
    assert roundtripped["output_bounds_hi"] == [5.0, 5.0]
    assert roundtripped["artifacts"] == [
        {
            "name": "dummy",
            "hf_filename": "rosenbrock_eq/dummy.pt",
            "hf_revision": "abc123",
            "local_subdir": "rosenbrock_eq",
            "location": "hf",
            "repo_path": "",
        }
    ]
    assert roundtripped["precision"] == "fp32"


def test_spec_rejects_output_bounds_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="output_bounds"):
        _minimal_spec(output_bounds=(torch.zeros(3), torch.ones(3)))


def test_spec_rejects_constraint_name_count_mismatch() -> None:
    with pytest.raises(ValueError, match="constraint_names"):
        _minimal_spec(
            n_eq=1, n_ineq=1,
            constraint_names=["only_one"],
            constraint_types=["eq", "ineq"],
        )


def test_spec_rejects_constraint_types_count_mismatch() -> None:
    with pytest.raises(ValueError, match="constraint_types length"):
        _minimal_spec(
            n_eq=1, n_ineq=1,
            constraint_names=["a", "b"],
            constraint_types=["eq"],
        )


def test_spec_rejects_constraint_types_eq_count_disagreement() -> None:
    with pytest.raises(ValueError, match="eq-count"):
        _minimal_spec(
            n_eq=1, n_ineq=1,
            constraint_names=["a", "b"],
            constraint_types=["ineq", "ineq"],
        )


def test_spec_is_frozen() -> None:
    spec = _minimal_spec()
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.tolerance = 1e-5  # type: ignore[misc]


def test_query_length_and_device_move() -> None:
    q = Query(zeta=torch.randn(8, 4), conditions=torch.randn(8, 0))
    assert len(q) == 8
    assert q.conditions.shape == (8, 0)
    moved = q.to("cpu")
    assert moved.zeta.device.type == "cpu"


def test_query_rejects_batch_mismatch() -> None:
    with pytest.raises(ValueError, match="leading dim"):
        Query(zeta=torch.randn(4, 2), conditions=torch.randn(5, 0))


def test_benchmark_protocol_is_duck_typed() -> None:
    class _DummyBench:
        def __init__(self) -> None:
            self.spec = _minimal_spec()

        def objective(self, x, conditions=None):
            return (x**2).sum(-1)

        def constraints(self, x, conditions=None):
            return x[..., :1]

        def sample_queries(self, n, split, seed):
            g = torch.Generator().manual_seed(seed)
            return Query(
                zeta=torch.randn(n, self.spec.zeta_dim, generator=g),
                conditions=torch.zeros(n, self.spec.condition_dim),
            )

        def eval_queries(self, seed):
            return self.sample_queries(16, "eval", seed)

        def check_env(self) -> None:
            return None

        def visualize_train(self, x, conditions=None):
            return None

        def visualize_final(self, x, conditions=None):
            return None

    bench = _DummyBench()
    assert isinstance(bench, Benchmark)
    q = bench.eval_queries(0)
    assert len(q) == 16
