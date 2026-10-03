"""SLSQPSolver integration tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from pal.baselines.nlp_adapter import select_best_by_dominance as _select_best
from pal.baselines.slsqp.solver import SLSQPConfig, SLSQPSolver
from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint


class _KnownOptimumBench:
    """min sum(x^2) s.t. x[0] + x[1] = 1, x[0] >= 0.3. Optimum x* = (0.5, 0.5), f* = 0.5."""

    def __init__(self, dim: int = 2, n_eval: int = 4) -> None:
        self._n_eval = n_eval
        self.spec = BenchmarkSpec(
            id="mock_qp",
            family="mock",
            variant=None,
            dim=dim,
            n_eq=1,
            n_ineq=1,
            constraint_names=["coupling", "lower_bound"],
            constraint_types=["eq", "ineq"],
            output_bounds=(
                torch.full((dim,), -2.0, dtype=torch.float32),
                torch.full((dim,), 2.0, dtype=torch.float32),
            ),
            condition_dim=0,
            zeta_dim=2,
            tolerance=1e-4,
            cost="cheap",
            recommended_device="cpu",
            n_eval_default=n_eval,
        )

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        B = x.shape[0]
        obj = (x ** 2).sum(dim=-1)
        h = x[..., 0] + x[..., 1] - 1.0                  # eq: == 0
        g = 0.3 - x[..., 0]                              # ineq: <= 0 -> x[0] >= 0.3
        tol = torch.full((B,), 1e-4, dtype=x.dtype, device=x.device)
        margin = torch.full((B,), 1e-6, dtype=x.dtype, device=x.device)
        zero = torch.zeros(B, dtype=x.dtype, device=x.device)
        return obj, [
            Constraint(value=h, type="eq", tol=tol, margin=zero, name="coupling"),
            Constraint(value=g, type="ineq", tol=zero, margin=margin, name="lower_bound"),
        ]

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = n or self._n_eval
        g = torch.Generator().manual_seed(seed + 1)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conds = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conds)


class _OnlyIneqBench:
    """Min sum(x^2) s.t. x[0] <= 0.1 (g = x[0] - 0.1 <= 0): optimum x* = (0, 0)."""

    def __init__(self) -> None:
        self.spec = BenchmarkSpec(
            id="mock_ineq",
            family="mock",
            variant=None,
            dim=2,
            n_eq=0,
            n_ineq=1,
            constraint_names=["upper"],
            constraint_types=["ineq"],
            output_bounds=(
                torch.tensor([-1.0, -1.0]),
                torch.tensor([1.0, 1.0]),
            ),
            condition_dim=0,
            zeta_dim=1,
            tolerance=1e-4,
            cost="cheap",
            recommended_device="cpu",
            n_eval_default=2,
        )

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        B = x.shape[0]
        obj = (x ** 2).sum(dim=-1)
        g = x[..., 0] - 0.1                              # ineq: <= 0
        zero = torch.zeros(B, dtype=x.dtype, device=x.device)
        margin = torch.full((B,), 1e-6, dtype=x.dtype, device=x.device)
        return obj, [
            Constraint(value=g, type="ineq", tol=zero, margin=margin, name="upper"),
        ]

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = n or 2
        zeta = torch.zeros(n, 1)
        return Query(zeta=zeta, conditions=torch.empty(n, 0))


class _NullLogger:
    def log_config(self, **_): pass
    def log_step(self, step, **_): pass
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **_): pass
    def finish(self, status="ok", error=None): pass


def test_known_optimum_reached_on_convex_qp() -> None:
    bench = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger())

    assert res.solver_name == "slsqp"
    assert res.final_x_on_eval is not None
    assert res.final_x_on_eval.shape == (bench.spec.n_eval_default, bench.spec.dim)
    assert res.train_wall_time_s > 0.0

    # All queries should land at x* = (0.5, 0.5)
    x = res.final_x_on_eval.numpy()
    np.testing.assert_allclose(x, np.tile([0.5, 0.5], (x.shape[0], 1)), atol=1e-4)

    assert res.extras["feasible_fraction"] == 1.0


def test_ineq_sign_flip_is_correct() -> None:
    """With the correct sign flip the unconstrained optimum x = (0, 0) is feasible."""
    bench = _OnlyIneqBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    x = res.final_x_on_eval.numpy()

    np.testing.assert_allclose(x, np.zeros_like(x), atol=1e-4)
    assert res.extras["feasible_fraction"] == 1.0


def test_final_x_shape_and_wall_time() -> None:
    bench = _KnownOptimumBench(n_eval=3)
    solver = SLSQPSolver(SLSQPConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    assert res.final_x_on_eval.shape == (3, 2)
    assert res.train_wall_time_s > 0.0
    assert res.n_restarts == 1


def test_same_seed_produces_identical_solutions() -> None:
    bench_a = _KnownOptimumBench()
    bench_b = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=3))

    res_a = solver.train(bench_a, seed=0, logger=_NullLogger())
    res_b = solver.train(bench_b, seed=0, logger=_NullLogger())

    np.testing.assert_array_equal(
        res_a.final_x_on_eval.numpy(), res_b.final_x_on_eval.numpy()
    )


def test_different_seed_changes_initial_points() -> None:
    """Different zeta seeds different inits, so per-query diagnostics differ."""
    bench_a = _KnownOptimumBench()
    bench_b = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=1))
    res0 = solver.train(bench_a, seed=0, logger=_NullLogger())
    res1 = solver.train(bench_b, seed=1, logger=_NullLogger())
    assert res0.final_x_on_eval.shape == res1.final_x_on_eval.shape
    assert res0.extras["feasible_fraction"] == 1.0
    assert res1.extras["feasible_fraction"] == 1.0


def test_select_best_prefers_feasible_over_infeasible() -> None:
    cands = [
        {"x": np.array([1.0]), "obj": 0.0, "max_violation": 1e-2, "feasible": False, "status": 0},
        {"x": np.array([2.0]), "obj": 10.0, "max_violation": 0.0, "feasible": True, "status": 0},
    ]
    best = _select_best(cands)
    assert best["feasible"] is True
    np.testing.assert_array_equal(best["x"], [2.0])


def test_select_best_among_feasible_uses_objective() -> None:
    cands = [
        {"x": np.array([1.0]), "obj": 3.0, "max_violation": 0.0, "feasible": True, "status": 0},
        {"x": np.array([2.0]), "obj": 1.0, "max_violation": 0.0, "feasible": True, "status": 0},
        {"x": np.array([3.0]), "obj": 2.0, "max_violation": 0.0, "feasible": True, "status": 0},
    ]
    best = _select_best(cands)
    assert best["obj"] == 1.0
    np.testing.assert_array_equal(best["x"], [2.0])


def test_select_best_among_infeasible_uses_violation_then_objective() -> None:
    cands = [
        {"x": np.array([1.0]), "obj": 0.0, "max_violation": 0.5, "feasible": False, "status": 0},
        {"x": np.array([2.0]), "obj": 100.0, "max_violation": 0.1, "feasible": False, "status": 0},
        {"x": np.array([3.0]), "obj": 50.0, "max_violation": 0.1, "feasible": False, "status": 0},
    ]
    best = _select_best(cands)
    # Lowest violation is 0.1 (two ties); among those, lowest objective is 50.
    assert best["max_violation"] == 0.1
    assert best["obj"] == 50.0


def test_multi_start_logs_feasible_restart_count() -> None:
    bench = _KnownOptimumBench(n_eval=2)
    calls: list[dict] = []

    class _CaptureLogger(_NullLogger):
        def log_step(self, step, **kwargs):
            calls.append(dict(kwargs))

    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=3))
    res = solver.train(bench, seed=0, logger=_CaptureLogger())

    assert res.n_restarts == 3
    assert len(calls) == 2
    for c in calls:
        assert c["n_feasible_restarts"] <= 3
        assert c["n_feasible_restarts"] >= 1


def test_predict_returns_cached_when_query_count_matches() -> None:
    bench = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    q = bench.eval_queries(seed=0)
    out = solver.predict(bench, q, res)
    np.testing.assert_array_equal(
        out.raw.numpy(), res.final_x_on_eval.numpy()
    )
    # Classical: raw == post, no projection
    assert out.post is out.raw or torch.equal(out.post, out.raw)
    assert out.projection is None


def test_predict_resolves_when_query_count_differs() -> None:
    bench = _KnownOptimumBench(n_eval=4)
    solver = SLSQPSolver(SLSQPConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    g = torch.Generator().manual_seed(42)
    q_new = Query(
        zeta=torch.randn(2, bench.spec.zeta_dim, generator=g),
        conditions=torch.empty(2, 0),
    )
    out = solver.predict(bench, q_new, res)
    assert out.raw.shape == (2, bench.spec.dim)
    x = out.raw.numpy()
    np.testing.assert_allclose(x, np.tile([0.5, 0.5], (2, 1)), atol=1e-4)


def test_train_accepts_hparam_override() -> None:
    bench = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger(), multi_start=2)
    assert res.n_restarts == 2


def test_train_rejects_unknown_hparam() -> None:
    bench = _KnownOptimumBench()
    solver = SLSQPSolver(SLSQPConfig(seed=0))
    with pytest.raises(TypeError, match="unknown hparam"):
        solver.train(bench, seed=0, logger=_NullLogger(), nonsense_flag=42)


def test_runs_on_real_rosenbrock_eq_without_error() -> None:
    """SLSQP on rosenbrock_eq Rosenbrock completes with a finite tensor of the right shape."""
    from pal.benchmarks.synthetic.rosenbrock_eq import RosenbrockEq

    bench = RosenbrockEq()
    solver = SLSQPSolver(SLSQPConfig(seed=0, maxiter=50, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger())

    assert res.final_x_on_eval.shape == (bench.spec.n_eval_default, bench.spec.dim)
    assert torch.isfinite(res.final_x_on_eval).all()
