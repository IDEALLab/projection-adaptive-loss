"""IPOPTSolver integration tests (skipped without cyipopt)."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

pytest.importorskip("cyipopt")

from pal.baselines.ipopt.solver import (  # noqa: E402
    IPOPTConfig,
    IPOPTSolver,
    _make_constraint_bounds,
)
from pal.baselines.nlp_adapter import NLPView  # noqa: E402
from pal.baselines.slsqp.solver import SLSQPConfig, SLSQPSolver  # noqa: E402
from pal.benchmarks.base import BenchmarkSpec, Query  # noqa: E402
from pal.constraints import Constraint  # noqa: E402


class _KnownOptimumBench:
    """Min sum(x^2) s.t. x[0] + x[1] = 1, x[0] >= 0.3. Optimum (0.5, 0.5)."""

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
        h = x[..., 0] + x[..., 1] - 1.0
        g = 0.3 - x[..., 0]
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
        return Query(zeta=zeta, conditions=torch.empty(n, 0))


class _SignedIneqBench:
    """Min sum(x^2) s.t. x[0] <= 0.1 (g = x[0] - 0.1 <= 0): optimum at zero."""

    def __init__(self) -> None:
        self.spec = BenchmarkSpec(
            id="mock_signed",
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
        g = x[..., 0] - 0.1
        zero = torch.zeros(B, dtype=x.dtype, device=x.device)
        margin = torch.full((B,), 1e-6, dtype=x.dtype, device=x.device)
        return obj, [Constraint(value=g, type="ineq", tol=zero, margin=margin, name="upper")]

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = n or 2
        return Query(zeta=torch.zeros(n, 1), conditions=torch.empty(n, 0))


class _HS071Bench:
    """Canonical IPOPT test problem HS071.

    min x[0]*x[3]*(x[0]+x[1]+x[2]) + x[2]
    s.t.  x[0]*x[1]*x[2]*x[3] - 25 >= 0              -> pal: 25 - prod(x) <= 0
          sum(x^2) == 40
          1 <= x[i] <= 5

    Known optimum: x* ~ [1.0, 4.743, 3.821, 1.379], f* ~ 17.014.
    """

    def __init__(self) -> None:
        self.spec = BenchmarkSpec(
            id="hs071",
            family="hs",
            variant="071",
            dim=4,
            n_eq=1,
            n_ineq=1,
            constraint_names=["sum_sq", "prod_ge_25"],
            constraint_types=["eq", "ineq"],
            output_bounds=(
                torch.ones(4),
                torch.full((4,), 5.0),
            ),
            condition_dim=0,
            zeta_dim=1,
            tolerance=1e-4,
            cost="cheap",
            recommended_device="cpu",
            n_eval_default=1,
        )

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        B = x.shape[0]
        obj = x[..., 0] * x[..., 3] * (x[..., 0] + x[..., 1] + x[..., 2]) + x[..., 2]
        h = (x ** 2).sum(dim=-1) - 40.0
        prod = x[..., 0] * x[..., 1] * x[..., 2] * x[..., 3]
        g = 25.0 - prod                       # pal ineq: <= 0 means prod >= 25
        tol = torch.full((B,), 1e-4, dtype=x.dtype, device=x.device)
        margin = torch.full((B,), 1e-6, dtype=x.dtype, device=x.device)
        zero = torch.zeros(B, dtype=x.dtype, device=x.device)
        return obj, [
            Constraint(value=h, type="eq", tol=tol, margin=zero, name="sum_sq"),
            Constraint(value=g, type="ineq", tol=zero, margin=margin, name="prod_ge_25"),
        ]

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = n or 1
        zeta = torch.zeros(n, 1)
        return Query(zeta=zeta, conditions=torch.empty(n, 0))


class _NullLogger:
    def log_config(self, **_): pass
    def log_step(self, step, **_): pass
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **_): pass
    def finish(self, status="ok", error=None): pass


def test_constraint_bounds_mixed_order() -> None:
    bench = _KnownOptimumBench()  # types=["eq", "ineq"]
    view = NLPView(bench, conditions=None)
    cl, cu = _make_constraint_bounds(view)
    # eq row: [0, 0]; ineq row: [-inf, 0]
    np.testing.assert_array_equal(cl, [0.0, -np.inf])
    np.testing.assert_array_equal(cu, [0.0, 0.0])


def test_hs071_reaches_known_optimum() -> None:
    bench = _HS071Bench()
    solver = IPOPTSolver(IPOPTConfig(seed=0, max_iter=300, print_level=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())

    x = res.final_x_on_eval.numpy()[0]
    expected = np.array([1.0, 4.743, 3.821, 1.379])
    np.testing.assert_allclose(x, expected, atol=1e-3)
    assert res.extras["feasible_fraction"] == 1.0


def test_known_optimum_reached_on_convex_qp() -> None:
    bench = _KnownOptimumBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())

    assert res.solver_name == "ipopt"
    assert res.final_x_on_eval.shape == (bench.spec.n_eval_default, bench.spec.dim)
    assert res.train_wall_time_s > 0.0

    x = res.final_x_on_eval.numpy()
    np.testing.assert_allclose(x, np.tile([0.5, 0.5], (x.shape[0], 1)), atol=1e-5)
    assert res.extras["feasible_fraction"] == 1.0


def test_signed_constraint_passes_through_unchanged() -> None:
    """pal `g(x) <= 0` feasible -> IPOPT `cl=-inf, cu=0`. No sign flip."""
    bench = _SignedIneqBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    x = res.final_x_on_eval.numpy()
    # The unconstrained optimum (zero) is feasible.
    np.testing.assert_allclose(x, np.zeros_like(x), atol=1e-5)
    assert res.extras["feasible_fraction"] == 1.0


def test_ipopt_slsqp_agree_on_convex_qp() -> None:
    bench = _KnownOptimumBench()
    ip = IPOPTSolver(IPOPTConfig(seed=0)).train(bench, 0, _NullLogger())
    sl = SLSQPSolver(SLSQPConfig(seed=0)).train(bench, 0, _NullLogger())
    np.testing.assert_allclose(
        ip.final_x_on_eval.numpy(), sl.final_x_on_eval.numpy(), atol=1e-4
    )


def test_extras_per_query_carries_all_restart_records() -> None:
    """Every multistart candidate is persisted JSON-safe in extras["per_query"][i]["restarts"]."""
    bench = _KnownOptimumBench(n_eval=2)
    solver = IPOPTSolver(IPOPTConfig(seed=0, multi_start=3))
    res = solver.train(bench, seed=0, logger=_NullLogger())

    per_query = res.extras["per_query"]
    assert len(per_query) == bench.spec.n_eval_default

    for i, diag in enumerate(per_query):
        restarts = diag["restarts"]
        assert len(restarts) == 3
        assert [r["restart_idx"] for r in restarts] == [0, 1, 2]
        for r in restarts:
            assert isinstance(r["x"], list) and len(r["x"]) == bench.spec.dim
            assert all(isinstance(v, float) for v in r["x"])
            assert r["obj"] is None or isinstance(r["obj"], float)
            assert r["max_violation"] is None or isinstance(r["max_violation"], float)
            assert isinstance(r["feasible"], bool)
            assert isinstance(r["status"], int)

        # solutions were cast to fp32, so allow fp32 round-trip tolerance.
        best_x = res.final_x_on_eval[i].numpy().astype(np.float64)
        candidate_xs = [np.asarray(r["x"]) for r in restarts]
        assert any(np.allclose(best_x, cx, atol=1e-5) for cx in candidate_xs)


def test_restart_shard_runs_only_assigned_slice() -> None:
    """--restart-shard r/R runs only restarts r, r+R, ... and keeps their restart_idx."""
    bench = _KnownOptimumBench(n_eval=1)
    # multi_start=8 with shard 1/4 -> restart_indices = [1, 5]
    cfg = IPOPTConfig(seed=0, multi_start=8, restart_shard=(1, 4))
    res = IPOPTSolver(cfg).train(bench, seed=0, logger=_NullLogger())

    assert res.n_restarts == 2  # local count, not the full multi_start=8
    assert res.extras["multi_start_total"] == 8
    assert res.extras["restart_shard"] == [1, 4]

    diag = res.extras["per_query"][0]
    assert [r["restart_idx"] for r in diag["restarts"]] == [1, 5]


def test_restart_shard_union_equals_unsharded_run() -> None:
    """The union of restart records from R shards equals an unsharded multi_start=N run."""
    bench = _KnownOptimumBench(n_eval=1)

    full = IPOPTSolver(IPOPTConfig(seed=0, multi_start=4)).train(
        bench, seed=0, logger=_NullLogger()
    )
    shards = [
        IPOPTSolver(IPOPTConfig(seed=0, multi_start=4, restart_shard=(r, 2))).train(
            bench, seed=0, logger=_NullLogger()
        )
        for r in range(2)
    ]

    full_records = full.extras["per_query"][0]["restarts"]
    union = []
    for s in shards:
        union.extend(s.extras["per_query"][0]["restarts"])
    union.sort(key=lambda r: r["restart_idx"])

    assert [r["restart_idx"] for r in full_records] == [0, 1, 2, 3]
    assert [r["restart_idx"] for r in union] == [0, 1, 2, 3]
    for fr, ur in zip(full_records, union, strict=True):
        np.testing.assert_allclose(fr["x"], ur["x"], atol=1e-10)
        if fr["obj"] is not None and ur["obj"] is not None:
            assert abs(fr["obj"] - ur["obj"]) < 1e-8


def test_shard_merge_post_metrics_match_unsharded_final_eval() -> None:
    """Single-shard aggregate_post_metrics matches an unsharded run within fp32 noise."""
    from pal.baselines.ipopt.shard_merge import aggregate_post_metrics

    bench = _KnownOptimumBench(n_eval=4)
    cfg = IPOPTConfig(seed=0, multi_start=1, restart_shard=(0, 1))
    res_sharded = IPOPTSolver(cfg).train(bench, seed=0, logger=_NullLogger())
    res_full = IPOPTSolver(IPOPTConfig(seed=0, multi_start=1)).train(
        bench, seed=0, logger=_NullLogger()
    )

    sharded_post = aggregate_post_metrics(res_sharded.extras["per_query"])
    full_post = aggregate_post_metrics(res_full.extras["per_query"])

    # Same RNG and NLP give byte-identical per-query records.
    assert sharded_post["obj_mean_post"] == pytest.approx(full_post["obj_mean_post"])
    assert sharded_post["viol_max_post"] == pytest.approx(full_post["viol_max_post"])
    assert sharded_post["feasibility_post"] == pytest.approx(full_post["feasibility_post"])


def test_per_query_restarts_is_json_serializable() -> None:
    """Strict JSON round-trip: no Infinity/NaN literals leak into final.json."""
    import json as _json

    bench = _KnownOptimumBench(n_eval=2)
    solver = IPOPTSolver(IPOPTConfig(seed=0, multi_start=2))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    # allow_nan=False rejects Infinity/NaN.
    _json.dumps(res.extras["per_query"], allow_nan=False)


def test_same_seed_produces_identical_solutions() -> None:
    bench_a = _KnownOptimumBench()
    bench_b = _KnownOptimumBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0, multi_start=3))
    res_a = solver.train(bench_a, seed=0, logger=_NullLogger())
    res_b = solver.train(bench_b, seed=0, logger=_NullLogger())
    np.testing.assert_array_equal(
        res_a.final_x_on_eval.numpy(), res_b.final_x_on_eval.numpy()
    )


def test_predict_returns_cached_on_matching_queries() -> None:
    bench = _KnownOptimumBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    q = bench.eval_queries(seed=0)
    out = solver.predict(bench, q, res)
    np.testing.assert_array_equal(
        out.raw.numpy(), res.final_x_on_eval.numpy()
    )
    assert out.projection is None


def test_predict_resolves_on_fresh_queries() -> None:
    bench = _KnownOptimumBench(n_eval=4)
    solver = IPOPTSolver(IPOPTConfig(seed=0))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    g = torch.Generator().manual_seed(42)
    q_new = Query(
        zeta=torch.randn(2, bench.spec.zeta_dim, generator=g),
        conditions=torch.empty(2, 0),
    )
    out = solver.predict(bench, q_new, res)
    assert out.raw.shape == (2, bench.spec.dim)
    np.testing.assert_allclose(out.raw.numpy(), np.tile([0.5, 0.5], (2, 1)), atol=1e-5)


def test_train_accepts_hparam_override() -> None:
    bench = _KnownOptimumBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger(), multi_start=2)
    assert res.n_restarts == 2


def test_train_rejects_unknown_hparam() -> None:
    bench = _KnownOptimumBench()
    solver = IPOPTSolver(IPOPTConfig(seed=0))
    with pytest.raises(TypeError, match="unknown hparam"):
        solver.train(bench, seed=0, logger=_NullLogger(), nonsense_flag=42)


def test_runs_on_real_rosenbrock_eq_without_error() -> None:
    from pal.benchmarks.synthetic.rosenbrock_eq import RosenbrockEq

    bench = RosenbrockEq()
    solver = IPOPTSolver(IPOPTConfig(seed=0, max_iter=200, multi_start=1))
    res = solver.train(bench, seed=0, logger=_NullLogger())
    assert res.final_x_on_eval.shape == (bench.spec.n_eval_default, bench.spec.dim)
    assert torch.isfinite(res.final_x_on_eval).all()
