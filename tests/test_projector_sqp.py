"""Unit tests for the `"sqp"` elastic-mode repair step in `Projector`.

Requires `qpsolvers` + `osqp` + `clarabel`.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pal.projection import projector as projector_module
from pal.projection.projector import Projector

pytest.importorskip("qpsolvers", reason="pal_sqp repair step needs qpsolvers + osqp")
pytest.importorskip("clarabel", reason="pal_sqp solver chain needs clarabel")


def _all_active(n_batch: int, K: int) -> torch.Tensor:
    return torch.ones(n_batch, K, dtype=torch.bool)


def _post_linearized(
    c: torch.Tensor, J: torch.Tensor, y: torch.Tensor, y_new: torch.Tensor
) -> torch.Tensor:
    """`c + J*(y_new - y)`, the row values the step *intends* to reach."""
    return c + torch.bmm(J, (y_new - y).unsqueeze(-1)).squeeze(-1)


class TestEqualityOnlyReducesToMinNorm:
    """No inequalities + consistent equalities => `sqp` == `lm_k(lambda->0)`."""

    def test_matches_undamped_lm_k(self) -> None:
        torch.manual_seed(0)
        n_batch, K, D = 4, 3, 6
        J = torch.randn(n_batch, K, D)
        c = torch.randn(n_batch, K) * 0.3
        y = torch.randn(n_batch, D)
        active = _all_active(n_batch, K)

        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        lm_undamped = Projector(
            K, ["eq"] * K, prescale=False, method="lm_k",
            delta=0.0, lambda_min=0.0,
        )
        y_sqp, info = sqp._project_step_inner(y, c, J, active)
        y_lm, _ = lm_undamped._project_step_inner(y, c, J, active)

        assert torch.allclose(y_sqp, y_lm, atol=1e-5)
        # Every row tight, no elastic rows, and the QP agrees with the re-solve.
        assert bool(info["tight_mask"].all())
        assert info["sqp_qp_failures"] == 0
        assert info["sqp_resolve_max_gap"] < 1e-5

    def test_drives_equalities_to_zero(self) -> None:
        """Sanity anchor: `Delta = -J^T (JJ^T)^-1c` zeroes the linearized rows."""
        torch.manual_seed(1)
        n_batch, K, D = 2, 2, 5
        J = torch.randn(n_batch, K, D)
        c = torch.randn(n_batch, K)
        y = torch.zeros(n_batch, D)
        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        y_new, _ = sqp._project_step_inner(y, c, J, _all_active(n_batch, K))
        assert _post_linearized(c, J, y, y_new).abs().max() < 1e-5


class TestInequalityNoReviolation:
    """`lm_k` drops the satisfied row and re-violates it; `sqp` keeps both: Delta = (-1, +0.9)."""

    J = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
    c = torch.tensor([[1.0, -0.1]])
    y = torch.tensor([[2.0, 0.0]])

    def test_lm_k_reviolates_the_satisfied_row(self) -> None:
        lm = Projector(
            2, ["ineq", "ineq"], prescale=False, method="lm_k",
            delta=0.0, lambda_min=1e-10,
        )
        active = lm._build_active_mask(self.c)
        assert active.tolist() == [[True, False]]  # only the violated row
        y_new, _ = lm._project_step_inner(self.y, self.c, self.J, active)
        post = _post_linearized(self.c, self.J, self.y, y_new)
        assert post[0, 0].abs() < 1e-5          # row 1 repaired
        assert post[0, 1] > 0.5                 # row 2 newly violated

    def test_sqp_holds_both_rows(self) -> None:
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(
            self.y, self.c, self.J, sqp._build_active_mask(self.c)
        )
        post = _post_linearized(self.c, self.J, self.y, y_new)
        assert post.max() < 1e-5                # neither row violated
        assert info["tight_mask"].tolist() == [[True, True]]
        assert torch.allclose(
            y_new, torch.tensor([[1.0, 0.9]]), atol=1e-5
        )


class TestInconsistentLinearization:
    """5 equalities in 3 variables: elastic mode must return a finite step."""

    def test_finite_elastic_step_and_exact_tight_rows(self) -> None:
        torch.manual_seed(0)
        K, D = 5, 3
        J = torch.randn(1, K, D)
        c = torch.randn(1, K)
        y = torch.zeros(1, D)
        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(y, c, J, _all_active(1, K))

        assert torch.isfinite(y_new).all()
        assert info["sqp_qp_failures"] == 0
        tight = info["tight_mask"][0]
        # At most D rows can be held (rank), and at least one must go elastic.
        assert 0 < int(tight.sum()) <= D
        assert not bool(tight.all())
        post = _post_linearized(c, J, y, y_new)[0]
        assert post[tight].abs().max() < 1e-5   # tight rows exactly repaired
        assert post[~tight].abs().max() > 1e-3  # elastic rows keep a residual

    def test_project_loop_does_not_raise(self) -> None:
        """The inference loop must survive an inconsistent linearization."""
        from types import SimpleNamespace

        K, D = 5, 3
        torch.manual_seed(3)
        A = torch.randn(K, D)
        b = torch.randn(K)

        def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
            value = yy @ A.T - b
            clist = [
                SimpleNamespace(value=value[..., k], type="eq", name=f"eq_{k}")
                for k in range(K)
            ]
            return value.pow(2).sum(dim=-1), clist

        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        y_final, info = sqp.project(
            torch.zeros(2, D), constraint_fn, conditions=None, max_iters=5,
        )
        assert torch.isfinite(y_final).all()
        assert info["iters"] == 5           # never converges; that is the point
        assert torch.isfinite(torch.tensor(info["max_residual"]))


class TestDisplacementProxyBackward:
    """`step()`'s displacement proxy must carry finite grads on the sqp path."""

    @staticmethod
    def _run(method: str, **kwargs) -> tuple[float, torch.Tensor, dict]:
        torch.manual_seed(7)
        D, K, n_batch = 4, 2, 6
        layer = torch.nn.Linear(3, D)
        z = torch.randn(n_batch, 3)
        A = torch.randn(K, D)
        offset = torch.tensor([0.5, -0.2])

        def values_fn(yy, conditions):  # noqa: ARG001 - unused
            return yy @ A.T - offset

        projector = Projector(
            K, ["eq", "ineq"], prescale=False, method=method, **kwargs
        )
        y_hat = layer(z)
        c_pre = values_fn(y_hat, None)
        _, info = projector.step(y_hat, c_pre, values_fn, None)
        loss = info["displacement_live"].pow(2).sum()
        loss.backward()
        assert layer.weight.grad is not None
        return float(loss.item()), layer.weight.grad.clone(), info

    def test_sqp_backward_is_finite(self) -> None:
        loss, grad, info = self._run("sqp")
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0.0
        assert torch.isfinite(torch.tensor(loss))
        assert info["displacement_live"].dtype == torch.float32
        # At the linearization point the proxy must reproduce the step actually taken.
        assert torch.allclose(
            info["displacement_live"].detach(), info["displacement"], atol=1e-6
        )

    def test_lm_k_backward_is_finite(self) -> None:
        """Structure-parallel control: same assertions on the shipped path."""
        loss, grad, info = self._run("lm_k", delta=1e-3)
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0.0
        assert torch.isfinite(torch.tensor(loss))
        assert torch.allclose(
            info["displacement_live"].detach(), info["displacement"], atol=1e-6
        )

    def test_jacobian_stays_detached(self) -> None:
        """No second-order terms reach the parameters through J."""
        _, _, info = self._run("sqp")
        assert not info["J"].requires_grad


def _osqp_status_solution(status: str, x: torch.Tensor | None, found: bool):
    """A stand-in for `qpsolvers`' OSQP `Solution` with a chosen status."""
    return SimpleNamespace(
        found=found,
        x=None if x is None else x.numpy(),
        extras={"info": SimpleNamespace(status=status)},
    )


def _patch_osqp(monkeypatch, make_solution) -> dict:
    """Replace OSQP's result with `make_solution(problem)`; return a backend call log."""
    import qpsolvers

    real = qpsolvers.solve_problem
    calls: dict = {"osqp": 0, "clarabel": 0}

    def fake(problem, solver, *args, **kwargs):
        calls[solver] = calls.get(solver, 0) + 1
        if solver == "osqp":
            return make_solution(problem)
        return real(problem, solver, *args, **kwargs)

    monkeypatch.setattr(qpsolvers, "solve_problem", fake)
    return calls


def _forbid_gauss_newton(monkeypatch) -> None:
    """Make the `lm_k` / `eigh` inner solves explode if the sqp branch reaches them."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test guard
        raise AssertionError(
            "sqp branch fell back to a Gauss-Newton inner solve, the v1.1 "
            "chain must never report lm_k's step under the SQP label"
        )

    monkeypatch.setattr(Projector, "_project_step_inner_lm_k", _boom)
    monkeypatch.setattr(Projector, "_project_step_inner_eigh", _boom)


class TestSolverChain:
    """OSQP -> Clarabel -> transparent failure (v1.1 locked decision 1)."""

    # A well-posed 2-inequality toy: both rows tight, unique step (-1, +0.9).
    J = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
    c = torch.tensor([[1.0, -0.1]])
    y = torch.tensor([[2.0, 0.0]])

    def _step(self, projector: Projector):
        return projector._project_step_inner(
            self.y, self.c, self.J, projector._build_active_mask(self.c)
        )

    def test_reference_step_without_patching(self) -> None:
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = self._step(sqp)
        assert torch.allclose(y_new, torch.tensor([[1.0, 0.9]]), atol=1e-5)
        assert info["sqp_clarabel_rescues"] == 0
        assert info["sqp_qp_failures"] == 0

    def test_osqp_failure_is_rescued_by_clarabel(self, monkeypatch) -> None:
        calls = _patch_osqp(
            monkeypatch,
            lambda problem: _osqp_status_solution(
                "maximum iterations reached", None, found=False
            ),
        )
        _forbid_gauss_newton(monkeypatch)
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = self._step(sqp)

        assert calls["osqp"] == 1 and calls["clarabel"] == 1
        assert info["sqp_clarabel_rescues"] == 1
        assert info["sqp_qp_failures"] == 0
        assert sqp.sqp_clarabel_rescues_total == 1
        assert sqp.sqp_qp_failures_total == 0
        assert torch.allclose(y_new, torch.tensor([[1.0, 0.9]]), atol=1e-5)
        assert info["tight_mask"].tolist() == [[True, True]]
        assert info["sqp_resolve_max_gap"] < 1e-8

    def test_solved_inaccurate_also_triggers_clarabel(self, monkeypatch) -> None:
        """`solved inaccurate` returns an iterate; it must still be rejected."""
        bogus = torch.tensor([-5.0, 7.0, 3.0, 3.0])  # [Delta (2), t (2)]
        calls = _patch_osqp(
            monkeypatch,
            lambda problem: _osqp_status_solution(
                "solved inaccurate", bogus, found=True
            ),
        )
        _forbid_gauss_newton(monkeypatch)
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = self._step(sqp)

        assert calls["clarabel"] == 1
        assert info["sqp_clarabel_rescues"] == 1
        assert info["sqp_qp_failures"] == 0
        assert torch.allclose(y_new, torch.tensor([[1.0, 0.9]]), atol=1e-5)

    def test_both_backends_failing_is_a_transparent_failure(
        self, monkeypatch
    ) -> None:
        import qpsolvers

        calls: dict = {"osqp": 0, "clarabel": 0}

        def fake(problem, solver, *args, **kwargs):
            calls[solver] = calls.get(solver, 0) + 1
            if solver == "osqp":
                return _osqp_status_solution(
                    "maximum iterations reached", None, found=False
                )
            return SimpleNamespace(
                found=False, x=None, extras={"status": "SolverStatus.MaxIterations"}
            )

        monkeypatch.setattr(qpsolvers, "solve_problem", fake)
        _forbid_gauss_newton(monkeypatch)
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = self._step(sqp)

        assert calls["osqp"] == 1 and calls["clarabel"] == 1
        assert info["sqp_qp_failures"] == 1
        assert info["sqp_clarabel_rescues"] == 0
        assert sqp.sqp_qp_failures_total == 1
        # Delta = 0 exactly: y_tilde = y_hat, empty tight set, no elastic rows.
        assert torch.equal(y_new, self.y)
        assert not bool(info["tight_mask"].any())
        assert torch.count_nonzero(info["disp_elastic"]) == 0

    def test_transparent_failure_zeroes_the_proxy_contribution(
        self, monkeypatch
    ) -> None:
        """A failed sample must contribute an exact zero to the proxy + grads."""
        import qpsolvers

        def fake(problem, solver, *args, **kwargs):
            if solver == "osqp":
                return _osqp_status_solution("unsolved", None, found=False)
            return SimpleNamespace(
                found=False, x=None, extras={"status": "SolverStatus.NumericalError"}
            )

        monkeypatch.setattr(qpsolvers, "solve_problem", fake)
        _forbid_gauss_newton(monkeypatch)

        torch.manual_seed(7)
        D, K, n_batch = 4, 2, 3
        layer = torch.nn.Linear(3, D)
        z = torch.randn(n_batch, 3)
        A = torch.randn(K, D)

        def values_fn(yy, conditions):  # noqa: ARG001 - unused
            return yy @ A.T - torch.tensor([0.5, -0.2])

        sqp = Projector(K, ["eq", "ineq"], prescale=False, method="sqp")
        y_hat = layer(z)
        c_pre = values_fn(y_hat, None)
        y_tilde, info = sqp.step(y_hat, c_pre, values_fn, None)

        assert info["sqp_qp_failures"] == n_batch
        assert torch.equal(y_tilde, y_hat.detach())
        assert torch.count_nonzero(info["displacement"]) == 0
        assert torch.count_nonzero(info["displacement_live"]) == 0
        info["displacement_live"].pow(2).sum().backward()
        assert layer.weight.grad is not None
        assert float(layer.weight.grad.abs().max()) == 0.0

    def test_inference_loop_counts_failures(self, monkeypatch) -> None:
        """`project()` reports the same counters with the same semantics."""
        import qpsolvers

        def fake(problem, solver, *args, **kwargs):
            if solver == "osqp":
                return _osqp_status_solution("unsolved", None, found=False)
            return SimpleNamespace(found=False, x=None, extras={"status": "MaxIterations"})

        monkeypatch.setattr(qpsolvers, "solve_problem", fake)
        _forbid_gauss_newton(monkeypatch)

        K, D = 2, 3
        A = torch.randn(K, D)

        def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
            value = yy @ A.T - torch.ones(K)
            clist = [
                SimpleNamespace(value=value[..., k], type="eq", name=f"eq_{k}")
                for k in range(K)
            ]
            return value.pow(2).sum(dim=-1), clist

        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        y0 = torch.zeros(2, D)
        y_final, info = sqp.project(y0, constraint_fn, None, max_iters=3)

        assert torch.equal(y_final, y0)             # nothing moved
        assert info["sqp_qp_failures"] == 3 * 2     # iters x batch
        assert info["sqp_clarabel_rescues"] == 0
        assert sqp.sqp_qp_failures_total == 6


class TestRankDeficientResolve:
    """v1.1 locked decision 3: min-norm rank-revealing re-solve on tight rows."""

    def test_redundant_tight_rows_reproduce_the_qp_step(self) -> None:
        """s6-style: exact duplicates + a scaled copy => singular `J_T J_T^T`."""
        J = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]]
        )
        c = torch.tensor([[0.5, 0.5, 0.3, 1.0]])
        y = torch.zeros(1, 3)
        sqp = Projector(4, ["ineq"] * 4, prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(
            y, c, J, sqp._build_active_mask(c)
        )

        assert bool(info["tight_mask"].all())       # the QP holds all four
        # The Gram matrix is rank 2 of 4; the pseudo-inverse must not drift.
        gram = torch.bmm(info["B_mat"], info["B_mat"].transpose(1, 2))
        assert int(torch.linalg.matrix_rank(gram)[0]) == 2
        assert info["sqp_resolve_max_gap"] < 1e-10
        assert _post_linearized(c, J, y, y_new).abs().max() < 1e-10
        assert torch.allclose(y_new, torch.tensor([[-0.5, -0.3, 0.0]]), atol=1e-9)

    def test_near_redundant_batch_stays_accurate(self) -> None:
        """The s6 generator perturbs its copies by 1e-4; also check that."""
        torch.manual_seed(0)
        D, n_real, n_red = 8, 4, 6
        a = torch.randn(n_real, D)
        a = a / a.norm(dim=-1, keepdim=True)
        parent = torch.randint(0, n_real, (n_red,))
        for eps in (0.0, 1e-4):
            a_red = a[parent] + eps * torch.randn(n_red, D)
            a_red = a_red / a_red.norm(dim=-1, keepdim=True)
            K = n_real + n_red
            J = torch.cat([a, a_red], dim=0).unsqueeze(0)
            c = torch.full((1, K), 0.3)
            y = torch.zeros(1, D)
            sqp = Projector(K, ["ineq"] * K, prescale=False, method="sqp")
            _, info = sqp._project_step_inner(y, c, J, sqp._build_active_mask(c))
            assert info["sqp_qp_failures"] == 0
            assert info["sqp_resolve_max_gap"] < 1e-10, f"eps={eps}"

    def test_jitter_knob_is_gone(self) -> None:
        """The `sqp_jitter` kwarg was removed with the jittered Cholesky."""
        with pytest.raises(TypeError):
            Projector(1, ["eq"], method="sqp", sqp_jitter=1e-8)
        assert not hasattr(Projector(1, ["eq"], method="sqp"), "sqp_jitter")


def _resolve_with_partition(
    J: torch.Tensor, c: torch.Tensor, tight: torch.Tensor, mu_elastic: torch.Tensor
) -> torch.Tensor:
    """Second-stage re-solve for an arbitrary partition, as `dy = -Delta`."""
    mask = tight.to(torch.float64)
    B_T = J * mask.unsqueeze(-1)
    disp = torch.bmm(J.transpose(1, 2), mu_elastic.unsqueeze(-1)).squeeze(-1)
    offset = torch.bmm(B_T, disp.unsqueeze(-1)).squeeze(-1)
    gram_pinv = torch.linalg.pinv(
        torch.bmm(B_T, B_T.transpose(1, 2)), rtol=1e-10, hermitian=True
    )
    rhs = (c * mask - offset) * mask
    z = torch.bmm(gram_pinv, rhs.unsqueeze(-1)).squeeze(-1)
    return torch.bmm(B_T.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1) + disp


class TestDualPartitionFixesBarelyInactiveRow:
    """A barely-inactive row inside the residual tight test must not perturb the step."""

    EPS = 5e-3
    RESIDUAL = -4e-7

    @staticmethod
    def _rows(eps: float) -> np.ndarray:
        a0 = np.array([1.0, 0.0, 0.0])
        a1 = np.array([1.0, eps, 0.0])
        a2 = np.array([1.0, -0.6 * eps, 0.4 * eps])
        return np.stack([a0, a1 / np.linalg.norm(a1), a2 / np.linalg.norm(a2)])

    def _problem(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rows = self._rows(self.EPS)
        y = torch.zeros(1, 3, dtype=torch.float64)
        # Row 2 is inactive at the pair's step, so adding it leaves the QP step unchanged.
        pair = Projector(2, ["ineq"] * 2, prescale=False, method="sqp")
        J_pair = torch.from_numpy(rows[:2])[None]
        c_pair = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
        y_pair, _ = pair._project_step_inner(
            y, c_pair, J_pair, pair._build_active_mask(c_pair)
        )
        delta = (y_pair - y)[0].numpy()
        c_extra = self.RESIDUAL - rows[2] @ delta
        J = torch.from_numpy(rows)[None]
        c = torch.tensor([[0.5, 0.5, c_extra]], dtype=torch.float64)
        return y, c, J

    def test_barely_inactive_row_is_classified_inactive(self) -> None:
        y, c, J = self._problem()
        sqp = Projector(3, ["ineq"] * 3, prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(y, c, J, sqp._build_active_mask(c))

        post = _post_linearized(c, J, y, y_new)[0]
        assert post[2] == pytest.approx(self.RESIDUAL, rel=1e-6)
        assert abs(float(post[2])) < 1e-6 * (1.0 + abs(float(c[0, 2])))
        # Dual partition: row 2 out, the near-redundant pair in.
        assert info["tight_mask"].tolist() == [[True, True, False]]
        assert info["sqp_qp_failures"] == 0
        assert info["sqp_partition_mismatches"] == 0
        assert info["sqp_resolve_max_gap"] < 1e-10

    def test_v11_residual_partition_would_miss_the_step(self) -> None:
        """Same QP, v1.1's classification: row 2 enters T and the step drifts."""
        y, c, J = self._problem()
        sqp = Projector(3, ["ineq"] * 3, prescale=False, method="sqp")
        y_new, _ = sqp._project_step_inner(y, c, J, sqp._build_active_mask(c))
        dy_qp = (y - y_new)[0]

        post = _post_linearized(c, J, y, y_new)
        tight_v11 = post.abs() <= sqp.sqp_tight_tol * (1.0 + c.abs())
        assert tight_v11.tolist() == [[True, True, True]]   # the defect
        dy_v11 = _resolve_with_partition(J, c, tight_v11, torch.zeros_like(c))
        assert float((dy_v11[0] - dy_qp).abs().max()) > 1e-5

    def test_weakly_active_rows_are_inactive_by_design(self) -> None:
        """mu ~ 0 with residual ~ 0 => inactive; the step is unchanged by the row."""
        y = torch.zeros(1, 3, dtype=torch.float64)
        rows = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # Row 2 exactly on its boundary but with nothing to repair: c_2 = 0.
        c = torch.tensor([[0.4, 0.3, 0.0]], dtype=torch.float64)
        J = torch.from_numpy(rows)[None]
        sqp = Projector(3, ["ineq"] * 3, prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(y, c, J, sqp._build_active_mask(c))

        assert info["tight_mask"].tolist() == [[True, True, False]]
        assert info["sqp_resolve_max_gap"] < 1e-10
        assert torch.allclose(
            y_new, torch.tensor([[-0.4, -0.3, 0.0]], dtype=torch.float64), atol=1e-12
        )


class TestStepGuard:
    """v1.2 locked decision 2: disagreement with the QP's Delta falls back to it."""

    J = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
    c = torch.tensor([[1.0, -0.1]])
    y = torch.tensor([[2.0, 0.0]])

    @staticmethod
    def _break_partition(monkeypatch) -> None:
        """Keep the QP solve, then throw the partition away (empty tight set)."""
        real = Projector._sqp_elastic_partition

        def wrong(self, J_np, c_np):
            tight, mu_el, delta_qp, solved, nf, nr = real(self, J_np, c_np)
            return (
                np.zeros_like(tight),
                np.zeros_like(mu_el),
                delta_qp,
                solved,
                nf,
                nr,
            )

        monkeypatch.setattr(Projector, "_sqp_elastic_partition", wrong)

    def test_guard_substitutes_the_qp_step(self, monkeypatch) -> None:
        self._break_partition(monkeypatch)
        _forbid_gauss_newton(monkeypatch)
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y_new, info = sqp._project_step_inner(
            self.y, self.c, self.J, sqp._build_active_mask(self.c)
        )

        assert info["sqp_partition_mismatches"] == 1
        assert sqp.sqp_partition_mismatches_total == 1
        assert torch.allclose(y_new, torch.tensor([[1.0, 0.9]]), atol=1e-6)
        assert not bool(info["tight_mask"].any())     # proxy keeps the partition
        assert info["sqp_resolve_max_gap"] > 0.5

    def test_guard_does_not_fire_on_a_good_partition(self) -> None:
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        _, info = sqp._project_step_inner(
            self.y, self.c, self.J, sqp._build_active_mask(self.c)
        )
        assert info["sqp_partition_mismatches"] == 0
        assert sqp.sqp_partition_mismatches_total == 0

    def test_guard_keeps_the_proxy_differentiable(self, monkeypatch) -> None:
        """A guarded sample still produces a finite-gradient displacement proxy."""
        self._break_partition(monkeypatch)
        _forbid_gauss_newton(monkeypatch)

        torch.manual_seed(7)
        D, K, n_batch = 4, 2, 3
        layer = torch.nn.Linear(3, D)
        z = torch.randn(n_batch, 3)
        A = torch.randn(K, D)

        def values_fn(yy, conditions):  # noqa: ARG001 - unused
            return yy @ A.T - torch.tensor([0.5, -0.2])

        sqp = Projector(K, ["eq", "ineq"], prescale=False, method="sqp")
        y_hat = layer(z)
        c_pre = values_fn(y_hat, None)
        _, info = sqp.step(y_hat, c_pre, values_fn, None)

        assert info["sqp_partition_mismatches"] == n_batch
        assert torch.isfinite(info["displacement_live"]).all()
        info["displacement_live"].pow(2).sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()

    def test_inference_loop_reports_mismatches(self, monkeypatch) -> None:
        self._break_partition(monkeypatch)
        _forbid_gauss_newton(monkeypatch)

        K, D = 2, 3
        torch.manual_seed(11)
        A = torch.randn(K, D)

        def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
            value = yy @ A.T - torch.ones(K)
            clist = [
                SimpleNamespace(value=value[..., k], type="eq", name=f"eq_{k}")
                for k in range(K)
            ]
            return value.pow(2).sum(dim=-1), clist

        sqp = Projector(K, ["eq"] * K, prescale=False, method="sqp")
        _, info = sqp.project(torch.zeros(2, D), constraint_fn, None, max_iters=2)
        assert info["sqp_partition_mismatches"] >= 2
        assert info["sqp_partition_mismatches"] == sqp.sqp_partition_mismatches_total


class TestDualExtraction:
    """The partition is only as good as the duals it reads."""

    def test_missing_duals_count_as_a_solver_failure(self, monkeypatch) -> None:
        """A backend that returns `x` but no usable `z` must not be trusted."""
        import qpsolvers

        calls: dict = {"osqp": 0, "clarabel": 0}

        def fake(problem, solver, *args, **kwargs):
            calls[solver] = calls.get(solver, 0) + 1
            return SimpleNamespace(
                found=True,
                x=np.zeros(problem.q.shape[0]),
                z=None,
                extras=(
                    {"info": SimpleNamespace(status="solved")}
                    if solver == "osqp"
                    else {"status": "Solved"}
                ),
            )

        monkeypatch.setattr(qpsolvers, "solve_problem", fake)
        _forbid_gauss_newton(monkeypatch)
        sqp = Projector(2, ["ineq", "ineq"], prescale=False, method="sqp")
        y = torch.tensor([[2.0, 0.0]])
        c = torch.tensor([[1.0, -0.1]])
        J = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
        y_new, info = sqp._project_step_inner(y, c, J, sqp._build_active_mask(c))

        assert calls["osqp"] == 1 and calls["clarabel"] == 1
        assert info["sqp_qp_failures"] == 1
        assert torch.equal(y_new, y)          # transparent failure, Delta = 0

    def test_dual_tolerances_are_internal_constants(self) -> None:
        """No config knob was added for the dual tolerances (v1.2 decision 3)."""
        assert projector_module._SQP_DUAL_ZERO_TOL == 1e-8
        assert projector_module._SQP_DUAL_RHO_TOL == 1e-6
        for name in ("sqp_dual_zero_tol", "sqp_dual_rho_tol", "sqp_guard_atol"):
            with pytest.raises(TypeError):
                Projector(1, ["eq"], method="sqp", **{name: 1e-8})
