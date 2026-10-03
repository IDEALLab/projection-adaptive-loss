"""Unit tests for the `"ip"` barrier-Newton repair step in `Projector`."""

from __future__ import annotations

from types import SimpleNamespace

import torch

from pal.projection.projector import _IP_FTB, Projector


def _all_active(n_batch: int, K: int) -> torch.Tensor:
    return torch.ones(n_batch, K, dtype=torch.bool)


def _post_linearized(
    c: torch.Tensor, J: torch.Tensor, y: torch.Tensor, y_new: torch.Tensor
) -> torch.Tensor:
    """`c + J*(y_new - y)`, the row values the step *intends* to reach."""
    return c + torch.bmm(J, (y_new - y).unsqueeze(-1)).squeeze(-1)


def _forbid_gauss_newton(monkeypatch) -> None:
    """Make the `lm_k` / `eigh` inner solves explode if the ip branch reaches them."""

    def _boom(*args, **kwargs):  # noqa: ANN002, ANN003 - test guard
        raise AssertionError(
            "ip branch fell back to a Gauss-Newton inner solve, a failed "
            "barrier-Newton solve must be a transparent Delta = 0, never lm_k's "
            "step under the IP label"
        )

    monkeypatch.setattr(Projector, "_project_step_inner_lm_k", _boom)
    monkeypatch.setattr(Projector, "_project_step_inner_eigh", _boom)


class TestEqualityOnlyAnchor:
    """No inequalities + consistent equalities => min-norm direction as rho -> inf."""

    @staticmethod
    def _setup():
        torch.manual_seed(0)
        n_batch, K, D = 4, 3, 6
        J = torch.randn(n_batch, K, D)
        c = torch.randn(n_batch, K) * 0.3
        y = torch.randn(n_batch, D)
        lm_undamped = Projector(
            K, ["eq"] * K, prescale=False, method="lm_k",
            delta=0.0, lambda_min=0.0,
        )
        y_lm, _ = lm_undamped._project_step_inner(y, c, J, _all_active(n_batch, K))
        return (n_batch, K, D), J, c, y, y_lm - y

    def _direction(self, rho: float):
        (n_batch, K, D), J, c, y, d_lm = self._setup()
        ip = Projector(K, ["eq"] * K, prescale=False, method="ip", ip_rho=rho)
        y_ip, info = ip._project_step_inner(y, c, J, _all_active(n_batch, K))
        alpha = info["ip_alpha"].to(y.dtype).unsqueeze(-1)
        return (y_ip - y) / alpha, d_lm, info

    def test_matches_min_norm_at_default_rho(self) -> None:
        d_ip, d_lm, info = self._direction(1e3)
        assert torch.allclose(d_ip, d_lm, atol=1e-3)
        assert info["ip_newton_residual_max"] < 1e-12
        assert info["ip_solve_failures"] == 0
        assert info["ip_pinv_fallbacks"] == 0

    def test_elastic_bias_vanishes_as_rho_grows(self) -> None:
        d_ip, d_lm, _ = self._direction(1e9)
        assert torch.allclose(d_ip, d_lm, atol=1e-6)

    def test_alpha_is_the_ftb_safeguard_not_quenching(self) -> None:
        """alpha ~ 0.995 uniformly: the p/q barrier, not a conditioning problem."""
        _, _, info = self._direction(1e3)
        alpha = info["ip_alpha"]
        assert float(alpha.min()) > 0.99
        assert torch.allclose(alpha, torch.full_like(alpha, _IP_FTB), atol=2e-3)


class TestFixedPoint:
    """R1 endgame: v = 0 => mu = 0 => Delta = 0, with an exactly zero proxy gradient."""

    @staticmethod
    def _feasible_batch():
        torch.manual_seed(7)
        D, K, n_batch = 4, 3, 5
        layer = torch.nn.Linear(3, D)
        z = torch.randn(n_batch, 3)
        A = torch.randn(K, D)

        def values_fn(yy, conditions):  # noqa: ARG001 - unused
            lin = yy @ A.T
            # Row 0: identically zero equality. Rows 1-2: strictly negative.
            return torch.stack(
                [
                    lin[..., 0] - lin[..., 0],
                    -1.0 - lin[..., 1] ** 2,
                    -1.0 - lin[..., 2] ** 2,
                ],
                dim=-1,
            )

        projector = Projector(
            K, ["eq", "ineq", "ineq"], prescale=False, method="ip"
        )
        return projector, layer, layer(z), values_fn

    def test_step_is_exactly_zero(self) -> None:
        projector, _layer, y_hat, values_fn = self._feasible_batch()
        c_pre = values_fn(y_hat, None)
        c_det = c_pre.detach()
        assert float(c_det[:, 0].abs().max()) == 0.0
        assert float(c_det[:, 1:].max()) < 0.0
        y_tilde, info = projector.step(y_hat, c_pre, values_fn, None)

        assert torch.equal(y_tilde, y_hat.detach())
        assert torch.count_nonzero(info["displacement"]) == 0
        assert torch.count_nonzero(info["displacement_live"]) == 0
        assert info["ip_mu_mean"] == 0.0
        assert info["ip_solve_failures"] == 0

    def test_proxy_gradient_is_exactly_zero(self) -> None:
        projector, layer, y_hat, values_fn = self._feasible_batch()
        c_pre = values_fn(y_hat, None)
        _, info = projector.step(y_hat, c_pre, values_fn, None)
        info["displacement_live"].pow(2).sum().backward()
        assert layer.weight.grad is not None
        assert float(layer.weight.grad.abs().max()) == 0.0


def _simplex_constraint_fn(n: int):
    """s2-flavored: `sum y = 1` (eq) plus `-y_i <= 0` (ineq), all linear."""

    def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
        value = torch.cat([yy.sum(-1, keepdim=True) - 1.0, -yy], dim=-1)
        clist = [
            SimpleNamespace(
                value=value[..., k],
                type=("eq" if k == 0 else "ineq"),
                name=f"row_{k}",
            )
            for k in range(n + 1)
        ]
        return value.pow(2).sum(-1), clist

    return constraint_fn


class TestInferenceSelfAnnealing:
    """`project()` is the training map iterated; mu shrinks with the violation."""

    def test_converges_with_monotone_residuals(self) -> None:
        n = 3
        types = ["eq"] + ["ineq"] * n
        ip = Projector(n + 1, types, prescale=False, method="ip")
        constraint_fn = _simplex_constraint_fn(n)
        y0 = torch.tensor([[0.9, -0.4, 0.2]])

        _, info = ip.project(y0, constraint_fn, None, max_iters=30, tol=1e-6)
        assert info["iters"] < 30
        assert info["max_residual"] <= 1e-6
        assert info["ip_solve_failures"] == 0

        _, trace, converged = ip.project_trace(
            y0, constraint_fn, None, max_iters=30, tol=1e-6
        )
        assert converged
        residuals = [
            max(
                float(c[0, 0].abs()),
                float(c[0, 1:].clamp(min=0).max()),
            )
            for _i, c, _y in trace
        ]
        # Strictly decreasing from the first step (iteration 0 is the pre-projection state).
        assert len(residuals) >= 3
        for prev, cur in zip(residuals[1:-1], residuals[2:], strict=True):
            assert cur < prev, residuals


class TestStrictInteriority:
    """The damped step keeps every barrier slack strictly positive."""

    def test_alpha_damps_and_slacks_stay_positive(self) -> None:
        # Violated rows: the full Newton step gives s^+ = -c_I - J_I d < 0, so FTB must bite.
        J = torch.tensor([[[1.0, 0.0], [-1.0, -1.0]]])
        c = torch.tensor([[1.0, -0.1]])
        y = torch.tensor([[2.0, 0.0]])
        ip = Projector(2, ["ineq", "ineq"], prescale=False, method="ip")
        _, info = ip._project_step_inner(y, c, J, ip._build_active_mask(c))

        alpha = info["ip_alpha"]
        assert float(alpha.max()) < 1.0
        s_new = info["ip_s0"] + alpha.unsqueeze(-1) * info["ip_ds"]
        assert float(s_new.min()) > 0.0
        # The undamped step really goes negative, else the assertion above is vacuous.
        assert float((info["ip_s0"] + info["ip_ds"]).min()) < 0.0

    def test_elastic_slacks_stay_positive(self) -> None:
        torch.manual_seed(0)
        K, D = 4, 3
        J = torch.randn(1, K, D)
        c = torch.randn(1, K)
        y = torch.zeros(1, D)
        ip = Projector(K, ["eq"] * K, prescale=False, method="ip")
        _, info = ip._project_step_inner(y, c, J, _all_active(1, K))
        alpha = info["ip_alpha"].unsqueeze(-1)
        assert float((info["ip_p0"] + alpha * info["ip_dp"]).min()) > 0.0
        assert float((info["ip_q0"] + alpha * info["ip_dq"]).min()) > 0.0


class TestDisplacementProxyBackward:
    """`step()`'s displacement proxy must carry finite grads on the ip path."""

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

    def test_ip_backward_is_finite(self) -> None:
        loss, grad, info = self._run("ip")
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
        _, _, info = self._run("ip")
        assert not info["J"].requires_grad


class TestInconsistentEqualities:
    """5 equalities in 3 variables (s5 shape): elastic saturation, live alpha."""

    @staticmethod
    def _setup():
        torch.manual_seed(0)
        K, D = 5, 3
        return K, D, torch.randn(1, K, D), torch.randn(1, K), torch.zeros(1, D)

    def test_saturates_nu_without_quenching_alpha(self) -> None:
        K, _D, J, c, y = self._setup()
        ip = Projector(K, ["eq"] * K, prescale=False, method="ip")
        y_new, info = ip._project_step_inner(y, c, J, _all_active(1, K))

        assert torch.isfinite(y_new).all()
        assert info["ip_solve_failures"] == 0
        assert info["ip_newton_residual_max"] < 1e-10
        # Elastic saturation: nu reaches O(rho), not O(1), and stays under rho.
        assert info["ip_nu_abs_max"] > 10.0
        assert info["ip_nu_abs_max"] < ip.ip_rho
        # The split cold start keeps alpha alive; p0 = q0 = mu/rho would push nu to O(rho^2c/mu).
        assert float(info["ip_alpha"].min()) > 0.1

    def test_project_loop_does_not_raise(self) -> None:
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

        ip = Projector(K, ["eq"] * K, prescale=False, method="ip")
        y_final, info = ip.project(
            torch.zeros(2, D), constraint_fn, conditions=None, max_iters=5
        )
        assert torch.isfinite(y_final).all()
        assert info["iters"] == 5  # never converges; that is the point
        assert torch.isfinite(torch.tensor(info["max_residual"]))
        assert info["ip_solve_failures"] == 0


class TestNoPartition:
    """`lm_k` re-violation toy: the barrier keeps the satisfied row in the system."""

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
        assert post[0, 1] > 0.5  # row 2 newly violated

    def test_ip_does_not_reviolate(self) -> None:
        ip = Projector(2, ["ineq", "ineq"], prescale=False, method="ip")
        y_new, info = ip._project_step_inner(
            self.y, self.c, self.J, ip._build_active_mask(self.c)
        )
        post = _post_linearized(self.c, self.J, self.y, y_new)
        assert float(post[0, 1]) < 1e-2
        # One centered step at finite mu does not reach the boundary on the violated row.
        assert float(post[0, 0]) > 0.0
        assert info["ip_solve_failures"] == 0

    def test_iterated_projection_reaches_both_rows(self) -> None:
        def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
            value = torch.stack(
                [yy[..., 0] - 1.0, -yy[..., 0] - yy[..., 1] + 1.9], dim=-1
            )
            clist = [
                SimpleNamespace(value=value[..., k], type="ineq", name=f"g{k}")
                for k in range(2)
            ]
            return value.pow(2).sum(-1), clist

        ip = Projector(2, ["ineq", "ineq"], prescale=False, method="ip")
        _, info = ip.project(
            torch.tensor([[2.0, 0.0]]), constraint_fn, None, max_iters=30, tol=1e-6
        )
        assert info["iters"] < 30
        assert info["max_residual"] <= 1e-6


class TestDegenerateGeometry:
    """H >= I always, so no rank issue exists; the edge shapes must still run."""

    def test_duplicated_inequality_rows(self) -> None:
        """s6-style: exact duplicates + a scaled copy. Clean solve, no pinv."""
        J = torch.tensor(
            [[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [2.0, 0.0, 0.0]]]
        )
        c = torch.tensor([[0.5, 0.5, 0.3, 1.0]])
        y = torch.zeros(1, 3)
        ip = Projector(4, ["ineq"] * 4, prescale=False, method="ip")
        y_new, info = ip._project_step_inner(y, c, J, ip._build_active_mask(c))

        assert torch.isfinite(y_new).all()
        assert info["ip_pinv_fallbacks"] == 0
        assert info["ip_solve_failures"] == 0
        assert info["ip_newton_residual_max"] < 1e-12

    def test_all_inequalities_violated(self) -> None:
        """s0 = eps on every row => W = I exactly; step must reduce the violation."""
        torch.manual_seed(2)
        K, D = 3, 4
        J = torch.randn(1, K, D)
        c = torch.full((1, K), 0.5)
        y = torch.zeros(1, D)
        ip = Projector(K, ["ineq"] * K, prescale=False, method="ip")
        y_new, info = ip._project_step_inner(y, c, J, ip._build_active_mask(c))

        W = info["ip_map"]["W"]
        assert torch.allclose(W, torch.ones_like(W))
        post = _post_linearized(c, J, y, y_new)
        assert torch.isfinite(post).all()
        assert float(post.max()) < float(c.max())
        assert info["ip_solve_failures"] == 0

    def test_no_inequalities_bypasses_the_barrier(self) -> None:
        """m_I = 0 (s1/s5): H = I, no factorization, empty s / W tensors."""
        torch.manual_seed(4)
        K, D = 2, 4
        J = torch.randn(3, K, D)
        c = torch.randn(3, K)
        y = torch.zeros(3, D)
        ip = Projector(K, ["eq"] * K, prescale=False, method="ip")
        y_new, info = ip._project_step_inner(y, c, J, _all_active(3, K))

        assert torch.isfinite(y_new).all()
        assert info["ip_s0"].shape == (3, 0)
        assert info["ip_map"]["W"].shape == (3, 0)
        assert info["ip_newton_residual_max"] < 1e-12
        eye = torch.eye(D, dtype=torch.float64).expand(3, D, D)
        assert torch.allclose(info["ip_map"]["Hinv"], eye)

    def test_no_equalities_runs_cleanly(self) -> None:
        """m_E = 0: no Schur block, nu empty."""
        torch.manual_seed(4)
        K, D = 2, 4
        J = torch.randn(3, K, D)
        c = torch.randn(3, K).abs()
        y = torch.zeros(3, D)
        ip = Projector(K, ["ineq"] * K, prescale=False, method="ip")
        y_new, info = ip._project_step_inner(y, c, J, _all_active(3, K))

        assert torch.isfinite(y_new).all()
        assert info["ip_nu"].shape == (3, 0)
        assert info["ip_nu_abs_max"] == 0.0
        assert info["ip_newton_residual_max"] < 1e-12

    def test_interleaved_row_layout(self) -> None:
        """s4-style eq/ineq interleaving must route through the index tensors."""
        torch.manual_seed(5)
        types = ["ineq", "eq", "ineq", "eq", "ineq"]
        K, D = len(types), 4
        J = torch.randn(2, K, D)
        c = torch.randn(2, K) * 0.4
        y = torch.zeros(2, D)
        ip = Projector(K, types, prescale=False, method="ip")
        assert ip._eq_idx.tolist() == [1, 3]
        assert ip._ineq_idx.tolist() == [0, 2, 4]
        y_new, info = ip._project_step_inner(y, c, J, _all_active(2, K))
        assert torch.isfinite(y_new).all()
        assert info["ip_newton_residual_max"] < 1e-10


class TestTransparentFailure:
    """Non-finite Newton solve => Delta = 0. Never lm_k, never eigh."""

    @staticmethod
    def _break_the_solves(monkeypatch) -> None:
        real = torch.linalg.cholesky_ex

        def failing_cholesky_ex(A, **kwargs):
            L, _info = real(A, **kwargs)
            return L, torch.ones(A.shape[:-2], dtype=torch.int32, device=A.device)

        monkeypatch.setattr(torch.linalg, "cholesky_ex", failing_cholesky_ex)
        monkeypatch.setattr(
            torch.linalg, "pinv", lambda A, **kwargs: torch.full_like(A, float("nan"))
        )

    def test_step_takes_no_move_and_counts(self, monkeypatch) -> None:
        self._break_the_solves(monkeypatch)
        _forbid_gauss_newton(monkeypatch)

        torch.manual_seed(7)
        D, K, n_batch = 4, 2, 3
        layer = torch.nn.Linear(3, D)
        z = torch.randn(n_batch, 3)
        A = torch.randn(K, D)

        def values_fn(yy, conditions):  # noqa: ARG001 - unused
            return yy @ A.T - torch.tensor([0.5, -0.2])

        ip = Projector(K, ["eq", "ineq"], prescale=False, method="ip")
        y_hat = layer(z)
        c_pre = values_fn(y_hat, None)
        y_tilde, info = ip.step(y_hat, c_pre, values_fn, None)

        assert info["ip_solve_failures"] == n_batch
        assert info["ip_pinv_fallbacks"] == n_batch
        assert ip.ip_solve_failures_total == n_batch
        assert ip.ip_pinv_fallbacks_total == n_batch
        assert torch.equal(y_tilde, y_hat.detach())
        assert torch.count_nonzero(info["displacement"]) == 0
        assert torch.count_nonzero(info["displacement_live"]) == 0
        assert info["ip_newton_residual_max"] == 0.0
        assert info["ip_nu_abs_max"] == 0.0

        info["displacement_live"].pow(2).sum().backward()
        assert layer.weight.grad is not None
        assert float(layer.weight.grad.abs().max()) == 0.0

    def test_inference_loop_sums_counters(self, monkeypatch) -> None:
        self._break_the_solves(monkeypatch)
        _forbid_gauss_newton(monkeypatch)

        K, D = 2, 3
        torch.manual_seed(1)
        A = torch.randn(K, D)

        def constraint_fn(yy, conditions):  # noqa: ARG001 - unused
            value = yy @ A.T - torch.ones(K)
            clist = [
                SimpleNamespace(value=value[..., k], type="eq", name=f"eq_{k}")
                for k in range(K)
            ]
            return value.pow(2).sum(dim=-1), clist

        ip = Projector(K, ["eq"] * K, prescale=False, method="ip")
        y0 = torch.zeros(2, D)
        y_final, info = ip.project(y0, constraint_fn, None, max_iters=3)

        assert torch.equal(y_final, y0)  # nothing moved
        assert info["ip_solve_failures"] == 3 * 2  # iters x batch
        assert info["ip_pinv_fallbacks"] == 3 * 2
        assert ip.ip_solve_failures_total == 6


class TestFixedMuOverride:
    """The inert R2 knob: `ip_fixed_mu` replaces the R1 rule when set."""

    def test_default_is_the_r1_rule(self) -> None:
        ip = Projector(2, ["eq", "ineq"], prescale=False, method="ip")
        assert ip.ip_fixed_mu is None

    def test_fixed_mu_breaks_the_fixed_point(self) -> None:
        """R2's documented endgame bias: a feasible sample still moves."""
        J = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        c = torch.tensor([[0.0, -1.0]])  # eq satisfied, ineq strictly inside
        y = torch.zeros(1, 2)
        r1 = Projector(2, ["eq", "ineq"], prescale=False, method="ip")
        y_r1, _ = r1._project_step_inner(y, c, J, _all_active(1, 2))
        assert torch.equal(y_r1, y)

        r2 = Projector(
            2, ["eq", "ineq"], prescale=False, method="ip", ip_fixed_mu=0.1
        )
        y_r2, info = r2._project_step_inner(y, c, J, _all_active(1, 2))
        assert not torch.equal(y_r2, y)
        assert info["ip_mu_mean"] == 0.1
