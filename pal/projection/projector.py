"""Constraint projector with four interchangeable repair steps.

``eigh`` and ``lm_k`` take one linearized step ``y^+ = y - B^T (B B^T + R)^-1 c_active``
(Tikhonov and Yamashita-Fukushima damping), ``sqp`` solves an elastic-mode
feasibility QP, and ``ip`` takes one damped log-barrier Newton step at finite mu.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from torch import Tensor

# Floor on the Yamashita-Fukushima damping: covers rank-deficient active sets at
# feasibility and fp32 roundoff on BB^T (~1e-7 * K).
_LM_LAMBDA_MIN = 1e-6

# rho is *the* knob: exact projection whenever the linearization is consistent
# and rho > ||mu*||inf; below that the step goes elastic on the rows it cannot hold.
_SQP_RHO = 1e3
_SQP_OSQP_EPS = 1e-8
# Relative tolerance of the tight-vs-inactive residual test.
_SQP_TIGHT_TOL = 1e-6
# |mu_i| at or below the solver tolerance: the row is treated as inactive.
_SQP_DUAL_ZERO_TOL = 1e-8
# |mu_i| >= (1 - tol)*rho: the row sits at the l1 bound, i.e. it is elastic.
_SQP_DUAL_RHO_TOL = 1e-6
# Fall back to the QP's own Delta where the torch re-solve disagrees by more than this.
_SQP_GUARD_RTOL = 1e-6
_SQP_GUARD_ATOL = 1e-4
# Relative singular-value cutoff of the min-norm re-solve on the tight rows.
_SQP_RCOND = 1e-10

# mu0 in the rule mu = mu0 * v(y_hat).
_IP_MU0 = 0.1
# Elastic penalty on the equality slack split, same role as _SQP_RHO.
_IP_RHO = 1e3
# Fraction-to-the-boundary safeguard factor (Nocedal & Wright Ch. 19).
_IP_FTB = 0.995
# Relative eigenvalue cutoff of the pseudo-inverse retry on failed Cholesky.
_IP_RCOND = 1e-10


_JACOBIAN_MODE_ANNOUNCED = False


def _announce_jacobian_mode_once(mode: str) -> None:
    """Print a one-line note about the active ``jacobian_mode`` once per process."""
    global _JACOBIAN_MODE_ANNOUNCED
    if _JACOBIAN_MODE_ANNOUNCED:
        return
    _JACOBIAN_MODE_ANNOUNCED = True
    if mode == "loop":
        print(
            "[pal.projector] jacobian_mode='loop' (default, probe-faithful). "
            "Set jacobian_mode='vmap_jacrev' to opt into batched VJP "
            "(torch.vmap+torch.func.jacrev; faster on large K but probe "
            "undercounts inner VJPs)."
        )
    else:
        print(
            f"[pal.projector] jacobian_mode={mode!r} active, "
            "probe fwd/bwd counts undercount inner VJPs on this path."
        )


def _sqp_solution_status(solver: str, solution) -> str:
    """Lower-cased backend status of a ``qpsolvers`` solution, or ``"unknown"``."""
    extras = getattr(solution, "extras", None) or {}
    if solver == "osqp":
        raw = getattr(extras.get("info"), "status", None)
    else:
        raw = extras.get("status")
    if raw is None:
        return "unknown"
    return str(raw).strip().lower().rsplit(".", 1)[-1]


class Projector:
    """Constraint projection with a displacement-proxy autograd path.

    One class, four inner solves selected by ``method``. The public API
    (``step`` / ``project`` / ``project_trace``) is method-agnostic.
    Args:
        n_constraints: Number of constraints K.
        constraint_types: Per-constraint ``"eq"`` / ``"ineq"`` flag.
        delta: Regularization coefficient. For ``method="eigh"``: Tikhonov
            ratio against the top eigenvalue, ``eps = max(delta * lambda_max, 1e-4)``.
            For ``method="lm_k"``: Yamashita-Fukushima coefficient, per-sample
            ``lambda = max(delta * ||c_active||^2, lambda_min)``.
        prescale: If True, normalize Jacobian rows to unit norm before solve
            (``c`` is rescaled consistently). Acts as a left-preconditioner
            that equalizes constraint scales.
        prescale_floor: Lower bound on the prescale divisor. Rows with norm
            below this pass through unscaled.
        eps_active: Threshold for eq activity in ``project_trace``
            diagnostics (used only for reporting, not the solve).
        method: Which inner solve to use: ``"eigh"`` (spectrum-adaptive
            Tikhonov), ``"lm_k"`` (Yamashita-Fukushima LM), ``"sqp"``
            (elastic-mode feasibility QP) or ``"ip"`` (one damped
            log-barrier Newton step at finite mu).
        sqp_rho: ``"sqp"`` only, elastic penalty rho (default 1e3).
        sqp_osqp_eps: ``"sqp"`` only, OSQP ``eps_abs``/``eps_rel``.
        sqp_tight_tol: ``"sqp"`` only, relative tolerance of the boundary
            (residual) discriminant that separates tight from inactive rows
            among the QP multipliers that are neither ~ 0 nor ~ +/-rho.
        ip_mu0: ``"ip"`` only, mu0 in the rule ``mu = mu0 * v(y_hat)`` (default 0.1).
        ip_rho: ``"ip"`` only, elastic penalty rho on the equality slack
            split (default 1e3).
        ip_fixed_mu: ``"ip"`` only, when not None, use this constant mu for
            every sample instead of the mu0 rule.
        jacobian_mode: How to build ``J = dc/dy`` in ``step()``.
            ``"loop"`` (default) does K ``autograd.grad`` VJPs on the live
            ``c_pre``. ``"vmap_jacrev"`` uses ``torch.vmap(torch.func.jacrev(...))``
            on ``values_fn``; faster for large K, but the probe undercounts
            inner VJPs because ``torch.vmap`` bypasses autograd hooks.
    """

    def __init__(
        self,
        n_constraints: int,
        constraint_types: list[str],
        delta: float = 1e-3,
        prescale: bool = True,
        prescale_floor: float = 1.0,
        eps_active: float = 1e-4,
        method: str = "eigh",
        box_lower: Tensor | None = None,
        box_upper: Tensor | None = None,
        detach_j: bool = True,
        jacobian_mode: str = "loop",
        lambda_min: float = _LM_LAMBDA_MIN,
        eigh_fallback_reg: float = 1e-6,
        sqp_rho: float = _SQP_RHO,
        sqp_osqp_eps: float = _SQP_OSQP_EPS,
        sqp_tight_tol: float = _SQP_TIGHT_TOL,
        ip_mu0: float = _IP_MU0,
        ip_rho: float = _IP_RHO,
        ip_fixed_mu: float | None = None,
    ):
        if method not in ("eigh", "lm_k", "sqp", "ip"):
            raise ValueError(
                f"unknown projection method {method!r}; "
                "expected 'eigh', 'lm_k', 'sqp', or 'ip'"
            )
        if jacobian_mode == "vmap":
            jacobian_mode = "vmap_jacrev"
        if jacobian_mode not in ("loop", "vmap_jacrev"):
            raise ValueError(
                f"unknown jacobian_mode {jacobian_mode!r}; "
                "expected 'loop', 'vmap', or 'vmap_jacrev'"
            )
        if (box_lower is None) != (box_upper is None):
            raise ValueError("box_lower and box_upper must either both be set or both be None")
        if box_lower is not None and box_upper is not None and box_lower.shape != box_upper.shape:
            raise ValueError("box_lower and box_upper must have the same shape")
        self.n_constraints = n_constraints
        self.constraint_types = list(constraint_types)
        self.delta = delta
        self.prescale = prescale
        self.prescale_floor = prescale_floor
        self.eps_active = eps_active
        self.method = method
        self.detach_j = detach_j
        self.jacobian_mode = jacobian_mode
        self.lambda_min = lambda_min
        # Pre-eigh diagonal regularizer; set 0 to probe undamped Newton.
        self.eigh_fallback_reg = float(eigh_fallback_reg)
        self.sqp_rho = float(sqp_rho)
        self.sqp_osqp_eps = float(sqp_osqp_eps)
        self.sqp_tight_tol = float(sqp_tight_tol)
        # Cumulative telemetry, incremented by training and inference entry points.
        self.sqp_qp_failures_total = 0
        self.sqp_clarabel_rescues_total = 0
        self.sqp_partition_mismatches_total = 0
        self.ip_mu0 = float(ip_mu0)
        self.ip_rho = float(ip_rho)
        self.ip_fixed_mu = None if ip_fixed_mu is None else float(ip_fixed_mu)
        self.ip_solve_failures_total = 0
        self.ip_pinv_fallbacks_total = 0
        self._is_eq = [t == "eq" for t in constraint_types]
        self._is_eq_np = np.array(self._is_eq, dtype=bool)
        self._eq_idx = torch.tensor(
            [i for i, e in enumerate(self._is_eq) if e], dtype=torch.long
        )
        self._ineq_idx = torch.tensor(
            [i for i, e in enumerate(self._is_eq) if not e], dtype=torch.long
        )
        self.box_lower = box_lower
        self.box_upper = box_upper
        if method == "sqp":
            try:
                import clarabel  # noqa: F401
                import qpsolvers  # noqa: F401
            except ImportError as e:  # pragma: no cover - env-dependent
                raise RuntimeError(
                    "method='sqp' needs the elastic-QP extras: "
                    "`pip install qpsolvers osqp clarabel` (all three declared "
                    "in pyproject.toml under the pal_sqp repair group; "
                    "Clarabel is the second solver in the chain). "
                    f"({type(e).__name__}: {e})"
                ) from e

    def set_jacobian_mode(self, mode: str) -> None:
        """Switch the Jacobian strategy used by `step()`.

        Solvers call this per-epoch under `jacobian_mode="sample"`.
        """
        if mode == "vmap":
            mode = "vmap_jacrev"
        if mode not in ("loop", "vmap_jacrev"):
            raise ValueError(
                f"unknown jacobian_mode {mode!r}; "
                "expected 'loop', 'vmap', or 'vmap_jacrev'"
            )
        self.jacobian_mode = mode

    def _clamp_to_box(self, y: Tensor) -> Tensor:
        if self.box_lower is None or self.box_upper is None:
            return y
        lower = self.box_lower.to(device=y.device, dtype=y.dtype)
        upper = self.box_upper.to(device=y.device, dtype=y.dtype)
        return torch.maximum(torch.minimum(y, upper), lower)

    def _jacobian_at_detached_y(
        self,
        y: Tensor,
        constraint_fn: Callable,
        conditions: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return `(c_pre [B,K], J_detached [B,K,D])`.

        Does a forward on a detached copy of ``y`` to get ``J``, then a second
        forward on the live ``y`` so ``c_pre`` carries gradient.
        """
        B, D = y.shape
        K = self.n_constraints

        with torch.enable_grad():
            y_det = y.detach().requires_grad_(True)
            _, constraint_list_det = constraint_fn(y_det, conditions)
            c_det = torch.stack([c.value for c in constraint_list_det], dim=-1)

            J = torch.zeros(B, K, D, device=y.device, dtype=y.dtype)
            for k in range(K):
                g = torch.autograd.grad(
                    c_det[:, k].sum(), y_det, retain_graph=(k < K - 1)
                )[0]
                J[:, k, :] = g

        _, constraint_list_live = constraint_fn(y, conditions)
        c_pre = torch.stack([c.value for c in constraint_list_live], dim=-1)

        return c_pre, (J.detach() if self.detach_j else J)

    def _jacobian_from_live_c(self, y: Tensor, c_pre: Tensor) -> Tensor:
        """Return `J_detached [B,K,D]` via K autograd.grad VJPs on ``c_pre``.

        Reuses the caller-supplied live ``c_pre`` as the VJP source, with
        ``retain_graph=True`` so the graph survives for ``loss.backward()``.
        """
        B, D = y.shape
        K = self.n_constraints
        J = torch.zeros(B, K, D, device=y.device, dtype=y.dtype)
        for k in range(K):
            g = torch.autograd.grad(
                c_pre[:, k].sum(), y, retain_graph=True
            )[0]
            J[:, k, :] = g
        return J.detach() if self.detach_j else J

    def _jacobian_vmap_jacrev(
        self,
        y: Tensor,
        values_fn: Callable,
        conditions: Tensor | None,
    ) -> Tensor:
        """Return `J_detached [B,K,D]` via `vmap(jacrev(values_fn))`.

        Rebuilds the per-sample Jacobian on ``y.detach()`` and batches it over
        B. The probe undercounts inner-VJP work on this path because
        ``torch.vmap`` bypasses ``Tensor.register_hook``.
        """
        B, D = y.shape
        y_det = y.detach()

        if conditions is not None:
            def _c_single(y_b: Tensor, cond_b: Tensor) -> Tensor:
                # values_fn expects a batch dim; re-add size-1 inside vmap.
                return values_fn(y_b.unsqueeze(0), cond_b.unsqueeze(0)).squeeze(0)

            J = torch.vmap(
                torch.func.jacrev(_c_single, argnums=0),
                in_dims=(0, 0),
            )(y_det, conditions)
        else:
            def _c_single_uncond(y_b: Tensor) -> Tensor:
                return values_fn(y_b.unsqueeze(0), None).squeeze(0)

            J = torch.vmap(
                torch.func.jacrev(_c_single_uncond, argnums=0),
            )(y_det)

        return J.detach() if self.detach_j else J

    def _build_active_mask(self, c_values: Tensor) -> Tensor:
        """`[B,K]` bool mask. Ineq active iff value > 0; eq always active."""
        is_eq = torch.tensor(self._is_eq, device=c_values.device)
        ineq_active = c_values > 0
        return torch.where(is_eq.unsqueeze(0), True, ineq_active)

    def _project_step_inner(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict]:
        """Dispatch to the inner solve selected by ``self.method``.

        Returns ``(y_tilde, info)``; ``info["method"]`` tags the inner solve.
        """
        if self.method == "ip":
            return self._project_step_inner_ip(y, c_values, J, active)
        if self.method == "sqp":
            return self._project_step_inner_sqp(y, c_values, J, active)
        if self.method == "lm_k":
            return self._project_step_inner_lm_k(y, c_values, J, active)
        return self._project_step_inner_eigh(y, c_values, J, active)

    def _project_step_inner_eigh(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict]:
        """One spectrum-adaptive Tikhonov step via eigendecomposition.

        Computes ``y^+ = y - B^T (B B^T + eps I)^-1 c_masked`` with
        ``eps = max(delta * lambda_max(BB^T), 1e-4)``, shifting every eigenvalue by eps.

        Returns:
            ``(y_tilde, eig_info)`` with ``U``, ``D_reg``, ``B_mat`` (and
            ``row_norms`` if prescaled) for the displacement proxy in ``step()``.
        """
        B, K, D = J.shape
        mask_float = active.to(J.dtype).unsqueeze(-1)
        B_mat = J * mask_float
        c_masked = c_values * active.to(c_values.dtype)

        if self.prescale:
            row_norms = B_mat.norm(dim=2, keepdim=True).clamp(min=self.prescale_floor)
            B_mat = B_mat / row_norms
            c_masked = c_masked / row_norms.squeeze(-1)

        BBT = torch.bmm(B_mat, B_mat.transpose(1, 2))
        # Pre-regularize so LAPACK sees a strictly PD matrix on rank-deficient BBT.
        if self.eigh_fallback_reg > 0.0:
            eye = torch.eye(BBT.shape[-1], device=BBT.device, dtype=BBT.dtype)
            BBT_pre = BBT + self.eigh_fallback_reg * eye
        else:
            BBT_pre = BBT
        eigenvalues, U = torch.linalg.eigh(BBT_pre)
        lam_max = eigenvalues[..., -1:].clamp(min=1e-12)
        eps = (self.delta * lam_max).clamp(min=1e-4)
        D_reg = 1.0 / (eigenvalues + eps)

        By = torch.bmm(B_mat, y.unsqueeze(-1)).squeeze(-1)
        v = By - c_masked

        def _reg_solve(rhs: Tensor) -> Tensor:
            is_vec = rhs.dim() == 2
            if is_vec:
                rhs = rhs.unsqueeze(-1)
            Ut_rhs = U.transpose(-1, -2) @ rhs
            result = U @ (D_reg.unsqueeze(-1) * Ut_rhs)
            return result.squeeze(-1) if is_vec else result

        inv_BBT_B = _reg_solve(B_mat)
        inv_BBT_v = _reg_solve(v)

        proj_By = torch.bmm(inv_BBT_B, y.unsqueeze(-1)).squeeze(-1)
        term1 = y - torch.bmm(
            B_mat.transpose(1, 2), proj_By.unsqueeze(-1)
        ).squeeze(-1)
        term2 = torch.bmm(
            B_mat.transpose(1, 2), inv_BBT_v.unsqueeze(-1)
        ).squeeze(-1)

        eig_info = {"method": "eigh", "U": U, "D_reg": D_reg, "B_mat": B_mat}
        if self.prescale:
            eig_info["row_norms"] = row_norms.squeeze(-1)
        return term1 + term2, eig_info

    def _project_step_inner_lm_k(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict]:
        """One K-space Levenberg-Marquardt step with Yamashita-Fukushima damping.

        Computes ``y^+ = y - B^T (B B^T + lambda I)^-1 c_masked`` with per-sample
        damping ``lambda = max(delta * ||c_masked||^2, lambda_min)``, solved by batched Cholesky.

        Returns:
            ``(y_tilde, info)`` with the Cholesky factor ``L``, ``B_mat`` (and
            ``row_norms`` if prescaled).
        """
        B, K, D = J.shape
        mask_float = active.to(J.dtype).unsqueeze(-1)
        B_mat = J * mask_float
        c_masked = c_values * active.to(c_values.dtype)

        if self.prescale:
            row_norms = B_mat.norm(dim=2, keepdim=True).clamp(min=self.prescale_floor)
            B_mat = B_mat / row_norms
            c_masked = c_masked / row_norms.squeeze(-1)

        BBT = torch.bmm(B_mat, B_mat.transpose(1, 2))
        c_norm_sq = c_masked.pow(2).sum(dim=1, keepdim=True)
        lam = (self.delta * c_norm_sq).clamp(min=self.lambda_min)
        eye = torch.eye(K, device=BBT.device, dtype=BBT.dtype)
        BBT_reg = BBT + lam.unsqueeze(-1) * eye

        # Cholesky fallback chain: fp32 -> fp64 -> eigh.
        try:
            L = torch.linalg.cholesky(BBT_reg)
        except torch._C._LinAlgError:
            try:
                L = torch.linalg.cholesky(BBT_reg.double()).to(BBT_reg.dtype)
            except torch._C._LinAlgError:
                return self._project_step_inner_eigh(y, c_values, J, active)
        z = torch.cholesky_solve(c_masked.unsqueeze(-1), L).squeeze(-1)
        dy = torch.bmm(B_mat.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1)
        y_tilde = y - dy

        info = {"method": "lm_k", "L": L, "B_mat": B_mat}
        if self.prescale:
            info["row_norms"] = row_norms.squeeze(-1)
        return y_tilde, info

    def _sqp_try_solver(
        self, problem, solver: str
    ) -> tuple[np.ndarray | None, np.ndarray | None, str]:
        """Run one backend on ``problem``; accept only a clean ``solved``.

        Returns ``(x, z, status)`` with ``x = None`` unless the backend reported
        ``solved`` with a finite primal iterate and finite duals. ``z`` holds
        the multipliers of the ``G x <= h`` block in the caller's row order.
        """
        import qpsolvers

        if solver == "osqp":
            kwargs = {
                "eps_abs": self.sqp_osqp_eps,
                "eps_rel": self.sqp_osqp_eps,
                "polishing": True,
                "max_iter": 10000,
            }
        else:
            kwargs = {
                "tol_gap_abs": self.sqp_osqp_eps,
                "tol_gap_rel": self.sqp_osqp_eps,
                "tol_feas": self.sqp_osqp_eps,
            }
        try:
            solution = qpsolvers.solve_problem(problem, solver=solver, **kwargs)
        except Exception as exc:  # noqa: BLE001 - any backend failure -> chain
            return None, None, f"exception:{type(exc).__name__}"

        status = _sqp_solution_status(solver, solution)
        if not bool(getattr(solution, "found", False)) or status != "solved":
            return None, None, status
        x = solution.x
        if x is None or not np.all(np.isfinite(x)):
            return None, None, f"{status}:non-finite"
        z = getattr(solution, "z", None)
        n_dual = problem.G.shape[0]
        if z is None or np.shape(z) != (n_dual,) or not np.all(np.isfinite(z)):
            return None, None, f"{status}:no-duals"
        return (
            np.asarray(x, dtype=np.float64),
            np.asarray(z, dtype=np.float64),
            status,
        )

    def _sqp_solve_one(
        self, problem
    ) -> tuple[np.ndarray | None, np.ndarray | None, bool, str]:
        """Solve one elastic QP with OSQP, retrying once with Clarabel.

        Returns ``(x, z, rescued, status)``: ``x`` / ``z`` are ``None`` if both
        backends failed, ``rescued`` is True when Clarabel produced the solution.
        """
        x, z, status = self._sqp_try_solver(problem, "osqp")
        if x is not None:
            return x, z, False, status
        osqp_status = status
        x, z, status = self._sqp_try_solver(problem, "clarabel")
        if x is not None:
            return x, z, True, status
        return None, None, False, f"osqp={osqp_status}, clarabel={status}"

    def _sqp_elastic_partition(
        self, J_np: np.ndarray, c_np: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int, int]:
        """Solve the per-sample elastic feasibility QP and read off its partition.

        Args:
            J_np: ``[B, K, D]`` float64 Jacobian (already row-prescaled if
                ``self.prescale``).
            c_np: ``[B, K]`` float64 constraint values, same scaling.

        Returns ``(tight, mu_elastic, delta_qp, solved, n_failures, n_rescues)``:
            tight: ``[B, K]`` bool, rows the QP step holds at equality.
            mu_elastic: ``[B, K]`` float64, +/-rho on elastic rows, 0 elsewhere.
            delta_qp: ``[B, D]`` float64, the QP's own step.
            solved: ``[B]`` bool, False where both QP backends failed.
            n_failures: number of samples where neither backend succeeded.
            n_rescues: number of samples OSQP failed on and Clarabel solved.

        The QP is posed in ``x = [Delta, t]`` with ``t_i >= 0`` the l1 split of the
        slack. The partition is read from the duals, ``mu_i = z^+_i - z^-_i`` on eq
        rows and ``mu_i = z^+_i`` on ineq rows, so that ``Delta = -J^T mu``. A sample
        both backends fail on gets an empty tight set and hence Delta = 0.
        """
        import qpsolvers
        from scipy import sparse

        n_batch, K, D = J_np.shape
        is_eq = self._is_eq_np
        eq_idx = np.flatnonzero(is_eq)
        eye_k = np.eye(K)
        n_var = D + K
        rho = self.sqp_rho

        P = sparse.diags(
            np.concatenate([np.ones(D), np.zeros(K)]), format="csc"
        )
        q = np.concatenate([np.zeros(D), np.full(K, rho)])
        lb = np.concatenate([np.full(D, -np.inf), np.zeros(K)])
        ub = np.full(n_var, np.inf)

        tight = np.zeros((n_batch, K), dtype=bool)
        mu_elastic = np.zeros((n_batch, K), dtype=np.float64)
        delta_qp = np.zeros((n_batch, D), dtype=np.float64)
        solved = np.ones(n_batch, dtype=bool)
        n_failures = 0
        n_rescues = 0
        last_failure_status = ""

        for b in range(n_batch):
            Jb = J_np[b]
            cb = c_np[b]
            # rows: [J_i, -e_i] Delta,t <= -c_i   for every row (eq and ineq)
            #       [-J_i, -e_i] Delta,t <=  c_i   for eq rows only
            G_dense = np.vstack(
                [
                    np.hstack([Jb, -eye_k]),
                    np.hstack([-Jb[eq_idx], -eye_k[eq_idx]]),
                ]
            )
            G = sparse.csc_matrix(G_dense)
            h = np.concatenate([-cb, cb[eq_idx]])
            problem = qpsolvers.Problem(P, q, G=G, h=h, lb=lb, ub=ub)
            x, z, rescued, status = self._sqp_solve_one(problem)
            if rescued:
                n_rescues += 1
            if x is None:
                solved[b] = False
                n_failures += 1
                last_failure_status = status
                continue

            delta = x[:D]
            delta_qp[b] = delta
            # Multiplier per row; the mirrored -J block of an eq row subtracts.
            mu = z[:K].copy()
            mu[eq_idx] -= z[K:]
            abs_mu = np.abs(mu)
            r = cb + Jb @ delta
            on_boundary = np.abs(r) <= self.sqp_tight_tol * (1.0 + np.abs(cb))
            elastic = (abs_mu >= (1.0 - _SQP_DUAL_RHO_TOL) * rho) & ~on_boundary
            tight[b] = (abs_mu > _SQP_DUAL_ZERO_TOL) & on_boundary
            mu_elastic[b] = np.where(elastic, rho * np.sign(mu), 0.0)

        self.sqp_qp_failures_total += n_failures
        self.sqp_clarabel_rescues_total += n_rescues
        if n_failures > 0:
            print(
                f"[pal.projector] sqp: elastic QP FAILED on {n_failures}/"
                f"{n_batch} samples (last status: {last_failure_status}). "
                "Those samples take Delta=0 (y_tilde=y_hat, zero proxy contribution) and "
                "are counted in sqp/qp_failures. No lm_k fallback is applied.",
                flush=True,
            )
        return tight, mu_elastic, delta_qp, solved, n_failures, n_rescues

    def _project_step_inner_sqp(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict]:
        """One elastic-mode SQP feasibility step (QP tight set + torch re-solve).

        OSQP (Clarabel on failure) solves the elastic QP over all rows and its
        duals give the tight set T and elastic set E. A batched torch re-solve
        on T then gives the step and the linear map the displacement proxy
        re-applies in ``step()``::

            (J_T J_T^T) mu_T = c_T - J_T J_E^T mu_E
            Delta*             = -J_T^T mu_T - J_E^T mu_E

        with ``dy = -Delta*``. Where the re-solve disagrees with the QP's Delta, the QP's
        Delta is taken as the step. ``active`` is unused.

        Returns:
            ``(y_tilde, info)`` with ``gram_pinv``, ``B_mat``, ``tight_mask``,
            ``disp_elastic``, ``rhs_offset`` and the ``sqp_*`` diagnostics.
        """
        n_batch, K, D = J.shape
        B_all = J
        c_all = c_values

        if self.prescale:
            row_norms = B_all.norm(dim=2, keepdim=True).clamp(min=self.prescale_floor)
            B_all = B_all / row_norms
            c_all = c_all / row_norms.squeeze(-1)

        J_np = B_all.detach().to(dtype=torch.float64, device="cpu").numpy()
        c_np = c_all.detach().to(dtype=torch.float64, device="cpu").numpy()
        (
            tight_np,
            mu_el_np,
            delta_qp_np,
            solved_np,
            n_failures,
            n_rescues,
        ) = self._sqp_elastic_partition(J_np, c_np)
        tight = torch.from_numpy(tight_np).to(device=J.device)
        mu_elastic = torch.from_numpy(mu_el_np).to(device=J.device, dtype=J.dtype)

        # fp64: J_T^T mu_T and J_E^T mu_E are each O(rho) and cancel.
        work = torch.float64
        mask = tight.to(dtype=work)
        B_all_w = B_all.to(work)
        B_T = B_all_w * mask.unsqueeze(-1)
        # mu_elastic is zero on T and on inactive rows, so B^T mu_elastic = J_E^T mu_E.
        disp_elastic = torch.bmm(
            B_all_w.detach().transpose(1, 2), mu_elastic.to(work).unsqueeze(-1)
        ).squeeze(-1)
        rhs_offset = torch.bmm(
            B_T.detach(), disp_elastic.unsqueeze(-1)
        ).squeeze(-1)

        # Min-norm solve; non-tight rows are all-zero in the Gram matrix.
        gram = torch.bmm(B_T, B_T.transpose(1, 2))
        gram_pinv = self._sqp_gram_pinv(gram)
        rhs = (c_all.to(work) * mask - rhs_offset) * mask
        z = torch.bmm(gram_pinv, rhs.unsqueeze(-1)).squeeze(-1)
        dy_w = torch.bmm(B_T.transpose(1, 2), z.unsqueeze(-1)).squeeze(-1) + disp_elastic

        ok = torch.from_numpy(solved_np).to(device=J.device)
        delta_qp = torch.from_numpy(delta_qp_np).to(device=J.device, dtype=work)
        per_sample_gap = (dy_w.detach() + delta_qp).abs().amax(dim=-1)
        threshold = (
            _SQP_GUARD_RTOL * delta_qp.abs().amax(dim=-1).clamp(min=1.0)
            + _SQP_GUARD_ATOL
        )
        mismatch = ok & (per_sample_gap > threshold)
        n_mismatches = int(mismatch.sum().item())
        if n_mismatches:
            dy_w = torch.where(mismatch.unsqueeze(-1), -delta_qp, dy_w)
        self.sqp_partition_mismatches_total += n_mismatches
        gap = (
            float(per_sample_gap[ok].max().item()) if bool(ok.any()) else float("nan")
        )
        y_tilde = y - dy_w.to(y.dtype)

        info = {
            "method": "sqp",
            "gram_pinv": gram_pinv,
            "B_mat": B_T,
            "tight_mask": tight,
            "disp_elastic": disp_elastic,
            "rhs_offset": rhs_offset,
            "sqp_qp_failures": n_failures,
            "sqp_clarabel_rescues": n_rescues,
            "sqp_partition_mismatches": n_mismatches,
            "sqp_resolve_max_gap": gap,
        }
        if self.prescale:
            info["row_norms"] = row_norms.squeeze(-1)
        return y_tilde, info

    @staticmethod
    def _sqp_gram_pinv(gram: Tensor) -> Tensor:
        """Rank-revealing pseudo-inverse of the tight-row Gram matrix.

        Drops eigendirections with ``lambda <= _SQP_RCOND * lambda_max``; expects float64
        input, since fp32 noise on the null directions exceeds the cutoff.
        """
        try:
            return torch.linalg.pinv(gram, rtol=_SQP_RCOND, hermitian=True)
        except torch._C._LinAlgError:  # pragma: no cover - LAPACK-dependent
            return torch.linalg.pinv(gram, rtol=_SQP_RCOND)

    @staticmethod
    def _ip_spd_inverse(A: Tensor) -> tuple[Tensor, Tensor]:
        """Batched inverse of an SPD ``[B, m, m]`` stack; ``(inverse, bad)``.
        Cholesky first; samples whose factorization fails are re-solved with the
        hermitian pseudo-inverse, and ``bad`` flags them.
        """
        L, info = torch.linalg.cholesky_ex(A)
        bad = info != 0
        eye = torch.eye(A.shape[-1], device=A.device, dtype=A.dtype)
        # Substitute I for failing factors so cholesky_inverse cannot see NaNs.
        L_safe = torch.where(bad.view(-1, 1, 1), eye.expand_as(L), L)
        A_inv = torch.cholesky_inverse(L_safe)
        if bool(bad.any()):
            try:
                retry = torch.linalg.pinv(A[bad], rtol=_IP_RCOND, hermitian=True)
            except torch._C._LinAlgError:  # pragma: no cover - LAPACK-dependent
                retry = torch.linalg.pinv(A[bad], rtol=_IP_RCOND)
            A_inv = A_inv.clone()
            A_inv[bad] = retry
        return A_inv, bad

    @staticmethod
    def _ip_newton_d(m: dict, c_eq: Tensor, c_ineq: Tensor) -> tuple[Tensor, Tensor]:
        """Full (undamped) Newton displacement ``d`` and ``nu^+`` from cached maps.

        ``c_eq`` / ``c_ineq`` are detached when building the step and live when
        ``step()`` re-applies the same map to ``c_pre``::

            r1 = -J_I^T W c_I + k
            nu^+ = M^-1 (c_E + J_E H^-1 r1)
            d  = H^-1 (r1 - J_E^T nu^+)
        """
        if m["has_ineq"]:
            r1 = m["k"] - torch.bmm(
                m["JI_T"], (m["W"] * c_ineq).unsqueeze(-1)
            ).squeeze(-1)
        else:
            r1 = m["k"]
        Hinv_r1 = torch.bmm(m["Hinv"], r1.unsqueeze(-1))
        if m["has_eq"]:
            nu = torch.bmm(
                m["Minv"], (c_eq.unsqueeze(-1) + torch.bmm(m["JE"], Hinv_r1))
            )
            d = torch.bmm(
                m["Hinv"], r1.unsqueeze(-1) - torch.bmm(m["JE_T"], nu)
            ).squeeze(-1)
        else:
            nu = c_eq.unsqueeze(-1)  # [B, 0, 1]
            d = Hinv_r1.squeeze(-1)
        return d, nu.squeeze(-1)

    @staticmethod
    def _ip_damped_dy(m: dict, d: Tensor) -> Tensor:
        """``dy = -alpha*d`` under the per-sample gate, in the projector's sign convention.

        ``alpha`` and the gate are frozen constants of the proxy. Gated-out samples
        (feasible, or failed) contribute an exact zero.
        """
        dy = -m["alpha"].unsqueeze(-1) * d
        return torch.where(m["gate"].unsqueeze(-1), dy, torch.zeros_like(dy))

    def _project_step_inner_ip(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
    ) -> tuple[Tensor, dict]:
        """One damped barrier-Newton step at finite mu.

        Cold start -> ``H`` -> M-side Schur solve for ``nu^+`` -> ``d`` -> one
        fraction-to-the-boundary alpha per sample -> ``y_tilde = y + alpha d``. All algebra
        runs in float64. ``active`` is unused: every row is live.

        Returns:
            ``(y_tilde, info)`` with the cached affine map in ``info["ip_map"]``
            and ``ip_*`` telemetry.
        """
        n_batch, K, D = J.shape
        work = torch.float64
        rho = self.ip_rho
        dev = J.device

        B_all = J
        c_all = c_values
        if self.prescale:
            row_norms = B_all.norm(dim=2, keepdim=True).clamp(min=self.prescale_floor)
            B_all = B_all / row_norms
            c_all = c_all / row_norms.squeeze(-1)

        eq_idx = self._eq_idx.to(dev)
        ineq_idx = self._ineq_idx.to(dev)
        m_E = int(eq_idx.numel())
        m_I = int(ineq_idx.numel())
        has_eq = m_E > 0
        has_ineq = m_I > 0

        # Infeasibility v on the unscaled values, so mu vanishes exactly at feasibility.
        c_raw = c_values.detach().to(work)
        zero_b = torch.zeros(n_batch, device=dev, dtype=work)
        v_eq = c_raw[:, eq_idx].abs().amax(dim=-1) if has_eq else zero_b
        v_in = c_raw[:, ineq_idx].clamp(min=0).amax(dim=-1) if has_ineq else zero_b
        v = torch.maximum(v_eq, v_in)

        if self.ip_fixed_mu is None:
            mu_true = self.ip_mu0 * v
            # v == 0: no step; mu = 1 is a dummy to keep the algebra nonsingular.
            gate = v > 0
        else:
            mu_true = torch.full_like(v, self.ip_fixed_mu)
            gate = torch.ones_like(v, dtype=torch.bool)
        mu = torch.where(mu_true > 0, mu_true, torch.ones_like(mu_true))
        eps = mu.sqrt()
        mu_c = mu.unsqueeze(-1)
        eps_c = eps.unsqueeze(-1)

        c_w = c_all.detach().to(work)
        J_w = B_all.detach().to(work)
        c_eq = c_w[:, eq_idx]
        c_ineq = c_w[:, ineq_idx]
        JE = J_w[:, eq_idx, :]
        JI = J_w[:, ineq_idx, :]
        eye_d = torch.eye(D, device=dev, dtype=work)

        s0 = torch.maximum(-c_ineq, eps_c)          # lambda0 = mu/s0 => s0*lambda0 = mu
        W = mu_c / s0.pow(2)                        # <= I by eps = sqrt(mu)
        p0 = c_eq.clamp(min=0) + eps_c              # p0 - q0 = c_E
        q0 = (-c_eq).clamp(min=0) + eps_c
        sigma0 = c_eq.abs() + 2.0 * eps_c           # = p0 + q0
        D_E = sigma0 / rho

        if has_ineq:
            JI_T = JI.transpose(1, 2)
            H = eye_d.expand(n_batch, D, D) + torch.bmm(JI_T, W.unsqueeze(-1) * JI)
            Hinv, bad_H = self._ip_spd_inverse(H)
            # Centering force -2mu J_I^T s0^-11.
            k = -2.0 * mu_c * torch.bmm(JI_T, s0.reciprocal().unsqueeze(-1)).squeeze(-1)
        else:
            # No inequalities: H = I exactly.
            JI_T = JI.transpose(1, 2)
            H = eye_d.expand(n_batch, D, D)
            Hinv = H
            bad_H = torch.zeros(n_batch, dtype=torch.bool, device=dev)
            k = torch.zeros(n_batch, D, device=dev, dtype=work)

        JE_T = JE.transpose(1, 2)
        if has_eq:
            M = torch.bmm(JE, torch.bmm(Hinv, JE_T)) + torch.diag_embed(D_E)
            Minv, bad_M = self._ip_spd_inverse(M)
        else:
            Minv = M = torch.zeros(n_batch, 0, 0, device=dev, dtype=work)
            bad_M = torch.zeros(n_batch, dtype=torch.bool, device=dev)

        ip_map: dict = {
            "work_dtype": work,
            "has_eq": has_eq,
            "has_ineq": has_ineq,
            "eq_idx": eq_idx,
            "ineq_idx": ineq_idx,
            "Hinv": Hinv,
            "Minv": Minv,
            "JE": JE,
            "JE_T": JE_T,
            "JI_T": JI_T,
            "W": W,
            "k": k,
        }
        d_full, nu = self._ip_newton_d(ip_map, c_eq, c_ineq)

        # Newton residual of the two condensed blocks (health check).
        r1 = (
            k - torch.bmm(JI_T, (W * c_ineq).unsqueeze(-1)).squeeze(-1)
            if has_ineq
            else k
        )
        res1 = (
            torch.bmm(H, d_full.unsqueeze(-1)).squeeze(-1)
            + torch.bmm(JE_T, nu.unsqueeze(-1)).squeeze(-1)
            - r1
        )
        res2 = (
            torch.bmm(JE, d_full.unsqueeze(-1)).squeeze(-1) - D_E * nu + c_eq
            if has_eq
            else torch.zeros(n_batch, 0, device=dev, dtype=work)
        )

        ds = -(c_ineq + s0) - torch.bmm(JI, d_full.unsqueeze(-1)).squeeze(-1)
        dp = (mu_c - rho * p0) / rho + (p0 / rho) * nu
        dq = (mu_c - rho * q0) / rho - (q0 / rho) * nu
        alpha = torch.ones(n_batch, device=dev, dtype=work)
        for x0, dx in ((s0, ds), (p0, dp), (q0, dq)):
            if x0.shape[-1] == 0:
                continue
            ratio = torch.where(
                dx < 0, x0 / (-dx), torch.full_like(dx, float("inf"))
            )
            alpha = torch.minimum(alpha, _IP_FTB * ratio.amin(dim=-1))
        alpha = alpha.clamp(max=1.0)

        # A non-finite step gives Delta = 0 for that sample.
        finite = torch.isfinite(d_full).all(dim=-1) & torch.isfinite(alpha)
        failed = gate & ~finite
        n_failures = int(failed.sum().item())
        n_pinv = int((bad_H | bad_M).sum().item())
        gate = gate & finite
        if n_failures > 0:
            # Zero the cached map too: a NaN in Hinv would poison the backward pass.
            Hinv = Hinv.clone()
            Hinv[failed] = 0.0
            ip_map["Hinv"] = Hinv
            if has_eq:
                Minv = Minv.clone()
                Minv[failed] = 0.0
                ip_map["Minv"] = Minv
            k = k.clone()
            k[failed] = 0.0
            ip_map["k"] = k
            if has_ineq:
                W = W.clone()
                W[failed] = 0.0
                ip_map["W"] = W
            alpha = torch.where(finite, alpha, torch.ones_like(alpha))
        ip_map["alpha"] = alpha
        ip_map["gate"] = gate

        self.ip_solve_failures_total += n_failures
        self.ip_pinv_fallbacks_total += n_pinv
        if n_failures > 0:
            print(
                f"[pal.projector] ip: Newton solve FAILED on {n_failures}/"
                f"{n_batch} samples. Those samples take Delta=0 (y_tilde=y_hat, zero proxy "
                "contribution) and are counted in ip/solve_failures. No lm_k "
                "fallback is applied.",
                flush=True,
            )

        dy_w = self._ip_damped_dy(ip_map, d_full)
        y_tilde = y - dy_w.to(y.dtype)

        gate_f = gate.to(work)
        n_gated = float(gate_f.sum().item())

        def _masked_max(t: Tensor) -> float:
            # torch.where, not a multiply: 0 * NaN would leak into telemetry.
            if t.shape[-1] == 0 or n_gated == 0.0:
                return 0.0
            keep = gate.unsqueeze(-1).expand_as(t)
            return float(torch.where(keep, t.abs(), torch.zeros_like(t)).max().item())

        # alpha stats only over samples that take a step.
        if n_gated > 0.0:
            alpha_kept = alpha[gate]
            alpha_mean = float(alpha_kept.mean().item())
            alpha_min = float(alpha_kept.min().item())
        else:
            alpha_mean = alpha_min = 1.0

        info = {
            "method": "ip",
            "ip_map": ip_map,
            "ip_mu": mu_true,
            "ip_alpha": alpha,
            "ip_gate": gate,
            "ip_nu": nu,
            "ip_s0": s0,
            "ip_ds": ds,
            "ip_p0": p0,
            "ip_dp": dp,
            "ip_q0": q0,
            "ip_dq": dq,
            "ip_solve_failures": n_failures,
            "ip_pinv_fallbacks": n_pinv,
            "ip_alpha_mean": alpha_mean,
            "ip_alpha_min": alpha_min,
            "ip_mu_mean": float(mu_true.mean().item()),
            "ip_nu_abs_max": _masked_max(nu),
            "ip_newton_residual_max": max(_masked_max(res1), _masked_max(res2)),
        }
        if self.prescale:
            info["row_norms"] = row_norms.squeeze(-1)
        return y_tilde, info

    def step(
        self,
        y: Tensor,
        c_pre: Tensor,
        values_fn: Callable,
        conditions: Tensor | None,
    ) -> tuple[Tensor, dict]:
        """Single projection step with a live-gradient displacement proxy.

        Returns ``(y_tilde_detached, info)``. ``info["displacement_live"]``
        and ``info["c_pre"]`` carry gradients back to ``y -> theta`` for the
        training loss; ``info["c_post"]`` is the detached post-projection
        constraint vector for multiplier updates.

        Args:
            y: live input, typically the NN output. Gradients flow through
                here to the model parameters on ``loss.backward()``.
            c_pre: ``[B, K]`` constraint values at ``y`` with a live
                autograd graph; J is built from it by K VJPs.
            values_fn: ``(y, conds) -> Tensor[B, K]`` constraint evaluator,
                used for the post-projection values.
            conditions: optional condition tensor passed to ``values_fn``.
        """
        if self.jacobian_mode == "vmap_jacrev":
            J = self._jacobian_vmap_jacrev(y, values_fn, conditions)
        else:
            J = self._jacobian_from_live_c(y, c_pre)

        active = self._build_active_mask(c_pre)
        jac_row_norms = J.norm(dim=2)

        y_tilde, eig_info = self._project_step_inner(y, c_pre, J, active)
        y_tilde = self._clamp_to_box(y_tilde)

        c_post = values_fn(y_tilde.detach(), conditions).detach()

        c_scaled = c_pre
        if "row_norms" in eig_info:
            c_scaled = c_scaled / eig_info["row_norms"].clamp(min=self.prescale_floor)
        # sqp masks by the QP's tight set, ip masks nothing, eigh/lm_k by `active`.
        is_sqp = eig_info["method"] == "sqp"
        is_ip = eig_info["method"] == "ip"
        row_mask = (
            eig_info["tight_mask"].to(c_scaled.dtype)
            if is_sqp
            else torch.ones_like(c_scaled)
            if is_ip
            else active.to(c_scaled.dtype)
        )
        c_pre_masked = c_scaled * row_mask

        if is_ip:
            # Re-apply the cached affine map to the live c_pre.
            ip_map = eig_info["ip_map"]
            c_live = c_pre_masked.to(ip_map["work_dtype"])
            d_live, _ = self._ip_newton_d(
                ip_map,
                c_live[:, ip_map["eq_idx"]],
                c_live[:, ip_map["ineq_idx"]],
            )
            displacement_live = self._ip_damped_dy(ip_map, d_live).to(c_pre.dtype)
            y_tilde_live = self._clamp_to_box(y - displacement_live)
            displacement_live = y_tilde_live - y
            info = {
                "c_pre": c_pre,
                "c_post": c_post,
                "J": J,
                "active_mask": active,
                "displacement": (y_tilde - y).detach(),
                "displacement_live": displacement_live,
                "jac_row_norms": jac_row_norms,
            }
            for key in (
                "ip_solve_failures",
                "ip_pinv_fallbacks",
                "ip_alpha_mean",
                "ip_alpha_min",
                "ip_mu_mean",
                "ip_nu_abs_max",
                "ip_newton_residual_max",
            ):
                info[key] = eig_info[key]
            return y_tilde, info

        B_mat = eig_info["B_mat"]
        if eig_info["method"] == "eigh":
            U = eig_info["U"]
            D_reg = eig_info["D_reg"]
            Ut_c = U.transpose(-1, -2) @ c_pre_masked.unsqueeze(-1)
            inv_BBT_c = U @ (D_reg.unsqueeze(-1) * Ut_c)
        elif is_sqp:
            # Cached pseudo-inverse applied to the live c_T, in float64.
            gram_pinv = eig_info["gram_pinv"]
            rhs = (
                c_pre_masked.to(gram_pinv.dtype) - eig_info["rhs_offset"]
            ) * row_mask.to(gram_pinv.dtype)
            inv_BBT_c = torch.bmm(gram_pinv, rhs.unsqueeze(-1))
        else:
            L = eig_info["L"]
            inv_BBT_c = torch.cholesky_solve(c_pre_masked.unsqueeze(-1), L)
        displacement_live = torch.bmm(
            B_mat.transpose(1, 2), inv_BBT_c
        ).squeeze(-1)
        if is_sqp:
            displacement_live = (
                displacement_live + eig_info["disp_elastic"]
            ).to(c_pre.dtype)

        y_tilde_live = self._clamp_to_box(y - displacement_live)
        displacement_live = y_tilde_live - y

        info = {
            "c_pre": c_pre,
            "c_post": c_post,
            "J": J,
            "active_mask": active,
            "displacement": (y_tilde - y).detach(),
            "displacement_live": displacement_live,
            "jac_row_norms": jac_row_norms,
        }
        if is_sqp:
            info["tight_mask"] = eig_info["tight_mask"]
            info["sqp_qp_failures"] = eig_info["sqp_qp_failures"]
            info["sqp_clarabel_rescues"] = eig_info["sqp_clarabel_rescues"]
            info["sqp_partition_mismatches"] = eig_info["sqp_partition_mismatches"]
            info["sqp_resolve_max_gap"] = eig_info["sqp_resolve_max_gap"]
        return y_tilde, info

    def project(
        self,
        y: Tensor,
        constraint_fn: Callable,
        conditions: Tensor | None,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> tuple[Tensor, dict]:
        """Multi-step projection used at inference (no gradient through loop).

        Under ``method="sqp"`` / ``"ip"`` the returned dict also carries the
        solver counters summed over the iterations of this call.
        """
        n_failures = 0
        n_rescues = 0
        n_mismatches = 0
        n_ip_failures = 0
        n_ip_pinv = 0

        def _out(iters: int, max_res: float) -> dict:
            info: dict = {"iters": iters, "max_residual": max_res}
            if self.method == "sqp":
                info["sqp_qp_failures"] = n_failures
                info["sqp_clarabel_rescues"] = n_rescues
                info["sqp_partition_mismatches"] = n_mismatches
            if self.method == "ip":
                info["ip_solve_failures"] = n_ip_failures
                info["ip_pinv_fallbacks"] = n_ip_pinv
            return info

        for i in range(max_iters):
            c_values, J = self._jacobian_at_detached_y(y, constraint_fn, conditions)
            active = self._build_active_mask(c_values)

            residuals = self._compute_residuals(c_values, active)
            max_res = residuals.max().item()
            if max_res < tol:
                return y, _out(i, max_res)

            y, step_info = self._project_step_inner(y, c_values, J, active)
            n_failures += int(step_info.get("sqp_qp_failures", 0))
            n_rescues += int(step_info.get("sqp_clarabel_rescues", 0))
            n_mismatches += int(step_info.get("sqp_partition_mismatches", 0))
            n_ip_failures += int(step_info.get("ip_solve_failures", 0))
            n_ip_pinv += int(step_info.get("ip_pinv_fallbacks", 0))
            y = self._clamp_to_box(y)

        c_final, _ = self._jacobian_at_detached_y(y, constraint_fn, conditions)
        active_final = self._build_active_mask(c_final)
        final_res = self._compute_residuals(c_final, active_final).max().item()
        return y, _out(max_iters, final_res)

    def _compute_residuals(self, c_values: Tensor, active: Tensor) -> Tensor:
        """Per-constraint residuals. Ineq: `max(g, 0)`; eq: `|h|`."""
        is_eq = torch.tensor(
            self._is_eq, device=c_values.device, dtype=c_values.dtype
        )
        ineq_r = c_values.clamp(min=0) * (1 - is_eq)
        eq_r = c_values.abs() * is_eq
        return (ineq_r + eq_r) * active.to(c_values.dtype)

    def project_trace(
        self,
        y: Tensor,
        constraint_fn: Callable,
        conditions: Tensor | None,
        objective_fn: Callable[[Tensor, Tensor | None], Tensor] | None = None,
        max_iters: int = 10,
        tol: float = 1e-6,
    ) -> tuple[Tensor, list, bool]:
        """Multi-step projection returning per-iter `(c_values, y, obj?)`.

        Caller converts the raw tuples into `ProjectionStep` dataclasses.

        Returns:
            `y_final`, list of `(iter_idx, c_values [B,K], y [B,D])` tuples
            including the pre-projection state (iter=0) and the post-step
            states, and a `converged` bool.

        """
        trace: list[tuple[int, Tensor, Tensor]] = []
        converged = False
        y_cur = y
        c_values, J = self._jacobian_at_detached_y(y_cur, constraint_fn, conditions)
        trace.append((0, c_values.detach().clone(), y_cur.detach().clone()))

        for i in range(max_iters):
            active = self._build_active_mask(c_values)
            residuals = self._compute_residuals(c_values, active)
            if residuals.max().item() < tol:
                converged = True
                break
            y_cur, _ = self._project_step_inner(y_cur, c_values, J, active)
            y_cur = self._clamp_to_box(y_cur)
            c_values, J = self._jacobian_at_detached_y(y_cur, constraint_fn, conditions)
            trace.append((i + 1, c_values.detach().clone(), y_cur.detach().clone()))

        return y_cur, trace, converged
