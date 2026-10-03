"""AdaNP: Adaptive-depth Neural Projection (Lastrucci & Schweidtmann, arXiv:2502.06774).

Gradient flows through the `y - B^T (B B^T + eps I)^{-1} (B y - c)` step, the Jacobian
is detached.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch import Tensor


class AdaNP:
    """Iteratively project onto active constraints via eigh-regularized solve.

    For ineq: drives `c + margin -> 0` (i.e. `c -> -margin`).
    For eq:   drives `c -> 0` (margins are 0 for eq slots).
    """

    def __init__(
        self,
        n_constraints: int,
        constraint_types: list[str],
        max_iters: int = 10,
        tol: float = 1e-6,
        delta: float = 1e-3,
        adaptive_delta: bool = False,
        prescale: bool = True,
        use_eigh: bool = True,
        eps: float = 1e-6,
    ):
        assert len(constraint_types) == n_constraints
        self.n_constraints = n_constraints
        self.constraint_types = constraint_types
        self.max_iters = max_iters
        self.tol = tol
        self.delta = delta
        self.adaptive_delta = adaptive_delta
        self.prescale = prescale
        self.use_eigh = use_eigh
        self.eps = eps
        self._is_eq = [t == "eq" for t in constraint_types]
        self._last_iters = 0
        self._last_max_residual = 0.0

    def _compute_jacobian(
        self,
        y: Tensor,
        raw_constraint_fn: Callable,
        conditions: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        B, D = y.shape
        K = self.n_constraints

        y_det = y.detach().requires_grad_(True)
        _, constraint_list = raw_constraint_fn(y_det, conditions)
        c_values = torch.stack([c.value for c in constraint_list], dim=-1)

        zeros = torch.zeros(B, device=y.device, dtype=y.dtype)
        margins = torch.stack(
            [c.margin if not self._is_eq[k] else zeros for k, c in enumerate(constraint_list)],
            dim=-1,
        )
        tols = torch.stack(
            [c.tol if self._is_eq[k] else zeros for k, c in enumerate(constraint_list)],
            dim=-1,
        )

        J = torch.zeros(B, K, D, device=y.device, dtype=y.dtype)
        for k in range(K):
            g = torch.autograd.grad(
                c_values[:, k].sum(),
                y_det,
                retain_graph=(k < K - 1),
            )[0]
            J[:, k, :] = g

        return c_values.detach(), J.detach(), margins.detach(), tols.detach()

    def _build_active_mask(
        self, c_values: Tensor, margins: Tensor, tols: Tensor
    ) -> Tensor:
        is_eq = torch.tensor(self._is_eq, device=c_values.device)
        ineq_active = (c_values + margins) > 0
        eq_active = c_values.abs() > tols
        return torch.where(is_eq.unsqueeze(0), eq_active, ineq_active)

    @staticmethod
    def _regularized_solve(
        U: Tensor, D_reg: Tensor, rhs: Tensor
    ) -> Tensor:
        is_vec = rhs.dim() == 2
        if is_vec:
            rhs = rhs.unsqueeze(-1)
        Ut_rhs = U.transpose(-1, -2) @ rhs
        result = U @ (D_reg.unsqueeze(-1) * Ut_rhs)
        if is_vec:
            result = result.squeeze(-1)
        return result

    def _project_step(
        self,
        y: Tensor,
        c_values: Tensor,
        J: Tensor,
        active: Tensor,
        margins: Tensor,
    ) -> Tensor:
        B, K, D = J.shape
        mask_float = active.float().unsqueeze(-1)
        B_mat = J * mask_float
        c_masked = (c_values + margins) * active.float()

        if not self.use_eigh:
            return self._project_step_solve(y, B_mat, c_masked, K, D)

        if self.prescale:
            row_norms = B_mat.norm(dim=2, keepdim=True).clamp(min=1e-12)
            B_mat = B_mat / row_norms
            c_masked = c_masked / row_norms.squeeze(-1)

        BBT = torch.bmm(B_mat, B_mat.transpose(1, 2))
        eigenvalues, U = torch.linalg.eigh(BBT)
        lam_max = eigenvalues[..., -1:].clamp(min=1e-12)

        if self.adaptive_delta:
            c_norm = c_masked.norm(dim=1, keepdim=True)
            delta_k = torch.clamp(c_norm, max=self.delta)
        else:
            delta_k = self.delta

        eps = delta_k * lam_max
        D_reg = 1.0 / (eigenvalues + eps)

        By = torch.bmm(B_mat, y.unsqueeze(-1)).squeeze(-1)
        v = By - c_masked

        inv_BBT_B = self._regularized_solve(U, D_reg, B_mat)
        inv_BBT_v = self._regularized_solve(U, D_reg, v)

        proj_By = torch.bmm(inv_BBT_B, y.unsqueeze(-1)).squeeze(-1)
        term1 = y - torch.bmm(B_mat.transpose(1, 2), proj_By.unsqueeze(-1)).squeeze(-1)

        term2 = torch.bmm(B_mat.transpose(1, 2), inv_BBT_v.unsqueeze(-1)).squeeze(-1)

        return term1 + term2

    def _project_step_solve(
        self,
        y: Tensor,
        B_mat: Tensor,
        c_masked: Tensor,
        K: int,
        D: int,
    ) -> Tensor:
        BBT = torch.bmm(B_mat, B_mat.transpose(1, 2))
        BBT = BBT + self.eps * torch.eye(K, device=y.device, dtype=y.dtype).unsqueeze(0)

        By = torch.bmm(B_mat, y.unsqueeze(-1)).squeeze(-1)
        v = By - c_masked

        inv_BBT_B = torch.linalg.solve(BBT, B_mat)
        inv_BBT_v = torch.linalg.solve(BBT, v.unsqueeze(-1)).squeeze(-1)

        proj_By = torch.bmm(inv_BBT_B, y.unsqueeze(-1)).squeeze(-1)
        term1 = y - torch.bmm(B_mat.transpose(1, 2), proj_By.unsqueeze(-1)).squeeze(-1)

        term2 = torch.bmm(B_mat.transpose(1, 2), inv_BBT_v.unsqueeze(-1)).squeeze(-1)

        return term1 + term2

    def project(
        self,
        y_hat: Tensor,
        raw_constraint_fn: Callable,
        conditions: Tensor | None,
    ) -> tuple[Tensor, dict]:
        is_eq_t = torch.tensor(self._is_eq, dtype=torch.float32)

        def _active_residuals(c: Tensor, margins: Tensor, tols: Tensor, active: Tensor) -> Tensor:
            ineq_r = (c + margins).clamp(min=0) * (1 - is_eq_t.to(c.device))
            eq_r = c.abs() * is_eq_t.to(c.device)
            return (ineq_r + eq_r) * active.float()

        y = y_hat
        for i in range(self.max_iters):
            c_values, J, margins, tols = self._compute_jacobian(y, raw_constraint_fn, conditions)
            active = self._build_active_mask(c_values, margins, tols)

            active_residuals = _active_residuals(c_values, margins, tols, active)
            max_residual = active_residuals.max().item()
            if max_residual < self.tol:
                self._last_iters = i
                self._last_max_residual = max_residual
                return y, {"iters": i, "max_residual": max_residual}

            y = self._project_step(y, c_values, J, active, margins)

        c_final, _, margins_f, tols_f = self._compute_jacobian(y, raw_constraint_fn, conditions)
        active_final = self._build_active_mask(c_final, margins_f, tols_f)
        final_residual = _active_residuals(c_final, margins_f, tols_f, active_final).max().item()
        self._last_iters = self.max_iters
        self._last_max_residual = final_residual
        return y, {"iters": self.max_iters, "max_residual": final_residual}

    def log_dict(self) -> dict:
        return {
            "adanp/iters": self._last_iters,
            "adanp/max_residual": self._last_max_residual,
        }
