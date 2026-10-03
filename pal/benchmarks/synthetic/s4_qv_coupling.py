"""S4 two-block coupling `A u = A_tilde c`, `v = sin(u) + B c`, `||v||^2 <= gamma(c)`, f on v.

`y_star(c) = (u* = A^+ A_tilde c, v* = sin(u*) + Bc)`; ineq slack is +1.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 30  # u (15) + v (15)
_U_DIM = 15
_V_DIM = 15
_COND_DIM = 5
_C_LO = 0.5
_C_HI = 1.5
_N_EQ_MAIN = 10
_EVAL_SALT = 2104
_CTOR_SEED = 2104


def _make_spec() -> BenchmarkSpec:
    # y*(c) components reach [-3.74, +4.12], so +/-5 keeps them inside the tanh head's range.
    bounds_lo = torch.full((_DIM,), -5.0)
    bounds_hi = torch.full((_DIM,), 5.0)
    names = (
        [f"main_{i}" for i in range(_N_EQ_MAIN)]
        + [f"coupling_{i}" for i in range(_V_DIM)]
        + ["v_norm_bound"]
    )
    types = ["eq"] * _N_EQ_MAIN + ["eq"] * _V_DIM + ["ineq"]
    return BenchmarkSpec(
        id="s4_qv_coupling",
        family="s4",
        variant=None,
        dim=_DIM,
        n_eq=_N_EQ_MAIN + _V_DIM,
        n_ineq=1,
        constraint_names=names,
        constraint_types=types,
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=_COND_DIM,
        zeta_dim=0,
        tolerance=1e-4,
        tau=1e-4,
        cost="cheap",
        recommended_device="cpu",
        train_batch_size=256,
        n_eval_default=64,
        notes="two-block Q-V coupling; v = sin(u) + Bc; fixed-SPD weighted quad obj",
    )


def _make_spd(dim: int, cond: float, g: torch.Generator) -> Tensor:
    """Build SPD matrix with specified condition number."""
    M = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(M)
    eigs = torch.linspace(1.0, cond, dim)
    return Q @ torch.diag(eigs) @ Q.T


class S4QvCoupling:
    def __init__(self) -> None:
        self.spec = _make_spec()
        g = torch.Generator().manual_seed(_CTOR_SEED)

        # A in R^{10 x 15} rank-10, rows unit-normed
        A = torch.randn(_N_EQ_MAIN, _U_DIM, generator=g)
        A = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self._A = A
        self._A_pinv = torch.linalg.pinv(A)  # [15, 10]

        self._A_tilde = torch.randn(_N_EQ_MAIN, _COND_DIM, generator=g) * 0.5

        self._B = torch.randn(_V_DIM, _COND_DIM, generator=g) * 0.3

        # W_v in R^{15 x 15} SPD, cond ~ 5 (operates on v block only)
        self._W = _make_spd(_V_DIM, cond=5.0, g=g)

    def _u_star(self, c: Tensor) -> Tensor:
        """u*(c) = A^+ A_tilde c: [B, 15]."""
        A_pinv = self._A_pinv.to(device=c.device, dtype=c.dtype)
        A_tilde = self._A_tilde.to(device=c.device, dtype=c.dtype)
        return c @ A_tilde.T @ A_pinv.T

    def _v_star(self, c: Tensor, u_star: Tensor) -> Tensor:
        """v*(c) = sin(u*) + B c: [B, 15]."""
        B_mat = self._B.to(device=c.device, dtype=c.dtype)
        return torch.sin(u_star) + c @ B_mat.T

    def _y_star(self, c: Tensor) -> Tensor:
        u_star = self._u_star(c)
        v_star = self._v_star(c, u_star)
        return torch.cat([u_star, v_star], dim=-1)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        u = x[:, :_U_DIM]
        v = x[:, _U_DIM:]

        u_star = self._u_star(conditions)
        v_star = self._v_star(conditions, u_star)

        diff_v = v - v_star
        W = self._W.to(device=x.device, dtype=x.dtype)
        obj = 0.5 * (diff_v @ W * diff_v).sum(dim=-1)

        # eq main: A u - A_tilde c = 0
        A = self._A.to(device=x.device, dtype=x.dtype)
        A_tilde = self._A_tilde.to(device=x.device, dtype=x.dtype)
        main_val = u @ A.T - conditions @ A_tilde.T  # [B, 10]

        # eq coupling: v - sin(u) - B c = 0
        B_mat = self._B.to(device=x.device, dtype=x.dtype)
        coupling_val = v - torch.sin(u) - conditions @ B_mat.T  # [B, 15]

        # ineq: ||v||^2 - gamma(c) <= 0, gamma = ||v*||^2 + 1
        gamma = (v_star * v_star).sum(dim=-1) + 1.0  # [B]
        ineq_val = (v * v).sum(dim=-1) - gamma  # [B]

        B = x.shape[0]
        device = x.device
        tol_eq = torch.full((B,), 1e-3, device=device)
        margin_zero = torch.zeros(B, device=device)

        constraints = []
        for i in range(_N_EQ_MAIN):
            constraints.append(
                Constraint(value=main_val[:, i], type="eq", tol=tol_eq, margin=margin_zero, name=f"main_{i}")
            )
        for i in range(_V_DIM):
            constraints.append(
                Constraint(value=coupling_val[:, i], type="eq", tol=tol_eq, margin=margin_zero, name=f"coupling_{i}")
            )
        constraints.append(
            Constraint(
                value=ineq_val, type="ineq",
                tol=torch.zeros(B, device=device),
                margin=torch.full((B,), 0.1, device=device),
                name="v_norm_bound",
            )
        )
        return obj, constraints

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        assert conditions is not None
        u = x[:, :_U_DIM]
        v = x[:, _U_DIM:]
        u_star = self._u_star(conditions)
        v_star = self._v_star(conditions, u_star)
        A = self._A.to(device=x.device, dtype=x.dtype)
        A_tilde = self._A_tilde.to(device=x.device, dtype=x.dtype)
        B_mat = self._B.to(device=x.device, dtype=x.dtype)
        main_val = u @ A.T - conditions @ A_tilde.T
        coupling_val = v - torch.sin(u) - conditions @ B_mat.T
        gamma = (v_star * v_star).sum(dim=-1) + 1.0
        ineq_val = (v * v).sum(dim=-1) - gamma
        return torch.cat([main_val, coupling_val, ineq_val.unsqueeze(-1)], dim=-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def _sample_c(self, n: int, g: torch.Generator) -> Tensor:
        return _C_LO + (_C_HI - _C_LO) * torch.rand(n, _COND_DIM, generator=g)

    def sample_queries(
        self, n: int, split: Literal["train", "eval"], seed: int
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        c = self._sample_c(n, g)
        zeta = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=c)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + _EVAL_SALT)
        c = self._sample_c(n, g)
        zeta = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=c)

    def y_star(self, conditions: Tensor) -> Tensor:
        return self._y_star(conditions)

    def check_env(self) -> None:
        return None

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None
