"""S6 15 coupling eqs `y[i] - alpha*y[i+15] = const(c)`, 75 redundant ineqs, f on dims 15..49.

`y*(c) = t(c)` with `t(c)_i = c_{i mod 20}` satisfies all constraints with f(y*) = 0.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 50
_COND_DIM = 20
_C_LO = 0.5
_C_HI = 1.5
_N_REAL = 25
_N_REDUNDANT = 50
_N_INEQ = _N_REAL + _N_REDUNDANT
_N_COUPLING_EQ = 15  # round 3: 15 coupling eqs y[i] - alpha*y[i+15] = const(c)
_COUPLING_ALPHA = 0.5
_F_HEAD = 15  # round 3: f starts at dim 15; dims 0..14 are not in f
_EVAL_SALT = 2106
_CTOR_SEED = 2106
_RHS_SLACK = 0.2


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((_DIM,), 0.3)
    bounds_hi = torch.full((_DIM,), 1.7)
    names = (
        [f"couple_{i}" for i in range(_N_COUPLING_EQ)]
        + [f"real_{i}" for i in range(_N_REAL)]
        + [f"redundant_{i}" for i in range(_N_REDUNDANT)]
    )
    types = ["eq"] * _N_COUPLING_EQ + ["ineq"] * _N_INEQ
    return BenchmarkSpec(
        id="s6_redundant_ineq",
        family="s6",
        variant=None,
        dim=_DIM,
        n_eq=_N_COUPLING_EQ,
        n_ineq=_N_INEQ,
        constraint_names=names,
        constraint_types=types,
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=_COND_DIM,
        zeta_dim=0,
        tolerance=1e-4,
        tau=1e-4,
        cost="mid",
        recommended_device="cpu",
        train_batch_size=64,
        n_eval_default=64,
        notes="round 3: 15 coupling eqs + 25 real + 50 near-redundant ineqs; f on tail only",
    )


class S6RedundantIneq:
    def __init__(self) -> None:
        self.spec = _make_spec()
        g = torch.Generator().manual_seed(_CTOR_SEED)

        a_real = torch.randn(_N_REAL, _DIM, generator=g)
        a_real = a_real / a_real.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        parent_idx = torch.randint(0, _N_REAL, (_N_REDUNDANT,), generator=g)
        perturb = 1e-4 * torch.randn(_N_REDUNDANT, _DIM, generator=g)
        a_red = a_real[parent_idx] + perturb
        a_red = a_red / a_red.norm(dim=-1, keepdim=True).clamp(min=1e-12)

        self._A = torch.cat([a_real, a_red], dim=0)

    @staticmethod
    def _t_of_c(c: Tensor) -> Tensor:
        """t(c)_i = c_{i mod 20}."""
        idx = torch.arange(_DIM, device=c.device) % _COND_DIM
        return c[:, idx]

    @staticmethod
    def _coupling_rhs(c: Tensor) -> Tensor:
        """RHS of coupling eqs: c[i mod 20] - alpha*c[(i+15) mod 20] for i = 0..14."""
        idx_lo = torch.arange(_N_COUPLING_EQ, device=c.device) % _COND_DIM
        idx_hi = (torch.arange(_N_COUPLING_EQ, device=c.device) + _F_HEAD) % _COND_DIM
        return c[:, idx_lo] - _COUPLING_ALPHA * c[:, idx_hi]

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        t = self._t_of_c(conditions)
        diff_tail = x[:, _F_HEAD:] - t[:, _F_HEAD:]
        obj = 0.5 * (diff_tail * diff_tail).sum(dim=-1)

        # 15 coupling eqs: y[i] - alpha*y[i+_F_HEAD] = const(c)
        rhs = self._coupling_rhs(conditions)
        couple_resid = x[:, :_N_COUPLING_EQ] - _COUPLING_ALPHA * x[:, _F_HEAD:_F_HEAD + _N_COUPLING_EQ] - rhs

        A = self._A.to(device=x.device, dtype=x.dtype)
        a_y = x @ A.T
        a_t = t @ A.T
        ineq_val = a_y - a_t - _RHS_SLACK

        B = x.shape[0]
        device = x.device
        tol_eq = torch.full((B,), 1e-3, device=device)
        tol_zero = torch.zeros(B, device=device)
        margin_zero = torch.zeros(B, device=device)
        margin_ineq = torch.full((B,), 0.1, device=device)
        constraints = []
        for i in range(_N_COUPLING_EQ):
            constraints.append(
                Constraint(
                    value=couple_resid[:, i], type="eq", tol=tol_eq, margin=margin_zero,
                    name=f"couple_{i}",
                )
            )
        for i in range(_N_INEQ):
            name = f"real_{i}" if i < _N_REAL else f"redundant_{i - _N_REAL}"
            constraints.append(
                Constraint(
                    value=ineq_val[:, i], type="ineq", tol=tol_zero, margin=margin_ineq,
                    name=name,
                )
            )
        return obj, constraints

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        assert conditions is not None
        t = self._t_of_c(conditions)
        rhs = self._coupling_rhs(conditions)
        couple_resid = x[:, :_N_COUPLING_EQ] - _COUPLING_ALPHA * x[:, _F_HEAD:_F_HEAD + _N_COUPLING_EQ] - rhs
        A = self._A.to(device=x.device, dtype=x.dtype)
        a_y = x @ A.T
        a_t = t @ A.T
        ineq_val = a_y - a_t - _RHS_SLACK
        return torch.cat([couple_resid, ineq_val], dim=-1)

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
        return self._t_of_c(conditions)

    def check_env(self) -> None:
        return None

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None
