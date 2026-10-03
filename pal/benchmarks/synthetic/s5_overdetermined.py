"""S5 overdetermined: 5 mutually consistent equalities on `y in R^4` (DC3-inapplicable).

`y*(c) = (c0 cos phi1, c0 sin phi1, c1 cos phi2, c1 sin phi2)`,
`phi1 = c2 pi`, `phi2 = c3 pi + c4 pi/2`.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 4
_COND_DIM = 5
_N_EQ = 5
_C_LO = 0.5
_C_HI = 1.5
_EVAL_SALT = 2105
_CTOR_SEED = 2105


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((_DIM,), -2.0)
    bounds_hi = torch.full((_DIM,), 2.0)
    names = ["circle_0", "circle_1", "product_02", "product_13", "cross_diff"]
    types = ["eq"] * _N_EQ
    return BenchmarkSpec(
        id="s5_overdetermined",
        family="s5",
        variant=None,
        dim=_DIM,
        n_eq=_N_EQ,
        n_ineq=0,
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
        notes="5 eqs on 4 vars: DC3 structurally inapplicable (m > n)",
    )


def _make_spd(dim: int, cond: float, g: torch.Generator) -> Tensor:
    M = torch.randn(dim, dim, generator=g)
    Q, _ = torch.linalg.qr(M)
    eigs = torch.linspace(1.0, cond, dim)
    return Q @ torch.diag(eigs) @ Q.T


class S5Overdetermined:
    def __init__(self) -> None:
        self.spec = _make_spec()
        g = torch.Generator().manual_seed(_CTOR_SEED)
        self._P = _make_spd(_DIM, cond=3.0, g=g)

    @staticmethod
    def _y_star_static(c: Tensor) -> Tensor:
        phi1 = c[:, 2] * math.pi
        phi2 = c[:, 3] * math.pi + c[:, 4] * (math.pi / 2.0)
        c0 = c[:, 0]
        c1 = c[:, 1]
        y_star = torch.stack(
            [c0 * torch.cos(phi1), c0 * torch.sin(phi1),
             c1 * torch.cos(phi2), c1 * torch.sin(phi2)],
            dim=-1,
        )
        return y_star

    def _y_star(self, c: Tensor) -> Tensor:
        return self._y_star_static(c)

    def _eq_values(self, x: Tensor, y_star: Tensor, c: Tensor) -> Tensor:
        """Return [B, 5] raw eq values. Zero at x = y_star by construction."""
        y0, y1, y2, y3 = x[:, 0], x[:, 1], x[:, 2], x[:, 3]
        ys0, ys1, ys2, ys3 = y_star[:, 0], y_star[:, 1], y_star[:, 2], y_star[:, 3]

        c0 = c[:, 0]
        c1 = c[:, 1]

        eq1 = y0 * y0 + y1 * y1 - c0 * c0
        eq2 = y2 * y2 + y3 * y3 - c1 * c1
        alpha = ys0 * ys2
        eq3 = y0 * y2 - alpha
        beta = ys1 * ys3
        eq4 = y1 * y3 - beta
        delta = ys0 * ys3 - ys1 * ys2
        eq5 = y0 * y3 - y1 * y2 - delta

        return torch.stack([eq1, eq2, eq3, eq4, eq5], dim=-1)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        y_star = self._y_star(conditions)
        diff = x - y_star
        P = self._P.to(device=x.device, dtype=x.dtype)
        obj = 0.5 * (diff @ P * diff).sum(dim=-1)

        eq_vals = self._eq_values(x, y_star, conditions)

        B = x.shape[0]
        device = x.device
        tol_eq = torch.full((B,), 1e-3, device=device)
        margin_zero = torch.zeros(B, device=device)
        constraints = []
        names = self.spec.constraint_names
        for i in range(_N_EQ):
            constraints.append(
                Constraint(
                    value=eq_vals[:, i], type="eq", tol=tol_eq, margin=margin_zero,
                    name=names[i],
                )
            )
        return obj, constraints

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        assert conditions is not None
        y_star = self._y_star(conditions)
        return self._eq_values(x, y_star, conditions)

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
