"""S1 sphere `||y||^2 = c^2` with shifted Rosenbrock on (y0, y1) only.

The unique feasible zero of f is y* = (c/sqrt(2), c/sqrt(2), 0, 0).
"""

from __future__ import annotations

import math as _math
from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 4
_COND_DIM = 1
_C_LO = 0.5
_C_HI = 2.0
_EVAL_SALT = 2101

_ROSEN_A = 1.0
_ROSEN_B = 100.0

# y*(c) = c * (1/sqrt(2), 1/sqrt(2), 0, 0); the shift puts Rosenbrock's min there.
_INV_SQRT2 = 1.0 / _math.sqrt(2.0)


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((_DIM,), -2.5)
    bounds_hi = torch.full((_DIM,), 2.5)
    return BenchmarkSpec(
        id="s1_sphere_track",
        family="s1",
        variant=None,
        dim=_DIM,
        n_eq=1,
        n_ineq=0,
        constraint_names=["sphere"],
        constraint_types=["eq"],
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=_COND_DIM,
        zeta_dim=0,
        tolerance=1e-4,
        tau=1e-4,
        cost="cheap",
        recommended_device="cpu",
        train_batch_size=256,
        n_eval_default=64,
        notes="hardened sphere; Rosenbrock obj on ||y||^2=c^2",
    )


class S1SphereTrack:
    def __init__(self) -> None:
        self.spec = _make_spec()

    @staticmethod
    def _y_star(c: Tensor) -> Tensor:
        """Constrained obj-min and feasible reference: y*(c) = (c/sqrt(2), c/sqrt(2), 0, 0)."""
        B = c.shape[0]
        c0 = c[:, 0:1]
        head = c0 * _INV_SQRT2  # [B, 1]
        zeros_tail = torch.zeros(B, _DIM - 2, device=c.device, dtype=c.dtype)
        return torch.cat([head, head, zeros_tail], dim=-1)

    @staticmethod
    def _shift(c: Tensor) -> Tensor:
        """s(c) = y*(c) - (a, a^2, 0, 0) so Rosenbrock(y - s(c)) has min at y*(c)."""
        B = c.shape[0]
        c0 = c[:, 0:1]
        s0 = c0 * _INV_SQRT2 - _ROSEN_A
        s1 = c0 * _INV_SQRT2 - _ROSEN_A * _ROSEN_A
        zeros_tail = torch.zeros(B, _DIM - 2, device=c.device, dtype=c.dtype)
        return torch.cat([s0, s1, zeros_tail], dim=-1)

    def _rosenbrock(self, y: Tensor, c: Tensor) -> Tensor:
        """Rosenbrock on (z0,z1) only. Tail dims z2,z3 do not enter f.

        {f=0} is a 2D plane in R^4. Sphere intersect plane = single point y*.
        """
        z = y - self._shift(c)
        return (
            (_ROSEN_A - z[:, 0]) ** 2
            + _ROSEN_B * (z[:, 1] - z[:, 0] ** 2) ** 2
        )

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        obj = self._rosenbrock(x, conditions)

        c_sq = (conditions[:, 0]) ** 2
        raw_eq = (x * x).sum(dim=-1) - c_sq

        B = x.shape[0]
        device = x.device
        return obj, [
            Constraint(
                value=raw_eq, type="eq",
                tol=torch.full((B,), 1e-3, device=device),
                margin=torch.zeros(B, device=device),
                name="sphere",
            ),
        ]

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        assert conditions is not None
        c_sq = (conditions[:, 0]) ** 2
        raw_eq = (x * x).sum(dim=-1) - c_sq
        return raw_eq.unsqueeze(-1)

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
