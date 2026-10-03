"""Rosenbrock with equality + inequality constraints.

Two linear subsystems coupled through a Rosenbrock objective.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((4,), -5.0)
    bounds_hi = torch.full((4,), 5.0)
    return BenchmarkSpec(
        id="rosenbrock_eq",
        family="rosenbrock_eq",
        variant=None,
        dim=4,
        n_eq=1,
        n_ineq=1,
        constraint_names=["coupling", "bound"],
        constraint_types=["eq", "ineq"],
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=0,
        zeta_dim=4,
        tolerance=1e-4,
        tau=1e-4,
        cost="cheap",
        recommended_device="cpu",
        train_batch_size=256,
        n_eval_default=64,
        notes="two linear subsystems coupled by a Rosenbrock objective",
    )


class RosenbrockEq:
    """Rosenbrock objective on two coupled linear subsystems.

    Decision variable `x in R^4`. Two 2x2 linear subsystems `s1 = x[:2] @ A1.T`,
    `s2 = x[2:] @ A2.T` (with `A1`, `A2` fixed and seeded). Objective is
    Rosenbrock on `(s1[0], s2[0])`; one equality couples `s1[1] + s2[1] = 1`;
    one inequality caps `s1[0] >= -1`.
    """

    def __init__(self) -> None:
        self.spec = _make_spec()
        rng = torch.Generator().manual_seed(12345)
        self._A1 = torch.randn(2, 2, generator=rng)
        self._A2 = torch.randn(2, 2, generator=rng)
        self._target = 1.0
        self._threshold = -1.0

    def _subsystems(self, x: Tensor) -> tuple[Tensor, Tensor]:
        A1 = self._A1.to(device=x.device, dtype=x.dtype)
        A2 = self._A2.to(device=x.device, dtype=x.dtype)
        s1 = x[..., :2] @ A1.T
        s2 = x[..., 2:] @ A2.T
        return s1, s2

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        """Single-pass `(obj, list[Constraint])`."""
        A1 = self._A1.to(device=x.device, dtype=x.dtype)
        A2 = self._A2.to(device=x.device, dtype=x.dtype)
        s1 = x[..., :2] @ A1.T
        s2 = x[..., 2:] @ A2.T
        x0, y0 = s1[..., 0], s2[..., 0]
        obj = (1.0 - x0) ** 2 + 100.0 * (y0 - x0**2) ** 2

        raw_eq = s1[..., 1] + s2[..., 1] - self._target
        raw_ineq = self._threshold - s1[..., 0]
        B = x.shape[0]
        device = x.device
        return obj, [
            Constraint(
                value=raw_eq, type="eq",
                tol=torch.full((B,), 1e-3, device=device),
                margin=torch.zeros(B, device=device),
                name="coupling",
            ),
            Constraint(
                value=raw_ineq, type="ineq",
                tol=torch.zeros(B, device=device),
                margin=torch.full((B,), 0.1, device=device),
                name="bound",
            ),
        ]

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        s1, s2 = self._subsystems(x)
        raw_eq = s1[..., 1] + s2[..., 1] - self._target
        raw_ineq = self._threshold - s1[..., 0]
        return torch.stack([raw_eq, raw_ineq], dim=-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def sample_queries(
        self,
        n: int,
        split: Literal["train", "eval"],
        seed: int,
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        """Fixed eval set; default `n=spec.n_eval_default` (64)."""
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + 7919)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def check_env(self) -> None:
        return None

    def visualize_train(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> None:
        return None

    def visualize_final(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> None:
        return None
