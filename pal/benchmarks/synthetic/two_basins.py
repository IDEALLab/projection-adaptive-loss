"""Two-basin objective with infeasible strip.

Smooth-min of two quadratic wells with one linear equality and one ineq barrier.
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
        id="two_basins",
        family="two_basins",
        variant=None,
        dim=4,
        n_eq=1,
        n_ineq=1,
        constraint_names=["sum_zero", "basin_sep"],
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
        notes="smooth-min of two quadratic wells with an infeasible strip",
    )


class TwoBasins:
    def __init__(self) -> None:
        self.spec = _make_spec()
        self._c1 = torch.tensor([-2.0, 0.0, 0.0, 0.0])
        self._c2 = torch.tensor([2.0, 0.0, 0.0, 0.0])
        self._alpha = 10.0

    def _raw(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        c1 = self._c1.to(device=x.device, dtype=x.dtype)
        c2 = self._c2.to(device=x.device, dtype=x.dtype)
        d1 = ((x - c1) ** 2).sum(dim=-1)
        d2 = ((x - c2) ** 2).sum(dim=-1)
        stacked = torch.stack(
            [-self._alpha * d1, -self._alpha * d2], dim=-1
        )
        obj = -torch.logsumexp(stacked, dim=-1) / self._alpha
        raw_eq = x[..., 1] + x[..., 2]
        raw_ineq = 0.5 - x[..., 0].abs()
        return obj, raw_eq, raw_ineq

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        """Single-pass `(obj, list[Constraint])`."""
        c1 = self._c1.to(device=x.device, dtype=x.dtype)
        c2 = self._c2.to(device=x.device, dtype=x.dtype)
        d1 = ((x - c1) ** 2).sum(dim=-1)
        d2 = ((x - c2) ** 2).sum(dim=-1)
        stacked = torch.stack(
            [-self._alpha * d1, -self._alpha * d2], dim=-1
        )
        objective = -torch.logsumexp(stacked, dim=-1) / self._alpha
        raw_eq = x[..., 1] + x[..., 2]
        raw_ineq = 0.5 - x[..., 0].abs()
        B = x.shape[0]
        dev = x.device
        return objective, [
            Constraint(
                value=raw_eq, type="eq",
                tol=torch.full((B,), 1e-3, device=dev),
                margin=torch.zeros(B, device=dev),
                name="sum_zero",
            ),
            Constraint(
                value=raw_ineq, type="ineq",
                tol=torch.zeros(B, device=dev),
                margin=torch.full((B,), 0.05, device=dev),
                name="basin_sep",
            ),
        ]

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        _, raw_eq, raw_ineq = self._raw(x)
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
