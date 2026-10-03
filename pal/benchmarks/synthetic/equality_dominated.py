"""Equality-dominated benchmark (8 eq + 2 ineq on 10 outputs).

Eight rank-8 linear equalities leave a 2D feasible manifold.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((10,), -5.0)
    bounds_hi = torch.full((10,), 5.0)
    names = [f"linear_{k}" for k in range(8)] + ["x0_upper", "x1_lower"]
    types = ["eq"] * 8 + ["ineq"] * 2
    return BenchmarkSpec(
        id="equality_dominated",
        family="equality_dominated",
        variant=None,
        dim=10,
        n_eq=8,
        n_ineq=2,
        constraint_names=names,
        constraint_types=types,
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=0,
        zeta_dim=4,
        tolerance=1e-4,
        tau=1e-4,
        cost="cheap",
        recommended_device="cpu",
        train_batch_size=256,
        n_eval_default=64,
        notes="8 linear equalities on 10 outputs + 2 ineq bounds",
    )


class EqualityDominated:
    def __init__(self) -> None:
        self.spec = _make_spec()
        rng = torch.Generator().manual_seed(66666)
        self._A = torch.randn(8, 10, generator=rng)
        self._b = torch.randn(8, generator=rng) * 0.1

    def _parts(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        A = self._A.to(device=x.device, dtype=x.dtype)
        b = self._b.to(device=x.device, dtype=x.dtype)
        obj = (x[..., 0] - 1.0) ** 2 + (x[..., 1] + 0.5) ** 2
        residual = x @ A.T - b  # [B, 8]
        ineq0 = x[..., 0] - 3.0
        ineq1 = -2.0 - x[..., 1]
        return obj, residual, ineq0, ineq1

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        """Single-pass `(obj, list[Constraint])`."""
        A = self._A.to(device=x.device, dtype=x.dtype)
        b = self._b.to(device=x.device, dtype=x.dtype)
        objective = (x[..., 0] - 1.0) ** 2 + (x[..., 1] + 0.5) ** 2
        B = x.shape[0]
        dev = x.device
        out: list[Constraint] = []
        residual = x @ A.T - b
        for k in range(8):
            out.append(
                Constraint(
                    value=residual[:, k], type="eq",
                    tol=torch.full((B,), 1e-3, device=dev),
                    margin=torch.zeros(B, device=dev),
                    name=f"linear_{k}",
                )
            )
        out.append(
            Constraint(
                value=x[..., 0] - 3.0, type="ineq",
                tol=torch.zeros(B, device=dev),
                margin=torch.full((B,), 0.1, device=dev),
                name="x0_upper",
            )
        )
        out.append(
            Constraint(
                value=-2.0 - x[..., 1], type="ineq",
                tol=torch.zeros(B, device=dev),
                margin=torch.full((B,), 0.1, device=dev),
                name="x1_lower",
            )
        )
        return objective, out

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        _, residual, ineq0, ineq1 = self._parts(x)
        return torch.cat(
            [residual, torch.stack([ineq0, ineq1], dim=-1)], dim=-1
        )

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
