"""S2 unit simplex with chained Rosenbrock on dims 0..4 only.

`y_star(c)` puts the head at the f-min and a uniform tail satisfying sum y = 1.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 10
_COND_DIM = 2
_C_LO = -1.0
_C_HI = 1.0
_EVAL_SALT = 2102
_TEMP = 0.5

_D_TAIL = torch.tensor([0.15 + 0.05 * i for i in range(_DIM - 2)])

_ROSEN_B = 10.0
_TAIL_W = 0.1


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((_DIM,), 0.0)
    bounds_hi = torch.full((_DIM,), 1.0)
    names = ["sum_to_one"] + [f"nonneg_{i}" for i in range(_DIM)]
    types = ["eq"] + ["ineq"] * _DIM
    return BenchmarkSpec(
        id="s2_active_set_switch",
        family="s2",
        variant=None,
        dim=_DIM,
        n_eq=1,
        n_ineq=_DIM,
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
        notes="hardened simplex; chained Rosenbrock obj on first 5 dims",
    )


class S2ActiveSetSwitch:
    def __init__(self) -> None:
        self.spec = _make_spec()
        self._d_tail = _D_TAIL.clone()

    def _d_of_c(self, c: Tensor) -> Tensor:
        tail = self._d_tail.to(device=c.device, dtype=c.dtype).unsqueeze(0).expand(c.shape[0], -1)
        return torch.cat([c, tail], dim=-1)

    def _y_star(self, c: Tensor) -> Tensor:
        """f-min head (Rosenbrock chain) + uniform tail satisfying sum y=1.

        Head: y0=shift, y1=shift^2, y2=shift^4, y3=shift^8, y4=shift^16
        Tail: y5..y9 all equal to (1 - sum_head)/5 (positive since sum_head < 1).
        """
        shift = 0.4 + 0.05 * c[:, 0] + 0.02 * c[:, 1]  # [B]
        y0 = shift
        y1 = shift ** 2
        y2 = shift ** 4
        y3 = shift ** 8
        y4 = shift ** 16
        head = torch.stack([y0, y1, y2, y3, y4], dim=-1)  # [B, 5]
        sum_head = head.sum(dim=-1, keepdim=True)  # [B, 1]
        tail_val = (1.0 - sum_head) / 5.0  # [B, 1]
        tail = tail_val.expand(-1, _DIM - 5)  # [B, 5]
        return torch.cat([head, tail], dim=-1)

    @staticmethod
    def _rosenbrock_chain(y: Tensor, shift: Tensor) -> Tensor:
        """Chained Rosenbrock on dims 0..4 only, tail (y5..y9) does not enter f.

        shift: [B] -> first chain term `(shift - y0)^2` is c-conditional.
        """
        head_misfit = (shift - y[:, 0]) ** 2
        chain = sum(
            _ROSEN_B * (y[:, i + 1] - y[:, i] ** 2) ** 2 for i in range(4)
        )
        return head_misfit + chain

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        shift = 0.4 + 0.05 * conditions[:, 0] + 0.02 * conditions[:, 1]
        obj = self._rosenbrock_chain(x, shift)

        raw_eq = x.sum(dim=-1) - 1.0
        raw_ineq = -x

        B = x.shape[0]
        device = x.device
        constraints = [
            Constraint(
                value=raw_eq, type="eq",
                tol=torch.full((B,), 1e-3, device=device),
                margin=torch.zeros(B, device=device),
                name="sum_to_one",
            ),
        ]
        for i in range(_DIM):
            constraints.append(
                Constraint(
                    value=raw_ineq[:, i], type="ineq",
                    tol=torch.zeros(B, device=device),
                    margin=torch.zeros(B, device=device),
                    name=f"nonneg_{i}",
                )
            )
        return obj, constraints

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        raw_eq = (x.sum(dim=-1) - 1.0).unsqueeze(-1)
        raw_ineq = -x
        return torch.cat([raw_eq, raw_ineq], dim=-1)

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
