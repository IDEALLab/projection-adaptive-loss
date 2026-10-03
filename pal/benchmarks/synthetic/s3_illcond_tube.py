"""S3 ill-conditioned head quadratic, vanishing tube `||y[10:]||^2 <= c^4`, one mixing eq.

`y*(c) = t(c)` with f* = 0; the mixing eq couples head and tail.
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 20
_COND_DIM = 1
_C_LO = 0.1
_C_HI = 1.0
_N_PARALLEL = 5  # row index 0 is the mixing eq, indices 1..4 stay as ineqs
_EVAL_SALT = 2103
_CTOR_SEED = 2103

_HEAD = 10  # first 10 coords (outside tube, f-relevant)
_TAIL = _DIM - _HEAD  # last 10 coords (inside tube, f-irrelevant)


def _make_spec() -> BenchmarkSpec:
    bounds_lo = torch.full((_DIM,), -1.5)
    bounds_hi = torch.full((_DIM,), 1.5)
    names = ["mix_eq"] + ["tube"] + [f"near_par_{i}" for i in range(1, _N_PARALLEL)]
    types = ["eq"] + ["ineq"] * _N_PARALLEL  # 1 eq + (1 tube + 4 near_par) ineqs
    return BenchmarkSpec(
        id="s3_illcond_tube",
        family="s3",
        variant=None,
        dim=_DIM,
        n_eq=1,
        n_ineq=_N_PARALLEL,  # 1 tube + 4 remaining near-par
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
        notes="ill-cond Q (head only) + 1 mixing eq + tube + 4 near-parallel",
    )


class S3IllcondTube:
    def __init__(self) -> None:
        self.spec = _make_spec()
        g = torch.Generator().manual_seed(_CTOR_SEED)
        a0 = torch.randn(_DIM, generator=g)
        a0 = a0 / a0.norm().clamp(min=1e-12)
        perturb = 1e-3 * torch.randn(_N_PARALLEL, _DIM, generator=g)
        A = a0.unsqueeze(0) + perturb  # [5, 20]
        A = A / A.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        self._A = A
        # diagonal exponents for Q(c): q_i = c^(2i/9) over head dims (i=0..9)
        self._exponents = torch.tensor([2.0 * i / (_HEAD - 1) for i in range(_HEAD)])

    @staticmethod
    def _t_of_c(c: Tensor) -> Tensor:
        """target t(c): [B, 20]. head = c*1, tail = (c^2 / sqrt(20))*1.

        Out-of-place cat (vmap-safe for SnareNet/FSNet per-sample Jacobians).
        """
        c0 = c[:, 0:1]  # [B, 1]
        head = c0.expand(-1, _HEAD)
        tail = (c0**2).expand(-1, _TAIL) / (_TAIL**0.5)
        return torch.cat([head, tail], dim=-1)

    def _q_diag_head(self, c: Tensor) -> Tensor:
        """Q(c) diagonal over head only: [B, _HEAD]."""
        c0 = c[:, 0].clamp(min=1e-12)
        exponents = self._exponents.to(device=c.device, dtype=c.dtype)
        return c0.unsqueeze(-1) ** exponents.unsqueeze(0)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        t = self._t_of_c(conditions)
        q_head = self._q_diag_head(conditions)
        diff_head = x[:, :_HEAD] - t[:, :_HEAD]
        obj = 0.5 * (q_head * diff_head * diff_head).sum(dim=-1)

        c0 = conditions[:, 0]
        c4 = c0**4
        tube_val = (x[:, _HEAD:] * x[:, _HEAD:]).sum(dim=-1) - c4  # [B]

        A = self._A.to(device=x.device, dtype=x.dtype)
        a_y = x @ A.T  # [B, 5]
        a_t = t @ A.T  # [B, 5]
        # row 0: mixing equality a_0^T y - a_0^T t = 0
        mix_eq_val = a_y[:, 0] - a_t[:, 0]  # [B]
        # rows 1..4: near-parallel inequalities a_i^T y - a_i^T t - 0.2 <= 0
        near_val = a_y[:, 1:] - a_t[:, 1:] - 0.2  # [B, 4]

        B = x.shape[0]
        device = x.device
        tol_eq = torch.full((B,), 1e-3, device=device)
        tol_zero = torch.zeros(B, device=device)
        margin_zero = torch.zeros(B, device=device)
        constraints = [
            Constraint(value=mix_eq_val, type="eq", tol=tol_eq, margin=margin_zero, name="mix_eq"),
            Constraint(value=tube_val, type="ineq", tol=tol_zero, margin=margin_zero, name="tube"),
        ]
        for i in range(1, _N_PARALLEL):
            constraints.append(
                Constraint(
                    value=near_val[:, i - 1], type="ineq", tol=tol_zero, margin=margin_zero,
                    name=f"near_par_{i}",
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
        c0 = conditions[:, 0]
        c4 = c0**4
        tube_val = (x[:, _HEAD:] * x[:, _HEAD:]).sum(dim=-1) - c4
        A = self._A.to(device=x.device, dtype=x.dtype)
        a_y = x @ A.T
        a_t = t @ A.T
        mix_eq_val = (a_y[:, 0] - a_t[:, 0]).unsqueeze(-1)  # [B, 1]
        near_val = a_y[:, 1:] - a_t[:, 1:] - 0.2  # [B, 4]
        return torch.cat([mix_eq_val, tube_val.unsqueeze(-1), near_val], dim=-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def _sample_c(self, n: int, g: torch.Generator) -> Tensor:
        # log-uniform in [0.1, 1.0]: c = 0.1 * (10)^u, u ~ U[0, 1]
        u = torch.rand(n, _COND_DIM, generator=g)
        return _C_LO * (_C_HI / _C_LO) ** u

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
