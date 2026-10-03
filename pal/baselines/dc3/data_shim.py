"""Adapter from a ``Benchmark`` to the DC3 ``data`` duck type used by ``method.py``.

Upstream ``obj_fn(Y)`` takes no ``X``, so conditions are passed via ``bind_x``.
"""

from __future__ import annotations

from typing import Any, Callable, Sequence

import torch
from torch import Tensor

from pal.baselines.dc3._completion import newton_complete
from pal.baselines.dc3._completion_acopf import (
    ACOPFPartition,
    acopf_two_step_complete,
)
from pal.baselines.dc3._partial_grad import ineq_partial_grad_generic


def make_eq_resid_per_sample(bench):
    """Return ``(y_full[ydim], x[cond_dim]) -> [n_eq]`` wrapping ``bench.forward``."""
    constraint_types = list(bench.spec.constraint_types)
    has_conditions = bench.spec.condition_dim > 0

    def f(y_full: Tensor, x_single: Tensor) -> Tensor:
        conds = x_single.unsqueeze(0) if has_conditions else None
        _, cons = bench.forward(y_full.unsqueeze(0), conds)
        eqs = [c.value.squeeze(0) for c, t in zip(cons, constraint_types) if t == "eq"]
        if not eqs:
            return torch.zeros(0, device=y_full.device, dtype=y_full.dtype)
        return torch.stack(eqs)

    return f


def make_ineq_dist_per_sample(bench):
    """Return ``(y_full, x) -> [n_ineq]`` with signed ineq clamped to ``>= 0``."""
    constraint_types = list(bench.spec.constraint_types)
    has_conditions = bench.spec.condition_dim > 0

    def f(y_full: Tensor, x_single: Tensor) -> Tensor:
        conds = x_single.unsqueeze(0) if has_conditions else None
        _, cons = bench.forward(y_full.unsqueeze(0), conds)
        ineqs = [c.value.squeeze(0) for c, t in zip(cons, constraint_types) if t == "ineq"]
        if not ineqs:
            return torch.zeros(0, device=y_full.device, dtype=y_full.dtype)
        return torch.clamp(torch.stack(ineqs), min=0.0)

    return f


def make_ineq_resid_per_sample(bench):
    """Return ``(y_full, x) -> [n_ineq]`` with signed (unclamped) residuals."""
    constraint_types = list(bench.spec.constraint_types)
    has_conditions = bench.spec.condition_dim > 0

    def f(y_full: Tensor, x_single: Tensor) -> Tensor:
        conds = x_single.unsqueeze(0) if has_conditions else None
        _, cons = bench.forward(y_full.unsqueeze(0), conds)
        ineqs = [c.value.squeeze(0) for c, t in zip(cons, constraint_types) if t == "ineq"]
        if not ineqs:
            return torch.zeros(0, device=y_full.device, dtype=y_full.dtype)
        return torch.stack(ineqs)

    return f


def make_eq_resid_batched(bench):
    """Return ``(Y[B, ydim], X[B, cond_dim]) -> [B, n_eq]``.

    Batched so VJPs through it reach the probe hooks, which vmap tracers bypass.
    """
    constraint_types = list(bench.spec.constraint_types)
    has_conditions = bench.spec.condition_dim > 0

    def f(Y: Tensor, X: Tensor) -> Tensor:
        conds = X if has_conditions else None
        _, cons = bench.forward(Y, conds)
        eqs = [c.value for c, t in zip(cons, constraint_types) if t == "eq"]
        if not eqs:
            return torch.zeros(Y.shape[0], 0, device=Y.device, dtype=Y.dtype)
        return torch.stack(eqs, dim=-1)  # [B, n_eq]

    return f


def make_ineq_dist_batched(bench):
    """Return ``(Y[B, ydim], X[B, cond_dim]) -> [B, n_ineq]`` clamped >= 0."""
    constraint_types = list(bench.spec.constraint_types)
    has_conditions = bench.spec.condition_dim > 0

    def f(Y: Tensor, X: Tensor) -> Tensor:
        conds = X if has_conditions else None
        _, cons = bench.forward(Y, conds)
        ineqs = [c.value for c, t in zip(cons, constraint_types) if t == "ineq"]
        if not ineqs:
            return torch.zeros(Y.shape[0], 0, device=Y.device, dtype=Y.dtype)
        return torch.clamp(torch.stack(ineqs, dim=-1), min=0.0)  # [B, n_ineq]

    return f


class _DC3DataShim:
    """Adapter exposing a PAL benchmark through the DC3 ``data`` interface."""

    def __init__(
        self,
        bench,
        *,
        partial_vars: Sequence[int] | None,
        other_vars: Sequence[int] | None,
        linear: bool,
        newton_max_iter: int,
        newton_tol: float,
        newton_reg: float,
        newton_yf_damping: float = 0.0,
        newton_lm_damping: float = 0.0,
        warm_start_fn: Callable | None = None,
        warm_start_ctx: dict[str, Any] | None = None,
        known_vars: Sequence[int] | None = None,
        known_values: Sequence[float] | None = None,
        completion_strategy: str = "generic_newton",
        acopf_partition: ACOPFPartition | None = None,
    ):
        spec = bench.spec
        self.bench = bench
        self._spec = spec
        self._constraint_types = list(spec.constraint_types)

        self.xdim = spec.condition_dim
        self.ydim = spec.dim
        self.neq = spec.n_eq
        self.nineq = spec.n_ineq
        self._device = "cpu"

        self.partial_vars = list(partial_vars) if partial_vars else []
        self.other_vars = list(other_vars) if other_vars else []
        self.known_vars = list(known_vars) if known_vars else []
        self.known_values = list(known_values) if known_values else []
        self.nknowns = len(self.known_vars)

        if len(self.known_vars) != len(self.known_values):
            raise ValueError(
                f"known_vars ({len(self.known_vars)}) and known_values "
                f"({len(self.known_values)}) length mismatch"
            )
        if self.known_vars and (
            set(self.known_vars) & set(self.partial_vars)
            or set(self.known_vars) & set(self.other_vars)
        ):
            raise ValueError(
                "known_vars must be disjoint from partial_vars and other_vars"
            )

        self._linear = bool(linear)
        self._strategy = str(completion_strategy)
        self._acopf_partition = acopf_partition
        if self._strategy == "acopf_two_step" and self._acopf_partition is None:
            raise ValueError(
                "completion_strategy='acopf_two_step' requires an "
                "ACOPFPartition; pass one via acopf_partition=..."
            )

        self._newton_max_iter = int(newton_max_iter)
        self._newton_tol = float(newton_tol)
        self._newton_reg = float(newton_reg)
        self._newton_yf_damping = float(newton_yf_damping)
        self._newton_lm_damping = float(newton_lm_damping)
        self._warm_start_fn = warm_start_fn
        self._warm_start_ctx = dict(warm_start_ctx or {})

        self._eq_resid_per = make_eq_resid_per_sample(bench)
        self._ineq_dist_per = make_ineq_dist_per_sample(bench)
        self._ineq_resid_per = make_ineq_resid_per_sample(bench)
        self._eq_resid_batched = make_eq_resid_batched(bench)
        self._ineq_dist_batched = make_ineq_dist_batched(bench)

        self._jacobian_mode: str = "loop"

        self._X_current: Tensor | None = None
        # Objective divisor (the solver sets genbase.mean()**2 on ACOPF).
        self._obj_scale: float = 1.0

    def set_device(self, device: str) -> None:
        self._device = device

    def set_jacobian_mode(self, mode: str) -> None:
        """Set the Jacobian construction strategy (``"loop"`` or ``"vmap"``)."""
        if mode not in ("loop", "vmap"):
            raise ValueError(
                f"unknown jacobian_mode {mode!r}; expected 'loop' or 'vmap'"
            )
        self._jacobian_mode = mode

    def bind_x(self, X: Tensor | None) -> None:
        """Store the current batch's conditions so ``obj_fn`` can reach them."""
        self._X_current = X

    def _conditions(self, X: Tensor) -> Tensor | None:
        """Return ``conditions`` arg for ``bench.forward``, None when unconditional."""
        return X if self.xdim > 0 else None

    def obj_fn(self, Y: Tensor) -> Tensor:
        obj, _ = self.bench.forward(Y, self._conditions(self._X_current))
        return obj / self._obj_scale

    def eq_resid(self, X: Tensor, Y: Tensor) -> Tensor:
        _, cons = self.bench.forward(Y, self._conditions(X))
        eqs = [c.value for c, t in zip(cons, self._constraint_types) if t == "eq"]
        if not eqs:
            return Y.new_zeros(Y.shape[0], 0)
        return torch.stack(eqs, dim=-1)

    def ineq_resid(self, X: Tensor, Y: Tensor) -> Tensor:
        _, cons = self.bench.forward(Y, self._conditions(X))
        ineqs = [c.value for c, t in zip(cons, self._constraint_types) if t == "ineq"]
        if not ineqs:
            return Y.new_zeros(Y.shape[0], 0)
        return torch.stack(ineqs, dim=-1)

    def ineq_dist(self, X: Tensor, Y: Tensor) -> Tensor:
        return torch.clamp(self.ineq_resid(X, Y), min=0.0)

    def eq_grad(self, X: Tensor, Y: Tensor) -> Tensor:
        """d ||eq_resid||^2 / dY. Used only when ``corr_mode='full'``."""
        if self.neq == 0:
            return torch.zeros_like(Y)
        with torch.enable_grad():
            Y_live = Y.detach().requires_grad_(True)
            resid = self._eq_resid_batched(Y_live, self._x_for(Y))  # [B, n_eq]
            loss = resid.pow(2).sum()
            return torch.autograd.grad(loss, Y_live)[0].detach()

    def ineq_grad(self, X: Tensor, Y: Tensor) -> Tensor:
        """d ||ineq_dist||^2 / dY. Used only when ``corr_mode='full'``."""
        if self.nineq == 0:
            return torch.zeros_like(Y)
        with torch.enable_grad():
            Y_live = Y.detach().requires_grad_(True)
            dist = self._ineq_dist_batched(Y_live, self._x_for(Y))  # [B, n_ineq]
            loss = dist.pow(2).sum()
            return torch.autograd.grad(loss, Y_live)[0].detach()

    def ineq_partial_grad(self, X: Tensor, Y: Tensor) -> Tensor:
        """DC3's partial-correction gradient, full ``[B, ydim]`` (zero on ``known_vars``)."""
        if not self.partial_vars:
            return torch.zeros_like(Y)
        return ineq_partial_grad_generic(
            self._eq_resid_batched,
            self._ineq_dist_batched,
            self._x_for(Y),
            Y,
            self.partial_vars,
            self.other_vars,
            self.ydim,
            reg=self._newton_reg,
            known_vars=self.known_vars,
            jacobian_mode=self._jacobian_mode,
            eq_resid_per_sample_fn=self._eq_resid_per,
            ineq_dist_per_sample_fn=self._ineq_dist_per,
        )

    def complete_partial(self, X: Tensor, Z: Tensor) -> Tensor:
        """Solve eq for the completion vars given ``Z = y_partial``. Returns ``[B, ydim]``."""
        if not self.partial_vars:
            raise RuntimeError(
                "complete_partial called with empty partial_vars, this "
                "signals use_compl=False mode; caller should branch earlier."
            )
        if self._strategy == "acopf_two_step":
            assert self._acopf_partition is not None  # validated in __init__
            return acopf_two_step_complete(
                self._eq_resid_per,
                self._x_for(Z),
                Z,
                self._acopf_partition,
                max_iter=self._newton_max_iter,
                tol=self._newton_tol,
                reg=self._newton_reg,
                accept_floor=getattr(self, "_accept_floor_override", None),
                in_loop_cap=getattr(self, "_in_loop_cap_override", 1e3),
            )
        return newton_complete(
            self._eq_resid_batched,
            self._x_for(Z),
            Z,
            self.partial_vars,
            self.other_vars,
            self.ydim,
            linear=self._linear,
            max_iter=self._newton_max_iter,
            tol=self._newton_tol,
            reg=self._newton_reg,
            yf_damping=self._newton_yf_damping,
            lm_damping=self._newton_lm_damping,
            warm_start_fn=self._warm_start_fn,
            warm_start_ctx=self._warm_start_ctx,
            jacobian_mode=self._jacobian_mode,
            eq_resid_per_sample_fn=self._eq_resid_per,
        )

    def process_output(self, X: Tensor, Y: Tensor) -> Tensor:
        """Pass-through. ``CoordinationMLP`` already applies tanh bounds."""
        return Y

    def _x_for(self, Y: Tensor) -> Tensor:
        """Return the bound X ``[B, cond_dim]``, or zeros if ``bind_x`` was not called."""
        B = Y.shape[0]
        if self._X_current is not None:
            return self._X_current
        return torch.zeros(B, self.xdim, device=Y.device, dtype=Y.dtype)
