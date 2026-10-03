"""Gradient-share instrumentation shared by every learned solver.

Logs ``grad_norm_con`` (norm of the constraint-loss gradient) and
``grad_norm_tot`` (total-loss gradient before clipping) on a sparse schedule.
Neither touches ``.grad`` or optimizer state.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor

#: Log cadence in epochs. The final epoch is always logged on top of this.
GRAD_SHARE_EVERY = 10

#: Metric keys. Identical across every method so the analyzer reads uniformly.
KEY_CON = "grad_norm_con"
KEY_TOT = "grad_norm_tot"


def should_log_grad_share(
    epoch: int, total_epochs: int, every: int = GRAD_SHARE_EVERY
) -> bool:
    """True on every `every`-th epoch and on the final epoch (1-based)."""
    return epoch % every == 0 or epoch >= total_epochs


def constraint_grad_norm(loss_con: Tensor, params: Iterable) -> float:
    """||d `loss_con` / d theta||2 over all trainable params, without touching `.grad`.

    Returns 0.0 when the constraint term is a detached constant.
    """
    plist = [p for p in params if p.requires_grad]
    if not plist or not isinstance(loss_con, Tensor) or not loss_con.requires_grad:
        return 0.0
    grads = torch.autograd.grad(
        loss_con, plist, retain_graph=True, create_graph=False, allow_unused=True,
    )
    total_sq = 0.0
    for g in grads:
        if g is None:
            continue
        total_sq += float(g.detach().pow(2).sum().item())
    return total_sq ** 0.5


def param_grad_norm(params: Iterable) -> float:
    """||d L_total / d theta||2 read off `.grad`. Call after `backward()`, before clipping."""
    total_sq = 0.0
    for p in params:
        if p.grad is not None:
            total_sq += float(p.grad.detach().pow(2).sum().item())
    return total_sq ** 0.5
