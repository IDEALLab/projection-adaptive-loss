"""Repair-contraction instrumentation for solvers with a training-time repair.

Logs the mean violation before (``repair_c_pre``) and after (``repair_c_post``)
the training-time repair step, on the gradient-share schedule.
"""

from __future__ import annotations

import os

from torch import Tensor

from pal.solvers.grad_share import should_log_grad_share

#: Metric keys. Identical across every method so the analyzer reads uniformly.
KEY_C_PRE = "repair_c_pre"
KEY_C_POST = "repair_c_post"

#: Set to "1" to disable the probe everywhere.
DISABLE_ENV = "PAL_DISABLE_REPAIR_CONTRACTION"


def repair_contraction_enabled() -> bool:
    """False when the kill switch is set, i.e. the probe can never fire."""
    return os.environ.get(DISABLE_ENV) != "1"


def should_log_repair_contraction(epoch: int, total_epochs: int) -> bool:
    """Same cadence as the gradient-share probe, plus the kill switch."""
    if not repair_contraction_enabled():
        return False
    return should_log_grad_share(epoch, total_epochs)


def mean_violation(violation: Tensor) -> float:
    """Mean of a non-negative per-(sample, constraint) violation tensor.

    Returns 0.0 for an empty tensor instead of NaN.
    """
    if violation.numel() == 0:
        return 0.0
    return float(violation.detach().mean().item())
