"""MetricLogGap clamps the log-gap both ways, so per-step ``|Delta w| <= rate * max_decades``."""

from __future__ import annotations

import torch

from pal.method.loggap.loss import _W_FLOOR, MetricLogGap


def _residual(c: float) -> torch.Tensor:
    return torch.tensor([[c]], dtype=torch.float32)


def test_decay_side_capped_at_rate_times_max_decades() -> None:
    """Deeply feasible batch drops w by exactly `rate * max_decades`, no more."""
    mlg = MetricLogGap(n_metrics=1, tau=1e-3, rate=0.01, max_decades=1.0)
    mlg.weights = torch.tensor([5.0], dtype=mlg.weights.dtype)

    mlg.update(_residual(1e-30))

    # Unclamped gap log10(1e-30/1e-3) = -27 clamps to -1, so w drops by 0.01.
    assert torch.allclose(mlg.weights, torch.tensor([4.99]), atol=1e-7), mlg.weights


def test_growth_side_still_capped_at_rate_times_max_decades() -> None:
    """Symmetric clamp does not change the (already-bounded) growth side."""
    mlg = MetricLogGap(n_metrics=1, tau=1e-3, rate=0.01, max_decades=1.0)
    mlg.weights = torch.tensor([0.0], dtype=mlg.weights.dtype)

    mlg.update(_residual(1.0))

    # Unclamped gap = log10(1/1e-3) = +3 -> clamped to +1. w rises by 0.01.
    assert torch.allclose(mlg.weights, torch.tensor([0.01]), atol=1e-7), mlg.weights


def test_monotone_still_holds_on_feasible_batch() -> None:
    """`monotone=True` floors the gap at 0; deeply feasible leaves w alone."""
    mlg = MetricLogGap(
        n_metrics=1, tau=1e-3, rate=0.01, max_decades=1.0, monotone=True
    )
    mlg.weights = torch.tensor([5.0], dtype=mlg.weights.dtype)

    mlg.update(_residual(1e-30))

    assert torch.allclose(mlg.weights, torch.tensor([5.0]), atol=1e-12), mlg.weights


def test_w_floor_enforced_on_weight_not_gap() -> None:
    """`_W_FLOOR` clamps the post-update weight; gap stays signed."""
    mlg = MetricLogGap(n_metrics=1, tau=1e-3, rate=0.01, max_decades=1.0)
    mlg.weights = torch.tensor([0.005], dtype=mlg.weights.dtype)

    mlg.update(_residual(1e-30))

    # Pre-clamp weight = 0.005 - 0.01 = -0.005 -> clamped to _W_FLOOR.
    assert torch.allclose(
        mlg.weights, torch.tensor([float(_W_FLOOR)]), atol=1e-12
    ), mlg.weights


def test_no_one_step_crash_e2_flavoured() -> None:
    """A deeply feasible batch clamps the gap to -5, so one step shifts w by exactly 0.5."""
    mlg = MetricLogGap(n_metrics=1, tau=1e-4, rate=0.1, max_decades=5.0)
    mlg.weights = torch.tensor([3.0], dtype=mlg.weights.dtype)

    mlg.update(_residual(1e-30))

    assert torch.allclose(mlg.weights, torch.tensor([2.5]), atol=1e-7), mlg.weights
