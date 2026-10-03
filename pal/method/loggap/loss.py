"""Log-gap multiplier for PAL.

Linear `w * residual` loss on `c_pre`; detached `c_post` drives the
per-metric weight with the log-space gap rule:

    gap = clamp(log10(c_mean / tau), min=-max_decades, max=max_decades)
    w   = clamp(w + rate * gap, min=_W_FLOOR)

Per-step `|Delta w|` saturates at `rate * max_decades` in both directions.
`monotone=True` clamps the gap at 0 so `w` never decays.
"""

from __future__ import annotations

import torch
from torch import Tensor

# Floor on the per-metric weight, so weights can always grow back.
_W_FLOOR = 1e-6


class MetricLogGap:
    """Log-gap multiplier with saturated positive growth and additive decay."""

    def __init__(
        self,
        n_metrics: int,
        tau: float,
        rate: float = 1e-2,
        max_decades: float = 1.0,
        device: str = "cpu",
        monotone: bool = False,
    ):
        self.n_metrics = n_metrics
        self.tau = tau
        self.rate = rate
        self.max_decades = max_decades
        self.monotone = monotone
        self.weights = torch.full((n_metrics,), _W_FLOOR, device=device)

    def compute_loss(self, c_pre_residuals: Tensor) -> Tensor:
        """`[B]` per-sample loss: `sum_k w_k * residual_k`."""
        return (self.weights.detach().unsqueeze(0) * c_pre_residuals).sum(dim=1)

    def update(self, c_post_residuals: Tensor) -> None:
        """Per-metric `w` update from detached per-batch `c_post` residuals `[B, K]`."""
        with torch.no_grad():
            c_mean = c_post_residuals.mean(dim=0)
            gap = torch.log10(c_mean.clamp(min=1e-30) / self.tau).clamp(
                min=-self.max_decades, max=self.max_decades
            )
            if self.monotone:
                gap = gap.clamp(min=0.0)
            self.weights = (self.weights + self.rate * gap).clamp(min=_W_FLOOR)

    def log_dict(self, prefix: str) -> dict:
        return {
            f"{prefix}/w_{k}": self.weights[k].item()
            for k in range(self.n_metrics)
        }

    def state_dict(self) -> dict:
        """Checkpoint payload: the per-metric weights (all that evolves)."""
        return {"weights": self.weights.detach().clone()}

    def load_state_dict(self, state: dict) -> None:
        """Restore `weights` from `state_dict()`, onto the current device."""
        self.weights = state["weights"].to(self.weights.device)
