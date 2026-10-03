"""Fixed-weight quadratic penalty, the base penalty of `enforce_orig` and `enforce_v4`.

Uses the `"standard"` constraint convention (eq `h == 0`, ineq `g <= 0`).
"""

from __future__ import annotations

import torch
from torch import Tensor


class FixedPenalty:
    """Quadratic penalty with fixed weights. `compute_loss` returns `[B]`.

    `w = 1.0` per constraint, no dual update.
    """

    def __init__(
        self,
        n_constraints: int,
        constraint_types: list[str],
        weights: list[float] | None = None,
        constraint_convention: str = "standard",
    ):
        assert len(constraint_types) == n_constraints
        self.n_constraints = n_constraints
        self.constraint_types = constraint_types
        self.constraint_convention = constraint_convention
        w = weights or [1.0] * n_constraints
        self._weights = torch.tensor(w)

    def compute_loss(self, constraints: Tensor) -> Tensor:
        if self.constraint_convention == "nonneg" and (constraints < 0).any():
            raise ValueError(
                f"Constraints must be >= 0 at interface. "
                f"Got min={constraints.min().item():.6f}"
            )
        w = self._weights.to(constraints.device)
        return (w * constraints.pow(2)).sum(dim=1)

    def update(self, constraints: Tensor, loss: float | None = None) -> None:
        """No-op for fixed penalty."""

    def log_dict(self) -> dict:
        return {f"penalty_weight_{k}": w for k, w in enumerate(self._weights.tolist())}
