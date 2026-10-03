"""Constraint dataclass and dead-Huber loss functions for standard convention."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

SHIFT = 0.2
EPS = 1e-8


@dataclass
class Constraint:
    """A single named constraint with tolerance/margin metadata.

    Args:
        value: [B] raw signed constraint value.
        type: "eq" or "ineq".
        tol: [B] eq tolerance band (|h| <= tol = satisfied).
        margin: [B] ineq safety margin (g <= -margin = satisfied).
        name: Human-readable label.
    """

    value: Tensor
    type: str
    tol: Tensor
    margin: Tensor
    name: str = ""

    def __post_init__(self) -> None:
        if self.type not in ("eq", "ineq"):
            raise ValueError(f"Constraint type must be 'eq' or 'ineq', got '{self.type}'")


def dead_huber_eq(y: Tensor, tol: Tensor) -> Tensor:
    """Eq constraint loss. Zero + zero gradient inside tolerance band.

    Args:
        y: [B] raw signed equality constraint value.
        tol: [B] tolerance (must be > 0).

    Returns:
        [B] non-negative loss.
    """
    delta = 0.5 * tol
    h = torch.relu(y.abs() - (1 - SHIFT) * tol)
    return torch.where(h <= delta, 0.5 * h**2 / (delta + EPS), h - 0.5 * delta)


def dead_huber_ineq(y: Tensor, margin: Tensor) -> Tensor:
    """Ineq constraint loss. Zero + zero gradient deep inside margin.

    Args:
        y: [B] raw signed inequality constraint value (<=0 = satisfied).
        margin: [B] safety margin (must be > 0).

    Returns:
        [B] non-negative loss.
    """
    h = torch.relu(y + margin * (1 + 2 * SHIFT))
    delta = margin
    return torch.where(h <= delta, 0.5 * h**2 / (delta + EPS), h - 0.5 * delta)


def constraints_to_violation(constraints: list[Constraint]) -> Tensor:
    """Transform standard-convention constraints to non-negative violations via dead-Huber.

    Args:
        constraints: List of Constraint objects with raw signed values.

    Returns:
        [B, K] non-negative violation tensor, suitable for penalty methods.

    Raises:
        ValueError: If tol <= 0 for eq.
    """
    violations = []
    for c in constraints:
        if c.type == "eq":
            if (c.tol <= 0).any():
                raise ValueError(
                    f"Constraint '{c.name}': tol must be > 0. "
                    f"Use a small value (e.g. 1e-4) instead of 0."
                )
            violations.append(dead_huber_eq(c.value, c.tol))
        else:
            # margin == 0 is allowed; `delta + EPS` avoids division by zero.
            violations.append(dead_huber_ineq(c.value, c.margin))
    return torch.stack(violations, dim=-1)


def compute_feasibility_standard(constraints: list[Constraint]) -> float:
    """Fraction of batch where all constraints are satisfied (standard convention).

    eq: |value| <= tol
    ineq: value <= -margin

    Args:
        constraints: List of Constraint objects.

    Returns:
        Feasibility fraction in [0, 1].
    """
    if not constraints:
        return 1.0
    B = constraints[0].value.shape[0]
    all_sat = torch.ones(B, dtype=torch.bool, device=constraints[0].value.device)
    for c in constraints:
        if c.type == "eq":
            all_sat &= c.value.abs() <= c.tol
        else:
            all_sat &= c.value <= -c.margin
    return all_sat.float().mean().item()
