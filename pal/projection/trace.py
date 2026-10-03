"""Projection trajectory dataclasses.

Shared by PAL and the baselines with an iterative projection.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ProjectionStep:
    """One iteration of an inner projection loop."""

    iter: int
    obj: float
    constraints: list[float]


@dataclass
class ProjectionResult:
    """Output of a multi-iteration projection call."""

    x_final: object
    trajectory: list[ProjectionStep]
    converged: bool
    n_iters: int
