"""Analytical BWB planform (cubic Bezier surface patches), used by the wingbox builder."""

from __future__ import annotations

from .geometry import (
    AIRFOIL_CP_INNER,
    AIRFOIL_CP_TIP,
    build_yaml_config,
    compute_control_nets,
    compute_stations,
)

__all__ = [
    "AIRFOIL_CP_INNER",
    "AIRFOIL_CP_TIP",
    "build_yaml_config",
    "compute_control_nets",
    "compute_stations",
]
