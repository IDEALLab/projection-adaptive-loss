"""
geometry Standard Library - SDF Primitives and Operations

Contains:
- 2D Primitives: circle, rectangle, rectangle_sharp
- 3D Primitives: sphere, box, box_sharp
- Neural Primitives: neural3d, neural2d
- Curve primitives: Curve2D, CubicBezier2D
- Workplane: Workplane
- Boolean operations: union, intersection, difference
- Smooth boolean operations: smooth_union, smooth_intersection, smooth_difference, inverse
- Transform operations: extrude, translate, rotate, mirror, scale, revolve, loft
"""

from .curves import (
    CubicBezier2D,
    Curve2D,
)
from .neural_primitives import (
    neural2d,
    neural3d,
)
from .operations import (
    difference,
    extrude,
    intersection,
    inverse,
    loft,
    mirror,
    revolve,
    rotate,
    scale,
    shell,
    smooth_difference,
    smooth_intersection,
    smooth_union,
    translate,
    union,
)
from .operations.shell import ShellMode
from .primitives import (
    PLANE_AXES,
    VALID_AXES,
    VALID_PLANES,
    bezier_halfplane,
    box,
    box_sharp,
    circle,
    formula2d,
    formula3d,
    rectangle,
    rectangle_sharp,
    sphere,
)
from .surface import (
    bezier_surface,
)
from .workplane import (
    Workplane,
)

__all__ = [
    # 2D Primitives
    "circle",
    "rectangle",
    "rectangle_sharp",
    # 3D Primitives
    "sphere",
    "box",
    "box_sharp",
    # Formula Primitives
    "formula3d",
    "formula2d",
    # Bezier Primitives
    "bezier_halfplane",
    # Neural Primitives
    "neural3d",
    "neural2d",
    # Boolean operations
    "union",
    "intersection",
    "difference",
    # Smooth boolean operations
    "smooth_union",
    "smooth_intersection",
    "smooth_difference",
    "inverse",
    # Transform operations
    "extrude",
    "translate",
    "rotate",
    "mirror",
    "scale",
    "revolve",
    "loft",
    # Curve primitives
    "Curve2D",
    "CubicBezier2D",
    # Workplane
    "Workplane",
    # Surface operations
    "bezier_surface",
    # Constants
    "PLANE_AXES",
    "VALID_PLANES",
    "VALID_AXES",
    # Shell
    "shell",
    "ShellMode",
]
