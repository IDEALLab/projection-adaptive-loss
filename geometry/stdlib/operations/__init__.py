"""
SDF Operations - Boolean and Transform Operations

Strict tensor validation: ALL parameters must be [B, K] tensors.
Exception: smooth boolean `k` parameter remains a scalar float.

Boolean operations (CSG - Constructive Solid Geometry):
    - union(a, b, ...) = min(sdf_a, sdf_b, ...)      -- inside ANY shape
    - intersection(a, b, ...) = max(sdf_a, sdf_b, ...) -- inside ALL shapes
    - difference(a, b, ...) = max(sdf_a, -sdf_b, ...) -- a minus all others

Transform operations:
    - extrude(shape_2d, axis, min, max) -- extrude 2D shape along axis
    - translate(shape, offset) -- move shape by offset
    - rotate(shape, axis, angle) -- rotate shape around axis
    - mirror(shape, plane, offset) -- reflect shape across plane
    - revolve(shape_2d, axis, start_angle, end_angle) -- revolve 2D shape

All operations support batched parameters (where applicable).
"""

import torch
from torch import Tensor

from geometry.stdlib.operations.difference import difference as difference
from geometry.stdlib.operations.extrude import extrude as extrude
from geometry.stdlib.operations.intersection import intersection as intersection
from geometry.stdlib.operations.inverse import inverse as inverse
from geometry.stdlib.operations.loft import loft as loft
from geometry.stdlib.operations.mirror import mirror as mirror
from geometry.stdlib.operations.revolve import revolve as revolve
from geometry.stdlib.operations.rotate import rotate as rotate
from geometry.stdlib.operations.scale import scale as scale
from geometry.stdlib.operations.shell import shell as shell
from geometry.stdlib.operations.smooth_difference import (
    smooth_difference as smooth_difference,
)
from geometry.stdlib.operations.smooth_intersection import (
    smooth_intersection as smooth_intersection,
)
from geometry.stdlib.operations.smooth_union import smooth_union as smooth_union
from geometry.stdlib.operations.translate import translate as translate
from geometry.stdlib.operations.union import union as union

# Spatial Transform Operations


def _build_rotation_matrix(angle: Tensor, axis: str) -> Tensor:
    """
    Build 3x3 rotation matrix for rotation around axis.

    Args:
        angle: Rotation angle in radians (scalar tensor)
        axis: 'x', 'y', or 'z'

    Returns:
        [3, 3] rotation matrix
    """
    c = torch.cos(angle)
    s = torch.sin(angle)
    zero = torch.zeros_like(c)
    one = torch.ones_like(c)

    if axis == "x":
        return torch.stack(
            [
                torch.stack([one, zero, zero]),
                torch.stack([zero, c, -s]),
                torch.stack([zero, s, c]),
            ]
        )
    if axis == "y":
        return torch.stack(
            [
                torch.stack([c, zero, s]),
                torch.stack([zero, one, zero]),
                torch.stack([-s, zero, c]),
            ]
        )
    if axis == "z":
        return torch.stack(
            [
                torch.stack([c, -s, zero]),
                torch.stack([s, c, zero]),
                torch.stack([zero, zero, one]),
            ]
        )
    raise ValueError(f"Invalid axis: {axis}")
