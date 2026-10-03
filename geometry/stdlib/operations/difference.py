"""Union operation."""

from typing import Any

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.plane_ops import _get_common_plane, _validate_same_plane
from geometry.stdlib.operations.warning import _warn_neural_csg
from geometry.utils import validate_batch_sizes


def check_shape_def(shape_def: dict[str, Any]) -> None:
    shapes = shape_def["shapes"]
    if len(shapes) < 2:
        raise ValueError(
            f"difference requires base shape + at least 1 to subtract. "
            f"Got {len(shapes)} shapes. Example: shapes: [base, hole]"
        )


@loadable_shape(feed_positionals_from="shapes", check_shape_def=check_shape_def)
def difference(base: Shape, /, *to_subtract: Shape) -> Shape:
    """
    Compute the difference: base shape minus all other shapes.

    Difference is the region inside the base shape but NOT inside any
    of the shapes to subtract.
    SDF formula: max(sdf_base, -sdf_sub1, -sdf_sub2, ...)

    Note: Unlike union/intersection, difference is NOT symmetric.
    difference(a, b) != difference(b, a)
    The first argument is the base; all subsequent arguments are subtracted.

    Args:
        base: The base shape to subtract from
        *to_subtract: One or more shapes to subtract from the base
                      For 2D shapes, all must be in the same plane.

    Returns:
        Shape representing base with all other shapes removed

    Raises:
        ValueError: If no shapes to subtract provided
        ValueError: If batch sizes are incompatible
        ValueError: If 2D shapes have different planes

    Example:
        >>> cube = box(size=torch.tensor([[2., 2., 2.]]))
        >>> hole = sphere(radius=torch.tensor([[0.5]]), center=torch.tensor([[0., 0., 0.]]))
        >>> hollow_cube = difference(cube, hole)
    """
    if len(to_subtract) < 1:
        raise ValueError(
            f"difference requires at least 1 shape to subtract, got {len(to_subtract)}"
        )

    all_shapes = [base] + list(to_subtract)

    # Warn if any neural shapes are involved
    _warn_neural_csg("difference", all_shapes)

    # Validate 2D shapes have same plane
    _validate_same_plane(all_shapes, "difference")

    batch_size = validate_batch_sizes(all_shapes, "difference")

    def sdf_fn(p: Tensor) -> Tensor:
        # Start with base shape's SDF
        result = base(p)
        # Subtract each shape by taking max with its negated SDF
        for shape in to_subtract:
            result = torch.maximum(result, -shape(p))
        return result

    # Preserve plane if all inputs have the same plane
    common_plane = _get_common_plane(all_shapes)
    return Shape(sdf_fn, batch_size=batch_size, plane=common_plane, device=base.device)
