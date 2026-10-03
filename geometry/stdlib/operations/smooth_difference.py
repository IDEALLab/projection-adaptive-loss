"""Union operation."""

from typing import Any

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.inverse import inverse
from geometry.stdlib.operations.plane_ops import _validate_same_plane
from geometry.stdlib.operations.smooth_intersection import smooth_intersection
from geometry.stdlib.operations.smooth_union import deserialize_float, smooth_union


def check_shape_def(shape_def: dict[str, Any]) -> None:
    shapes = shape_def["shapes"]
    if len(shapes) < 2:
        raise ValueError(
            f"smooth_difference requires base shape + at least 1 to subtract. "
            f"Got {len(shapes)} shapes. Example: shapes: [base, hole]"
        )
    k = shape_def.get("k")
    if k is None:
        raise ValueError(
            "smooth_difference requires 'k' parameter for blend sharpness. Example: k: 8.0"
        )


@loadable_shape(
    check_shape_def=check_shape_def,
    deserialize_args={"k": deserialize_float},
    feed_positionals_from="shapes",
)
def smooth_difference(base: Shape, /, *to_subtract: Shape, k: float) -> Shape:
    """
    Compute the smooth difference: base shape minus other shapes.

    Unlike standard difference (hard max with negation), smooth_difference
    creates rounded edges at the subtraction boundaries. Has continuous
    gradients everywhere.

    Implemented as: smooth_intersection(base, inverse(subtractor), k=k)
    For multiple subtractions, subtractors are first combined with smooth_union.

    Note: Like standard difference, this is NOT symmetric.
    smooth_difference(a, b, k) != smooth_difference(b, a, k)

    Args:
        base: The base shape to subtract from
        *to_subtract: One or more shapes to subtract from the base
                      For 2D shapes, all must be in the same plane.
        k: Blend sharpness parameter (REQUIRED, scalar float, must be > 0)
           - Larger k -> sharper transition (approaches hard difference)
           - Smaller k -> smoother/rounder blend
           - Typical range: 2.0 to 50.0

    Returns:
        Shape representing base with all other shapes smoothly removed

    Raises:
        ValueError: If no shapes to subtract provided
        ValueError: If k is not positive
        ValueError: If batch sizes are incompatible
        ValueError: If 2D shapes have different planes
        TypeError: If k is not a number

    Note:
        k=0 is equivalent to hard difference but is not differentiable
        at the boundaries. Use k > 0 for differentiability.

    Example:
        >>> cube = box(size=torch.tensor([[2., 2., 2.]]))
        >>> hole = sphere(radius=torch.tensor([[0.5]]), center=torch.tensor([[0., 0., 0.]]))
        >>> hollow = smooth_difference(cube, hole, k=8.0)
    """
    if len(to_subtract) < 1:
        raise ValueError(
            f"smooth_difference requires at least 1 shape to subtract, "
            f"got {len(to_subtract)}"
        )

    # Validate 2D shapes have same plane
    all_shapes = [base] + list(to_subtract)
    _validate_same_plane(all_shapes, "smooth_difference")

    # Validate k is a scalar number
    if not isinstance(k, (int, float)):
        raise TypeError(f"k must be a number, got {type(k).__name__}")

    if k <= 0:
        raise ValueError(
            f"smooth_difference requires k > 0, got {k}. "
            "Note: k=0 is equivalent to hard difference but is not differentiable."
        )

    # For multiple subtractions, first smooth_union the subtractors
    if len(to_subtract) == 1:
        subtractor = to_subtract[0]
    else:
        subtractor = smooth_union(*to_subtract, k=k)

    # smooth_difference(base, sub) = smooth_intersection(base, inverse(sub))
    return smooth_intersection(base, inverse(subtractor), k=k)
