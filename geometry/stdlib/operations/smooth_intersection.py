"""Union operation."""

from typing import Any

from geometry.core import Shape
from geometry.loader import assert_at_least_2_shapes, loadable_shape
from geometry.stdlib.operations.inverse import inverse
from geometry.stdlib.operations.plane_ops import _validate_same_plane
from geometry.stdlib.operations.smooth_union import deserialize_float, smooth_union


def check_shape_def(shape_def: dict[str, Any]) -> None:
    assert_at_least_2_shapes("smooth_intersection")(shape_def)
    k = shape_def.get("k")
    if k is None:
        raise ValueError(
            "smooth_intersection requires 'k' parameter for blend sharpness. Example: k: 10.0"
        )


@loadable_shape(
    check_shape_def=check_shape_def,
    deserialize_args={"k": deserialize_float},
)
def smooth_intersection(*shapes: Shape, k: float) -> Shape:
    """
    Compute the smooth intersection of multiple shapes.

    Unlike standard intersection (hard max), smooth_intersection creates
    a rounded blend at the boundary. Has continuous gradients everywhere.

    Implemented as: inverse(smooth_union(inverse(s1), inverse(s2), ..., k=k))

    Args:
        *shapes: Two or more Shape objects to combine
                 For 2D shapes, all must be in the same plane.
        k: Blend sharpness parameter (REQUIRED, scalar float, must be > 0)
           - Larger k -> sharper transition (approaches hard max)
           - Smaller k -> smoother/rounder blend
           - Typical range: 2.0 to 50.0

    Returns:
        Shape representing the smooth intersection of all inputs

    Raises:
        ValueError: If fewer than 2 shapes provided
        ValueError: If k is not positive
        ValueError: If batch sizes are incompatible
        ValueError: If 2D shapes have different planes
        TypeError: If k is not a number

    Note:
        k=0 is equivalent to hard intersection (max) but is not differentiable
        at the boundaries where shapes meet. Use k > 0 for differentiability.

    Example:
        >>> s = sphere(radius=torch.tensor([[1.5]]), center=torch.tensor([[0., 0., 0.]]))
        >>> b = box(size=torch.tensor([[2., 2., 2.]]))
        >>> rounded_cube = smooth_intersection(s, b, k=10.0)
    """
    if len(shapes) < 2:
        raise ValueError(
            f"smooth_intersection requires at least 2 shapes, got {len(shapes)}"
        )

    # Validate 2D shapes have same plane
    _validate_same_plane(list(shapes), "smooth_intersection")

    # Validate k is a scalar number
    if not isinstance(k, (int, float)):
        raise TypeError(f"k must be a number, got {type(k).__name__}")

    if k <= 0:
        raise ValueError(
            f"smooth_intersection requires k > 0, got {k}. "
            "Note: k=0 is equivalent to hard intersection (max) but is not differentiable."
        )

    # smooth_max(a, b) = -smooth_min(-a, -b)
    inverted_shapes = [inverse(s) for s in shapes]
    smooth_min_result = smooth_union(*inverted_shapes, k=k)
    return inverse(smooth_min_result)
