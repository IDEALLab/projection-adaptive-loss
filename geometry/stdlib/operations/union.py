"""Union operation."""

from functools import reduce

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import assert_at_least_2_shapes, loadable_shape
from geometry.stdlib.operations.plane_ops import _get_common_plane, _validate_same_plane
from geometry.stdlib.operations.warning import _warn_neural_csg
from geometry.utils import validate_batch_sizes


@loadable_shape(check_shape_def=assert_at_least_2_shapes("union"))
def union(*shapes: Shape) -> Shape:
    """
    Compute the union of multiple shapes.

    Union is the region that is inside ANY of the input shapes.
    SDF formula: min(sdf_a, sdf_b, ...)

    Args:
        *shapes: Two or more Shape objects to combine
                 For 2D shapes, all must be in the same plane.

    Returns:
        Shape representing the union of all inputs

    Raises:
        ValueError: If fewer than 2 shapes provided
        ValueError: If batch sizes are incompatible
        ValueError: If 2D shapes have different planes

    Example:
        >>> s1 = sphere(radius=torch.tensor([[1.0]]), center=torch.tensor([[0., 0., 0.]]))
        >>> s2 = sphere(radius=torch.tensor([[1.0]]), center=torch.tensor([[1., 0., 0.]]))
        >>> combined = union(s1, s2)  # Two overlapping spheres
    """
    if len(shapes) < 2:
        raise ValueError(f"union requires at least 2 shapes, got {len(shapes)}")

    # Warn if any neural shapes are involved
    _warn_neural_csg("union", list(shapes))

    # Validate 2D shapes have same plane
    _validate_same_plane(list(shapes), "union")

    batch_size = validate_batch_sizes(list(shapes), "union")

    def sdf_fn(p: Tensor) -> Tensor:
        # Evaluate all SDFs and take elementwise minimum
        sdfs = [shape(p) for shape in shapes]
        return reduce(torch.minimum, sdfs)

    # Preserve plane if all inputs have the same plane
    common_plane = _get_common_plane(list(shapes))
    return Shape(
        sdf_fn, batch_size=batch_size, plane=common_plane, device=shapes[0].device
    )
