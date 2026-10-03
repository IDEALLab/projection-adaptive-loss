import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.utils import validate_tensor


@loadable_shape(input_shape_dim=2)
def extrude(
    shape: Shape,
    axis: str,
    min: Tensor,
    max: Tensor,
) -> Shape:
    """
    Extrude a 2D shape along an axis to create a 3D shape.

    The 2D shape is evaluated in the plane perpendicular to the extrusion axis,
    then bounded by min/max along the extrusion axis.

    SDF formula:
        d_2d = shape_2d(p_perpendicular)  # 2D SDF in perpendicular plane
        d_axis = max(min - p_axis, p_axis - max)  # distance to axis bounds
        d = max(d_2d, d_axis)  # intersection

    Args:
        shape: 2D shape to extrude - REQUIRED
               The shape should be defined in the plane perpendicular to the axis:
               - axis='x' expects shape in YZ plane (plane='yz')
               - axis='y' expects shape in XZ plane (plane='xz')
               - axis='z' expects shape in XY plane (plane='xy')
        axis: Extrusion axis - REQUIRED. One of: 'x', 'y', 'z'
        min: [B, 1] tensor - minimum bound along axis - REQUIRED
        max: [B, 1] tensor - maximum bound along axis - REQUIRED

    Returns:
        Shape object representing the extruded 3D shape

    Examples:
        >>> # Create a cylinder (circle extruded along Z)
        >>> c = circle(
        ...     radius=torch.tensor([[1.0]]),
        ...     center=torch.tensor([[0., 0.]]),
        ...     plane='xy'
        ... )
        >>> cylinder = extrude(
        ...     c, axis='z',
        ...     min=torch.tensor([[0.]]),
        ...     max=torch.tensor([[2.]])
        ... )

        >>> # Batched extrusion with different heights
        >>> heights = torch.linspace(1, 5, 10).unsqueeze(1)  # [10, 1]
        >>> cylinders = extrude(c, axis='z', min=torch.zeros(10, 1), max=heights)
    """
    # Validate axis
    if axis is None:
        raise ValueError(
            f"extrude() missing required argument: 'axis'. "
            f"Must be one of: {VALID_EXTRUDE_AXES}"
        )
    if axis not in VALID_EXTRUDE_AXES:
        raise ValueError(
            f"extrude() got invalid axis '{axis}'. Must be one of: {VALID_EXTRUDE_AXES}"
        )

    # Validate min/max provided
    if min is None:
        raise ValueError("extrude() missing required argument: 'min'")
    if max is None:
        raise ValueError("extrude() missing required argument: 'max'")

    # Strict tensor validation - [B, 1] format
    min_t = validate_tensor(min, vec_size=1, name="min")  # [B, 1]
    max_t = validate_tensor(max, vec_size=1, name="max")  # [B, 1]

    # Determine batch size (strict matching)
    batch_size = _get_extrude_batch_size(shape, min_t, max_t)

    # Get axis configuration
    axis_idx = AXIS_CONFIG[axis]["axis_idx"]

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate extruded shape SDF: [B, N, 3] -> [B, N]"""
        # Get the coordinate along the extrusion axis
        p_axis = p[..., axis_idx]  # [B, N]

        # Evaluate the 2D shape (returns [B, N])
        d_2d = shape(p)  # [B, N]

        # Distance to axis bounds
        d_axis = torch.maximum(min_t - p_axis, p_axis - max_t)  # [B, N]

        # Intersection: inside both the 2D shape and the axis bounds
        return torch.maximum(d_2d, d_axis)  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, device=min_t.device)


def _get_extrude_batch_size(shape: Shape, min_t: Tensor, max_t: Tensor) -> int:
    """
    Determine batch size for extrude operation.

    All batch sizes must match (strict - no broadcasting).
    """
    # All parameters are now [B, 1] format, so just get B from first dim
    batch_sizes = [shape.batch_size, min_t.shape[0], max_t.shape[0]]

    # Check for None (shouldn't happen in new architecture)
    if shape.batch_size is None:
        raise ValueError(
            "Internal error: shape.batch_size is None. "
            "All shapes must have batch_size >= 1."
        )

    # All batch sizes must match
    unique = set(batch_sizes)
    if len(unique) > 1:
        raise ValueError(
            f"extrude() requires all batched parameters to have the same batch size. "
            f"Got: shape.batch_size={shape.batch_size}, "
            f"min batch_size={min_t.shape[0]}, max batch_size={max_t.shape[0]}"
        )

    return batch_sizes[0]


VALID_EXTRUDE_AXES = ("x", "y", "z")

AXIS_CONFIG = {
    "x": {"plane_axes": (1, 2), "axis_idx": 0},  # YZ plane, extrude along X
    "y": {"plane_axes": (0, 2), "axis_idx": 1},  # XZ plane, extrude along Y
    "z": {"plane_axes": (0, 1), "axis_idx": 2},  # XY plane, extrude along Z
}
"""Axis configuration for extrude: which plane the 2D shape is in, and axis index"""
