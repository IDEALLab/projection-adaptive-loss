import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.coordinates import PLANE_PERPENDICULAR_AXIS
from geometry.utils import get_batch_size, validate_tensor


@loadable_shape()
def rotate(shape: Shape, axis: str, angle: Tensor) -> Shape:
    """
    Rotate shape around an axis through the origin.

    Uses inverse rotation on query points (rotate points opposite direction,
    then evaluate original shape).

    Args:
        shape: Shape to rotate
        axis: Rotation axis - 'x', 'y', or 'z'
               For 2D shapes, must be the axis perpendicular to the plane:
               - XY plane -> axis='z'
               - XZ plane -> axis='y'
               - YZ plane -> axis='x'
        angle: [B, 1] tensor - rotation angle in DEGREES - REQUIRED

    Returns:
        Rotated Shape

    Example:
        >>> b = box(size=torch.tensor([[2., 1., 1.]]))
        >>> rotated = rotate(b, axis='z', angle=torch.tensor([[45.]]))  # 45 degrees around Z
    """
    if axis not in VALID_ROTATE_AXES:
        raise ValueError(
            f"rotate() got invalid axis '{axis}'. Must be one of: {VALID_ROTATE_AXES}"
        )

    # Validate 2D shape rotation
    _validate_2d_rotation(shape, axis)

    # Validate angle is a tensor
    angle_t = validate_tensor(angle, vec_size=1, name="angle")  # [B, 1]

    # Convert degrees to radians internally
    angle_rad = angle_t * (torch.pi / 180.0)  # [B, 1]

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(shape, angle_t, names=["shape", "angle"])

    # Precompute cos/sin of inverse rotation angle
    neg_angle_rad = -angle_rad  # [B, 1]
    cos_neg = torch.cos(neg_angle_rad)  # [B, 1]
    sin_neg = torch.sin(neg_angle_rad)  # [B, 1]

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate rotated shape: [B, N, 3] -> [B, N]"""
        x = p[..., 0]  # [B, N]
        y = p[..., 1]  # [B, N]
        z = p[..., 2]  # [B, N]

        # Apply rotation per axis (broadcasts [B,1] with [B,N])
        if axis == "x":
            xr, yr, zr = x, cos_neg * y - sin_neg * z, sin_neg * y + cos_neg * z
        elif axis == "y":
            xr, yr, zr = cos_neg * x + sin_neg * z, y, -sin_neg * x + cos_neg * z
        elif axis == "z":
            xr, yr, zr = cos_neg * x - sin_neg * y, sin_neg * x + cos_neg * y, z

        xr, yr, zr = torch.broadcast_tensors(xr, yr, zr)
        p_rotated = torch.stack([xr, yr, zr], dim=-1)  # [B, N, 3]

        return shape(p_rotated)  # [B, N]

    # Preserve plane if rotating around the perpendicular axis (stays in same plane)
    # e.g., rotating XY plane shape around Z keeps it in XY
    PERPENDICULAR_AXIS = {"xy": "z", "xz": "y", "yz": "x"}
    preserved_plane = None
    if shape.plane is not None and PERPENDICULAR_AXIS.get(shape.plane) == axis:
        preserved_plane = shape.plane
    return Shape(
        sdf_fn, batch_size=batch_size, plane=preserved_plane, device=angle_rad.device
    )


def _validate_2d_rotation(shape: "Shape", axis: str) -> None:
    """
    Validate that a 2D shape is rotated around its perpendicular axis.

    Raises:
        ValueError: If the rotation axis is not perpendicular to the shape's plane.
    """
    if shape.plane is None:
        return  # 3D shape, no validation needed

    perpendicular = PLANE_PERPENDICULAR_AXIS[shape.plane]
    if axis != perpendicular.name:
        raise ValueError(
            f"Cannot rotate {shape.plane.upper()}-plane shape around {axis.upper()} axis. "
            f"Use axis='{perpendicular.name}' (perpendicular to the plane)."
        )


VALID_ROTATE_AXES = ("x", "y", "z")
