"""Revolve Operation."""

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.utils import validate_tensor

VALID_REVOLVE_AXES = ("x", "y", "z")

# Valid 2D planes for each revolve axis. The plane must contain the axis letter.
REVOLVE_AXIS_PLANES = {
    "x": ("xy", "xz"),
    "y": ("xy", "yz"),
    "z": ("xz", "yz"),
}

# Configuration for each axis: which coords form the radial plane, and which is height
# For axis='y': radial plane is XZ (indices 0, 2), height is Y (index 1)
REVOLVE_AXIS_CONFIG = {
    "x": {"radial_indices": (1, 2), "height_idx": 0},  # YZ radial, X height
    "y": {"radial_indices": (0, 2), "height_idx": 1},  # XZ radial, Y height
    "z": {"radial_indices": (0, 1), "height_idx": 2},  # XY radial, Z height
}

# Synthetic point mapping: (axis, plane) -> which axis index gets replaced by r
# For revolve around Y with XY plane: r replaces X (index 0)
# For revolve around Y with YZ plane: r replaces Z (index 2)
REVOLVE_RADIAL_SUBSTITUTION = {
    ("x", "xy"): 1,  # r replaces Y
    ("x", "xz"): 2,  # r replaces Z
    ("y", "xy"): 0,  # r replaces X
    ("y", "yz"): 2,  # r replaces Z
    ("z", "xz"): 0,  # r replaces X
    ("z", "yz"): 1,  # r replaces Y
}


def _is_full_revolution(start_angle: float, end_angle: float) -> bool:
    """Check if the angle range represents a full 360 degree revolution."""
    # Full revolution: difference is exactly 360 degrees (handles both 0->360 and 360->0)
    diff = abs(end_angle - start_angle)
    return abs(diff - 360.0) < 1e-6


@loadable_shape(input_shape_dim=2)
def revolve(
    shape: Shape,
    axis: str,
    start_angle: Tensor,
    end_angle: Tensor,
) -> Shape:
    """
    Revolve a 2D shape around an axis to create a 3D shape.

    The 2D shape is rotated around the specified axis from start_angle to end_angle
    (in degrees, right-handed rotation). For partial revolutions (< 360), flat end
    caps are created at the angle boundaries.

    Args:
        shape: 2D shape to revolve - REQUIRED
               The shape must have a valid plane attribute that contains the axis letter:
               - axis='x' requires plane='xy' or plane='xz'
               - axis='y' requires plane='xy' or plane='yz'
               - axis='z' requires plane='xz' or plane='yz'
        axis: Revolution axis - REQUIRED. One of: 'x', 'y', 'z'
        start_angle: [1, 1] tensor - start angle in degrees - REQUIRED
                     (Only scalar angles supported)
        end_angle: [1, 1] tensor - end angle in degrees - REQUIRED
                   For full revolution, use start_angle=[[0]], end_angle=[[360]].
                   (Only scalar angles supported)

    Returns:
        Shape object representing the revolved 3D shape

    Examples:
        >>> # Create a torus (circle revolved around Y axis)
        >>> c = circle(
        ...     radius=torch.tensor([[0.5]]),
        ...     center=torch.tensor([[2., 0.]]),
        ...     plane='xy'
        ... )
        >>> torus = revolve(
        ...     c, axis='y',
        ...     start_angle=torch.tensor([[0.]]),
        ...     end_angle=torch.tensor([[360.]])
        ... )

        >>> # Partial revolution (quarter turn)
        >>> quarter = revolve(
        ...     c, axis='y',
        ...     start_angle=torch.tensor([[0.]]),
        ...     end_angle=torch.tensor([[90.]])
        ... )
    """
    # Validate axis
    if axis is None:
        raise ValueError(
            f"revolve() missing required argument: 'axis'. "
            f"Must be one of: {VALID_REVOLVE_AXES}"
        )
    if axis not in VALID_REVOLVE_AXES:
        raise ValueError(
            f"revolve() got invalid axis '{axis}'. Must be one of: {VALID_REVOLVE_AXES}"
        )

    # Validate angles provided
    if start_angle is None:
        raise ValueError("revolve() missing required argument: 'start_angle'")
    if end_angle is None:
        raise ValueError("revolve() missing required argument: 'end_angle'")

    # Validate angles are tensors
    start_t = validate_tensor(start_angle, vec_size=1, name="start_angle")  # [B, 1]
    end_t = validate_tensor(end_angle, vec_size=1, name="end_angle")  # [B, 1]

    # Only scalar angles supported
    if start_t.shape[0] > 1 or end_t.shape[0] > 1:
        raise NotImplementedError(
            "Batched revolve angles not yet supported. "
            "Use scalar angle values with shape [1, 1]."
        )

    # Validate that start != end (0 degree revolve is invalid)
    start_val = float(start_t.squeeze())
    end_val = float(end_t.squeeze())
    if abs(start_val - end_val) < 1e-6:
        raise ValueError(
            f"revolve() requires start_angle != end_angle. "
            f"Got start_angle={start_val}, end_angle={end_val} (0 degree revolution is invalid). "
            f"For a full revolution, use start_angle=0, end_angle=360."
        )

    # Validate plane compatibility
    valid_planes = REVOLVE_AXIS_PLANES[axis]
    if shape.plane is None:
        raise ValueError(
            "revolve() requires a 2D shape with a plane attribute. "
            "Got shape with plane=None. Use circle(), rectangle(), etc."
        )
    if shape.plane not in valid_planes:
        raise ValueError(
            f"revolve() with axis='{axis}' requires a 2D shape in plane "
            f"{valid_planes}, but got shape with plane='{shape.plane}'. "
            f"The 2D plane must contain the axis letter '{axis}'."
        )

    # Convert to radians
    start_rad = start_t.squeeze() * (torch.pi / 180.0)
    end_rad = end_t.squeeze() * (torch.pi / 180.0)

    # Check if full revolution (no end caps needed)
    is_full = _is_full_revolution(start_val, end_val)

    # Get axis configuration
    config = REVOLVE_AXIS_CONFIG[axis]
    radial_idx1, radial_idx2 = config["radial_indices"]
    height_idx = config["height_idx"]

    # Determine which radial coordinate maps to the 2D shape's first coordinate
    # Based on the shape's plane and the axis
    plane = shape.plane

    # Get which axis index to substitute with r (and -r)
    substitution_idx = REVOLVE_RADIAL_SUBSTITUTION[(axis, plane)]

    # Inherit batch_size from input shape
    batch_size = shape.batch_size

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate revolved shape SDF at points p: [B, N, 3]"""
        # Extract coordinates
        p_r1 = p[..., radial_idx1]  # [B, N] - first radial coord
        p_r2 = p[..., radial_idx2]  # [B, N] - second radial coord

        # Compute radial distance from axis
        r = torch.sqrt(p_r1**2 + p_r2**2)  # [B, N]

        # Create synthetic points by substituting r (and -r) for the appropriate coord
        zeros = torch.zeros_like(r)
        p_h = p[..., height_idx]  # [B, N] Height coordinate

        # Build synthetic point: place r, p_h, and zeros in correct positions
        coords_pos = [None, None, None]
        coords_neg = [None, None, None]

        coords_pos[substitution_idx] = r
        coords_neg[substitution_idx] = -r
        coords_pos[height_idx] = p_h
        coords_neg[height_idx] = p_h

        # Fill remaining slot with zeros
        for i in range(3):
            if coords_pos[i] is None:
                coords_pos[i] = zeros
                coords_neg[i] = zeros

        p_pos = torch.stack(coords_pos, dim=-1)  # [B, N, 3]
        p_neg = torch.stack(coords_neg, dim=-1)  # [B, N, 3]

        # Evaluate 2D shape at both synthetic points and take union (min)
        # This handles shapes that cross the axis
        d_pos = shape(p_pos)  # [B, N]
        d_neg = shape(p_neg)  # [B, N]
        d_2d = torch.minimum(d_pos, d_neg)

        # For full revolution, just return the 2D SDF
        if is_full:
            return d_2d

        # For partial revolution, add angular constraints (end caps)
        # Start boundary plane (at start_angle, normal points toward end)
        n_start_1 = -torch.sin(start_rad)  # normal component in radial_idx1 direction
        n_start_2 = torch.cos(start_rad)  # normal component in radial_idx2 direction
        d_start = p_r1 * n_start_1 + p_r2 * n_start_2  # [B, N] positive = past start

        # End boundary plane (at end_angle, normal points toward start)
        n_end_1 = torch.sin(end_rad)
        n_end_2 = -torch.cos(end_rad)
        d_end = p_r1 * n_end_1 + p_r2 * n_end_2  # [B, N] positive = before end

        # Check angular span to decide intersection vs union
        span = end_rad - start_rad  # in radians
        span_deg = span * (180.0 / torch.pi)

        # Convert to SDF (negative inside, positive outside)
        if span_deg <= 180.0:
            # Small span: intersection (point must satisfy BOTH constraints)
            d_angular = torch.maximum(-d_start, -d_end)  # [B, N]
        else:
            # Large span: union (point must satisfy EITHER constraint)
            d_angular = torch.minimum(-d_start, -d_end)  # [B, N]

        # Final SDF: intersection of revolved shape and angular wedge
        # max(d_2d, d_angular) for CSG intersection
        return torch.maximum(d_2d, d_angular)

    return Shape(sdf_fn, batch_size=batch_size, device=start_t.device)
