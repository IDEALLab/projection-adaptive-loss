"""Loft Operation"""

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.coordinates import PLANE_PERPENDICULAR_AXIS
from geometry.utils import validate_tensor

VALID_LOFT_AXES = ("x", "y", "z")

# In-plane coordinate indices for each plane
LOFT_PLANE_AXES = {
    "xy": (0, 1),  # x, y
    "xz": (0, 2),  # x, z
    "yz": (1, 2),  # y, z
}


def _get_loft_batch_size(
    shape_start: Shape,
    shape_end: Shape,
    min_t: Tensor,
    max_t: Tensor,
    pos_start_t: Tensor | None,
    pos_end_t: Tensor | None,
    rot_start_t: Tensor | None,
    rot_end_t: Tensor | None,
    scale_start_t: Tensor | None = None,
    scale_end_t: Tensor | None = None,
) -> int:
    """
    Determine batch size for loft operation.

    All non-None batch sizes must match (strict - no broadcasting).
    """
    if shape_start.batch_size is None:
        raise ValueError(
            "Internal error: shape_start.batch_size is None. "
            "All shapes must have batch_size >= 1."
        )
    if shape_end.batch_size is None:
        raise ValueError(
            "Internal error: shape_end.batch_size is None. "
            "All shapes must have batch_size >= 1."
        )

    batch_sizes = [
        shape_start.batch_size,
        shape_end.batch_size,
        min_t.shape[0],
        max_t.shape[0],
    ]
    names = ["shape_start", "shape_end", "min", "max"]

    if pos_start_t is not None:
        batch_sizes.append(pos_start_t.shape[0])
        names.append("pos_start")
    if pos_end_t is not None:
        batch_sizes.append(pos_end_t.shape[0])
        names.append("pos_end")
    if rot_start_t is not None:
        batch_sizes.append(rot_start_t.shape[0])
        names.append("rotation_start")
    if rot_end_t is not None:
        batch_sizes.append(rot_end_t.shape[0])
        names.append("rotation_end")
    if scale_start_t is not None:
        batch_sizes.append(scale_start_t.shape[0])
        names.append("scale_start")
    if scale_end_t is not None:
        batch_sizes.append(scale_end_t.shape[0])
        names.append("scale_end")

    unique = set(batch_sizes)
    if len(unique) > 1:
        details = ", ".join(f"{n}={b}" for n, b in zip(names, batch_sizes))
        raise ValueError(
            f"loft() requires all batched parameters to have the same batch size. "
            f"Got: {details}"
        )

    return batch_sizes[0]


@loadable_shape(input_shape_dim=2, skip_validation=True)
def loft(
    shape_start: Shape,
    shape_end: Shape,
    axis: str,
    min: Tensor,
    max: Tensor,
    pos_start: Tensor | None = None,
    pos_end: Tensor | None = None,
    rotation_start: Tensor | None = None,
    rotation_end: Tensor | None = None,
    scale_start: Tensor | None = None,
    scale_end: Tensor | None = None,
) -> Shape:
    """
    Loft between two 2D shapes along an axis, blending their SDFs linearly.

    At each height along the axis, both shapes are evaluated in a shared
    local frame that interpolates from (pos_start, rotation_start) at min
    to (pos_end, rotation_end) at max. The SDF values are linearly blended:
    100% shape_start at min, 100% shape_end at max.

    This is the standard F-rep loft approach (same as libfive, nTop, Curv).

    Limitations:
        Linear SDF blending interpolates distance VALUES, not geometry.
        Intermediate cross-sections may appear slightly curved or pinched
        compared to a true geometric interpolation, especially when the
        two shapes differ significantly in size or position. This is a
        known limitation of all F-rep loft implementations.

    Transform chain (forward): shape -> SCALE -> ROTATE -> TRANSLATE
    SDF inverse: point -> subtract pos -> rotate(-angle) -> divide by scale -> eval SDF

    SDF formula:
        t = clamp((p_axis - min) / (max - min), 0, 1)
        pos = pos_start + t * (pos_end - pos_start)
        angle = rotation_start + t * (rotation_end - rotation_start)
        scale = scale_start + t * (scale_end - scale_start)
        local = rotate(p_2d - pos, -angle) / scale
        sdf = ((1-t) * sdf_start(local) + t * sdf_end(local)) * min(scale)
        result = max(p_axis - max, max(min - p_axis, sdf))

    Args:
        shape_start: 2D shape at the min end - REQUIRED
        shape_end: 2D shape at the max end - REQUIRED
                   Must be in the same plane as shape_start.
        axis: Loft axis - REQUIRED. Must be perpendicular to the shapes' plane:
              - XY plane -> axis='z'
              - XZ plane -> axis='y'
              - YZ plane -> axis='x'
        min: [B, 1] tensor - start coordinate along axis - REQUIRED
        max: [B, 1] tensor - end coordinate along axis - REQUIRED
             min can be greater than max to reverse the loft direction.
        pos_start: [B, 2] tensor - in-plane position at min - OPTIONAL (default [0, 0])
        pos_end: [B, 2] tensor - in-plane position at max - OPTIONAL (default [0, 0])
        rotation_start: [B, 1] tensor - rotation at min in degrees - OPTIONAL (default 0)
        rotation_end: [B, 1] tensor - rotation at max in degrees - OPTIONAL (default 0)
        scale_start: [B, 1] or [B, 2] tensor - in-plane scale at min - OPTIONAL (default 1.0)
                     [B, 1] for uniform scale, [B, 2] for per-axis (applied in shape's local frame).
                     All components must be > 0.
        scale_end: [B, 1] or [B, 2] tensor - in-plane scale at max - OPTIONAL (default 1.0)
                   [B, 1] for uniform scale, [B, 2] for per-axis (applied in shape's local frame).
                   All components must be > 0.

    Returns:
        3D Shape object representing the lofted shape

    Examples:
        >>> # Circle to square morph along Z
        >>> c = circle(radius=torch.tensor([[1.0]]),
        ...            center=torch.tensor([[0., 0.]]), plane='xy')
        >>> r = rectangle(size=torch.tensor([[2., 2.]]),
        ...               center=torch.tensor([[0., 0.]]), plane='xy')
        >>> lofted = loft(c, r, axis='z',
        ...               min=torch.tensor([[0.]]),
        ...               max=torch.tensor([[3.]]))

        >>> # Twisted column (same shape, 90deg rotation)
        >>> lofted = loft(r, r, axis='z',
        ...               min=torch.tensor([[0.]]),
        ...               max=torch.tensor([[3.]]),
        ...               rotation_end=torch.tensor([[90.]]))

        >>> # Tapered cone (circle shrinks from full size to half)
        >>> lofted = loft(c, c, axis='z',
        ...               min=torch.tensor([[0.]]),
        ...               max=torch.tensor([[5.]]),
        ...               scale_end=torch.tensor([[0.5]]))
    """
    # Validate axis
    if axis is None:
        raise ValueError(
            f"loft() missing required argument: 'axis'. "
            f"Must be one of: {VALID_LOFT_AXES}"
        )
    if axis not in VALID_LOFT_AXES:
        raise ValueError(
            f"loft() got invalid axis '{axis}'. Must be one of: {VALID_LOFT_AXES}"
        )

    # Validate shapes are 2D
    if shape_start.plane is None:
        raise ValueError(
            "loft() requires shape_start to be a 2D shape (with a plane attribute). "
            "Got a 3D shape. Only 2D shapes (circle, rectangle, etc.) can be lofted."
        )
    if shape_end.plane is None:
        raise ValueError(
            "loft() requires shape_end to be a 2D shape (with a plane attribute). "
            "Got a 3D shape. Only 2D shapes (circle, rectangle, etc.) can be lofted."
        )

    # Validate same plane
    if shape_start.plane != shape_end.plane:
        raise ValueError(
            f"loft() requires both shapes to be in the same plane. "
            f"Got shape_start.plane='{shape_start.plane}' and "
            f"shape_end.plane='{shape_end.plane}'."
        )

    # Validate axis perpendicular to plane
    plane = shape_start.plane
    expected_axis = PLANE_PERPENDICULAR_AXIS[plane]
    if axis != expected_axis.name:
        raise ValueError(
            f"loft() cannot loft {plane}-plane shapes along '{axis}' axis. "
            f"The axis must be perpendicular to the shape plane. "
            f"Use axis='{expected_axis.name}'."
        )

    # Validate min/max provided
    if min is None:
        raise ValueError("loft() missing required argument: 'min'")
    if max is None:
        raise ValueError("loft() missing required argument: 'max'")

    # Strict tensor validation
    min_t = validate_tensor(min, vec_size=1, name="min")  # [B, 1]
    max_t = validate_tensor(max, vec_size=1, name="max")  # [B, 1]

    pos_start_t = (
        validate_tensor(pos_start, vec_size=2, name="pos_start")
        if pos_start is not None
        else None
    )
    pos_end_t = (
        validate_tensor(pos_end, vec_size=2, name="pos_end")
        if pos_end is not None
        else None
    )
    rot_start_t = (
        validate_tensor(rotation_start, vec_size=1, name="rotation_start")
        if rotation_start is not None
        else None
    )
    rot_end_t = (
        validate_tensor(rotation_end, vec_size=1, name="rotation_end")
        if rotation_end is not None
        else None
    )

    # Validate scale (accept [B, 1] or [B, 2])
    scale_start_t = None
    scale_end_t = None
    if scale_start is not None:
        if not isinstance(scale_start, Tensor):
            raise TypeError(
                f"scale_start must be a Tensor, got {type(scale_start).__name__}"
            )
        if scale_start.dim() != 2 or scale_start.shape[1] not in (1, 2):
            raise ValueError(
                f"scale_start must be [B, 1] (uniform) or [B, 2] (per-axis), "
                f"got shape {list(scale_start.shape)}"
            )
        if torch.any(scale_start <= 0):
            raise ValueError(
                f"scale_start must have all components > 0. "
                f"Got values: {scale_start.tolist()}"
            )
        scale_start_t = (
            scale_start if scale_start.shape[1] == 2 else scale_start.expand(-1, 2)
        )
    if scale_end is not None:
        if not isinstance(scale_end, Tensor):
            raise TypeError(
                f"scale_end must be a Tensor, got {type(scale_end).__name__}"
            )
        if scale_end.dim() != 2 or scale_end.shape[1] not in (1, 2):
            raise ValueError(
                f"scale_end must be [B, 1] (uniform) or [B, 2] (per-axis), "
                f"got shape {list(scale_end.shape)}"
            )
        if torch.any(scale_end <= 0):
            raise ValueError(
                f"scale_end must have all components > 0. "
                f"Got values: {scale_end.tolist()}"
            )
        scale_end_t = scale_end if scale_end.shape[1] == 2 else scale_end.expand(-1, 2)

    # Determine batch size (strict matching)
    batch_size = _get_loft_batch_size(
        shape_start,
        shape_end,
        min_t,
        max_t,
        pos_start_t,
        pos_end_t,
        rot_start_t,
        rot_end_t,
        scale_start_t,
        scale_end_t,
    )

    # Default optional parameters
    dev = min_t.device
    if pos_start_t is None:
        pos_start_t = torch.zeros(batch_size, 2, device=dev)
    if pos_end_t is None:
        pos_end_t = torch.zeros(batch_size, 2, device=dev)
    if rot_start_t is None:
        rot_start_t = torch.zeros(batch_size, 1, device=dev)
    if rot_end_t is None:
        rot_end_t = torch.zeros(batch_size, 1, device=dev)
    if scale_start_t is None:
        scale_start_t = torch.ones(batch_size, 2, device=dev)
    if scale_end_t is None:
        scale_end_t = torch.ones(batch_size, 2, device=dev)

    # Validate min != max
    if torch.all(torch.abs(min_t - max_t) < 1e-6):
        raise ValueError(
            f"loft() requires min != max. "
            f"Got min={min_t.squeeze().tolist()}, max={max_t.squeeze().tolist()}. "
            f"Note: min can be greater than max to reverse the loft direction."
        )

    # Convert rotations to radians
    rot_start_rad = rot_start_t * (torch.pi / 180.0)  # [B, 1]
    rot_end_rad = rot_end_t * (torch.pi / 180.0)  # [B, 1]

    # Get coordinate indices
    ax1, ax2 = LOFT_PLANE_AXES[plane]
    axis_idx = expected_axis.index

    # Detect same-shape optimization
    same_shape = shape_start is shape_end

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate lofted shape SDF: [B, N, 3] -> [B, N]"""
        N = p.shape[-2]
        B = batch_size

        # Extract coordinates from query points
        p_axis = p[..., axis_idx]  # [B, N]
        p_2d = torch.stack([p[..., ax1], p[..., ax2]], dim=-1)  # [B, N, 2]

        # Interpolation parameter t: [B, N]
        t = (p_axis - min_t) / (max_t - min_t)  # [B, N]
        t = t.clamp(0, 1)

        # Shared interpolated position and rotation at each t.
        pos_delta = pos_end_t - pos_start_t  # [B, 2]
        t_2d = t.unsqueeze(-1)  # [B, N, 1]
        pd = pos_delta.unsqueeze(1)  # [B, 1, 2]

        pos = pos_start_t.unsqueeze(1) + t_2d * pd  # [B, N, 2]

        rot_delta = rot_end_rad - rot_start_rad  # [B, 1]
        angle = rot_start_rad + t * rot_delta  # [B, N]

        # Interpolate scale
        scale_delta = scale_end_t - scale_start_t  # [B, 2]
        sd = scale_delta.unsqueeze(1)  # [B, 1, 2]
        sc = scale_start_t.unsqueeze(1) + t_2d * sd  # [B, N, 2]
        min_sc = sc.min(dim=-1).values  # [B, N]

        # Transform query points to the interpolated local frame
        local = p_2d - pos  # [B, N, 2]
        local = _rotate_2d_points(local, -angle)  # [B, N, 2]
        local = local / sc  # [B, N, 2] - apply inverse scale

        # Rebuild synthetic 3D points (out-of-place for torch.compile)
        channels = [torch.zeros(B, N, 1, device=p.device) for _ in range(3)]
        channels[ax1] = local[:, :, 0:1]
        channels[ax2] = local[:, :, 1:2]
        synth = torch.cat(channels, dim=-1)

        # Evaluate shapes at the local frame
        sdf_a = shape_start(synth)  # [B, N]
        if same_shape:
            # (1-t)*sdf + t*sdf = sdf; skip redundant eval and blend
            blended = sdf_a * min_sc  # [B, N]
        else:
            sdf_b = shape_end(synth)  # [B, N]
            blended = ((1 - t) * sdf_a + t * sdf_b) * min_sc  # [B, N]

        # Bounds clipping (flat end caps)
        lo = torch.minimum(min_t, max_t)  # [B, 1]
        hi = torch.maximum(min_t, max_t)  # [B, 1]
        result = torch.maximum(
            p_axis - hi,  # [B, N] distance above high end
            torch.maximum(
                lo - p_axis,  # [B, N] distance below low end
                blended,
            ),
        )

        return result  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, device=min_t.device)


def _rotate_2d_points(points: Tensor, angles: Tensor) -> Tensor:
    """
    Rotate 2D points by per-point angles.

    Args:
        points: [..., 2] tensor of 2D points
        angles: [...] tensor of rotation angles in radians,
                broadcastable with points[..., 0]

    Returns:
        Rotated points with same shape as input
    """
    cos_a = torch.cos(angles)
    sin_a = torch.sin(angles)

    x = points[..., 0]
    y = points[..., 1]

    x_rot = cos_a * x - sin_a * y
    y_rot = sin_a * x + cos_a * y

    return torch.stack([x_rot, y_rot], dim=-1)
