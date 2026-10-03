"""Scale Operation."""

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.coordinates import PLANE_PERPENDICULAR_AXIS
from geometry.utils import get_batch_size


@loadable_shape()
def scale(shape: Shape, factor: Tensor) -> Shape:
    """
    Scale a shape by a factor (uniform or per-axis).

    SDF formula: sdf(p / factor) * min_factor
    Exact for uniform scaling, conservative bound for non-uniform.

    Args:
        shape: Shape to scale
        factor: [B, 1] for uniform, [B, 2] for 2D shapes, or [B, 3] for per-axis - REQUIRED
                All components must be > 0 (use mirror for reflections).
                [B, 2] only valid for 2D shapes: auto-inserts 1.0 on perpendicular axis.
                For [B, 3] with 2D shapes, perpendicular axis component must be 1.0:
                - XY plane -> factor[:, 2] (Z) must be 1.0
                - XZ plane -> factor[:, 1] (Y) must be 1.0
                - YZ plane -> factor[:, 0] (X) must be 1.0

    Returns:
        Scaled Shape

    Example:
        >>> s = sphere(
        ...     radius=torch.tensor([[1.0]]),
        ...     center=torch.tensor([[0., 0., 0.]])
        ... )
        >>> big = scale(s, factor=torch.tensor([[2.0]]))  # Uniform 2x
        >>> ellipsoid = scale(s, factor=torch.tensor([[2., 1., 1.]]))  # Stretch X
    """
    # Validate factor tensor: accept [B, 1], [B, 2] (2D shapes), or [B, 3]
    if not isinstance(factor, Tensor):
        raise TypeError(
            f"factor must be a Tensor with shape [B, 1], [B, 2], or [B, 3], "
            f"got {type(factor).__name__}."
        )
    if factor.dim() != 2 or factor.shape[1] not in (1, 2, 3):
        raise ValueError(
            f"factor must be [B, 1] (uniform), [B, 2] (2D), or [B, 3] (per-axis), "
            f"got shape {list(factor.shape)}."
        )

    # Validate all components > 0
    if (factor <= 0).any():
        raise ValueError(
            f"scale factor must be > 0 for all components, got {factor.tolist()}. "
            f"Use mirror() for reflections."
        )

    # Normalize to [B, 3]
    if factor.shape[1] == 1:
        factor_t = factor.expand(-1, 3)
    elif factor.shape[1] == 2:
        # [B, 2] for 2D shapes: insert 1.0 on the perpendicular axis
        plane = getattr(shape, "plane", None)
        if plane is None:
            raise ValueError(
                "factor [B, 2] only valid for 2D shapes (with a plane attribute). "
                "Use [B, 1] or [B, 3] for 3D shapes."
            )
        # Map plane -> perpendicular axis index: xy->2(z), xz->1(y), yz->0(x)
        perp_idx = {"xy": 2, "xz": 1, "yz": 0}[plane]
        ones = torch.ones(factor.shape[0], 1, dtype=factor.dtype, device=factor.device)
        parts = (
            [factor[:, :perp_idx], ones, factor[:, perp_idx:]]
            if perp_idx < 2
            else [factor, ones]
        )
        factor_t = torch.cat(parts, dim=1)
    else:
        factor_t = factor

    # Validate 2D shape scaling (perpendicular axis must be 1.0)
    _validate_2d_scale(shape, factor_t)

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(shape, factor_t, names=["shape", "factor"])

    # Precompute min scale factor
    min_factor = factor_t.min(dim=-1).values  # [B]

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate scaled shape: [B, N, 3] -> [B, N]"""
        f_exp = factor_t.unsqueeze(1)  # [B, 1, 3]
        scaled = p / f_exp  # [B, N, 3]

        return shape(scaled) * min_factor.unsqueeze(1)  # [B, N]

    # Scale preserves plane
    return Shape(
        sdf_fn, batch_size=batch_size, plane=shape.plane, device=factor_t.device
    )


def _validate_2d_scale(shape: "Shape", factor: Tensor) -> None:
    """
    Validate that a 2D shape is only scaled within its plane.

    The perpendicular axis component of factor must be 1.0 (no scaling
    out of plane).

    Raises:
        ValueError: If factor has a non-1.0 component on the perpendicular axis.
    """
    if shape.plane is None:
        return  # 3D shape, no validation needed

    perpendicular = PLANE_PERPENDICULAR_AXIS[shape.plane]

    perp_values = factor[:, perpendicular.index]
    if (perp_values - 1.0).abs().max() > 1e-6:
        raise ValueError(
            f"Cannot scale {shape.plane.upper()}-plane shape in {perpendicular.name.upper()} direction. "
            f"Perpendicular axis factor must be 1.0 for 2D shapes."
        )
