"""Differentiable 2D rasterization of SDF shapes.

Evaluates an SDF on a regular 2D grid at a slice plane and returns
soft-sigmoid occupancy. The output is differentiable w.r.t. shape
parameters, making it suitable for gradient-based optimization through
image-based surrogates (e.g. WinDiNet).

Provides:
    rasterize_2d(): Batched 2D occupancy rasterization
    RasterResult: Dataclass with occupancy, raw SDF, and grid metadata
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core import Shape

# Plane -> (grid_axis_0, grid_axis_1, normal_axis)
_PLANE_AXIS_MAP: dict[str, tuple[int, int, int]] = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}


@dataclass
class RasterResult:
    """Result of 2D SDF rasterization.

    Attributes:
        occupancy: [B, H, W] soft-sigmoid occupancy (differentiable).
            ~1.0 inside the shape, ~0.0 outside.
        sdf: [B, H, W] raw signed distance values.
        plane: Which axis-aligned plane was sliced ('xy', 'xz', 'yz').
        offset: Position along the normal axis.
        extents: ((u_min, u_max), (v_min, v_max)) of the rasterized region.
    """

    occupancy: Tensor
    sdf: Tensor
    plane: str
    offset: float
    extents: tuple[tuple[float, float], tuple[float, float]]


def rasterize_2d(
    shape: Shape,
    plane: str = "xy",
    offset: float = 0.0,
    resolution: tuple[int, int] = (128, 128),
    extents: tuple[tuple[float, float], tuple[float, float]] | None = None,
    epsilon: float = 0.01,
) -> RasterResult:
    """Rasterize an SDF shape into 2D occupancy at a slice plane.

    Creates a regular grid in the specified plane at the given offset
    along the normal axis, evaluates the SDF, and applies a soft sigmoid
    to produce differentiable occupancy.

    Args:
        shape: Shape to rasterize.
        plane: Axis-aligned slice plane ('xy', 'xz', or 'yz').
        offset: Position along the normal axis (e.g. z-value for 'xy').
        resolution: (H, W) grid resolution in pixels.
        extents: ((u_min, u_max), (v_min, v_max)) bounds for the grid
            axes. Required, no default.
        epsilon: Sigmoid sharpness. Smaller -> sharper boundary.

    Returns:
        RasterResult with occupancy [B, H, W] and raw SDF [B, H, W].

    Raises:
        ValueError: If plane is invalid or extents not provided.
    """
    if plane not in _PLANE_AXIS_MAP:
        raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")
    if extents is None:
        raise ValueError(
            "extents is required: ((u_min, u_max), (v_min, v_max)). "
            "Use CADProgram.rasterize_2d() to auto-derive from bounds."
        )

    ax_u, ax_v, ax_n = _PLANE_AXIS_MAP[plane]
    H, W = resolution

    # Build 2D meshgrid
    u = torch.linspace(extents[0][0], extents[0][1], H)
    v = torch.linspace(extents[1][0], extents[1][1], W)
    uu, vv = torch.meshgrid(u, v, indexing="ij")  # [H, W] each

    # Assemble [H*W, 3] query points
    points = torch.zeros(H * W, 3)
    points[:, ax_u] = uu.reshape(-1)
    points[:, ax_v] = vv.reshape(-1)
    points[:, ax_n] = offset

    # Move to shape's device if needed
    if shape.device is not None:
        points = points.to(shape.device)

    # Evaluate SDF, [B, H*W] (no torch.no_grad: must be differentiable)
    sdf_flat = shape(points)  # auto-unsqueezes [H*W, 3] -> [1, H*W, 3]
    B = sdf_flat.shape[0]

    # Reshape and compute occupancy
    sdf_grid = sdf_flat.reshape(B, H, W)
    occupancy = torch.sigmoid(-sdf_grid / epsilon)

    return RasterResult(
        occupancy=occupancy,
        sdf=sdf_grid,
        plane=plane,
        offset=offset,
        extents=extents,
    )
