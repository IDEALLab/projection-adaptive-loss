"""Overlap volume computation for SDF shapes.

Provides:
- sdf_overlap_volume(): Hierarchical SDF overlap with sigmoid weighting
- aabb_overlap_volume(): Fast AABB intersection (internal utility)
- OverlapResult: Dataclass with volume and uncertainty
"""

import copy
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from ..core import Shape
from ..loader import substitute_params
from ..utils import tensorify_shape_def


@dataclass
class OverlapResult:
    """Result of overlap volume computation.

    Attributes:
        volume: [B] sigmoid-weighted overlap volume (differentiable)
        uncertainty: [B] Lipschitz error bound (absolute, same cubic units as volume)
    """

    volume: Tensor
    uncertainty: Tensor


def aabb_overlap_volume(
    size_a: Tensor,
    center_a: Tensor,
    size_b: Tensor,
    center_b: Tensor,
) -> Tensor:
    """Axis-aligned box-box intersection volume.

    Args:
        size_a: [B, 3] dimensions of box A
        center_a: [B, 3] center of box A
        size_b: [B, 3] dimensions of box B
        center_b: [B, 3] center of box B

    Returns:
        [B] intersection volume (0 when no overlap)
    """
    lo_a = center_a - size_a / 2
    hi_a = center_a + size_a / 2
    lo_b = center_b - size_b / 2
    hi_b = center_b + size_b / 2

    overlap = torch.clamp(
        torch.min(hi_a, hi_b) - torch.max(lo_a, lo_b),
        min=0,
    )
    return overlap.prod(dim=-1)  # [B]


def _resolve_box_params(
    config: dict[str, Any],
    params: dict[str, Any],
    name: str,
    batch_size: int,
    device=None,
) -> tuple[Tensor, Tensor] | None:
    """Walk shape graph to find underlying box.

    Handles: direct box, box wrapped in translate (accumulates offset).
    Rejects: rotate, union, intersection, difference, etc.

    Args:
        config: Full YAML config dict
        params: Current parameter values
        name: Shape name to resolve
        batch_size: Batch size for tensorification
        device: PyTorch device for tensor creation (None = CPU)

    Returns:
        (size [B,3], center [B,3]) or None if not an unrotated box
    """
    shapes = config["shapes"]
    if name not in shapes:
        raise ValueError(
            f"Shape '{name}' not found. Available shapes: {sorted(shapes.keys())}"
        )

    offset = torch.zeros(batch_size, 3, device=device)

    while True:
        if name not in shapes:
            return None

        shape_def = copy.deepcopy(shapes[name])
        shape_def = substitute_params(shape_def, params)
        shape_def = tensorify_shape_def(shape_def, batch_size, device=device)

        t = shape_def["type"]

        if t in ("box", "box_sharp"):
            size = shape_def["size"]  # [B, 3]
            center = shape_def.get("center", torch.zeros(batch_size, 3, device=device))
            center = center + offset
            return size, center

        if t == "translate":
            offset = offset + shape_def["offset"]  # [B, 3]
            name = shape_def["shape"]  # follow child

        else:
            return None


def _make_grid(
    bounds_min: Tensor, bounds_max: Tensor, resolution: int
) -> tuple[Tensor, Tensor]:
    """Uniform grid cell centers.

    Args:
        bounds_min: [3] lower bounds
        bounds_max: [3] upper bounds
        resolution: Cells per axis

    Returns:
        Tuple of (centers [R^3, 3], half_size [3])
    """
    size = bounds_max - bounds_min
    cell_size = size / resolution
    half_step = cell_size / 2

    coords = [
        torch.linspace(
            (bounds_min[d] + half_step[d]).item(),
            (bounds_max[d] - half_step[d]).item(),
            resolution,
            device=bounds_min.device,
        )
        for d in range(3)
    ]

    gx, gy, gz = torch.meshgrid(coords[0], coords[1], coords[2], indexing="ij")
    centers = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    return centers, half_step


def _subdivide_cells(centers: Tensor, half_size: Tensor, step: int = 1) -> Tensor:
    """Split each cell into (2^step)^3 subcells.

    Args:
        centers: [M, 3] cell centers
        half_size: [3] half-size of current cells
        step: Number of octree levels per subdivision (default 1).
            step=1: 8 subcells (2x per axis), step=2: 64 subcells (4x per axis).

    Returns:
        [M * (2^step)^3, 3] subcell centers
    """
    n = 2**step  # children per axis
    child_half = half_size / n

    # Offsets per axis: {-(n-1), -(n-3), ..., (n-3), (n-1)} * child_half
    # For step=1: [-1, 1], step=2: [-3, -1, 1, 3]
    ticks = torch.arange(1 - n, n, 2, dtype=centers.dtype, device=centers.device)
    ox, oy, oz = torch.meshgrid(ticks, ticks, ticks, indexing="ij")
    offsets = torch.stack([ox, oy, oz], dim=-1).reshape(-1, 3) * child_half  # [n^3, 3]

    return (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 3)


def sdf_overlap_volume(
    shape1: Shape,
    shape2: Shape,
    bounds_min: Tensor,
    bounds_max: Tensor,
    base_res: int = 8,
    levels: int = 4,
    step: int = 1,
    lipschitz: float = 1.0,
    epsilon: float = 0.01,
) -> OverlapResult:
    """Hierarchical SDF overlap volume with sigmoid weighting.

    Uses Lipschitz-1 property for efficient cell classification with
    hierarchical refinement. Volume is differentiable via sigmoid weighting.

    All B geometry variants share the same grid. At each level, cells that
    are mixed for ANY batch element are subdivided. Resolved cells use hard
    weights (1.0 for inside-both, 0.0 for outside-either). Mixed cells on
    the final level use sigmoid weights for differentiability.

    Args:
        shape1: First SDF shape
        shape2: Second SDF shape
        bounds_min: [3] lower bounds of evaluation domain
        bounds_max: [3] upper bounds of evaluation domain
        base_res: Grid cells per axis at coarsest level
        levels: Number of refinement levels
        step: Octree levels per refinement (default 1). step=2 subdivides by
            4x per axis (64 subcells) instead of 2x (8 subcells), skipping
            intermediate evaluations. Effective res = base_res * (2^step)^(levels-1).
        lipschitz: Lipschitz constant of the SDFs (default 1.0 for true SDFs)
        epsilon: Sigmoid sharpness (smaller = closer to hard boundary)

    Returns:
        OverlapResult with volume [B] and uncertainty [B]
    """
    n_children = (2**step) ** 3  # subcells per subdivision
    factor = 2**step  # size reduction per level

    centers, half_size = _make_grid(bounds_min, bounds_max, base_res)

    # First evaluation to determine batch size
    sdf1 = shape1(centers)  # [B1, N]
    sdf2 = shape2(centers)  # [B2, N]
    B = max(sdf1.shape[0], sdf2.shape[0])
    device = sdf1.device

    volume = torch.zeros(B, device=device)
    uncertainty = torch.zeros(B, device=device)
    active = torch.ones(B, centers.shape[0], dtype=torch.bool, device=device)

    for level in range(levels):
        if level > 0:
            sdf1 = shape1(centers)
            sdf2 = shape2(centers)

        # Cell geometry at this level
        cell_vol = (half_size * 2).prod()
        half_diag = half_size.norm()
        r = lipschitz * half_diag

        # Lipschitz classification [B, N]
        inside_both = (sdf1 < -r) & (sdf2 < -r)
        outside_either = (sdf1 > r) | (sdf2 > r)

        # Accumulate resolved-inside cells with hard weight 1.0
        resolved_inside = active & inside_both
        volume = volume + resolved_inside.float().sum(-1) * cell_vol

        # Mark resolved cells inactive
        newly_resolved = inside_both | outside_either
        active = active & ~newly_resolved

        if level == levels - 1:
            # Final level: accumulate remaining mixed cells with sigmoid
            w = torch.sigmoid(-sdf1 / epsilon) * torch.sigmoid(-sdf2 / epsilon)
            volume = volume + (active.float() * w).sum(-1) * cell_vol
            uncertainty = active.float().sum(-1) * cell_vol
        else:
            # Find cells mixed for ANY batch element
            any_active = active.any(dim=0)  # [N]

            if not any_active.any():
                break  # All cells resolved

            # Select and subdivide mixed cells
            mixed_idx = any_active.nonzero(as_tuple=True)[0]
            centers = _subdivide_cells(centers[mixed_idx], half_size, step=step)

            # Propagate per-batch active mask to children
            active = active[:, mixed_idx].repeat_interleave(n_children, dim=1)

            # Child cells are smaller by factor
            half_size = half_size / factor

    return OverlapResult(volume=volume, uncertainty=uncertainty)
