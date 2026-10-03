"""Sparse 2D quadtree for adaptive SDF evaluation on slice planes.

Provides:
- quadtree_sdf_eval(): Adaptive 2D SDF evaluation via quadtree refinement
- QuadtreeResult: Dataclass with classified inside and boundary cells
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core import Shape

_AXIS_MAP: dict[str, tuple[int, int, int, str, str]] = {
    "x": (0, 1, 2, "y", "z"),
    "y": (1, 0, 2, "x", "z"),
    "z": (2, 0, 1, "x", "y"),
}


@dataclass
class QuadtreeResult:
    """Classified cells from adaptive 2D SDF evaluation on a slice plane.

    Inside cells are resolved at various quadtree levels (varying sizes).
    Boundary cells are at the finest resolution level (uniform size).
    Outside cells are discarded.

    Attributes:
        inside_centers: [M_in, 2] cell centers in (u, v) coordinates
        inside_half_sizes: [M_in, 2] half-width per cell (varies by level)
        inside_mask: [B, M_in] bool, True where batch element is inside
        boundary_centers: [M_bnd, 2] cell centers at finest level
        boundary_sdf: [B, M_bnd] SDF values for sigmoid weighting
        boundary_half_size: [2] half-size of boundary cells (uniform)
        axis: slicing axis name ('x', 'y', or 'z')
        station: position along the slicing axis
        axis_u: name of the first in-plane axis
        axis_v: name of the second in-plane axis
    """

    inside_centers: Tensor
    inside_half_sizes: Tensor
    inside_mask: Tensor
    boundary_centers: Tensor
    boundary_sdf: Tensor
    boundary_half_size: Tensor
    axis: str
    station: float
    axis_u: str
    axis_v: str


def _axis_mapping(axis: str) -> tuple[int, int, int, str, str]:
    """Map axis name to index tuple.

    Args:
        axis: Slicing axis ('x', 'y', or 'z')

    Returns:
        (axis_idx, u_idx, v_idx, axis_u_name, axis_v_name)

    Raises:
        ValueError: If axis is not 'x', 'y', or 'z'
    """
    if axis not in _AXIS_MAP:
        raise ValueError(f"axis must be 'x', 'y', or 'z', got '{axis}'")
    return _AXIS_MAP[axis]


def _make_grid_2d(
    bounds_min: Tensor, bounds_max: Tensor, resolution: int
) -> tuple[Tensor, Tensor]:
    """Uniform 2D grid of cell centers.

    Args:
        bounds_min: [2] lower bounds (u, v)
        bounds_max: [2] upper bounds (u, v)
        resolution: Cells per axis

    Returns:
        (centers [R², 2], half_size [2])
    """
    size = bounds_max - bounds_min
    cell_size = size / resolution
    half_size = cell_size / 2

    coords = [
        torch.linspace(
            (bounds_min[d] + half_size[d]).item(),
            (bounds_max[d] - half_size[d]).item(),
            resolution,
            device=bounds_min.device,
        )
        for d in range(2)
    ]

    gu, gv = torch.meshgrid(coords[0], coords[1], indexing="ij")
    centers = torch.stack([gu.reshape(-1), gv.reshape(-1)], dim=-1)
    return centers, half_size


def _subdivide_cells_2d(centers: Tensor, half_size: Tensor) -> Tensor:
    """Split each 2D cell into 4 children.

    Args:
        centers: [M, 2] cell centers
        half_size: [2] half-size of current cells

    Returns:
        [M * 4, 2] child cell centers
    """
    child_half = half_size / 2
    offsets = (
        torch.tensor(
            [[-1, -1], [-1, 1], [1, -1], [1, 1]],
            dtype=centers.dtype,
            device=centers.device,
        )
        * child_half
    )
    return (centers[:, None, :] + offsets[None, :, :]).reshape(-1, 2)


def _build_query_3d(
    centers_2d: Tensor,
    station: float,
    axis_idx: int,
    u_idx: int,
    v_idx: int,
) -> Tensor:
    """Build 3D query points from 2D cell centers and station position.

    Args:
        centers_2d: [N, 2] cell centers in (u, v)
        station: Position along the slicing axis
        axis_idx: Index of the slicing axis (0, 1, or 2)
        u_idx: Index of the u axis
        v_idx: Index of the v axis

    Returns:
        [N, 3] query points
    """
    n = centers_2d.shape[0]
    query = torch.zeros(n, 3, dtype=centers_2d.dtype, device=centers_2d.device)
    query[:, u_idx] = centers_2d[:, 0]
    query[:, v_idx] = centers_2d[:, 1]
    query[:, axis_idx] = station
    return query


def quadtree_sdf_eval(
    shape: Shape,
    axis: str,
    station: float,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    base_res: int = 8,
    levels: int = 4,
    lipschitz: float = 1.0,
) -> QuadtreeResult:
    """Adaptive 2D SDF evaluation on a slice plane via quadtree refinement.

    Evaluates a 3D shape's SDF on a 2D slice plane, using Lipschitz-based
    cell classification to skip fine evaluation in regions that are clearly
    inside or outside the shape.

    All B batch elements share the same quadtree structure. A cell is
    subdivided if it is mixed (neither clearly inside nor outside) for
    ANY batch element. Cells that are inside for some batch elements and
    outside for others are stored with a per-batch mask.

    Args:
        shape: Shape object to evaluate
        axis: Slicing axis ('x', 'y', or 'z'). Stations step along this
            axis; the grid lives in the remaining two axes.
        station: Position along the slicing axis
        bounds_min: 3D bounding box minimum (x, y, z)
        bounds_max: 3D bounding box maximum (x, y, z)
        base_res: Coarse grid resolution (cells per axis at level 0)
        levels: Number of refinement levels. Effective finest resolution
            is base_res * 2^(levels-1).
        lipschitz: Lipschitz constant of the SDF (default 1.0 for true SDFs)

    Returns:
        QuadtreeResult with classified inside and boundary cells
    """
    axis_idx, u_idx, v_idx, axis_u, axis_v = _axis_mapping(axis)

    device = torch.device(shape.device) if shape.device else torch.device("cpu")
    bb_min = torch.tensor(bounds_min, dtype=torch.float32, device=device)
    bb_max = torch.tensor(bounds_max, dtype=torch.float32, device=device)
    bounds_min_2d = torch.stack([bb_min[u_idx], bb_min[v_idx]])
    bounds_max_2d = torch.stack([bb_max[u_idx], bb_max[v_idx]])

    centers, half_size = _make_grid_2d(bounds_min_2d, bounds_max_2d, base_res)

    # First SDF eval to determine batch size
    query_3d = _build_query_3d(centers, station, axis_idx, u_idx, v_idx)
    sdf = shape(query_3d)  # [B, N]
    B = sdf.shape[0]

    inside_centers_list: list[Tensor] = []
    inside_half_sizes_list: list[Tensor] = []
    inside_masks_list: list[Tensor] = []

    active = torch.ones(B, centers.shape[0], dtype=torch.bool, device=device)

    # Defaults for early-break case
    boundary_centers = torch.empty(0, 2, device=device)
    boundary_sdf_out = torch.empty(B, 0, device=device)
    boundary_half_size = half_size.clone()

    for level in range(levels):
        if level > 0:
            query_3d = _build_query_3d(centers, station, axis_idx, u_idx, v_idx)
            sdf = shape(query_3d)

        half_diag = half_size.norm()
        r = lipschitz * half_diag

        # Per-batch Lipschitz classification
        inside = active & (sdf < -r)  # [B, N]
        outside = active & (sdf > r)  # [B, N]

        # Store cells that are inside for at least one batch element
        any_inside = inside.any(dim=0)  # [N]
        if any_inside.any():
            idx = any_inside.nonzero(as_tuple=True)[0]
            inside_centers_list.append(centers[idx])
            inside_half_sizes_list.append(
                half_size.unsqueeze(0).expand(idx.shape[0], -1).clone()
            )
            inside_masks_list.append(inside[:, idx])

        # Deactivate resolved cells per batch element
        resolved = inside | outside
        active = active & ~resolved

        if level == levels - 1:
            # Final level: remaining active cells become boundary
            any_active = active.any(dim=0)  # [N]
            if any_active.any():
                bnd_idx = any_active.nonzero(as_tuple=True)[0]
                boundary_centers = centers[bnd_idx]
                boundary_sdf_out = sdf[:, bnd_idx]
                boundary_half_size = half_size.clone()
        else:
            any_active = active.any(dim=0)  # [N]
            if not any_active.any():
                # All cells resolved, set boundary half_size to finest level
                remaining = levels - 1 - level
                boundary_half_size = half_size / (2**remaining)
                break

            # Subdivide cells that are mixed for any batch element
            mixed_idx = any_active.nonzero(as_tuple=True)[0]
            centers = _subdivide_cells_2d(centers[mixed_idx], half_size)
            active = active[:, mixed_idx].repeat_interleave(4, dim=1)
            half_size = half_size / 2

    # Concatenate inside cells from all levels
    if inside_centers_list:
        all_inside_centers = torch.cat(inside_centers_list, dim=0)
        all_inside_half_sizes = torch.cat(inside_half_sizes_list, dim=0)
        all_inside_masks = torch.cat(inside_masks_list, dim=1)
    else:
        all_inside_centers = torch.empty(0, 2, device=device)
        all_inside_half_sizes = torch.empty(0, 2, device=device)
        all_inside_masks = torch.empty(B, 0, dtype=torch.bool, device=device)

    return QuadtreeResult(
        inside_centers=all_inside_centers,
        inside_half_sizes=all_inside_half_sizes,
        inside_mask=all_inside_masks,
        boundary_centers=boundary_centers,
        boundary_sdf=boundary_sdf_out,
        boundary_half_size=boundary_half_size,
        axis=axis,
        station=station,
        axis_u=axis_u,
        axis_v=axis_v,
    )
