"""Isocontour extraction via adaptive quadtree + marching squares.

Extracts 2D contour polylines (1D line meshes) at slice planes through
3D SDF shapes. Each vertex has an outward normal and arc-length quadrature
weight, ready for downstream integration (e.g. Cp surface loads).

Pipeline:
    quadtree (corner-based, Lipschitz-pruned)
    -> marching squares (16-case lookup, linear edge interpolation)
    -> segment chaining (endpoint hashing, CCW orientation)
    -> normals + weights (tangent rotation, arc-length)

Provides:
    isocontour_extract(): Batched extraction across B geometries and S stations
    IsocontourResult: Dataclass with per-batch polylines, normals, weights
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from ..core import Shape
from .quadtree import _axis_mapping

# Constants

# Corner layout:       Edge layout:
#   3 --- 2             e3
#   |     |           --+--
#   0 --- 1          e0|   |e2
# +
#                       e1
_CORNER_OFFSETS = torch.tensor([[0, 0], [1, 0], [1, 1], [0, 1]])
_EDGE_VERTS = torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]])

# Marching squares lookup table.
# Case index = sum of (corner_i > 0) << i for i in 0..3.
# Each case maps to a list of (edge_a, edge_b) pairs forming segments.
# fmt: off
_MS_TABLE: list[list[tuple[int, int]]] = [
    [],              # 0:  ----
    [(0, 3)],        # 1:  +---
    [(0, 1)],        # 2:  -+--
    [(1, 3)],        # 3:  ++--
    [(1, 2)],        # 4:  --+-
    [(0, 1), (2, 3)],# 5:  +-+- (ambiguous)
    [(0, 2)],        # 6:  -++-
    [(2, 3)],        # 7:  +++-
    [(2, 3)],        # 8:  ---+
    [(0, 2)],        # 9:  +--+
    [(0, 3), (1, 2)],# 10: -+-+ (ambiguous)
    [(1, 2)],        # 11: ++-+
    [(1, 3)],        # 12: --++
    [(0, 1)],        # 13: +-++
    [(0, 3)],        # 14: -+++
    [],              # 15: ++++
]
# fmt: on

# Pre-encode for vectorized marching squares:
# For each of the 16 cases, the number of segments (0, 1, or 2)
# and the edge pairs.
_MS_NUM_SEGS = torch.tensor([len(c) for c in _MS_TABLE])  # [16]
_MS_EDGES_A = torch.zeros(16, 2, dtype=torch.long)
_MS_EDGES_B = torch.zeros(16, 2, dtype=torch.long)
for _ci, _pairs in enumerate(_MS_TABLE):
    for _si, (_ea, _eb) in enumerate(_pairs):
        _MS_EDGES_A[_ci, _si] = _ea
        _MS_EDGES_B[_ci, _si] = _eb

_PLANE_TO_AXIS = {"xy": "z", "xz": "y", "yz": "x"}


# Result type


@dataclass
class IsocontourResult:
    """Isocontour extraction result at a single slice station.

    Per-batch polyline lists because each batch element may produce
    different numbers of contours with different vertex counts.
    Follows the MeshResult pattern (list[Tensor] for variable-size
    per-batch data).

    Attributes:
        polylines: polylines[b] is a list of [K_i, 2] tensors (CCW, no dup endpoint)
        normals: normals[b] is a list of [K_i, D] outward unit normals, where
            D=2 for normal_mode='in_plane' or D=3 for normal_mode='3d'
        weights: weights[b] is a list of [K_i] arc-length quadrature weights
        lengths: [B] total contour length per batch element
        axis: slicing axis name ('x', 'y', or 'z')
        station: position along the slicing axis
        axis_u: in-plane u axis name
        axis_v: in-plane v axis name
        normal_mode: 'in_plane' or '3d'
    """

    polylines: list[list[Tensor]]
    normals: list[list[Tensor]]
    weights: list[list[Tensor]]
    lengths: Tensor
    axis: str
    station: float
    axis_u: str
    axis_v: str
    normal_mode: str

    def visualize(
        self,
        batch: int = 0,
        show_normals: bool = False,
        ax=None,
    ):
        """Plot the isocontour on the 2D slice plane.

        Args:
            batch: Which batch element to show (default 0).
            show_normals: If True, overlay outward normal arrows.
            ax: Optional matplotlib Axes. If None, creates a new figure.

        Returns:
            The matplotlib Axes used for plotting.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(1, 1, figsize=(8, 6))

        polys = self.polylines[batch]
        norms = self.normals[batch]

        for i, poly in enumerate(polys):
            pts = poly.detach().cpu().numpy()
            # Close the loop for plotting
            closed = __import__("numpy").concatenate([pts, pts[:1]], axis=0)
            ax.plot(closed[:, 0], closed[:, 1], "-", lw=1.5, label=f"contour {i}")
            ax.plot(pts[:, 0], pts[:, 1], ".", ms=2)

            if show_normals and i < len(norms):
                n = norms[i].detach().cpu().numpy()
                step = max(1, len(pts) // 30)
                # For 3D normals, project onto the slice plane using axis indices
                if n.shape[1] == 3:
                    _, u_idx, v_idx, _, _ = _axis_mapping(self.axis)
                    nu, nv = n[::step, u_idx], n[::step, v_idx]
                else:
                    nu, nv = n[::step, 0], n[::step, 1]
                ax.quiver(
                    pts[::step, 0],
                    pts[::step, 1],
                    nu,
                    nv,
                    scale=25,
                    width=0.003,
                    color="#AAAAAA",
                )

        ax.set_aspect("equal")
        ax.set_xlabel(self.axis_u)
        ax.set_ylabel(self.axis_v)
        ax.set_title(
            f"Isocontour at {self.axis}={self.station:.3f} "
            f"(batch {batch}, L={self.lengths[batch]:.4f})"
        )
        return ax


# Corner deduplication


def _unique_int_coords_2d(
    coords: Tensor,
    max_coord: int,
) -> tuple[Tensor, Tensor]:
    """Deduplicate rows of 2D integer coordinates using 1D scalar keys.

    Same pattern as sparsedmc's _unique_int_coords but for [N, 2].
    Avoids broken torch.unique(dim=0) on MPS.

    Args:
        coords: [N, 2] int tensor with values in [0, max_coord].
        max_coord: Maximum coordinate value (inclusive).

    Returns:
        unique_coords: [M, 2] unique rows (same dtype as input).
        inverse_idx: [N] int64 mapping from input rows to unique rows.
    """
    R = max_coord + 1
    keys = coords[:, 0].long() * R + coords[:, 1].long()
    unique_keys, inverse_idx = torch.unique(keys, return_inverse=True)
    unique_coords = torch.stack([unique_keys // R, unique_keys % R], dim=1).to(
        coords.dtype
    )
    return unique_coords, inverse_idx


# Corner-based quadtree


def _quadtree_corners(
    shape: Shape,
    axis_idx: int,
    u_idx: int,
    v_idx: int,
    stations: list[float],
    bounds_min_2d: Tensor,
    bounds_max_2d: Tensor,
    base_res: int = 8,
    levels: int = 5,
    lipschitz: float = 1.0,
) -> tuple[Tensor, Tensor, int]:
    """Corner-based adaptive quadtree batched across B and S stations.

    Evaluates SDF at grid corners (not centers), sharing corners between
    adjacent cells. Quadtree structure is shared across all B batch
    elements and all S stations (union of active cells).

    Args:
        shape: Shape object to evaluate.
        axis_idx: Index of the slicing axis (0, 1, or 2).
        u_idx: Index of the u (first in-plane) axis.
        v_idx: Index of the v (second in-plane) axis.
        stations: Positions along the slicing axis.
        bounds_min_2d: [2] lower bounds in (u, v).
        bounds_max_2d: [2] upper bounds in (u, v).
        base_res: Cells per axis at level 0.
        levels: Number of refinement levels.
        lipschitz: Lipschitz constant.

    Returns:
        cell_corner_positions: [M_bnd, 4, 2] world-space corners of boundary cells.
        corner_sdf: [B, S, M_bnd, 4] SDF values at corners.
        B: batch size.
    """
    device = bounds_min_2d.device
    domain_size = bounds_max_2d - bounds_min_2d
    S = len(stations)
    stations_t = torch.tensor(stations, dtype=torch.float32, device=device)

    corner_offsets = _CORNER_OFFSETS.to(device)

    # Level 0: all cells as integer (ix, iy) indices
    ix = torch.arange(base_res, device=device)
    cell_indices = torch.stack(torch.meshgrid(ix, ix, indexing="ij"), dim=-1).reshape(
        -1, 2
    )

    B = None

    for level in range(levels):
        res = base_res * (2**level)
        cell_size = domain_size / res

        M = cell_indices.shape[0]

        # All corner indices: [M, 4, 2]
        all_corner_idx = cell_indices[:, None, :] + corner_offsets[None, :, :]
        flat_corners = all_corner_idx.reshape(-1, 2)  # [M*4, 2]

        # Deduplicate corners
        unique_corners, inverse_idx = _unique_int_coords_2d(flat_corners, max_coord=res)
        C = unique_corners.shape[0]

        # Convert to world-space 2D positions
        unique_pos_2d = bounds_min_2d + unique_corners.float() * cell_size  # [C, 2]

        # Build 3D query points for ALL stations: [C*S, 3]
        # Repeat each 2D corner S times, with different station values
        pos_2d_rep = unique_pos_2d.repeat(S, 1)  # [C*S, 2]
        station_rep = stations_t.repeat_interleave(C)  # [C*S]
        query_3d = torch.zeros(C * S, 3, dtype=torch.float32, device=device)
        query_3d[:, u_idx] = pos_2d_rep[:, 0]
        query_3d[:, v_idx] = pos_2d_rep[:, 1]
        query_3d[:, axis_idx] = station_rep

        # Evaluate SDF: [B, C*S]
        sdf_flat = shape(query_3d)
        B = sdf_flat.shape[0]

        # Reshape to [B, S, C]
        sdf = sdf_flat.reshape(B, S, C)

        # Gather SDF at cell corners: [B, S, M, 4]
        # inverse_idx: [M*4] -> unique corner index
        inv = inverse_idx.long()  # [M*4]
        corner_sdf = sdf[:, :, inv].reshape(B, S, M, 4)

        # Lipschitz classification
        half_diag = cell_size.norm() / 2
        threshold = lipschitz * half_diag * 2  # full diagonal

        # min |sdf| across 4 corners: [B, S, M]
        min_abs = corner_sdf.abs().min(dim=-1).values

        # Cell is boundary if ANY (b, s) pair has min_abs < threshold
        any_boundary = (min_abs < threshold).any(dim=0).any(dim=0)  # [M]

        if level == levels - 1:
            # Final level: return boundary cells
            bnd_mask = any_boundary
            if not bnd_mask.any():
                # No boundary cells at all
                empty_pos = torch.empty(0, 4, 2, device=device)
                empty_sdf = torch.empty(B, S, 0, 4, device=device)
                return empty_pos, empty_sdf, B

            bnd_idx = bnd_mask.nonzero(as_tuple=True)[0]
            bnd_cell_indices = cell_indices[bnd_idx]  # [M_bnd, 2]

            # Corner positions in world space: [M_bnd, 4, 2]
            bnd_corner_idx = bnd_cell_indices[:, None, :] + corner_offsets[None, :, :]
            bnd_corner_pos = bounds_min_2d + bnd_corner_idx.float() * cell_size

            # Corner SDF: [B, S, M_bnd, 4]
            bnd_corner_sdf = corner_sdf[:, :, bnd_idx, :]

            return bnd_corner_pos, bnd_corner_sdf, B

        else:
            # Subdivide boundary cells
            if not any_boundary.any():
                # All resolved, return empty
                empty_pos = torch.empty(0, 4, 2, device=device)
                empty_sdf = torch.empty(B, S, 0, 4, device=device)
                return empty_pos, empty_sdf, B

            bnd_idx = any_boundary.nonzero(as_tuple=True)[0]
            parent_cells = cell_indices[bnd_idx]  # [P, 2]

            # Each cell (ix, iy) at level L becomes 4 children at L+1
            child_offsets = torch.tensor(
                [[0, 0], [1, 0], [1, 1], [0, 1]], device=device
            )
            children = parent_cells[:, None, :] * 2 + child_offsets[None, :, :]
            cell_indices = children.reshape(-1, 2)

    # Should not reach here
    empty_pos = torch.empty(0, 4, 2, device=device)
    empty_sdf = torch.empty(B or 1, S, 0, 4, device=device)
    return empty_pos, empty_sdf, B or 1


# Marching squares


def _marching_squares_segments(
    cell_corners: Tensor,
    cell_sdf: Tensor,
) -> Tensor:
    """Vectorized marching squares for a single (batch, station) pair.

    Args:
        cell_corners: [M, 4, 2] world-space corner positions.
        cell_sdf: [M, 4] SDF values at corners.

    Returns:
        segments: [S, 2, 2] line segments (start_xy, end_xy).
    """
    if cell_corners.shape[0] == 0:
        return torch.empty(0, 2, 2, device=cell_corners.device)

    M = cell_corners.shape[0]
    device = cell_corners.device

    # Case index per cell
    signs = (cell_sdf > 0).int()
    case_idx = signs[:, 0] + 2 * signs[:, 1] + 4 * signs[:, 2] + 8 * signs[:, 3]

    # Precompute all 4 edge crossings for all M cells
    edge_verts = _EDGE_VERTS.to(device)  # [4, 2]
    crossings = torch.zeros(4, M, 2, device=device)
    for e in range(4):
        i, j = edge_verts[e, 0].item(), edge_verts[e, 1].item()
        si = cell_sdf[:, i]  # [M]
        sj = cell_sdf[:, j]  # [M]
        t = si / (si - sj + 1e-12)  # [M]
        crossings[e] = cell_corners[:, i] + t[:, None] * (
            cell_corners[:, j] - cell_corners[:, i]
        )

    # Gather segments by iterating over the 16 cases
    seg_list: list[Tensor] = []
    for ci in range(16):
        pairs = _MS_TABLE[ci]
        if not pairs:
            continue
        mask = case_idx == ci
        if not mask.any():
            continue
        for ea, eb in pairs:
            pa = crossings[ea][mask]  # [K, 2]
            pb = crossings[eb][mask]  # [K, 2]
            seg = torch.stack([pa, pb], dim=1)  # [K, 2, 2]
            seg_list.append(seg)

    if not seg_list:
        return torch.empty(0, 2, 2, device=device)

    return torch.cat(seg_list, dim=0)


# Segment chaining


def _chain_segments(
    segments: Tensor,
    tol: float | None = None,
) -> list[Tensor]:
    """Chain unordered segments into closed polylines (CCW oriented).

    Preserves gradient flow: topology discovery uses numpy on detached
    copies, but output polylines are gathered from the original
    differentiable ``segments`` tensor.

    Args:
        segments: [S, 2, 2] line segments on the original device.
        tol: Distance tolerance for endpoint matching. If None, auto-computed
            as 1e-6 times the segment bounding box diagonal.

    Returns:
        List of [K_i, 2] closed polylines (CCW, no duplicate endpoint).
    """
    if segments.shape[0] == 0:
        return []

    import numpy as np

    # Detached numpy copy for topology discovery only
    segs = segments.detach().cpu().numpy()
    S = len(segs)

    # Auto-compute tolerance from data extent
    if tol is None:
        all_pts = segs.reshape(-1, 2)
        extent = all_pts.max(axis=0) - all_pts.min(axis=0)
        tol = max(float(extent.max()), 1e-12) * 1e-6

    # Hash endpoints
    def _key(pt):
        return (round(pt[0] / tol), round(pt[1] / tol))

    adj: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for si in range(S):
        for end in range(2):
            k = _key(segs[si, end])
            adj.setdefault(k, []).append((si, end))

    used = [False] * S
    polylines: list[Tensor] = []

    for seed in range(S):
        if used[seed]:
            continue

        # Track (segment_idx, endpoint_idx), topology only
        chain_indices: list[tuple[int, int]] = [(seed, 0), (seed, 1)]
        used[seed] = True

        # Walk forward
        while True:
            last_seg, last_end = chain_indices[-1]
            tail = segs[last_seg, last_end]
            k = _key(tail)
            neighbors = adj.get(k, [])
            found = False
            for si, end in neighbors:
                if used[si]:
                    continue
                other = 1 - end
                chain_indices.append((si, other))
                used[si] = True
                found = True
                break
            if not found:
                break

        # Check close/CCW on detached numpy values
        pts_np = np.array([segs[i, e] for i, e in chain_indices])
        if np.linalg.norm(pts_np[-1] - pts_np[0]) < tol * 10:
            chain_indices = chain_indices[:-1]
            pts_np = pts_np[:-1]

        x, y = pts_np[:, 0], pts_np[:, 1]
        x_next = np.roll(x, -1)
        y_next = np.roll(y, -1)
        area = 0.5 * np.sum(x * y_next - x_next * y)
        needs_flip = area < 0

        # Gather from original differentiable tensor
        seg_idxs = torch.tensor(
            [i for i, _ in chain_indices], dtype=torch.long,
        )
        end_idxs = torch.tensor(
            [e for _, e in chain_indices], dtype=torch.long,
        )
        polyline = segments[seg_idxs, end_idxs]  # [K, 2], differentiable

        if needs_flip:
            polyline = torch.flip(polyline, dims=[0])

        polylines.append(polyline)

    return polylines


# Normals and weights


def _contour_normals_weights(
    polyline: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute outward normals and arc-length weights for a closed polyline.

    Args:
        polyline: [K, 2] closed polyline (CCW, no duplicate endpoint).

    Returns:
        normals: [K, 2] outward unit normals.
        weights: [K] arc-length quadrature weights.
        length: scalar total arc length.
    """
    K = polyline.shape[0]
    if K < 3:
        return (
            torch.zeros_like(polyline),
            torch.zeros(K, device=polyline.device),
            torch.tensor(0.0, device=polyline.device),
        )

    # Indices for closed loop
    next_idx = torch.arange(1, K + 1, device=polyline.device) % K
    prev_idx = torch.arange(-1, K - 1, device=polyline.device) % K

    # Edge vectors and lengths
    edges = polyline[next_idx] - polyline  # [K, 2]
    edge_lengths = edges.norm(dim=1)  # [K]

    # Arc-length weights: w_i = (|e_{i-1}| + |e_i|) / 2
    prev_edge_lengths = edge_lengths[prev_idx]
    weights = (prev_edge_lengths + edge_lengths) / 2

    total_length = edge_lengths.sum()

    # Tangent via central difference
    tangent = polyline[next_idx] - polyline[prev_idx]  # [K, 2]
    tangent_norm = tangent.norm(dim=1, keepdim=True).clamp(min=1e-12)
    tangent = tangent / tangent_norm

    # Normal = rotate tangent 90 deg (outward for CCW)
    normals = torch.stack([tangent[:, 1], -tangent[:, 0]], dim=1)

    return normals, weights, total_length


# 3D autograd normals


def _compute_3d_normals(
    shape: Shape,
    polylines_2d: list[Tensor],
    station: float,
    axis_idx: int,
    u_idx: int,
    v_idx: int,
    batch_idx: int,
) -> list[Tensor]:
    """Compute true 3D surface normals via autograd on the SDF.

    Lifts 2D contour points to 3D, evaluates the SDF, and uses
    ``torch.autograd.grad`` with ``create_graph=True`` to get spatial
    gradients (= surface normals). The computation graph is preserved
    so that downstream backprop can flow through to shape parameters.

    Args:
        shape: Shape object to evaluate.
        polylines_2d: List of [K_i, 2] contour polylines in the slice plane.
        station: Position along the slicing axis.
        axis_idx: Index of the slicing axis (0=x, 1=y, 2=z).
        u_idx: Index of the in-plane u axis.
        v_idx: Index of the in-plane v axis.
        batch_idx: Which batch element these polylines belong to.

    Returns:
        List of [K_i, 3] unit normal tensors, one per polyline.
    """
    if not polylines_2d:
        return []

    # Concatenate all polylines for one batched eval
    sizes = [p.shape[0] for p in polylines_2d]
    pts_2d = torch.cat(polylines_2d, dim=0)  # [K_total, 2]
    K = pts_2d.shape[0]
    device = pts_2d.device

    # autograd.grad needs a live graph; callers (e.g. eval trajectory loop)
    # may invoke isocontour inside `with torch.no_grad():`, which would
    # silently disable the requires_grad below.
    with torch.enable_grad():
        # Lift to 3D
        pts_3d = torch.zeros(K, 3, device=device)
        pts_3d[:, u_idx] = pts_2d[:, 0]
        pts_3d[:, v_idx] = pts_2d[:, 1]
        pts_3d[:, axis_idx] = station
        pts_3d = pts_3d.requires_grad_(True)

        # Forward through SDF: shape expects [B, N, 3] -> [B, N]
        # Input [1, K, 3] broadcasts to [B, K, 3]; select the right batch row
        sdf = shape(pts_3d.unsqueeze(0))  # [B, K]
        sdf_b = sdf[batch_idx]  # [K]

        # Spatial gradient with graph kept alive for outer backward
        (grad,) = torch.autograd.grad(
            sdf_b.sum(),
            pts_3d,
            create_graph=True,
        )

        # Normalize to unit normals
        grad_norm = grad.norm(dim=1, keepdim=True).clamp(min=1e-12)
        normals = grad / grad_norm  # [K, 3]

    # Split back per polyline
    return list(normals.split(sizes, dim=0))


# Main extraction function


_VALID_NORMAL_MODES = ("in_plane", "3d")


def isocontour_extract(
    shape: Shape,
    axis: str,
    stations: list[float],
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    *,
    normal_mode: str,
    base_res: int = 8,
    levels: int = 5,
    lipschitz: float = 1.0,
) -> list[IsocontourResult]:
    """Extract 2D isocontour polylines at slice planes through a 3D SDF.

    Pipeline: adaptive quadtree (corner-based, batched across B and S)
    -> marching squares -> segment chaining -> normals/weights.

    Args:
        shape: Shape object to evaluate.
        axis: Slicing axis ('x', 'y', or 'z').
        stations: Positions along the slicing axis.
        bounds_min: 3D bounding box minimum (x, y, z).
        bounds_max: 3D bounding box maximum (x, y, z).
        normal_mode: How to compute surface normals. Required.
            'in_plane': [K, 2] normals from tangent rotation in the slice plane.
            '3d': [K, 3] true surface normals via autograd on the SDF
            (differentiable w.r.t. shape parameters).
        base_res: Coarse grid resolution (cells per axis at level 0).
        levels: Number of refinement levels.
        lipschitz: Lipschitz constant (1.0 for true SDFs).

    Returns:
        List of IsocontourResult, one per station.
    """
    if normal_mode not in _VALID_NORMAL_MODES:
        raise ValueError(
            f"normal_mode must be one of {_VALID_NORMAL_MODES}, "
            f"got '{normal_mode}'"
        )

    axis_idx, u_idx, v_idx, axis_u, axis_v = _axis_mapping(axis)

    device = torch.device(shape.device) if shape.device else torch.device("cpu")
    bb_min = torch.tensor(bounds_min, dtype=torch.float32, device=device)
    bb_max = torch.tensor(bounds_max, dtype=torch.float32, device=device)
    bounds_min_2d = torch.stack([bb_min[u_idx], bb_min[v_idx]])
    bounds_max_2d = torch.stack([bb_max[u_idx], bb_max[v_idx]])

    S = len(stations)

    # Step 1: Corner-based quadtree (batched across B and S)
    cell_corner_pos, corner_sdf, B = _quadtree_corners(
        shape,
        axis_idx,
        u_idx,
        v_idx,
        stations,
        bounds_min_2d,
        bounds_max_2d,
        base_res,
        levels,
        lipschitz,
    )
    # cell_corner_pos: [M_bnd, 4, 2]
    # corner_sdf: [B, S, M_bnd, 4]

    # Step 2: Per-(b, s) marching squares + chaining + normals
    results: list[IsocontourResult] = []

    use_3d = normal_mode == "3d"

    for s_idx in range(S):
        all_polylines: list[list[Tensor]] = []
        all_normals: list[list[Tensor]] = []
        all_weights: list[list[Tensor]] = []
        all_lengths: list[Tensor] = []

        for b in range(B):
            sdf_bs = corner_sdf[b, s_idx]  # [M_bnd, 4]

            # Marching squares
            segments = _marching_squares_segments(cell_corner_pos, sdf_bs)

            # Chain segments
            polylines = _chain_segments(segments)

            # Weights (always from arc-length) + in-plane normals
            b_normals_inplane: list[Tensor] = []
            b_weights: list[Tensor] = []
            b_length = torch.tensor(0.0, device=device)

            for poly in polylines:
                n, w, length = _contour_normals_weights(poly)
                b_normals_inplane.append(n)
                b_weights.append(w)
                b_length = b_length + length

            # Normals: in-plane or 3D
            if use_3d:
                b_normals = _compute_3d_normals(
                    shape, polylines, stations[s_idx],
                    axis_idx, u_idx, v_idx, b,
                )
            else:
                b_normals = b_normals_inplane

            all_polylines.append(polylines)
            all_normals.append(b_normals)
            all_weights.append(b_weights)
            all_lengths.append(b_length)

        results.append(
            IsocontourResult(
                polylines=all_polylines,
                normals=all_normals,
                weights=all_weights,
                lengths=torch.stack(all_lengths),
                axis=axis,
                station=stations[s_idx],
                axis_u=axis_u,
                axis_v=axis_v,
                normal_mode=normal_mode,
            )
        )

    return results
