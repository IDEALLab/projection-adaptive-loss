"""Structural cross-section properties from SDF shapes.

Computes area, centroid, second moments of area, principal moments,
radii of gyration, maximum first moments (for shear), and optionally
the torsion constant J (via Prandtl stress function PDE solve) at
slice planes through 3D shapes. Uses the sparse 2D quadtree for
efficient adaptive SDF evaluation.
"""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

from ..core import Shape
from .quadtree import QuadtreeResult, _axis_mapping, quadtree_sdf_eval


@dataclass
class SectionProperties:
    """Cross-section structural properties at one or more stations.

    All tensor fields have shape [B, S] where B = batch size,
    S = number of stations.

    Attributes:
        A: Cross-section area
        u_cg: Centroid position along in-plane u axis
        v_cg: Centroid position along in-plane v axis
        I_uu: Second moment of area about centroidal u axis
        I_vv: Second moment of area about centroidal v axis
        I_uv: Product of inertia about centroid
        I_1: Maximum principal second moment
        I_2: Minimum principal second moment
        theta_p: Principal axis angle [rad] from u axis to I_1 direction
        r_u: Radius of gyration about u axis
        r_v: Radius of gyration about v axis
        Q_u_max: Maximum first moment about u axis (for vertical shear)
        Q_v_max: Maximum first moment about v axis (for horizontal shear)
        J: Torsion constant (Saint-Venant). None if compute_torsion=False.
            Computed via Prandtl stress function PDE: ∇²φ = -2, φ = 0 on boundary.
            NOT differentiable, detached from computation graph. Attempting
            to backprop through J will raise RuntimeError.
        axis: Slicing axis name
        axis_u: In-plane u axis name
        axis_v: In-plane v axis name
        stations: [S] station positions along slicing axis
    """

    A: Tensor
    u_cg: Tensor
    v_cg: Tensor
    I_uu: Tensor
    I_vv: Tensor
    I_uv: Tensor
    I_1: Tensor
    I_2: Tensor
    theta_p: Tensor
    r_u: Tensor
    r_v: Tensor
    Q_u_max: Tensor
    Q_v_max: Tensor
    J: Tensor | None
    axis: str
    axis_u: str
    axis_v: str
    stations: Tensor


def _rasterize_quadtree(
    result: QuadtreeResult,
    bounds_min_2d: Tensor,
    bounds_max_2d: Tensor,
    res: int,
    B: int,
    epsilon: float,
) -> Tensor:
    """Convert quadtree result to a uniform boolean mask at finest resolution.

    Args:
        result: QuadtreeResult from quadtree_sdf_eval
        bounds_min_2d: [2] lower bounds in slice plane
        bounds_max_2d: [2] upper bounds in slice plane
        res: Grid resolution (cells per axis at finest level)
        B: Batch size
        epsilon: Sigmoid sharpness for boundary cell classification

    Returns:
        [B, res, res] bool mask, True for cells inside the shape
    """
    device = result.boundary_half_size.device
    mask = torch.zeros(B, res, res, dtype=torch.bool, device=device)

    domain_size = bounds_max_2d - bounds_min_2d  # [2]
    cell_size = domain_size / res  # [2]

    # Rasterize inside cells (varying sizes -> multiple fine cells each)
    M_in = result.inside_centers.shape[0]
    if M_in > 0:
        lo = result.inside_centers - result.inside_half_sizes  # [M_in, 2]
        hi = result.inside_centers + result.inside_half_sizes  # [M_in, 2]

        i_lo = ((lo[:, 0] - bounds_min_2d[0]) / cell_size[0]).long().clamp(0, res - 1)
        i_hi = ((hi[:, 0] - bounds_min_2d[0]) / cell_size[0]).ceil().long().clamp(0, res)
        j_lo = ((lo[:, 1] - bounds_min_2d[1]) / cell_size[1]).long().clamp(0, res - 1)
        j_hi = ((hi[:, 1] - bounds_min_2d[1]) / cell_size[1]).ceil().long().clamp(0, res)

        for k in range(M_in):
            for b in range(B):
                if result.inside_mask[b, k]:
                    mask[b, i_lo[k] : i_hi[k], j_lo[k] : j_hi[k]] = True

    # Rasterize boundary cells (finest level -> one fine cell each)
    M_bnd = result.boundary_centers.shape[0]
    if M_bnd > 0:
        occ = torch.sigmoid(-result.boundary_sdf / epsilon)  # [B, M_bnd]
        inside_bnd = occ > 0.5  # [B, M_bnd]

        i_bnd = (
            (result.boundary_centers[:, 0] - bounds_min_2d[0]) / cell_size[0]
        ).long().clamp(0, res - 1)
        j_bnd = (
            (result.boundary_centers[:, 1] - bounds_min_2d[1]) / cell_size[1]
        ).long().clamp(0, res - 1)

        for k in range(M_bnd):
            for b in range(B):
                if inside_bnd[b, k]:
                    mask[b, i_bnd[k], j_bnd[k]] = True

    return mask


def _solve_poisson_sor(
    mask: Tensor,
    hu: float,
    hv: float,
    max_iter: int = 2000,
    tol: float = 1e-6,
) -> Tensor:
    """Solve ∇²φ = -2 on masked domain with φ = 0 outside via Red-Black SOR.

    Uses the Prandtl stress function formulation for the Saint-Venant
    torsion problem. The torsion constant is J = 2 ∫∫ φ dA.

    NOT differentiable. The iterative solver does not propagate gradients.
    Output is detached from the computation graph.

    Args:
        mask: [B, H, W] bool, True for interior cells
        hu: Grid spacing in u direction
        hv: Grid spacing in v direction
        max_iter: Maximum SOR iterations
        tol: Convergence tolerance on max residual

    Returns:
        [B, H, W] stress function φ (detached, no grad)
    """
    B, H, W = mask.shape
    device = mask.device
    phi = torch.zeros(B, H, W, device=device)
    mask_f = mask.float()

    au = 1.0 / (hu * hu)
    av = 1.0 / (hv * hv)
    diag = 2.0 * (au + av)

    # Near-optimal SOR relaxation parameter
    N = max(H, W)
    omega = 2.0 / (1.0 + math.sin(math.pi / N)) if N > 2 else 1.0

    # Red-black checkerboard masks [H, W]
    rows = torch.arange(H, device=device).unsqueeze(1)
    cols = torch.arange(W, device=device).unsqueeze(0)
    red = (rows + cols) % 2 == 0  # [H, W]
    black = ~red

    for it in range(max_iter):
        for color in [red, black]:
            # Neighbor values (zero-padded at domain edges = φ=0 BC)
            u_plus = F.pad(phi[:, 1:, :], (0, 0, 0, 1))  # φ[i+1, j]
            u_minus = F.pad(phi[:, :-1, :], (0, 0, 1, 0))  # φ[i-1, j]
            v_plus = F.pad(phi[:, :, 1:], (0, 1))  # φ[i, j+1]
            v_minus = F.pad(phi[:, :, :-1], (1, 0))  # φ[i, j-1]

            # Gauss-Seidel target
            phi_gs = (
                au * (u_plus + u_minus) + av * (v_plus + v_minus) + 2.0
            ) / diag

            # SOR update at colored interior cells only
            update = color.unsqueeze(0) & mask  # [B, H, W]
            phi = torch.where(update, phi + omega * (phi_gs - phi), phi)

        # Check convergence every 50 iterations
        if it % 50 == 49:
            u_plus = F.pad(phi[:, 1:, :], (0, 0, 0, 1))
            u_minus = F.pad(phi[:, :-1, :], (0, 0, 1, 0))
            v_plus = F.pad(phi[:, :, 1:], (0, 1))
            v_minus = F.pad(phi[:, :, :-1], (1, 0))

            laplacian = au * (u_plus - 2 * phi + u_minus) + av * (
                v_plus - 2 * phi + v_minus
            )
            residual = ((laplacian + 2.0) * mask_f).abs().max().item()
            if residual < tol:
                break

    return phi


def _compute_torsion_constant(
    result: QuadtreeResult,
    bounds_min_2d: Tensor,
    bounds_max_2d: Tensor,
    res: int,
    B: int,
    epsilon: float,
    poisson_tol: float = 1e-6,
    poisson_max_iter: int = 2000,
) -> Tensor:
    """Compute torsion constant J from quadtree result via Poisson solve.

    Args:
        result: QuadtreeResult from quadtree_sdf_eval
        bounds_min_2d: [2] lower bounds in slice plane
        bounds_max_2d: [2] upper bounds in slice plane
        res: Grid resolution at finest quadtree level
        B: Batch size
        epsilon: Sigmoid sharpness for boundary classification
        poisson_tol: Convergence tolerance for Poisson solver
        poisson_max_iter: Maximum solver iterations

    Returns:
        [B] torsion constant J per batch element
    """
    domain_size = bounds_max_2d - bounds_min_2d
    hu = (domain_size[0] / res).item()
    hv = (domain_size[1] / res).item()
    cell_area = hu * hv

    # Rasterize quadtree to uniform grid
    mask = _rasterize_quadtree(
        result, bounds_min_2d, bounds_max_2d, res, B, epsilon
    )

    # Handle empty sections
    if not mask.any():
        return torch.zeros(B, device=result.boundary_half_size.device)

    # Solve Prandtl stress function PDE (not differentiable)
    with torch.no_grad():
        phi = _solve_poisson_sor(mask, hu, hv, poisson_max_iter, poisson_tol)

    # J = 2 ∫∫ φ dA, detached, no gradient graph
    J = 2.0 * (phi * mask.float()).sum(dim=(-2, -1)) * cell_area

    return J.detach()


def _compute_section_from_quadtree(
    result: QuadtreeResult, epsilon: float
) -> dict[str, Tensor]:
    """Compute section properties from a single station's quadtree result.

    Args:
        result: QuadtreeResult from quadtree_sdf_eval
        epsilon: Sigmoid sharpness for boundary cell occupancy

    Returns:
        Dict of [B] tensors: A, u_cg, v_cg, I_uu, I_vv, I_uv, Q_u_max, Q_v_max
    """
    device = result.boundary_half_size.device

    # Determine batch size
    if result.inside_mask.numel() > 0:
        B = result.inside_mask.shape[0]
    else:
        B = result.boundary_sdf.shape[0]

    # Accumulators for pass 1 (area + centroid)
    A = torch.zeros(B, device=device)
    Su = torch.zeros(B, device=device)
    Sv = torch.zeros(B, device=device)

    # Pass 1: Area and centroid

    # Inside cells (varying sizes, per-batch mask)
    M_in = result.inside_centers.shape[0]
    if M_in > 0:
        hs = result.inside_half_sizes  # [M_in, 2]
        cell_areas = 4 * hs[:, 0] * hs[:, 1]  # [M_in]
        u_c = result.inside_centers[:, 0]  # [M_in]
        v_c = result.inside_centers[:, 1]  # [M_in]
        w = result.inside_mask.float()  # [B, M_in]

        A = A + (w * cell_areas.unsqueeze(0)).sum(-1)
        Su = Su + (w * cell_areas.unsqueeze(0) * u_c.unsqueeze(0)).sum(-1)
        Sv = Sv + (w * cell_areas.unsqueeze(0) * v_c.unsqueeze(0)).sum(-1)

    # Boundary cells (uniform fine size, sigmoid occupancy)
    M_bnd = result.boundary_centers.shape[0]
    if M_bnd > 0:
        bnd_area = 4 * result.boundary_half_size[0] * result.boundary_half_size[1]
        occ = torch.sigmoid(-result.boundary_sdf / epsilon)  # [B, M_bnd]
        u_b = result.boundary_centers[:, 0]  # [M_bnd]
        v_b = result.boundary_centers[:, 1]  # [M_bnd]

        A = A + (occ * bnd_area).sum(-1)
        Su = Su + (occ * bnd_area * u_b.unsqueeze(0)).sum(-1)
        Sv = Sv + (occ * bnd_area * v_b.unsqueeze(0)).sum(-1)

    # Centroid (guard against zero area)
    safe_A = torch.where(A > 0, A, torch.ones_like(A))
    u_cg = Su / safe_A
    v_cg = Sv / safe_A

    # Pass 2: Second moments + Q_max (need centroid)

    I_uu = torch.zeros(B, device=device)
    I_vv = torch.zeros(B, device=device)
    I_uv = torch.zeros(B, device=device)
    Q_u = torch.zeros(B, device=device)
    Q_v = torch.zeros(B, device=device)

    # Inside cells, parallel axis theorem
    if M_in > 0:
        du = u_c.unsqueeze(0) - u_cg.unsqueeze(-1)  # [B, M_in]
        dv = v_c.unsqueeze(0) - v_cg.unsqueeze(-1)  # [B, M_in]
        cell_w = 2 * hs[:, 0]  # width in u [M_in]
        cell_h = 2 * hs[:, 1]  # height in v [M_in]

        # I_self_uu = w * h³ / 12, I_self_vv = h * w³ / 12
        I_self_uu = (cell_w * cell_h**3 / 12).unsqueeze(0)  # [1, M_in]
        I_self_vv = (cell_h * cell_w**3 / 12).unsqueeze(0)  # [1, M_in]
        ca = cell_areas.unsqueeze(0)  # [1, M_in]

        I_uu = I_uu + (w * (I_self_uu + ca * dv**2)).sum(-1)
        I_vv = I_vv + (w * (I_self_vv + ca * du**2)).sum(-1)
        I_uv = I_uv + (w * ca * du * dv).sum(-1)

        # Q_max: first moment of area above/right of centroid
        above = (v_c.unsqueeze(0) > v_cg.unsqueeze(-1)).float()  # [B, M_in]
        Q_u = Q_u + (w * above * dv * ca).sum(-1)
        right = (u_c.unsqueeze(0) > u_cg.unsqueeze(-1)).float()  # [B, M_in]
        Q_v = Q_v + (w * right * du * ca).sum(-1)

    # Boundary cells, fine resolution, no self-inertia needed
    if M_bnd > 0:
        du_b = u_b.unsqueeze(0) - u_cg.unsqueeze(-1)  # [B, M_bnd]
        dv_b = v_b.unsqueeze(0) - v_cg.unsqueeze(-1)  # [B, M_bnd]

        I_uu = I_uu + (occ * bnd_area * dv_b**2).sum(-1)
        I_vv = I_vv + (occ * bnd_area * du_b**2).sum(-1)
        I_uv = I_uv + (occ * bnd_area * du_b * dv_b).sum(-1)

        above_b = (v_b.unsqueeze(0) > v_cg.unsqueeze(-1)).float()  # [B, M_bnd]
        Q_u = Q_u + (occ * above_b * dv_b * bnd_area).sum(-1)
        right_b = (u_b.unsqueeze(0) > u_cg.unsqueeze(-1)).float()  # [B, M_bnd]
        Q_v = Q_v + (occ * right_b * du_b * bnd_area).sum(-1)

    # Zero out results when area is zero (station outside shape)
    zero_mask = A <= 0
    u_cg = torch.where(zero_mask, torch.zeros_like(u_cg), u_cg)
    v_cg = torch.where(zero_mask, torch.zeros_like(v_cg), v_cg)

    return {
        "A": A,
        "u_cg": u_cg,
        "v_cg": v_cg,
        "I_uu": I_uu,
        "I_vv": I_vv,
        "I_uv": I_uv,
        "Q_u_max": Q_u,
        "Q_v_max": Q_v,
    }


def structural_properties(
    shape: Shape,
    axis: str,
    stations: list[float] | Tensor,
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
    base_res: int = 8,
    levels: int = 4,
    lipschitz: float = 1.0,
    epsilon: float = 1e-3,
    compute_torsion: bool = False,
    poisson_tol: float = 1e-6,
    poisson_max_iter: int = 2000,
    torsion_levels: int | None = None,
) -> SectionProperties:
    """Compute structural cross-section properties at slice stations.

    Evaluates area, centroid, second moments of area (with parallel axis
    theorem for large resolved cells), principal moments, radii of
    gyration, and maximum first moments for shear. Optionally computes
    the torsion constant J via the Prandtl stress function PDE.

    Args:
        shape: Shape object to evaluate
        axis: Slicing axis ('x', 'y', or 'z'). Stations step along this
            axis; the grid lives in the remaining two axes.
        stations: Positions along the slicing axis
        bounds_min: 3D bounding box minimum (x, y, z)
        bounds_max: 3D bounding box maximum (x, y, z)
        base_res: Coarse quadtree grid resolution (cells per axis)
        levels: Number of quadtree refinement levels
        lipschitz: Lipschitz constant of the SDF (default 1.0)
        epsilon: Sigmoid sharpness for boundary cell occupancy
        compute_torsion: If True, solve the Prandtl PDE for torsion constant J
        poisson_tol: Convergence tolerance for Poisson solver (torsion only)
        poisson_max_iter: Maximum Poisson solver iterations (torsion only)
        torsion_levels: If set, rasterize the quadtree to a grid sized as if
            this number of levels had been used (``base_res * 2**(torsion_levels-1)``)
            before running the Poisson solve. Defaults to ``levels`` (no
            downsampling). Smaller values dramatically speed up the Poisson
            solve, cost scales as O(torsion_res^2 * iterations) and SOR
            converges faster on smaller grids.

    Returns:
        SectionProperties with all fields as [B, S] tensors.
        J is None if compute_torsion=False.
    """
    if isinstance(stations, Tensor):
        stations_list = stations.tolist()
        stations_t = stations.float()
    else:
        stations_list = list(stations)
        stations_t = torch.tensor(stations_list, dtype=torch.float32)

    # Precompute 2D bounds for torsion
    axis_idx, u_idx, v_idx, _, _ = _axis_mapping(axis)
    bb_min = torch.tensor(bounds_min, dtype=torch.float32)
    bb_max = torch.tensor(bounds_max, dtype=torch.float32)
    bounds_min_2d = torch.stack([bb_min[u_idx], bb_min[v_idx]])
    bounds_max_2d = torch.stack([bb_max[u_idx], bb_max[v_idx]])
    _torsion_levels = levels if torsion_levels is None else torsion_levels
    if _torsion_levels > levels:
        raise ValueError(
            f"torsion_levels ({_torsion_levels}) must be <= levels ({levels}); "
            f"you cannot rasterize to a finer grid than the quadtree produced."
        )
    torsion_res = base_res * 2 ** (_torsion_levels - 1)

    # Compute properties at each station
    per_station: list[dict[str, Tensor]] = []
    J_per_station: list[Tensor] = []
    axis_u = ""
    axis_v = ""

    for s in stations_list:
        qt = quadtree_sdf_eval(
            shape,
            axis=axis,
            station=s,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            base_res=base_res,
            levels=levels,
            lipschitz=lipschitz,
        )
        axis_u = qt.axis_u
        axis_v = qt.axis_v
        props = _compute_section_from_quadtree(qt, epsilon)
        per_station.append(props)

        if compute_torsion:
            B = props["A"].shape[0]
            J_val = _compute_torsion_constant(
                qt, bounds_min_2d, bounds_max_2d, torsion_res, B, epsilon,
                poisson_tol, poisson_max_iter,
            )
            J_per_station.append(J_val)

    # Stack along station dimension: [B] -> [B, S]
    keys = ["A", "u_cg", "v_cg", "I_uu", "I_vv", "I_uv", "Q_u_max", "Q_v_max"]
    stacked = {
        k: torch.stack([ps[k] for ps in per_station], dim=-1) for k in keys
    }

    # Derived quantities
    I_uu = stacked["I_uu"]
    I_vv = stacked["I_vv"]
    I_uv = stacked["I_uv"]
    A = stacked["A"]

    # Principal moments (eigenvalues of 2x2 inertia matrix)
    avg = (I_uu + I_vv) / 2
    diff = (I_uu - I_vv) / 2
    R = torch.sqrt(diff**2 + I_uv**2)
    I_1 = avg + R
    I_2 = avg - R

    # Principal angle
    theta_p = 0.5 * torch.atan2(-2 * I_uv, I_uu - I_vv)

    # Radii of gyration (guard against zero area)
    safe_A = torch.where(A > 0, A, torch.ones_like(A))
    r_u = torch.sqrt(I_uu / safe_A)
    r_v = torch.sqrt(I_vv / safe_A)
    # Zero out when area is zero
    zero_mask = A <= 0
    r_u = torch.where(zero_mask, torch.zeros_like(r_u), r_u)
    r_v = torch.where(zero_mask, torch.zeros_like(r_v), r_v)

    # Torsion constant
    J: Tensor | None = None
    if compute_torsion and J_per_station:
        J = torch.stack(J_per_station, dim=-1)  # [B, S]

    return SectionProperties(
        A=stacked["A"],
        u_cg=stacked["u_cg"],
        v_cg=stacked["v_cg"],
        I_uu=I_uu,
        I_vv=I_vv,
        I_uv=I_uv,
        I_1=I_1,
        I_2=I_2,
        theta_p=theta_p,
        r_u=r_u,
        r_v=r_v,
        Q_u_max=stacked["Q_u_max"],
        Q_v_max=stacked["Q_v_max"],
        J=J,
        axis=axis,
        axis_u=axis_u,
        axis_v=axis_v,
        stations=stations_t,
    )
