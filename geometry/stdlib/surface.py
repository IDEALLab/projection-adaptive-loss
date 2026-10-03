"""
Bicubic Bezier Surface SDF, signed distance to a parametric surface patch.

Two input modes:

1. **Curve-based**: two CubicBezier2D curves on workplanes, with r1/r2 bulge
   parameters controlling interior shape (r=0 -> ruled, r=1 -> smooth).

2. **Control-net**: explicit [B, 4, 4, 3] control net for full control over
   the bicubic patch (e.g. for nose cones, fairings, free-form surfaces).

API:
    bezier_surface(curve1, curve2, r1=..., r2=..., flip=False) -> Shape
    bezier_surface(control_net=net, flip=False) -> Shape
"""

import torch
from torch import Tensor

from ..core import Shape
from ..utils import get_batch_size, validate_tensor
from .curves import CubicBezier2D

# Bernstein basis helpers


def _bernstein3(t: Tensor) -> Tensor:
    """
    Cubic Bernstein basis values.

    Args:
        t: [...] parameter values in [0, 1]

    Returns:
        [..., 4] basis values [B0(t), B1(t), B2(t), B3(t)]
    """
    s = 1.0 - t
    s2 = s * s
    t2 = t * t
    return torch.stack(
        [
            s2 * s,  # (1-t)^3
            3.0 * s2 * t,  # 3(1-t)^2 t
            3.0 * s * t2,  # 3(1-t) t^2
            t2 * t,  # t^3
        ],
        dim=-1,
    )


def _bernstein3_deriv(t: Tensor) -> Tensor:
    """
    Derivatives of cubic Bernstein basis.

    Args:
        t: [...] parameter values

    Returns:
        [..., 4] derivative values [B0'(t), B1'(t), B2'(t), B3'(t)]
    """
    s = 1.0 - t
    return torch.stack(
        [
            -3.0 * s * s,  # -3(1-t)^2
            3.0 * s * (1.0 - 3.0 * t),  # 3(1-t)(1-3t) = 3(1-t) - 9(1-t)t
            3.0 * t * (2.0 - 3.0 * t),  # 6t(1-t) - 3t^2 = 3t(2-3t)
            3.0 * t * t,  # 3t^2
        ],
        dim=-1,
    )


# Surface evaluation


def _eval_surface(net: Tensor, u: Tensor, v: Tensor) -> Tensor:
    """
    Evaluate bicubic Bezier surface S(u,v) via two-step bmm contraction.

    Avoids the O(B*N*4*4*3) intermediate of the 5D broadcast approach,
    using bmm to contract over u first, then a pointwise contraction over v.

    Args:
        net: [B, 4, 4, 3] control net
        u: [B, N] parameter along curve direction
        v: [B, N] parameter across curves (0=curve1, 1=curve2)

    Returns:
        [B, N, 3] surface points
    """
    Bu = _bernstein3(u)  # [B, N, 4]
    Bv = _bernstein3(v)  # [B, N, 4]

    B_sz, N_sz = u.shape
    net_flat = net.reshape(B_sz, 4, 12)  # [B, 4, 4*3]

    # Contract over u: tmp[b,n,j*3] = sum_i Bu[b,n,i] * net[b,i,j*3]
    tmp = torch.bmm(Bu, net_flat)  # [B, N, 12]
    tmp = tmp.reshape(B_sz, N_sz, 4, 3)  # [B, N, 4, 3]

    # Contract over v: S[b,n,c] = sum_j Bv[b,n,j] * tmp[b,n,j,c]
    S = (Bv.unsqueeze(-1) * tmp).sum(dim=-2)  # [B, N, 3]

    return S


def _eval_surface_derivs(net: Tensor, u: Tensor, v: Tensor):
    """
    Partial derivatives dS/du and dS/dv of bicubic Bezier surface.

    Uses the same bmm two-step contraction as _eval_surface.

    Args:
        net: [B, 4, 4, 3] control net
        u: [B, N] parameter along curve direction
        v: [B, N] parameter across curves

    Returns:
        dSdu: [B, N, 3] partial derivative w.r.t. u
        dSdv: [B, N, 3] partial derivative w.r.t. v
    """
    Bu = _bernstein3(u)  # [B, N, 4]
    Bv = _bernstein3(v)  # [B, N, 4]
    dBu = _bernstein3_deriv(u)  # [B, N, 4]
    dBv = _bernstein3_deriv(v)  # [B, N, 4]

    B_sz, N_sz = u.shape
    net_flat = net.reshape(B_sz, 4, 12)  # [B, 4, 4*3]

    # dS/du: contract dBu over u, then Bv over v
    tmp_du = torch.bmm(dBu, net_flat).reshape(B_sz, N_sz, 4, 3)
    dSdu = (Bv.unsqueeze(-1) * tmp_du).sum(dim=-2)  # [B, N, 3]

    # dS/dv: contract Bu over u, then dBv over v
    tmp_dv = torch.bmm(Bu, net_flat).reshape(B_sz, N_sz, 4, 3)
    dSdv = (dBv.unsqueeze(-1) * tmp_dv).sum(dim=-2)  # [B, N, 3]

    return dSdu, dSdv


def _eval_curve_3d(cp: Tensor, t: Tensor) -> Tensor:
    """
    Evaluate a cubic Bezier curve in 3D at per-query-point parameter values.

    Args:
        cp: [B, 4, 3] control points
        t: [B, N] or [B, N, K] parameter values

    Returns:
        [B, N, 3] or [B, N, K, 3] curve points
    """
    B = _bernstein3(t)
    cp_exp = cp
    for _ in range(t.dim() - 1):
        cp_exp = cp_exp.unsqueeze(1)
    return (B.unsqueeze(-1) * cp_exp).sum(dim=-2)


def _eval_curve_3d_deriv(cp: Tensor, t: Tensor) -> Tensor:
    """
    Evaluate the derivative of a cubic Bezier curve in 3D.

    Args:
        cp: [B, 4, 3] control points
        t: [B, N] parameter values

    Returns:
        [B, N, 3] tangent vectors
    """
    dB = _bernstein3_deriv(t)
    cp_exp = cp
    for _ in range(t.dim() - 1):
        cp_exp = cp_exp.unsqueeze(1)
    return (dB.unsqueeze(-1) * cp_exp).sum(dim=-2)


def _closest_t_curve_3d(
    query: Tensor,
    cp: Tensor,
    n_coarse: int = 16,
    n_newton: int = 4,
) -> Tensor:
    """
    Find closest t on a cubic 3D Bezier curve for each query point.

    This is used by _closest_uv() to explicitly test the four patch boundary
    curves. Boundary closest points are common on trimmed/open patches, and the
    2D surface search alone can snap to the wrong edge branch.
    """
    B, N, _ = query.shape
    device = query.device
    _T_EPS = 1e-3

    with torch.no_grad():
        t_coarse = torch.linspace(0.0, 1.0, n_coarse, device=device)
        t_expanded = t_coarse.unsqueeze(0).unsqueeze(0).expand(B, N, n_coarse)
        pts_coarse = _eval_curve_3d(cp, t_expanded)  # [B, N, K, 3]
        diff = query.unsqueeze(2) - pts_coarse
        dist_sq = (diff * diff).sum(dim=-1)
        best_idx = dist_sq.argmin(dim=-1)
        t_star = t_coarse[best_idx]

    t_star = t_star.clamp(_T_EPS, 1.0 - _T_EPS)

    for _ in range(n_newton):
        C = _eval_curve_3d(cp, t_star)  # [B, N, 3]
        r = C - query
        dCdt = _eval_curve_3d_deriv(cp, t_star)

        # Gauss-Newton in 1D: solve ||C(t) - q||^2 locally.
        denom = (dCdt * dCdt).sum(dim=-1).clamp(min=1e-12)
        dt = -((dCdt * r).sum(dim=-1)) / denom

        obj_cur = (r * r).sum(dim=-1)
        best_obj = obj_cur
        best_t = t_star

        for alpha in (1.0, 0.5, 0.25, 0.125):
            t_try = (t_star + alpha * dt).clamp(_T_EPS, 1.0 - _T_EPS)
            C_try = _eval_curve_3d(cp, t_try)
            obj_try = ((C_try - query) ** 2).sum(dim=-1)
            improved = obj_try < best_obj
            best_obj = torch.where(improved, obj_try, best_obj)
            best_t = torch.where(improved, t_try, best_t)

        t_star = best_t

    return t_star


# Control net construction


def _build_control_net(
    curve1: CubicBezier2D,
    curve2: CubicBezier2D,
    r1: Tensor,
    r2: Tensor,
) -> Tensor:
    """
    Build 4x4 bicubic Bezier control net from two curves.

    Row j=0: curve1 control points (3D)
    Row j=1: curve1 CPs + r1 * (span/3) * n1_oriented
    Row j=2: curve2 CPs - r2 * (span/3) * n2_oriented
    Row j=3: curve2 control points (3D)

    r=0 gives ruled surface, r=1 gives standard smooth interpolation.

    Args:
        curve1: CubicBezier2D with workplane (v=0 boundary)
        curve2: CubicBezier2D with workplane (v=1 boundary)
        r1: [B, 1] bulge parameter for curve1 side
        r2: [B, 1] bulge parameter for curve2 side

    Returns:
        [B, 4, 4, 3] control net (first index = u, second = v)
    """
    # Get 3D control points: [B, 4, 3]
    P1 = curve1.control_points_3d()  # [B, 4, 3]
    P2 = curve2.control_points_3d()  # [B, 4, 3]

    # Workplane normals: [B, 3]
    n1 = curve1.workplane.normal  # [B, 3]
    n2 = curve2.workplane.normal  # [B, 3]

    # Center of each set of control points: [B, 3]
    center1 = P1.mean(dim=1)  # [B, 3]
    center2 = P2.mean(dim=1)  # [B, 3]

    # Orient normals toward the other curve
    dir_12 = center2 - center1  # [B, 3]
    dir_21 = center1 - center2  # [B, 3]

    sign1 = torch.sign((dir_12 * n1).sum(dim=-1, keepdim=True))  # [B, 1]
    sign2 = torch.sign((dir_21 * n2).sum(dim=-1, keepdim=True))  # [B, 1]

    # Handle degenerate case where curves are coplanar (sign=0)
    sign1 = torch.where(sign1 == 0, torch.ones_like(sign1), sign1)
    sign2 = torch.where(sign2 == 0, torch.ones_like(sign2), sign2)

    n1_oriented = n1 * sign1  # [B, 3], points toward curve2
    n2_oriented = n2 * sign2  # [B, 3], points toward curve1

    # Per-control-point span: [B, 4]
    span = (P2 - P1).norm(dim=-1)  # [B, 4]

    # Build interior rows
    # r1: [B, 1], span: [B, 4], n1_oriented: [B, 3]
    # offset1: [B, 4, 3] = r1 * (span/3) * n1_oriented
    offset1 = (
        r1.unsqueeze(-1) * (span / 3.0).unsqueeze(-1) * n1_oriented.unsqueeze(1)
    )  # [B, 4, 3]
    offset2 = (
        r2.unsqueeze(-1) * (span / 3.0).unsqueeze(-1) * n2_oriented.unsqueeze(1)
    )  # [B, 4, 3]

    row0 = P1  # [B, 4, 3]
    row1 = P1 + offset1  # [B, 4, 3]
    row2 = P2 + offset2  # [B, 4, 3]  (n2_oriented points toward C1, so + is correct)
    row3 = P2  # [B, 4, 3]

    # Stack: net[i, j] where i=u (along curves), j=v (across curves)
    # Rows are v-direction, columns are u-direction
    # net[:, i, j, :] = row_j[:, i, :]
    net = torch.stack([row0, row1, row2, row3], dim=2)  # [B, 4(u), 4(v), 3]

    return net


# Closest point search on surface


def _closest_uv(
    query: Tensor,
    net: Tensor,
    n_coarse: int = 16,
    n_newton: int = 4,
) -> tuple:
    """
    Find closest (u, v) on bicubic Bezier surface for each query point.

    Two-pass algorithm:
    1. Coarse pass (no_grad): evaluate on n_coarse × n_coarse grid, hard argmin
    2. Gauss-Newton refinement with backtracking line search
    3. Boundary pass: explicitly search all four patch edges as cubic curves
       and keep the closest candidate

    Newton replaces the old softmin fine grid.  It is 9× faster
    (58 ms vs 525 ms at B=256, N=4096 on MPS), 20× more precise, and uses
    ~18× less memory.  See profiling/bezier_surface_optimization_results.md.

    Args:
        query: [B, N, 3] query points in 3D
        net: [B, 4, 4, 3] control net
        n_coarse: Coarse grid samples per dimension (default 8 -> 64 candidates)
        n_newton: Number of Gauss-Newton iterations (default 2)

    Returns:
        u_star: [B, N] closest u parameter
        v_star: [B, N] closest v parameter
    """
    B, N, _ = query.shape
    device = query.device

    # Small epsilon: avoids exact 0/1 where ruled surfaces (r=0)
    # have dSdv=0, making the Jacobian singular (zero Newton step).
    _UV_EPS = 1e-3

    # Pass 1: Coarse (no_grad)
    with torch.no_grad():
        uc = torch.linspace(0.0, 1.0, n_coarse, device=device)
        vc = torch.linspace(0.0, 1.0, n_coarse, device=device)
        ug, vg = torch.meshgrid(uc, vc, indexing="ij")
        ug_flat = ug.reshape(-1)  # (G,)
        vg_flat = vg.reshape(-1)  # (G,)
        G = n_coarse * n_coarse

        ug_exp = ug_flat.unsqueeze(0).expand(B, G)  # [B, G]
        vg_exp = vg_flat.unsqueeze(0).expand(B, G)  # [B, G]

        pts_coarse = _eval_surface(net, ug_exp, vg_exp)  # [B, G, 3]

        diff = query.unsqueeze(2) - pts_coarse.unsqueeze(1)  # [B, N, G, 3]
        dist_sq = (diff**2).sum(dim=-1)  # [B, N, G]

        best_idx = dist_sq.argmin(dim=-1)  # [B, N]
        u_star = ug_flat[best_idx]  # [B, N]
        v_star = vg_flat[best_idx]  # [B, N]

    # Nudge seed away from exact 0/1 to avoid singular Jacobian
    u_star = u_star.clamp(_UV_EPS, 1.0 - _UV_EPS)
    v_star = v_star.clamp(_UV_EPS, 1.0 - _UV_EPS)

    # Pass 2: Gauss-Newton with backtracking line search
    for _ in range(n_newton):
        S = _eval_surface(net, u_star, v_star)  # [B, N, 3]
        r = S - query  # [B, N, 3]
        dSdu, dSdv = _eval_surface_derivs(net, u_star, v_star)

        # Normal equations: (J^T J) @ [du, dv] = -J^T r
        Jtr_u = (dSdu * r).sum(dim=-1)  # [B, N]
        Jtr_v = (dSdv * r).sum(dim=-1)  # [B, N]
        a = (dSdu * dSdu).sum(dim=-1)  # [B, N]
        b = (dSdu * dSdv).sum(dim=-1)  # [B, N]
        d = (dSdv * dSdv).sum(dim=-1)  # [B, N]
        det = (a * d - b * b).clamp(min=1e-12)

        du = -(d * Jtr_u - b * Jtr_v) / det
        dv = -(-b * Jtr_u + a * Jtr_v) / det

        # Backtracking line search: pick best step size per point
        obj_cur = (r * r).sum(dim=-1)  # [B, N]
        best_obj = obj_cur
        best_u = u_star
        best_v = v_star

        for alpha in (1.0, 0.5, 0.25, 0.125):
            u_try = (u_star + alpha * du).clamp(_UV_EPS, 1.0 - _UV_EPS)
            v_try = (v_star + alpha * dv).clamp(_UV_EPS, 1.0 - _UV_EPS)
            S_try = _eval_surface(net, u_try, v_try)
            obj_try = ((S_try - query) ** 2).sum(dim=-1)
            improved = obj_try < best_obj
            best_obj = torch.where(improved, obj_try, best_obj)
            best_u = torch.where(improved, u_try, best_u)
            best_v = torch.where(improved, v_try, best_v)

        u_star = best_u
        v_star = best_v

    # Surface Gauss-Newton handles interior closest points well, but when the
    # true closest point lives on a patch boundary it can lock onto the wrong
    # edge/corner basin. Search all four Bezier boundary curves explicitly and
    # keep the closest candidate by Euclidean distance.
    S_best = _eval_surface(net, u_star, v_star)
    best_dist_sq = ((S_best - query) ** 2).sum(dim=-1)
    best_u = u_star
    best_v = v_star

    edge_specs = (
        ("u0", net[:, 0, :, :]),
        ("u1", net[:, 3, :, :]),
        ("v0", net[:, :, 0, :]),
        ("v1", net[:, :, 3, :]),
    )

    for edge_name, edge_cp in edge_specs:
        t_edge = _closest_t_curve_3d(query, edge_cp)
        if edge_name == "u0":
            u_edge = torch.full_like(t_edge, _UV_EPS)
            v_edge = t_edge
        elif edge_name == "u1":
            u_edge = torch.full_like(t_edge, 1.0 - _UV_EPS)
            v_edge = t_edge
        elif edge_name == "v0":
            u_edge = t_edge
            v_edge = torch.full_like(t_edge, _UV_EPS)
        else:
            u_edge = t_edge
            v_edge = torch.full_like(t_edge, 1.0 - _UV_EPS)

        S_edge = _eval_surface(net, u_edge, v_edge)
        dist_sq = ((S_edge - query) ** 2).sum(dim=-1)
        better = dist_sq < best_dist_sq

        best_dist_sq = torch.where(better, dist_sq, best_dist_sq)
        best_u = torch.where(better, u_edge, best_u)
        best_v = torch.where(better, v_edge, best_v)

    return best_u, best_v


# Public API


def bezier_surface(
    curve1: CubicBezier2D | None = None,
    curve2: CubicBezier2D | None = None,
    *,
    r1: Tensor | None = None,
    r2: Tensor | None = None,
    control_net: Tensor | None = None,
    flip: bool = False,
) -> Shape:
    """
    Signed distance to a bicubic Bezier surface patch.

    Two input modes:

    **Curve-based** (existing): builds control net from two curves.
        bezier_surface(curve1, curve2, r1=..., r2=...)

    **Control-net** (new): explicit 4x4 control net for free-form patches.
        bezier_surface(control_net=net)

    Args:
        curve1: CubicBezier2D with workplane (v=0 boundary). Curve mode only.
        curve2: CubicBezier2D with workplane (v=1 boundary). Curve mode only.
        r1: [B, 1] bulge on curve1 side (0=flat/ruled, 1=standard smooth). Curve mode only.
        r2: [B, 1] bulge on curve2 side. Curve mode only.
        control_net: [B, 4, 4, 3] explicit control net. Control-net mode only.
        flip: If True, invert sign convention (default: normal side = outside)

    Returns:
        Shape with sdf_fn: [N, 3] -> [B, N]

    Raises:
        ValueError: If inputs are invalid or both modes specified
    """
    has_curves = curve1 is not None or curve2 is not None
    has_net = control_net is not None

    if has_curves and has_net:
        raise ValueError(
            "bezier_surface: provide either (curve1, curve2, r1, r2) "
            "or control_net, not both"
        )
    if not has_curves and not has_net:
        raise ValueError(
            "bezier_surface: must provide either (curve1, curve2, r1, r2) "
            "or control_net"
        )

    if has_net:
        # Control-net mode
        if not isinstance(control_net, Tensor):
            raise TypeError(
                f"bezier_surface: control_net must be a Tensor, "
                f"got {type(control_net).__name__}"
            )
        if control_net.dim() != 4 or control_net.shape[1:] != (4, 4, 3):
            raise ValueError(
                f"bezier_surface: control_net must be [B, 4, 4, 3], "
                f"got {list(control_net.shape)}"
            )
        net = control_net
        batch_size = net.shape[0]
    else:
        # Curve-based mode (existing)
        if curve1 is None or curve2 is None:
            raise ValueError(
                "bezier_surface: curve mode requires both curve1 and curve2"
            )
        if r1 is None or r2 is None:
            raise ValueError("bezier_surface: curve mode requires both r1 and r2")

        wp1 = curve1.workplane
        wp2 = curve2.workplane
        if wp1 is None:
            raise ValueError(
                "bezier_surface: curve1 has no workplane. "
                "Create curve with: CubicBezier2D(cp, workplane='xz')"
            )
        if wp2 is None:
            raise ValueError(
                "bezier_surface: curve2 has no workplane. "
                "Create curve with: CubicBezier2D(cp, workplane='xz')"
            )

        r1_t = validate_tensor(r1, vec_size=1, name="r1")
        r2_t = validate_tensor(r2, vec_size=1, name="r2")

        # r < 0.05 produces a near-degenerate control net (adjacent rows
        # nearly identical -> dSdv ≈ 0 at boundaries -> ill-conditioned
        # Jacobian -> SDF artifacts).  Reject early with a clear message.
        _R_MIN = 0.05
        if (r1_t < _R_MIN).any():
            raise ValueError(
                f"bezier_surface: r1 must be >= {_R_MIN}, got "
                f"{r1_t.min().item():.4f}. Values near 0 produce a "
                f"degenerate surface with SDF artifacts at patch boundaries."
            )
        if (r2_t < _R_MIN).any():
            raise ValueError(
                f"bezier_surface: r2 must be >= {_R_MIN}, got "
                f"{r2_t.min().item():.4f}. Values near 0 produce a "
                f"degenerate surface with SDF artifacts at patch boundaries."
            )

        batch_size = get_batch_size(r1_t, r2_t, names=["r1", "r2"])

        if curve1.batch_size != batch_size:
            raise ValueError(
                f"bezier_surface: curve1 batch_size={curve1.batch_size} "
                f"doesn't match parameter batch_size={batch_size}"
            )
        if curve2.batch_size != batch_size:
            raise ValueError(
                f"bezier_surface: curve2 batch_size={curve2.batch_size} "
                f"doesn't match parameter batch_size={batch_size}"
            )

        net = _build_control_net(curve1, curve2, r1_t, r2_t)  # [B, 4, 4, 3]

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate bicubic Bezier surface SDF: [B, N, 3] -> [B, N]"""
        # p is already [B, N, 3] from the convenience wrapper
        query = p.expand(batch_size, -1, -1)

        # Find closest (u*, v*) on surface
        u_star, v_star = _closest_uv(query, net)  # [B, N], [B, N]

        # Evaluate surface point and derivatives at (u*, v*)
        S = _eval_surface(net, u_star, v_star)  # [B, N, 3]
        dSdu, dSdv = _eval_surface_derivs(net, u_star, v_star)  # [B, N, 3] each

        # Displacement from surface to query
        diff = query - S  # [B, N, 3]

        # Raw normal via cross product
        raw_normal = torch.linalg.cross(dSdu, dSdv)  # [B, N, 3]
        raw_len_sq = (raw_normal * raw_normal).sum(dim=-1, keepdim=True)  # [B, N, 1]

        # Detect degeneracy: |cross|^2 < threshold
        _DEGEN_SQ = 1e-8  # |cross| < 1e-4
        is_degen = raw_len_sq < _DEGEN_SQ  # [B, N, 1]

        # Branchless: always compute retraction (~10-20µs extra when no degen)
        # Avoids .any() graph break for torch.compile
        _ALPHA = 0.05
        u_r = u_star + _ALPHA * (0.5 - u_star)
        v_r = v_star + _ALPHA * (0.5 - v_star)
        dSdu_r, dSdv_r = _eval_surface_derivs(net, u_r, v_r)
        ret_normal = torch.linalg.cross(dSdu_r, dSdv_r)
        ret_len_sq = (ret_normal * ret_normal).sum(dim=-1, keepdim=True)

        _DEGEN_INNER_SQ = 1e-10
        double_degen = is_degen & (ret_len_sq < _DEGEN_INNER_SQ)

        chosen = torch.where(is_degen, ret_normal, raw_normal)
        chosen = torch.where(double_degen, diff, chosen)

        normal_len = chosen.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        normal = chosen / normal_len

        # Signed distance to tangent plane at closest point.
        # Same idea as bezier_halfplane's cross-product trick:
        # the tangent plane varies smoothly with (u,v), so small
        # search errors don't cause discontinuities.
        result = (diff * normal).sum(dim=-1)  # [B, N]

        if flip:
            result = -result

        return result

    return Shape(sdf_fn, batch_size=batch_size, plane=None, device=net.device)
