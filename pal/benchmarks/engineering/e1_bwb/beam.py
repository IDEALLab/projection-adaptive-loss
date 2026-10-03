"""Beam math and stress aggregation (`ComputeStress` protocol), pure torch.

L-scales the nondimensional `StructProps`, integrates cantilever resultants
from the tip and Euler-Bernoulli deflection from the innermost station.
"""

from __future__ import annotations

import torch
from torch import Tensor

from .interfaces import Loads, StructProps
from .x_layout import DecodedX

NOMINAL_SKIN_THICKNESS_M: float = 3e-3      # 3 mm, typical composite skin
# CFRP quasi-iso density and modulus (callers pass the canonical values).
RHO_MATERIAL_KG_PER_M3: float = 1600.0
E_MODULUS_PA: float = 55e9
# Enclosed (Bredt) area over wall area for a typical thin-walled wingbox.
A_ENC_OVER_A_WALL: float = 20.0


def _sort_ascending(
    y_stations: Tensor,
    loads: Loads,
    props: StructProps,
) -> tuple[Tensor, Loads, StructProps]:
    """Sort stations ascending in y, re-ordering every per-station field."""
    y_sorted, idx = torch.sort(y_stations, dim=-1)

    def g(t: Tensor) -> Tensor:
        return torch.gather(t, -1, idx)

    loads_s = Loads(q_z=g(loads.q_z), m=g(loads.m), x_cp=g(loads.x_cp))
    props_s = StructProps(
        I_uu=g(props.I_uu),
        I_vv=g(props.I_vv),
        J=g(props.J),
        A=g(props.A),
        u_cg=g(props.u_cg),
        v_cg=g(props.v_cg),
        Q_max=g(props.Q_max),
    )
    return y_sorted, loads_s, props_s


def _reverse_cumtrapz(f: Tensor, y: Tensor) -> Tensor:
    """Compute F(y_i) = int_{y_i}^{y_tip} f(y') dy' via reverse cumulative trapezoid.

    Args:
        f: [B, N] integrand values at station y.
        y: [B, N] sorted ascending.

    Returns:
        [B, N] with `out[..., -1] = 0` (tip boundary) and
        `out[..., 0]` = integral over the whole span.
    """
    dy = y[..., 1:] - y[..., :-1]            # [B, N-1]
    f_avg = 0.5 * (f[..., 1:] + f[..., :-1])  # [B, N-1]
    seg = f_avg * dy                          # [B, N-1]
    # Reverse + cumsum gives the running sum from the tip side.
    seg_rev = torch.flip(seg, dims=[-1])
    cum_rev = torch.cumsum(seg_rev, dim=-1)
    cum = torch.flip(cum_rev, dims=[-1])
    tip_zero = torch.zeros_like(cum[..., :1])
    return torch.cat([cum, tip_zero], dim=-1)


def _forward_cumtrapz(f: Tensor, y: Tensor) -> Tensor:
    """Compute F(y_i) = int_{y_0}^{y_i} f(y') dy' via forward cumulative trapezoid."""
    dy = y[..., 1:] - y[..., :-1]
    f_avg = 0.5 * (f[..., 1:] + f[..., :-1])
    seg = f_avg * dy
    cum = torch.cumsum(seg, dim=-1)
    root_zero = torch.zeros_like(cum[..., :1])
    return torch.cat([root_zero, cum], dim=-1)


def _l_scale(
    props: StructProps, L: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Apply L-scaling to a `StructProps` in one place.

    Returns: (A_phys, I_uu_phys, I_vv_phys, J_phys, u_cg_phys, v_cg_phys, Q_phys).
    All shapes `[B, N]`.
    """
    L1 = L.unsqueeze(-1)
    L2 = L1 * L1
    L3 = L2 * L1
    L4 = L2 * L2
    return (
        props.A * L2,
        props.I_uu * L4,
        props.I_vv * L4,
        props.J * L4,
        props.u_cg * L1,
        props.v_cg * L1,
        props.Q_max * L3,
    )


def compute_stress(
    loads: Loads,
    props: StructProps,
    y_stations: Tensor,
    x: DecodedX,
    *,
    skin_thickness: Tensor | float = NOMINAL_SKIN_THICKNESS_M,
    rho_material: Tensor | float = RHO_MATERIAL_KG_PER_M3,
    a_enc_over_a: Tensor | float = A_ENC_OVER_A_WALL,
    e_modulus: Tensor | float = E_MODULUS_PA,
) -> tuple[Tensor, Tensor, Tensor]:
    """Satisfies the `ComputeStress` protocol.

    Returns:
        sigma_max: [B, N]  von Mises composite stress, Pa.
        mass_struct: [B]   structural mass, kg.
    """
    y_sorted, loads_s, props_s = _sort_ascending(y_stations, loads, props)

    L = x.L.squeeze(-1)  # [B]
    A_phys, I_uu_phys, I_vv_phys, J_phys, u_cg_phys, v_cg_phys, Q_phys = _l_scale(
        props_s, L,
    )

    # `skin_thickness` can be scalar, `[B]`, or `[B, 1]`.
    if isinstance(skin_thickness, Tensor) and skin_thickness.dim() == 1:
        skin_thickness = skin_thickness.unsqueeze(-1)

    V = _reverse_cumtrapz(loads_s.q_z, y_sorted)              # [B, N] N
    M = _reverse_cumtrapz(V, y_sorted)                        # [B, N] N*m
    # Distributed torsion t(y) = q_z*(x_cp - x_cg) + m, with x_sc := x_cg.
    t_dist = loads_s.q_z * (loads_s.x_cp - u_cg_phys) + loads_s.m
    T = _reverse_cumtrapz(t_dist, y_sorted)                   # [B, N] N*m

    safe_I = I_uu_phys.clamp_min(1e-18)
    safe_A = A_phys.clamp_min(1e-12)
    # Outer-fibre distance. Exact for rectangles, reasonable proxy elsewhere.
    z_max = torch.sqrt(torch.tensor(3.0, device=safe_I.device, dtype=safe_I.dtype)) * torch.sqrt(
        safe_I / safe_A,
    )
    sigma_b = M.abs() * z_max / safe_I
    tau_s = V.abs() * Q_phys / (safe_I * skin_thickness)
    A_enc = a_enc_over_a * A_phys
    tau_t = T.abs() / (2.0 * A_enc.clamp_min(1e-12) * skin_thickness)
    tau_total = tau_s + tau_t
    # eps guards dsqrt(x)/dx = 1/(2sqrt(x)) at the tip station where M=V=T=0 exactly.
    sigma_vm = torch.sqrt(sigma_b * sigma_b + 3.0 * tau_total * tau_total + 1e-30)

    dy = y_sorted[..., 1:] - y_sorted[..., :-1]
    A_avg = 0.5 * (A_phys[..., 1:] + A_phys[..., :-1])
    mass_struct = rho_material * (A_avg * dy).sum(dim=-1)      # [B]

    # Deflection w(y): double forward cumulative trapezoid of M/(E*I).
    EI = e_modulus * safe_I                                     # [B, N] Pa*m^4
    curvature = M / EI.clamp_min(1e-18)                         # [B, N] 1/m
    slope = _forward_cumtrapz(curvature, y_sorted)              # [B, N] rad
    w = _forward_cumtrapz(slope, y_sorted)                      # [B, N] m

    return sigma_vm, mass_struct, w
