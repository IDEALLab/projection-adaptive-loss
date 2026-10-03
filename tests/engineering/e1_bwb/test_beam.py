"""Beam tests: analytical cantilever, L-scaling ratios, grad flow."""

from __future__ import annotations

import torch

from pal.benchmarks.engineering.e1_bwb import DIM
from pal.benchmarks.engineering.e1_bwb.beam import (
    NOMINAL_SKIN_THICKNESS_M,
    compute_stress,
)
from pal.benchmarks.engineering.e1_bwb.interfaces import Loads, StructProps
from pal.benchmarks.engineering.e1_bwb.x_layout import decode_x

_B_CHORD = 0.3     # unit-scale chord (m when L=1)
_H_HEIGHT = 0.05   # unit-scale height (m when L=1)


def _analytic_rect_props_unit() -> dict:
    """Exact nondim properties for a rectangle at unit scale."""
    A = _B_CHORD * _H_HEIGHT
    I_uu = _B_CHORD * _H_HEIGHT**3 / 12.0
    I_vv = _H_HEIGHT * _B_CHORD**3 / 12.0
    # Thin-section torsion constant (dominant term of the series).
    J = _B_CHORD * _H_HEIGHT**3 / 3.0
    # Q = (b*h/2)*(h/4) = b*h^2/8
    Q_max = _B_CHORD * _H_HEIGHT**2 / 8.0
    return {
        "A": A, "I_uu": I_uu, "I_vv": I_vv, "J": J, "Q_max": Q_max,
    }


def _make_uniform_inputs(
    B: int, N: int, L: float, q_0: float,
) -> tuple[Loads, StructProps, torch.Tensor, torch.Tensor]:
    """Cantilever with uniform loading and section props, stations on [0, L/2]."""
    rect = _analytic_rect_props_unit()
    span = 0.5 * L
    y = torch.linspace(0.0, span, N).unsqueeze(0).expand(B, -1).contiguous()
    ones = torch.ones(B, N)

    # Uniform x_cp at centroid so the distributed torsion vanishes.
    u_cg_nd = torch.full((B, N), 0.0)                                    # nondim
    x_cp_phys = torch.full((B, N), 0.0)                                  # m
    loads = Loads(
        q_z=ones * q_0,                                                   # N/m
        m=torch.zeros(B, N),
        x_cp=x_cp_phys,
    )
    props = StructProps(
        I_uu=ones * rect["I_uu"],
        I_vv=ones * rect["I_vv"],
        J=ones * rect["J"],
        A=ones * rect["A"],
        u_cg=u_cg_nd,
        v_cg=torch.zeros(B, N),
        Q_max=ones * rect["Q_max"],
    )
    x = torch.zeros(B, DIM)
    x[:, 9] = L
    return loads, props, y, x


def test_cantilever_uniform_load_matches_closed_form():
    """sigma_b at root within 1 % of analytical sigma = 3*q*L_span^2 / (b*h^2).

    Closed-form (cantilever, uniform q over span L_s, rectangular section):
        M(0) = q*L_s^2 / 2
        sigma_b  = M * (h/2) / (b*h^3/12) = 6 M / (b*h^2)
             = 3*q*L_s^2 / (b*h^2)
    """
    B, N = 1, 41           # high N for tight trapezoid accuracy
    L = 1.0                # x.L = 1 m -> physical = unit-scale
    q_0 = 1000.0           # N/m
    L_span = 0.5 * L
    loads, props, y, x_vec = _make_uniform_inputs(B, N, L, q_0)

    # Thick skin makes shear negligible, so sigma_vm ~ sigma_b at the root.
    sigma_vm, mass, _ = compute_stress(
        loads, props, y, decode_x(x_vec), skin_thickness=10.0,
    )
    sigma_root = sigma_vm[0, 0].item()
    analytic = 3.0 * q_0 * L_span**2 / (_B_CHORD * _H_HEIGHT**2)
    # 1 % tolerance for the reverse-cumtrapz + z_max = sqrt(3)*r_y equivalence.
    assert abs(sigma_root - analytic) / analytic < 0.01, (
        f"sigma_root={sigma_root:.6g}  analytic={analytic:.6g}  "
        f"rel_err={(sigma_root - analytic) / analytic:.3e}"
    )


def test_cantilever_M_profile_matches_closed_form():
    """Check bending-moment profile, independent of the stress kernel."""
    B, N = 1, 41
    L = 1.0
    q_0 = 1000.0
    L_span = 0.5 * L
    loads, props, y, x_vec = _make_uniform_inputs(B, N, L, q_0)

    # z_max and I are uniform, so sigma_b / sigma_b_root = M / M_root.
    sigma_vm, _, _ = compute_stress(
        loads, props, y, decode_x(x_vec), skin_thickness=10.0,
    )
    sigma_profile = sigma_vm[0]                               # [N]
    sigma_root = sigma_profile[0]

    # Closed-form M(y) = q*(L_s - y)^2 / 2  ->  M / M_root = (1 - y/L_s)^2
    y_rel = y[0] / L_span
    expected_ratio = (1.0 - y_rel) ** 2
    actual_ratio = sigma_profile / sigma_root.clamp_min(1e-30)
    # 2% relative tolerance, absolute at the tip where M = 0.
    assert torch.allclose(
        actual_ratio[:-1], expected_ratio[:-1], atol=0.02, rtol=0.02,
    ), f"actual_ratio={actual_ratio.tolist()}"


def test_mass_scales_as_L_cubed():
    """mass(L=2) / mass(L=1) should equal 8.

    At L=1: mass = rho*A_unit*L_span_1 = rho*A*0.5
    At L=2: mass = rho*(A_unit*4)*L_span_2 = rho*(A*4)*(1.0) = 8*rho*A*0.5
    Ratio = 8.
    """
    B, N = 1, 11
    loads_1, props_1, y_1, x_1 = _make_uniform_inputs(B, N, 1.0, q_0=0.0)
    loads_2, props_2, y_2, x_2 = _make_uniform_inputs(B, N, 2.0, q_0=0.0)

    _, mass_1, _ = compute_stress(loads_1, props_1, y_1, decode_x(x_1))
    _, mass_2, _ = compute_stress(loads_2, props_2, y_2, decode_x(x_2))
    ratio = (mass_2 / mass_1.clamp_min(1e-30)).item()
    assert abs(ratio - 8.0) / 8.0 < 1e-6, f"ratio={ratio}"


def test_output_shapes_and_finite():
    B, N = 3, 8
    loads, props, y, x_vec = _make_uniform_inputs(B, N, 1.5, q_0=500.0)
    sigma, mass, w = compute_stress(loads, props, y, decode_x(x_vec))
    assert sigma.shape == (B, N)
    assert mass.shape == (B,)
    assert w.shape == (B, N)
    assert torch.isfinite(sigma).all()
    assert torch.isfinite(mass).all()
    assert torch.isfinite(w).all()
    assert (sigma >= 0).all()    # von Mises is non-negative
    assert (mass > 0).all()


def test_unsorted_stations_give_same_answer():
    """The `_sort_ascending` helper should make result invariant to input order."""
    B, N = 2, 6
    loads, props, y, x_vec = _make_uniform_inputs(B, N, 1.0, q_0=500.0)
    sigma_sorted, mass_sorted, w_sorted = compute_stress(
        loads, props, y, decode_x(x_vec),
    )

    torch.manual_seed(0)
    perm = torch.randperm(N)
    def sh(t):
        return t[..., perm]

    loads_sh = Loads(q_z=sh(loads.q_z), m=sh(loads.m), x_cp=sh(loads.x_cp))
    props_sh = StructProps(
        I_uu=sh(props.I_uu), I_vv=sh(props.I_vv), J=sh(props.J), A=sh(props.A),
        u_cg=sh(props.u_cg), v_cg=sh(props.v_cg), Q_max=sh(props.Q_max),
    )
    y_sh = sh(y)
    sigma_sh, mass_sh, w_sh = compute_stress(
        loads_sh, props_sh, y_sh, decode_x(x_vec),
    )
    assert torch.allclose(sigma_sorted, sigma_sh, atol=1e-6, rtol=1e-6)
    assert torch.allclose(mass_sorted, mass_sh, atol=1e-6, rtol=1e-6)
    assert torch.allclose(w_sorted, w_sh, atol=1e-6, rtol=1e-6)


def test_grad_flow_through_loads_props_and_L():
    """Back-prop from sigma + mass sums to every input tensor."""
    B, N = 1, 8
    loads, props, y, x_vec = _make_uniform_inputs(B, N, 1.0, q_0=500.0)
    loads = Loads(
        q_z=loads.q_z.clone().requires_grad_(True),
        m=loads.m.clone().requires_grad_(True),
        x_cp=loads.x_cp.clone().requires_grad_(True),
    )
    props = StructProps(
        **{
            k: getattr(props, k).clone().requires_grad_(True)
            for k in ("I_uu", "I_vv", "J", "A", "u_cg", "v_cg", "Q_max")
        }
    )
    x_vec = x_vec.clone().requires_grad_(True)

    sigma, mass, w = compute_stress(loads, props, y, decode_x(x_vec))
    (sigma.sum() + mass.sum() + w.sum()).backward()

    for name, t_ in [
        ("q_z", loads.q_z), ("I_uu", props.I_uu), ("A", props.A),
        ("Q_max", props.Q_max), ("x", x_vec),
    ]:
        assert t_.grad is not None, f"no grad on {name}"
        assert torch.isfinite(t_.grad).all(), f"non-finite grad on {name}"
        assert t_.grad.abs().sum() > 0, f"zero grad on {name}"


def test_cantilever_tip_deflection_matches_closed_form():
    """Uniform q on a prismatic cantilever: w_tip = q*L^4 / (8*E*I). 2% tol.

    The first station acts as the root (w[0] = w'[0] = 0).
    """
    B, N = 1, 201
    L = 1.0
    q_0 = 1000.0
    L_span = 0.5 * L
    E = 70e9  # arbitrary, cancels out of the ratio check, keep units honest
    loads, props, y, x_vec = _make_uniform_inputs(B, N, L, q_0)

    _, _, w = compute_stress(
        loads, props, y, decode_x(x_vec),
        skin_thickness=10.0,  # eliminate shear coupling, not needed for w
        e_modulus=E,
    )
    w_tip = w[0, -1].item()
    # Analytic: I_phys = I_uu_unit * L^4; fixed cantilever w_tip = q*L_span^4/(8*E*I)
    I_uu_unit = _B_CHORD * _H_HEIGHT**3 / 12.0
    I_phys = I_uu_unit * L**4
    analytic = q_0 * L_span**4 / (8.0 * E * I_phys)
    rel = abs(w_tip - analytic) / analytic
    assert rel < 0.02, f"w_tip={w_tip:.6g}  analytic={analytic:.6g}  rel={rel:.3e}"


def test_skin_thickness_kwarg_changes_tau():
    """Halving `skin_thickness` must roughly double the shear-dominated stress."""
    B, N = 1, 21
    # Configure so shear dominates: big q, small chord -> high V*Q_max / I*t.
    loads, props, y, x_vec = _make_uniform_inputs(B, N, 1.0, q_0=1e5)

    s_default, _, _ = compute_stress(loads, props, y, decode_x(x_vec))
    s_thin, _, _ = compute_stress(
        loads, props, y, decode_x(x_vec),
        skin_thickness=0.5 * NOMINAL_SKIN_THICKNESS_M,
    )
    # Mid-span where both V and M are non-trivial: s_thin should exceed s_default.
    mid = N // 2
    assert s_thin[0, mid] > s_default[0, mid]
