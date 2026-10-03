"""Stub implementations of the component protocols.

Each stub returns correctly-shaped placeholder tensors that still carry gradients from the inputs.
"""

from __future__ import annotations

from torch import Tensor

from .interfaces import Loads, StructProps
from .x_layout import DecodedX


def _batch_scalar(x: Tensor, value: float = 0.0) -> Tensor:
    """[B] scalar that carries grad from `x` (via sum-reduction then rescale)."""
    # x.sum()*0 + value keeps device, dtype and graph without numerical dependence.
    return x.sum(dim=-1) * 0.0 + value


def _batched_stations(x: Tensor, y_stations: Tensor, value: float) -> Tensor:
    """[B, N] tensor carrying grad from `x` and `y_stations`."""
    base = y_stations * 0.0 + value
    grad_carrier = x.sum(dim=-1, keepdim=True) * 0.0
    return base + grad_carrier


def compute_aero_stub(x: DecodedX, conditions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Constant CL/CD/CM with grad through x and conditions."""
    carrier = x.shape.sum(dim=-1) + conditions.sum(dim=-1)
    CL = carrier * 0.0 + 0.4
    CD = carrier * 0.0 + 0.03
    CM = carrier * 0.0 + 0.0
    return CL, CD, CM


def compute_loads_stub(
    program: object,
    x: DecodedX,
    conditions: Tensor,
    y_stations: Tensor,
) -> Loads:
    """Flat load distribution (1000 N/m, 0 moment, x_cp at quarter-chord)."""
    q_z = _batched_stations(x.shape, y_stations, 1000.0)
    m = _batched_stations(x.shape, y_stations, 0.0)
    x_cp = _batched_stations(x.shape, y_stations, 0.25)
    cond_carrier = conditions.sum(dim=-1, keepdim=True) * 0.0
    return Loads(q_z=q_z + cond_carrier, m=m + cond_carrier, x_cp=x_cp + cond_carrier)


def compute_structural_stub(x: DecodedX, y_stations: Tensor) -> StructProps:
    """Uniform nondim section props (reasonable O(1) values)."""
    def mk(v):
        return _batched_stations(x.struct, y_stations, v)
    return StructProps(
        I_uu=mk(1e-4),
        I_vv=mk(1e-3),
        J=mk(5e-5),
        A=mk(1e-2),
        u_cg=mk(0.25),
        v_cg=mk(0.0),
        Q_max=mk(5e-4),
    )


def compute_stress_stub(
    loads: Loads,
    props: StructProps,
    y_stations: Tensor,
    x: DecodedX,
    **kwargs: object,
) -> tuple[Tensor, Tensor, Tensor]:
    """Constant sigma well below yield, constant mass, zero deflection.

    `**kwargs` (skin_thickness, rho_material, e_modulus) are ignored.
    """
    carrier = (
        loads.q_z.sum(dim=-1)
        + props.A.sum(dim=-1)
        + y_stations.sum(dim=-1)
        + x.L.squeeze(-1)
    )
    sigma_max = _batched_stations(x.shape, y_stations, 1e7)  # 10 MPa
    sigma_max = sigma_max + (carrier.unsqueeze(-1) * 0.0)
    mass_struct = carrier * 0.0 + 50.0
    # Deflection ramps linearly with y so the gradient check is non-trivial.
    w = y_stations * 1e-3 + (carrier.unsqueeze(-1) * 0.0)
    return sigma_max, mass_struct, w
