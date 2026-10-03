"""Internal Protocol contracts (tensor shapes) for the E1 BWB components.

B = batch size, N_stations = spanwise station count. Tensors are float32 by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torch import Tensor

from .x_layout import DecodedX


@dataclass(frozen=True)
class StructProps:
    """Nondimensional section properties `[B, N_stations]`, L-scaled in `beam.py`."""

    I_uu: Tensor    # [B, N] second moment about chord-axis
    I_vv: Tensor    # [B, N] second moment about vertical axis
    J: Tensor       # [B, N] torsion constant
    A: Tensor       # [B, N] enclosed area
    u_cg: Tensor    # [B, N] centroid chordwise offset
    v_cg: Tensor    # [B, N] centroid vertical offset
    Q_max: Tensor   # [B, N] first-moment for shear stress


@dataclass(frozen=True)
class Loads:
    """Spanwise aerodynamic load distribution in physical units (per unit span).

    Fields are `[B, N_stations]`.
    """

    q_z: Tensor     # [B, N] vertical force per unit span (N/m)
    m: Tensor       # [B, N] pitching moment per unit span (N*m/m)
    x_cp: Tensor    # [B, N] chordwise centre-of-pressure (m)


@runtime_checkable
class ComputeAero(Protocol):
    """Global coefficients from A_aero surrogate.

    Args:
        x: Decoded design vector.
        conditions: `[B, 2]` = (alt, V_cruise).

    Returns:
        (CL, CD, CM) each `[B]`, dimensionless.
    """

    def __call__(self, x: DecodedX, conditions: Tensor) -> tuple[Tensor, Tensor, Tensor]: ...


@runtime_checkable
class ComputeLoads(Protocol):
    """Integrated sectional loads from FiLM surface-pressure surrogate.

    Args:
        program: geometry `CADProgram` for the current x (holds SDF + params).
        x: Decoded design vector.
        conditions: `[B, 2]`.
        y_stations: `[B, N_stations]`, sorted ascending on `[0, semi_span]`.

    Returns:
        `Loads` container (all fields `[B, N_stations]`).
    """

    def __call__(
        self,
        program: object,
        x: DecodedX,
        conditions: Tensor,
        y_stations: Tensor,
    ) -> Loads: ...


@runtime_checkable
class ComputeStructural(Protocol):
    """Structural-surrogate section-property evaluation.

    Args:
        x: Decoded design vector.
        y_stations: `[B, N_stations]`.

    Returns:
        `StructProps` container, nondimensional.
    """

    def __call__(self, x: DecodedX, y_stations: Tensor) -> StructProps: ...


@runtime_checkable
class ComputeStress(Protocol):
    """Beam-theory stress aggregation.

    Args:
        loads: `Loads` container (physical units).
        props: `StructProps` container (nondimensional, L-scaled here).
        y_stations: `[B, N_stations]`.
        x: Decoded design vector (for `L` and `alpha_cr`).

    Returns:
        (sigma_max [B, N_stations], mass_struct [B], w [B, N_stations]).
        `w` is Euler-Bernoulli deflection (m), `w[..., -1]` is the tip.
    """

    def __call__(
        self,
        loads: Loads,
        props: StructProps,
        y_stations: Tensor,
        x: DecodedX,
    ) -> tuple[Tensor, Tensor, Tensor]: ...
