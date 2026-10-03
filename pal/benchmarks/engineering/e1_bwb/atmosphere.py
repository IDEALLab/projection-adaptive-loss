"""ISA troposphere (below 11 km) air properties: Mach, density, dynamic pressure, Re.

All functions are torch-native, batch-broadcastable and autograd-safe.
"""

from __future__ import annotations

import torch
from torch import Tensor

G: float = 9.80665                # m/s^2
GAMMA_AIR: float = 1.4            # ratio of specific heats
R_AIR: float = 287.05287          # J/(kg*K) specific gas constant
MU_AIR_REF: float = 1.7894e-5     # Pa*s at 288.15 K (Sutherland reference)
T_REF: float = 288.15             # K
SUTHERLAND_S: float = 110.4       # K

_T0: float = 288.15                # K (sea-level)
_P0: float = 101325.0              # Pa
_L_LAPSE: float = 0.0065           # K/m
_RHO0: float = _P0 / (R_AIR * _T0)

gravity: float = G


def _as_tensor(x: Tensor | float, like: Tensor | None = None) -> Tensor:
    """Coerce scalars to tensors matching an anchor tensor's dtype/device."""
    if isinstance(x, Tensor):
        return x
    if like is None:
        return torch.as_tensor(x, dtype=torch.float32)
    return torch.as_tensor(x, dtype=like.dtype, device=like.device)


def temperature(alt: Tensor | float) -> Tensor:
    """ISA static temperature [K] at altitude [m]."""
    a = _as_tensor(alt)
    return _T0 - _L_LAPSE * a


def pressure(alt: Tensor | float) -> Tensor:
    """ISA static pressure [Pa] at altitude [m]."""
    a = _as_tensor(alt)
    T = temperature(a)
    exponent = G / (R_AIR * _L_LAPSE)
    return _P0 * (T / _T0) ** exponent


def rho_air(alt: Tensor | float) -> Tensor:
    """ISA density [kg/m^3] at altitude [m]."""
    a = _as_tensor(alt)
    return pressure(a) / (R_AIR * temperature(a))


def speed_of_sound(alt: Tensor | float) -> Tensor:
    """Speed of sound [m/s] at altitude [m]."""
    return torch.sqrt(GAMMA_AIR * R_AIR * temperature(alt))


def dynamic_viscosity(alt: Tensor | float) -> Tensor:
    """Sutherland dynamic viscosity [Pa*s] at altitude [m]."""
    T = temperature(alt)
    return MU_AIR_REF * (T / T_REF) ** 1.5 * (T_REF + SUTHERLAND_S) / (T + SUTHERLAND_S)


def mach(V: Tensor | float, alt: Tensor | float) -> Tensor:
    """Mach number from true airspeed [m/s] and altitude [m]."""
    v = _as_tensor(V)
    return v / speed_of_sound(alt)


def q_dyn(alt: Tensor | float, V: Tensor | float) -> Tensor:
    """Dynamic pressure [Pa] = 0.5*rho*V^2."""
    v = _as_tensor(V)
    return 0.5 * rho_air(alt) * v * v


def reynolds(alt: Tensor | float, V: Tensor | float, L: Tensor | float) -> Tensor:
    """Reynolds number based on reference length `L` [m]."""
    v = _as_tensor(V)
    L_t = _as_tensor(L)
    return rho_air(alt) * v * L_t / dynamic_viscosity(alt)
