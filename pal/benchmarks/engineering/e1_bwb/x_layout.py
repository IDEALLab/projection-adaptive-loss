"""Design-variable layout for E1 BWB.

x[B, 36] = [ shape(9) | L(1) | struct(19) | battery(6) | alpha_cr(1) ]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch
from torch import Tensor

N_SHAPE: Final[int] = 9       # [B1, B2, B3, C2, C3, C4, S1, S2, S3], C1 fixed at 1000 mm
N_L: Final[int] = 1
N_STRUCT: Final[int] = 19
N_BATTERY: Final[int] = 6     # cx, cy, cz, w, d, h
N_ALPHA: Final[int] = 1
DIM: Final[int] = N_SHAPE + N_L + N_STRUCT + N_BATTERY + N_ALPHA  # 36

_OFF_SHAPE = 0
_OFF_L = _OFF_SHAPE + N_SHAPE
_OFF_STRUCT = _OFF_L + N_L
_OFF_BATTERY = _OFF_STRUCT + N_STRUCT
_OFF_ALPHA = _OFF_BATTERY + N_BATTERY


@dataclass(frozen=True)
class DecodedX:
    """Named view over a `[B, DIM]` design vector. All fields share batch dim B."""

    shape: Tensor     # [B, 9]  BWB SDF conditioning [B1, B2, B3, C2, C3, C4, S1, S2, S3]
    L: Tensor         # [B, 1]  overall span scale (metres, log-uniform [0.1, 10])
    struct: Tensor    # [B, 19] structural-surrogate latent / geometry knobs
    battery: Tensor   # [B, 6]  (cx, cy, cz, w, d, h) in SDF frame, nondim (scaled by L)
    alpha_cr: Tensor  # [B, 1]  cruise AoA (rad)


def decode_x(x: Tensor) -> DecodedX:
    """Slice `x[B, DIM]` into the five named fields (views on `x`)."""
    if x.shape[-1] != DIM:
        raise ValueError(f"decode_x expected last dim {DIM}, got {tuple(x.shape)}")
    return DecodedX(
        shape=x[..., _OFF_SHAPE:_OFF_SHAPE + N_SHAPE],
        L=x[..., _OFF_L:_OFF_L + N_L],
        struct=x[..., _OFF_STRUCT:_OFF_STRUCT + N_STRUCT],
        battery=x[..., _OFF_BATTERY:_OFF_BATTERY + N_BATTERY],
        alpha_cr=x[..., _OFF_ALPHA:_OFF_ALPHA + N_ALPHA],
    )


def encode_x(
    shape: Tensor,
    L: Tensor,
    struct: Tensor,
    battery: Tensor,
    alpha_cr: Tensor,
) -> Tensor:
    """Inverse of `decode_x`. All inputs must share device, dtype, and batch dim."""
    for name, t, width in [
        ("shape", shape, N_SHAPE),
        ("L", L, N_L),
        ("struct", struct, N_STRUCT),
        ("battery", battery, N_BATTERY),
        ("alpha_cr", alpha_cr, N_ALPHA),
    ]:
        if t.shape[-1] != width:
            raise ValueError(f"encode_x: {name} expected last dim {width}, got {tuple(t.shape)}")
    return torch.cat([shape, L, struct, battery, alpha_cr], dim=-1)


def default_bounds(dtype: torch.dtype = torch.float32) -> tuple[Tensor, Tensor]:
    """Output box bounds for the 36-dim design vector."""
    lo = torch.empty(DIM, dtype=dtype)
    hi = torch.empty(DIM, dtype=dtype)

    # shape(9): unit box, the SDF net normalizes internally.
    lo[_OFF_SHAPE:_OFF_SHAPE + N_SHAPE] = -1.0
    hi[_OFF_SHAPE:_OFF_SHAPE + N_SHAPE] = 1.0

    # L(1): [0.5, 2.0] m, UAV-scale BWB.
    lo[_OFF_L:_OFF_L + N_L] = 0.5
    hi[_OFF_L:_OFF_L + N_L] = 2.0

    # struct(19): +/-3sigma on the latent dims.
    lo[_OFF_STRUCT:_OFF_STRUCT + N_STRUCT] = -3.0
    hi[_OFF_STRUCT:_OFF_STRUCT + N_STRUCT] = 3.0

    # battery(6): centre in [-2, 2], extent in [0.05, 0.15] (nondim by L).
    lo[_OFF_BATTERY + 0:_OFF_BATTERY + 3] = -2.0
    hi[_OFF_BATTERY + 0:_OFF_BATTERY + 3] = 2.0
    import os as _os
    _ext_hi = float(_os.environ.get("E1_BATTERY_EXT_HI", "0.15"))
    lo[_OFF_BATTERY + 3:_OFF_BATTERY + 6] = 0.05
    hi[_OFF_BATTERY + 3:_OFF_BATTERY + 6] = _ext_hi

    # alpha_cr(1): [-3 deg, 3 deg] in radians. Negative AoAs rare but legal.
    three_deg = 3.0 * torch.pi / 180.0
    lo[_OFF_ALPHA:_OFF_ALPHA + N_ALPHA] = -three_deg
    hi[_OFF_ALPHA:_OFF_ALPHA + N_ALPHA] = three_deg

    return lo, hi
