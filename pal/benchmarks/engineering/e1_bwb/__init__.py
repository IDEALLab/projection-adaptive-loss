"""E1 BWB multi-physics MDO benchmark."""

from __future__ import annotations

from .benchmark import E1BWB
from .spec import make_spec
from .x_layout import DIM, DecodedX, decode_x, encode_x

__all__ = [
    "E1BWB",
    "DIM",
    "DecodedX",
    "decode_x",
    "encode_x",
    "make_spec",
]
