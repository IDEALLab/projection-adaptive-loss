"""Structural CVAE surrogate architecture.

Encoder(struct | bwb) -> mu, log_var. Decoder(z | bwb) -> struct_recon.
Predictor(z, bwb, thick, y) -> section properties. RibPredictor(z, bwb,
rib_thickness) -> rib volume.
"""

from __future__ import annotations

from .build_wingbox import (
    BWB_PARAM_NAMES,
    STRUCT_PARAM_NAMES,
    build_config,
    build_program,
)
from .model import StructuralCVAE, kl_divergence

__all__ = [
    "BWB_PARAM_NAMES",
    "STRUCT_PARAM_NAMES",
    "StructuralCVAE",
    "build_config",
    "build_program",
    "kl_divergence",
]
