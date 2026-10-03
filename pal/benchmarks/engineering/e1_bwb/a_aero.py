"""A_aero: global (CL, CD, CM) surrogate from trained MLP on DeCoDe cases.

Inputs are `(shape[-1,1]^9, Ma, alt, alpha_deg, log10 L)`, normalised as in training.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import Tensor, nn

from . import atmosphere
from ._artifacts import ensure_artifact_path
from .x_layout import DecodedX

# Indices inside the normalised flight vector (shared with training script).
_IDX_L_LOG = 3  # L is log10-transformed
_IDX_CD_LOG = 1  # CD (output) is log10-transformed


class _AAeroMLP(nn.Module):
    def __init__(self, in_dim: int = 13, hidden: int = 128, out_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class AAeroSurrogate(nn.Module):
    """Satisfies the `ComputeAero` protocol: `(x, conditions) -> (CL, CD, CM)`.

    Conditions layout: `conditions[B, 2] = (alt_m, V_cruise_m_per_s)`.
    """

    def __init__(
        self,
        model: _AAeroMLP,
        stats: dict,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        if device is not None:
            self.model = self.model.to(device)

        anchor = next(self.model.parameters())
        tk = {"dtype": torch.float32, "device": anchor.device}

        self.register_buffer("shape_min", torch.tensor(stats["shape_min"], **tk))
        self.register_buffer("shape_max", torch.tensor(stats["shape_max"], **tk))
        self.register_buffer("flight_mean", torch.tensor(stats["flight_mean"], **tk))
        self.register_buffer("flight_std", torch.tensor(stats["flight_std"], **tk))
        self.register_buffer("target_mean", torch.tensor(stats["target_mean"], **tk))
        self.register_buffer("target_std", torch.tensor(stats["target_std"], **tk))

    @classmethod
    def load_default(
        cls,
        checkpoint: Path | str | None = None,
        norm_stats: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> AAeroSurrogate:
        ckpt_path = Path(
            ensure_artifact_path(checkpoint if checkpoint else "a_aero_weights")
        )
        stats_path = Path(
            ensure_artifact_path(norm_stats if norm_stats else "a_aero_norm_stats")
        )

        bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        arch = bundle.get("arch", {"in_dim": 13, "hidden": 128, "out_dim": 3})
        with open(stats_path) as f:
            stats = json.load(f)

        model = _AAeroMLP(**arch)
        model.load_state_dict(bundle["state_dict"])
        model.eval()
        return cls(model, stats, device=device)

    def _normalise_inputs(self, x: DecodedX, conditions: Tensor) -> Tensor:
        alt = conditions[..., 0]
        V = conditions[..., 1]
        Ma = atmosphere.mach(V, alt)
        alpha_deg = x.alpha_cr.squeeze(-1) * (180.0 / torch.pi)
        L = x.L.squeeze(-1)
        log_L = torch.log10(L)

        shape_min = self.shape_min.to(device=x.shape.device, dtype=x.shape.dtype)
        shape_max = self.shape_max.to(device=x.shape.device, dtype=x.shape.dtype)
        # Design-vector shape is already in [-1, 1], as in training.
        shape_norm = x.shape
        _ = (shape_min, shape_max)

        flight_raw = torch.stack([Ma, alt, alpha_deg, log_L], dim=-1)
        flight_mean = self.flight_mean.to(device=flight_raw.device, dtype=flight_raw.dtype)
        flight_std = self.flight_std.to(device=flight_raw.device, dtype=flight_raw.dtype)
        flight_norm = (flight_raw - flight_mean) / flight_std

        return torch.cat([shape_norm, flight_norm], dim=-1)

    def _unnormalise_outputs(self, y_norm: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean = self.target_mean.to(device=y_norm.device, dtype=y_norm.dtype)
        std = self.target_std.to(device=y_norm.device, dtype=y_norm.dtype)
        y = y_norm * std + mean
        CL = y[..., 0]
        CD = torch.pow(10.0, y[..., 1])
        CM = y[..., 2]
        return CL, CD, CM

    def forward(self, x: DecodedX, conditions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        features = self._normalise_inputs(x, conditions)
        y_norm = self.model(features)
        return self._unnormalise_outputs(y_norm)

    def __call__(self, x: DecodedX, conditions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return super().__call__(x, conditions)
