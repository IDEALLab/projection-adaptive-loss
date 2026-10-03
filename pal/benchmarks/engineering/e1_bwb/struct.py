"""Structural surrogate wrapper satisfying the `ComputeStructural` protocol.

Returns 7 nondimensional `StructProps` fields `[B, N]`, L-scaling happens in `beam.py`.
x.struct[:, :16] is the VAE latent z, x.struct[:, 16:19] the z-normalised thickness.
Outputs are log-standardised: props_unit = exp(pred_norm * std + mean).
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn

from ._artifacts import ensure_artifact_path
from .interfaces import StructProps
from .loads import _shape_raw_to_sdf_ratio, _unnormalise_x_shape
from .struct_surrogate.model import StructuralCVAE
from .x_layout import DecodedX

# Index into the surrogate's 11-field output.
_IDX_A = 0
_IDX_U_CG = 1
_IDX_V_CG = 2
_IDX_I_UU = 3
_IDX_I_VV = 4
_IDX_Q_U_MAX = 8
_IDX_Q_V_MAX = 9
_IDX_J = 10

_LATENT_SLICE = slice(0, 16)
_THICK_SLICE = slice(16, 19)


class StructSurrogate(nn.Module):
    """`ComputeStructural` implementation wrapping `StructuralCVAE`."""

    def __init__(
        self,
        model: nn.Module,
        norms: dict[str, dict[str, Tensor]],
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.model = model.eval()
        if device is not None:
            self.model = self.model.to(device)

        anchor = next(self.model.parameters())
        tk = {"dtype": torch.float32, "device": anchor.device}

        self.register_buffer("bwb_mean", norms["bwb"]["mean"].to(**tk))
        self.register_buffer("bwb_std", norms["bwb"]["std"].to(**tk))
        self.register_buffer("y_mean", norms["y"]["mean"].to(**tk))
        self.register_buffer("y_std", norms["y"]["std"].to(**tk))
        self.register_buffer("props_mean", norms["props"]["mean"].to(**tk))
        self.register_buffer("props_std", norms["props"]["std"].to(**tk))
        # thick stats are optional (older checkpoints lack them).
        if "thick" in norms:
            self.register_buffer("thick_mean", norms["thick"]["mean"].to(**tk))
            self.register_buffer("thick_std", norms["thick"]["std"].to(**tk))

    @classmethod
    def load_default(
        cls,
        checkpoint: Path | str | None = None,
        device: torch.device | str | None = None,
    ) -> StructSurrogate:
        ckpt_path = Path(
            ensure_artifact_path(checkpoint if checkpoint else "struct_weights"),
        )
        ck = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
        cfg = ck["config"]
        model = StructuralCVAE(
            latent_dim=cfg["latent_dim"],
            hidden_dim=cfg["hidden_dim"],
            depth=cfg["depth"],
            dropout=cfg["dropout"],
        )
        model.load_state_dict(ck["model_state_dict"])
        norms = {
            "bwb": ck["norm_bwb"],
            "y": ck["norm_y"],
            "props": ck["norm_props"],
        }
        if "norm_thick" in ck:
            norms["thick"] = ck["norm_thick"]
        return cls(model, norms, device=device)

    def _build_bwb10(self, x: DecodedX) -> Tensor:
        """`x.shape` + `x.L` -> raw BWB-10 `[B1..C4(ratios), S1..S3(deg), L(m)]`."""
        shape_raw = _unnormalise_x_shape(x.shape)        # [B, 9]  DeCoDe units
        shape_ratio = _shape_raw_to_sdf_ratio(shape_raw)  # mm -> ratio for B/C
        return torch.cat([shape_ratio, x.L], dim=-1)     # [B, 10]

    def __call__(self, x: DecodedX, y_stations: Tensor) -> StructProps:
        B, N = y_stations.shape

        z = x.struct[..., _LATENT_SLICE]                 # [B, 16]
        thick_n = x.struct[..., _THICK_SLICE]            # [B,  3]  already z-normed
        bwb_raw = self._build_bwb10(x)                   # [B, 10]
        bwb_n = (bwb_raw - self.bwb_mean) / self.bwb_std

        # y: physical -> unit-frame -> z-scored.
        L = x.L.squeeze(-1).clamp_min(1e-12)
        y_unit = y_stations / L.unsqueeze(-1)            # [B, N]
        y_norm = (y_unit - self.y_mean) / self.y_std     # [B, N]

        # Flatten (B, N) -> (B*N) so one batched forward covers every station.
        z_rep = z.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        bwb_rep = bwb_n.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        thick_rep = thick_n.unsqueeze(1).expand(B, N, -1).reshape(B * N, -1)
        y_rep = y_norm.reshape(B * N, 1)

        pred_n = self.model.predict(z_rep, bwb_rep, thick_rep, y_rep)  # [B*N, 11]
        pred_unit = torch.exp(pred_n * self.props_std + self.props_mean)
        pred_unit = pred_unit.reshape(B, N, -1)          # [B, N, 11]

        # Q_u_max, Q_v_max are non-negative, so max gives the worst transverse shear.
        Q_max = torch.maximum(
            pred_unit[..., _IDX_Q_U_MAX],
            pred_unit[..., _IDX_Q_V_MAX],
        )

        return StructProps(
            I_uu=pred_unit[..., _IDX_I_UU],
            I_vv=pred_unit[..., _IDX_I_VV],
            J=pred_unit[..., _IDX_J],
            A=pred_unit[..., _IDX_A],
            u_cg=pred_unit[..., _IDX_U_CG],
            v_cg=pred_unit[..., _IDX_V_CG],
            Q_max=Q_max,
        )

    def forward(self, x: DecodedX, y_stations: Tensor) -> StructProps:  # pragma: no cover
        return self.__call__(x, y_stations)
