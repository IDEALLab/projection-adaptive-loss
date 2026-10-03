"""FiLM-based aerodynamic loads (q_z, m, x_cp per spanwise station), physical units.

Slices the BWB CADProgram per station, runs FiLM on the contour points and
integrates chordwise with arc-length weights. `y_stations` are physical metres.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from . import atmosphere
from ._artifacts import ensure_artifact_path
from .film_surface.model import FiLMNet
from .interfaces import Loads
from .x_layout import DecodedX

_THIS_DIR = Path(__file__).resolve().parent
_BWB_YAML = _THIS_DIR / "bwb_sdf" / "bwb_wing.yaml"

_SHAPE_COLS_9 = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
_MM_DIMS = 6  # B1, B2, B3, C2, C3, C4
_DEG_DIMS = 3  # S1, S2, S3
_C1_MM = 1000.0  # Dataset-fixed centreline length, inserted at FiLM index 3.


def _load_film_net(checkpoint: Path, device: torch.device | str) -> nn.Module:
    """Instantiate FiLMNet + load the shipped checkpoint."""
    sd = torch.load(str(checkpoint), map_location=device, weights_only=False)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]

    model = FiLMNet(
        cond_dim=13,
        coord_dim=6,
        output_dim=3,
        hidden_dim=256,
        num_layers=4,
        extra_layers=3,
    )
    model.load_state_dict(sd)
    model.to(device).eval()
    return model


def build_bwb_program(x: DecodedX, device: torch.device | str | None = None) -> Any:
    """Construct a batched `CADProgram` from `x.shape` (unit frame, `x.L` not applied)."""
    from geometry.program import CADProgram  # lazy

    shape_raw = _unnormalise_x_shape(x.shape)  # [B, 9]  (mm / deg in DeCoDe units)
    shape_ratio = _shape_raw_to_sdf_ratio(shape_raw)  # [B, 9]  SDF ingests ratios

    prog = CADProgram.load_from_yaml(str(_BWB_YAML), device=device)
    prog = prog.with_params(
        **{
            name: shape_ratio[..., i : i + 1]
            for i, name in enumerate(_SHAPE_COLS_9)
        },
    )
    return prog


def _unnormalise_x_shape(x_shape: Tensor) -> Tensor:
    """Map `x.shape in [-1, 1]^9` to raw DeCoDe units (mm for B/C, deg for S)."""
    shape_min, shape_max = _load_aaero_shape_ranges(
        device=x_shape.device, dtype=x_shape.dtype,
    )
    return shape_min + 0.5 * (x_shape + 1.0) * (shape_max - shape_min)


def _shape_raw_to_sdf_ratio(shape_raw: Tensor) -> Tensor:
    """mm / deg -> ratio / deg (SDF convention). B/C columns / 1000."""
    out = shape_raw.clone()
    out[..., :_MM_DIMS] = out[..., :_MM_DIMS] / 1000.0
    return out


def _shape_raw_to_film_10(shape_raw: Tensor) -> Tensor:
    """Insert `C1 = 1000 mm` at index 3 -> FiLM's 10-dim shape vector."""
    B = shape_raw.shape[:-1]
    c1 = shape_raw.new_full((*B, 1), _C1_MM)
    return torch.cat(
        [shape_raw[..., :3], c1, shape_raw[..., 3:]], dim=-1,
    )


_AAERO_CACHE: dict[str, Tensor] = {}


def _load_aaero_shape_ranges(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    path: Path | str | None = None,
) -> tuple[Tensor, Tensor]:
    p = Path(ensure_artifact_path(path if path else "a_aero_norm_stats"))
    key = f"{p}|{device}|{dtype}"
    if key not in _AAERO_CACHE:
        with open(p) as f:
            stats = json.load(f)
        _AAERO_CACHE[f"{key}|min"] = torch.tensor(
            stats["shape_min"], device=device, dtype=dtype,
        )
        _AAERO_CACHE[f"{key}|max"] = torch.tensor(
            stats["shape_max"], device=device, dtype=dtype,
        )
    return _AAERO_CACHE[f"{key}|min"], _AAERO_CACHE[f"{key}|max"]


@dataclass(frozen=True)
class _FiLMNormStats:
    coord_min: Tensor
    coord_max: Tensor
    flight_mean: Tensor
    flight_std: Tensor
    shape_mean: Tensor
    shape_std: Tensor
    output_mean: Tensor
    output_std: Tensor


def _load_film_norm_stats(
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
    path: Path | str | None = None,
) -> _FiLMNormStats:
    p = Path(ensure_artifact_path(path if path else "film_norm_stats"))
    with open(p) as f:
        raw = json.load(f)
    def t(k):
        return torch.tensor(raw[k], device=device, dtype=dtype)
    return _FiLMNormStats(
        coord_min=t("coord_min"),
        coord_max=t("coord_max"),
        flight_mean=t("flight_mean"),
        flight_std=t("flight_std"),
        shape_mean=t("shape_mean"),
        shape_std=t("shape_std"),
        output_mean=t("output_mean"),
        output_std=t("output_std"),
    )


class FiLMLoads(nn.Module):
    """Satisfies the `ComputeLoads` protocol.

    Args:
        film_model: the trained FiLMNet (eval mode).
        film_stats: parsed `norm_stats.json`.
        isocontour_base_res: quadtree base grid resolution per axis (8 default,
            4 is fine for unit tests).
        isocontour_levels: quadtree refinement levels (5 default; 3 OK for tests).
    """

    def __init__(
        self,
        film_model: nn.Module,
        film_stats: _FiLMNormStats,
        isocontour_base_res: int = 8,
        isocontour_levels: int = 5,
    ) -> None:
        super().__init__()
        self.film = film_model
        self.stats = film_stats
        self.isocontour_base_res = int(isocontour_base_res)
        self.isocontour_levels = int(isocontour_levels)

    @classmethod
    def load_default(
        cls,
        checkpoint: Path | str | None = None,
        norm_stats: Path | str | None = None,
        device: torch.device | str | None = None,
        isocontour_base_res: int = 8,
        isocontour_levels: int = 5,
    ) -> FiLMLoads:
        device = torch.device(device) if device is not None else torch.device("cpu")
        ckpt = Path(ensure_artifact_path(checkpoint if checkpoint else "film_weights"))
        stats = _load_film_norm_stats(device=device, path=norm_stats)
        model = _load_film_net(ckpt, device)
        return cls(
            model, stats,
            isocontour_base_res=isocontour_base_res,
            isocontour_levels=isocontour_levels,
        )

    def _build_film_cond(
        self, Re: Tensor, Ma: Tensor, alpha_deg: Tensor, shape_raw: Tensor,
    ) -> Tensor:
        """Build z-normalised condition vector `[B, 13]`.

        Args:
            Re, Ma, alpha_deg: each `[B]`.
            shape_raw: `[B, 9]` in DeCoDe units (mm / deg).
        """
        flight = torch.stack([Re, Ma, alpha_deg], dim=-1)
        flight_norm = (flight - self.stats.flight_mean) / self.stats.flight_std
        shape_10 = _shape_raw_to_film_10(shape_raw)  # [B, 10]
        shape_norm = (shape_10 - self.stats.shape_mean) / self.stats.shape_std
        return torch.cat([flight_norm, shape_norm], dim=-1)

    def _film_forward(self, xyz_unit: Tensor, normals: Tensor, cond: Tensor) -> Tensor:
        """Run FiLM on one slice.

        Args:
            xyz_unit: `[K, 3]` unit-scale surface points (LE at +x).
            normals: `[K, 3]` raw outward 3D normals.
            cond: `[13]` pre-built z-normalised condition for this batch index.

        Returns:
            `[K, 3]` physical (cp, cf_x, cf_z).
        """
        denom = (self.stats.coord_max - self.stats.coord_min).clamp_min(1e-12)
        xyz_norm = 2.0 * (xyz_unit - self.stats.coord_min) / denom - 1.0
        coords6 = torch.cat([xyz_norm, normals], dim=-1)
        cond_K = cond.unsqueeze(0).expand(coords6.shape[0], -1)
        pred_norm = self.film(coords6, cond_K)
        return pred_norm * self.stats.output_std + self.stats.output_mean

    def __call__(
        self,
        program: Any,
        x: DecodedX,
        conditions: Tensor,
        y_stations: Tensor,
    ) -> Loads:
        if program is None:
            program = build_bwb_program(x, device=x.L.device)

        B, N = y_stations.shape
        L = x.L.squeeze(-1)                                   # [B]
        alt = conditions[..., 0]
        V = conditions[..., 1]
        Ma = atmosphere.mach(V, alt)
        Re = atmosphere.reynolds(alt, V, L)                   # Physical L
        q_inf = atmosphere.q_dyn(alt, V)                      # [B] Pa
        alpha_deg = x.alpha_cr.squeeze(-1) * (180.0 / torch.pi)

        shape_raw = _unnormalise_x_shape(x.shape)             # [B, 9]
        cond_all = self._build_film_cond(Re, Ma, alpha_deg, shape_raw)  # [B, 13]

        y_unit = y_stations / L.unsqueeze(-1).clamp_min(1e-12)

        # Out-of-place accumulation (in-place scatter trips autograd version checks).
        zero = q_inf.new_zeros(())                           # [] scalar
        q_z_cells: list[torch.Tensor] = []
        m_cells: list[torch.Tensor] = []
        x_cp_cells: list[torch.Tensor] = []

        # isocontour takes one station list shared across the batch, so call per sample.
        for b in range(B):
            stations_b = [float(y_unit[b, s].detach().cpu()) for s in range(N)]
            results = program.isocontour(
                plane="xz", stations=stations_b, normal_mode="3d",
                base_res=self.isocontour_base_res, levels=self.isocontour_levels,
            )
            cond_b = cond_all[b]  # [13]
            for s in range(N):
                slice_res = results[s]
                polys = slice_res.polylines[b]       # list[Tensor [K_i, 2]]
                norms = slice_res.normals[b]         # list[Tensor [K_i, 3]]
                ws = slice_res.weights[b]            # list[Tensor [K_i]]
                if not polys:
                    # No contour intersection at this y (e.g. beyond wing tip).
                    q_z_cells.append(zero)
                    m_cells.append(zero)
                    x_cp_cells.append(zero)
                    continue

                poly_xz = torch.cat(polys, dim=0)            # [K, 2]
                normals3d = torch.cat(norms, dim=0)          # [K, 3]
                weights = torch.cat(ws, dim=0)               # [K]

                # Stations are sampling locations, detached from the graph.
                y_fill = torch.full_like(
                    poly_xz[:, 0:1], float(y_unit[b, s].detach()),
                )
                xyz_unit = torch.cat(
                    [poly_xz[:, 0:1], y_fill, poly_xz[:, 1:2]], dim=-1,
                )  # [K, 3]

                preds = self._film_forward(xyz_unit, normals3d, cond_b)  # [K, 3]
                cp = preds[:, 0]
                cf_x = preds[:, 1]
                cf_z = preds[:, 2]

                # Non-dimensional surface force density: F/(q_inf*A) = -cp * n_hat + (cf_x, 0, cf_z)
                fx_nd = -cp * normals3d[:, 0] + cf_x
                fz_nd = -cp * normals3d[:, 2] + cf_z

                Iz = (fz_nd * weights).sum()
                # Moment about LE (x=0), nose-up +: m_unit = sum (x * F_z - z * F_x) * w
                Im = (poly_xz[:, 0] * fz_nd - poly_xz[:, 1] * fx_nd) * weights
                Im = Im.sum()

                # Force per span q_inf * L * I_nd, moment per span q_inf * L^2 * I_nd.
                q_zL = q_inf[b] * L[b]
                q_zL2 = q_zL * L[b]
                q_z_bs = q_zL * Iz
                m_bs = q_zL2 * Im
                # Signed x_cp about LE (physical metres), guarded against q_z ~ 0.
                safe_qz = torch.where(
                    q_z_bs.abs() < 1e-6,
                    q_z_bs.new_full((), 1e-6),
                    q_z_bs,
                )
                q_z_cells.append(q_z_bs)
                m_cells.append(m_bs)
                x_cp_cells.append(m_bs / safe_qz)

        q_z = torch.stack(q_z_cells, dim=0).reshape(B, N)
        m = torch.stack(m_cells, dim=0).reshape(B, N)
        x_cp = torch.stack(x_cp_cells, dim=0).reshape(B, N)
        return Loads(q_z=q_z, m=m, x_cp=x_cp)
