"""E1BWB: blended-wing-body multi-physics MDO benchmark.

x (36D) = 9 shape + 1 span scale + 19 struct + 6 battery + 1 cruise AoA;
conditions (2D) = (alt, V_cruise). Objective is Breguet range.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
from torch import Tensor

from pal.benchmarks.base import Query
from pal.constraints import Constraint

from ._artifacts import check_runtime_artifacts
from .atmosphere import G, q_dyn
from .loads import _unnormalise_x_shape, build_bwb_program
from .spec import N_STATIONS_DEFAULT, make_spec
from .stubs import (
    compute_aero_stub,
    compute_loads_stub,
    compute_stress_stub,
    compute_structural_stub,
)
from .x_layout import DIM, DecodedX, decode_x

VIZ_FINAL_DATA_SCHEMA = "e1/viz_final_data@1"

# High-performance Li-ion pack.
RHO_BAT: float = 2400.0         # kg/m^3
E_STAR_BAT: float = 300.0 * 3600.0  # Wh/kg -> J/kg
M_PAYLOAD: float = 3.0          # kg
ETA_PROP: float = 0.8           # motor + prop chain efficiency
# CFRP quasi-isotropic laminate, averaged into one effective modulus and strain allowable.
E_MAT: float = 55e9             # Pa (quasi-iso averaged modulus, 0 deg/+/-45 deg/90 deg)
EPS_ALLOW: float = 4e-3         # 4000 ustrain, post-knockdown tension allowable
RHO_MAT: float = 1600.0         # kg/m^3
# Tip deflection constraint: |w_tip| / (DEFLECTION_FRAC * semi_span) <= 1.
DEFLECTION_FRAC: float = 0.10
# Cruise envelope, overridable via env for diagnostic sweeps.
_ALT_LO = float(os.environ.get("E1_ALT_LO", "0.0"))
_ALT_HI = float(os.environ.get("E1_ALT_HI", "4000.0"))
_V_LO = float(os.environ.get("E1_V_LO", "25.0"))
_V_HI = float(os.environ.get("E1_V_HI", "80.0"))

# obj = -R / R_REF, R_REF keeps the objective gradient O(0.1) at init.
R_REF: float = 1.0e7


class E1BWB:
    """Blended-wing-body multi-physics MDO benchmark."""

    supports_final_data_dump: bool = True

    def __init__(
        self,
        device: str | torch.device | None = None,
        n_stations: int = N_STATIONS_DEFAULT,
        live: bool = True,
        compute_aero: Callable | None = None,
        compute_loads: Callable | None = None,
        compute_structural: Callable | None = None,
        compute_stress: Callable | None = None,
    ) -> None:
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.n_stations = int(n_stations)
        self.spec = make_spec(n_stations=self.n_stations)

        if live:
            compute_aero = compute_aero or _default_aero(self.device)
            compute_loads = compute_loads or _default_loads(self.device)
            compute_structural = compute_structural or _default_structural(self.device)
            compute_stress = compute_stress or _default_stress()

        self._compute_aero = compute_aero or compute_aero_stub
        self._compute_loads = compute_loads or compute_loads_stub
        self._compute_structural = compute_structural or compute_structural_stub
        self._compute_stress = compute_stress or compute_stress_stub

        # Only FiLM loads needs a CADProgram.
        self._needs_program = self._compute_loads is not compute_loads_stub

        self._last_program: Any | None = None
        # Per-station strain residuals [B, N] of the last forward, detached.
        self._last_strain_per_station: Tensor | None = None
        # Mass breakdown of the last forward (each [B], detached).
        self._last_mass_breakdown: dict[str, Tensor] | None = None

    def _planform(self, x_decoded: DecodedX) -> tuple[Tensor, Tensor]:
        """Closed-form (semi_span [m], S_ref [m^2]) from `x.shape` + `x.L` (C1 = 1 m)."""
        shape_raw = _unnormalise_x_shape(x_decoded.shape)              # [B, 9]
        B = shape_raw[..., 0:3] / 1000.0                               # [B, 3] station lengths (ratio)
        C_tail = shape_raw[..., 3:6] / 1000.0                          # [B, 3] chords C2..C4 (ratio)
        C1 = torch.ones_like(x_decoded.L)                              # [B, 1] C1 = 1.0
        C = torch.cat([C1, C_tail], dim=-1)                            # [B, 4] C1..C4
        S_half_unit = 0.5 * ((C[..., :-1] + C[..., 1:]) * B).sum(dim=-1)  # [B]
        L = x_decoded.L.squeeze(-1)                                    # [B]
        semi_span = B.sum(dim=-1) * L                                  # [B] m
        S_ref = 2.0 * S_half_unit * L * L                              # [B] m^2
        return semi_span, S_ref

    def _sample_y_stations(self, zeta: Tensor, semi_span: Tensor) -> Tensor:
        """Sobol stations on [0, semi_span], seeded per sample by zeta, sorted ascending.

        Uses scipy Sobol rather than torch SobolEngine so it works under functorch transforms.
        """
        import numpy as np
        from scipy.stats import qmc

        B, N = zeta.shape[0], self.n_stations
        zeta_cpu = zeta.detach().cpu()
        draws_np: list[np.ndarray] = []
        for b in range(B):
            seed = int(torch.sum(zeta_cpu[b] * 1e6).abs()) % (2**31)
            sampler = qmc.Sobol(d=1, scramble=True, seed=seed)
            draw = sampler.random(N).reshape(N)               # [N] numpy
            draws_np.append(draw)
        stacked = torch.from_numpy(np.stack(draws_np, axis=0)).to(
            device=zeta.device, dtype=zeta.dtype
        )                                                      # [B, N]
        scaled = stacked * semi_span.unsqueeze(-1)
        return scaled.sort(dim=-1).values

    def _battery_volume(self, x_decoded: DecodedX) -> Tensor:
        """Physical battery-box volume (m^3) = (w*d*h)*L^3."""
        w = x_decoded.battery[..., 3]
        d = x_decoded.battery[..., 4]
        h = x_decoded.battery[..., 5]
        L = x_decoded.L.squeeze(-1)
        return w * d * h * L**3

    def _skin_thickness(self, x_decoded: DecodedX) -> Tensor:
        """Physical skin thickness (m) from `x.struct[:, 16]`, nominal if the plug is the stub."""
        from .beam import NOMINAL_SKIN_THICKNESS_M

        mean_buf = getattr(self._compute_structural, "thick_mean", None)
        std_buf = getattr(self._compute_structural, "thick_std", None)
        if mean_buf is None or std_buf is None:
            return x_decoded.L.new_full((x_decoded.L.shape[0],), NOMINAL_SKIN_THICKNESS_M)
        z = x_decoded.struct[:, 16]
        return z * std_buf[0].to(z.device, z.dtype) + mean_buf[0].to(z.device, z.dtype)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None,
    ) -> tuple[Tensor, list[Constraint]]:
        """End-to-end forward. Returns (objective [B], constraint list).

        Objective: `-R / R_REF` (pure Breguet range).
        Constraints: lift_balance (eq), strain_agg (ineq, = mean(ReLU(c_i))
        over N_stations cross-sections), tip_deflection (ineq).
        """
        if x.shape[-1] != DIM:
            raise ValueError(f"E1BWB.forward: x last dim must be {DIM}, got {tuple(x.shape)}")
        if x.dim() != 2:
            raise ValueError(f"E1BWB.forward: x must be [B, {DIM}], got {tuple(x.shape)}")
        if conditions is None:
            raise ValueError("E1BWB requires conditions=[B, 2] (alt, V_cruise)")
        if conditions.shape != (x.shape[0], 2):
            raise ValueError(
                f"conditions must be [B, 2], got {tuple(conditions.shape)}"
            )

        B = x.shape[0]
        dev = x.device
        dec = decode_x(x)
        alt = conditions[..., 0]
        V = conditions[..., 1]

        q = q_dyn(alt, V)
        semi_span, S_ref = self._planform(dec)

        zeta_surrogate = x
        y_stations = self._sample_y_stations(zeta_surrogate, semi_span)

        if self._needs_program:
            self._last_program = build_bwb_program(dec, device=dev)

        skin_thickness = self._skin_thickness(dec)

        CL, CD, CM = self._compute_aero(dec, conditions)
        loads = self._compute_loads(self._last_program, dec, conditions, y_stations)
        props = self._compute_structural(dec, y_stations)
        sigma_max, m_struct, w = self._compute_stress(
            loads, props, y_stations, dec,
            skin_thickness=skin_thickness,
            rho_material=RHO_MAT,
            e_modulus=E_MAT,
        )

        V_bat = self._battery_volume(dec)
        m_bat = RHO_BAT * V_bat
        m_total = m_struct + m_bat + M_PAYLOAD

        LD = CL / (CD + 1e-12)
        R = (ETA_PROP / G) * E_STAR_BAT * LD * (m_bat / (m_total + 1e-12))
        objective = -R / R_REF

        weight = m_total * G
        lift = CL * q * S_ref
        # Equality: lift == weight.
        lift_balance_val = (weight - lift) / (weight + 1e-12)

        self._last_mass_breakdown = {
            "m_struct": m_struct.detach(),
            "m_bat": m_bat.detach(),
            "m_payload": torch.full_like(m_struct.detach(), float(M_PAYLOAD)),
            "m_total": m_total.detach(),
            "weight": weight.detach(),
            "lift": lift.detach(),
            "L": dec.L.squeeze(-1).detach(),
            "semi_span": semi_span.detach(),
            "S_ref": S_ref.detach(),
        }

        # Per-station residuals are kept for reporting; training sees mean(ReLU(c_i)).
        strain_vals_per_station = (sigma_max / E_MAT) / EPS_ALLOW - 1.0  # [B, N]
        strain_agg_val = strain_vals_per_station.clamp_min(0.0).mean(dim=-1)  # [B]
        self._last_strain_per_station = strain_vals_per_station.detach()

        w_tip = w[..., -1].abs()                              # [B]
        tip_allow = DEFLECTION_FRAC * semi_span + 1e-12
        tip_deflection_val = w_tip / tip_allow - 1.0          # [B]

        constraints: list[Constraint] = []
        tol = torch.full((B,), float(self.spec.tolerance), device=dev)
        margin = torch.full((B,), 1e-4, device=dev)
        # lift_balance is an equality, margin unused.
        constraints.append(
            Constraint(
                value=lift_balance_val, type="eq", tol=tol,
                margin=torch.zeros(B, device=dev),
                name="lift_balance",
            )
        )
        constraints.append(
            Constraint(
                value=strain_agg_val, type="ineq",
                tol=tol, margin=margin, name="strain_agg",
            )
        )
        constraints.append(
            Constraint(
                value=tip_deflection_val, type="ineq",
                tol=tol, margin=margin, name="tip_deflection",
            )
        )
        return objective, constraints

    def diagnose_mass_sensitivity(
        self, x: Tensor, conditions: Tensor,
    ) -> dict[str, Tensor]:
        """One-shot autograd probe: dm_struct / dx for the 3 thickness slots.
        Slots 26/27/28 are skin / front-spar / rear-spar thickness (z-scored).
        Returns a dict of [B] tensors, one sensitivity per thickness slot.
        """
        x_probe = x.detach().clone().requires_grad_(True)
        dec_p = decode_x(x_probe)
        semi_span_p, _ = self._planform(dec_p)
        y_p = self._sample_y_stations(x_probe, semi_span_p)
        program_p = (
            build_bwb_program(dec_p, device=x_probe.device)
            if self._needs_program else None
        )
        skin_t_p = self._skin_thickness(dec_p)
        loads_p = self._compute_loads(program_p, dec_p, conditions, y_p)
        props_p = self._compute_structural(dec_p, y_p)
        _, m_struct_p, _ = self._compute_stress(
            loads_p, props_p, y_p, dec_p,
            skin_thickness=skin_t_p,
            rho_material=RHO_MAT,
            e_modulus=E_MAT,
        )
        grads = torch.autograd.grad(m_struct_p.sum(), x_probe)[0]
        return {
            "dm_struct_dthick_skin": grads[:, 26].detach(),
            "dm_struct_dthick_fspar": grads[:, 27].detach(),
            "dm_struct_dthick_rspar": grads[:, 28].detach(),
        }

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        """Stacked `[B, n_eq + n_ineq]` in spec order."""
        _, clist = self.forward(x, conditions)
        return torch.stack([c.value for c in clist], dim=-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None,
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def sample_queries(
        self, n: int, split: Literal["train", "eval"], seed: int,
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        alt = torch.rand(n, 1, generator=g) * (_ALT_HI - _ALT_LO) + _ALT_LO
        V = torch.rand(n, 1, generator=g) * (_V_HI - _V_LO) + _V_LO
        conditions = torch.cat([alt, V], dim=-1).to(torch.float32)
        return Query(zeta=zeta, conditions=conditions)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + 7919)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        alt = torch.rand(n, 1, generator=g) * (_ALT_HI - _ALT_LO) + _ALT_LO
        V = torch.rand(n, 1, generator=g) * (_V_HI - _V_LO) + _V_LO
        conditions = torch.cat([alt, V], dim=-1).to(torch.float32)
        return Query(zeta=zeta, conditions=conditions)

    def check_env(self) -> None:
        """Verify the 6 live surrogates' on-disk artefacts are available."""
        check_runtime_artifacts()

    def visualize_train(
        self, x: Tensor, conditions: Tensor | None = None,
    ) -> Any | None:
        """Cheap-ish train viz: BWB OML + spars + Sobol stations overlaid."""
        try:
            from .viz import render_train_stations
        except Exception:
            return None
        try:
            return render_train_stations(self, x, conditions)
        except Exception as exc:
            import sys
            import traceback
            print(f"[e1 visualize_train] failed: {exc}", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            return None

    def save_final_data(
        self,
        path: Path | str,
        x: Tensor,
        conditions: Tensor | None = None,
        *,
        resolution: int = 256,
    ) -> None:
        """Pickle the data needed to render the hero composite offline.

        `resolution` is the SDF meshing resolution (256 for hero quality)."""
        from .viz import collect_hero_data

        data = collect_hero_data(self, x, conditions, resolution=resolution)
        payload = {
            "schema": VIZ_FINAL_DATA_SCHEMA,
            "x": x.detach().cpu(),
            "conditions": (
                conditions.detach().cpu() if conditions is not None else None
            ),
            **data,
            "metadata": {
                "benchmark_id": "e1/bwb",
                "git_sha": os.environ.get("PAL_GIT_SHA", ""),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, out)

    def visualize_final(
        self, x: Tensor, conditions: Tensor | None = None,
    ) -> Any | None:
        """Hero composite: BWB + spars + ribs + battery left, Cp + deflection right.

        With `PAL_VIZ_FINAL_DUMP=1`, pickles the arrays to `PAL_VIZ_FINAL_DUMP_PATH` instead."""
        if os.environ.get("PAL_VIZ_FINAL_DUMP") == "1":
            out = Path(
                os.environ.get(
                    "PAL_VIZ_FINAL_DUMP_PATH", "/tmp/viz_final_data.pt"
                )
            )
            resolution = int(os.environ.get("PAL_VIZ_FINAL_RESOLUTION", "256"))
            try:
                self.save_final_data(
                    out, x, conditions, resolution=resolution,
                )
            except Exception as exc:
                import sys
                print(
                    f"[e1 visualize_final] dump failed: {exc}",
                    file=sys.stderr,
                )
            return None

        try:
            from .viz import render_hero
        except Exception:
            return None
        try:
            fig = render_hero(self, x, conditions)
        except Exception as exc:
            import sys
            print(f"[e1 visualize_final] failed: {exc}", file=sys.stderr)
            return None
        return {"hero": fig}


def _default_aero(device: torch.device) -> Callable:
    from .a_aero import AAeroSurrogate
    return AAeroSurrogate.load_default(device=device)


def _default_loads(device: torch.device) -> Callable:
    from .loads import FiLMLoads
    return FiLMLoads.load_default(device=device)


def _default_structural(device: torch.device) -> Callable:
    from .struct import StructSurrogate
    return StructSurrogate.load_default(device=device)


def _default_stress() -> Callable:
    from .beam import compute_stress
    return compute_stress
