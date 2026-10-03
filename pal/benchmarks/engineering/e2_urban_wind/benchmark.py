"""E2: Urban Wind 50D box benchmark (WinDiNet surrogate).

Wraps `UrbanWindBenchmark` from the vendored windinet slice
(`_vendor/windinet`, `_vendor/inverse`, plus sibling `urban_benchmark.py`)
as a pal `Benchmark`-compliant class.

Decision variable (50D default, N=10 buildings): 10 x (cx, cy, w_raw, d_raw, h_raw).
Objective: `-(total_volume / V_ref)`. Constraints: `site`, `clearance`,
`height`, `danger` (inequalities) and `coverage` (equality).
Set `E2_N_BUILDINGS=20` for the 100D variant.

sys.path is prepared by the package `__init__.py` (or `conftest.py` under
pytest) before this module loads, so the unqualified imports below resolve
against the ported files and vendored packages.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import Tensor
from urban_benchmark import (
    CITY_YAML_PATH,
    UrbanWindBenchmark,
    choose_device,
    load_default_surrogate,
)

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

VIZ_FINAL_DATA_SCHEMA = "e2/viz_final_data@1"


_N_BUILDINGS = int(os.environ.get("E2_N_BUILDINGS", "10"))
_DIM = 5 * _N_BUILDINGS  # 50 for N=10 default; set E2_N_BUILDINGS=20 for the 100D variant
_DOMAIN_LO = 175.0
_DOMAIN_HI = 925.0

# E2_CLEARANCE_PER_PAIR=1 splits the aggregated clearance constraint into
# N(N-1)/2 per-pair scalars (45 for N=10), each with raw relu overlap.
_PER_PAIR_CLEARANCE = os.environ.get("E2_CLEARANCE_PER_PAIR", "0") == "1"
_N_PAIRS = _N_BUILDINGS * (_N_BUILDINGS - 1) // 2
if _PER_PAIR_CLEARANCE:
    _CLEARANCE_NAMES = [f"clearance_p{i:02d}" for i in range(_N_PAIRS)]
else:
    _CLEARANCE_NAMES = ["clearance"]
_CONSTRAINT_NAMES = ["site", *_CLEARANCE_NAMES, "height", "danger", "coverage"]


def _kept_constraint_names() -> list[str]:
    """Constraint names kept after `E2_DROP_CONSTRAINTS` (comma-separated) filtering."""
    drop = os.environ.get("E2_DROP_CONSTRAINTS", "")
    drop_set = {n.strip() for n in drop.split(",") if n.strip()}
    return [n for n in _CONSTRAINT_NAMES if n not in drop_set]

# Pre-scaling applied to obj and c before the solver sees them, so active
# constraints land in an O(0.1-1) range. Override via constructor args or env
# (E2_OBJ_SCALE / E2_C_SCALES_JSON).
_DEFAULT_OBJECTIVE_SCALE = 1.0
_DEFAULT_CONSTRAINT_SCALES = {
    "site": 10.0,
    # Per-pair clearance (relu, m^2) is O(0.1-10) when violating; aggregated
    # clearance (squared-mean, normalised) is O(1e-7).
    **({name: 1.0 for name in _CLEARANCE_NAMES}
       if _PER_PAIR_CLEARANCE else {"clearance": 1000.0}),
    "height": 1.0,
    "danger": 1.0,
    "coverage": 1.0,
}


def _make_spec() -> BenchmarkSpec:
    # cx/cy bounds extend past the feasible [175, 925] site so the design range
    # stays in the tanh head's linear regime; the site constraint enforces feasibility.
    lo = torch.tensor(
        [-100.0] * _N_BUILDINGS  # cx
        + [-100.0] * _N_BUILDINGS  # cy
        + [-5.0] * _N_BUILDINGS  # w_raw
        + [-5.0] * _N_BUILDINGS  # d_raw
        + [-5.0] * _N_BUILDINGS,  # h_raw
        dtype=torch.float32,
    )
    hi = torch.tensor(
        [1200.0] * _N_BUILDINGS
        + [1200.0] * _N_BUILDINGS
        + [100.0] * _N_BUILDINGS
        + [100.0] * _N_BUILDINGS
        + [250.0] * _N_BUILDINGS,
        dtype=torch.float32,
    )
    kept = _kept_constraint_names()
    n_kept = len(kept)
    # `coverage` is an equality (footprint == max_coverage * site); all others are inequalities.
    constraint_types = ["eq" if n == "coverage" else "ineq" for n in kept]
    n_eq = constraint_types.count("eq")
    n_ineq = n_kept - n_eq
    return BenchmarkSpec(
        id="e2/urban_wind",
        family="e2",
        variant="urban_wind",
        dim=_DIM,
        n_eq=n_eq,
        n_ineq=n_ineq,
        constraint_names=list(kept),
        constraint_types=constraint_types,
        output_bounds=(lo, hi),
        condition_dim=0,
        zeta_dim=16,
        tolerance=1e-4,
        tau=1e-4,
        cost="expensive",
        recommended_device="gpu",
        precision="fp32",
        recommended_batch_per_gpu={"A100-80GB": 4, "H100-80GB": 8},
        train_batch_size=32,
        n_eval_default=32,
        # pal_loggap: rate=0.1 lets lambda ramp fast, max_decades=5 raises the per-step growth cap.
        solver_hparams={
            "pal_loggap": {"rate": 0.1, "max_decades": 5.0, "epochs": 200},
        },
        notes="10-building WinDiNet surrogate; fp32 for gradient fidelity",
    )


def _render_final_pyvista(
    *,
    program_config: Any,
    program_params: Any,
    program_batch_size: int,
    tail_mean_speed: Tensor,
    tail_mean_u: Tensor,
    tail_mean_v: Tensor,
    building_mask: Tensor,
    decoded_cx: Tensor,
    decoded_cy: Tensor,
    decoded_w: Tensor,
    decoded_d: Tensor,
    decoded_h: Tensor,
    hard_danger_fraction: float,
    constraint_vec: Tensor,
    total_volume: float,
) -> dict[str, Any] | None:
    """Render the e2 3D plots from precomputed tensors (no forward pass).

    Returns `{"3d_wind": Figure, "3d_comfort": Figure}`, or `None` if
    PyVista/VTK are unavailable."""
    try:
        import matplotlib.pyplot as plt
        import pyvista as pv
        from matplotlib import cm, colors

        from geometry import FieldSlice
        from geometry.render.box_detect import extract_boxes
        from geometry.render.pyvista import _add_field_slice
    except ImportError:
        return None

    def _build_plotter() -> pv.Plotter:
        plotter = pv.Plotter(off_screen=True, window_size=(2800, 2100))
        plotter.set_background("white")
        boxes = extract_boxes(
            program_config, program_params, program_batch_size,
        )
        if boxes is not None:
            for box in boxes:
                if box.name == "inner_domain":
                    continue  # skip the domain wireframe
                plotter.add_mesh(
                    pv.Box(bounds=box.bounds),
                    color="lightgrey", smooth_shading=False,
                    specular=0.25, specular_power=10,
                )
        plotter.camera_position = [
            (2400, -1000, 1500), (550, 550, 40), (0, 0, 1),
        ]
        plotter.camera.zoom(1.18)
        plotter.add_axes()
        return plotter

    def _composite(arr, *, cmap: str, clim: tuple[float, float],
                   cbar_label: str, title: str):
        fig, ax = plt.subplots(figsize=(13.5, 9.5), dpi=170)
        ax.imshow(arr)
        ax.set_axis_off()
        sm = cm.ScalarMappable(
            norm=colors.Normalize(vmin=clim[0], vmax=clim[1]), cmap=cmap,
        )
        cbar = fig.colorbar(sm, ax=ax, shrink=0.78, pad=0.015,
                            orientation="vertical")
        cbar.set_label(cbar_label, fontsize=11)
        cbar.ax.tick_params(labelsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=8)
        fig.tight_layout()
        return fig

    # 3d_wind: tail-mean wind speed slice over the full raster domain.
    plotter_w = _build_plotter()
    _add_field_slice(plotter_w, FieldSlice(
        values=tail_mean_speed,
        extents=((0, 1100), (0, 1100)),
        plane="xy", offset=2.0,
        name="tail mean wind speed",
        cmap="coolwarm", clim=(0, 15), opacity=0.8,
    ), actor_name="wind")
    plotter_w.remove_scalar_bar()
    wind_arr = plotter_w.screenshot(return_img=True)
    plotter_w.close()
    fig_wind = _composite(
        wind_arr, cmap="coolwarm", clim=(0.0, 15.0),
        cbar_label="tail-mean wind speed (m/s)",
        title=(
            f"volume = {total_volume:.0f} m^3    "
            f"danger = {100 * hard_danger_fraction:.2f} %"
        ),
    )

    # 3d_comfort: comfort overlay (wind speed on fluid cells, NaN under
    # buildings so they read as bare grey).
    plotter_c = _build_plotter()
    zone_speed = tail_mean_speed.clone()
    zone_speed[building_mask > 0.5] = float("nan")
    _add_field_slice(plotter_c, FieldSlice(
        values=zone_speed,
        extents=((0, 1100), (0, 1100)),
        plane="xy", offset=1.0,
        name="comfort zone",
        cmap="RdYlGn_r", clim=(0, 15), opacity=0.85,
    ), actor_name="comfort")
    plotter_c.remove_scalar_bar()
    comfort_arr = plotter_c.screenshot(return_img=True)
    plotter_c.close()
    # constraint_vec follows _CONSTRAINT_NAMES order:
    # [site, clearance, height, danger, coverage].
    fig_comfort = _composite(
        comfort_arr, cmap="RdYlGn_r", clim=(0.0, 15.0),
        cbar_label="comfort zone wind speed (m/s)",
        title=(
            f"c_site={constraint_vec[0].item():.3f}   "
            f"c_clearance={constraint_vec[1].item():.3f}   "
            f"c_height={constraint_vec[2].item():.3f}   "
            f"c_danger={constraint_vec[3].item():.3f}   "
            f"c_coverage={constraint_vec[4].item():.3f}"
        ),
    )

    return {"3d_wind": fig_wind, "3d_comfort": fig_comfort}


class E2UrbanWind:
    """pal Benchmark wrapper around `UrbanWindBenchmark`."""

    _batch_diag_printed: bool = False
    supports_final_data_dump: bool = True

    def __init__(
        self,
        device: str | None = None,
        *,
        objective_scale: float | None = None,
        constraint_scales: dict[str, float] | None = None,
    ) -> None:
        self.spec = _make_spec()
        self.device = choose_device(device)
        surrogate, self._diffusion_dir = load_default_surrogate(device=self.device)
        self._bench = UrbanWindBenchmark(
            yaml_path=CITY_YAML_PATH,
            device=self.device,
            surrogate=surrogate,
        )
        # Most recent BenchmarkResult, reused by visualize_train / visualize_final.
        self._last_result: Any | None = None

        # Pre-scaling: ctor arg > env override > default.
        env_obj = os.environ.get("E2_OBJ_SCALE")
        if objective_scale is not None:
            self.objective_scale = float(objective_scale)
        elif env_obj is not None:
            self.objective_scale = float(env_obj)
        else:
            self.objective_scale = _DEFAULT_OBJECTIVE_SCALE

        scales = dict(_DEFAULT_CONSTRAINT_SCALES)
        env_scales = os.environ.get("E2_C_SCALES_JSON")
        if env_scales:
            scales.update({k: float(v) for k, v in json.loads(env_scales).items()})
        # Per-constraint single-value env overrides (E2_SCALE_<NAME>).
        for name in list(scales.keys()):
            single = os.environ.get(f"E2_SCALE_{name.upper()}")
            if single:
                scales[name] = float(single)
        if constraint_scales is not None:
            scales.update({k: float(v) for k, v in constraint_scales.items()})
        # All constraints are computed, then indexed into the kept subset.
        self._kept_names: list[str] = _kept_constraint_names()
        self._kept_indices: list[int] = [
            _CONSTRAINT_NAMES.index(n) for n in self._kept_names
        ]
        self._constraint_scale_vec = torch.tensor(
            [scales[n] for n in self._kept_names],
            dtype=torch.float32, device=self.device,
        )
        self._constraint_scales = {n: scales[n] for n in self._kept_names}
        dropped = [n for n in _CONSTRAINT_NAMES if n not in self._kept_names]
        print(
            f"[e2] pre-scale: objectivex{self.objective_scale:g}, "
            f"constraintsx{self._constraint_scales}"
            + (f" (dropped: {dropped})" if dropped else "")
        )

    def _split(self, x: Tensor) -> dict[str, Tensor]:
        N = _N_BUILDINGS
        return {
            "cx": x[..., 0:N],
            "cy": x[..., N:2 * N],
            "w": x[..., 2 * N:3 * N],
            "d": x[..., 3 * N:4 * N],
            "h": x[..., 4 * N:5 * N],
        }

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        # Move x to the surrogate's device (callers may pass CPU tensors).
        if x.device.type != self.device.split(":")[0]:
            x = x.to(self.device)
        B = x.shape[0]
        if not self._batch_diag_printed:
            from parallel_surrogate import ParallelSurrogate, memory_warning
            n_replicas = (
                self._bench.surrogate.n_replicas
                if isinstance(self._bench.surrogate, ParallelSurrogate)
                else 1
            )
            per_gpu = B // n_replicas
            rem = B % n_replicas
            print(
                f"[e2] first forward: global batch={B}, "
                f"replicas={n_replicas}, per-GPU={per_gpu}"
                + (f" (+{rem} remainder)" if rem else "")
            )
            warn = memory_warning(per_gpu_batch=max(per_gpu, 1))
            if warn:
                print(f"[e2] {warn}")
            self._batch_diag_printed = True

        result = self._bench.evaluate_raw_params(self._split(x))
        self._last_result = result
        if _PER_PAIR_CLEARANCE:
            c_kept = self._build_per_pair_kept(result, x.device)  # [B, n_kept]
        else:
            c_full = result.constraint_tensor()  # [B, 5]
            c_kept = c_full[:, self._kept_indices]  # [B, n_kept]
        c = c_kept * self._constraint_scale_vec.to(
            device=x.device, dtype=result.objective.dtype
        )
        obj = result.objective * self.objective_scale
        B = x.shape[0]
        dev = x.device
        # `coverage` is an equality (|ratio - max_coverage| <= tol, tol > 0);
        # the inequalities use tol=0, margin=1e-4.
        return obj, [
            Constraint(
                value=c[:, j],
                type="eq" if name == "coverage" else "ineq",
                tol=(
                    torch.full((B,), 1e-3, device=dev)
                    if name == "coverage"
                    else torch.zeros(B, device=dev)
                ),
                margin=(
                    torch.zeros(B, device=dev)
                    if name == "coverage"
                    else torch.full((B,), 1e-4, device=dev)
                ),
                name=name,
            )
            for j, name in enumerate(self._kept_names)
        ]

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        if x.device.type != self.device.split(":")[0]:
            x = x.to(self.device)
        result = self._bench.evaluate_raw_params(self._split(x))
        if _PER_PAIR_CLEARANCE:
            c_kept = self._build_per_pair_kept(result, x.device)
        else:
            c_full = result.constraint_tensor()
            c_kept = c_full[:, self._kept_indices]
        return c_kept * self._constraint_scale_vec.to(device=c_kept.device, dtype=c_kept.dtype)

    def _build_per_pair_kept(self, result: Any, x_device: torch.device) -> Tensor:
        """Stack scalars + per-pair clearance into [B, n_kept] in `_kept_names` order.

        Each pair's value is the unnormalised m^2 overlap of inflated AABBs
        (raw relu), exactly 0 when feasible."""
        decoded = result.decoded
        bench = self._bench
        pair_i = bench._pair_i.to(decoded.cx.device)
        pair_j = bench._pair_j.to(decoded.cx.device)
        half_w_sum = 0.5 * (decoded.w[:, pair_i] + decoded.w[:, pair_j]) + 0.5 * bench.d_min
        half_d_sum = 0.5 * (decoded.d[:, pair_i] + decoded.d[:, pair_j]) + 0.5 * bench.d_min
        gx = half_w_sum - (decoded.cx[:, pair_i] - decoded.cx[:, pair_j]).abs()
        gy = half_d_sum - (decoded.cy[:, pair_i] - decoded.cy[:, pair_j]).abs()
        clearance_pairs = F.relu(gx) * F.relu(gy)  # [B, n_pairs]

        scalars = result.constraints  # dict: site/clearance/height/danger/coverage
        cols: list[Tensor] = []
        for name in self._kept_names:
            if name.startswith("clearance_p"):
                idx = int(name[len("clearance_p"):])
                cols.append(clearance_pairs[:, idx:idx + 1])
            else:
                cols.append(scalars[name].unsqueeze(1))
        return torch.cat(cols, dim=1).to(x_device)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def sample_queries(
        self,
        n: int,
        split: Literal["train", "eval"],
        seed: int,
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + 7919)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def check_env(self) -> None:
        """Verify the WinDiNet surrogate can be loaded end-to-end.

        Raises on missing checkpoints, revision mismatch, or failure to
        initialize the LTX-Video surrogate (e.g. no network access for
        base LTXV weights).
        """
        # Lazily instantiated in __init__; if we got here, the surrogate loaded.
        _ = self._bench

    def _result_for_viz(self, x: Tensor) -> Any:
        """Return the cached BenchmarkResult, or run and cache a forward if none exists."""
        result = self._last_result
        if result is None:
            import sys

            print(
                "  [e2] viz: no cached forward; running an extra "
                "WinDiNet pass (~30s on CPU). Plot from training/eval "
                "forward to avoid.",
                file=sys.stderr,
            )
            result = self._bench.evaluate_raw_params(
                self._split(x.to(self.device))
            )
            self._last_result = result
        return result

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None):
        """3-panel composite (tail-mean wind speed, hard comfort zones,
        clearance overlay).

        Reuses `self._last_result` from the most recent forward pass, or runs
        one forward if the cache is empty.
        """
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import BoundaryNorm, ListedColormap
        except ImportError:
            return None

        x_single = (
            x.detach().cpu().reshape(1, -1) if x.dim() == 1
            else x.detach().cpu()[:1]
        )
        result = self._result_for_viz(x_single)
        d_min = 4.0

        speed = result.tail_mean_speed[0].detach().cpu().numpy()
        bldg = result.building_mask[0].detach().cpu().numpy() > 0.5
        zones = result.hard_zone_labels[0].detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(22, 7))
        ext = [0, 1100, 0, 1100]

        # Panel 0: tail-mean wind speed
        ax = axes[0]
        im = ax.imshow(speed, origin="lower", cmap="coolwarm",
                       vmin=0, vmax=15, extent=ext)
        ax.contour(bldg, levels=[0.5], colors="black", linewidths=1.0,
                   extent=ext)
        ax.add_patch(plt.Rectangle(
            (175, 175), 750, 750, fill=False, edgecolor="green",
            linewidth=1.5, linestyle="--",
        ))
        ax.set_title("Tail-mean wind speed [m/s]", fontsize=12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")
        fig.colorbar(im, ax=ax, shrink=0.8, label="m/s")

        # Panel 1: hard comfort zones (categorical)
        ax = axes[1]
        zone_cmap = ListedColormap(
            ["#404040", "#4dac26", "#b8e186", "#f4a742", "#d7191c"]
        )
        zone_norm = BoundaryNorm(
            [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], zone_cmap.N,
        )
        im = ax.imshow(
            zones, origin="lower", cmap=zone_cmap, norm=zone_norm, extent=ext,
        )
        cbar = fig.colorbar(im, ax=ax, shrink=0.8, ticks=[0, 1, 2, 3, 4])
        cbar.ax.set_yticklabels([
            "Building", "Calm\n<1", "Acceptable\n1-5",
            "Uncomfortable\n5-15", "Danger\n>15",
        ])
        ax.add_patch(plt.Rectangle(
            (175, 175), 750, 750, fill=False, edgecolor="white",
            linewidth=1.5, linestyle="--",
        ))
        ax.set_title("Hard comfort zones (tail mean)", fontsize=12)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_aspect("equal")

        # Panel 2: clearance overlay (red lines for violating pairs)
        ax = axes[2]
        ax.set_xlim(0, 1100)
        ax.set_ylim(0, 1100)
        ax.set_aspect("equal")
        ax.add_patch(plt.Rectangle(
            (175, 175), 750, 750, fill=False, edgecolor="green",
            linewidth=1.5, linestyle="--",
        ))
        cx = result.decoded.cx[0].detach().cpu().numpy()
        cy = result.decoded.cy[0].detach().cpu().numpy()
        w = result.decoded.w[0].detach().cpu().numpy()
        d = result.decoded.d[0].detach().cpu().numpy()
        N = cx.shape[0]
        for i in range(N):
            ax.add_patch(plt.Rectangle(
                (cx[i] - w[i] / 2, cy[i] - d[i] / 2), w[i], d[i],
                facecolor="lightgrey", edgecolor="black", linewidth=0.8,
            ))
        n_violations = 0
        for i in range(N):
            for j in range(i + 1, N):
                half_w = 0.5 * (w[i] + w[j]) + 0.5 * d_min
                half_d = 0.5 * (d[i] + d[j]) + 0.5 * d_min
                gap_x = half_w - abs(cx[i] - cx[j])
                gap_y = half_d - abs(cy[i] - cy[j])
                if gap_x > 0 and gap_y > 0:
                    ax.plot([cx[i], cx[j]], [cy[i], cy[j]], "r-",
                            linewidth=1.5, alpha=0.7)
                    n_violations += 1
        ax.set_title(
            f"Clearance (d_min={d_min} m, {n_violations} pairs violating)",
            fontsize=12,
        )
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")

        c_tensor = result.constraint_tensor()[0].detach().cpu().numpy()
        coverage_str = f"  |  coverage_viol={c_tensor[4]:+.3e}"
        fig.suptitle(
            (
                f"volume={result.total_volume[0].item():.0f} m^3"
                f"  |  danger={100*result.hard_danger_fraction[0].item():.2f}%"
                f"{coverage_str}"
            ),
            fontsize=14,
        )
        plt.tight_layout()
        return fig

    def _viz_final_payload(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> dict[str, Any]:
        """Build the tensors/metadata needed by `_render_final_pyvista`."""
        x_single = (
            x.detach().cpu().reshape(1, -1) if x.dim() == 1
            else x.detach().cpu()[:1]
        )
        result = self._result_for_viz(x_single)
        return {
            "program_config": result.program.config,
            "program_params": result.program.params,
            "program_batch_size": result.program.batch_size,
            "tail_mean_speed": result.tail_mean_speed[0].detach().cpu(),
            "tail_mean_u": result.tail_mean_u[0].detach().cpu(),
            "tail_mean_v": result.tail_mean_v[0].detach().cpu(),
            "building_mask": result.building_mask[0].detach().cpu(),
            "decoded_cx": result.decoded.cx[0].detach().cpu(),
            "decoded_cy": result.decoded.cy[0].detach().cpu(),
            "decoded_w": result.decoded.w[0].detach().cpu(),
            "decoded_d": result.decoded.d[0].detach().cpu(),
            "decoded_h": result.decoded.h[0].detach().cpu(),
            "hard_danger_fraction": float(
                result.hard_danger_fraction[0].item()
            ),
            "constraint_vec": result.constraint_tensor()[0].detach().cpu(),
            "total_volume": float(result.total_volume[0].item()),
        }

    def save_final_data(
        self,
        path: Path | str,
        x: Tensor,
        conditions: Tensor | None = None,
    ) -> None:
        """Save the tensors/metadata needed to render the 3D plots offline."""
        payload = self._viz_final_payload(x, conditions)
        data = {
            "schema": VIZ_FINAL_DATA_SCHEMA,
            "x": x.detach().cpu(),
            "conditions": (
                conditions.detach().cpu() if conditions is not None else None
            ),
            **{
                k: v for k, v in payload.items()
                if k not in ("program_config", "program_params",
                             "program_batch_size")
            },
            "program": {
                "config": payload["program_config"],
                "params": payload["program_params"],
                "batch_size": payload["program_batch_size"],
            },
            "metadata": {
                "benchmark_id": "e2/urban_wind",
                "git_sha": os.environ.get("PAL_GIT_SHA", ""),
                "timestamp": datetime.now(UTC).isoformat(),
            },
        }
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(data, out)

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None):
        """PyVista renders wrapped in matplotlib figures with colorbars.

        Returns `{"3d_wind": Figure, "3d_comfort": Figure}`.

        When `PAL_VIZ_FINAL_DUMP=1`, skips rendering, saves the required
        tensors to `PAL_VIZ_FINAL_DUMP_PATH` (default `/tmp/viz_final_data.pt`)
        and returns `None`.

        Reuses `self._last_result` from the most recent forward pass.
        """
        if os.environ.get("PAL_VIZ_FINAL_DUMP") == "1":
            out = Path(
                os.environ.get(
                    "PAL_VIZ_FINAL_DUMP_PATH", "/tmp/viz_final_data.pt"
                )
            )
            self.save_final_data(out, x, conditions)
            return None

        try:
            import pyvista  # noqa: F401, probe VTK availability
        except ImportError:
            return None

        payload = self._viz_final_payload(x, conditions)
        return _render_final_pyvista(**payload)
