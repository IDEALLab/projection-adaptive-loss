"""e1 visualisation: train station plot and final hero composite."""

from __future__ import annotations

import gc
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from .loads import _shape_raw_to_sdf_ratio, _unnormalise_x_shape
from .x_layout import decode_x

_OML_COLOR = "#B0B0B0"
_OML_OPACITY = 0.35
_SPAR_COLOR = "#7DA4E1"      # light blue
_RIB_COLOR = "#CDF0C8"       # pale green
_BATTERY_COLOR = "#F08A2A"   # orange
_STATION_COLOR = "#1F1F1F"

_BWB_PARAM_NAMES = ("B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3")

# Fixed wingbox knobs that are not in the 36-D design vector.
_WINGBOX_FIXED_DEFAULTS: dict[str, float] = {
    "front_spar_x": -0.25,
    "front_spar_dev": 3.0,
    "rear_spar_x": -0.50,
    "bat_x": 1.0,
    "bat_y": 0.15,
    "bat_z": 0.06,
    "bat_z_center": 0.0,
    "rib_start_y": 0.20,
    "rib_end_y": 0.85,
    "rib_w": 0.025,
    "center_rib_w": 0.020,
}
_THICKNESS_FALLBACK: dict[str, float] = {
    "skin_t": 0.008,
    "front_spar_w": 0.030,
    "rear_spar_w": 0.035,
}


def _as_single_row(x: Tensor) -> Tensor:
    """Coerce `x` to a 2D [1, DIM] tensor, benchmarks receive batched x."""
    if x.dim() == 1:
        return x.unsqueeze(0)
    if x.dim() == 2:
        return x[:1]
    raise ValueError(f"viz expects x of shape [DIM] or [B, DIM], got {tuple(x.shape)}")


def _release_viz_gpu_state(bench: Any) -> None:
    """Drop GPU refs held by the viz CADProgram, then GC and empty the CUDA cache."""
    if hasattr(bench, "_last_program"):
        bench._last_program = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _as_single_cond(c: Tensor | None) -> Tensor | None:
    """Coerce `conditions` (1D `[2]` or `[B, 2]`) to `[1, 2]`, passing None through."""
    if c is None:
        return None
    if c.dim() == 1:
        return c.unsqueeze(0)
    if c.dim() == 2:
        return c[:1]
    raise ValueError(
        f"viz expects conditions of shape [2] or [B, 2], got {tuple(c.shape)}"
    )


def _shape_params_for(x_row: Tensor) -> dict[str, float]:
    """Convert `x_row.shape` (9-d, [-1,1]^9) to the SDF ratio param dict for `build_config`."""
    dec = decode_x(x_row)
    shape_raw = _unnormalise_x_shape(dec.shape)
    shape_ratio = _shape_raw_to_sdf_ratio(shape_raw).detach().cpu()
    return {
        name: float(shape_ratio[0, i])
        for i, name in enumerate(_BWB_PARAM_NAMES)
    }


def _thickness_params_for(bench: Any, x_row: Tensor) -> dict[str, float]:
    """Read thick_mean/thick_std from the struct surrogate if available."""
    struct = bench._compute_structural
    mean_buf = getattr(struct, "thick_mean", None)
    std_buf = getattr(struct, "thick_std", None)
    if mean_buf is None or std_buf is None:
        return dict(_THICKNESS_FALLBACK)
    z = decode_x(x_row).struct[0, 16:19]
    raw = (z * std_buf.to(z.device, z.dtype) + mean_buf.to(z.device, z.dtype))
    raw = raw.detach().cpu().tolist()
    return {"skin_t": float(raw[0]), "front_spar_w": float(raw[1]), "rear_spar_w": float(raw[2])}


def _build_wingbox_program(
    bench: Any,
    x_row: Tensor,
    resolution: int,
    extra_outputs: tuple[str, ...] = ("bwb", "spars", "ribs"),
) -> Any:
    """Build the analytical-BWB + wingbox CADProgram for viz, exposing `extra_outputs`."""
    import yaml as _yaml

    from geometry.program import CADProgram

    from .struct_surrogate.build_wingbox import HERE, build_config

    params = _shape_params_for(x_row)
    params.update(_WINGBOX_FIXED_DEFAULTS)
    params.update(_thickness_params_for(bench, x_row))

    cfg = build_config(params, resolution=resolution)
    for name in extra_outputs:
        if name not in cfg["output"] and name in cfg["shapes"]:
            cfg["output"][name] = name

    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", dir=HERE, delete=False,
    ) as f:
        _yaml.safe_dump(cfg, f)
        tmp_path = Path(f.name)
    try:
        prog = CADProgram.load_from_yaml(str(tmp_path), device=str(x_row.device))
    finally:
        tmp_path.unlink(missing_ok=True)
    prog._build_shapes()
    return prog


def _mesh_shape(prog: Any, shape_name: str, backend: str = "sparse-dmc") -> Any:
    """Mesh one named shape from `prog`, returns a pyvista.PolyData (or None)."""
    import numpy as _np

    from geometry.render.pyvista import _mesh_result_to_pv

    shape = prog._shapes[shape_name]
    mesh_result = shape.get_mesh(
        prog.xyz_min, prog.xyz_max, prog.resolution,
        backend=backend, device=prog._device,
    )
    pv_mesh, _bounds = _mesh_result_to_pv(mesh_result, _np)
    return pv_mesh


def _battery_box_mesh(x_row: Tensor):
    """Build a PyVista Box for the battery (unit frame)."""
    import pyvista as pv

    dec = decode_x(x_row)
    cx, cy, cz, w, d, h = dec.battery[0].detach().cpu().tolist()
    half = (0.5 * w, 0.5 * d, 0.5 * h)
    bounds = (cx - half[0], cx + half[0], cy - half[1], cy + half[1], cz - half[2], cz + half[2])
    return pv.Box(bounds=bounds)


def _offscreen_screenshot(add_actors, window_size=(900, 900), camera="iso",
                          zoom: float = 1.7, screen_shift: tuple[float, float] = (0.0, 0.0),
                          elev_delta: float = 0.0):
    """Render actors off-screen and return the RGBA array."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=window_size)
    plotter.set_background("white")
    add_actors(plotter)
    if camera == "iso":
        plotter.view_isometric()
        if elev_delta:
            plotter.camera.elevation(elev_delta)
        plotter.reset_camera(render=False)
        plotter.camera.zoom(zoom)
    if screen_shift != (0.0, 0.0):
        plotter.camera.SetWindowCenter(screen_shift[0], screen_shift[1])
    img = plotter.screenshot(return_img=True)
    plotter.close()
    return img


def render_train_stations(
    bench: Any,
    x_row: Tensor,
    conditions: Tensor,
    resolution: int = 64,
):
    """Train viz: BWB station polylines (xz isocontours mirrored to +/-y) in matplotlib 3D."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers projection)

    x_row = _as_single_row(x_row).to(bench.device)
    if conditions is not None:
        conditions = _as_single_cond(conditions).to(bench.device)
    dec = decode_x(x_row)

    semi_span, _S_ref = bench._planform(dec)
    y_stations_phys = bench._sample_y_stations(x_row, semi_span)        # [1, N]
    y_unit = (y_stations_phys / dec.L)[0].detach().cpu().tolist()

    prog = _build_wingbox_program(bench, x_row, resolution=resolution,
                                  extra_outputs=("bwb",))

    contours = prog.isocontour(plane="xz", stations=y_unit, normal_mode="3d",
                               name="bwb", base_res=8, levels=5)
    station_lines: list[np.ndarray] = []
    for res in contours:
        polys = res.polylines[0]     # per-batch (B=1), List[Tensor [K, 2]]
        y_val = float(res.station)
        for poly in polys:
            if poly.shape[0] < 2:
                continue
            xz = poly.detach().cpu().numpy()
            closed_xz = np.vstack([xz, xz[:1]])
            for y_signed in (y_val, -y_val):
                xyz = np.column_stack([
                    closed_xz[:, 0],
                    np.full(len(closed_xz), y_signed),
                    closed_xz[:, 1],
                ])
                station_lines.append(xyz)

    obj, clist = bench.forward(
        x_row, _as_single_cond(conditions),
    )
    R_km = -float(obj[0].detach()) / 1000.0
    name_to_val = {c.name: float(c.value[0].detach()) for c in clist}
    n_strain_viol = 0
    if bench._last_strain_per_station is not None:
        n_strain_viol = int((bench._last_strain_per_station[0] > 0).sum().item())

    # Battery box wireframe (unit frame, same as the station polylines).
    cx, cy, cz, w, d, h = dec.battery[0].detach().cpu().tolist()
    hw, hd, hh = 0.5 * w, 0.5 * d, 0.5 * h
    bat_corners = np.array([
        [cx - hw, cy - hd, cz - hh],
        [cx + hw, cy - hd, cz - hh],
        [cx + hw, cy + hd, cz - hh],
        [cx - hw, cy + hd, cz - hh],
        [cx - hw, cy - hd, cz + hh],
        [cx + hw, cy - hd, cz + hh],
        [cx + hw, cy + hd, cz + hh],
        [cx - hw, cy + hd, cz + hh],
    ])
    bat_edges = (
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    )

    fig = plt.figure(figsize=(9, 7.2), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    if station_lines:
        for xyz in station_lines:
            ax.plot(
                xyz[:, 0], xyz[:, 1], xyz[:, 2],
                color=_STATION_COLOR, linewidth=1.6,
            )
        for i, j in bat_edges:
            seg = bat_corners[[i, j]]
            ax.plot(
                seg[:, 0], seg[:, 1], seg[:, 2],
                color=_BATTERY_COLOR, linewidth=1.4,
            )
        all_pts = np.vstack([np.vstack(station_lines), bat_corners])
        x_lo, x_hi = float(all_pts[:, 0].min()), float(all_pts[:, 0].max())
        y_lo, y_hi = float(all_pts[:, 1].min()), float(all_pts[:, 1].max())
        z_lo, z_hi = float(all_pts[:, 2].min()), float(all_pts[:, 2].max())
        # Equal-aspect cube around the data so chord/span/thickness ratios read.
        spans = (x_hi - x_lo, y_hi - y_lo, z_hi - z_lo)
        max_span = max(spans) if max(spans) > 0 else 1.0
        for setter, lo, hi in (
            (ax.set_xlim, x_lo, x_hi),
            (ax.set_ylim, y_lo, y_hi),
            (ax.set_zlim, z_lo, z_hi),
        ):
            mid = 0.5 * (lo + hi)
            setter(mid - 0.5 * max_span, mid + 0.5 * max_span)
    ax.set_xlabel("x")
    ax.set_ylabel("y (span)")
    ax.set_zlabel("z")
    ax.view_init(elev=22, azim=-65)

    n_stations = len(y_unit)
    feas = (name_to_val.get("lift_balance", 0.0) <= 1e-2
            and name_to_val.get("tip_deflection", 0.0) <= 0.0
            and n_strain_viol == 0)
    title = (f"e1 train: R = {R_km:.1f} km | "
             f"strain violations {n_strain_viol}/{n_stations} | "
             f"tip_defl {name_to_val.get('tip_deflection', 0.0):+.3f} | "
             f"{'FEAS' if feas else 'INFEAS'}")
    ax.set_title(title, fontsize=10, loc="left", pad=8)
    fig.tight_layout()

    del prog, station_lines, contours
    _release_viz_gpu_state(bench)
    return fig


def _mesh_to_arrays(mesh) -> dict[str, np.ndarray] | None:
    """Strip a PyVista PolyData down to plain numpy arrays so it pickles cleanly."""
    if mesh is None:
        return None
    return {
        "points": np.asarray(mesh.points, dtype=np.float32).copy(),
        "faces": np.asarray(mesh.faces).copy(),
    }


def _arrays_to_mesh(data: dict[str, np.ndarray] | None):
    """Inverse of `_mesh_to_arrays`. Returns a fresh `pv.PolyData`."""
    if data is None:
        return None
    import pyvista as pv
    return pv.PolyData(data["points"], data["faces"])


def collect_hero_data(
    bench: Any,
    x_row: Tensor,
    conditions: Tensor,
    resolution: int = 256,
) -> dict[str, Any]:
    """Run the expensive parts of the hero pipeline and return a pickleable dict."""
    # Viz receives CPU tensors, the surrogates live on bench.device.
    x_row = _as_single_row(x_row).to(bench.device)
    if conditions is not None:
        conditions = _as_single_cond(conditions).to(bench.device)
    dec = decode_x(x_row)

    prog = _build_wingbox_program(
        bench, x_row, resolution=resolution,
        extra_outputs=("bwb", "spars", "ribs"),
    )
    bwb_mesh = _mesh_shape(prog, "bwb")
    spar_mesh = _mesh_shape(prog, "spars")
    rib_mesh = _mesh_shape(prog, "ribs")

    cx, cy, cz, w, d, h = dec.battery[0].detach().cpu().tolist()
    battery_bounds = (
        cx - 0.5 * w, cx + 0.5 * w,
        cy - 0.5 * d, cy + 0.5 * d,
        cz - 0.5 * h, cz + 0.5 * h,
    )

    obj, clist = bench.forward(
        x_row, _as_single_cond(conditions),
    )
    name_to_val = {c.name: float(c.value[0].detach()) for c in clist}
    R_km = -float(obj[0].detach()) / 1000.0
    n_stations = bench.n_stations
    n_strain_viol = 0
    if bench._last_strain_per_station is not None:
        n_strain_viol = int((bench._last_strain_per_station[0] > 0).sum().item())

    w_profile, y_profile = _recover_deflection_profile(
        bench, x_row, conditions,
    )
    cp_scalar = _sample_cp_on_mesh(bench, x_row, conditions, bwb_mesh)

    payload = {
        "meshes": {
            "bwb": _mesh_to_arrays(bwb_mesh),
            "spars": _mesh_to_arrays(spar_mesh),
            "ribs": _mesh_to_arrays(rib_mesh),
        },
        "battery_bounds": battery_bounds,
        "cp_scalar": np.asarray(cp_scalar, dtype=np.float32),
        "deflection": {
            "y_profile": np.asarray(y_profile, dtype=np.float64),
            "w_profile": np.asarray(w_profile, dtype=np.float64),
        },
        "stats": {
            "R_km": R_km,
            "name_to_val": name_to_val,
            "n_strain_viol": int(n_strain_viol),
            "n_stations": int(n_stations),
        },
    }

    del prog, bwb_mesh, spar_mesh, rib_mesh, cp_scalar
    _release_viz_gpu_state(bench)
    return payload


def render_hero_from_data(data: dict[str, Any]):
    """Render the composite Figure from the dict returned by `collect_hero_data`."""
    import matplotlib.pyplot as plt
    import pyvista as pv
    from matplotlib import cm
    from matplotlib import colors as mcolors
    from matplotlib.gridspec import GridSpec

    bwb_mesh = _arrays_to_mesh(data["meshes"]["bwb"])
    spar_mesh = _arrays_to_mesh(data["meshes"]["spars"])
    rib_mesh = _arrays_to_mesh(data["meshes"]["ribs"])
    battery_mesh = pv.Box(bounds=tuple(data["battery_bounds"]))

    cp_scalar = np.asarray(data["cp_scalar"])
    cp_clim = (-1.5, 0.5)
    w_profile = np.asarray(data["deflection"]["w_profile"])
    y_profile = np.asarray(data["deflection"]["y_profile"])
    stats = data["stats"]
    R_km = stats["R_km"]

    # Optional env knobs: E1_VIZ_HIDE_BATTERY, E1_VIZ_CAD_ZOOM, E1_VIZ_DEFL_OPACITY.
    hide_battery = os.environ.get("E1_VIZ_HIDE_BATTERY") == "1"
    cad_zoom = float(os.environ.get("E1_VIZ_CAD_ZOOM", "1.7"))
    defl_opacity = float(os.environ.get("E1_VIZ_DEFL_OPACITY", "1.0"))
    win_scale = float(os.environ.get("E1_VIZ_WINDOW_SCALE", "1.0"))
    _cad_win = (int(1200 * win_scale), int(1200 * win_scale))
    _side_win = (int(800 * win_scale), int(600 * win_scale))

    def _cad_actors(plotter: pv.Plotter) -> None:
        if bwb_mesh is not None:
            plotter.add_mesh(bwb_mesh, color=_OML_COLOR, opacity=_OML_OPACITY,
                             smooth_shading=True, specular=0.35, specular_power=18.0)
        if spar_mesh is not None:
            plotter.add_mesh(spar_mesh, color=_SPAR_COLOR, opacity=0.95,
                             smooth_shading=True)
        if rib_mesh is not None:
            plotter.add_mesh(rib_mesh, color=_RIB_COLOR, opacity=0.95,
                             smooth_shading=True)
        if not hide_battery:
            plotter.add_mesh(battery_mesh, color=_BATTERY_COLOR, opacity=1.0,
                             smooth_shading=False)

    img_cad = _offscreen_screenshot(
        _cad_actors, window_size=_cad_win, screen_shift=(0.3, 0.0),
        zoom=cad_zoom,
        elev_delta=float(os.environ.get("E1_VIZ_CAD_ELEV_DELTA", "0.0")),
    )

    def _cp_actors(plotter: pv.Plotter) -> None:
        if bwb_mesh is None:
            return
        m = bwb_mesh.copy()
        m["Cp"] = cp_scalar
        plotter.add_mesh(
            m, scalars="Cp", cmap="coolwarm", clim=cp_clim,
            smooth_shading=True, show_scalar_bar=False,
        )

    img_cp = _offscreen_screenshot(
        _cp_actors, window_size=_side_win, screen_shift=(0.25, 0.0),
    )

    warp_amp = 5.0
    w_abs_max = float(np.abs(w_profile).max()) if len(w_profile) else 0.0
    defl_clim = (0.0, max(w_abs_max, 1e-9))

    def _defl_actors(plotter: pv.Plotter) -> None:
        if bwb_mesh is None:
            return
        warped, w_on_mesh = _warp_mesh_by_deflection(
            bwb_mesh, y_profile, w_profile, amplification=warp_amp,
        )
        warped["|w|"] = np.abs(w_on_mesh)
        plotter.add_mesh(
            warped, scalars="|w|", cmap="viridis", clim=defl_clim,
            smooth_shading=True, show_scalar_bar=False, opacity=defl_opacity,
        )

    img_defl = _offscreen_screenshot(
        _defl_actors, window_size=_side_win, screen_shift=(0.25, 0.0),
    )

    fig = plt.figure(figsize=(16, 10), dpi=120)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[2.0, 1.0],
                  left=0.02, right=0.98, top=0.92, bottom=0.04,
                  wspace=0.05, hspace=0.1)
    ax_cad = fig.add_subplot(gs[:, 0])
    ax_cp = fig.add_subplot(gs[0, 1])
    ax_defl = fig.add_subplot(gs[1, 1])

    ax_cad.imshow(img_cad)
    ax_cad.axis("off")
    ax_cad.set_title(f"E1 BWB: R = {R_km:.1f} km", fontsize=12, loc="left", pad=8)

    ax_cp.imshow(img_cp)
    ax_cp.axis("off")
    sm_cp = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=cp_clim[0], vmax=cp_clim[1]), cmap="coolwarm",
    )
    cbar_cp = fig.colorbar(sm_cp, ax=ax_cp, shrink=0.82, pad=0.015,
                           orientation="vertical")
    cbar_cp.set_label("Cp  (FiLM)", fontsize=10)
    cbar_cp.ax.tick_params(labelsize=9)
    ax_cp.set_title("Surface pressure coefficient", fontsize=11, loc="left", pad=6)

    ax_defl.imshow(img_defl)
    ax_defl.axis("off")
    sm_defl = cm.ScalarMappable(
        norm=mcolors.Normalize(vmin=defl_clim[0] * 1000, vmax=defl_clim[1] * 1000),
        cmap="viridis",
    )
    cbar_defl = fig.colorbar(sm_defl, ax=ax_defl, shrink=0.82, pad=0.015,
                             orientation="vertical")
    cbar_defl.set_label("|w|  (mm)", fontsize=10)
    cbar_defl.ax.tick_params(labelsize=9)
    tip_mm = abs(w_profile[-1]) * 1000 if len(w_profile) else 0.0
    ax_defl.set_title(
        f"Deflection (warp x{warp_amp:.0f}, tip = {tip_mm:.0f} mm)",
        fontsize=11, loc="left", pad=6,
    )

    return fig


def render_hero(
    bench: Any,
    x_row: Tensor,
    conditions: Tensor,
    resolution: int = 256,
):
    """Final hero composite, `E1_VIZ_RESOLUTION` overrides the SDF mesh resolution."""
    resolution = int(os.environ.get("E1_VIZ_RESOLUTION", resolution))
    data = collect_hero_data(bench, x_row, conditions, resolution=resolution)
    return render_hero_from_data(data)


def _recover_deflection_profile(bench: Any, x_row: Tensor, conditions: Tensor):
    """Re-run the compute_stress pipeline for this single sample to grab w(y)."""
    from . import benchmark as bench_mod

    x_row = _as_single_row(x_row)
    dec = decode_x(x_row)
    conds = _as_single_cond(conditions) if conditions is not None else torch.zeros(1, 2)
    semi_span, _ = bench._planform(dec)
    y_stations = bench._sample_y_stations(x_row, semi_span)
    if bench._needs_program:
        from .loads import build_bwb_program
        bench._last_program = build_bwb_program(dec, device=x_row.device)
    loads = bench._compute_loads(bench._last_program, dec, conds, y_stations)
    props = bench._compute_structural(dec, y_stations)
    skin_t = bench._skin_thickness(dec)
    _, _, w = bench._compute_stress(
        loads, props, y_stations, dec,
        skin_thickness=skin_t,
        rho_material=bench_mod.RHO_MAT,
        e_modulus=bench_mod.E_MAT,
    )
    # `compute_stress` internally sorts stations ascending, rebuild matching y.
    y_sorted, _ = torch.sort(y_stations, dim=-1)
    w = w[0].detach().cpu().numpy()
    y = y_sorted[0].detach().cpu().numpy()
    return w, y


def _warp_mesh_by_deflection(mesh, y_profile, w_profile, amplification: float = 1.0):
    """Warp a PyVista mesh by a 1D `w(y)` profile along +z."""
    warped = mesh.copy()
    verts = np.asarray(warped.points, dtype=np.float64)
    y = verts[:, 1]
    y_norm = y / max(y_profile[-1], 1e-12)
    y_prof_norm = y_profile / max(y_profile[-1], 1e-12)
    w_on_mesh = np.interp(np.abs(y_norm), y_prof_norm, w_profile, left=0.0,
                          right=w_profile[-1])
    verts[:, 2] = verts[:, 2] + amplification * w_on_mesh
    warped.points = verts.astype(np.float32)
    return warped, w_on_mesh


def _sample_cp_on_mesh(bench: Any, x_row: Tensor, conditions: Tensor, mesh):
    """Query FiLM Cp on every vertex of the BWB mesh."""
    if mesh is None or mesh.n_points == 0:
        return np.zeros(0, dtype=np.float32)

    from . import atmosphere
    from .loads import FiLMLoads, _unnormalise_x_shape

    film_loads = bench._compute_loads
    if not isinstance(film_loads, FiLMLoads):
        return np.zeros(mesh.n_points, dtype=np.float32)

    dec = decode_x(x_row)
    conds = _as_single_cond(conditions) if conditions is not None else torch.zeros(1, 2)
    alt = conds[..., 0]
    V = conds[..., 1]
    Ma = atmosphere.mach(V, alt)
    Re = atmosphere.reynolds(alt, V, dec.L.squeeze(-1))
    alpha_deg = dec.alpha_cr.squeeze(-1) * (180.0 / torch.pi)
    shape_raw = _unnormalise_x_shape(dec.shape)
    cond_all = film_loads._build_film_cond(Re, Ma, alpha_deg, shape_raw)  # [1, 13]

    verts_np = np.asarray(mesh.points, dtype=np.float32)
    normals_np = np.asarray(mesh.point_normals, dtype=np.float32)
    if normals_np.shape[0] != verts_np.shape[0]:
        mesh_n = mesh.compute_normals(point_normals=True, cell_normals=False,
                                      auto_orient_normals=True, inplace=False)
        normals_np = np.asarray(mesh_n.point_normals, dtype=np.float32)

    verts_t = torch.from_numpy(verts_np).to(cond_all.device)
    norms_t = torch.from_numpy(normals_np).to(cond_all.device)
    with torch.no_grad():
        preds = film_loads._film_forward(verts_t, norms_t, cond_all[0])     # [N, 3]
    cp = preds[:, 0].detach().cpu().numpy().astype(np.float32)
    return cp
