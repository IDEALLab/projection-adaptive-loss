"""Generate structural-surrogate training data: one CSV row per (sample, thickness, station)."""

import argparse
import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from scipy.stats.qmc import LatinHypercube

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pal.benchmarks.engineering.e1_bwb.struct_surrogate.build_wingbox import (  # noqa: E402
    build_program,
)

BOUNDS_PATH = HERE / "bounds.yaml"

BWB_ORDER = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
STRUCT_ORDER = [
    "skin_t", "front_spar_x", "front_spar_w", "front_spar_dev",
    "rear_spar_x", "rear_spar_w",
    "bat_x", "bat_y", "bat_z", "bat_z_center",
    "rib_start_y", "rib_end_y", "rib_w", "center_rib_w",
]
THICKNESS_PARAMS = ["skin_t", "front_spar_w", "rear_spar_w"]

# One LHS scalar sets both rib_w and center_rib_w (5-50 mm at L=1 m).
RIB_THICKNESS_RANGE = (0.005, 0.050)
ALL_PARAMS = ["L"] + BWB_ORDER + STRUCT_ORDER

LOG_UNIFORM_PARAMS = {"L"}

# Sampled as a fraction of span (B1+B2+B3), then multiplied by span.
SPAN_SCALED_PARAMS = {"rib_end_y"}

PROP_COLS = [
    "A", "u_cg", "v_cg", "I_uu", "I_vv", "I_uv",
    "I_1", "I_2", "Q_u_max", "Q_v_max", "J",
]


def load_bounds():
    with open(BOUNDS_PATH) as f:
        raw = yaml.safe_load(f)
    bounds = {}   # name -> (lo, hi)
    fixed = {}    # name -> value
    for group in ("bwb_conditions", "structural"):
        for name, val in raw[group].items():
            if isinstance(val, list):
                bounds[name] = (float(val[0]), float(val[1]))
            else:
                fixed[name] = float(val)
    return bounds, fixed


def _bimodal_rear_spar_x(rng, lo, hi):
    if rng.random() < 0.5:
        x = rng.normal(-0.50, 0.07)
    else:
        x = rng.normal(-1.30, 0.10)
    return float(np.clip(x, lo, hi))


def sample_params(bounds, fixed, rng):
    params = {}
    for name, (lo, hi) in bounds.items():
        if name == "rear_spar_x":
            params[name] = _bimodal_rear_spar_x(rng, lo, hi)
        elif name in LOG_UNIFORM_PARAMS:
            params[name] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        else:
            params[name] = rng.uniform(lo, hi)
    span = params["B1"] + params["B2"] + params["B3"]
    for name in SPAN_SCALED_PARAMS:
        if name in params:
            params[name] = params[name] * span
    for name, val in fixed.items():
        params[name] = val
    return params


def sample_thickness_lhs(bounds, n_thickness, rng):
    """Latin Hypercube Sample across THICKNESS_PARAMS, returned as a list of dicts."""
    seed = int(rng.integers(0, 2**31 - 1))
    lhs = LatinHypercube(d=len(THICKNESS_PARAMS), seed=seed)
    u = lhs.random(n=n_thickness)  # (n_thickness, 3) in [0, 1]
    combos = []
    for i in range(n_thickness):
        combo = {}
        for j, name in enumerate(THICKNESS_PARAMS):
            lo, hi = bounds[name]
            combo[name] = float(lo + u[i, j] * (hi - lo))
        combos.append(combo)
    return combos


def sample_rib_thickness_lhs(n_rib, rng):
    """1-D LHS of rib thickness (same scalar sets rib_w and center_rib_w)."""
    seed = int(rng.integers(0, 2**31 - 1))
    lhs = LatinHypercube(d=1, seed=seed)
    u = lhs.random(n=n_rib).ravel()
    lo, hi = RIB_THICKNESS_RANGE
    return [float(lo + ui * (hi - lo)) for ui in u]


def validate_geometric(p):
    """Fast geometric checks (all params in unit coords)."""
    angle_deg = p["S3"] - p["front_spar_dev"]
    tan_a = np.tan(np.radians(angle_deg))
    reasons = []
    if p["rib_start_y"] >= p["rib_end_y"]:
        reasons.append("rib_start_y >= rib_end_y")
    rib_w = p.get("rib_w", 0.025)
    if p["bat_y"] >= p["rib_start_y"] - rib_w / 2:
        reasons.append("battery hole overlaps rib_start")
    rear_on = p["rear_spar_x"] > -0.8
    if rear_on and tan_a > 1e-6:
        dx = p["front_spar_x"] - p["rear_spar_x"]
        if dx > 0:
            y_cross = dx / tan_a
            if y_cross < p["rib_start_y"]:
                reasons.append("spars cross before rib_start")
        else:
            reasons.append("front spar behind rear spar")
    if reasons:
        return False, reasons
    return True, []


def validate_sdf(p, bwb_shape, device):
    """SDF checks (all params in unit coords)."""
    tan_a = np.tan(np.radians(p["S3"] - p["front_spar_dev"]))
    reasons = []
    spar_x_at_rib5 = p["front_spar_x"] - p["rib_end_y"] * tan_a
    pts = torch.tensor(
        [[[spar_x_at_rib5, p["rib_end_y"], 0.0]]],
        dtype=torch.float32, device=device,
    )
    with torch.no_grad():
        sdf = bwb_shape(pts)
    if sdf[0, 0].item() > 0.015:
        reasons.append("front spar at rib5 outside BWB")
    bat_y = p["bat_y"]
    bat_z = p["bat_z"]
    bat_z_c = p["bat_z_center"]
    margin = 0.005
    z_top = bat_z_c + bat_z / 2 + margin
    z_bot = bat_z_c - bat_z / 2 - margin
    y_stations = [i * 0.01 for i in range(1, 26)]
    top_pts, bot_pts = [], []
    for y in y_stations:
        if y > bat_y:
            break
        spar_x = p["front_spar_x"] - y * tan_a
        top_pts.append([spar_x, y, z_top])
        bot_pts.append([spar_x, y, z_bot])
    if top_pts:
        all_pts = torch.tensor(
            [top_pts + bot_pts], dtype=torch.float32, device=device,
        )
        with torch.no_grad():
            sdf_all = bwb_shape(all_pts)
        n = len(top_pts)
        sdf_top = sdf_all[0, :n]
        sdf_bot = sdf_all[0, n:]
        both_outside = (sdf_top > 0) & (sdf_bot > 0)
        if both_outside.any():
            idx = both_outside.nonzero()[0][0].item()
            reasons.append(f"spar severed at y={y_stations[idx]:.2f}")
        min_wall = 0.02
        best_side = torch.minimum(sdf_top, sdf_bot)
        too_thin = best_side > -min_wall
        if too_thin.any() and not both_outside.any():
            idx = too_thin.nonzero()[0][0].item()
            wall = -best_side[idx].item()
            reasons.append(
                f"spar too thin at battery corner y={y_stations[idx]:.2f} (wall={wall:.3f})"
            )
    if reasons:
        return False, reasons
    return True, []


def get_bwb_light(prog_base, params, device, bwb_source="analytical"):
    """Build a low-res program and return its `bwb` SDF for rejection validation."""
    prog = build_program(
        params, resolution=64, device=device, bwb_source=bwb_source,
    )
    prog._build_shapes()
    return prog._shapes["bwb"]


def bind_program(prog_base, params, device, bwb_source="analytical"):
    """Build a full-res CADProgram for structural_properties / meshing."""
    return build_program(
        params, resolution=128, device=device, bwb_source=bwb_source,
    )


def mesh_to_pv(mesh, batch_idx=0):
    import pyvista as pv
    verts = mesh.vertices[batch_idx].cpu().numpy()
    faces_np = mesh.faces[batch_idx].cpu().numpy()
    n_faces = faces_np.shape[0]
    pv_faces = np.column_stack([
        np.full(n_faces, 3, dtype=np.int64),
        faces_np,
    ]).ravel()
    return pv.PolyData(verts, pv_faces)


def _render_bwb_spars(prog, path, resolution, label=None, label_color="black"):
    """Mesh BWB (transparent) + spars (solid) and save a screenshot."""
    import pyvista as pv
    prog.config["bounds"]["resolution"] = resolution
    try:
        meshes = prog.mesh(backend="sparse-dmc")
    except Exception:
        meshes = prog.mesh(backend="skimage")
    bwb_pv = mesh_to_pv(meshes["bwb"])
    spars_pv = mesh_to_pv(meshes["spars"])
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    plotter.add_mesh(bwb_pv, color="lightgray", opacity=0.3)
    plotter.add_mesh(spars_pv, color="steelblue")
    if label:
        plotter.add_text(label, position="lower_left", font_size=9, color=label_color)
    plotter.add_axes()
    plotter.camera_position = "xz"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 25
    plotter.screenshot(str(path))
    plotter.close()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def sample_stations(rng, n_stations, y_min, y_max):
    """Stratified random stations (one draw per bin, sorted), y_min always first."""
    n_rand = n_stations - 1
    bin_edges = np.linspace(y_min, y_max, n_rand + 1)
    stations = [y_min]
    for i in range(n_rand):
        stations.append(rng.uniform(bin_edges[i], bin_edges[i + 1]))
    stations.sort()
    return stations


def compute_properties(prog, stations, base_res, levels, compute_torsion, torsion_levels=None):
    """Compute structural cross-section properties at spanwise stations, per-station lists."""
    props = prog.structural_properties(
        axis="y",
        stations=stations,
        name="structural",
        base_res=base_res,
        levels=levels,
        compute_torsion=compute_torsion,
        torsion_levels=torsion_levels,
    )
    result = {
        "A": props.A[0].tolist(),
        "u_cg": props.u_cg[0].tolist(),
        "v_cg": props.v_cg[0].tolist(),
        "I_uu": props.I_uu[0].tolist(),
        "I_vv": props.I_vv[0].tolist(),
        "I_uv": props.I_uv[0].tolist(),
        "I_1": props.I_1[0].tolist(),
        "I_2": props.I_2[0].tolist(),
        "Q_u_max": props.Q_u_max[0].tolist(),
        "Q_v_max": props.Q_v_max[0].tolist(),
        "J": props.J[0].tolist() if props.J is not None else [0.0] * len(stations),
    }
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Generate structural surrogate training data",
    )
    parser.add_argument("--n", type=int, default=10, help="Number of accepted samples")
    parser.add_argument(
        "--n-thickness", type=int, default=32,
        help="LHS thickness combos (skin_t, front_spar_w, rear_spar_w) per accepted geometry",
    )
    parser.add_argument(
        "--n-rib-thickness", type=int, default=5,
        help="LHS rib-thickness combos (rib_w=center_rib_w) per accepted geometry for rib-volume output",
    )
    parser.add_argument(
        "--rib-mesh-res", type=int, default=128,
        help="Mesh resolution override for rib-volume computation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output CSV path (default: data/chunk_<seed>.csv)",
    )
    parser.add_argument("--n-stations", type=int, default=30)
    parser.add_argument("--y-min", type=float, default=0.02)
    parser.add_argument("--y-max", type=float, default=0.95)
    parser.add_argument("--base-res", type=int, default=8)
    parser.add_argument("--levels", type=int, default=9, help="Quadtree levels (9 = 2048 effective)")
    parser.add_argument(
        "--torsion-levels", type=int, default=7,
        help="Levels to use for the Poisson-solve grid (torsion only). Must be "
             "<= --levels. 7 = 512 grid; dramatically faster than running the "
             "solver on the full 2048 grid at levels=9.",
    )
    parser.add_argument("--no-torsion", action="store_true", help="Skip J computation")
    parser.add_argument(
        "--viz", type=int, default=0,
        help="Mesh + screenshot the first N valid and N invalid samples",
    )
    parser.add_argument("--viz-res", type=int, default=256, help="Mesh resolution for viz")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu/mps/cuda)")
    parser.add_argument(
        "--bwb-source", type=str, default="neural",
        choices=["neural", "analytical"],
        help="BWB SDF source. neural=fast (default, ~6mm/m tip err), "
             "analytical=accurate but ~1000x slower per SDF query.",
    )
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = (
            "mps" if torch.backends.mps.is_available()
            else "cuda" if torch.cuda.is_available()
            else "cpu"
        )
    print(f"Device: {device}")
    print(f"Samples: {args.n}, seed: {args.seed}")
    print(f"Stations: {args.n_stations} stratified-random in [{args.y_min}, {args.y_max}]")
    print(f"Quadtree: base_res={args.base_res}, levels={args.levels} "
          f"(effective {args.base_res * 2**(args.levels-1)})")
    print(f"Torsion: {'OFF' if args.no_torsion else 'ON'}")
    if args.viz > 0:
        print(f"Viz: first {args.viz} valid + {args.viz} invalid (res={args.viz_res})")

    out_dir = HERE / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"chunk_{args.seed}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Params stored in absolute units, properties at unit scale.
    header = ["sample_id", "thickness_id"] + ALL_PARAMS + ["y"] + PROP_COLS
    with open(out_path, "w") as f:
        f.write(",".join(header) + "\n")

    # V_ribs is unit-scale volume, physical volume = V_ribs * L^3.
    ribs_out_path = out_path.with_name(out_path.stem + "_ribs" + out_path.suffix)
    ribs_header = [
        "sample_id", "rib_combo_id", "L",
    ] + BWB_ORDER + [
        "rib_start_y", "rib_end_y",  # rib geometry context
        "rib_thickness",              # sampled scalar, = rib_w = center_rib_w
        "V_ribs",                     # unit-scale volume (m^3 at L=1m)
    ]
    with open(ribs_out_path, "w") as f:
        f.write(",".join(ribs_header) + "\n")

    bounds, fixed = load_bounds()
    rng = np.random.default_rng(args.seed)
    prog_base = None

    accepted = 0
    total_drawn = 0
    reject_stats = {}
    n_invalid_viz = 0
    stats_path = out_dir / f"rejection_stats_{args.seed}.json"
    t_start = time.perf_counter()

    if args.viz > 0:
        viz_valid_dir = out_dir / f"viz_valid_{args.seed}"
        viz_invalid_dir = out_dir / f"viz_invalid_{args.seed}"
        viz_valid_dir.mkdir(parents=True, exist_ok=True)
        viz_invalid_dir.mkdir(parents=True, exist_ok=True)

    for i in range(args.n):
        for _attempt in range(500):
            params = sample_params(bounds, fixed, rng)
            total_drawn += 1

            ok, reasons = validate_geometric(params)
            if not ok:
                for r in reasons:
                    reject_stats[r] = reject_stats.get(r, 0) + 1
                if args.viz > 0 and n_invalid_viz < args.viz:
                    try:
                        prog_inv = bind_program(prog_base, params, device, bwb_source=args.bwb_source)
                        _render_bwb_spars(
                            prog_inv,
                            viz_invalid_dir / f"{n_invalid_viz:04d}.png",
                            args.viz_res,
                            label=f"L={params['L']:.2f}m\n" + "\n".join(reasons),
                            label_color="red",
                        )
                        n_invalid_viz += 1
                    except Exception:
                        pass
                continue

            bwb_shape = get_bwb_light(prog_base, params, device, bwb_source=args.bwb_source)
            ok, reasons = validate_sdf(params, bwb_shape, device)
            if ok:
                break
            for r in reasons:
                reject_stats[r] = reject_stats.get(r, 0) + 1
            if args.viz > 0 and n_invalid_viz < args.viz:
                try:
                    prog_inv = bind_program(prog_base, params, device, bwb_source=args.bwb_source)
                    _render_bwb_spars(
                        prog_inv,
                        viz_invalid_dir / f"{n_invalid_viz:04d}.png",
                        args.viz_res,
                        label=f"L={params['L']:.2f}m\n" + "\n".join(reasons),
                        label_color="red",
                    )
                    n_invalid_viz += 1
                except Exception:
                    pass
        else:
            print(f"[{i+1}/{args.n}] Could not find valid params after 500 attempts")
            continue

        accepted += 1
        L = params["L"]
        sample_id = f"s{args.seed}_{accepted - 1:05d}"

        # Per-geometry y_max: span = B1+B2+B3, margin avoids tip-collapse singularity
        span = params["B1"] + params["B2"] + params["B3"]
        y_max_geom = min(span - 0.02, args.y_max)

        stations = sample_stations(rng, args.n_stations, args.y_min, y_max_geom)

        if args.viz > 0 and accepted <= args.viz:
            try:
                prog_viz = bind_program(prog_base, params, device, bwb_source=args.bwb_source)
                _render_bwb_spars(
                    prog_viz,
                    viz_valid_dir / f"{accepted - 1:04d}.png",
                    args.viz_res,
                    label=f"L = {L:.2f} m",
                )
                print(f"  viz: {viz_valid_dir / f'{accepted - 1:04d}.png'}")
                del prog_viz
            except Exception as e:
                print(f"  viz FAILED: {e}")

        thickness_combos = sample_thickness_lhs(bounds, args.n_thickness, rng)
        t0 = time.perf_counter()
        n_combos_ok = 0
        n_combos_failed = 0
        for t_idx, combo in enumerate(thickness_combos):
            params_t = dict(params)
            params_t.update(combo)
            try:
                prog = bind_program(prog_base, params_t, device, bwb_source=args.bwb_source)
                result = compute_properties(
                    prog, stations, args.base_res, args.levels,
                    compute_torsion=not args.no_torsion,
                    torsion_levels=args.torsion_levels,
                )
            except Exception as e:
                n_combos_failed += 1
                print(
                    f"  [{sample_id} t{t_idx:02d}] compute FAILED: "
                    f"{type(e).__name__}: {e}"
                )
                gc.collect()
                continue

            param_vals = ",".join(f"{params_t[k]:.8g}" for k in ALL_PARAMS)
            with open(out_path, "a") as f:
                for s_idx, y in enumerate(stations):
                    prop_vals = ",".join(
                        f"{result[col][s_idx]:.10g}" for col in PROP_COLS
                    )
                    f.write(
                        f"{sample_id},{t_idx},{param_vals},"
                        f"{y:.6f},{prop_vals}\n"
                    )
            n_combos_ok += 1

        dt = time.perf_counter() - t0

        rib_thicknesses = sample_rib_thickness_lhs(args.n_rib_thickness, rng)
        t0_ribs = time.perf_counter()
        n_rib_ok = 0
        for r_idx, rib_t in enumerate(rib_thicknesses):
            params_r = dict(params)
            params_r["rib_w"] = rib_t
            params_r["center_rib_w"] = rib_t
            try:
                prog_r = bind_program(prog_base, params_r, device, bwb_source=args.bwb_source)
                prog_r._build_shapes()
                orig_res = prog_r.config["bounds"]["resolution"]
                prog_r.config["bounds"]["resolution"] = args.rib_mesh_res
                ribs_shape = prog_r._shapes["ribs"]
                ribs_mesh = ribs_shape.get_mesh(
                    prog_r.xyz_min, prog_r.xyz_max, prog_r.resolution,
                    "sparse-dmc", device=prog_r._device,
                )
                prog_r.config["bounds"]["resolution"] = orig_res
                V = prog_r.compute_volume(ribs_mesh)
                v_val = float(V.item() if torch.is_tensor(V) else V)
            except Exception as e:
                print(
                    f"  [{sample_id} rib{r_idx}] volume FAILED: "
                    f"{type(e).__name__}: {e}"
                )
                gc.collect()
                continue

            bwb_vals = ",".join(f"{params[k]:.8g}" for k in BWB_ORDER)
            with open(ribs_out_path, "a") as f:
                f.write(
                    f"{sample_id},{r_idx},{params['L']:.8g},{bwb_vals},"
                    f"{params['rib_start_y']:.8g},{params['rib_end_y']:.8g},"
                    f"{rib_t:.8g},{v_val:.10g}\n"
                )
            n_rib_ok += 1
        dt_ribs = time.perf_counter() - t0_ribs

        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()

        elapsed = time.perf_counter() - t_start
        rate = accepted / elapsed if elapsed > 0 else 0
        print(
            f"[{accepted}/{args.n}] {sample_id} L={L:.2f}m  "
            f"struct {n_combos_ok}/{args.n_thickness} ok ({n_combos_failed} failed) {dt:.1f}s  "
            f"ribs {n_rib_ok}/{args.n_rib_thickness} ok {dt_ribs:.1f}s  "
            f"({total_drawn} drawn, {elapsed:.0f}s, {rate:.2f} geom/s)"
        )

    elapsed = time.perf_counter() - t_start
    with open(stats_path, "w") as f:
        json.dump({
            "seed": args.seed,
            "total_drawn": total_drawn,
            "accepted": accepted,
            "elapsed_s": elapsed,
            "reasons": reject_stats,
        }, f, indent=2)

    print(f"\nDone. {accepted}/{total_drawn} accepted "
          f"({100*accepted/max(total_drawn,1):.1f}%) in {elapsed:.0f}s")
    print(f"  Output: {out_path}")
    print(f"  Stats:  {stats_path}")
    print(f"  Max rows: {accepted * args.n_thickness * args.n_stations} "
          f"({accepted} geom x {args.n_thickness} thicknesses x "
          f"{args.n_stations} stations; actual = this minus failed combos)")
    if args.viz > 0:
        print(f"  Valid viz:   {viz_valid_dir}/ ({min(accepted, args.viz)} images)")
        print(f"  Invalid viz: {viz_invalid_dir}/ ({n_invalid_viz} images)")


if __name__ == "__main__":
    main()
