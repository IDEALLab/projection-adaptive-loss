"""Sample random wingbox configurations, render screenshots, and save a table.

For each sample:
  1. Draw params uniformly from bounds.yaml
  2. Mesh BWB + spars
  3. Save a screenshot to samples/<id>.png
  4. Append params + image path to samples/table.csv

Usage:
    python -m pal.benchmarks.engineering.e1_bwb.reproduce.structural.sample --n 20
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from pal.benchmarks.engineering.e1_bwb.struct_surrogate.build_wingbox import (  # noqa: E402
    build_program,
)

BOUNDS_PATH = HERE / "bounds.yaml"


LOG_UNIFORM_PARAMS = {"L"}                          # sampled log-uniform


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
    """Mixture of two Gaussians, soft on/off for rear spar."""
    if rng.random() < 0.5:
        x = rng.normal(-0.50, 0.07)   # "on", inside BWB
    else:
        x = rng.normal(-1.30, 0.10)   # "off", past TE, clipped away
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
    # Include fixed params in the dict (needed for tensor building)
    for name, val in fixed.items():
        params[name] = val
    return params


def validate_geometric(p):
    """Fast geometric checks (no SDF needed). Returns (ok, reasons_list).

    All params are in unit coords (ratios of C1). Physical = value * L.
    """
    angle_deg = p["S3"] - p["front_spar_dev"]
    tan_a = np.tan(np.radians(angle_deg))
    reasons = []

    # rib_start_y must be less than rib_end_y (guaranteed by bounds, but belt-and-suspenders)
    if p["rib_start_y"] >= p["rib_end_y"]:
        reasons.append("rib_start_y >= rib_end_y")

    # Battery hole must end before first outer rib
    rib_w = p.get("rib_w", 0.025)
    if p["bat_y"] >= p["rib_start_y"] - rib_w / 2:
        reasons.append("battery hole overlaps rib_start")

    # Spars must not cross before rib_start_y (only when rear spar is "on")
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
    """SDF-based checks against actual BWB shape. Returns (ok, reasons_list)."""
    tan_a = np.tan(np.radians(p["S3"] - p["front_spar_dev"]))
    reasons = []

    # 1) Front spar must be inside BWB at outermost rib
    #    Spar center must be at least 0.015 inside BWB surface (negative SDF)
    spar_x_at_rib5 = p["front_spar_x"] - p["rib_end_y"] * tan_a
    pts = torch.tensor([[[spar_x_at_rib5, p["rib_end_y"], 0.0]]], dtype=torch.float32, device=device)
    with torch.no_grad():
        sdf = bwb_shape(pts)
    if sdf[0, 0].item() > 0.015:
        reasons.append("front spar at rib5 outside BWB")

    # 2) Spar contiguity through battery bay (y=0 to bat_y):
    #    At each station, BWB must have material on at least ONE side
    #    (top OR bottom) of the battery hole. Both outside = fully severed.
    bat_y = p["bat_y"]
    bat_z = p["bat_z"]
    bat_z_c = p["bat_z_center"]
    margin = 0.005
    z_top = bat_z_c + bat_z / 2 + margin
    z_bot = bat_z_c - bat_z / 2 - margin

    y_stations = [i * 0.01 for i in range(1, 26)]
    top_pts = []
    bot_pts = []
    for y in y_stations:
        if y > bat_y:
            break
        spar_x = p["front_spar_x"] - y * tan_a
        top_pts.append([spar_x, y, z_top])
        bot_pts.append([spar_x, y, z_bot])

    if top_pts:
        all_pts = torch.tensor([top_pts + bot_pts], dtype=torch.float32, device=device)
        with torch.no_grad():
            sdf_all = bwb_shape(all_pts)  # [1, 2*N]
        n = len(top_pts)
        sdf_top = sdf_all[0, :n]
        sdf_bot = sdf_all[0, n:]
        # Severed: both top and bottom are outside BWB
        both_outside = (sdf_top > 0) & (sdf_bot > 0)
        if both_outside.any():
            idx = both_outside.nonzero()[0][0].item()
            reasons.append(f"spar severed at y={y_stations[idx]:.2f}")

        # Too thin: best side has < min_wall remaining between hole and BWB surface
        min_wall = 0.02
        best_side = torch.minimum(sdf_top, sdf_bot)  # most negative = deepest inside
        too_thin = best_side > -min_wall  # not deep enough inside
        if too_thin.any() and not both_outside.any():
            idx = too_thin.nonzero()[0][0].item()
            wall = -best_side[idx].item()
            reasons.append(f"spar too thin at battery corner y={y_stations[idx]:.2f} (wall={wall:.3f})")

    if reasons:
        return False, reasons
    return True, []


BWB_ORDER = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
STRUCT_ORDER = [
    "skin_t", "front_spar_x", "front_spar_w", "front_spar_dev",
    "rear_spar_x", "rear_spar_w",
    "bat_x", "bat_y", "bat_z", "bat_z_center",
    "rib_start_y", "rib_end_y", "rib_w", "center_rib_w",
]


def build_sample_program(params, device, resolution=128):
    """Build a full CADProgram for meshing / SDF access.

    The analytical BWB is rebuilt per sample, each (B1..S3) triple yields
    a different set of Bezier control nets, so there is no shared program
    to rebind.
    """
    prog = build_program(params, resolution=resolution, device=device)
    prog._build_shapes()
    return prog


def build_bwb_for_validation(params, device):
    """Build just the BWB shape (low-res program) for SDF rejection checks.

    The analytical builder is fast enough that caching is not worthwhile,
    and each draw needs fresh Bezier control nets.
    """
    prog = build_program(params, resolution=64, device=device)
    prog._build_shapes()
    return prog._shapes["bwb"]


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


def render_screenshot(meshes, path):
    import pyvista as pv
    plotter = pv.Plotter(off_screen=True, window_size=(1200, 900))
    bwb_pv = mesh_to_pv(meshes["bwb"])
    spars_pv = mesh_to_pv(meshes["spars"])
    plotter.add_mesh(bwb_pv, color="lightgray", opacity=0.3)
    plotter.add_mesh(spars_pv, color="steelblue")
    plotter.add_axes()
    plotter.camera_position = "xz"
    plotter.camera.azimuth = 30
    plotter.camera.elevation = 25
    plotter.screenshot(str(path))
    plotter.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=20, help="Number of samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--res", type=int, default=128)
    args = parser.parse_args()

    device = (
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Device: {device}, samples: {args.n}, resolution: {args.res}")

    bounds, fixed = load_bounds()
    rng = np.random.default_rng(args.seed)

    # Output directory
    out_dir = HERE / "samples" / "valid"
    out_dir.mkdir(parents=True, exist_ok=True)

    # CSV, append if exists, else create with header
    # Include all params (sampled + fixed) for reproducibility
    param_names = list(bounds.keys()) + list(fixed.keys())
    csv_path = out_dir / "table.csv"
    if csv_path.exists():
        # Find next ID from existing rows
        with open(csv_path) as f:
            lines = f.readlines()
        start_id = len(lines) - 1  # subtract header
        print(f"Appending to existing table (starting at id {start_id:04d})")
    else:
        start_id = 0
        with open(csv_path, "w") as f:
            f.write("id," + ",".join(param_names) + ",image\n")

    accepted = 0
    total_drawn = 0
    reject_stats = {}
    stats_path = out_dir.parent / "rejection_stats.json"

    def flush_stats():
        with open(stats_path, "w") as f:
            json.dump({"total_drawn": total_drawn, "accepted": accepted, "reasons": reject_stats}, f, indent=2)

    for _i in range(args.n):
        # Two-stage rejection sampling
        for _attempt in range(500):
            params = sample_params(bounds, fixed, rng)
            total_drawn += 1
            # First pass: fast geometric checks
            ok, reasons = validate_geometric(params)
            if not ok:
                for r in reasons:
                    reject_stats[r] = reject_stats.get(r, 0) + 1
                if total_drawn % 50 == 0:
                    flush_stats()
                    print(f"  ... {total_drawn} drawn, {accepted} accepted so far")
                continue
            # Second pass: SDF-based checks against actual BWB shape
            bwb_shape = build_bwb_for_validation(params, device)
            ok, reasons = validate_sdf(params, bwb_shape, device)
            if ok:
                break
            for r in reasons:
                reject_stats[r] = reject_stats.get(r, 0) + 1
            if total_drawn % 50 == 0:
                flush_stats()
                print(f"  ... {total_drawn} drawn, {accepted} accepted so far")
        else:
            print(f"\n[{accepted+1}/{args.n}] Could not find valid params after 500 attempts")
            flush_stats()
            if accepted == 0 and total_drawn >= 1000:
                print("\nNo valid samples found in first 1000 draws. Exiting.")
                print(f"  rejection stats: {stats_path}")
                for r, n in sorted(reject_stats.items(), key=lambda x: -x[1]):
                    print(f"    {n:5d}  {r}")
                return
            continue

        idx = start_id + accepted
        accepted += 1
        L = params["L"]
        print(f"\n[{accepted}/{args.n}] Sample {idx:04d} (L={L:.2f} m, {total_drawn} drawn total)")
        try:
            prog = build_sample_program(params, device, resolution=args.res)
            meshes = prog.mesh(backend="skimage")
            img_name = f"{idx:04d}.png"
            render_screenshot(meshes, out_dir / img_name)
            print(f"  saved {img_name}")
        except Exception as e:
            print(f"  FAILED: {e}")
            img_name = "FAILED"

        # Append to CSV (store original absolute params, not unit-converted)
        vals = ",".join(f"{params[k]:.6f}" for k in param_names)
        with open(csv_path, "a") as f:
            f.write(f"{idx:04d},{vals},{img_name}\n")

    flush_stats()
    print(f"\nDone. Results in {out_dir}/")
    print(f"  table: {csv_path}")
    print(f"  rejection stats: {stats_path}")
    print(f"  {accepted}/{total_drawn} accepted ({100*accepted/max(total_drawn,1):.1f}%)")
    for r, n in sorted(reject_stats.items(), key=lambda x: -x[1]):
        print(f"    {n:5d}  {r}")


if __name__ == "__main__":
    main()
