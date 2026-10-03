"""Precompute SDF training samples from BlendedNet VTK meshes."""

import argparse
import json
import re
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pyvista as pv
import vtk


def parse_args():
    p = argparse.ArgumentParser(description="Precompute SDF samples from BlendedNet VTK meshes")

    p.add_argument("--data-dir", type=str,
                   default="blended_dataset",
                   help="Path to unzipped BlendedNet dataset")
    p.add_argument("--out-dir", type=str,
                   default="blendednet/data",
                   help="Output directory for precomputed data")

    p.add_argument("--n-near", type=int, default=12000,
                   help="Near-surface samples (Gaussian offset, sigma-near)")
    p.add_argument("--n-mid", type=int, default=8000,
                   help="Mid-field samples (Gaussian offset, sigma-mid)")
    p.add_argument("--n-far", type=int, default=10000,
                   help="Far-field samples (uniform in padded bbox)")
    p.add_argument("--n-surface", type=int, default=0,
                   help="On-surface samples (SDF=0, no VTK query needed)")
    p.add_argument("--sigma-near", type=float, default=0.01,
                   help="Std dev for near-surface Gaussian offset")
    p.add_argument("--sigma-mid", type=float, default=0.1,
                   help="Std dev for mid-field Gaussian offset")
    p.add_argument("--bbox-pad", type=float, default=10.0,
                   help="Padding around bounding box for far-field uniform sampling")
    p.add_argument("--n-farfield", type=int, default=0,
                   help="Extra far-field samples (Gaussian, sigma-farfield from centroid)")
    p.add_argument("--sigma-farfield", type=float, default=8.0,
                   help="Std dev (in bbox-extent units) for extra far-field Gaussian")
    p.add_argument("--n-curvature", type=int, default=0,
                   help="Curvature-weighted near-surface samples")
    p.add_argument("--sigma-curvature", type=float, default=0.005,
                   help="Std dev for curvature-band normal offset")
    p.add_argument("--n-tip", type=int, default=0,
                   help="Wing-tip region near-surface samples")
    p.add_argument("--sigma-tip", type=float, default=0.003,
                   help="Std dev for tip-band normal offset")
    p.add_argument("--n-edge", type=int, default=0,
                   help="Leading/trailing edge near-surface samples")
    p.add_argument("--sigma-edge", type=float, default=0.001,
                   help="Std dev for edge normal offset (very tight)")

    p.add_argument("--start", type=int, default=0,
                   help="Start geometry index (inclusive)")
    p.add_argument("--end", type=int, default=-1,
                   help="End geometry index (exclusive), -1 for all")
    p.add_argument("--split", type=str, default="train",
                   choices=["train", "test", "both"],
                   help="Which split to process")

    p.add_argument("--workers", type=int, default=8,
                   help="Number of parallel workers")

    p.add_argument("--skip-existing", action="store_true", default=True,
                   help="Skip geometries that already have .npz files")
    p.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    p.add_argument("--consolidate", action="store_true", default=True,
                   help="Consolidate per-geometry .npz into single .pt after precompute")
    p.add_argument("--no-consolidate", dest="consolidate", action="store_false")

    return p.parse_args()


def build_geom_index(data_dir, split="train"):
    """Map geom_id -> {vtk_path, params}, using the first VTK case per geometry."""
    split_dir = Path(data_dir) / split

    params_text = (split_dir / "geom_params.ini").read_text()
    geom_params = {}
    current_geom = None
    for line in params_text.strip().split("\n"):
        line = line.strip()
        m = re.match(r"\[geom_(\d+)\]", line)
        if m:
            current_geom = f"geom_{int(m.group(1)):03d}"
            geom_params[current_geom] = {}
        elif "=" in line and current_geom:
            key, val = line.split("=")
            geom_params[current_geom][key.strip()] = float(val.strip())

    case_lines = (split_dir / "case_data.dat").read_text().strip().split("\n")
    geom_to_vtk = {}
    for line in case_lines[1:]:  # skip header
        parts = line.split("\t")
        case_name = parts[0].strip()
        geom_name = parts[1].strip()
        if geom_name not in geom_to_vtk:
            vtk_path = split_dir / "vtk" / f"{case_name}.vtk"
            if vtk_path.exists():
                geom_to_vtk[geom_name] = str(vtk_path)

    index = {}
    param_keys = ["B1", "B2", "B3", "C1", "C2", "C3", "C4", "S1", "S2", "S3"]
    for geom_id, params in geom_params.items():
        if geom_id in geom_to_vtk:
            param_vec = [params[k] / params["C1"] for k in param_keys if k != "C1"]
            # S1..S3 are sweep angles in degrees, so only lengths are divided by C1.
            param_vec = []
            for k in param_keys:
                if k == "C1":
                    continue
                v = params[k]
                if k.startswith("S"):
                    param_vec.append(v)  # sweep angles in degrees
                else:
                    param_vec.append(v / params["C1"])  # normalize lengths by C1
            index[geom_id] = {
                "vtk_path": geom_to_vtk[geom_id],
                "params": param_vec,
                "param_names": [k for k in param_keys if k != "C1"],
            }

    return index


def sample_surface_points(surf, n):
    """Sample random points on the mesh surface (area-weighted triangles + barycentric)."""
    faces = np.array(surf.faces).reshape(-1, 4)[:, 1:]  # [T, 3]
    pts = np.array(surf.points)  # [V, 3]

    v0 = pts[faces[:, 0]]
    v1 = pts[faces[:, 1]]
    v2 = pts[faces[:, 2]]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * areas[:, None] + 1e-30)  # unit face normals

    probs = areas / areas.sum()
    tri_idx = np.random.choice(len(areas), size=n, p=probs)

    r1 = np.random.rand(n)
    r2 = np.random.rand(n)
    sqrt_r1 = np.sqrt(r1)
    bary_a = 1 - sqrt_r1
    bary_b = sqrt_r1 * (1 - r2)
    bary_c = sqrt_r1 * r2

    surface_pts = (bary_a[:, None] * v0[tri_idx] +
                   bary_b[:, None] * v1[tri_idx] +
                   bary_c[:, None] * v2[tri_idx])

    surface_normals = normals[tri_idx]

    return surface_pts.astype(np.float32), surface_normals.astype(np.float32)


def compute_triangle_curvature(surf):
    """Per-triangle absolute mean curvature as sampling weights.

    Args:
        surf: pyvista PolyData (triangulated)

    Returns:
        weights: [T] array of abs(mean_curvature) per triangle
    """
    vertex_curv = np.abs(surf.curvature("mean"))  # [V]
    faces = np.array(surf.faces).reshape(-1, 4)[:, 1:]  # [T, 3]
    tri_curv = vertex_curv[faces].mean(axis=1)  # avg vertex curvatures per tri
    # floor at small value so flat regions still get some samples
    tri_curv = np.maximum(tri_curv, tri_curv.max() * 0.01)
    return tri_curv


def sample_curvature_weighted(surf, n, sigma):
    """Sample near-surface points weighted by curvature magnitude.

    Args:
        surf: pyvista PolyData (triangulated)
        n: number of points
        sigma: normal offset std dev

    Returns:
        (points [n,3], normals [n,3])
    """
    faces = np.array(surf.faces).reshape(-1, 4)[:, 1:]
    pts = np.array(surf.points)

    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * areas[:, None] + 1e-30)

    curv = compute_triangle_curvature(surf)
    weights = areas * curv
    probs = weights / weights.sum()

    tri_idx = np.random.choice(len(probs), size=n, p=probs)

    r1, r2 = np.random.rand(n), np.random.rand(n)
    s = np.sqrt(r1)
    ba, bb, bc = 1 - s, s * (1 - r2), s * r2
    surface_pts = ba[:, None] * v0[tri_idx] + bb[:, None] * v1[tri_idx] + bc[:, None] * v2[tri_idx]

    offsets = np.random.randn(n, 1).astype(np.float32) * sigma
    return (surface_pts + offsets * normals[tri_idx]).astype(np.float32), normals[tri_idx].astype(np.float32)


def sample_wingtip_points(surf, n, sigma):
    """Sample near-surface points in wing-tip regions (high |y|).

    Args:
        surf: pyvista PolyData (triangulated)
        n: number of points
        sigma: normal offset std dev

    Returns:
        (points [n,3], normals [n,3])
    """
    faces = np.array(surf.faces).reshape(-1, 4)[:, 1:]
    pts = np.array(surf.points)

    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * areas[:, None] + 1e-30)

    centroids = (v0 + v1 + v2) / 3.0
    abs_y = np.abs(centroids[:, 1])

    # tip = triangles with |y| > 80th percentile
    threshold = np.percentile(abs_y, 80)
    tip_mask = abs_y >= threshold

    tip_areas = areas[tip_mask]
    if tip_areas.sum() < 1e-30:
        # fallback: area-weighted from whole surface
        tip_mask = np.ones(len(areas), dtype=bool)
        tip_areas = areas

    tip_indices = np.where(tip_mask)[0]
    probs = tip_areas / tip_areas.sum()
    sel = np.random.choice(len(tip_indices), size=n, p=probs)
    tri_idx = tip_indices[sel]

    r1, r2 = np.random.rand(n), np.random.rand(n)
    s = np.sqrt(r1)
    ba, bb, bc = 1 - s, s * (1 - r2), s * r2
    surface_pts = ba[:, None] * v0[tri_idx] + bb[:, None] * v1[tri_idx] + bc[:, None] * v2[tri_idx]

    offsets = np.random.randn(n, 1).astype(np.float32) * sigma
    return (surface_pts + offsets * normals[tri_idx]).astype(np.float32), normals[tri_idx].astype(np.float32)


def sample_edge_points(surf, n, sigma):
    """Sample near-surface points at leading/trailing edges.

    Args:
        surf: pyvista PolyData (triangulated)
        n: number of points
        sigma: normal offset std dev (recommend 0.001-0.002)

    Returns:
        (points [n,3], normals [n,3])
    """
    faces = np.array(surf.faces).reshape(-1, 4)[:, 1:]
    pts = np.array(surf.points)

    v0, v1, v2 = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    normals = cross / (2 * areas[:, None] + 1e-30)

    curv = compute_triangle_curvature(surf)

    centroids = (v0 + v1 + v2) / 3.0

    # Edge score: high curvature and extreme chordwise position.
    x_vals = centroids[:, 0]
    x_min, x_max = x_vals.min(), x_vals.max()
    x_norm = (x_vals - x_min) / (x_max - x_min + 1e-8)

    edge_score = np.maximum(1.0 - x_norm, x_norm)  # high at both edges
    edge_score = edge_score ** 2

    weights = areas * curv * edge_score
    probs = weights / weights.sum()

    tri_idx = np.random.choice(len(probs), size=n, p=probs)

    r1, r2 = np.random.rand(n), np.random.rand(n)
    s = np.sqrt(r1)
    ba, bb, bc = 1 - s, s * (1 - r2), s * r2
    surface_pts = ba[:, None] * v0[tri_idx] + bb[:, None] * v1[tri_idx] + bc[:, None] * v2[tri_idx]

    offsets = np.random.randn(n, 1).astype(np.float32) * sigma
    return (surface_pts + offsets * normals[tri_idx]).astype(np.float32), normals[tri_idx].astype(np.float32)


def sample_query_points(surf, n_near, n_mid, n_far, n_surface,
                        sigma_near, sigma_mid, bbox_pad,
                        n_farfield=0, sigma_farfield=3.0,
                        n_curvature=0, sigma_curvature=0.005,
                        n_tip=0, sigma_tip=0.003,
                        n_edge=0, sigma_edge=0.001):
    """Sample query points from all bands (near, mid, far, surface, curvature, tip, edge)."""
    pts = np.array(surf.points)
    bbox_min = pts.min(axis=0) - bbox_pad
    bbox_max = pts.max(axis=0) + bbox_pad

    all_points = []

    if n_near > 0:
        surf_pts, surf_normals = sample_surface_points(surf, n_near)
        offsets = np.random.randn(n_near, 1).astype(np.float32) * sigma_near
        all_points.append(surf_pts + offsets * surf_normals)

    if n_mid > 0:
        surf_pts, _ = sample_surface_points(surf, n_mid)
        offsets = np.random.randn(n_mid, 3).astype(np.float32) * sigma_mid
        all_points.append(surf_pts + offsets)

    if n_far > 0:
        far_pts = np.random.uniform(bbox_min, bbox_max, size=(n_far, 3)).astype(np.float32)
        all_points.append(far_pts)

    if n_farfield > 0:
        centroid = pts.mean(axis=0)
        extent = (bbox_max - bbox_min).max()  # largest bbox dimension
        sigma = sigma_farfield * extent
        ff_pts = (centroid + np.random.randn(n_farfield, 3).astype(np.float32) * sigma)
        all_points.append(ff_pts)

    if n_surface > 0:
        surf_pts, _ = sample_surface_points(surf, n_surface)
        all_points.append(surf_pts)

    if n_curvature > 0:
        curv_pts, _ = sample_curvature_weighted(surf, n_curvature, sigma_curvature)
        all_points.append(curv_pts)

    if n_tip > 0:
        tip_pts, _ = sample_wingtip_points(surf, n_tip, sigma_tip)
        all_points.append(tip_pts)

    if n_edge > 0:
        edge_pts, _ = sample_edge_points(surf, n_edge, sigma_edge)
        all_points.append(edge_pts)

    query = np.concatenate(all_points, axis=0)
    return query, n_surface


def compute_vtk_sdf(query_points, surf):
    """Compute SDF at query points using VTK's vtkImplicitPolyDataDistance."""
    imp_dist = vtk.vtkImplicitPolyDataDistance()
    imp_dist.SetInput(surf)

    sdf = np.empty(len(query_points), dtype=np.float32)
    for i, p in enumerate(query_points):
        sdf[i] = imp_dist.EvaluateFunction(p.tolist())

    return sdf


def process_geometry(args):
    """Process a single geometry: load VTK, sample points, compute SDF, save."""
    (geom_id, info, out_dir, n_near, n_mid, n_far, n_surface,
     sigma_near, sigma_mid, bbox_pad, n_farfield, sigma_farfield,
     n_curvature, sigma_curvature, n_tip, sigma_tip,
     n_edge, sigma_edge,
     skip_existing) = args

    out_path = Path(out_dir) / "sdf_samples" / f"{geom_id}.npz"
    if skip_existing and out_path.exists():
        return geom_id, "skipped", 0.0

    t0 = time.time()

    surf = pv.read(info["vtk_path"])
    if not isinstance(surf, pv.PolyData):
        surf = surf.extract_surface()
    surf = surf.triangulate()

    query, n_on_surface = sample_query_points(
        surf, n_near, n_mid, n_far, n_surface, sigma_near, sigma_mid, bbox_pad,
        n_farfield, sigma_farfield,
        n_curvature, sigma_curvature, n_tip, sigma_tip,
        n_edge, sigma_edge,
    )

    n_to_query = len(query) - n_on_surface
    sdf = np.empty(len(query), dtype=np.float32)

    if n_to_query > 0:
        sdf[:n_to_query] = compute_vtk_sdf(query[:n_to_query], surf)
    if n_on_surface > 0:
        sdf[n_to_query:] = 0.0

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_path,
        points=query,
        sdf=sdf,
        params=np.array(info["params"], dtype=np.float32),
    )

    elapsed = time.time() - t0
    return geom_id, "done", elapsed


def consolidate(out_dir, geom_index):
    """Merge per-geometry .npz files into a single .pt file."""
    import torch

    sample_dir = Path(out_dir) / "sdf_samples"
    npz_files = sorted(sample_dir.glob("geom_*.npz"))

    if not npz_files:
        print("No .npz files found to consolidate.")
        return

    first = np.load(npz_files[0])
    n_pts = len(first["points"])
    n_params = len(first["params"])
    n_geoms = len(npz_files)

    print(f"Consolidating {n_geoms} geometries x {n_pts} points...")

    all_points = np.empty((n_geoms, n_pts, 3), dtype=np.float32)
    all_sdf = np.empty((n_geoms, n_pts), dtype=np.float32)
    all_params = np.empty((n_geoms, n_params), dtype=np.float32)
    geom_ids = []

    for i, f in enumerate(npz_files):
        data = np.load(f)
        all_points[i] = data["points"]
        all_sdf[i] = data["sdf"]
        all_params[i] = data["params"]
        geom_ids.append(f.stem)

    out_path = Path(out_dir) / "train_dataset.pt"
    torch.save({
        "points": torch.from_numpy(all_points),
        "sdf": torch.from_numpy(all_sdf),
        "params": torch.from_numpy(all_params),
        "geom_ids": geom_ids,
        "param_names": geom_index[next(iter(geom_index))]["param_names"],
    }, out_path)

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"Saved: {out_path} ({size_mb:.1f} MB)")
    print(f"  points: [{n_geoms}, {n_pts}, 3]")
    print(f"  sdf:    [{n_geoms}, {n_pts}]")
    print(f"  params: [{n_geoms}, {n_params}]")


def main():
    args = parse_args()

    n_total = (args.n_near + args.n_mid + args.n_far + args.n_farfield
               + args.n_surface + args.n_curvature + args.n_tip + args.n_edge)
    print("BlendedNet SDF Precomputation")
    print(f"  data_dir:    {args.data_dir}")
    print(f"  out_dir:     {args.out_dir}")
    print(f"  n_near:      {args.n_near} (sigma={args.sigma_near})")
    print(f"  n_mid:       {args.n_mid} (sigma={args.sigma_mid})")
    print(f"  n_far:       {args.n_far} (bbox_pad={args.bbox_pad})")
    print(f"  n_farfield:  {args.n_farfield} (sigma={args.sigma_farfield})")
    print(f"  n_curvature: {args.n_curvature} (sigma={args.sigma_curvature})")
    print(f"  n_tip:       {args.n_tip} (sigma={args.sigma_tip})")
    print(f"  n_edge:      {args.n_edge} (sigma={args.sigma_edge})")
    print(f"  n_surface:   {args.n_surface}")
    print(f"  total/geom:  {n_total}")
    print(f"  workers:     {args.workers}")
    print()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    geom_index = {}
    for split in splits:
        idx = build_geom_index(args.data_dir, split)
        print(f"  {split}: {len(idx)} unique geometries")
        geom_index.update(idx)

    geom_ids = sorted(geom_index.keys())
    end = args.end if args.end > 0 else len(geom_ids)
    geom_ids = geom_ids[args.start:end]
    print(f"  Processing: {len(geom_ids)} geometries [{args.start}:{end}]")
    print()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    config = vars(args)
    config["n_geometries"] = len(geom_ids)
    with open(Path(args.out_dir) / "precompute_config.json", "w") as f:
        json.dump(config, f, indent=2)
    with open(Path(args.out_dir) / "geom_index.json", "w") as f:
        json.dump({k: {**v, "param_names": v["param_names"]} for k, v in geom_index.items()}, f, indent=2)

    work = [
        (gid, geom_index[gid], args.out_dir,
         args.n_near, args.n_mid, args.n_far, args.n_surface,
         args.sigma_near, args.sigma_mid, args.bbox_pad,
         args.n_farfield, args.sigma_farfield,
         args.n_curvature, args.sigma_curvature,
         args.n_tip, args.sigma_tip,
         args.n_edge, args.sigma_edge,
         args.skip_existing)
        for gid in geom_ids
    ]

    t_start = time.time()
    n_done = 0
    n_skipped = 0
    n_total_geoms = len(work)

    if args.workers <= 1:
        for w in work:
            gid, status, elapsed = process_geometry(w)
            if status == "skipped":
                n_skipped += 1
            else:
                n_done += 1
            print(f"  [{n_done + n_skipped}/{n_total_geoms}] {gid}: {status} ({elapsed:.1f}s)")
    else:
        with Pool(args.workers) as pool:
            for _gid, status, _elapsed in pool.imap_unordered(process_geometry, work):
                if status == "skipped":
                    n_skipped += 1
                else:
                    n_done += 1
                done = n_done + n_skipped
                if done % 10 == 0 or done == n_total_geoms:
                    eta = (time.time() - t_start) / done * (n_total_geoms - done)
                    print(f"  [{done}/{n_total_geoms}] eta {eta:.0f}s")

    t_total = time.time() - t_start
    print(f"\nDone: {n_done} computed, {n_skipped} skipped in {t_total:.1f}s")
    if n_done > 0:
        print(f"  Avg: {t_total/n_done:.1f}s per geometry")

    if args.consolidate:
        print()
        consolidate(args.out_dir, geom_index)


if __name__ == "__main__":
    main()
