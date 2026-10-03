"""Offline training loop for the conditional BWB SDF network."""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import wandb
from pal.benchmarks.engineering.e1_bwb.bwb_sdf.sdf_net import BWBSDFNet

from .config import DEFAULT_CONFIG


def load_dataset(
    data_dir: str,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, list[str]]:
    """Load precomputed dataset and compute normalization stats.

    Args:
        data_dir: path to directory containing train_dataset.pt
        device: target device

    Returns:
        (points, sdf, params, param_min, param_max, xyz_min, xyz_max, geom_ids)
    """
    path = os.path.join(data_dir, "train_dataset.pt")
    print(f"loading {path}...")
    ds = torch.load(path, map_location="cpu", weights_only=False)

    points = ds["points"]      # [N, 30000, 3]
    sdf = ds["sdf"]            # [N, 30000]
    params = ds["params"]      # [N, 9]
    geom_ids = ds["geom_ids"]  # list[str]

    param_min = params.min(dim=0).values  # [9]
    param_max = params.max(dim=0).values  # [9]

    pts_flat = points.reshape(-1, 3)
    xyz_min = pts_flat.min(dim=0).values  # [3]
    xyz_max = pts_flat.max(dim=0).values  # [3]

    print(f"  {len(geom_ids)} geometries, {points.shape[1]} pts/geom")
    print(f"  param ranges: min={param_min.tolist()}")
    print(f"                max={param_max.tolist()}")
    print(f"  xyz range: min={xyz_min.tolist()}")
    print(f"             max={xyz_max.tolist()}")

    # Keep large tensors on CPU, batches are moved to the device in sample_batch.
    param_min = param_min.to(device)
    param_max = param_max.to(device)
    xyz_min = xyz_min.to(device)
    xyz_max = xyz_max.to(device)

    return points, sdf, params, param_min, param_max, xyz_min, xyz_max, geom_ids


def normalize_params(
    params: Tensor,
    param_min: Tensor,
    param_max: Tensor,
) -> Tensor:
    """Normalize geom params to [-1, 1] via min/max scaling.

    Args:
        params: [B, 9] raw parameters
        param_min: [9] per-param minimum
        param_max: [9] per-param maximum

    Returns:
        [B, 9] normalized to [-1, 1]
    """
    return 2.0 * (params - param_min) / (param_max - param_min + 1e-8) - 1.0


def compute_extreme_weights(
    params: Tensor,
    param_min: Tensor,
    param_max: Tensor,
    max_weight: float = 3.0,
) -> Tensor:
    """Compute per-geometry sampling weights favoring param-space extremes.

    Args:
        params: [N, 9] raw parameters
        param_min: [9] per-param minimum
        param_max: [9] per-param maximum
        max_weight: weight for most extreme geometries (center gets ~1.0)

    Returns:
        [N] normalized probability vector
    """
    normed = (params - param_min) / (param_max - param_min + 1e-8)
    extremity = (normed - 0.5).abs().max(dim=1).values  # [N], range [0, 0.5]
    extremity = extremity / 0.5  # normalize to [0, 1]
    weights = 1.0 + (max_weight - 1.0) * extremity  # [1, max_weight]
    return weights / weights.sum()


def sample_batch(
    points: Tensor,
    sdf: Tensor,
    params: Tensor,
    batch_size: int,
    points_per_geom: int,
    weights: Tensor | None = None,
    device: torch.device | None = None,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sample a random batch of geometries and subsample points.

    Args:
        points: [N, P, 3] all points (may be on CPU)
        sdf: [N, P] all SDF values (may be on CPU)
        params: [N, 9] all parameters (may be on CPU)
        batch_size: number of geometries to sample
        points_per_geom: number of points to subsample per geom
        weights: [N] optional geometry sampling probabilities
        device: target device for output tensors (None = same as input)

    Returns:
        (batch_xyz, batch_sdf, batch_params): [B,K,3], [B,K], [B,9]
    """
    N, P, _ = points.shape

    if weights is not None:
        geom_idx = torch.multinomial(weights, batch_size, replacement=True)
    else:
        geom_idx = torch.randint(0, N, (batch_size,))

    pt_idx = torch.randint(0, P, (batch_size, points_per_geom))

    batch_pts = points[geom_idx]  # [B, P, 3]
    batch_sdf = sdf[geom_idx]    # [B, P]
    batch_params = params[geom_idx]  # [B, 9]

    batch_pts = torch.gather(
        batch_pts, 1, pt_idx.unsqueeze(-1).expand(-1, -1, 3)
    )  # [B, K, 3]
    batch_sdf = torch.gather(batch_sdf, 1, pt_idx)  # [B, K]

    if device is not None:
        batch_pts = batch_pts.to(device)
        batch_sdf = batch_sdf.to(device)
        batch_params = batch_params.to(device)

    return batch_pts, batch_sdf, batch_params


def compute_eikonal_loss(
    model: BWBSDFNet,
    xyz: Tensor,
    cond: Tensor,
    subsample: float = 0.25,
) -> Tensor:
    """Eikonal regularization: (|grad_xyz SDF| - 1)^2.

    Args:
        model: SDF network
        xyz: [B, Q, 3] query points
        cond: [B, 9] normalized conditions
        subsample: fraction of points to use

    Returns:
        scalar eikonal loss
    """
    B, Q, _ = xyz.shape
    n_sub = max(1, int(Q * subsample))

    idx = torch.randint(0, Q, (B, n_sub), device=xyz.device)
    xyz_sub = torch.gather(xyz, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    xyz_sub = xyz_sub.detach().requires_grad_(True)

    sdf_pred = model(xyz_sub, cond)  # [B, n_sub]

    grad = torch.autograd.grad(
        sdf_pred,
        xyz_sub,
        grad_outputs=torch.ones_like(sdf_pred),
        create_graph=True,
    )[0]  # [B, n_sub, 3]

    grad_norm = grad.norm(dim=-1)  # [B, n_sub]
    return ((grad_norm - 1.0) ** 2).mean()


def _compute_sdf_loss(pred: Tensor, target: Tensor, cfg: dict) -> Tensor:
    """Compute SDF reconstruction loss based on cfg['loss_type'].

    Args:
        pred: [B, K] predicted SDF
        target: [B, K] ground truth SDF
        cfg: config dict with 'loss_type' and loss-specific params

    Returns:
        scalar loss
    """
    loss_type = cfg.get("loss_type", "relative_mse")

    if loss_type == "relative_mse":
        rel_err = (pred - target) / (target.abs() + cfg.get("rel_eps", 1e-3))
        return (rel_err ** 2).mean()

    elif loss_type == "absolute_mse":
        return ((pred - target) ** 2).mean()

    elif loss_type == "l1":
        return (pred - target).abs().mean()

    elif loss_type == "clamped_l1":
        delta = cfg.get("clamp_delta", 0.1)
        return (pred.clamp(-delta, delta) - target.clamp(-delta, delta)).abs().mean()

    elif loss_type == "weighted_mse":
        # weight by |sdf| band: w_near below t_near, w_far above t_far, else 1.
        t_near = cfg.get("w_near_thresh", 0.01)
        t_far = cfg.get("w_far_thresh", 0.1)
        w_near = cfg.get("w_near", 10.0)
        w_far = cfg.get("w_far", 0.1)
        abs_target = target.abs()
        weights = torch.ones_like(target)
        weights[abs_target < t_near] = w_near
        weights[abs_target >= t_far] = w_far
        return (weights * (pred - target) ** 2).mean()

    else:
        raise ValueError(f"unknown loss_type: {loss_type}")


def train(cfg: dict | None = None) -> None:
    """Main training loop.

    Args:
        cfg: config dict, defaults to DEFAULT_CONFIG
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}")

    os.makedirs(cfg["output_dir"], exist_ok=True)

    data_dir = cfg.get("data_dir", os.path.join(os.path.dirname(__file__), "data"))
    points, sdf_gt, params, param_min, param_max, xyz_min, xyz_max, geom_ids = load_dataset(
        data_dir, device,
    )

    if cfg.get("normalize_xyz", False):
        print("  normalizing XYZ to [-1, 1] per axis")
        points = 2.0 * (points - xyz_min.cpu()) / (xyz_max.cpu() - xyz_min.cpu() + 1e-8) - 1.0

    val_frac = cfg.get("val_split", 0.0)
    N = points.shape[0]
    if val_frac > 0:
        rng = np.random.RandomState(42)
        perm = rng.permutation(N)
        n_val = max(1, int(N * val_frac))
        val_idx = torch.tensor(perm[:n_val], device=points.device)
        train_idx = torch.tensor(perm[n_val:], device=points.device)
        val_points, val_sdf, val_params = points[val_idx], sdf_gt[val_idx], params[val_idx]
        points, sdf_gt, params = points[train_idx], sdf_gt[train_idx], params[train_idx]
        all_geom_ids = geom_ids
        geom_ids = [all_geom_ids[i] for i in perm[n_val:]]
        val_geom_ids = [all_geom_ids[i] for i in perm[:n_val]]
        print(f"  train/val split: {len(geom_ids)} train, {len(val_geom_ids)} val")
    else:
        val_points = val_sdf = val_params = None

    extreme_max_w = cfg.get("extreme_weight", 1.0)
    if extreme_max_w > 1.0:
        geom_weights = compute_extreme_weights(params, param_min.cpu(), param_max.cpu(), extreme_max_w)
        print(f"  extreme oversampling: max_weight={extreme_max_w:.1f}")
    else:
        geom_weights = None

    geom_index_path = os.path.join(data_dir, "geom_index.json")
    vis_geom_indices = _pick_extreme_geometries(params, param_min.cpu(), param_max.cpu())
    print(f"  viz geometries: {[geom_ids[i] for i in vis_geom_indices]}")

    vtk_meshes = _load_vtk_meshes(geom_index_path, geom_ids, vis_geom_indices)

    model = BWBSDFNet(
        fourier_bands=cfg["fourier_bands"],
        hidden_dim=cfg["hidden_dim"],
        cond_dim=cfg["cond_dim"],
        n_blocks=cfg.get("n_blocks", 3),
        activation=cfg.get("activation", "gelu"),
        omega_0=cfg.get("omega_0", 30.0),
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"BWB SDF net: {n_params / 1e6:.2f}M params")

    opt_cls = torch.optim.AdamW if cfg.get("optimizer", "adam") == "adamw" else torch.optim.Adam
    optimizer = opt_cls(model.parameters(), lr=cfg["lr"])
    def lr_decay(step):
        return 1.0 - step / cfg["total_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_decay)

    use_amp = cfg.get("amp", False) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("AMP enabled (eikonal in fp32)")

    start_step = 0
    resume_path = cfg.get("resume")
    if resume_path:
        print(f"resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        if cfg.get("fresh_lr", False):
            print(f"  fresh LR: {cfg['lr']}, {cfg['total_steps']} steps")
        else:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_step = ckpt["step"]
        print(f"  resumed at step {start_step}")

    wandb_kwargs = {}
    if resume_path and cfg.get("wandb_run_id"):
        wandb_kwargs["id"] = cfg["wandb_run_id"]
        wandb_kwargs["resume"] = "must"
    wandb.init(
        project="bwb-sdf",
        group="bwb-sdf",
        tags=["bwb-sdf"],
        config=cfg,
        **wandb_kwargs,
    )
    wandb.watch(model, log="gradients", log_freq=cfg["log_interval"])

    t0 = time.time()
    running_mse = 0.0
    running_eik = 0.0

    for step in range(start_step + 1, cfg["total_steps"] + 1):
        batch_xyz, batch_sdf, batch_params = sample_batch(
            points, sdf_gt, params,
            cfg["batch_size"], cfg["points_per_geom"],
            weights=geom_weights, device=device,
        )

        cond_norm = normalize_params(batch_params, param_min, param_max)

        with torch.amp.autocast("cuda", enabled=use_amp):
            pred_sdf = model(batch_xyz, cond_norm)
            loss_mse = _compute_sdf_loss(pred_sdf, batch_sdf, cfg)

        # eikonal loss in fp32 (autograd.grad needs full precision)
        loss_eik = compute_eikonal_loss(
            model, batch_xyz, cond_norm, subsample=cfg["eik_subsample"],
        )
        loss = loss_mse + cfg["lambda_eik"] * loss_eik

        with torch.no_grad():
            sign_acc = ((pred_sdf.sign() == batch_sdf.sign()).float().mean()).item()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        running_mse += loss_mse.item()
        running_eik += loss_eik.item()

        lr = optimizer.param_groups[0]["lr"]
        wandb.log({
            "loss/mse": loss_mse.item(),
            "loss/eikonal": loss_eik.item(),
            "loss/total": loss.item(),
            "lr": lr,
            "sign_accuracy": sign_acc,
        }, step=step)

        if step % cfg["log_interval"] == 0:
            avg_mse = running_mse / cfg["log_interval"]
            avg_eik = running_eik / cfg["log_interval"]
            elapsed = time.time() - t0
            ms_per_step = elapsed / step * 1000
            print(
                f"step {step:6d} | mse {avg_mse:.6f} | eik {avg_eik:.4f} | "
                f"sign {sign_acc:.3f} | lr {lr:.2e} | {ms_per_step:.0f} ms/step"
            )
            running_mse = 0.0
            running_eik = 0.0

            if val_points is not None:
                model.eval()
                with torch.no_grad():
                    vb_xyz, vb_sdf, vb_params = sample_batch(
                        val_points, val_sdf, val_params,
                        cfg["batch_size"], cfg["points_per_geom"],
                        device=device,
                    )
                    vc = normalize_params(vb_params, param_min, param_max)
                    vp = model(vb_xyz, vc)
                    v_mse = _compute_sdf_loss(vp, vb_sdf, cfg).item()
                    v_sign = ((vp.sign() == vb_sdf.sign()).float().mean()).item()
                wandb.log({"val/mse": v_mse, "val/sign_accuracy": v_sign}, step=step)
                print(f"        val | mse {v_mse:.6f} | sign {v_sign:.3f}")
                model.train()

        if step % cfg["vis_interval"] == 0:
            from viz_training import save_cross_section_figure
            vis_path = save_cross_section_figure(
                model, points, params, param_min, param_max,
                vis_geom_indices, geom_ids, vtk_meshes,
                cfg, step, device,
            )
            if vis_path is not None:
                wandb.log({"cross_sections": wandb.Image(vis_path)}, step=step)

        if step % cfg["ckpt_interval"] == 0:
            ckpt_path = os.path.join(cfg["output_dir"], f"ckpt_{step:06d}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "config": cfg,
                "param_min": param_min.cpu(),
                "param_max": param_max.cpu(),
                "xyz_min": xyz_min.cpu(),
                "xyz_max": xyz_max.cpu(),
            }, ckpt_path)
            print(f"saved {ckpt_path}")

            art = wandb.Artifact(
                f"bwb-sdf-ckpt-{step:06d}",
                type="model",
                metadata={"step": step, "sign_acc": sign_acc},
            )
            art.add_file(ckpt_path)
            wandb.log_artifact(art)

    final_path = os.path.join(cfg["output_dir"], "final.pt")
    torch.save({
        "step": cfg["total_steps"],
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": cfg,
        "param_min": param_min.cpu(),
        "param_max": param_max.cpu(),
        "xyz_min": xyz_min.cpu(),
        "xyz_max": xyz_max.cpu(),
    }, final_path)
    print(f"training done. saved {final_path}")

    art = wandb.Artifact("bwb-sdf-final", type="model", metadata={"step": cfg["total_steps"]})
    art.add_file(final_path)
    wandb.log_artifact(art)

    wandb.finish()


def _pick_extreme_geometries(
    params: Tensor,
    param_min: Tensor,
    param_max: Tensor,
) -> list[int]:
    """Pick 8 geometries spanning param space extremes.

    Args:
        params: [N, 9] raw parameters
        param_min: [9] min values
        param_max: [9] max values

    Returns:
        list of 8 geometry indices
    """
    # param indices: B1=0, B2=1, B3=2, C2=3, C3=4, C4=5, S1=6, S2=7, S3=8
    B3 = params[:, 2]  # span
    C2 = params[:, 3]  # chord ratio
    S3 = params[:, 8]  # sweep

    B3_n = (B3 - B3.min()) / (B3.max() - B3.min() + 1e-8)
    C2_n = (C2 - C2.min()) / (C2.max() - C2.min() + 1e-8)
    S3_n = (S3 - S3.min()) / (S3.max() - S3.min() + 1e-8)

    corners = [
        (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
        (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1),
    ]

    indices = []
    used = set()
    for b3t, c2t, s3t in corners:
        dist = (B3_n - b3t) ** 2 + (C2_n - c2t) ** 2 + (S3_n - s3t) ** 2
        sorted_idx = dist.argsort()
        for idx in sorted_idx:
            idx_int = idx.item()
            if idx_int not in used:
                indices.append(idx_int)
                used.add(idx_int)
                break

    return indices


def _load_vtk_meshes(
    geom_index_path: str,
    geom_ids: list[str],
    vis_indices: list[int],
) -> list:
    """Load VTK meshes for the 8 visualization geometries.

    Returns empty list if pyvista not available or files missing.
    """
    try:
        import pyvista as pv
    except ImportError:
        print("  pyvista not available, viz will skip GT overlay")
        return []

    if not os.path.exists(geom_index_path):
        print("  geom_index.json not found, viz will skip GT overlay")
        return []

    with open(geom_index_path) as f:
        geom_index = json.load(f)

    meshes = []
    for idx in vis_indices:
        gid = geom_ids[idx]
        entry = geom_index.get(gid, {})
        vtk_path = entry.get("vtk_path", "")
        if os.path.exists(vtk_path):
            mesh = pv.read(vtk_path)
            if not mesh.is_all_triangles:
                mesh = mesh.triangulate()
            meshes.append(mesh)
        else:
            print(f"  VTK missing for {gid}: {vtk_path}")
            meshes.append(None)

    return meshes


def _build_parser() -> argparse.ArgumentParser:
    d = DEFAULT_CONFIG
    p = argparse.ArgumentParser(
        description="train conditional BWB SDF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--steps", type=int, default=d["total_steps"])
    p.add_argument("--batch", type=int, default=d["batch_size"])
    p.add_argument("--points", type=int, default=d["points_per_geom"])
    p.add_argument("--hidden-dim", type=int, default=d["hidden_dim"])
    p.add_argument("--fourier-bands", type=int, default=d["fourier_bands"])
    p.add_argument("--n-blocks", type=int, default=d.get("n_blocks", 3),
                   help="layers per block in SDF network")
    p.add_argument("--activation", type=str, default="gelu",
                   choices=["gelu", "siren"], help="network activation type")
    p.add_argument("--omega-0", type=float, default=30.0,
                   help="SIREN frequency scaling (only for --activation siren)")
    p.add_argument("--lr", type=float, default=d["lr"])
    p.add_argument("--optimizer", type=str, default="adam",
                   choices=["adam", "adamw"], help="optimizer type")
    p.add_argument("--loss-type", type=str, default="relative_mse",
                   choices=["relative_mse", "absolute_mse", "l1", "clamped_l1", "weighted_mse"])
    p.add_argument("--clamp-delta", type=float, default=0.1,
                   help="clamp threshold for clamped_l1 loss")
    p.add_argument("--rel-eps", type=float, default=1e-3,
                   help="epsilon for relative_mse denominator")
    p.add_argument("--w-near", type=float, default=10.0,
                   help="weight for near-surface points (weighted_mse)")
    p.add_argument("--w-near-thresh", type=float, default=0.01,
                   help="SDF threshold for near-surface (weighted_mse)")
    p.add_argument("--w-far", type=float, default=0.1,
                   help="weight for far-field points (weighted_mse)")
    p.add_argument("--w-far-thresh", type=float, default=0.1,
                   help="SDF threshold for far-field (weighted_mse)")
    p.add_argument("--normalize-xyz", action="store_true",
                   help="normalize XYZ coords to [-1,1] per axis")
    p.add_argument("--extreme-weight", type=float, default=1.0,
                   help="max sampling weight for param-extreme geometries (1.0=uniform, 3.0=3x oversample)")
    p.add_argument("--val-split", type=float, default=0.0,
                   help="fraction of geometries for validation (0=no val)")
    p.add_argument("--lambda-eik", type=float, default=d["lambda_eik"])
    p.add_argument("--log-interval", type=int, default=d["log_interval"])
    p.add_argument("--vis-interval", type=int, default=d["vis_interval"])
    p.add_argument("--ckpt-interval", type=int, default=d["ckpt_interval"])
    p.add_argument("--output", type=str, default=d["output_dir"])
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--resume", type=str, default=None,
                   help="path to checkpoint .pt to resume from")
    p.add_argument("--fresh-lr", action="store_true",
                   help="reset optimizer+scheduler on resume (use --lr and --steps for new schedule)")
    p.add_argument("--wandb-run-id", type=str, default=None,
                   help="wandb run id to resume (optional)")
    p.add_argument("--wandb-dir", type=str, default=None, help="wandb cache dir")
    p.add_argument("--amp", action="store_true",
                   help="enable mixed-precision training (fp16 forward, fp32 eikonal)")
    p.add_argument("--scratch", type=str, default=None,
                   help="scratch root, sets output to <scratch>/bwb_sdf and wandb to <scratch>/wandb")
    return p


def _args_to_config(args: argparse.Namespace) -> dict:
    cfg = {
        "total_steps": args.steps,
        "batch_size": args.batch,
        "points_per_geom": args.points,
        "hidden_dim": args.hidden_dim,
        "fourier_bands": args.fourier_bands,
        "n_blocks": args.n_blocks,
        "activation": args.activation,
        "omega_0": args.omega_0,
        "lr": args.lr,
        "optimizer": args.optimizer,
        "loss_type": args.loss_type,
        "clamp_delta": args.clamp_delta,
        "rel_eps": args.rel_eps,
        "w_near": args.w_near,
        "w_near_thresh": args.w_near_thresh,
        "w_far": args.w_far,
        "w_far_thresh": args.w_far_thresh,
        "normalize_xyz": args.normalize_xyz,
        "extreme_weight": args.extreme_weight,
        "val_split": args.val_split,
        "lambda_eik": args.lambda_eik,
        "log_interval": args.log_interval,
        "vis_interval": args.vis_interval,
        "ckpt_interval": args.ckpt_interval,
        "output_dir": args.output,
    }
    if args.data_dir is not None:
        cfg["data_dir"] = args.data_dir
    if args.resume is not None:
        cfg["resume"] = args.resume
        cfg["fresh_lr"] = args.fresh_lr
    if args.wandb_run_id is not None:
        cfg["wandb_run_id"] = args.wandb_run_id
    cfg["amp"] = args.amp
    return cfg


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    if args.scratch is not None:
        if args.output == DEFAULT_CONFIG["output_dir"]:
            args.output = os.path.join(args.scratch, "bwb_sdf")
        if args.wandb_dir is None:
            args.wandb_dir = os.path.join(args.scratch, "wandb")

    if args.wandb_dir is not None:
        os.environ["WANDB_DIR"] = args.wandb_dir

    cfg = _args_to_config(args)
    train(cfg)
