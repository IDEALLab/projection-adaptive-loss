#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from pal.benchmarks.engineering.e1_bwb.film_surface.model import FiLMNet  # noqa: E402

from .dataset import UnifiedDesignDataset, design_collate_fn, split_designs  # noqa: E402

# CLI args (wandb)
parser = argparse.ArgumentParser()
parser.add_argument("--no-wandb", action="store_true", help="disable wandb logging")
parser.add_argument("--wandb-entity", default=os.environ.get("WANDB_ENTITY"))
parser.add_argument("--wandb-project", default="blendednet-repro")
parser.add_argument("--wandb-dir", default="./wandb/")
parser.add_argument("--wandb-run-name", default=None, help="optional run name")
args = parser.parse_args()

USE_WANDB = not args.no_wandb
if USE_WANDB:
    import wandb

# Paths
REPO_ROOT = Path(__file__).resolve().parent
_DATA = Path(os.environ.get("BLENDEDNET_DATA", REPO_ROOT / "data"))
_CSV = Path(os.environ.get("BLENDEDNET_CSV", REPO_ROOT / "csv_files"))

CSV = str(_CSV / "case_with_geom_params_train.csv")
H5  = str(_DATA / "surface_data.hdf5")
NORM_JSON = str(REPO_ROOT / "norm_stats.json")
CKPT_DIR  = Path(os.environ.get("BLENDEDNET_CHECKPOINTS", REPO_ROOT / "checkpoints"))
CKPT_DIR.mkdir(parents=True, exist_ok=True)
BEST_PATH  = CKPT_DIR / "film_best.pth"
FINAL_PATH = CKPT_DIR / "film_final.pth"
CFG_PATH   = CKPT_DIR / "train_config.json"

# Hyperparams
EPOCHS = int(os.getenv("FILM_EPOCHS", "20000"))
BATCH_SIZE = int(os.getenv("FILM_BATCH_SIZE", "64"))
LR = float(os.getenv("FILM_LR", "5e-4"))
TRAIN_RATIO = float(os.getenv("FILM_TRAIN_RATIO", "0.9"))
NUM_WORKERS = int(os.getenv("FILM_NUM_WORKERS", "0"))
GRAD_CLIP = float(os.getenv("FILM_GRAD_CLIP", "0"))  # 0 = disabled
DISABLE_AMP = int(os.getenv("FILM_DISABLE_AMP", "0")) == 1
LR_END = float(os.getenv("FILM_LR_END", "0"))  # 0 = constant LR

# Device & AMP
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    device_name = torch.cuda.get_device_name(0)
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = torch.device("mps")
    device_name = "Apple Metal (MPS)"
else:
    device = torch.device("cpu")
    device_name = "CPU"
print(f"Using device: {device_name}")

use_amp = device.type == "cuda" and not DISABLE_AMP
scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

# Load norm stats
if os.path.isfile(NORM_JSON):
    raw = json.load(open(NORM_JSON))
    NORM = {k: np.array(v, dtype=np.float32) for k, v in raw.items()}
else:
    tmp_ds = UnifiedDesignDataset(csv_path=CSV, hdf5_path=H5, norm_stats=None, mode="train")
    NORM = {k: np.array(v, dtype=np.float32) for k, v in tmp_ds.norm_stats.items()}
    with open(NORM_JSON, "w") as f:
        json.dump({k: v.tolist() for k, v in NORM.items()}, f, indent=2)
    tmp_ds.close()

# Dataset & Loaders
PRELOAD = int(os.getenv("FILM_PRELOAD", "0")) == 1
ds_full = UnifiedDesignDataset(csv_path=CSV, hdf5_path=H5, norm_stats=NORM, mode="train", preload=PRELOAD)
train_ds, val_ds = split_designs(ds_full, train_ratio=TRAIN_RATIO)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=design_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)
val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=design_collate_fn,
    num_workers=NUM_WORKERS,
    pin_memory=device.type == "cuda",
)

# Model
model = FiLMNet(cond_dim=13, coord_dim=6, output_dim=3,
                hidden_dim=256, num_layers=4, extra_layers=3).to(device)
opt = torch.optim.Adam(model.parameters(), lr=LR)

# Save config
cfg = dict(
    csv=str(CSV), h5=str(H5), norm_json=str(NORM_JSON),
    epochs=EPOCHS, batch_size=BATCH_SIZE, lr=LR, train_ratio=TRAIN_RATIO,
    device=str(device), amp=use_amp, model=dict(cond_dim=13, coord_dim=6, output_dim=3,
                                                hidden_dim=256, num_layers=4, extra_layers=3)
)
json.dump(cfg, open(CFG_PATH, "w"), indent=2)

# Wandb
if USE_WANDB:
    os.makedirs(args.wandb_dir, exist_ok=True)
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name,
        dir=args.wandb_dir,
        config=cfg,
    )

# Resume
RESUME_PATH = Path(os.getenv("FILM_RESUME", ""))
start_epoch = 1
best_val = float("inf")
if RESUME_PATH.is_file():
    ckpt = torch.load(RESUME_PATH, map_location=device, weights_only=False)
    if "model" in ckpt:  # new format
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            opt.load_state_dict(ckpt["optimizer"])
        if "scaler" in ckpt:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val = ckpt.get("best_val", float("inf"))
    else:  # old format (model state_dict only)
        model.load_state_dict(ckpt)
        start_epoch = int(os.getenv("FILM_START_EPOCH", "1"))
    print(f"Resumed from {RESUME_PATH} (starting epoch {start_epoch}, best_val={best_val:.4e})")

# Train/Val Loop
LOSS_PLOT = CKPT_DIR / "loss_curves.png"
train_losses, val_losses = [], []
t0 = time.time()
print(f"Starting training: epochs {start_epoch}-{EPOCHS}, batch_size={BATCH_SIZE}, device={device}")
print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")
sys.stdout.flush()

for epoch in range(start_epoch, EPOCHS + 1):
    # linear LR decay
    if LR_END > 0:
        frac = (epoch - 1) / max(EPOCHS - 1, 1)
        cur_lr = LR + (LR_END - LR) * frac
        for pg in opt.param_groups:
            pg["lr"] = cur_lr

    model.train()
    running = 0.0
    count = 0

    for batch_idx, (coords, conds, targets) in enumerate(train_loader):
        if epoch == 1 and batch_idx == 0:
            print(f"  First batch: coords={coords.shape}, conds={conds.shape}, targets={targets.shape}")
            sys.stdout.flush()
        coords, conds, targets = coords.to(device), conds.to(device), targets.to(device)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred = model(coords, conds)
            loss = torch.mean((pred - targets) ** 2)

        scaler.scale(loss).backward()
        if GRAD_CLIP > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scaler.step(opt)
        scaler.update()

        running += loss.item() * coords.size(0)
        count   += coords.size(0)

    train_mse = running / max(count, 1)

    # validation
    model.eval()
    v_running, v_count = 0.0, 0
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
        for coords, conds, targets in val_loader:
            coords, conds, targets = coords.to(device), conds.to(device), targets.to(device)
            pred = model(coords, conds)
            vloss = torch.mean((pred - targets) ** 2)
            v_running += vloss.item() * coords.size(0)
            v_count   += coords.size(0)
    val_mse = v_running / max(v_count, 1)

    train_losses.append(train_mse)
    val_losses.append(val_mse)

    if USE_WANDB:
        wandb.log({"train_mse": train_mse, "val_mse": val_mse, "epoch": epoch})

    if epoch % 500 == 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogy(train_losses, label="train", linewidth=0.8)
        ax.semilogy(val_losses, label="val", linewidth=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.set_title(f"FiLM training: epoch {epoch}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(LOSS_PLOT, dpi=150)
        plt.close(fig)

    # Periodic checkpoint (every 500 epochs)
    if epoch % 500 == 0:
        ckpt_path = CKPT_DIR / f"film_ep{epoch:05d}.pth"
        torch.save(dict(
            epoch=epoch, model=model.state_dict(), optimizer=opt.state_dict(),
            scaler=scaler.state_dict(), best_val=best_val,
        ), ckpt_path)

    # Track best val
    saved = False
    if val_mse < best_val:
        best_val = val_mse
        saved = True

    if saved or epoch % 10 == 0 or epoch == 1:
        tag = "  <-- BEST" if saved else ""
        print(f"[epoch {epoch:05d}] train={train_mse:.4e} | val={val_mse:.4e}{tag}")

# Save final
torch.save(dict(
    epoch=EPOCHS, model=model.state_dict(), optimizer=opt.state_dict(),
    scaler=scaler.state_dict(), best_val=best_val,
), FINAL_PATH)
dt = time.time() - t0
print(f"\nDone in {dt/60:.1f} min")
print(f"Best weights:  {BEST_PATH}")
print(f"Final weights: {FINAL_PATH}")
ds_full.close()
if USE_WANDB:
    wandb.finish()
