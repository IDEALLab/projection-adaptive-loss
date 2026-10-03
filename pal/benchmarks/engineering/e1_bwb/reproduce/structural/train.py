"""Train structural surrogate cVAE + predictor.

Usage:
    python -m pal.benchmarks.engineering.e1_bwb.reproduce.structural.train \\
        --data /path/to/merged/structural.parquet
    python -m pal.benchmarks.engineering.e1_bwb.reproduce.structural.train \\
        --data ... --ribs-data .../ribs.parquet --epochs 200
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from pal.benchmarks.engineering.e1_bwb.struct_surrogate.model import StructuralCVAE, kl_divergence

# Column definitions matching generate_data.py
BWB_COLS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3", "L"]
STRUCT_COLS = [
    "front_spar_x", "front_spar_dev", "rear_spar_x",
    "bat_x", "bat_y", "bat_z", "bat_z_center",
    "rib_start_y", "rib_end_y",
]
THICK_COLS = ["skin_t", "front_spar_w", "rear_spar_w"]
PROP_COLS = [
    "A", "u_cg", "v_cg", "I_uu", "I_vv", "I_uv",
    "I_1", "I_2", "Q_u_max", "Q_v_max", "J",
]
LOG_EPS = 1e-15  # clamp before log transform


def load_and_split(
    path: str, test_frac: float = 0.2, seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load parquet, split by sample_id into train/test."""
    df = pd.read_parquet(path)
    sample_ids = np.array(df["sample_id"].unique())
    rng = np.random.default_rng(seed)
    rng.shuffle(sample_ids)
    n_test = int(len(sample_ids) * test_frac)
    test_ids = set(sample_ids[:n_test])
    mask = df["sample_id"].isin(test_ids)
    return df[~mask].reset_index(drop=True), df[mask].reset_index(drop=True)


class Normalizer:
    """Standardize columns, optionally log-transforming first."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, x: torch.Tensor) -> "Normalizer":
        self.mean = x.mean(dim=0)
        self.std = x.std(dim=0).clamp(min=1e-8)
        return self

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def to(self, device: torch.device) -> "Normalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


def df_to_tensors(df: pd.DataFrame) -> dict[str, torch.Tensor]:
    """Extract model input/output tensors from dataframe."""
    bwb = torch.tensor(df[BWB_COLS].values, dtype=torch.float32)
    struct = torch.tensor(df[STRUCT_COLS].values, dtype=torch.float32)
    thick = torch.tensor(df[THICK_COLS].values, dtype=torch.float32)
    y = torch.tensor(df[["y"]].values, dtype=torch.float32)

    # Log-transform properties, clamp small values
    props_raw = df[PROP_COLS].values.copy()
    props_raw = np.abs(props_raw)  # handle sign (I_uv can be negative)
    props_raw = np.clip(props_raw, LOG_EPS, None)
    props_log = np.log(props_raw)

    # We'll store log(|val|) and train on that; sign handled separately if needed
    props = torch.tensor(props_log, dtype=torch.float32)
    return {"bwb": bwb, "struct": struct, "thick": thick, "y": y, "props": props}


def make_dataloader(
    tensors: dict[str, torch.Tensor], batch_size: int, shuffle: bool = True,
) -> DataLoader:
    ds = TensorDataset(
        tensors["bwb"], tensors["struct"], tensors["thick"],
        tensors["y"], tensors["props"],
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(
    model: StructuralCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    norm_struct: Normalizer,
    norm_props: Normalizer,
    norm_bwb: Normalizer,
    norm_thick: Normalizer,
    norm_y: Normalizer,
    beta: float,
    lam: float,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0, "recon": 0, "kl": 0, "pred": 0, "n": 0}

    for bwb, struct, thick, y, props in loader:
        bwb, struct, thick, y, props = (
            bwb.to(device), struct.to(device), thick.to(device),
            y.to(device), props.to(device),
        )

        # Normalize inputs
        bwb_n = norm_bwb.transform(bwb)
        struct_n = norm_struct.transform(struct)
        thick_n = norm_thick.transform(thick)
        y_n = norm_y.transform(y)
        props_n = norm_props.transform(props)

        out = model(struct_n, bwb_n, thick_n, y_n)

        # Losses
        l_recon = nn.functional.mse_loss(out["struct_recon"], struct_n)
        l_kl = kl_divergence(out["mu"], out["log_var"])
        l_pred = nn.functional.mse_loss(out["properties"], props_n)

        loss = l_recon + beta * l_kl + lam * l_pred

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = bwb.shape[0]
        totals["loss"] += loss.item() * bs
        totals["recon"] += l_recon.item() * bs
        totals["kl"] += l_kl.item() * bs
        totals["pred"] += l_pred.item() * bs
        totals["n"] += bs

    n = totals["n"]
    return {k: v / n for k, v in totals.items() if k != "n"}


@torch.no_grad()
def eval_epoch(
    model: StructuralCVAE,
    loader: DataLoader,
    norm_struct: Normalizer,
    norm_props: Normalizer,
    norm_bwb: Normalizer,
    norm_thick: Normalizer,
    norm_y: Normalizer,
    beta: float,
    lam: float,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals = {"loss": 0, "recon": 0, "kl": 0, "pred": 0, "n": 0}

    for bwb, struct, thick, y, props in loader:
        bwb, struct, thick, y, props = (
            bwb.to(device), struct.to(device), thick.to(device),
            y.to(device), props.to(device),
        )

        bwb_n = norm_bwb.transform(bwb)
        struct_n = norm_struct.transform(struct)
        thick_n = norm_thick.transform(thick)
        y_n = norm_y.transform(y)
        props_n = norm_props.transform(props)

        out = model(struct_n, bwb_n, thick_n, y_n)

        l_recon = nn.functional.mse_loss(out["struct_recon"], struct_n)
        l_kl = kl_divergence(out["mu"], out["log_var"])
        l_pred = nn.functional.mse_loss(out["properties"], props_n)
        loss = l_recon + beta * l_kl + lam * l_pred

        bs = bwb.shape[0]
        totals["loss"] += loss.item() * bs
        totals["recon"] += l_recon.item() * bs
        totals["kl"] += l_kl.item() * bs
        totals["pred"] += l_pred.item() * bs
        totals["n"] += bs

    n = totals["n"]
    return {k: v / n for k, v in totals.items() if k != "n"}


def main():
    parser = argparse.ArgumentParser(description="Train structural surrogate cVAE")
    parser.add_argument("--data", type=str, required=True, help="Path to structural.parquet")
    parser.add_argument("--ribs-data", type=str, default=None, help="Path to ribs.parquet")
    parser.add_argument("--output-dir", type=str, default=None, help="Checkpoint/log dir")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--latent-dim", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate in MLPs")
    parser.add_argument("--beta", type=float, default=0.1, help="Final KL weight")
    parser.add_argument("--lam", type=float, default=10.0, help="Prediction loss weight")
    parser.add_argument("--warmup-frac", type=float, default=0.05, help="VAE warmup fraction")
    parser.add_argument("--test-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--wandb-dir", type=str, default=None)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Device: {device}")

    # Output dir
    if args.output_dir is None:
        args.output_dir = f"runs/{int(time.time())}"
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # Save config
    config = vars(args)
    config["device"] = str(device)
    with open(out_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    # Wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        import wandb
        wandb_kwargs = {"dir": args.wandb_dir} if args.wandb_dir else {}
        wandb.init(
            project="structural-surrogate",
            config=config,
            name=out_dir.name,
            **wandb_kwargs,
        )

    # Load data
    print(f"Loading {args.data} ...")
    train_df, test_df = load_and_split(args.data, args.test_frac, args.seed)
    print(f"Train: {len(train_df):,} rows ({train_df['sample_id'].nunique()} geom)")
    print(f"Test:  {len(test_df):,} rows ({test_df['sample_id'].nunique()} geom)")

    train_t = df_to_tensors(train_df)
    test_t = df_to_tensors(test_df)

    # Fit normalizers on train data
    norm_bwb = Normalizer().fit(train_t["bwb"]).to(device)
    norm_struct = Normalizer().fit(train_t["struct"]).to(device)
    norm_thick = Normalizer().fit(train_t["thick"]).to(device)
    norm_y = Normalizer().fit(train_t["y"]).to(device)
    norm_props = Normalizer().fit(train_t["props"]).to(device)

    train_loader = make_dataloader(train_t, args.batch_size, shuffle=True)
    test_loader = make_dataloader(test_t, args.batch_size, shuffle=False)

    # Model
    model = StructuralCVAE(
        struct_dim=len(STRUCT_COLS),
        bwb_dim=len(BWB_COLS),
        thick_dim=len(THICK_COLS),
        prop_dim=len(PROP_COLS),
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    warmup_epochs = int(args.epochs * args.warmup_frac)
    best_val_loss = float("inf")

    print(f"\nTraining {args.epochs} epochs (warmup: {warmup_epochs}, cosine LR decay) ...")
    for epoch in range(args.epochs):
        # Beta annealing: 0 -> target over warmup period
        if epoch < warmup_epochs:
            beta = args.beta * (epoch + 1) / warmup_epochs
            lam = 0.0  # no prediction loss during warmup
        else:
            beta = args.beta
            lam = args.lam

        t0 = time.perf_counter()
        train_metrics = train_epoch(
            model, train_loader, optimizer,
            norm_struct, norm_props, norm_bwb, norm_thick, norm_y,
            beta, lam, device,
        )
        val_metrics = eval_epoch(
            model, test_loader,
            norm_struct, norm_props, norm_bwb, norm_thick, norm_y,
            beta, lam, device,
        )
        dt = time.perf_counter() - t0

        # Log
        log = {
            "epoch": epoch,
            "beta": beta,
            "lambda": lam,
            "train/loss": train_metrics["loss"],
            "train/recon": train_metrics["recon"],
            "train/kl": train_metrics["kl"],
            "train/pred": train_metrics["pred"],
            "val/loss": val_metrics["loss"],
            "val/recon": val_metrics["recon"],
            "val/kl": val_metrics["kl"],
            "val/pred": val_metrics["pred"],
        }

        if use_wandb:
            wandb.log(log, step=epoch)

        # Checkpoint best
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "config": config,
                "norm_bwb": {"mean": norm_bwb.mean.cpu(), "std": norm_bwb.std.cpu()},
                "norm_struct": {"mean": norm_struct.mean.cpu(), "std": norm_struct.std.cpu()},
                "norm_thick": {"mean": norm_thick.mean.cpu(), "std": norm_thick.std.cpu()},
                "norm_y": {"mean": norm_y.mean.cpu(), "std": norm_y.std.cpu()},
                "norm_props": {"mean": norm_props.mean.cpu(), "std": norm_props.std.cpu()},
            }, out_dir / "best.pt")
            marker = " *"
        else:
            marker = ""

        scheduler.step()

        if epoch % 10 == 0 or epoch == args.epochs - 1 or marker:
            lr_now = optimizer.param_groups[0]["lr"]
            print(
                f"[{epoch:3d}/{args.epochs}] "
                f"loss={train_metrics['loss']:.4f} "
                f"recon={train_metrics['recon']:.4f} "
                f"kl={train_metrics['kl']:.4f} "
                f"pred={train_metrics['pred']:.4f} "
                f"| val={val_metrics['loss']:.4f} "
                f"vpred={val_metrics['pred']:.4f} "
                f"lr={lr_now:.2e} ({dt:.1f}s){marker}"
            )

    # Extract latent bounds for MDO
    print("\nExtracting latent bounds ...")
    model.eval()
    z_all = []
    with torch.no_grad():
        for bwb, struct, _thick, _y, _props in train_loader:
            bwb_n = norm_bwb.transform(bwb.to(device))
            struct_n = norm_struct.transform(struct.to(device))
            mu, _ = model.encode(struct_n, bwb_n)
            z_all.append(mu.cpu())
    z_all = torch.cat(z_all, dim=0)
    # Deduplicate to geometry level (same sample_id shares same struct params -> same z)
    # z_all has one entry per row, but many rows share the same geometry
    z_stats = {
        "z_mean": z_all.mean(dim=0),
        "z_std": z_all.std(dim=0),
        "z_min": z_all.min(dim=0).values,
        "z_max": z_all.max(dim=0).values,
        "z_p01": z_all.quantile(0.01, dim=0),
        "z_p99": z_all.quantile(0.99, dim=0),
    }
    print(f"  z range per dim: [{z_stats['z_min'].min():.2f}, {z_stats['z_max'].max():.2f}]")
    print(f"  z std per dim:   [{z_stats['z_std'].min():.2f}, {z_stats['z_std'].max():.2f}]")

    # Save final checkpoint (includes z_stats for MDO bounds)
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "config": config,
        "z_stats": z_stats,
        "norm_bwb": {"mean": norm_bwb.mean.cpu(), "std": norm_bwb.std.cpu()},
        "norm_struct": {"mean": norm_struct.mean.cpu(), "std": norm_struct.std.cpu()},
        "norm_thick": {"mean": norm_thick.mean.cpu(), "std": norm_thick.std.cpu()},
        "norm_y": {"mean": norm_y.mean.cpu(), "std": norm_y.std.cpu()},
        "norm_props": {"mean": norm_props.mean.cpu(), "std": norm_props.std.cpu()},
    }, out_dir / "final.pt")

    print(f"\nDone. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints: {out_dir}")

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
