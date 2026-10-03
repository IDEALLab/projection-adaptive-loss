"""Train A_aero: a small MLP mapping (shape, Ma, alt, alpha, L) -> (CL, CD, CM).

3-layer MLP (13 -> 128 -> 128 -> 3), GELU, MSE loss. Normalisation:
- shape (9-dim): min-max to [-1, 1] using per-column train-set ranges, so the
  design-vector `x.shape in [-1, 1]` feeds straight in at inference.
- Ma, alt, alpha: plain z-score.
- L: z-score of `log10(L)` (training range is log-uniform on [0.1, 10]).
- CD: target is z-score of `log10(CD)` (spans decades).
- CL, CM: plain z-score.

Data split: 80 / 20 of train.csv, grouped by `geom_name`, so val geometries are
never seen during training. The held-out DeCoDe test.csv is used as a second,
fully-unseen generalisation report.

Outputs:
- `<ckpt-dir>/a_aero.pt`           (best-val weights)
- `<ckpt-dir>/a_aero_norm_stats.json`
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
DEFAULT_CKPT_DIR = Path(__file__).resolve().parents[1] / "checkpoints"

SHAPE_COLS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]
FLIGHT_COLS = ["Ma", "alt", "alpha", "L"]  # L z-scored in log-space
TARGET_COLS = ["CL", "CD", "CM"]           # CD z-scored in log-space


class AAeroMLP(nn.Module):
    def __init__(self, in_dim: int = 13, hidden: int = 128, out_dim: int = 3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _norm_stats(
    df: pd.DataFrame,
) -> dict:
    """Min/max (shape) + mean/std (flight, outputs) from the training rows."""
    shape = df[SHAPE_COLS].to_numpy(dtype=np.float64)
    flight = np.stack(
        [
            df["Ma"].to_numpy(dtype=np.float64),
            df["alt"].to_numpy(dtype=np.float64),
            df["alpha"].to_numpy(dtype=np.float64),
            np.log10(df["L"].to_numpy(dtype=np.float64)),
        ],
        axis=1,
    )
    targets = np.stack(
        [
            df["CL"].to_numpy(dtype=np.float64),
            np.log10(df["CD"].to_numpy(dtype=np.float64)),
            df["CM"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    return {
        "shape_cols": SHAPE_COLS,
        "shape_min": shape.min(axis=0).tolist(),
        "shape_max": shape.max(axis=0).tolist(),
        "flight_cols": FLIGHT_COLS,
        "flight_transform": ["identity", "identity", "identity", "log10"],
        "flight_mean": flight.mean(axis=0).tolist(),
        "flight_std": (flight.std(axis=0) + 1e-12).tolist(),
        "target_cols": TARGET_COLS,
        "target_transform": ["identity", "log10", "identity"],
        "target_mean": targets.mean(axis=0).tolist(),
        "target_std": (targets.std(axis=0) + 1e-12).tolist(),
    }


def _apply_norm(df: pd.DataFrame, stats: dict) -> tuple[np.ndarray, np.ndarray]:
    shape = df[SHAPE_COLS].to_numpy(dtype=np.float64)
    shape_min = np.array(stats["shape_min"])
    shape_max = np.array(stats["shape_max"])
    shape_norm = 2.0 * (shape - shape_min) / (shape_max - shape_min + 1e-12) - 1.0

    flight = np.stack(
        [
            df["Ma"].to_numpy(dtype=np.float64),
            df["alt"].to_numpy(dtype=np.float64),
            df["alpha"].to_numpy(dtype=np.float64),
            np.log10(df["L"].to_numpy(dtype=np.float64)),
        ],
        axis=1,
    )
    flight_norm = (flight - np.array(stats["flight_mean"])) / np.array(stats["flight_std"])

    targets = np.stack(
        [
            df["CL"].to_numpy(dtype=np.float64),
            np.log10(df["CD"].to_numpy(dtype=np.float64)),
            df["CM"].to_numpy(dtype=np.float64),
        ],
        axis=1,
    )
    targets_norm = (targets - np.array(stats["target_mean"])) / np.array(stats["target_std"])

    X = np.concatenate([shape_norm, flight_norm], axis=1).astype(np.float32)
    Y = targets_norm.astype(np.float32)
    return X, Y


def _unnorm_targets(y_norm: np.ndarray, stats: dict) -> np.ndarray:
    """Un-normalise model outputs back to physical (CL, CD, CM)."""
    m = np.array(stats["target_mean"])
    s = np.array(stats["target_std"])
    y = y_norm * s + m
    y_out = y.copy()
    # CD is the log10-transformed column.
    y_out[:, 1] = np.power(10.0, y[:, 1])
    return y_out


def _r2(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0)) ** 2).sum(axis=0) + 1e-12
    return 1.0 - ss_res / ss_tot


def _split_by_geom(df: pd.DataFrame, val_frac: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    geoms = np.array(sorted(df["geom_name"].unique()))
    rng.shuffle(geoms)
    n_val = max(1, int(round(val_frac * len(geoms))))
    val_geoms = set(geoms[:n_val].tolist())
    val_mask = df["geom_name"].isin(val_geoms)
    return df.loc[~val_mask].reset_index(drop=True), df.loc[val_mask].reset_index(drop=True)


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    ckpt_dir: Path = DEFAULT_CKPT_DIR,
    epochs: int = 400,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    device_str: str | None = None,
    smoke: bool = False,
) -> dict:
    device = torch.device(device_str) if device_str else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    if smoke:
        epochs = 5

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_csv = pd.read_csv(data_dir / "aero_targets_train.csv")
    test_csv = pd.read_csv(data_dir / "aero_targets_test.csv")

    train_df, val_df = _split_by_geom(train_csv, val_frac=0.2, seed=seed)
    stats = _norm_stats(train_df)

    X_tr, Y_tr = _apply_norm(train_df, stats)
    X_val, Y_val = _apply_norm(val_df, stats)
    X_te, Y_te = _apply_norm(test_csv, stats)

    X_tr_t = torch.from_numpy(X_tr).to(device)
    Y_tr_t = torch.from_numpy(Y_tr).to(device)
    X_val_t = torch.from_numpy(X_val).to(device)
    Y_val_t = torch.from_numpy(Y_val).to(device)

    model = AAeroMLP(in_dim=X_tr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n_train = X_tr_t.shape[0]
    best_val = math.inf
    best_state: dict | None = None

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total = 0.0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            pred = model(X_tr_t[idx])
            loss = ((pred - Y_tr_t[idx]) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            total += loss.item() * idx.numel()
        sched.step()
        train_loss = total / n_train

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = ((val_pred - Y_val_t) ** 2).mean().item()

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % max(1, epochs // 20) == 0 or epoch == epochs - 1:
            print(f"epoch {epoch:4d}  train {train_loss:.5f}  val {val_loss:.5f}  (best {best_val:.5f})")

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    def _r2_on(X: np.ndarray, Y_norm: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            pred_norm = model(torch.from_numpy(X).to(device)).cpu().numpy()
        true_phys = _unnorm_targets(Y_norm, stats)
        pred_phys = _unnorm_targets(pred_norm, stats)
        return _r2(true_phys, pred_phys)

    r2_train = _r2_on(X_tr, Y_tr)
    r2_val = _r2_on(X_val, Y_val)
    r2_test = _r2_on(X_te, Y_te)

    def _row(name: str, r2: np.ndarray) -> str:
        return f"{name:>5}  CL={r2[0]:+.4f}  CD={r2[1]:+.4f}  CM={r2[2]:+.4f}"

    print()
    print("R^2 (on physical CL, CD, CM):")
    print(_row("train", r2_train))
    print(_row("val", r2_val))
    print(_row("test", r2_test))
    print()
    tgt = {"CL": 0.95, "CD": 0.90, "CM": 0.85}
    ok = all(r2_test[i] >= v for i, v in enumerate(tgt.values()))
    print(f"plan targets (test-split): CL>0.95, CD>0.9, CM>0.85 -> {'PASS' if ok else 'MISS'}")

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "a_aero.pt"
    stats_path = ckpt_dir / "a_aero_norm_stats.json"

    torch.save(
        {
            "state_dict": best_state,
            "arch": {"in_dim": X_tr.shape[1], "hidden": 128, "out_dim": 3},
            "r2_train": r2_train.tolist(),
            "r2_val": r2_val.tolist(),
            "r2_test": r2_test.tolist(),
        },
        ckpt_path,
    )
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\nsaved: {ckpt_path}")
    print(f"saved: {stats_path}")

    return {
        "r2_train": r2_train.tolist(),
        "r2_val": r2_val.tolist(),
        "r2_test": r2_test.tolist(),
        "best_val_loss": best_val,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_CKPT_DIR)
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--smoke", action="store_true", help="5-epoch smoke run")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        ckpt_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device_str=args.device,
        smoke=args.smoke,
    )


if __name__ == "__main__":
    main()
