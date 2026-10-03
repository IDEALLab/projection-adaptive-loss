"""Prepare A_aero training targets from DeCoDe BlendedNet CSVs.

Loads `case_with_geom_params_{train,test}.csv` from clarc_blended_wing_body,
derives the reference length `L` (m) from Re_L, altitude, and Mach via the
shared ISA atmosphere helper, renames columns to the E1 internal convention,
and writes the result to `e1_bwb/data/aero_targets_{train,test}.csv`.

Output columns:
    geom_name, B1, B2, B3, C2, C3, C4, S1, S2, S3, Ma, alt, alpha, L, CL, CD, CM

Notes:
- `alt_kft` -> metres (x 304.8). DeCoDe covers ~[0.75, 12192] m; ISA in
  `atmosphere.py` is only strictly valid below the 11 km tropopause. A handful
  of rows sit above that and use mildly-extrapolated ISA. The benchmark
  altitude range stays inside the valid region.
- `C1` is dropped, it is a fixed 1000-mm normalisation constant in the DeCoDe
  dataset, so it is 9 shape params on the live axis. Matches the 9-dim shape
  slice in `x_layout.DecodedX.shape`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from pal.benchmarks.engineering.e1_bwb import atmosphere

KFT_TO_M = 304.8

# Point --in-dir at the DeCoDe `case_with_geom_params_*.csv` directory.
DECODE_DIR = Path("csv_files")
DEFAULT_OUT_DIR = Path(__file__).resolve().parents[2] / "data"

SHAPE_COLS = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]  # C1 dropped
INPUT_COLS = SHAPE_COLS + ["Ma", "alt", "alpha", "L"]
TARGET_COLS = ["CL", "CD", "CM"]


def _derive_L(Re_L: torch.Tensor, alt_m: torch.Tensor, Ma: torch.Tensor) -> torch.Tensor:
    V = Ma * atmosphere.speed_of_sound(alt_m)
    mu = atmosphere.dynamic_viscosity(alt_m)
    rho = atmosphere.rho_air(alt_m)
    return Re_L * mu / (rho * V)


def prepare_split(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    alt_m = torch.as_tensor(df["alt_kft"].to_numpy(), dtype=torch.float64) * KFT_TO_M
    Re_L = torch.as_tensor(df["Re_L"].to_numpy(), dtype=torch.float64)
    Ma = torch.as_tensor(df["M_inf"].to_numpy(), dtype=torch.float64)
    L = _derive_L(Re_L, alt_m, Ma)

    out = pd.DataFrame({"geom_name": df["geom_name"].values})
    for col in SHAPE_COLS:
        out[col] = df[col].values
    out["Ma"] = df["M_inf"].values
    out["alt"] = alt_m.numpy()
    out["alpha"] = df["alpha_deg"].values
    out["L"] = L.numpy()
    out["CL"] = df["CL"].values
    out["CD"] = df["CD"].values
    out["CM"] = df["CMy"].values
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-dir", type=Path, default=DECODE_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "test"):
        src = args.in_dir / f"case_with_geom_params_{split}.csv"
        dst = args.out_dir / f"aero_targets_{split}.csv"
        df = prepare_split(src)
        df.to_csv(dst, index=False)
        L_min, L_max = df["L"].min(), df["L"].max()
        print(
            f"{split}: {len(df):>5d} rows  ->  {dst}\n"
            f"        L in [{L_min:.3f}, {L_max:.3f}] m   "
            f"alt in [{df['alt'].min():.1f}, {df['alt'].max():.1f}] m   "
            f"Ma in [{df['Ma'].min():.3f}, {df['Ma'].max():.3f}]   "
            f"alpha in [{df['alpha'].min():.2f}, {df['alpha'].max():.2f}] deg"
        )


if __name__ == "__main__":
    main()
