#!/usr/bin/env python3
"""
Prepare local BlendedNet train/test assets for FiLM training.

This script:
1. Builds `csv_files/case_with_geom_params_{train,test}.csv` from the raw
   `geom_params.ini` + `case_data.dat` files.
2. Builds `data/surface_data.hdf5` and `data/surface_data_test.hdf5` from the
   raw VTK folders.
"""

from __future__ import annotations

import argparse
import configparser
import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_CSV_DIR = REPO_ROOT / "csv_files"
HDF5_BUILDER = REPO_ROOT / "create_hdf5.py"

CASE_COLUMNS = [
    "case_name",
    "geom_name",
    "mesh",
    "B1",
    "B2",
    "B3",
    "C1",
    "C2",
    "C3",
    "C4",
    "S1",
    "S2",
    "S3",
    "alt_kft",
    "Re_L",
    "M_inf",
    "alpha_deg",
    "beta_deg",
    "CD",
    "CL",
    "CMy",
]


def load_geom_params(ini_path: Path) -> pd.DataFrame:
    parser = configparser.ConfigParser()
    parser.optionxform = str
    with ini_path.open("r", encoding="utf-8") as f:
        parser.read_file(f)

    rows = []
    for section in parser.sections():
        row = {"geom_name": section}
        for key, value in parser.items(section):
            row[key] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def load_case_data(dat_path: Path) -> pd.DataFrame:
    with dat_path.open("r", encoding="utf-8") as f:
        header = f.readline().lstrip("#").strip().split()
    df = pd.read_csv(dat_path, sep=r"\s+", names=header, skiprows=1, engine="python")
    return df.rename(columns={"Re": "Re_L"})


def build_case_csv(split_dir: Path, out_csv: Path) -> None:
    geom_df = load_geom_params(split_dir / "geom_params.ini")
    case_df = load_case_data(split_dir / "case_data.dat")
    merged = case_df.merge(geom_df, on="geom_name", how="inner", validate="many_to_one")
    merged["mesh"] = merged["geom_name"]
    merged = merged[CASE_COLUMNS].copy()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    print(f"Wrote CSV: {out_csv} ({len(merged)} rows)")


def build_hdf5(vtk_dir: Path, out_h5: Path) -> None:
    case_files = sorted(vtk_dir.glob("case_*.vtk"))
    if not case_files:
        raise FileNotFoundError(f"No case_*.vtk files found in {vtk_dir}")

    if out_h5.exists():
        out_h5.unlink()

    def run_batch(offset: int, limit: int, append: bool) -> None:
        cmd = [
            sys.executable,
            str(HDF5_BUILDER),
            "--data-dir",
            str(vtk_dir),
            "--out-h5",
            str(out_h5),
            "--offset",
            str(offset),
            "--limit",
            str(limit),
        ]
        if append:
            cmd.append("--append")
        print("Running:", " ".join(cmd))
        subprocess.run(cmd, check=True)

    batch_size = 500
    append = False
    for offset in range(0, len(case_files), batch_size):
        limit = min(batch_size, len(case_files) - offset)
        try:
            run_batch(offset, limit, append=append)
            append = True
        except subprocess.CalledProcessError:
            print(f"Batch starting at index {offset} failed; retrying one case at a time.")
            for local_idx in range(limit):
                try:
                    run_batch(offset + local_idx, 1, append=append)
                    append = True
                except subprocess.CalledProcessError:
                    bad_case = case_files[offset + local_idx].name
                    print(f"Skipping unreadable VTK: {bad_case}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare local BlendedNet CSV/HDF5 assets.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--csv-dir", type=Path, default=DEFAULT_CSV_DIR)
    parser.add_argument("--skip-csv", action="store_true")
    parser.add_argument("--skip-hdf5", action="store_true")
    args = parser.parse_args()

    train_dir = args.data_root / "train"
    test_dir = args.data_root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError(f"Expected train/test folders under {args.data_root}")

    if not args.skip_csv:
        build_case_csv(train_dir, args.csv_dir / "case_with_geom_params_train.csv")
        build_case_csv(test_dir, args.csv_dir / "case_with_geom_params_test.csv")

    if not args.skip_hdf5:
        build_hdf5(train_dir / "vtk", args.data_root / "surface_data.hdf5")
        build_hdf5(test_dir / "vtk", args.data_root / "surface_data_test.hdf5")


if __name__ == "__main__":
    main()
