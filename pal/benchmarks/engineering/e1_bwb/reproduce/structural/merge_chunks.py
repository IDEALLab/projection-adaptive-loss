"""Merge chunk CSVs from parallel data generation into single files.

Concatenates chunk_*.csv and chunk_*_ribs.csv across job dirs,
validates data quality, and exports CSV + Parquet.

Usage:
    python -m pal.benchmarks.engineering.e1_bwb.reproduce.structural.merge_chunks \
        --jobs <job_id> [<job_id> ...] [--out <dir>]
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_ROOT = Path("structural_surrogate_data")

PROP_COLS = [
    "A", "u_cg", "v_cg", "I_uu", "I_vv", "I_uv",
    "I_1", "I_2", "Q_u_max", "Q_v_max", "J",
]


def find_chunks(job_ids: list[int], data_root: Path, suffix: str = ".csv") -> list[Path]:
    """Find all chunk CSVs matching suffix across job dirs."""
    chunks = []
    for jid in job_ids:
        job_dir = data_root / f"job_{jid}"
        if not job_dir.exists():
            print(f"WARNING: {job_dir} not found, skipping")
            continue
        pattern = f"chunk_*{suffix}"
        found = sorted(job_dir.glob(pattern))
        chunks.extend(found)
    return chunks


def merge_csvs(paths: list[Path]) -> pd.DataFrame:
    """Read and concatenate CSVs, skipping duplicate headers."""
    dfs = []
    for p in paths:
        df = pd.read_csv(p)
        if len(df) > 0:
            dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def validate_struct(df: pd.DataFrame) -> list[str]:
    """Validate structural properties dataframe. Returns list of warnings."""
    warnings = []

    # Duplicate sample_id + thickness_id + y combos
    key_cols = ["sample_id", "thickness_id", "y"]
    dupes = df.duplicated(subset=key_cols, keep=False)
    n_dupes = dupes.sum()
    if n_dupes > 0:
        warnings.append(f"  {n_dupes} duplicate (sample_id, thickness_id, y) rows")

    # NaN/inf in property columns
    for col in PROP_COLS:
        if col not in df.columns:
            warnings.append(f"  missing column: {col}")
            continue
        n_nan = df[col].isna().sum()
        n_inf = np.isinf(df[col]).sum()
        if n_nan > 0:
            warnings.append(f"  {col}: {n_nan} NaN values")
        if n_inf > 0:
            warnings.append(f"  {col}: {n_inf} inf values")

    # Zero area (stations past wingtip)
    if "A" in df.columns:
        n_zero = (df["A"].abs() < 1e-12).sum()
        if n_zero > 0:
            warnings.append(f"  {n_zero} rows with A~0 (stations past wingtip?)")

    return warnings


def validate_ribs(df: pd.DataFrame) -> list[str]:
    """Validate ribs dataframe."""
    warnings = []

    key_cols = ["sample_id", "rib_combo_id"]
    dupes = df.duplicated(subset=key_cols, keep=False)
    if dupes.sum() > 0:
        warnings.append(f"  {dupes.sum()} duplicate (sample_id, rib_combo_id) rows")

    if "V_ribs" in df.columns:
        n_nan = df["V_ribs"].isna().sum()
        n_neg = (df["V_ribs"] < 0).sum()
        if n_nan > 0:
            warnings.append(f"  V_ribs: {n_nan} NaN values")
        if n_neg > 0:
            warnings.append(f"  V_ribs: {n_neg} negative values")

    return warnings


def print_stats(df: pd.DataFrame, name: str) -> None:
    """Print summary statistics."""
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Rows:              {len(df):,}")
    print(f"  Unique sample_ids: {df['sample_id'].nunique():,}")

    if "y" in df.columns:
        stations_per = df.groupby("sample_id")["y"].nunique()
        print(f"  Stations/geometry: {stations_per.min()}-{stations_per.max()}")
    if "thickness_id" in df.columns:
        thick_per = df.groupby("sample_id")["thickness_id"].nunique()
        print(f"  Thickness combos:  {thick_per.min()}-{thick_per.max()}")

    # Parameter ranges
    param_cols = [c for c in df.columns if c not in
                  ["sample_id", "thickness_id", "rib_combo_id"] + PROP_COLS + ["y"]]
    print("\n  Parameter ranges:")
    for col in param_cols:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            print(f"    {col:20s}  [{df[col].min():.6g}, {df[col].max():.6g}]")

    # Property ranges
    props_in_df = [c for c in PROP_COLS if c in df.columns]
    if props_in_df:
        print("\n  Property ranges:")
        for col in props_in_df:
            print(f"    {col:10s}  [{df[col].min():.6g}, {df[col].max():.6g}]")


def main():
    parser = argparse.ArgumentParser(description="Merge structural surrogate data chunks")
    parser.add_argument(
        "--jobs", type=int, nargs="+", required=True,
        help="SLURM job IDs to merge (space-separated)",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output directory (default: DATA_ROOT/merged/)",
    )
    parser.add_argument(
        "--data-root", type=str, default=str(DATA_ROOT),
        help=f"Root dir containing job_* folders (default: {DATA_ROOT})",
    )
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out) if args.out else data_root / "merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root: {data_root}")
    print(f"Output:    {out_dir}")
    print(f"Jobs:      {args.jobs}")

    # Structural properties
    struct_chunks = find_chunks(args.jobs, data_root, suffix=".csv")
    # Exclude ribs files
    struct_chunks = [p for p in struct_chunks if "_ribs" not in p.name]
    print(f"\nFound {len(struct_chunks)} structural chunk files")

    if struct_chunks:
        df_struct = merge_csvs(struct_chunks)
        print(f"Merged: {len(df_struct):,} rows")

        warnings = validate_struct(df_struct)
        if warnings:
            print("\nWARNINGS (structural):")
            for w in warnings:
                print(w)
        else:
            print("Validation: OK")

        print_stats(df_struct, "Structural Properties")

        # Write
        csv_path = out_dir / "structural.csv"
        pq_path = out_dir / "structural.parquet"
        df_struct.to_csv(csv_path, index=False)
        df_struct.to_parquet(pq_path, index=False)
        print(f"\n  Saved: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Saved: {pq_path} ({pq_path.stat().st_size / 1e6:.1f} MB)")

    # Ribs
    ribs_chunks = find_chunks(args.jobs, data_root, suffix="_ribs.csv")
    print(f"\nFound {len(ribs_chunks)} ribs chunk files")

    if ribs_chunks:
        df_ribs = merge_csvs(ribs_chunks)
        print(f"Merged: {len(df_ribs):,} rows")

        warnings = validate_ribs(df_ribs)
        if warnings:
            print("\nWARNINGS (ribs):")
            for w in warnings:
                print(w)
        else:
            print("Validation: OK")

        print_stats(df_ribs, "Rib Volumes")

        csv_path = out_dir / "ribs.csv"
        pq_path = out_dir / "ribs.parquet"
        df_ribs.to_csv(csv_path, index=False)
        df_ribs.to_parquet(pq_path, index=False)
        print(f"\n  Saved: {csv_path} ({csv_path.stat().st_size / 1e6:.1f} MB)")
        print(f"  Saved: {pq_path} ({pq_path.stat().st_size / 1e6:.1f} MB)")

    print("\nDone.")


if __name__ == "__main__":
    main()
