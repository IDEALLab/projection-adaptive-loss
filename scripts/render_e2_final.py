#!/usr/bin/env python
"""Offline renderer for the e2 hero plots from a `PAL_VIZ_FINAL_DUMP=1` data dump.

Usage: python scripts/render_e2_final.py --data-pt viz_final_data_<id>.pt --out-dir DIR
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ensure_pal_on_path() -> None:
    repo = Path(__file__).resolve().parent.parent
    for p in (repo, repo / "pal" / "benchmarks" / "engineering" / "e2_urban_wind"):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-pt", required=True, type=Path,
                        help="Path to the viz_final_data .pt dump")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to write 3d_wind and 3d_comfort into")
    parser.add_argument("--open", action="store_true",
                        help="Run `open <out-dir>` after rendering (macOS)")
    args = parser.parse_args()

    if not args.data_pt.exists():
        parser.error(f"--data-pt does not exist: {args.data_pt}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_pal_on_path()

    import torch

    from pal.benchmarks.engineering.e2_urban_wind.benchmark import (
        VIZ_FINAL_DATA_SCHEMA,
        _render_final_pyvista,
    )

    data = torch.load(args.data_pt, weights_only=False, map_location="cpu")
    schema = data.get("schema")
    if schema != VIZ_FINAL_DATA_SCHEMA:
        print(
            f"WARNING: dump schema {schema!r} != expected "
            f"{VIZ_FINAL_DATA_SCHEMA!r}; proceeding anyway",
            file=sys.stderr,
        )

    render_kwargs = dict(
        program_config=data["program"]["config"],
        program_params=data["program"]["params"],
        program_batch_size=data["program"]["batch_size"],
        tail_mean_speed=data["tail_mean_speed"],
        tail_mean_u=data["tail_mean_u"],
        tail_mean_v=data["tail_mean_v"],
        building_mask=data["building_mask"],
        decoded_cx=data["decoded_cx"],
        decoded_cy=data["decoded_cy"],
        decoded_w=data["decoded_w"],
        decoded_d=data["decoded_d"],
        decoded_h=data["decoded_h"],
        svf_mean=data["svf_mean"],
        hard_danger_fraction=data["hard_danger_fraction"],
        constraint_vec=data["constraint_vec"],
        total_volume=data["total_volume"],
    )
    figs = _render_final_pyvista(**render_kwargs)
    if figs is None:
        print(
            "ERROR: _render_final_pyvista returned None, pyvista/vtk/"
            "matplotlib not importable in this environment.",
            file=sys.stderr,
        )
        return 2

    for name, fig in figs.items():
        png = args.out_dir / f"{name}.png"
        pdf = args.out_dir / f"{name}.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        print(f"wrote {png}")
        print(f"wrote {pdf}")

    if args.open:
        subprocess.run(["open", str(args.out_dir)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
