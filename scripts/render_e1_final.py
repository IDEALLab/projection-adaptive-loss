#!/usr/bin/env python
"""Offline renderer for the e1 hero composite from a `PAL_VIZ_FINAL_DUMP=1` data dump.

Usage: python scripts/render_e1_final.py --data-pt viz_final_data_<id>.pt --out-dir DIR
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _ensure_pal_on_path() -> None:
    repo = Path(__file__).resolve().parent.parent
    sp = str(repo)
    if sp not in sys.path:
        sys.path.insert(0, sp)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-pt", required=True, type=Path,
                        help="Path to the viz_final_data .pt dump")
    parser.add_argument("--out-dir", required=True, type=Path,
                        help="Directory to write hero.{png,pdf} into")
    parser.add_argument("--open", action="store_true",
                        help="Run `open <out-dir>` after rendering (macOS)")
    args = parser.parse_args()

    if not args.data_pt.exists():
        parser.error(f"--data-pt does not exist: {args.data_pt}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _ensure_pal_on_path()

    import torch

    from pal.benchmarks.engineering.e1_bwb.benchmark import (
        VIZ_FINAL_DATA_SCHEMA,
    )
    from pal.benchmarks.engineering.e1_bwb.viz import render_hero_from_data

    data = torch.load(args.data_pt, weights_only=False, map_location="cpu")
    schema = data.get("schema")
    if schema != VIZ_FINAL_DATA_SCHEMA:
        print(
            f"WARNING: dump schema {schema!r} != expected "
            f"{VIZ_FINAL_DATA_SCHEMA!r}; proceeding anyway",
            file=sys.stderr,
        )

    fig = render_hero_from_data(data)
    if fig is None:
        print(
            "ERROR: render_hero_from_data returned None.",
            file=sys.stderr,
        )
        return 2

    png = args.out_dir / "hero.png"
    pdf = args.out_dir / "hero.pdf"
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    print(f"wrote {png}")
    print(f"wrote {pdf}")

    if args.open:
        subprocess.run(["open", str(args.out_dir)], check=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
