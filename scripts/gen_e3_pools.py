"""Generate e3 MATLAB-accepted load pools.

Writes the pool files into the package-data directory.

Usage
-----
    python scripts/gen_e3_pools.py --case ieee30
    python scripts/gen_e3_pools.py --case ieee57 --verify
    python scripts/gen_e3_pools.py --all
    python scripts/gen_e3_pools.py --all --verify --use-numba
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _data_dir() -> Path:
    from pal.benchmarks.engineering.e3_acopf import data as _data_pkg
    return Path(_data_pkg.__file__).resolve().parent


def _run(case: str, *, use_numba: bool, verify: bool) -> None:
    from pal.benchmarks.engineering.e3_acopf.grid_adapter import (
        pandapower_to_ml4opf,
    )
    from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
        _pool_filename,
        describe_pool,
        generate_matlab_accepted_pool,
        load_pool,
        save_pool,
    )

    print(f"\n=== {case}: generating MATLAB-accepted pool ===")
    t0 = time.monotonic()
    pool = generate_matlab_accepted_pool(case, use_numba=use_numba, verbose=True)
    dt = time.monotonic() - t0
    print(f"  elapsed: {dt:.1f}s")

    out = _data_dir() / _pool_filename(case)
    save_pool(pool, out)
    print(f"  wrote {out.relative_to(Path.cwd())}  ({out.stat().st_size / 1024:.1f} KB)")

    print("\n" + describe_pool(pool))

    if verify:
        print("\n  verifying reload + fingerprint match against live grid data...")
        live = pandapower_to_ml4opf(case)
        reloaded = load_pool(case, live_data=live)
        assert reloaded["pd"].shape == pool["pd"].shape
        assert reloaded["metadata"]["case_fingerprint"] == pool["metadata"]["case_fingerprint"]
        print("  verify: OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--case", choices=["ieee30", "ieee57", "ieee118"])
    parser.add_argument("--all", action="store_true",
                        help="generate all three cases")
    parser.add_argument("--verify", action="store_true",
                        help="reload + fingerprint-check each pool after writing")
    parser.add_argument("--use-numba", action="store_true",
                        help="enable pandapower's numba acceleration (requires numba)")
    args = parser.parse_args(argv)

    if args.all and args.case:
        parser.error("pass --case or --all, not both")
    if not args.all and not args.case:
        parser.error("one of --case / --all is required")

    cases = ("ieee30", "ieee57", "ieee118") if args.all else (args.case,)
    for c in cases:
        _run(c, use_numba=args.use_numba, verify=args.verify)
    return 0


if __name__ == "__main__":
    sys.exit(main())
