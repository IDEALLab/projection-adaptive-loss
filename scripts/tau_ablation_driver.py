"""tau ablation driver for the curvature_warp warp-control probe (pal_loggap, only `tau` varied).

Arms: `loose` sets tau via --set. `lossoff` sets tau=1e10 and monkeypatches `_W_FLOOR = 0`,
since the default 1e-6 floor would leak constraint pressure. Each run is its own subprocess.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

#: Other hparams are bench defaults. train_dataset_size=1000 is not a default, so it is pinned.
BASE_ARGS = [
    "run",
    "--method", "pal_loggap",
    "--seeds", "0",
    "--epochs", "2000",
    "--device", "cpu",
    "--n-eval", "512",
    "--set", "train_dataset_size=1000",
]


def _exec(argv: list[str], patch_floor: bool) -> int:
    if patch_floor:
        import pal.method.loggap.loss as loss_mod

        loss_mod._W_FLOOR = 0.0
        assert loss_mod._W_FLOOR == 0.0
    from pal.runner.cli import main as cli_main

    return int(cli_main(argv) or 0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("loose", "lossoff"), default=None)
    ap.add_argument("--tau", type=float, default=None,
                    help="tau for the `loose` arm; ignored for `lossoff`")
    ap.add_argument("--variants", default="k0,k6,k10")
    ap.add_argument("--runs-root", default=None)
    ap.add_argument("--exec", dest="exec_argv", nargs=argparse.REMAINDER,
                    default=None, help="internal: run cli.main with these args")
    ap.add_argument("--patch-floor", action="store_true",
                    help="internal: zero `_W_FLOOR` before running")
    args = ap.parse_args()

    if args.exec_argv is not None:
        raise SystemExit(_exec(args.exec_argv, args.patch_floor))

    if args.arm is None or args.runs_root is None:
        ap.error("--arm and --runs-root are required")
    tau = 1.0e10 if args.arm == "lossoff" else args.tau
    if tau is None:
        ap.error("--tau is required for the `loose` arm")
    patch = args.arm == "lossoff"

    for variant in args.variants.split(","):
        bench_id = f"curvature_warp_{variant}"
        cli_argv = [
            *BASE_ARGS,
            "--benchmarks", bench_id,
            "--runs-root", args.runs_root,
            "--set", f"tau={tau:.6e}",
            "--set", f"ablation_name=curvature_warp_tau_{args.arm}",
            "--set", f"scenario_label=tau_{tau:.0e}",
        ]
        cmd = [
            sys.executable, str(Path(__file__).resolve()),
            *(["--patch-floor"] if patch else []),
            "--exec", *cli_argv,
        ]
        print(f"[driver] {args.arm} {variant} tau={tau:g} patch_floor={patch}",
              flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            raise SystemExit(f"[driver] run failed for {bench_id} (rc={rc})")


if __name__ == "__main__":
    main()
