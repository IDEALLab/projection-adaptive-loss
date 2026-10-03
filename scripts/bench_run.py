#!/usr/bin/env python3
"""Wrapper for clean CPU timing runs: pins threads, forces CPU, then calls the `pal` CLI.

Usage:
    python scripts/bench_run.py run --method alm --benchmarks s1_sphere_track ...
    python scripts/bench_run.py eval --run-id <id> ...
    BENCH_THREADS=2 python scripts/bench_run.py run ...   # override the thread pin
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_THREADS = os.environ.get("BENCH_THREADS", "1")

# Set BEFORE importing torch so OpenMP/MKL/OpenBLAS pick up the pin.
for _var in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_var] = _THREADS
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # no CUDA visible -> CPU only
os.environ["TQDM_DISABLE"] = "1"  # silence progress bars
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ.setdefault("PYTHONUNBUFFERED", "1")

import torch  # noqa: E402  (imported after the env pin on purpose)

torch.set_num_threads(int(_THREADS))
try:
    torch.set_num_interop_threads(int(_THREADS))
except RuntimeError:
    pass

logging.disable(logging.WARNING)


def main() -> int:
    from pal.runner.cli import main as cli_main

    argv = list(sys.argv[1:])
    # argparse: last flag wins, so this overrides any earlier --device.
    if argv and argv[0] in {"run", "eval"}:
        argv = argv + ["--device", "cpu"]
    return cli_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
