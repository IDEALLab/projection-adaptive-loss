#!/usr/bin/env python3
"""Freeze the e2 IPOPT campaign's eval queries to per-shard eval-points JSON.

`--mode replicate` (default) reproduces the eval_queries RNG without loading the
surrogate, `--mode bench` loads the real benchmark.

Usage:
    python scripts/gen_e2_ipopt_points.py --seed 0 --n-eval 64 --pack 1 \
        --out $SCRATCH/pal_engineering/e2_ipopt_x86/points
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

# Must match pal/benchmarks/engineering/e2_urban_wind/benchmark.py.
_E2_ZETA_DIM = 16
_E2_EVAL_SEED_OFFSET = 7919
_E2_EVAL_SEED_STRIDE = 1000

# Echoed into each JSON; the solver only honors max_iter/tol/multi_start/device.
_IPOPT_BLOCK = {
    "n_multistarts": 2,
    "max_iter": 500,
    "tol": 1e-6,
    "constr_viol_tol": 1e-4,
    "mu_strategy": "adaptive",
    "hessian_approximation": "limited-memory",
}


def _zetas_replicate(seed: int, n: int):
    import torch

    g = torch.Generator("cpu").manual_seed(
        int(seed) * _E2_EVAL_SEED_STRIDE + _E2_EVAL_SEED_OFFSET
    )
    return torch.randn(n, _E2_ZETA_DIM, generator=g).tolist()


def _zetas_bench(seed: int, n: int):
    from pal.benchmarks import registry as bench_registry

    bench = bench_registry.get("e2/urban_wind", device="cpu")
    q = bench.eval_queries(seed, n=n)
    return q.zeta.detach().cpu().tolist()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-eval", type=int, default=64)
    ap.add_argument("--pack", type=int, default=1,
                    help="queries per shard file (=> ceil(n_eval/pack) shards)")
    ap.add_argument("--mode", choices=["replicate", "bench"], default="replicate")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    zetas = (
        _zetas_bench(args.seed, args.n_eval)
        if args.mode == "bench"
        else _zetas_replicate(args.seed, args.n_eval)
    )
    assert len(zetas) == args.n_eval, (len(zetas), args.n_eval)

    args.out.mkdir(parents=True, exist_ok=True)
    n_shards = math.ceil(args.n_eval / args.pack)
    written = []
    for s in range(n_shards):
        q_lo = s * args.pack
        q_hi = min((s + 1) * args.pack, args.n_eval)
        points = [{"zeta": zetas[i], "condition": []} for i in range(q_lo, q_hi)]
        blob = {
            "benchmark": "e2",
            "frozen_at": "e2-ipopt-x86-campaign",
            "rationale": (
                f"e2 IPOPT parity: bench.eval_queries(seed={args.seed}, "
                f"n={args.n_eval}) rows [{q_lo}:{q_hi}] (mode={args.mode})."
            ),
            "meta": {
                "seed": args.seed,
                "n_eval": args.n_eval,
                "pack": args.pack,
                "shard": s,
                "query_indices": list(range(q_lo, q_hi)),
                "mode": args.mode,
            },
            "points": points,
            "ipopt": _IPOPT_BLOCK,
        }
        path = args.out / f"seed{args.seed}_shard{s:03d}.json"
        path.write_text(json.dumps(blob, indent=2))
        written.append(path)

    print(f"wrote {len(written)} shard file(s) to {args.out} "
          f"(n_eval={args.n_eval}, pack={args.pack}, mode={args.mode})")
    print(f"n_shards={n_shards}  -> submit with --array=0-{n_shards - 1}")


if __name__ == "__main__":
    main()
