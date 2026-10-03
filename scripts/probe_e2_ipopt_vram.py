#!/usr/bin/env python3
"""VRAM and latency probe for the e2 IPOPT campaign (peak memory, forward and Jacobian wall).

Usage: PAL_IPOPT_JAC_MODE=loop python scripts/probe_e2_ipopt_vram.py [--device cuda]
"""

from __future__ import annotations

import argparse
import os
import time

# Force loop mode: jacrev crashes on the checkpointed surrogate.
os.environ.setdefault("PAL_IPOPT_JAC_MODE", "loop")
# Probe at K=49 (per-pair clearance): loop-Jacobian cost scales linearly in K.
os.environ.setdefault("E2_CLEARANCE_PER_PAIR", "1")

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--forwards", type=int, default=5)
    ap.add_argument("--jacs", type=int, default=3)
    ap.add_argument("--max-iter", type=int, default=500,
                    help="only used to project a per-solve wall estimate")
    args = ap.parse_args()

    from pal.baselines.nlp_adapter import _JAC_MODE, NLPView
    from pal.benchmarks import registry as bench_registry

    print(f"[probe] jac_mode={_JAC_MODE} device={args.device} "
          f"torch={torch.__version__} cuda_avail={torch.cuda.is_available()}")
    assert _JAC_MODE == "loop", "set PAL_IPOPT_JAC_MODE=loop before running"

    t0 = time.perf_counter()
    bench = bench_registry.get("e2/urban_wind", device=args.device)
    print(f"[probe] bench+surrogate loaded in {time.perf_counter() - t0:.1f}s "
          f"dim={bench.spec.dim} K={len(bench.spec.constraint_types)}")

    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    view = NLPView(bench, conditions=None, device=args.device)
    rng = np.random.default_rng(0)
    x = rng.uniform(view.lo, view.hi)

    # warm-up (kernel autotune / lazy init), excluded from timing
    _ = view.f(x)
    _ = view.jac_g(x)

    fwd_t = []
    for _ in range(args.forwards):
        xi = rng.uniform(view.lo, view.hi)
        view.reset_diag()
        t = time.perf_counter()
        view.f(xi)  # triggers one surrogate forward (fresh x => cache miss)
        _sync(args.device)
        fwd_t.append(time.perf_counter() - t)

    jac_t = []
    K = view.n_constraints
    for _ in range(args.jacs):
        xi = rng.uniform(view.lo, view.hi)
        view.reset_diag()
        t = time.perf_counter()
        view.jac_g(xi)  # forward + K sequential backward passes (loop mode)
        _sync(args.device)
        jac_t.append(time.perf_counter() - t)

    peak_gb = (
        torch.cuda.max_memory_allocated() / 1e9
        if (args.device == "cuda" and torch.cuda.is_available())
        else float("nan")
    )
    fwd_mean = float(np.mean(fwd_t))
    jac_mean = float(np.mean(jac_t))
    per_row = jac_mean / max(K, 1)

    print("\n=== e2 IPOPT probe ===")
    print(f"peak_gpu_mem_GB:      {peak_gb:.2f}")
    print(f"per_forward_s:        {fwd_mean:.3f}  (n={args.forwards})")
    print(f"per_jac_call_s:       {jac_mean:.3f}  (forward + {K} backward rows)")
    print(f"per_jac_row_s:        {per_row:.4f}")
    # Rough per-iteration cost: one Jacobian call plus one forward.
    iter_cost = jac_mean + fwd_mean
    est_solve_h = iter_cost * args.max_iter / 3600.0
    print(f"est_iter_cost_s:      {iter_cost:.3f}")
    print(f"est_solve_wall_h@{args.max_iter}: {est_solve_h:.2f}  "
          f"(upper bound; real solves often stop < max_iter)")
    if peak_gb == peak_gb:  # not NaN
        print(f"suggested PACK (24 GB / peak, capped 4): "
              f"{max(1, min(4, int(22.0 / max(peak_gb, 0.1))))}")


def _sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
