"""E1 BWB bottleneck profiler.

Two modes, both emit a JSON report so runs on different hardware can be
diffed directly:

  --mode=timing    per-region wall-clock + (on CUDA) peak memory during fwd+bwd.
                   Instruments: build_bwb_program, program.isocontour,
                   FiLMLoads._film_forward, per-surrogate callables, backward.

  --mode=bf16-grad fp32 reference vs `torch.autocast(bf16)` gradient cosine
                   similarity w.r.t. x. Uses a scalar test loss
                   L = obj.sum() + sum(c.value.sum() for c in constraints).

  --mode=both      run both (default).

Output: JSON dict on stdout + optional --out path. Shape:
    {
      "config":  {...},
      "timing":  {region_name: {"n_calls": int, "total_s": float,
                                "peak_mem_delta_mb": float | null}, ...},
      "peak":    {"fwd_mb": ..., "bwd_mb": ...},          # CUDA only
      "bf16":    {"obj": {"cos_min": ..., "cos_mean": ...},
                  "loss": {"cos_min": ..., "cos_mean": ...}},
    }
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM
from pal.benchmarks.engineering.e1_bwb.x_layout import default_bounds


# Timing plumbing
@dataclass
class Region:
    n_calls: int = 0
    total_s: float = 0.0
    peak_mem_delta_mb: float | None = None  # CUDA only


@dataclass
class Timings:
    regions: dict[str, Region] = field(default_factory=lambda: defaultdict(Region))
    use_cuda: bool = False

    def record(self, name: str, elapsed: float, mem_delta_mb: float | None) -> None:
        r = self.regions[name]
        r.n_calls += 1
        r.total_s += elapsed
        if mem_delta_mb is not None:
            if r.peak_mem_delta_mb is None:
                r.peak_mem_delta_mb = mem_delta_mb
            else:
                r.peak_mem_delta_mb = max(r.peak_mem_delta_mb, mem_delta_mb)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, r in self.regions.items():
            out[name] = {
                "n_calls": r.n_calls,
                "total_s": round(r.total_s, 6),
                "avg_s": round(r.total_s / max(r.n_calls, 1), 6),
                "peak_mem_delta_mb": (
                    round(r.peak_mem_delta_mb, 2) if r.peak_mem_delta_mb is not None else None
                ),
            }
        return out


@contextlib.contextmanager
def timed(timings: Timings, name: str):
    if timings.use_cuda:
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated() / (1024 ** 2)
        # Reset peak within this region so nested calls don't pollute siblings.
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    try:
        yield
    finally:
        if timings.use_cuda:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
            mem_delta = peak - mem_before
        else:
            mem_delta = None
        timings.record(name, time.perf_counter() - t0, mem_delta)


# Monkey-patch helpers, wrap bench / loads surfaces in-place.
def _wrap_callable(obj: Any, attr: str, name: str, timings: Timings) -> None:
    """Wrap `obj.attr` (a callable) with a timing region named `name`.

    `obj` may be the bench itself or a nn.Module, we replace the bound
    attribute, not the class method, so other benches aren't affected.
    """
    original = getattr(obj, attr)

    def wrapped(*args, **kwargs):
        with timed(timings, name):
            return original(*args, **kwargs)

    setattr(obj, attr, wrapped)


def _wrap_method(cls: type, attr: str, name: str, timings: Timings):
    """Wrap an unbound method on a class. Returns the original for restore."""
    original = getattr(cls, attr)

    def wrapped(self, *args, **kwargs):
        with timed(timings, name):
            return original(self, *args, **kwargs)

    setattr(cls, attr, wrapped)
    return original


def instrument_bench(bench: E1BWB, timings: Timings) -> list[tuple[type, str, Callable]]:
    """Wrap every expensive surface on the bench. Returns restore list."""
    # Surrogates, per-instance wraps (callables stored on the bench).
    _wrap_callable(bench, "_compute_aero", "aero", timings)
    _wrap_callable(bench, "_compute_loads", "loads_total", timings)
    _wrap_callable(bench, "_compute_structural", "structural", timings)
    _wrap_callable(bench, "_compute_stress", "stress", timings)

    # build_bwb_program, module-level; patch the function on the module.
    from pal.benchmarks.engineering.e1_bwb import loads as loads_mod
    original_build = loads_mod.build_bwb_program

    def wrapped_build(*args, **kwargs):
        with timed(timings, "build_bwb_program"):
            return original_build(*args, **kwargs)

    # benchmark.py imports build_bwb_program via "from .loads import ...",
    # so we patch both the origin AND benchmark.py's binding.
    loads_mod.build_bwb_program = wrapped_build
    from pal.benchmarks.engineering.e1_bwb import benchmark as bench_mod
    bench_mod.build_bwb_program = wrapped_build

    # FiLMLoads internals, patch the class method so we see per-slice cost.
    from pal.benchmarks.engineering.e1_bwb.loads import FiLMLoads
    restores: list[tuple[type, str, Callable]] = []
    restores.append((FiLMLoads, "_film_forward",
                     _wrap_method(FiLMLoads, "_film_forward", "film_forward", timings)))

    # isocontour is a method on CADProgram instances created by build_bwb_program.
    # We can't patch cleanly at class import time (geometry is lazy-imported),
    # so patch the FiLMLoads.__call__ to wrap program.isocontour per call.
    orig_loads_call = FiLMLoads.__call__

    def wrapped_loads_call(self, program, x, conditions, y_stations):
        if program is not None and not getattr(program.isocontour, "_is_timed", False):
            original_iso = program.isocontour

            def wrapped_iso(*args, **kwargs):
                with timed(timings, "isocontour"):
                    return original_iso(*args, **kwargs)

            wrapped_iso._is_timed = True  # type: ignore[attr-defined]
            program.isocontour = wrapped_iso  # type: ignore[method-assign]
        return orig_loads_call(self, program, x, conditions, y_stations)

    restores.append((FiLMLoads, "__call__", orig_loads_call))
    FiLMLoads.__call__ = wrapped_loads_call  # type: ignore[method-assign]

    # Store build-function restore hook.
    restores.append((loads_mod, "build_bwb_program", original_build))
    restores.append((bench_mod, "build_bwb_program", original_build))
    return restores


def restore_bench(restores: list[tuple[Any, str, Any]]) -> None:
    for target, attr, original in restores:
        setattr(target, attr, original)


# Sample builders
def build_inputs(B: int, seed: int, device: torch.device) -> tuple[Tensor, Tensor]:
    lo, hi = default_bounds(dtype=torch.float32)
    mid = 0.5 * (lo + hi)
    g = torch.Generator("cpu").manual_seed(seed)
    jitter = 0.05 * (hi - lo) * (2.0 * torch.rand(B, DIM, generator=g) - 1.0)
    x = (mid.unsqueeze(0).expand(B, -1).clone() + jitter).clamp(lo, hi).to(device)
    x[:, -1] = math.radians(1.0)
    alt = torch.full((B, 1), 2000.0, device=device)
    V = torch.full((B, 1), 40.0, device=device)
    conditions = torch.cat([alt, V], dim=-1)
    return x, conditions


def scalar_loss(obj: Tensor, clist) -> Tensor:
    """Simple sum-over-everything loss, good enough for gradient-fidelity tests."""
    loss = obj.sum()
    for c in clist:
        loss = loss + c.value.sum()
    return loss


# Mode: timing
def run_timing(
    bench: E1BWB, x: Tensor, conditions: Tensor, device: torch.device,
    n_warmup: int, n_iters: int,
) -> dict[str, Any]:
    use_cuda = device.type == "cuda"
    timings = Timings(use_cuda=use_cuda)
    restores = instrument_bench(bench, timings)
    try:
        # Warmup, don't record.
        for _ in range(n_warmup):
            x_w = x.clone().requires_grad_(True)
            obj, clist = bench.forward(x_w, conditions)
            scalar_loss(obj, clist).backward()

        # Reset after warmup.
        timings.regions.clear()
        peak_fwd_mb = 0.0
        peak_bwd_mb = 0.0

        for _ in range(n_iters):
            x_t = x.clone().requires_grad_(True)
            if use_cuda:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with timed(timings, "forward_total"):
                obj, clist = bench.forward(x_t, conditions)
                loss = scalar_loss(obj, clist)
            if use_cuda:
                peak_fwd_mb = max(
                    peak_fwd_mb, torch.cuda.max_memory_allocated() / (1024 ** 2),
                )
                torch.cuda.reset_peak_memory_stats()
            with timed(timings, "backward_total"):
                loss.backward()
            if use_cuda:
                peak_bwd_mb = max(
                    peak_bwd_mb, torch.cuda.max_memory_allocated() / (1024 ** 2),
                )
    finally:
        restore_bench(restores)

    out = {"regions": timings.to_dict()}
    if use_cuda:
        out["peak"] = {"fwd_mb": round(peak_fwd_mb, 2), "bwd_mb": round(peak_bwd_mb, 2)}
    return out


# Mode: bf16 gradient fidelity
def _cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """Per-row cosine between `[B, D]` tensors."""
    an = a.norm(dim=-1).clamp_min(eps)
    bn = b.norm(dim=-1).clamp_min(eps)
    return (a * b).sum(dim=-1) / (an * bn)


def run_bf16_grad(
    bench: E1BWB, x: Tensor, conditions: Tensor, device: torch.device,
) -> dict[str, Any]:
    """Compare fp32 vs autocast(bf16) gradients w.r.t. x."""
    # fp32 reference.
    x32 = x.clone().requires_grad_(True)
    obj32, clist32 = bench.forward(x32, conditions)
    obj_loss32 = obj32.sum()
    loss32 = scalar_loss(obj32, clist32)

    # Gradients: dobj/dx (plain) and dloss/dx.
    g_obj_32 = torch.autograd.grad(obj_loss32, x32, retain_graph=True)[0].detach().float()
    g_loss_32 = torch.autograd.grad(loss32, x32)[0].detach().float()

    # bf16 under autocast.
    x16 = x.clone().requires_grad_(True)
    if device.type == "cuda":
        ac_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    elif device.type == "cpu":
        ac_ctx = torch.autocast(device_type="cpu", dtype=torch.bfloat16)
    else:
        # MPS etc, skip autocast, still useful for structure.
        ac_ctx = contextlib.nullcontext()

    with ac_ctx:
        obj16, clist16 = bench.forward(x16, conditions)
        obj_loss16 = obj16.sum()
        loss16 = scalar_loss(obj16, clist16)

    g_obj_16 = torch.autograd.grad(obj_loss16, x16, retain_graph=True)[0].detach().float()
    g_loss_16 = torch.autograd.grad(loss16, x16)[0].detach().float()

    def summarize(g_ref: Tensor, g_alt: Tensor) -> dict[str, float]:
        cos = _cosine(g_ref, g_alt)
        rel = (g_alt - g_ref).norm(dim=-1) / g_ref.norm(dim=-1).clamp_min(1e-12)
        return {
            "cos_min":  round(float(cos.min()), 6),
            "cos_mean": round(float(cos.mean()), 6),
            "cos_p10":  round(float(cos.kthvalue(max(1, cos.numel() // 10)).values), 6),
            "rel_l2_max":  round(float(rel.max()), 6),
            "rel_l2_mean": round(float(rel.mean()), 6),
            "ref_norm_mean": round(float(g_ref.norm(dim=-1).mean()), 6),
            "alt_norm_mean": round(float(g_alt.norm(dim=-1).mean()), 6),
        }

    return {
        "obj":  summarize(g_obj_32, g_obj_16),
        "loss": summarize(g_loss_32, g_loss_16),
        "note": (
            "obj = dobj.sum()/dx; loss = d(obj.sum() + sum c.value.sum())/dx. "
            "Per-row cosine across B samples, ref=fp32, alt=autocast(bf16)."
        ),
    }


# Driver
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["timing", "bf16-grad", "both"], default="both")
    ap.add_argument("--B", type=int, default=8, help="batch size")
    ap.add_argument("--N", type=int, default=12, help="spanwise stations")
    ap.add_argument("--device", type=str, default="cpu", help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--out", type=str, default="", help="write JSON report here")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"[profile_e1] device={device}  B={args.B}  N={args.N}  "
          f"mode={args.mode}", flush=True)

    bench = E1BWB(device=device, n_stations=args.N, live=True)
    x, conditions = build_inputs(args.B, args.seed, device)

    report: dict[str, Any] = {
        "config": {
            "mode": args.mode, "B": args.B, "N": args.N, "device": str(device),
            "seed": args.seed, "warmup": args.warmup, "iters": args.iters,
            "torch": torch.__version__,
            "cuda": torch.cuda.is_available(),
            "cuda_device": (
                torch.cuda.get_device_name() if torch.cuda.is_available() else None
            ),
        },
    }

    if args.mode in ("timing", "both"):
        print("[profile_e1] timing ...", flush=True)
        report["timing"] = run_timing(
            bench, x, conditions, device, args.warmup, args.iters,
        )

    if args.mode in ("bf16-grad", "both"):
        print("[profile_e1] bf16-grad ...", flush=True)
        report["bf16"] = run_bf16_grad(bench, x, conditions, device)

    as_json = json.dumps(report, indent=2)
    print(as_json, flush=True)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(as_json)
        print(f"[profile_e1] wrote {out_path}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f"[profile_e1] FAIL: {exc}", file=sys.stderr)
        sys.exit(1)
