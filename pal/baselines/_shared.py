"""Helpers shared by the ALM, enforce_orig and enforce_v4 solvers."""

from __future__ import annotations

import random
from collections.abc import Callable

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from pal.benchmarks.base import Benchmark, Query
from pal.constraints import Constraint


def dist_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def dist_rank() -> int:
    return dist.get_rank() if dist_is_active() else 0


def dist_world_size() -> int:
    return dist.get_world_size() if dist_is_active() else 1


def local_batch_size(global_batch_size: int) -> int:
    """Per-rank batch slice. Mirrors `pal.method.solver._local_batch_size`."""
    if not dist_is_active():
        return int(global_batch_size)
    world = dist.get_world_size()
    if global_batch_size % world != 0:
        raise ValueError(
            f"distributed baseline requires batch_size divisible by world_size; "
            f"got batch_size={global_batch_size}, world_size={world}"
        )
    return int(global_batch_size // world)


def seed_everything(seed: int) -> None:
    """Seed `random`, numpy and torch (all CUDA devices included)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_forward_fn(bench: Benchmark) -> Callable[[Tensor, Tensor | None], tuple[Tensor, list[Constraint]]]:
    """Return a `(x, conditions) -> (obj, list[Constraint])` forward function.

    Prefers `bench.forward(x, conditions)` when available so objective and
    constraints share one autograd subgraph, which matches the reference
    implementation to the last ulp.
    """
    if hasattr(bench, "forward"):
        return bench.forward  # type: ignore[return-value]

    has_list = hasattr(bench, "constraint_list")
    spec = bench.spec
    names = list(spec.constraint_names)
    types = list(spec.constraint_types)

    def forward(x: Tensor, conditions: Tensor | None) -> tuple[Tensor, list[Constraint]]:
        obj = bench.objective(x, conditions)
        if has_list:
            return obj, bench.constraint_list(x, conditions)
        c = bench.constraints(x, conditions)
        B = c.shape[0]
        zero = torch.zeros(B, device=c.device)
        eq_tol = torch.full((B,), 1e-3, device=c.device)
        out: list[Constraint] = []
        for k in range(c.shape[-1]):
            if types[k] == "eq":
                out.append(Constraint(value=c[..., k], type="eq", tol=eq_tol, margin=zero, name=names[k]))
            else:
                margin = torch.full((B,), 1e-4, device=c.device)
                out.append(Constraint(value=c[..., k], type="ineq", tol=zero, margin=margin, name=names[k]))
        return obj, out

    return forward


def sample_conditions(
    bench: Benchmark, n: int, device: str, seed: int
) -> Tensor | None:
    """Per-epoch condition draw. Returns `None` for unconditional benchmarks.

    Callers pass `cfg.seed * 10_000_000 + epoch` so every (run_seed, epoch)
    pair gets a distinct condition batch. Unconditional benchmarks
    (`condition_dim == 0`) short-circuit before any RNG consumption.
    """
    if bench.spec.condition_dim == 0:
        return None
    q = bench.sample_queries(n, split="train", seed=seed)
    return q.conditions.to(device)


def eval_conditions(q: Query, condition_dim: int) -> Tensor | None:
    return q.conditions if condition_dim > 0 else None


def output_bounds_list(spec) -> list[tuple[float, float]]:
    return [
        (float(lo), float(hi))
        for lo, hi in zip(
            spec.output_bounds[0].tolist(),
            spec.output_bounds[1].tolist(), strict=False,
        )
    ]
