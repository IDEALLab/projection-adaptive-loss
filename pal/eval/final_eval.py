"""Predictor-based end-of-run evaluation.

Aggregates objective and violation metrics for the raw NN output and the
post-repair output.
Violation conventions:
- eq feasible when `|v| <= tolerance` -> violation = `|v|`
- ineq feasible when `v <= 0`           -> violation = `max(v, 0)`
- per-query feasibility = `max_k violation_k < spec.tolerance`
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from pal.benchmarks.base import Benchmark, Query
from pal.solvers.base import PredictionOutputs, Solver, TrainResult
from pal.tracking.base import Logger


@dataclass
class StageMetrics:
    """Aggregates one stage (raw or post) over the eval set."""

    obj_mean: float
    obj_std: float
    viol_max: float
    viol_mean: float
    feasibility: float
    per_constraint_viol_max: list[float]


@dataclass
class EvalResult:
    """End-of-run evaluation payload.

    `inf_iters_*` are None for solvers without an inference repair loop.
    """

    raw: StageMetrics
    post: StageMetrics
    n_queries: int
    n_restarts: int
    train_wall_time_s: float
    predict_wall_time_s: float
    tolerance: float
    obj_mean_raw: float = field(init=False)
    obj_mean_post: float = field(init=False)
    viol_max_raw: float = field(init=False)
    viol_max_post: float = field(init=False)
    feasibility_raw: float = field(init=False)
    feasibility_post: float = field(init=False)
    eval_rows: list[dict[str, Any]] | None = None
    inf_iters_median: float | None = None
    inf_iters_p90: float | None = None
    inf_iters_max: float | None = None
    inf_iters_n_converged: int | None = None
    inf_iters_max_allowed: int | None = None

    def __post_init__(self) -> None:
        self.obj_mean_raw = self.raw.obj_mean
        self.obj_mean_post = self.post.obj_mean
        self.viol_max_raw = self.raw.viol_max
        self.viol_max_post = self.post.viol_max
        self.feasibility_raw = self.raw.feasibility
        self.feasibility_post = self.post.feasibility


def _violations(c: Tensor, constraint_types: list[str]) -> Tensor:
    """Raw non-negative violations per slot: `eq: |v|`, `ineq: max(v, 0)` -> `[B, K]`.

    Keeps the benchmark's native (possibly interleaved) constraint order.
    """
    if c.shape[-1] == 0:
        return c
    is_eq = torch.tensor(
        [t == "eq" for t in constraint_types], device=c.device
    )
    return torch.where(is_eq, c.abs(), c.clamp(min=0))


def _aggregate(
    bench: Benchmark,
    x: Tensor,
    conditions: Tensor | None,
    tolerance: float,
    chunk_size: int | None = None,
) -> StageMetrics:
    stats, per_constraint = _accumulate_local_stats(
        bench, x, conditions, tolerance, chunk_size=chunk_size,
    )
    n_total = max(int(stats[0].item()), 1)
    obj_mean = stats[1] / n_total
    obj_var = (stats[2] / n_total) - obj_mean.pow(2)
    obj_std = obj_var.clamp(min=0.0).sqrt()
    K = bench.spec.n_eq + bench.spec.n_ineq
    n_viol = n_total * K
    viol_mean = float((stats[3] / n_viol).item()) if n_viol > 0 else 0.0
    return StageMetrics(
        obj_mean=float(obj_mean.item()),
        obj_std=float(obj_std.item()),
        viol_max=float(stats[4].item()),
        viol_mean=viol_mean,
        feasibility=float((stats[5] / n_total).item()),
        per_constraint_viol_max=per_constraint.cpu().tolist(),
    )


def _dist_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _shard_query(queries: Query) -> Query | None:
    if not _dist_is_active():
        return queries
    world = dist.get_world_size()
    rank = dist.get_rank()
    n = len(queries)
    chunk = (n + world - 1) // world
    start = rank * chunk
    end = min(start + chunk, n)
    if start >= end:
        return None
    return Query(
        queries.zeta[start:end],
        queries.conditions[start:end],
    )


def _shard_query_with_bounds(queries: Query) -> tuple[Query | None, int, int]:
    if not _dist_is_active():
        return queries, 0, len(queries)
    world = dist.get_world_size()
    rank = dist.get_rank()
    n = len(queries)
    chunk = (n + world - 1) // world
    start = rank * chunk
    end = min(start + chunk, n)
    if start >= end:
        return None, start, end
    return Query(
        queries.zeta[start:end],
        queries.conditions[start:end],
    ), start, end


def _aggregate_distributed(
    bench: Benchmark,
    x: Tensor | None,
    conditions: Tensor | None,
    tolerance: float,
    chunk_size: int | None = None,
) -> StageMetrics:
    stats, per_constraint = _accumulate_local_stats(
        bench, x, conditions, tolerance, chunk_size=chunk_size,
    )
    K = bench.spec.n_eq + bench.spec.n_ineq

    dist.all_reduce(stats, op=dist.ReduceOp.SUM)
    if K > 0:
        dist.all_reduce(per_constraint, op=dist.ReduceOp.MAX)

    n_total = max(int(stats[0].item()), 1)
    obj_mean = stats[1] / n_total
    obj_var = (stats[2] / n_total) - obj_mean.pow(2)
    obj_std = obj_var.clamp(min=0.0).sqrt()

    n_viol = n_total * K
    viol_mean = float((stats[3] / n_viol).item()) if n_viol > 0 else 0.0

    return StageMetrics(
        obj_mean=float(obj_mean.item()),
        obj_std=float(obj_std.item()),
        viol_max=float(stats[4].item()),
        viol_mean=viol_mean,
        feasibility=float((stats[5] / n_total).item()),
        per_constraint_viol_max=per_constraint.cpu().tolist(),
    )


def _stats_device(x: Tensor | None, conditions: Tensor | None) -> torch.device:
    if x is not None:
        return x.device
    if conditions is not None:
        return conditions.device
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _accumulate_local_stats(
    bench: Benchmark,
    x: Tensor | None,
    conditions: Tensor | None,
    tolerance: float,
    *,
    chunk_size: int | None,
) -> tuple[Tensor, Tensor]:
    K = bench.spec.n_eq + bench.spec.n_ineq
    device = _stats_device(x, conditions)
    stats = torch.zeros(6, dtype=torch.float64, device=device)
    per_constraint = torch.zeros(K, dtype=torch.float64, device=device)

    if x is None:
        return stats, per_constraint

    effective_chunk = _resolve_chunk_size(chunk_size, len(x))
    for start, end in _batch_slices(len(x), effective_chunk):
        x_chunk = x[start:end]
        cond_chunk = conditions[start:end] if conditions is not None else None
        # Bench outputs may live on a different device than x.
        obj = bench.objective(x_chunk, cond_chunk).to(device)
        c = bench.constraints(x_chunk, cond_chunk).to(device)
        v = _violations(c, bench.spec.constraint_types)
        per_query_max = (
            v.max(dim=-1).values if v.numel() else torch.zeros(len(x_chunk), device=device)
        )

        stats[0] += float(len(x_chunk))
        stats[1] += obj.double().sum()
        stats[2] += obj.double().pow(2).sum()
        stats[3] += v.double().sum() if v.numel() else 0.0
        if v.numel():
            stats[4] = torch.maximum(stats[4], v.double().max())
        stats[5] += (per_query_max < tolerance).double().sum()
        if K > 0 and v.numel():
            per_constraint = torch.maximum(per_constraint, v.double().max(dim=0).values)
        del obj, c, v, per_query_max, x_chunk, cond_chunk
        if x.is_cuda and effective_chunk < len(x):
            torch.cuda.empty_cache()
    return stats, per_constraint


def _per_query_stats(
    bench: Benchmark,
    x: Tensor | None,
    conditions: Tensor | None,
    tolerance: float,
    *,
    chunk_size: int | None,
) -> dict[str, list[float] | list[bool]]:
    if x is None:
        return {
            "obj": [], "viol_max": [], "max_eq": [], "max_ineq": [], "feasible": [],
        }

    obj_values: list[Tensor] = []
    viol_values: list[Tensor] = []
    max_eq_values: list[Tensor] = []
    max_ineq_values: list[Tensor] = []
    feasible_values: list[Tensor] = []
    effective_chunk = _resolve_chunk_size(chunk_size, len(x))
    device = _stats_device(x, conditions)
    ctypes = bench.spec.constraint_types
    eq_mask = torch.tensor([t == "eq" for t in ctypes], device=device)
    ineq_mask = ~eq_mask

    for start, end in _batch_slices(len(x), effective_chunk):
        x_chunk = x[start:end]
        cond_chunk = conditions[start:end] if conditions is not None else None
        obj = bench.objective(x_chunk, cond_chunk).detach()
        c = bench.constraints(x_chunk, cond_chunk).detach()
        v = _violations(c, ctypes)
        n_chunk = len(x_chunk)
        if v.numel():
            per_query_max = v.max(dim=-1).values.detach()
            v_eq = v[:, eq_mask]
            per_query_max_eq = (
                v_eq.max(dim=-1).values.detach()
                if v_eq.numel()
                else torch.zeros(n_chunk, dtype=v.dtype, device=device)
            )
            v_ineq = v[:, ineq_mask]
            per_query_max_ineq = (
                v_ineq.max(dim=-1).values.detach()
                if v_ineq.numel()
                else torch.zeros(n_chunk, dtype=v.dtype, device=device)
            )
        else:
            per_query_max = torch.zeros(n_chunk, dtype=torch.float32, device=device)
            per_query_max_eq = torch.zeros(n_chunk, dtype=torch.float32, device=device)
            per_query_max_ineq = torch.zeros(n_chunk, dtype=torch.float32, device=device)
        feasible = (per_query_max < tolerance).detach()
        obj_values.append(obj.cpu())
        viol_values.append(per_query_max.cpu())
        max_eq_values.append(per_query_max_eq.cpu())
        max_ineq_values.append(per_query_max_ineq.cpu())
        feasible_values.append(feasible.cpu())
        del obj, c, v, per_query_max, per_query_max_eq, per_query_max_ineq, feasible, x_chunk, cond_chunk
        if x.is_cuda and effective_chunk < len(x):
            torch.cuda.empty_cache()

    obj_cat = torch.cat(obj_values, dim=0) if obj_values else torch.empty(0)
    viol_cat = torch.cat(viol_values, dim=0) if viol_values else torch.empty(0)
    max_eq_cat = torch.cat(max_eq_values, dim=0) if max_eq_values else torch.empty(0)
    max_ineq_cat = torch.cat(max_ineq_values, dim=0) if max_ineq_values else torch.empty(0)
    feasible_cat = (
        torch.cat(feasible_values, dim=0) if feasible_values else torch.empty(0, dtype=torch.bool)
    )
    return {
        "obj": [float(v) for v in obj_cat.tolist()],
        "viol_max": [float(v) for v in viol_cat.tolist()],
        "max_eq": [float(v) for v in max_eq_cat.tolist()],
        "max_ineq": [float(v) for v in max_ineq_cat.tolist()],
        "feasible": [bool(v) for v in feasible_cat.tolist()],
    }


def build_eval_rows(
    bench: Benchmark,
    raw_x: Tensor | None,
    post_x: Tensor | None,
    conditions: Tensor | None,
    tolerance: float,
    *,
    query_offset: int = 0,
    chunk_size: int | None = None,
) -> list[dict[str, Any]]:
    raw = _per_query_stats(
        bench, raw_x, conditions, tolerance, chunk_size=chunk_size,
    )
    post = _per_query_stats(
        bench, post_x, conditions, tolerance, chunk_size=chunk_size,
    )
    n = len(raw["obj"])
    rows: list[dict[str, Any]] = []
    for idx in range(n):
        rows.append(
            {
                "query_idx": int(query_offset + idx),
                "obj_raw": float(raw["obj"][idx]),
                "obj_post": float(post["obj"][idx]),
                "viol_max_raw": float(raw["viol_max"][idx]),
                "viol_max_post": float(post["viol_max"][idx]),
                "max_eq_raw": float(raw["max_eq"][idx]),
                "max_eq_post": float(post["max_eq"][idx]),
                "max_ineq_raw": float(raw["max_ineq"][idx]),
                "max_ineq_post": float(post["max_ineq"][idx]),
                "feasible_raw": bool(raw["feasible"][idx]),
                "feasible_post": bool(post["feasible"][idx]),
            }
        )
    return rows


def _resolve_chunk_size(chunk_size: int | None, n_items: int) -> int:
    if chunk_size is None:
        return int(n_items)
    if int(chunk_size) <= 0:
        return int(n_items)
    return min(int(chunk_size), int(n_items))


def _batch_slices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)


def _resolve_inf_iters_max_allowed(solver: Solver) -> int | None:
    """Look up the solver's inference-iter cap, if it exposes one.

    Checks `proj_max_iters`, `corr_test_max_steps`, `newton_maxiter` and
    `max_iter` on the solver config.
    """
    cfg = getattr(solver, "config", None)
    if cfg is None:
        return None
    for attr in ("proj_max_iters", "corr_test_max_steps", "newton_maxiter", "max_iter"):
        v = getattr(cfg, attr, None)
        if v is not None:
            return int(v)
    return None


def _aggregate_inf_iters(
    inf_iters: Tensor | None, max_allowed: int | None,
) -> tuple[float | None, float | None, float | None, int | None]:
    """Return (median, p90, max, n_converged) over the per-query iter tensor.

    `n_converged` counts queries whose iter count is strictly below the cap.
    Returns all-None when `inf_iters` is None or empty.
    """
    if inf_iters is None or inf_iters.numel() == 0:
        return None, None, None, None
    t = inf_iters.detach().to(torch.float32).cpu()
    median = float(t.median().item())
    p90 = float(t.quantile(0.9).item())
    max_v = float(t.max().item())
    if max_allowed is None:
        n_conv = None
    else:
        n_conv = int((t < float(max_allowed)).sum().item())
    return median, p90, max_v, n_conv


def run_final_eval(
    bench: Benchmark,
    solver: Solver,
    train_result: TrainResult,
    queries: Query,
    tolerance: float | None = None,
    logger: Logger | None = None,
    distributed: bool | None = None,
) -> EvalResult:
    """Run `solver.predict(queries)`, aggregate raw + post metrics.

    `queries` must be the fingerprinted eval set the runner recorded.
    """
    spec = bench.spec
    tol = float(spec.tolerance if tolerance is None else tolerance)

    t0 = time.monotonic()
    sharded = _dist_is_active() if distributed is None else (bool(distributed) and _dist_is_active())
    local_queries, local_start, _local_end = (
        _shard_query_with_bounds(queries) if sharded else (queries, 0, len(queries))
    )
    predict_batch_size = getattr(getattr(solver, "config", None), "predict_batch_size", None)
    if local_queries is None:
        outputs = None
    else:
        outputs = solver.predict(bench, local_queries, train_result, logger=logger)
    predict_wall = time.monotonic() - t0
    if sharded:
        wall_device = _stats_device(
            outputs.raw if outputs is not None else None,
            local_queries.conditions if local_queries is not None else None,
        )
        wall = torch.tensor([predict_wall], dtype=torch.float64, device=wall_device)
        dist.all_reduce(wall, op=dist.ReduceOp.MAX)
        predict_wall = float(wall.item())

    if outputs is not None and not isinstance(outputs, PredictionOutputs):
        raise TypeError(
            f"{type(solver).__name__}.predict must return PredictionOutputs; "
            f"got {type(outputs).__name__}"
        )

    if sharded:
        local_conditions = None
        if outputs is not None and spec.condition_dim > 0:
            local_conditions = local_queries.conditions.to(outputs.raw.device)
        local_rows = build_eval_rows(
            bench,
            outputs.raw if outputs is not None else None,
            outputs.post if outputs is not None else None,
            local_conditions,
            tol,
            query_offset=local_start,
            chunk_size=predict_batch_size,
        )
        raw_metrics = _aggregate_distributed(
            bench,
            outputs.raw if outputs is not None else None,
            local_conditions,
            tol,
            chunk_size=predict_batch_size,
        )
        post_metrics = _aggregate_distributed(
            bench,
            outputs.post if outputs is not None else None,
            local_conditions,
            tol,
            chunk_size=predict_batch_size,
        )
        gathered_rows: list[list[dict[str, Any]] | None] | None = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        dist.gather_object(local_rows, gathered_rows, dst=0)
        eval_rows = None
        if dist.get_rank() == 0 and gathered_rows is not None:
            eval_rows = sorted(
                [row for shard in gathered_rows if shard for row in shard],
                key=lambda row: int(row["query_idx"]),
            )
    else:
        conditions = None
        if spec.condition_dim > 0:
            conditions = queries.conditions.to(outputs.raw.device)
        raw_metrics = _aggregate(
            bench, outputs.raw, conditions, tol, chunk_size=predict_batch_size,
        )
        post_metrics = _aggregate(
            bench, outputs.post, conditions, tol, chunk_size=predict_batch_size,
        )
        eval_rows = build_eval_rows(
            bench,
            outputs.raw,
            outputs.post,
            conditions,
            tol,
            query_offset=0,
            chunk_size=predict_batch_size,
        )

    # Sharded mode gathers the per-rank iteration tensors on rank 0.
    local_inf_iters = (
        outputs.inference_iters if outputs is not None else None
    )
    if sharded:
        gathered_inf_iters: list[Tensor | None] | None = (
            [None] * dist.get_world_size() if dist.get_rank() == 0 else None
        )
        dist.gather_object(local_inf_iters, gathered_inf_iters, dst=0)
        if dist.get_rank() == 0 and gathered_inf_iters is not None:
            non_none = [t for t in gathered_inf_iters if t is not None]
            inf_iters_global: Tensor | None = (
                torch.cat([t.detach().cpu() for t in non_none], dim=0)
                if non_none else None
            )
        else:
            inf_iters_global = None
    else:
        inf_iters_global = local_inf_iters

    inf_iters_max_allowed = _resolve_inf_iters_max_allowed(solver)
    inf_iters_median, inf_iters_p90, inf_iters_max, inf_iters_n_conv = (
        _aggregate_inf_iters(inf_iters_global, inf_iters_max_allowed)
    )

    return EvalResult(
        raw=raw_metrics,
        post=post_metrics,
        n_queries=len(queries),
        n_restarts=train_result.n_restarts,
        train_wall_time_s=train_result.train_wall_time_s,
        predict_wall_time_s=predict_wall,
        tolerance=tol,
        eval_rows=eval_rows,
        inf_iters_median=inf_iters_median,
        inf_iters_p90=inf_iters_p90,
        inf_iters_max=inf_iters_max,
        inf_iters_n_converged=inf_iters_n_conv,
        inf_iters_max_allowed=(
            inf_iters_max_allowed if inf_iters_global is not None else None
        ),
    )
