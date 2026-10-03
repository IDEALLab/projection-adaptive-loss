"""Re-run inference on an already-trained solver.

Used by `pal eval --run-id <id>`. Per-query projection trajectories are logged
when the solver exposes `build_inference_projector`.
"""

from __future__ import annotations

import time

import torch

from pal.benchmarks.base import Benchmark, Query
from pal.eval.final_eval import (
    EvalResult,
    _aggregate,
    _aggregate_inf_iters,
    _resolve_inf_iters_max_allowed,
    build_eval_rows,
)
from pal.projection.trace import ProjectionStep
from pal.solvers.base import PredictionOutputs, Solver, TrainResult
from pal.tracking.base import Logger


def run_inference(
    bench: Benchmark,
    solver: Solver,
    train_result: TrainResult,
    queries: Query,
    logger: Logger,
    *,
    inference_trajectory_max: int = 200,
    inference_trajectory_downsample: int = 1,
) -> EvalResult:
    """Predict on `queries`, log per-query trajectories, return EvalResult.

    `inference_trajectory_max` caps the number of queries that get a full
    trajectory recorded, past the cap, they still get aggregate metrics but
    no per-iter trace. `inference_trajectory_downsample` keeps every Kth iter
    of each trajectory (1 = full).
    """
    spec = bench.spec
    tol = float(spec.tolerance)
    t0 = time.monotonic()
    pred = solver.predict(bench, queries, train_result, logger=None)
    predict_wall = time.monotonic() - t0

    if not isinstance(pred, PredictionOutputs):
        raise TypeError(
            f"{type(solver).__name__}.predict must return PredictionOutputs; "
            f"got {type(pred).__name__}"
        )

    conditions_full = None
    if spec.condition_dim > 0:
        conditions_full = queries.conditions.to(pred.raw.device)

    n_traj_logged = 0
    if hasattr(solver, "build_inference_projector"):
        projector, constraint_fn, max_iters, proj_tol = solver.build_inference_projector(bench)
        n_to_log = min(inference_trajectory_max, len(queries))
        for q_idx in range(n_to_log):
            cond_q = (
                conditions_full[q_idx : q_idx + 1] if conditions_full is not None else None
            )
            raw_q = pred.raw[q_idx : q_idx + 1].clone()
            with torch.enable_grad():
                _, trace, _converged = projector.project_trace(
                    raw_q.detach().requires_grad_(False),
                    constraint_fn,
                    cond_q,
                    max_iters=max_iters,
                    tol=proj_tol,
                )
            steps: list[ProjectionStep] = []
            for i, c_vals, y_i in trace:
                if i % inference_trajectory_downsample != 0 and i != trace[-1][0]:
                    continue
                with torch.no_grad():
                    obj_i = float(bench.objective(y_i, cond_q).mean().item())
                    c_means = [
                        float(c_vals[:, k].item()) for k in range(c_vals.shape[1])
                    ]
                steps.append(ProjectionStep(iter=i, obj=obj_i, constraints=c_means))
            logger.log_projection_trajectory(q_idx, "eval", steps)
            n_traj_logged += 1

    # Chunk aggregate calls like run_final_eval to avoid OOM.
    aggregate_chunk = getattr(getattr(solver, "config", None), "predict_batch_size", None)
    raw_metrics = _aggregate(bench, pred.raw, conditions_full, tol, chunk_size=aggregate_chunk)
    post_metrics = _aggregate(bench, pred.post, conditions_full, tol, chunk_size=aggregate_chunk)
    eval_rows = build_eval_rows(
        bench,
        pred.raw,
        pred.post,
        conditions_full,
        tol,
        query_offset=0,
        chunk_size=aggregate_chunk,
    )

    inf_iters_max_allowed = _resolve_inf_iters_max_allowed(solver)
    inf_iters_median, inf_iters_p90, inf_iters_max, inf_iters_n_conv = (
        _aggregate_inf_iters(pred.inference_iters, inf_iters_max_allowed)
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
            inf_iters_max_allowed if pred.inference_iters is not None else None
        ),
    )
