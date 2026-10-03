"""Shared infrastructure for the PAL solvers: `PALSolver.predict()` and train-loop helpers."""

from __future__ import annotations

import gc
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor

from pal.benchmarks.base import Benchmark, Query
from pal.model import CoordinationMLP
from pal.projection import ProjectionStep, Projector
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.tracking.base import Logger


@dataclass
class PALConfig:
    """Shared config, fields read by `PALSolver.predict()` and the helpers below.

    `proj_method` picks the repair step: "eigh" (spectrum-adaptive Tikhonov),
    "lm_k" (Yamashita-Fukushima LM), "sqp" (elastic-mode feasibility QP) or
    "ip" (one damped log-barrier Newton step at finite mu).
    """

    seed: int = 0
    batch_size: int = 32
    device: str = "cpu"

    hidden: int = 512
    n_layers: int = 4

    proj_method: str = "eigh"
    proj_delta: float = 1e-3
    proj_max_iters: int = 10
    proj_tol: float = 1e-6
    eps_active: float = 1e-4
    detach_j: bool = True
    jacobian_mode: str = "loop"
    measurement_window_epochs: int = 5
    measurement_period_epochs: int = 100
    predict_batch_size: int | None = None


class PALSolver:
    """Base class for the PAL family. Provides `predict()` and inference-projector hooks.

    Subclassed by `PALLogGapSolver`, which overrides `train()`.
    """

    name = "pal_base"

    def __init__(self, config: PALConfig | None = None):
        self.config = config or PALConfig()

    def build_inference_projector(self, bench: Benchmark):
        """Hook for `pal/eval/inference.py` to log per-query projection traces.

        Returns a `(Projector, constraint_fn)` pair configured with the same
        hparams the inference path uses.
        """
        cfg = self.config
        spec = bench.spec
        projector = _build_projector(cfg, spec, cfg.device)
        return projector, _make_constraint_fn(bench), cfg.proj_max_iters, cfg.proj_tol

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        """Return both the raw NN output and the post-projection output.

        Metrics should prefer `post`.
        """
        cfg = self.config
        spec = bench.spec
        device = cfg.device
        output_bounds_list = [
            (float(lo), float(hi))
            for lo, hi in zip(
                spec.output_bounds[0].tolist(),
                spec.output_bounds[1].tolist(), strict=False,
            )
        ]
        model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list,
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            output_init_std=spec.model_hparams.get("output_init_std"),
        ).to(device)
        if train_result.model_state is None:
            raise RuntimeError(
                "train_result.model_state is empty; PAL requires a learned model"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()
        q = queries.to(device)
        conditions = _eval_conditions(q, spec.condition_dim)
        with torch.no_grad():
            raw = model(q.zeta, conditions).detach()

        projector = _build_projector(cfg, spec, device)
        constraint_fn = _make_constraint_fn(bench)
        chunk_size = _resolve_predict_batch_size(cfg.predict_batch_size, len(q))
        if chunk_size >= len(q):
            post, info = projector.project(
                raw.clone(),
                constraint_fn,
                conditions,
                max_iters=cfg.proj_max_iters,
                tol=cfg.proj_tol,
            )
            inf_iters = torch.full(
                (len(q),), int(info.get("iters", 0)), dtype=torch.int32, device="cpu",
            )
            return PredictionOutputs(
                raw=raw,
                post=post.detach(),
                projection=None,
                inference_iters=inf_iters,
            )

        raw_chunks: list[Tensor] = []
        post_chunks: list[Tensor] = []
        iters_chunks: list[Tensor] = []
        import os as _os
        diag_mem = _os.environ.get("PAL_PREDICT_MEM_DIAG") == "1"
        for chunk_idx, (start, end) in enumerate(_batch_slices(len(q), chunk_size)):
            raw_chunk = raw[start:end]
            cond_chunk = conditions[start:end] if conditions is not None else None
            if diag_mem and raw.is_cuda:
                _dev = raw.device
                _alloc_pre = torch.cuda.memory_allocated(_dev) / 1024**3
                _resv_pre = torch.cuda.memory_reserved(_dev) / 1024**3
            post_chunk, info = projector.project(
                raw_chunk.clone(),
                constraint_fn,
                cond_chunk,
                max_iters=cfg.proj_max_iters,
                tol=cfg.proj_tol,
            )
            if diag_mem and raw.is_cuda:
                _alloc_post = torch.cuda.memory_allocated(_dev) / 1024**3
                _resv_post = torch.cuda.memory_reserved(_dev) / 1024**3
                _peak = torch.cuda.max_memory_allocated(_dev) / 1024**3
                print(
                    f"[predict-mem] chunk {chunk_idx:3d} q[{start}:{end}] "
                    f"alloc {_alloc_pre:6.2f}->{_alloc_post:6.2f} GiB "
                    f"resv {_resv_pre:6.2f}->{_resv_post:6.2f} GiB "
                    f"peak {_peak:6.2f} GiB",
                    flush=True,
                )
            raw_chunks.append(raw_chunk)
            post_chunks.append(post_chunk.detach())
            iters_chunks.append(
                torch.full(
                    (end - start,), int(info.get("iters", 0)),
                    dtype=torch.int32, device="cpu",
                )
            )
            # Free per-query intermediates to limit allocator fragmentation.
            del post_chunk, cond_chunk
            gc.collect()
            if raw.is_cuda:
                torch.cuda.empty_cache()
        return PredictionOutputs(
            raw=torch.cat(raw_chunks, dim=0),
            post=torch.cat(post_chunks, dim=0),
            projection=None,
            inference_iters=torch.cat(iters_chunks, dim=0),
        )


def _in_measurement_window(cfg, epoch: int) -> bool:
    """Return True when `epoch` falls inside a measurement window.

    `cfg.measurement_window_epochs` loop-mode epochs in every
    `cfg.measurement_period_epochs`.
    """
    period = max(1, int(getattr(cfg, "measurement_period_epochs", 100)))
    window = max(0, int(getattr(cfg, "measurement_window_epochs", 0)))
    return ((epoch - 1) % period) < window


def _apply_projector_schedule(cfg, projector, bench, epoch: int) -> None:
    """Flip the Projector's mode + probe's measurement flag for `epoch`.

    Respects three values of `cfg.jacobian_mode`:
      - "loop": always loop mode, measurement flag on (counters identical
        to regular counters).
      - "vmap" / "vmap_jacrev": always vmap, measurement flag off.
      - "sample": loop during measurement windows, vmap elsewhere.
    """
    mode = str(getattr(cfg, "jacobian_mode", "loop"))
    if mode == "loop":
        loop = True
    elif mode in ("vmap", "vmap_jacrev"):
        loop = False
    elif mode == "sample":
        loop = _in_measurement_window(cfg, epoch)
    else:
        raise ValueError(
            f"unknown PAL jacobian_mode {mode!r}; "
            "expected 'loop', 'vmap', 'vmap_jacrev', or 'sample'"
        )
    projector.set_jacobian_mode("loop" if loop else "vmap")
    set_meas = getattr(bench, "set_measurement", None)
    if callable(set_meas):
        set_meas(loop)


def _merge_cfg(base, overrides: dict[str, Any]):
    """Return `type(base)(...)` with `overrides` applied. Works for any
    dataclass config (`PALConfig`, `PALLogGapConfig`)."""
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"{type(base).__name__}: unknown hparam(s) {sorted(unknown)}")
    return type(base)(**{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}})


def _seed_everything(seed: int) -> None:
    """Seed `random`, numpy and torch."""
    import random

    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sample_conditions(
    bench: Benchmark, n: int, device: str, seed: int
) -> Tensor | None:
    """Per-epoch condition draw. Returns `None` for unconditional benchmarks.

    Unconditional benchmarks short-circuit before any RNG consumption.
    """
    if bench.spec.condition_dim == 0:
        return None
    q = bench.sample_queries(n, split="train", seed=seed)
    return q.conditions.to(device)


def _repair_projector_kwargs(cfg: Any) -> dict[str, Any]:
    """Repair-step `Projector` kwargs a config carries, `{}` otherwise.

    The inference projector must be built with the same repair-step knobs as
    the training projector.
    """
    kwargs: dict[str, Any] = {}
    if hasattr(cfg, "proj_sqp_rho"):
        kwargs["sqp_rho"] = float(cfg.proj_sqp_rho)
    if hasattr(cfg, "proj_ip_mu0"):
        kwargs["ip_mu0"] = float(cfg.proj_ip_mu0)
    if hasattr(cfg, "proj_ip_fixed_mu"):
        fixed = cfg.proj_ip_fixed_mu
        kwargs["ip_fixed_mu"] = None if fixed is None else float(fixed)
    return kwargs


def _build_projector(cfg: PALConfig, spec: Any, device: str | torch.device) -> Projector:
    """Construct the Projector used at train + inference time.

    With `spec.hard_output_box`, projected iterates are clamped into the
    declared output box.
    """
    if spec.hard_output_box:
        box_lower = spec.output_bounds[0].to(device)
        box_upper = spec.output_bounds[1].to(device)
    else:
        box_lower = None
        box_upper = None
    # "sample" is a solver-level mode; the Projector starts at loop.
    raw_mode = str(getattr(cfg, "jacobian_mode", "loop"))
    init_mode = "loop" if raw_mode in ("loop", "sample") else raw_mode
    # Thread damping knobs through so inference matches the training projector.
    damping_kwargs: dict[str, Any] = {}
    if hasattr(cfg, "proj_lambda_min"):
        damping_kwargs["lambda_min"] = cfg.proj_lambda_min
    if hasattr(cfg, "proj_eigh_fallback_reg"):
        damping_kwargs["eigh_fallback_reg"] = cfg.proj_eigh_fallback_reg
    damping_kwargs.update(_repair_projector_kwargs(cfg))
    return Projector(
        n_constraints=spec.n_eq + spec.n_ineq,
        constraint_types=list(spec.constraint_types),
        delta=cfg.proj_delta,
        prescale=False,
        eps_active=cfg.eps_active,
        method=cfg.proj_method,
        box_lower=box_lower,
        box_upper=box_upper,
        detach_j=cfg.detach_j,
        jacobian_mode=init_mode,
        **damping_kwargs,
    )


def _make_constraint_fn(bench: Benchmark) -> Callable:
    """Build the `(y, conditions) -> (obj, [SimpleConstraint])` closure the
    Projector expects. SimpleConstraint carries `(value, type, name)` only."""
    spec = bench.spec
    types = list(spec.constraint_types)
    names = list(spec.constraint_names)
    has_list = hasattr(bench, "constraint_list")

    def constraint_fn(y: Tensor, conditions: Tensor | None):
        obj = _objective(bench, y, conditions)
        if has_list:
            clist = bench.constraint_list(y, conditions)
            simple = [_SimpleConstraint(c.value, c.type, c.name) for c in clist]
            return obj, simple
        c = bench.constraints(y, conditions)
        simple = [
            _SimpleConstraint(c[..., k], types[k], names[k]) for k in range(len(names))
        ]
        return obj, simple

    return constraint_fn


def _make_constraint_values_fn(bench: Benchmark) -> Callable:
    """Build the `(y, conditions) -> Tensor[B, K]` closure, values only.

    Used where only c(y) is needed, avoiding the objective call.
    """
    has_list = hasattr(bench, "constraint_list")

    def values_fn(y: Tensor, conditions: Tensor | None) -> Tensor:
        if has_list:
            clist = bench.constraint_list(y, conditions)
            return torch.stack([c.value for c in clist], dim=-1)
        return bench.constraints(y, conditions)

    return values_fn


class _SimpleConstraint:
    """Minimal container the Projector reads from (value / type / name)."""

    __slots__ = ("value", "type", "name")

    def __init__(self, value: Tensor, type: str, name: str):
        self.value = value
        self.type = type
        self.name = name


def _objective(
    bench: Benchmark, x: Tensor, conditions: Tensor | None
) -> Tensor:
    return bench.objective(x, conditions)


def _log_projection_trajectory(
    *,
    bench: Benchmark,
    projector: Projector,
    constraint_fn: Callable,
    y_hat: Tensor,
    conditions: Tensor | None,
    step: int,
    phase: str,
    logger: Logger,
) -> None:
    with torch.enable_grad():
        _, trace, _converged = projector.project_trace(
            y_hat.detach().requires_grad_(False),
            constraint_fn,
            conditions,
            max_iters=10,
            tol=1e-6,
        )
    steps: list[ProjectionStep] = []
    for i, c_vals, y_i in trace:
        with torch.no_grad():
            obj_i = float(_objective(bench, y_i, conditions).mean().item())
            c_means = [float(c_vals[:, k].mean().item()) for k in range(c_vals.shape[1])]
        steps.append(ProjectionStep(iter=i, obj=obj_i, constraints=c_means))
    logger.log_projection_trajectory(step, phase, steps)


def _periodic_eval(
    *,
    model: CoordinationMLP,
    bench: Benchmark,
    constraint_fn: Callable,
    cfg: PALConfig,
    epoch: int,
    logger: Logger,
    local_batch_size: int | None = None,
    rank: int = 0,
) -> None:
    """DDP-aware health check: raw obj and |c| summary over `cfg.eval_samples`.

    Runs without `torch.no_grad()` because the BWB SDF probe needs grad mode;
    nothing calls `.backward()`, so the graph is freed on scope exit.
    """
    phase_ctx = getattr(bench, "phase_ctx", None)
    ctx = phase_ctx("periodic_eval") if phase_ctx is not None else _nullctx()
    world_size = dist.get_world_size() if _dist_is_active() else 1
    n_total = int(cfg.eval_samples)
    # Even split across ranks; tail ranks may pick up one extra sample.
    n_local = (n_total + world_size - 1 - rank) // world_size
    with ctx:
        model.eval()
        zeta = torch.randn(
            n_local, bench.spec.zeta_dim, device=cfg.device
        )
        conditions = _sample_conditions(
            bench, n_local, cfg.device,
            seed=cfg.seed * 10_000_000 + epoch * 1000 + rank,
        )
        y_hat = model(zeta, conditions).detach()
        obj_local = _objective(bench, y_hat, conditions).detach()
        c_vals = bench.constraints(y_hat, conditions).detach()
        c_abs = c_vals.abs()
        obj_sum = obj_local.sum()
        c_abs_sum = c_abs.sum()
        c_abs_max_local = c_abs.max() if c_abs.numel() > 0 else torch.tensor(
            0.0, device=cfg.device
        )
        n_cells_local = torch.tensor(
            float(c_abs.numel()), device=cfg.device, dtype=torch.float64,
        )
        n_obj_local = torch.tensor(
            float(obj_local.numel()), device=cfg.device, dtype=torch.float64,
        )
        model.train()
    if _dist_is_active():
        for t in (obj_sum, c_abs_sum, n_cells_local, n_obj_local):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
        max_t = c_abs_max_local.detach().clone()
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX)
        c_abs_max_local = max_t
    obj_mean = float((obj_sum / n_obj_local.clamp_min(1.0)).item())
    c_abs_mean = float((c_abs_sum / n_cells_local.clamp_min(1.0)).item())
    c_abs_max = float(c_abs_max_local.item())
    if rank == 0:
        logger.log_step(
            epoch,
            **{
                "eval/obj_raw": obj_mean,
                "eval/c_abs_max": c_abs_max,
                "eval/c_abs_mean": c_abs_mean,
            },
        )


@contextmanager
def _nullctx():
    yield


def _eval_conditions(q: Query, condition_dim: int) -> Tensor | None:
    return q.conditions if condition_dim > 0 else None


def _dist_is_active() -> bool:
    return dist.is_available() and dist.is_initialized()


def _local_batch_size(global_batch_size: int) -> int:
    if not _dist_is_active():
        return int(global_batch_size)
    world_size = dist.get_world_size()
    if global_batch_size % world_size != 0:
        raise ValueError(
            f"distributed PAL requires batch_size divisible by world_size; "
            f"got batch_size={global_batch_size}, world_size={world_size}"
        )
    return int(global_batch_size // world_size)


def _resolve_predict_batch_size(predict_batch_size: int | None, n_queries: int) -> int:
    if predict_batch_size is None:
        return int(n_queries)
    if int(predict_batch_size) <= 0:
        return int(n_queries)
    return min(int(predict_batch_size), int(n_queries))


def _batch_slices(n_items: int, batch_size: int):
    for start in range(0, n_items, batch_size):
        yield start, min(start + batch_size, n_items)
