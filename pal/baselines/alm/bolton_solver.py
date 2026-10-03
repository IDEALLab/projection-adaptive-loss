"""ALM+Bolt-On: plain ALM training, then the LM projector on the raw output at inference."""

from __future__ import annotations

import gc
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module

from pal.baselines._shared import eval_conditions, output_bounds_list
from pal.baselines.alm.solver import ALMConfig, ALMSolver
from pal.baselines.hparams import load_hparams
from pal.benchmarks.base import Benchmark, Query
from pal.model import CoordinationMLP
from pal.projection import Projector
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.tracking.base import Logger


@dataclass
class ALMBoltOnConfig(ALMConfig):
    """ALM hparams + LM projector hparams. Inherits all ALM fields verbatim."""

    proj_method: str = "lm_k"
    proj_delta: float = 1e-3
    proj_lambda_min: float = 1e-6
    proj_prescale: bool = False
    proj_prescale_floor: float = 1.0
    proj_max_iters: int = 10
    proj_tol: float = 1e-6
    eps_active: float = 1e-4
    detach_j: bool = True


class ALMBoltOnSolver:
    """ALM training + LM projector at inference."""

    name = "alm_bolton"

    def __init__(self, config: ALMBoltOnConfig | None = None):
        if config is None:
            config = ALMBoltOnConfig(**load_hparams("alm_bolton"))
        self.config = config

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Callable[[int, Module], None] | None = None,
        **hp: Any,
    ) -> TrainResult:
        cfg = self.config
        alm_fields = set(ALMConfig.__dataclass_fields__)
        alm_kwargs = {k: getattr(cfg, k) for k in alm_fields}
        inner = ALMSolver(ALMConfig(**alm_kwargs))
        result = inner.train(bench, seed, logger, on_epoch_end, **hp)
        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=result.train_wall_time_s,
            n_restarts=result.n_restarts,
            model_state=result.model_state,
            train_loss_trajectory=result.train_loss_trajectory,
            final_x_on_eval=result.final_x_on_eval,
        )

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        cfg = self.config
        spec = bench.spec
        device = cfg.device

        model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list(spec),
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            output_init_std=spec.model_hparams.get("output_init_std"),
        ).to(device)
        if train_result.model_state is None:
            raise RuntimeError(
                "ALMBoltOnSolver.predict: missing model_state in train_result"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()

        q = queries.to(device)
        conds = eval_conditions(q, spec.condition_dim)
        with torch.no_grad():
            raw = model(q.zeta, conds).detach()

        if spec.hard_output_box:
            box_lower = spec.output_bounds[0].to(device)
            box_upper = spec.output_bounds[1].to(device)
        else:
            box_lower = None
            box_upper = None

        projector = Projector(
            n_constraints=spec.n_eq + spec.n_ineq,
            constraint_types=list(spec.constraint_types),
            delta=cfg.proj_delta,
            prescale=cfg.proj_prescale,
            prescale_floor=cfg.proj_prescale_floor,
            eps_active=cfg.eps_active,
            method=cfg.proj_method,
            lambda_min=cfg.proj_lambda_min,
            box_lower=box_lower,
            box_upper=box_upper,
            detach_j=cfg.detach_j,
        )

        constraint_fn = _make_alm_constraint_fn(bench)
        n = len(q)
        chunk_size = cfg.predict_batch_size
        if chunk_size is None or chunk_size <= 0 or chunk_size >= n:
            post, info = projector.project(
                raw.clone(),
                constraint_fn,
                conds,
                max_iters=cfg.proj_max_iters,
                tol=cfg.proj_tol,
            )
            inf_iters = torch.full(
                (n,), int(info.get("iters", 0)), dtype=torch.int32, device="cpu",
            )
            return PredictionOutputs(
                raw=raw,
                post=post.detach(),
                projection=None,
                inference_iters=inf_iters,
            )

        post_chunks: list[torch.Tensor] = []
        iters_chunks: list[torch.Tensor] = []
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            cond_chunk = conds[start:end] if conds is not None else None
            post_chunk, info = projector.project(
                raw[start:end].clone(),
                constraint_fn,
                cond_chunk,
                max_iters=cfg.proj_max_iters,
                tol=cfg.proj_tol,
            )
            post_chunks.append(post_chunk.detach())
            iters_chunks.append(
                torch.full(
                    (end - start,), int(info.get("iters", 0)),
                    dtype=torch.int32, device="cpu",
                )
            )
            del post_chunk, cond_chunk
            gc.collect()
            if raw.is_cuda:
                torch.cuda.empty_cache()
        return PredictionOutputs(
            raw=raw,
            post=torch.cat(post_chunks, dim=0),
            projection=None,
            inference_iters=torch.cat(iters_chunks, dim=0),
        )


def _make_alm_constraint_fn(bench: Benchmark) -> Callable:
    """Build the `(y, conditions) -> (obj, [SimpleConstraint])` closure for Projector.project()."""
    spec = bench.spec
    types = list(spec.constraint_types)
    names = list(spec.constraint_names)
    has_list = hasattr(bench, "constraint_list")

    def constraint_fn(y, conditions):
        obj = bench.objective(y, conditions)
        if has_list:
            clist = bench.constraint_list(y, conditions)
            return obj, [_SimpleConstraint(c.value, c.type, c.name) for c in clist]
        c = bench.constraints(y, conditions)
        return obj, [
            _SimpleConstraint(c[..., k], types[k], names[k]) for k in range(len(names))
        ]

    return constraint_fn


class _SimpleConstraint:
    __slots__ = ("value", "type", "name")

    def __init__(self, value, type, name):
        self.value = value
        self.type = type
        self.name = name
