"""Original ENFORCE baseline: AdaNP projection + fixed quadratic penalty.

No projection during warmup. Afterwards the loss is evaluated at the AdaNP output
`y_tilde`, plus a displacement penalty `lambda_d*||y-y_tilde||^2`.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module
from torch.nn.utils import clip_grad_norm_

from pal.baselines._shared import (
    eval_conditions,
    make_forward_fn,
    output_bounds_list,
    sample_conditions,
    seed_everything,
)
from pal.baselines.enforce_orig.adanp import AdaNP
from pal.baselines.penalty import FixedPenalty
from pal.benchmarks.base import Benchmark, Query
from pal.constraints import constraints_to_violation
from pal.model import CoordinationMLP
from pal.runner.probe import mark_opt_step
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.solvers.grad_share import (
    KEY_CON,
    KEY_TOT,
    constraint_grad_norm,
    param_grad_norm,
    should_log_grad_share,
)
from pal.solvers.repair_contraction import (
    KEY_C_POST,
    KEY_C_PRE,
    mean_violation,
    should_log_repair_contraction,
)
from pal.tracking.base import Logger
from pal.utils.peak_memory import probe_peak_memory


@dataclass
class EnforceOrigConfig:
    seed: int = 0
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 1e-4
    grad_clip: float = 20.0
    device: str = "cpu"

    hidden: int = 512
    n_layers: int = 4

    adanp_max_iters: int = 10
    adanp_tol: float = 1e-6
    adanp_delta: float = 1e-3
    adanp_adaptive_delta: bool = False
    adanp_prescale: bool = True
    adanp_use_eigh: bool = True
    adanp_eps: float = 1e-6
    adanp_warmup_frac: float = 0.5
    adanp_lambda_d: float = 0.5

    # Pre-warmup `adanp.project` is a no-op, so the memory probe logs ~0 until then.
    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100


class EnforceOrigSolver:
    """FixedPenalty training with a post-warmup AdaNP projection step."""

    name = "enforce_orig"

    def __init__(self, config: EnforceOrigConfig | None = None):
        self.config = config or EnforceOrigConfig()

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Callable[[int, Module], None] | None = None,
        **hp: Any,
    ) -> TrainResult:
        cfg = _merge_cfg(self.config, dict(hp, seed=seed))
        seed_everything(cfg.seed)
        wall_start = time.monotonic()

        spec = bench.spec
        device = cfg.device
        K = spec.n_eq + spec.n_ineq
        constraint_types = list(spec.constraint_types)

        model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list(spec),
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            output_init_std=spec.model_hparams.get("output_init_std"),
        ).to(device)
        # Re-seed after model init so the training loop shares the reference RNG tape.
        seed_everything(cfg.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        forward_fn = make_forward_fn(bench)
        penalty = FixedPenalty(
            n_constraints=K,
            constraint_types=constraint_types,
            constraint_convention="standard",
        )
        adanp = AdaNP(
            n_constraints=K,
            constraint_types=constraint_types,
            max_iters=cfg.adanp_max_iters,
            tol=cfg.adanp_tol,
            delta=cfg.adanp_delta,
            adaptive_delta=cfg.adanp_adaptive_delta,
            prescale=cfg.adanp_prescale,
            use_eigh=cfg.adanp_use_eigh,
            eps=cfg.adanp_eps,
        )
        warmup_epoch = int(cfg.epochs * cfg.adanp_warmup_frac)

        loss_trajectory: list[float] = []

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            optimizer.zero_grad()

            zeta = torch.randn(cfg.batch_size, spec.zeta_dim, device=device)
            conditions = sample_conditions(
                bench, cfg.batch_size, device,
                seed=cfg.seed * 10_000_000 + epoch,
            )

            output = model(zeta, conditions)
            adanp_active = epoch > warmup_epoch
            do_probe = cfg.measure_repair_mem and (
                epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
            )

            with probe_peak_memory(enabled=do_probe) as probe:
                if adanp_active:
                    y_tilde, _info = adanp.project(output, forward_fn, conditions)
                    raw_result = forward_fn(y_tilde, conditions)
                    displacement_loss = (output - y_tilde).pow(2).mean()
                else:
                    y_tilde = output
                    raw_result = forward_fn(output, conditions)
                    displacement_loss = torch.tensor(0.0, device=device)

            objective = raw_result[0]
            constraint_list = raw_result[1]
            violations = constraints_to_violation(constraint_list)
            penalty_loss = penalty.compute_loss(violations)
            loss = (
                objective.mean()
                + penalty_loss.mean()
                + cfg.adanp_lambda_d * displacement_loss
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"epoch {epoch}: non-finite loss "
                    f"(obj={objective.mean().item():.3e}, "
                    f"pen={penalty_loss.mean().item():.3e})"
                )

            # Constraint part of the gradient = the fixed quadratic penalty.
            log_grad_share = should_log_grad_share(epoch, cfg.epochs)
            gn_con = (
                constraint_grad_norm(penalty_loss.mean(), model.parameters())
                if log_grad_share
                else 0.0
            )

            # Pre-repair violation at the raw backbone output (equal to post during warmup).
            log_repair = should_log_repair_contraction(epoch, cfg.epochs)
            c_pre_val = c_post_val = 0.0
            if log_repair:
                c_post_val = mean_violation(violations)
                if adanp_active:
                    with torch.no_grad():
                        pre_result = forward_fn(output.detach(), conditions)
                        c_pre_val = mean_violation(
                            constraints_to_violation(pre_result[1])
                        )
                else:
                    c_pre_val = c_post_val

            loss.backward()

            gn_tot = param_grad_norm(model.parameters()) if log_grad_share else 0.0
            grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            mark_opt_step(bench)

            if not adanp_active:
                penalty.update(violations.detach())

            step_metrics = {
                "loss": float(loss.item()),
                "objective": float(objective.mean().item()),
                "penalty": float(penalty_loss.mean().item()),
                "displacement": float(displacement_loss.item()),
                "max_violation": float(violations.max().item()) if violations.numel() else 0.0,
                "grad_norm": float(
                    grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
                ),
            }
            if adanp_active:
                step_metrics.update(adanp.log_dict())
            if log_grad_share:
                step_metrics[KEY_CON] = gn_con
                step_metrics[KEY_TOT] = gn_tot
            if log_repair:
                step_metrics[KEY_C_PRE] = c_pre_val
                step_metrics[KEY_C_POST] = c_post_val
            if cfg.measure_repair_mem:
                step_metrics["repair_peak_mem_bytes"] = (
                    float(probe["peak_bytes"]) if do_probe else float("nan")
                )
            logger.log_step(epoch, **step_metrics)
            loss_trajectory.append(float(loss.item()))

            if on_epoch_end is not None:
                on_epoch_end(epoch, model)

        wall = time.monotonic() - wall_start

        eval_queries = bench.eval_queries(seed).to(device)
        with torch.no_grad():
            model.eval()
            eval_conds = eval_conditions(eval_queries, spec.condition_dim)
            final_x = model(eval_queries.zeta, eval_conds).detach().cpu()

        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=1,
            model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            train_loss_trajectory=loss_trajectory,
            final_x_on_eval=final_x,
        )

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        # Keep outputs on cfg.device for the downstream all_reduce.
        spec = bench.spec
        cfg = self.config
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
                "EnforceOrigSolver.predict: missing model_state in train_result"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()
        q = queries.to(device)
        conds = eval_conditions(q, spec.condition_dim)

        # AdaNP needs autograd for its Jacobians, so only the backbone forward is no_grad.
        with torch.no_grad():
            raw = model(q.zeta, conds).detach()

        K = spec.n_eq + spec.n_ineq
        adanp = AdaNP(
            n_constraints=K,
            constraint_types=list(spec.constraint_types),
            max_iters=cfg.adanp_max_iters,
            tol=cfg.adanp_tol,
            delta=cfg.adanp_delta,
            adaptive_delta=cfg.adanp_adaptive_delta,
            prescale=cfg.adanp_prescale,
            use_eigh=cfg.adanp_use_eigh,
            eps=cfg.adanp_eps,
        )
        forward_fn = make_forward_fn(bench)
        y_tilde, info = adanp.project(raw.clone(), forward_fn, conds)
        post = y_tilde.detach()

        # Batch-wide early exit -> broadcast the batch's iteration count.
        inf_iters = torch.full(
            (len(q),), int(info.get("iters", 0)), dtype=torch.int32, device="cpu",
        )
        return PredictionOutputs(
            raw=raw, post=post, projection=None, inference_iters=inf_iters,
        )


def _merge_cfg(base: EnforceOrigConfig, overrides: dict[str, Any]) -> EnforceOrigConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"EnforceOrigSolver.train: unknown hparam(s) {sorted(unknown)}")
    return EnforceOrigConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
