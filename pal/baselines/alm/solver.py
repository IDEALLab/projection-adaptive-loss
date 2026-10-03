"""ALM adapter (Basir & Senocak, arXiv:2306.04904v2).

Algorithm 3 (adaptive penalty update) lives in ``_state.py``. The primal step
is one Adam step per epoch on a ``CoordinationMLP``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from pal.baselines._shared import (
    dist_is_active,
    dist_rank,
    eval_conditions,
    make_forward_fn,
    output_bounds_list,
    sample_conditions,
    seed_everything,
)
from pal.baselines._shared import (
    local_batch_size as _local_batch_size,
)
from pal.baselines.alm._state import ALMState
from pal.baselines.hparams import load_hparams
from pal.benchmarks.base import Benchmark, Query
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
from pal.tracking.base import Logger
from pal.tracking.run_dir import atomic_torch_save


@dataclass
class ALMConfig:
    """Schema for ALM hparams; values load from ``hparams/alm.yaml``."""

    seed: int = 0
    device: str = "cpu"

    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-3
    grad_clip: float = 0.0

    hidden: int = 256
    n_layers: int = 4

    gamma: float = 1e-2
    alpha: float = 0.99
    eps: float = 1e-8
    mu_init: float = 1.0
    lambda_init: float = 1.0

    # train_dataset_size > 0 trains on a fixed pool; zeta_zero zeros zeta everywhere.
    train_dataset_size: int = 0
    zeta_zero: bool = False

    # Logging tags, unused by the solver.
    ablation_name: str = ""
    scenario_label: str = ""

    # ALM has no repair step; kept for a uniform schema (always logs 0).
    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100

    # Eval-time per-query chunk size; None or <= 0 disables chunking.
    predict_batch_size: int | None = None


class ALMSolver:
    name = "alm"

    def __init__(self, config: ALMConfig | None = None):
        if config is None:
            config = ALMConfig(**load_hparams("alm"))
        self.config = config

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Callable[[int, Module], None] | None = None,
        *,
        checkpoint_every: int = 0,
        checkpoint_dir: Path | None = None,
        resume_state: dict | None = None,
        **hp: Any,
    ) -> TrainResult:
        cfg = _merge_cfg(self.config, dict(hp, seed=seed))
        seed_everything(cfg.seed)
        wall_start = time.monotonic()

        spec = bench.spec
        device = cfg.device
        ddp_active = dist_is_active()
        rank = dist_rank()
        is_main = rank == 0
        batch_size = _local_batch_size(cfg.batch_size)
        K = spec.n_eq + spec.n_ineq
        constraint_types = list(spec.constraint_types)
        ineq_mask = torch.tensor(
            [t == "ineq" for t in constraint_types], device=device
        )
        # Bench-side objective normalization (e.g. genbase.mean^2 on ACOPF).
        obj_scale = float(getattr(bench, "objective_scale", 1.0))

        base_model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list(spec),
            hidden=cfg.hidden,
            n_layers=cfg.n_layers,
            output_init_std=spec.model_hparams.get("output_init_std"),
        ).to(device)
        model: Module
        if ddp_active:
            dev = torch.device(device)
            ddp_kwargs: dict[str, Any] = {}
            if dev.type == "cuda":
                ddp_kwargs["device_ids"] = [dev.index]
                ddp_kwargs["output_device"] = dev.index
            model = DDP(base_model, **ddp_kwargs)
        else:
            model = base_model
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        forward_fn = make_forward_fn(bench)
        state = ALMState(
            n_constraints=K,
            constraint_types=constraint_types,
            device=device,
            gamma=cfg.gamma,
            alpha=cfg.alpha,
            eps=cfg.eps,
            mu_init=cfg.mu_init,
            lambda_init=cfg.lambda_init,
        )

        pool_zeta: Tensor | None = None
        pool_conditions: Tensor | None = None
        if cfg.train_dataset_size > 0:
            pool_query = bench.sample_queries(
                n=cfg.train_dataset_size, split="train", seed=cfg.seed,
            )
            pool_zeta = pool_query.zeta.to(device)
            if cfg.zeta_zero:
                pool_zeta = torch.zeros_like(pool_zeta)
            pool_conditions = (
                pool_query.conditions.to(device)
                if pool_query.conditions is not None
                and pool_query.conditions.numel() > 0
                else None
            )
        n_pool = int(pool_zeta.shape[0]) if pool_zeta is not None else 0

        loss_trajectory: list[float] = []

        # Sampling is a pure function of (seed, epoch, rank), so resume needs no RNG state.
        start_epoch = 0
        if resume_state is not None:
            start_epoch = int(resume_state["epoch"])
            base_model.load_state_dict(resume_state["model_state"])
            optimizer.load_state_dict(resume_state["optimizer_state"])
            state.load_state_dict(resume_state["alm_state"])
            loss_trajectory = list(resume_state["loss_trajectory"])

        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            model.train()
            optimizer.zero_grad()

            if pool_zeta is not None:
                gen = torch.Generator(device="cpu").manual_seed(
                    int(cfg.seed) * 10_000_000 + epoch,
                )
                idx = torch.randperm(n_pool, generator=gen)[: cfg.batch_size].to(device)
                zeta = pool_zeta.index_select(0, idx)
                conditions = (
                    pool_conditions.index_select(0, idx)
                    if pool_conditions is not None else None
                )
            else:
                # zeta is shared across ranks (no rank offset), conditions are not.
                gen = torch.Generator(device="cpu").manual_seed(
                    int(cfg.seed) * 10_000_000 + epoch * 1000,
                )
                zeta = torch.randn(
                    batch_size, spec.zeta_dim, generator=gen,
                ).to(device)
                conditions = sample_conditions(
                    bench, batch_size, device,
                    seed=cfg.seed * 10_000_000 + epoch * 1000 + rank,
                )
            output = model(zeta, conditions)
            objective, constraint_list = forward_fn(output, conditions)

            # Algorithm 3 is equality-only: ineq slots use C_i := max(0, g_i), eq stay signed.
            c_raw = torch.stack([c.value for c in constraint_list], dim=-1)  # [B, K]
            c_values = torch.where(ineq_mask, c_raw.clamp_min(0.0), c_raw)

            penalty_loss = state.compute_loss(c_values)
            loss = objective.mean() / obj_scale + penalty_loss.mean()

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"epoch {epoch}: non-finite loss "
                    f"(obj={objective.mean().item():.3e}, "
                    f"pen={penalty_loss.mean().item():.3e})"
                )

            log_grad_share = is_main and should_log_grad_share(epoch, cfg.epochs)
            gn_con = (
                constraint_grad_norm(penalty_loss.mean(), model.parameters())
                if log_grad_share
                else 0.0
            )

            loss.backward()

            gn_tot = param_grad_norm(model.parameters()) if log_grad_share else 0.0
            if cfg.grad_clip > 0:
                g = clip_grad_norm_(model.parameters(), cfg.grad_clip)
                grad_norm_val = float(g.item() if isinstance(g, Tensor) else g)
            else:
                total_sq = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        total_sq += float(p.grad.detach().pow(2).sum().item())
                grad_norm_val = total_sq ** 0.5
            optimizer.step()
            mark_opt_step(bench)

            # Multipliers update on rank-local residuals (no all-reduce under DDP).
            state.update(c_values.detach())

            step_metrics = {
                "loss": float(loss.item()),
                "objective": float(objective.mean().item()),
                "penalty": float(penalty_loss.mean().item()),
                "max_violation": float(c_values.abs().max().item())
                if c_values.numel() else 0.0,
                "grad_norm": grad_norm_val,
            }
            if c_values.numel():
                c_mean_abs = c_values.detach().abs().mean(dim=0)  # [K]
                c_mean_signed = c_values.detach().mean(dim=0)     # [K]
                lam_d = state.lambdas.detach()
                mu_d = state.mu.detach()
                # Gradient through theta is sum_i (lambda_i + mu_i*c_bar_i) * dc_i/d theta.
                eff_mult = lam_d + mu_d * c_mean_signed
                slot_pen = lam_d * c_mean_signed + 0.5 * mu_d * c_mean_signed.pow(2)
                for k, name in enumerate(spec.constraint_names):
                    step_metrics[f"c/{name}"] = float(c_mean_abs[k].item())
                    step_metrics[f"eff_mult/{name}"] = float(eff_mult[k].item())
                    step_metrics[f"slot_pen/{name}"] = float(slot_pen[k].item())
            step_metrics.update(state.log_dict())
            if log_grad_share:
                step_metrics[KEY_CON] = gn_con
                step_metrics[KEY_TOT] = gn_tot
            if cfg.measure_repair_mem:
                step_metrics["repair_peak_mem_bytes"] = 0.0
            if is_main:
                logger.log_step(epoch, **step_metrics)
                if epoch == 1 or epoch % 2 == 0:
                    obj_v = step_metrics["objective"]
                    pen_v = step_metrics["penalty"]
                    gn_v = step_metrics["grad_norm"]
                    parts = [f"obj={obj_v:+.3e}", f"pen={pen_v:.3e}", f"gn={gn_v:.2e}"]
                    if c_values.numel():
                        for k, name in enumerate(spec.constraint_names):
                            parts.append(f"{name}={float(c_mean_abs[k].item()):.2e}")
                            parts.append(f"lambda_{name}={state.lambdas[k].item():.2e}")
                    print(f"[alm ep{epoch:>4d}] " + " ".join(parts), flush=True)
            loss_trajectory.append(float(loss.item()))

            if on_epoch_end is not None and is_main:
                on_epoch_end(epoch, model)

            if (
                checkpoint_every > 0
                and checkpoint_dir is not None
                and (epoch % checkpoint_every == 0 or epoch == cfg.epochs)
            ):
                if is_main:
                    model_ckpt = model.module if isinstance(model, DDP) else model
                    atomic_torch_save(
                        {
                            "epoch": epoch,
                            "model_state": {
                                k: v.detach().cpu()
                                for k, v in model_ckpt.state_dict().items()
                            },
                            "optimizer_state": optimizer.state_dict(),
                            "alm_state": state.state_dict(),
                            "loss_trajectory": loss_trajectory,
                        },
                        Path(checkpoint_dir) / "checkpoint_latest.pt",
                    )
                if ddp_active:
                    dist.barrier()

        wall = time.monotonic() - wall_start

        model_to_save = model.module if isinstance(model, DDP) else model
        eval_queries = bench.eval_queries(seed).to(device)
        with torch.no_grad():
            model_to_save.eval()
            eval_conds = eval_conditions(eval_queries, spec.condition_dim)
            final_x = model_to_save(eval_queries.zeta, eval_conds).detach().cpu()

        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=1,
            model_state={k: v.detach().cpu() for k, v in model_to_save.state_dict().items()},
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
        # Predict on the configured device: NCCL all_reduce downstream rejects CPU tensors.
        device = self.config.device
        spec = bench.spec
        model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list(spec),
            hidden=self.config.hidden,
            n_layers=self.config.n_layers,
            output_init_std=spec.model_hparams.get("output_init_std"),
        ).to(device)
        if train_result.model_state is None:
            raise RuntimeError(
                "ALMSolver.predict: missing model_state in train_result"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()
        q = queries.to(device)
        conds = eval_conditions(q, spec.condition_dim)
        with torch.no_grad():
            raw = model(q.zeta, conds).detach()
        return PredictionOutputs(raw=raw, post=raw, projection=None)


def _merge_cfg(base: ALMConfig, overrides: dict[str, Any]) -> ALMConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"ALMSolver.train: unknown hparam(s) {sorted(unknown)}")
    return ALMConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
