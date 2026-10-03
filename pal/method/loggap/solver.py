"""PAL-LogGap: linear loss on `c_pre`, log-gap multiplier driven by `c_post`.

Uses `MetricLogGap` for both the constraint and displacement multipliers; the
rest comes from the shared `PALSolver` base.
"""

from __future__ import annotations

import os
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

from pal.benchmarks.base import Benchmark
from pal.method.loggap.loss import MetricLogGap
from pal.method.solver import (
    PALSolver,
    _apply_projector_schedule,
    _dist_is_active,
    _eval_conditions,
    _local_batch_size,
    _log_projection_trajectory,
    _make_constraint_fn,
    _make_constraint_values_fn,
    _merge_cfg,
    _objective,
    _periodic_eval,
    _repair_projector_kwargs,
    _sample_conditions,
    _seed_everything,
)
from pal.model import CoordinationMLP
from pal.projection import Projector, _announce_jacobian_mode_once
from pal.runner.probe import mark_opt_step
from pal.solvers.base import TrainResult
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
from pal.tracking.run_dir import atomic_torch_save
from pal.utils.peak_memory import probe_peak_memory


@dataclass
class PALLogGapConfig:
    """Hparams for PAL-LogGap."""

    seed: int = 0
    epochs: int = 200
    batch_size: int = 32
    lr: float = 1e-4
    grad_clip: float = 20.0
    device: str = "cpu"

    hidden: int = 512
    n_layers: int = 4

    proj_method: str = "lm_k"
    proj_delta: float = 1e-3
    proj_max_iters: int = 10
    proj_tol: float = 1e-6
    eps_active: float = 1e-4
    detach_j: bool = True
    # lambda = max(delta * ||c_active||^2, proj_lambda_min) inside the projector.
    proj_lambda_min: float = 1e-6
    proj_eigh_fallback_reg: float = 1e-6
    # "loop" = K VJPs, "vmap_jacrev" / "vmap" = batched jacrev,
    # "sample" = loop during measurement windows, vmap elsewhere.
    jacobian_mode: str = "loop"
    measurement_window_epochs: int = 5
    measurement_period_epochs: int = 100
    project_to_output_box: bool = False
    predict_batch_size: int | None = None

    # `tau` defaults to the bench's `spec.tau`; `disp_tol` bounds ||y_hat - y_tilde||.
    tau: float | None = None
    disp_tol: float = 1e-4
    rate: float = 1e-2
    max_decades: float = 1.0
    # Clamp the log-gap at 0 so the multiplier never decays.
    monotone_multiplier: bool = False
    # True: objective value at y_tilde, gradient through y_hat (straight-through).
    cpre_obj_straight_through: bool = False

    eval_every: int = 100
    eval_samples: int = 64
    projection_log_every: int = 0

    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100

    ablation_name: str = ""
    scenario_label: str = ""

    # Paper-faithful protocol: fixed training pool and zero zeta.
    train_dataset_size: int = 0
    zeta_zero: bool = False


class PALLogGapSolver(PALSolver):
    """PAL variant with linear `lambda * c_pre` loss and log-gap multiplier."""

    name = "pal_loggap"

    def __init__(self, config: PALLogGapConfig | None = None):
        self.config = config or PALLogGapConfig()

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
        _seed_everything(cfg.seed)
        wall_start = time.monotonic()

        device = cfg.device
        ddp_active = _dist_is_active()
        rank = dist.get_rank() if ddp_active else 0
        is_main = rank == 0
        local_batch_size = _local_batch_size(cfg.batch_size)
        spec = bench.spec
        K = spec.n_eq + spec.n_ineq
        constraint_types = list(spec.constraint_types)
        # Objective scale factor on the training loss only.
        obj_scale = float(getattr(bench, "objective_scale", 1.0))

        output_bounds_list = [
            (float(lo), float(hi))
            for lo, hi in zip(
                spec.output_bounds[0].tolist(),
                spec.output_bounds[1].tolist(), strict=False,
            )
        ]
        base_model = CoordinationMLP(
            dim_zeta=spec.zeta_dim,
            dim_conditions=spec.condition_dim,
            dim_output=spec.dim,
            output_bounds=output_bounds_list,
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

        constraint_fn = _make_constraint_fn(bench)
        values_fn = _make_constraint_values_fn(bench)
        init_mode = (
            "loop" if cfg.jacobian_mode in ("loop", "sample") else cfg.jacobian_mode
        )
        projector = Projector(
            n_constraints=K,
            constraint_types=constraint_types,
            delta=cfg.proj_delta,
            prescale=False,
            eps_active=cfg.eps_active,
            method=cfg.proj_method,
            detach_j=cfg.detach_j,
            jacobian_mode=init_mode,
            lambda_min=cfg.proj_lambda_min,
            eigh_fallback_reg=cfg.proj_eigh_fallback_reg,
            **_repair_projector_kwargs(cfg),
        )
        if is_main:
            _announce_jacobian_mode_once(cfg.jacobian_mode)
        if cfg.tau is not None:
            effective_tau = float(cfg.tau)
        elif spec.tau is not None:
            effective_tau = float(spec.tau)
        else:
            raise ValueError(
                f"pal_loggap on bench {spec.id!r}: no tau available, "
                "set `tau=` on the BenchmarkSpec or pass cfg.tau explicitly."
            )
        lg = MetricLogGap(
            n_metrics=K,
            tau=effective_tau,
            rate=cfg.rate,
            max_decades=cfg.max_decades,
            device=device,
            monotone=cfg.monotone_multiplier,
        )
        lg_disp = MetricLogGap(
            n_metrics=1,
            tau=cfg.disp_tol,
            rate=cfg.rate,
            max_decades=cfg.max_decades,
            device=device,
            monotone=cfg.monotone_multiplier,
        )

        is_eq = torch.tensor(
            [t == "eq" for t in constraint_types], device=device
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

        # Minibatches are a function of (seed, epoch, rank), so resume needs no RNG state.
        start_epoch = 0
        if resume_state is not None:
            start_epoch = int(resume_state["epoch"])
            base_model.load_state_dict(resume_state["model_state"])
            optimizer.load_state_dict(resume_state["optimizer_state"])
            lg.load_state_dict(resume_state["lg"])
            lg_disp.load_state_dict(resume_state["lg_disp"])
            loss_trajectory = list(resume_state["loss_trajectory"])

        for epoch in range(start_epoch + 1, cfg.epochs + 1):
            _apply_projector_schedule(cfg, projector, bench, epoch)
            model.train()
            optimizer.zero_grad()

            if pool_zeta is not None:
                gen = torch.Generator(device="cpu").manual_seed(
                    int(cfg.seed) * 10_000_000 + epoch * 1000 + rank,
                )
                idx = torch.randperm(n_pool, generator=gen)[:local_batch_size].to(device)
                zeta = pool_zeta.index_select(0, idx)
                conditions = (
                    pool_conditions.index_select(0, idx)
                    if pool_conditions is not None else None
                )
            else:
                # Keyed by (seed, epoch) so resume is exact; zeta is shared across ranks.
                gen = torch.Generator(device="cpu").manual_seed(
                    int(cfg.seed) * 10_000_000 + epoch * 1000,
                )
                zeta = torch.randn(
                    local_batch_size, spec.zeta_dim, generator=gen,
                ).to(device)
                conditions = _sample_conditions(
                    bench, local_batch_size, device,
                    seed=cfg.seed * 10_000_000 + epoch * 1000 + rank,
                )

            y_hat = model(zeta, conditions)
            obj_live, c_list_live = bench.forward(y_hat, conditions)
            c_pre = torch.stack([c.value for c in c_list_live], dim=-1)
            do_probe = cfg.measure_repair_mem and (
                epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
            )
            with probe_peak_memory(enabled=do_probe) as probe:
                y_tilde_detached, proj_info = projector.step(
                    y_hat, c_pre, values_fn, conditions,
                )

            c_pre_live = proj_info["c_pre"]
            c_post_detached = proj_info["c_post"]
            displacement_live = proj_info["displacement_live"]

            c_pre_residuals = torch.where(
                is_eq.unsqueeze(0),
                c_pre_live.abs(),
                c_pre_live.clamp(min=0),
            )
            residual_loss = lg.compute_loss(c_pre_residuals)

            disp_mag = displacement_live.pow(2).sum(dim=-1).add(1e-12).sqrt()
            disp_metric = disp_mag.unsqueeze(-1)
            displacement_loss = lg_disp.compute_loss(disp_metric)

            if cfg.cpre_obj_straight_through:
                y_for_obj = y_hat + (y_tilde_detached - y_hat).detach()
                objective = _objective(bench, y_for_obj, conditions)
            else:
                objective = obj_live
            constraint_loss = residual_loss.mean()
            disp_loss_mean = displacement_loss.mean()
            obj_term = objective.mean() / obj_scale
            loss = obj_term + constraint_loss + disp_loss_mean

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"epoch {epoch}: non-finite loss "
                    f"(obj={objective.mean().item():.3e}, "
                    f"constraint={constraint_loss.item():.3e})"
                )

            # Per-component gradient diagnostics at y_hat (env-gated).
            grad_diag: dict[str, float] = {}
            if os.environ.get("E3_LOG_GRAD_DIAG") == "1" and is_main:
                import torch as _t
                g_obj = _t.autograd.grad(obj_term, y_hat, retain_graph=True)[0]
                g_con = _t.autograd.grad(constraint_loss, y_hat, retain_graph=True)[0]
                g_dis = _t.autograd.grad(disp_loss_mean, y_hat, retain_graph=True)[0]
                g_tot = g_obj + g_con + g_dis
                def _norm(g):
                    return float(g.pow(2).sum(dim=-1).sqrt().mean().item())
                def _cos(a, b):
                    a_flat = a.reshape(-1)
                    b_flat = b.reshape(-1)
                    na = a_flat.norm()
                    nb = b_flat.norm()
                    if na < 1e-30 or nb < 1e-30:
                        return float("nan")
                    return float((a_flat @ b_flat / (na * nb)).item())
                grad_diag = {
                    "grady/obj_norm": _norm(g_obj),
                    "grady/constraint_norm": _norm(g_con),
                    "grady/disp_norm": _norm(g_dis),
                    "grady/total_norm": _norm(g_tot),
                    "grady/cos_obj_constraint": _cos(g_obj, g_con),
                    "grady/cos_obj_total": _cos(g_obj, g_tot),
                    "grady/cos_constraint_total": _cos(g_con, g_tot),
                    "grady/cancellation_ratio": (
                        _norm(g_tot) / max(_norm(g_obj) + _norm(g_con) + _norm(g_dis), 1e-30)
                    ),
                }

            # Numerator is the constraint term only.
            log_grad_share = is_main and should_log_grad_share(epoch, cfg.epochs)
            log_repair = is_main and should_log_repair_contraction(
                epoch, cfg.epochs
            )
            gn_con = (
                constraint_grad_norm(constraint_loss, model.parameters())
                if log_grad_share
                else 0.0
            )

            loss.backward()

            gn_tot = param_grad_norm(model.parameters()) if log_grad_share else 0.0

            for name, p in model.named_parameters():
                if p.grad is not None and not p.grad.isfinite().all():
                    raise RuntimeError(
                        f"epoch {epoch}: non-finite gradient in param '{name}'"
                    )

            # clip_grad_norm_ with max_norm=0 zeros every gradient.
            if cfg.grad_clip > 0:
                grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
            else:
                _grads = [p.grad for p in model.parameters() if p.grad is not None]
                grad_norm = (
                    torch.norm(torch.stack([g.norm() for g in _grads]))
                    if _grads
                    else torch.tensor(0.0)
                )

            param_step_diag: dict[str, float] = {}
            if os.environ.get("E3_LOG_GRAD_DIAG") == "1" and is_main:
                _params_pre = {}
                for name, p in model.named_parameters():
                    if "out" in name.lower() or name.endswith("weight") or name.endswith("bias"):
                        _params_pre[name] = p.detach().clone()
            optimizer.step()
            if os.environ.get("E3_LOG_GRAD_DIAG") == "1" and is_main:
                total_step_norm = 0.0
                for name, p_pre in _params_pre.items():
                    p_now = dict(model.named_parameters())[name]
                    delta = (p_now.detach() - p_pre).norm().item()
                    total_step_norm += delta * delta
                param_step_diag["grady/param_step_norm"] = total_step_norm ** 0.5
            mark_opt_step(bench)

            c_post_residuals = torch.where(
                is_eq.unsqueeze(0),
                c_post_detached.abs(),
                c_post_detached.clamp(min=0),
            )
            lg.update(c_post_residuals)
            lg_disp.update(disp_mag.detach().unsqueeze(-1))

            step_metrics = {
                "loss": float(loss.item()),
                "loss/objective": float(objective.mean().item()),
                "loss/residual": float(residual_loss.mean().item()),
                "loss/constraint": float(constraint_loss.item()),
                "loss/displacement": float(disp_loss_mean.item()),
                "displacement/mag_mean": float(disp_mag.detach().mean().item()),
                "residual/tau_effective": effective_tau,
                "grad_norm": float(
                    grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
                ),
            }
            for k in range(K):
                step_metrics[f"residual/c_pre_{k}_mean"] = float(
                    c_pre_residuals.detach()[:, k].mean().item()
                )
                step_metrics[f"residual/c_post_{k}_mean"] = float(
                    c_post_residuals[:, k].mean().item()
                )
            step_metrics.update(lg.log_dict("residual"))
            step_metrics.update(lg_disp.log_dict("displacement"))
            if "sqp_qp_failures" in proj_info:
                step_metrics["sqp/qp_failures"] = float(proj_info["sqp_qp_failures"])
                step_metrics["sqp/clarabel_rescues"] = float(
                    proj_info["sqp_clarabel_rescues"]
                )
                step_metrics["sqp/partition_mismatches"] = float(
                    proj_info["sqp_partition_mismatches"]
                )
                step_metrics["sqp/resolve_max_gap"] = float(
                    proj_info["sqp_resolve_max_gap"]
                )
            # ip/alpha_min << 1: boundary quenching; ip/nu_abs_max ~ rho: elastic saturation.
            if "ip_solve_failures" in proj_info:
                for _key in (
                    "solve_failures",
                    "pinv_fallbacks",
                    "alpha_mean",
                    "alpha_min",
                    "mu_mean",
                    "nu_abs_max",
                    "newton_residual_max",
                ):
                    step_metrics[f"ip/{_key}"] = float(proj_info[f"ip_{_key}"])
            step_metrics.update(grad_diag)
            step_metrics.update(param_step_diag)
            if log_grad_share:
                step_metrics[KEY_CON] = gn_con
                step_metrics[KEY_TOT] = gn_tot
            if log_repair:
                step_metrics[KEY_C_PRE] = mean_violation(c_pre_residuals)
                step_metrics[KEY_C_POST] = mean_violation(c_post_residuals)
            if cfg.measure_repair_mem:
                step_metrics["repair_peak_mem_bytes"] = (
                    float(probe["peak_bytes"]) if do_probe else float("nan")
                )
            if is_main:
                logger.log_step(epoch, **step_metrics)
                if epoch == 1 or epoch % 2 == 0:
                    cpre_str = " ".join(
                        f"c_pre_{k}={float(c_pre_residuals.detach()[:, k].mean().item()):.2e}"
                        for k in range(K)
                    )
                    print(
                        f"[{self.name} ep{epoch:>4d}] "
                        f"loss={float(loss.item()):+.3e} "
                        f"obj={float(objective.mean().item()):+.3e} "
                        f"resid={float(residual_loss.mean().item()):.3e} "
                        f"disp={float(disp_loss_mean.item()):.3e} "
                        f"gn={float(grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm):.2e} "
                        f"{cpre_str}",
                        flush=True,
                    )
            loss_trajectory.append(float(loss.item()))

            if on_epoch_end is not None and is_main:
                on_epoch_end(epoch, model)

            if (
                is_main
                and cfg.projection_log_every > 0
                and epoch % cfg.projection_log_every == 0
            ):
                _log_projection_trajectory(
                    bench=bench,
                    projector=projector,
                    constraint_fn=constraint_fn,
                    y_hat=y_hat.detach(),
                    conditions=conditions,
                    step=epoch,
                    phase="train",
                    logger=logger,
                )

            if cfg.eval_every > 0 and epoch % cfg.eval_every == 0:
                _periodic_eval(
                    model=model,
                    bench=bench,
                    constraint_fn=constraint_fn,
                    cfg=cfg,
                    epoch=epoch,
                    logger=logger,
                    local_batch_size=local_batch_size,
                    rank=rank,
                )

            # Rank 0 writes atomically; the barrier keeps ranks from racing ahead.
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
                            "lg": lg.state_dict(),
                            "lg_disp": lg_disp.state_dict(),
                            "loss_trajectory": loss_trajectory,
                        },
                        Path(checkpoint_dir) / "checkpoint_latest.pt",
                    )
                if ddp_active:
                    dist.barrier()

        wall = time.monotonic() - wall_start
        set_meas = getattr(bench, "set_measurement", None)
        if callable(set_meas):
            set_meas(False)
        model_to_save = model.module if isinstance(model, DDP) else model

        eval_queries = bench.eval_queries(seed).to(device)
        with torch.no_grad():
            model_to_save.eval()
            eval_conditions = _eval_conditions(eval_queries, spec.condition_dim)
            final_x = model_to_save(eval_queries.zeta, eval_conditions).detach().cpu()

        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=1,
            model_state={k: v.detach().cpu() for k, v in model_to_save.state_dict().items()},
            train_loss_trajectory=loss_trajectory,
            final_x_on_eval=final_x,
        )
