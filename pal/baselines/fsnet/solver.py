"""FSNet adapter (Nguyen & Donti, arXiv:2506.00362v2): FSNet's MLP plus its L-BFGS feasibility step.

Hparams come from ``pal/baselines/hparams/fsnet.yaml``. ``final_x_on_eval`` is the
post-L-BFGS output, the pre-L-BFGS one is kept in ``extras``.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from pal.baselines._shared import (
    make_forward_fn,
    sample_conditions,
    seed_everything,
)
from pal.baselines.fsnet.upstream.models.neural_networks import MLP
from pal.baselines.fsnet.upstream.utils.lbfgs import (
    hybrid_lbfgs_solve,
    nondiff_lbfgs_solve,
)
from pal.baselines.hparams import load_hparams
from pal.benchmarks.base import Benchmark, Query
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
from pal.utils.peak_memory import probe_peak_memory


@dataclass
class FSNetConfig:
    """Schema for FSNet hparams; values loaded from ``hparams/fsnet.yaml``."""

    seed: int = 0
    device: str = "cpu"

    epochs: int = 100
    steps_per_epoch: int = 20
    batch_size: int = 512
    lr: float = 5e-4
    lr_decay: float = 0.5
    lr_decay_step: int = 2000
    weight_decay: float = 1e-3
    grad_clip: float = 1.0
    dropout: float = 0.0

    # FSNet's MLP, not CoordinationMLP
    hidden_dim: int = 1024
    num_layers: int = 4

    # Hold zeta at zero during training (deterministic NN(x)).
    zeta_zero: bool = False

    obj_weight: float = 1.0
    dist_weight: float = 5.0
    eq_pen_weight: float = 10.0
    ineq_pen_weight: float = 10.0
    val_tol: float = 1e-7
    test_val_tol: float = 1e-9
    decay_tol_step: int = 100
    memory_size: int = 30
    max_iter: int = 50
    max_diff_iter: int = 30
    scale: float = 1000.0

    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100


class _FSNetDataShim:
    """Duck-types FSNet's problem class against a PAL ``Benchmark``.

    ``ineq_resid`` returns ``[B, n_ineq + 2*ydim]`` non-negative violations,
    including the ``(L-y)+`` and ``(y-U)+`` box terms. FSNet's ``x`` is PAL's
    ``conditions``.
    """

    def __init__(self, bench: Benchmark):
        self.bench = bench
        self._forward_fn = make_forward_fn(bench)
        spec = bench.spec
        self.L = spec.output_bounds[0].detach().clone()
        self.U = spec.output_bounds[1].detach().clone()

    def scale(self, y_raw: Tensor) -> Tensor:
        """Map ``[0,1]`` sigmoid output -> feasible box ``[L, U]``."""
        L = self.L.to(y_raw.device, y_raw.dtype)
        U = self.U.to(y_raw.device, y_raw.dtype)
        return L + (U - L) * y_raw

    def _constraint_residuals(
        self, x: Tensor | None, y: Tensor
    ) -> tuple[list[Tensor], list[Tensor]]:
        _, constraints = self._forward_fn(y, x)
        eqs: list[Tensor] = []
        ineqs: list[Tensor] = []
        for c in constraints:
            if c.type == "eq":
                eqs.append(c.value)
            else:
                ineqs.append(c.value.clamp_min(0.0))
        return eqs, ineqs

    def eq_resid(self, x: Tensor | None, y: Tensor) -> Tensor:
        eqs, _ = self._constraint_residuals(x, y)
        if not eqs:
            return y.new_zeros(y.shape[0], 0)
        return torch.stack(eqs, dim=-1)

    def ineq_resid(self, x: Tensor | None, y: Tensor) -> Tensor:
        _, ineqs = self._constraint_residuals(x, y)
        # L-BFGS iterates in unconstrained space, so box violations join the merit.
        L = self.L.to(y.device, y.dtype)
        U = self.U.to(y.device, y.dtype)
        box_lo = (L - y).clamp_min(0.0)   # [B, ydim]
        box_hi = (y - U).clamp_min(0.0)   # [B, ydim]
        parts: list[Tensor] = []
        if ineqs:
            parts.append(torch.stack(ineqs, dim=-1))
        parts.append(box_lo)
        parts.append(box_hi)
        return torch.cat(parts, dim=-1)

    def obj_fn(self, y: Tensor, x: Tensor | None) -> Tensor:
        obj, _ = self._forward_fn(y, x)
        return obj


class FSNetSolver:
    """FSNet baseline: MLP + feasibility-seeking L-BFGS step."""

    name = "fsnet"

    def __init__(self, config: FSNetConfig | None = None):
        if config is None:
            config = FSNetConfig(**load_hparams("fsnet"))
        self.config = config

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Callable[[int, torch.nn.Module], None] | None = None,
        **hp: Any,
    ) -> TrainResult:
        cfg = _merge_cfg(self.config, dict(hp, seed=seed))
        seed_everything(cfg.seed)
        wall_start = time.monotonic()

        spec = bench.spec
        device = cfg.device

        model = MLP(
            input_dim=spec.zeta_dim + spec.condition_dim,
            hidden_dim=cfg.hidden_dim,
            output_dim=spec.dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout,
        ).to(device)
        # No re-seed here: FSNet seeds once at the top.

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            fused=(device == "cuda"),
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=cfg.lr_decay_step, gamma=cfg.lr_decay,
        )

        shim = _FSNetDataShim(bench)
        # Objective normalisation for training only, eval stays in raw units.
        obj_scale = float(getattr(bench, "objective_scale", 1.0))
        val_tol = cfg.val_tol
        loss_trajectory: list[float] = []
        global_step = 0

        for epoch in range(1, cfg.epochs + 1):
            # val_tol decay per trainer.py:326-332.
            if epoch % cfg.decay_tol_step == 0:
                val_tol = float(np.clip(val_tol / 10, 1e-9, 1e-6))

            model.train()

            # Fresh conditions per step in place of FSNet's train_loader (trainer.py:294).
            epoch_losses: list[float] = []
            for step_in_epoch in range(cfg.steps_per_epoch):
                global_step += 1
                optimizer.zero_grad()

                zeta = torch.randn(cfg.batch_size, spec.zeta_dim, device=device)
                if cfg.zeta_zero:
                    zeta = torch.zeros_like(zeta)
                conditions = sample_conditions(
                    bench, cfg.batch_size, device,
                    seed=cfg.seed * 10_000_000 + global_step,
                )
                # Unconditional benches: empty [B, 0] conditions so the cat works.
                if conditions is None:
                    conditions = zeta.new_zeros(cfg.batch_size, 0)
                nn_input = torch.cat([zeta, conditions], dim=-1)

                x = conditions
                y_pred_raw = model(nn_input)            # [B, ydim] in [0, 1]
                y_pred = shim.scale(y_pred_raw)         # rescale to [L, U]

                # Large-violation safeguard inputs (trainer.py:217-221).
                pre_eq  = shim.eq_resid(x, y_pred).square().sum(dim=1)
                pre_ine = shim.ineq_resid(x, y_pred).square().sum(dim=1)

                do_probe = cfg.measure_repair_mem and (
                    epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
                )
                with probe_peak_memory(enabled=do_probe) as probe:
                    y_final = hybrid_lbfgs_solve(
                        x, y_pred, shim,
                        max_diff_iter=cfg.max_diff_iter,
                        val_tol=val_tol,
                        memory=cfg.memory_size,
                        max_iter=cfg.max_iter,
                        scale=cfg.scale,
                    )

                obj = shim.obj_fn(y_final, x) / obj_scale
                distance = (y_final - y_pred).norm(dim=1).square().mean()

                # Outside the large-violation safeguard the loss has no constraint term.
                if pre_eq.mean() >= 1e3 or pre_ine.mean() >= 1e3:
                    con_term = (
                        cfg.eq_pen_weight * pre_eq + cfg.ineq_pen_weight * pre_ine
                    )
                    loss = (
                        cfg.obj_weight * obj
                        + cfg.dist_weight * distance
                        + con_term
                    )
                else:
                    con_term = None
                    loss = cfg.obj_weight * obj + cfg.dist_weight * distance

                loss_mean = loss.mean()
                # Skip the step on a non-finite loss instead of poisoning AdamW.
                if not torch.isfinite(loss_mean):
                    optimizer.zero_grad(set_to_none=True)
                    # skipped: 0=ran, 1=nonfinite_loss, 2=nonfinite_grad
                    skip_metrics = dict(
                        loss=float("nan"),
                        objective=float(obj.mean().item()),
                        pre_eq_violation=float(pre_eq.mean().item()),
                        pre_ineq_violation=float(pre_ine.mean().item()),
                        distance=float(distance.item()),
                        val_tol=val_tol,
                        grad_norm=float("nan"),
                        skipped=1.0,
                    )
                    if cfg.measure_repair_mem:
                        skip_metrics["repair_peak_mem_bytes"] = (
                            float(probe["peak_bytes"]) if do_probe else float("nan")
                        )
                    logger.log_step(global_step, **skip_metrics)
                    continue

                # Gradient share on the last inner step of logged epochs.
                log_grad_share = (
                    step_in_epoch == cfg.steps_per_epoch - 1
                    and should_log_grad_share(epoch, cfg.epochs)
                )
                gn_con = (
                    constraint_grad_norm(
                        con_term.mean() if con_term is not None
                        else torch.zeros((), device=device),
                        model.parameters(),
                    )
                    if log_grad_share
                    else 0.0
                )

                loss_mean.backward()

                gn_tot = param_grad_norm(model.parameters()) if log_grad_share else 0.0
                grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
                # Backprop through L-BFGS can yield non-finite grads even with a finite loss.
                gn_val = (
                    grad_norm.item() if isinstance(grad_norm, Tensor)
                    else float(grad_norm)
                )
                if not math.isfinite(gn_val):
                    optimizer.zero_grad(set_to_none=True)
                    skip_metrics = dict(
                        loss=float(loss_mean.item()),
                        objective=float(obj.mean().item()),
                        pre_eq_violation=float(pre_eq.mean().item()),
                        pre_ineq_violation=float(pre_ine.mean().item()),
                        distance=float(distance.item()),
                        val_tol=val_tol,
                        grad_norm=float("nan"),
                        skipped=2.0,
                    )
                    if cfg.measure_repair_mem:
                        skip_metrics["repair_peak_mem_bytes"] = (
                            float(probe["peak_bytes"]) if do_probe else float("nan")
                        )
                    logger.log_step(global_step, **skip_metrics)
                    continue
                optimizer.step()
                mark_opt_step(bench)

                epoch_losses.append(float(loss_mean.item()))

                step_metrics = {
                    "loss": float(loss_mean.item()),
                    "objective": float(obj.mean().item()),
                    "pre_eq_violation": float(pre_eq.mean().item()),
                    "pre_ineq_violation": float(pre_ine.mean().item()),
                    "distance": float(distance.item()),
                    "val_tol": val_tol,
                    "grad_norm": float(
                        grad_norm.item() if isinstance(grad_norm, Tensor)
                        else grad_norm
                    ),
                    "skipped": 0.0,
                }
                if cfg.measure_repair_mem:
                    step_metrics["repair_peak_mem_bytes"] = (
                        float(probe["peak_bytes"]) if do_probe else float("nan")
                    )
                if log_grad_share:
                    step_metrics[KEY_CON] = gn_con
                    step_metrics[KEY_TOT] = gn_tot
                logger.log_step(global_step, **step_metrics)

            scheduler.step()  # per-epoch, matches trainer.py:310
            loss_trajectory.append(
                float(np.mean(epoch_losses)) if epoch_losses else float("nan")
            )

            if on_epoch_end is not None:
                on_epoch_end(epoch, model)

        wall = time.monotonic() - wall_start

        model.eval()
        eval_queries = bench.eval_queries(seed).to(device)
        raw_eval, post_eval, _iters = self._run_prediction(
            model, shim, eval_queries,
            val_tol=cfg.test_val_tol, memory=cfg.memory_size,
            max_iter=cfg.max_iter, scale=cfg.scale,
        )
        final_x_on_eval = post_eval.detach().cpu()

        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=1,
            model_state={k: v.detach().cpu() for k, v in model.state_dict().items()},
            train_loss_trajectory=loss_trajectory,
            final_x_on_eval=final_x_on_eval,
            extras={"final_x_pre_lbfgs": raw_eval.detach().cpu()},
        )

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        spec = bench.spec
        model = MLP(
            input_dim=spec.zeta_dim + spec.condition_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=spec.dim,
            num_layers=self.config.num_layers,
            dropout=self.config.dropout,
        )
        if train_result.model_state is None:
            raise RuntimeError("FSNetSolver.predict: missing model_state")
        model.load_state_dict(train_result.model_state)
        model.eval()

        shim = _FSNetDataShim(bench)
        raw, post, iters_taken = self._run_prediction(
            model, shim, queries,
            val_tol=self.config.test_val_tol,
            memory=self.config.memory_size,
            max_iter=self.config.max_iter,
            scale=self.config.scale,
        )
        n_queries = len(queries)
        inf_iters = torch.full(
            (n_queries,), int(iters_taken), dtype=torch.int32, device="cpu",
        )
        return PredictionOutputs(
            raw=raw, post=post, projection=None, inference_iters=inf_iters,
        )

    @staticmethod
    def _run_prediction(
        model: torch.nn.Module,
        shim: _FSNetDataShim,
        queries: Query,
        *,
        val_tol: float,
        memory: int,
        max_iter: int,
        scale: float,
    ) -> tuple[Tensor, Tensor, int]:
        """Shared inference path: MLP -> scale -> non-differentiable L-BFGS.

        Returns ``(raw, post, iters_taken)``, where ``iters_taken`` is the
        batch-wide L-BFGS iteration count at break (or ``max_iter``).
        """
        zeta = queries.zeta
        conditions = queries.conditions
        # conditions is [N, 0] for unconditional benches, cat still works.
        nn_input = torch.cat([zeta, conditions], dim=-1)
        x = conditions
        with torch.no_grad():
            raw_unscaled = model(nn_input)
            raw = shim.scale(raw_unscaled)
        iters_box: dict = {}
        post = nondiff_lbfgs_solve(
            x, raw.clone(), shim,
            val_tol=val_tol,
            memory=memory,
            max_iter=max_iter,
            scale=scale,
            iters_out=iters_box,
        )
        return raw.detach(), post.detach(), int(iters_box.get("iters", max_iter))


def _merge_cfg(base: FSNetConfig, overrides: dict[str, Any]) -> FSNetConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(
            f"FSNetSolver.train: unknown hparam(s) {sorted(unknown)}"
        )
    return FSNetConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
