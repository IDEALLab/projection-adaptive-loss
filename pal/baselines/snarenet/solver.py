"""SnareNet adapter (Chu, Boukas & Udell, arXiv:2602.09317v1) around the vendored upstream source.

Self-supervised loss: f(y_repaired) + soft_weight * ||residual(y_repaired)||^2.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from pal.baselines._shared import seed_everything
from pal.baselines.snarenet._adaptive_relaxation import AdaptiveRelaxation
from pal.baselines.snarenet._cfg_proxy import _make_cfg_proxy
from pal.baselines.snarenet._upstream_loader import (
    load_vendored,
)
from pal.baselines.snarenet.data_shim import _SnareNetDataShim
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
from pal.solvers.repair_contraction import (
    KEY_C_POST,
    KEY_C_PRE,
    mean_violation,
    repair_contraction_enabled,
    should_log_repair_contraction,
)
from pal.tracking.base import Logger
from pal.utils.peak_memory import probe_peak_memory

_BaseModel, _SnareNet, _SnareNetRepairLayer = load_vendored()


@dataclass
class SnareNetConfig:
    """Frozen noncvx-derived defaults. All citations to miniyachi/SnareNet@203f1363."""

    seed: int = 0
    device: str = "cpu"

    # configs/experiment/snarenet_noncvx.yaml
    epochs: int = 2000                 # snarenet_noncvx.yaml:5
    batch_size: int = 200              # snarenet_noncvx.yaml:6
    learning_rate: float = 1e-4        # snarenet_noncvx.yaml:7 (== config.yaml:21)
    soft_epochs: int = 0               # config.yaml:24
    soft_weight: float = 10.0          # config.yaml:25

    # configs/model/snarenet.yaml; the paper setting is hidden_size=200, num_hidden_layers=2.
    hidden_size: int = 512
    num_hidden_layers: int = 4
    dropout: float = 0.2               # snarenet.yaml:7
    batchnorm_dropout: bool = True     # snarenet.yaml:8

    newton_maxiter: int = 100          # snarenet.yaml:12
    rtol: float = 1e-8                 # snarenet.yaml:13
    lambd: float = 1e-2                # snarenet_noncvx.yaml:10 (== snarenet.yaml:16)

    # snarenet_noncvx.yaml:12-14
    adaptive_relaxation: bool = True
    decay_epochs: int = 500
    decay_schedule: str = "linear"     # 'linear' | 'harmonic' | 'linear_harmonic'
    n_calibration_batches: int = 45

    # Hold zeta at zero during training and calibration (deterministic NN(x)).
    zeta_zero: bool = False

    trust_region: bool = False         # snarenet_noncvx.yaml:11
    is_cg: bool = False                # snarenet.yaml:27
    cg_maxiter: int = 10

    # Newton-repair Jacobian strategy, see `DC3Config.jacobian_mode`.
    jacobian_mode: str = "sample"
    measurement_window_epochs: int = 5
    measurement_period_epochs: int = 100

    # Peak-memory probe; the repair is fused into `net(x)`, so this covers forward+repair.
    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100


def _compute_in_window(epoch: int, cfg: SnareNetConfig) -> bool:
    period = max(1, int(cfg.measurement_period_epochs))
    window = max(0, int(cfg.measurement_window_epochs))
    return ((epoch - 1) % period) < window


def _apply_jacobian_schedule(
    cfg: SnareNetConfig,
    shim: _SnareNetDataShim,
    bench,
    epoch: int,
) -> None:
    """Flip shim mode + probe measurement flag for this epoch's Newton solves."""
    mode = str(cfg.jacobian_mode)
    if mode == "loop":
        loop = True
    elif mode in ("vmap", "vmap_jacrev"):
        loop = False
    elif mode == "sample":
        loop = _compute_in_window(epoch, cfg)
    else:
        raise ValueError(
            f"unknown SnareNet jacobian_mode {mode!r}; "
            "expected 'loop', 'vmap', or 'sample'"
        )
    shim.set_jacobian_mode("loop" if loop else "vmap")
    set_meas = getattr(bench, "set_measurement", None)
    if callable(set_meas):
        set_meas(loop)


class SnareNetSolver:
    """SnareNet baseline: BaseModel backbone + Newton-pinv repair."""

    name = "snarenet"

    def __init__(self, config: SnareNetConfig | None = None):
        self.config = config or SnareNetConfig()

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

        # Opt-in memory-probe hack: BatchNorm in eval mode so BS=1 does not raise or NaN.
        import os as _os
        _force_bn_eval = (
            _os.environ.get("PAL_MEM_PROBE_DISABLE_BN_BS_CHECK") == "1"
        )

        device = cfg.device

        # Zeta is concatenated into the MLP input so unconditional benches get a non-constant input.
        q0 = bench.sample_queries(
            cfg.batch_size, split="train", seed=cfg.seed,
        )
        bench_x0 = q0.conditions.to(device=device)
        zeta0 = q0.zeta.to(device=device)
        if cfg.zeta_zero:
            zeta0 = torch.zeros_like(zeta0)
        model_x0 = torch.cat([zeta0, bench_x0], dim=-1)

        shim = _SnareNetDataShim(
            bench, model_x=model_x0, bench_x=bench_x0, device=device,
        )
        cfg_proxy = _make_cfg_proxy(cfg, prob_type="noncvx")

        net = _SnareNet(shim, cfg_proxy).to(device)
        seed_everything(cfg.seed)

        optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learning_rate)

        ar_handler: AdaptiveRelaxation | None = None
        if cfg.adaptive_relaxation:
            ar_handler = AdaptiveRelaxation(
                start_epoch=cfg.soft_epochs,
                decay_epochs=cfg.decay_epochs,
                device=device,
                decay_fn=cfg.decay_schedule,
            )

        loss_trajectory: list[float] = []
        obj_scale = float(getattr(bench, "objective_scale", 1.0))

        # Forward hook captures the pre-repair backbone output without re-running dropout/BatchNorm.
        raw_box: list[torch.Tensor] = []
        raw_hook_handle: RemovableHandle | None = None

        def _capture_backbone_out(_module, _inputs, output: torch.Tensor) -> None:
            raw_box.clear()
            raw_box.append(output.detach())

        base_module = getattr(net, "_base", None)
        if base_module is not None and repair_contraction_enabled():
            raw_hook_handle = base_module.register_forward_hook(
                _capture_backbone_out
            )

        for epoch in range(1, cfg.epochs + 1):
            _apply_jacobian_schedule(cfg, shim, bench, epoch)
            net.train()
            if _force_bn_eval:
                for _m in net.modules():
                    if isinstance(_m, torch.nn.modules.batchnorm._BatchNorm):
                        _m.eval()
            net.set_repair(epoch >= cfg.soft_epochs)

            q = bench.sample_queries(
                cfg.batch_size, split="train",
                seed=cfg.seed * 10_000_000 + epoch,
            )
            bench_x = q.conditions.to(device=device)
            zeta = q.zeta.to(device=device)
            if cfg.zeta_zero:
                zeta = torch.zeros_like(zeta)
            x = torch.cat([zeta, bench_x], dim=-1)
            shim.update_x(x, bench_x)

            if ar_handler is not None and epoch >= cfg.soft_epochs:
                if not ar_handler.initialized:
                    ar_handler.get_init_eps(
                        shim, net, bench,
                        seed=cfg.seed * 10_000_000 - 1,
                        n_batches=cfg.n_calibration_batches,
                        batch_size=cfg.batch_size,
                        zeta_zero=cfg.zeta_zero,
                    )
                    # Calibration mutates shim state; restore this epoch's x.
                    shim.update_x(x, bench_x)
                net.set_eps(ar_handler.get_eps(epoch).detach())

            optimizer.zero_grad()
            do_probe = cfg.measure_repair_mem and (
                epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
            )
            with probe_peak_memory(enabled=do_probe) as probe:
                y = net(x)                                      # repair runs inside
            obj = shim.evaluate(x, y) / obj_scale               # [B]
            resid = shim.get_resid(x, y)                        # [B, K] >= 0
            resid_sq = resid.square().sum(dim=1)                # [B]  (sum over K)

            per_sample_loss = obj + cfg.soft_weight * resid_sq
            loss = per_sample_loss.sum()

            if not torch.isfinite(loss):
                # Log one row first so the memory column has a measurement.
                if cfg.measure_repair_mem:
                    logger.log_step(
                        epoch,
                        loss=float("nan"),
                        objective=float(obj.mean().item()),
                        residual_max=float("nan"),
                        repair_iters=int(net.get_iter_taken()),
                        eps_max=0.0,
                        repair_peak_mem_bytes=(
                            float(probe["peak_bytes"]) if do_probe else float("nan")
                        ),
                    )
                raise RuntimeError(
                    f"epoch {epoch}: non-finite loss "
                    f"(obj={obj.mean().item():.3e}, "
                    f"resid_sq={resid_sq.mean().item():.3e})"
                )

            # Constraint part of the gradient = the soft-weighted residual term.
            log_grad_share = should_log_grad_share(epoch, cfg.epochs)
            gn_con = (
                constraint_grad_norm(
                    (cfg.soft_weight * resid_sq).sum(), net.parameters()
                )
                if log_grad_share
                else 0.0
            )

            # Pre-repair residual at the hook-captured backbone output.
            log_repair = should_log_repair_contraction(epoch, cfg.epochs)
            c_pre_val = c_post_val = 0.0
            if log_repair:
                c_post_val = mean_violation(resid)
                if epoch >= cfg.soft_epochs and raw_box:
                    with torch.no_grad():
                        c_pre_val = mean_violation(
                            shim.get_resid(x, raw_box[0])
                        )
                else:
                    c_pre_val = c_post_val

            loss.backward()

            gn_tot = param_grad_norm(net.parameters()) if log_grad_share else 0.0
            optimizer.step()
            mark_opt_step(bench)

            eps_max_val = 0.0
            if ar_handler is not None and ar_handler.initialized:
                eps_t = net.get_eps()
                if eps_t.numel():
                    eps_max_val = float(eps_t.max().item())

            log_kwargs = dict(
                loss=float(loss.item()),
                loss_per_sample=float(per_sample_loss.mean().item()),
                objective=float(obj.mean().item()),
                residual_max=(
                    float(resid_sq.sqrt().max().item())
                    if resid_sq.numel() else 0.0
                ),
                repair_iters=int(net.get_iter_taken()),
                eps_max=eps_max_val,
            )
            if cfg.measure_repair_mem:
                log_kwargs["repair_peak_mem_bytes"] = (
                    float(probe["peak_bytes"]) if do_probe else float("nan")
                )
            if log_grad_share:
                log_kwargs[KEY_CON] = gn_con
                log_kwargs[KEY_TOT] = gn_tot
            if log_repair:
                log_kwargs[KEY_C_PRE] = c_pre_val
                log_kwargs[KEY_C_POST] = c_post_val
            logger.log_step(epoch, **log_kwargs)
            loss_trajectory.append(float(loss.item()))

            if on_epoch_end is not None:
                on_epoch_end(epoch, net)

        wall = time.monotonic() - wall_start

        if raw_hook_handle is not None:
            raw_hook_handle.remove()
        raw_box.clear()

        # Loop-mode Jacobians for downstream eval.
        shim.set_jacobian_mode("loop")
        set_meas = getattr(bench, "set_measurement", None)
        if callable(set_meas):
            set_meas(False)

        net.eval()
        eval_queries = bench.eval_queries(seed).to(device)
        bench_x_eval = eval_queries.conditions.to(device=device)
        x_eval = torch.cat(
            [eval_queries.zeta.to(device=device), bench_x_eval], dim=-1,
        )
        shim.update_x(x_eval, bench_x_eval)
        with torch.no_grad():
            net.set_repair(True)
            final_post = net(x_eval).detach().cpu()

        # `_eps` can change shape to a 0-d scalar during decay, so it is not saved.
        state = {
            k: v.detach().cpu() for k, v in net.state_dict().items()
            if not k.endswith("._eps") and k != "_eps"
        }

        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=1,
            model_state=state,
            train_loss_trajectory=loss_trajectory,
            final_x_on_eval=final_post,
        )

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        cfg = self.config
        device = cfg.device

        bench_x = queries.conditions.to(device=device)
        x = torch.cat([queries.zeta.to(device=device), bench_x], dim=-1)
        shim = _SnareNetDataShim(
            bench, model_x=x, bench_x=bench_x, device=device,
        )
        cfg_proxy = _make_cfg_proxy(cfg, prob_type="noncvx")

        net = _SnareNet(shim, cfg_proxy).to(device)
        if train_result.model_state is None:
            raise RuntimeError("SnareNetSolver.predict: missing model_state")
        # strict=False: the unsaved `_eps` buffer defaults to zeros, correct after decay.
        net.load_state_dict(train_result.model_state, strict=False)
        net.eval()

        with torch.no_grad():
            net.set_repair(False)
            raw = net(x).detach()
            net.set_repair(True)
            post = net(x).detach()
            iters_taken = int(net.get_iter_taken())

        n_queries = len(queries)
        inf_iters = torch.full(
            (n_queries,), iters_taken, dtype=torch.int32, device="cpu",
        )
        return PredictionOutputs(
            raw=raw, post=post, projection=None, inference_iters=inf_iters,
        )


def _merge_cfg(base: SnareNetConfig, overrides: dict[str, Any]) -> SnareNetConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(
            f"SnareNetSolver.train: unknown hparam(s) {sorted(unknown)}"
        )
    return SnareNetConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
