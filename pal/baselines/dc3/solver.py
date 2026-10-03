"""DC3 adapter (Donti, Rolnick & Kolter, ICLR 2021, arXiv:2104.12225v1).

The algorithmic core (``grad_steps``, ``grad_steps_all``, ``total_loss``) is
imported unmodified from ``upstream/method.py``. Hparams come from ``dc3.yaml``,
or ``dc3_acopf.yaml`` for ``e3/*``.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor
from torch.nn import Module

from pal.baselines._shared import (
    eval_conditions,
    output_bounds_list,
    sample_conditions,
    seed_everything,
)
from pal.baselines.dc3._upstream_loader import load_vendored
from pal.baselines.dc3.bench_specs import BenchDC3Spec, resolve_partial_vars
from pal.baselines.dc3.data_shim import _DC3DataShim
from pal.baselines.hparams import load_hparams
from pal.benchmarks.base import Benchmark, Query
from pal.model import DC3MLP, CoordinationMLP
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
class DC3Config:
    """Type-checked schema for DC3 hparams. Values load from YAML."""

    seed: int = 0
    device: str = "cpu"

    epochs: int = 1000
    batch_size: int = 200
    lr: float = 1e-4
    hidden: int = 200
    n_layers: int = 2
    grad_clip: float = 0.0

    soft_weight: float = 10.0
    soft_eq_frac: float = 0.5
    use_train_corr: bool = True
    use_test_corr: bool = True
    corr_train_steps: int = 10
    corr_test_max_steps: int = 10
    corr_eps: float = 1e-4
    corr_lr: float = 1e-7
    corr_momentum: float = 0.5

    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100

    # Fixed training pool as in the paper; 0 samples fresh conditions per step.
    train_dataset_size: int = 0

    newton_max_iter: int = 20
    newton_tol: float = 1e-6
    newton_reg: float = 1e-8
    # Yamashita-Fukushima LM damping, mu = reg + c*||h||^2 (0 = undamped).
    newton_yf_damping: float = 0.0
    # Constant LM damping (mu = reg + c_const + c_yf*||h||^2). 0.0 = off.
    newton_lm_damping: float = 0.0
    # Force the completion strategy; "" lets the bench registry decide.
    completion_strategy_override: str = ""

    # "loop": explicit per-row VJPs (counted by the probe), "vmap": vmap(jacrev),
    # "sample": loop inside measurement windows, vmap elsewhere.
    jacobian_mode: str = "sample"
    measurement_window_epochs: int = 5
    measurement_period_epochs: int = 100


def _hparam_tier_for(bench_id: str | None) -> str:
    return "dc3_acopf" if (bench_id and bench_id.startswith("e3/")) else "dc3"


def _compute_in_window(epoch: int, cfg: DC3Config) -> bool:
    """Return True when `epoch` falls inside a measurement window.

    Windows are `cfg.measurement_window_epochs` long, repeating with
    period `cfg.measurement_period_epochs`. Epoch 1 is the start of the
    first window.
    """
    period = max(1, int(cfg.measurement_period_epochs))
    window = max(0, int(cfg.measurement_window_epochs))
    return ((epoch - 1) % period) < window


def _apply_jacobian_schedule(
    cfg: DC3Config,
    shim: _DC3DataShim,
    bench,
    epoch: int,
) -> None:
    """Flip shim mode + probe measurement flag for this epoch's training."""
    mode = str(cfg.jacobian_mode)
    if mode == "loop":
        loop = True
    elif mode in ("vmap", "vmap_jacrev"):
        loop = False
    elif mode == "sample":
        loop = _compute_in_window(epoch, cfg)
    else:
        raise ValueError(
            f"unknown DC3 jacobian_mode {mode!r}; "
            "expected 'loop', 'vmap', or 'sample'"
        )
    shim.set_jacobian_mode("loop" if loop else "vmap")
    set_meas = getattr(bench, "set_measurement", None)
    if callable(set_meas):
        set_meas(loop)


def _slice_output_bounds(
    spec, partial_vars: list[int]
) -> list[tuple[float, float]]:
    """Slice ``spec.output_bounds`` to the partial-vars rows as float pairs."""
    lo, hi = spec.output_bounds
    idx = torch.as_tensor(partial_vars, dtype=torch.long)
    lo_p = lo.index_select(0, idx).tolist()
    hi_p = hi.index_select(0, idx).tolist()
    return [(float(a), float(b)) for a, b in zip(lo_p, hi_p, strict=False)]


def _dc3_args_dict(
    cfg: DC3Config, *, use_compl: bool, corr_mode: str
) -> dict[str, Any]:
    """Build the camelCase ``args`` dict read by DC3's ``grad_steps`` / ``total_loss``."""
    # Upstream default_args.py: "use 100 if useCompl=False".
    soft_weight_eff = cfg.soft_weight * (1.0 if use_compl else 10.0)
    return {
        "useCompl": bool(use_compl),
        "corrMode": corr_mode,
        "useTrainCorr": bool(cfg.use_train_corr),
        "useTestCorr": bool(cfg.use_test_corr),
        "corrTrainSteps": int(cfg.corr_train_steps),
        "corrTestMaxSteps": int(cfg.corr_test_max_steps),
        "corrEps": float(cfg.corr_eps),
        "corrLr": float(cfg.corr_lr),
        "corrMomentum": float(cfg.corr_momentum),
        "softWeight": float(soft_weight_eff),
        "softWeightEqFrac": float(cfg.soft_eq_frac),
    }


def _dc3_constraint_loss(shim, X: Tensor, Y: Tensor, args: dict[str, Any]) -> Tensor:
    """Per-sample constraint part of DC3's ``total_loss`` (the two soft-weighted terms)."""
    ineq_cost = torch.norm(shim.ineq_dist(X, Y), dim=1)
    eq_cost = torch.norm(shim.eq_resid(X, Y), dim=1)
    return (
        args["softWeight"] * (1 - args["softWeightEqFrac"]) * ineq_cost
        + args["softWeight"] * args["softWeightEqFrac"] * eq_cost
    )


def _dc3_violation(shim, X: Tensor, Y: Tensor) -> Tensor:
    """Per-(sample, constraint) violation magnitude at ``Y``, as ``[B, K]``."""
    return torch.cat(
        [shim.ineq_dist(X, Y), shim.eq_resid(X, Y).abs()], dim=1
    )


class DC3Solver:
    name = "dc3"

    def __init__(
        self,
        config: DC3Config | None = None,
        bench_id: str | None = None,
    ):
        if config is None:
            tier = _hparam_tier_for(bench_id)
            config = DC3Config(**load_hparams(tier))
        self.config = config
        self._bench_id = bench_id

    def _resolve(self, bench: Benchmark):
        """Return ``(spec, use_compl, partial_vars, other_vars, model_dim_out, bounds)``.

        ``other_vars`` excludes the spec's ``known_vars``, so the three sets
        partition ``range(ydim)``.
        """
        spec = bench.spec
        dc3_spec: BenchDC3Spec | None = resolve_partial_vars(bench)

        if dc3_spec is None:
            return (
                None, False, [], [],
                spec.dim, output_bounds_list(spec),
            )

        partial_vars = list(dc3_spec.partial_vars)
        known_vars = list(dc3_spec.known_vars)
        reserved = set(partial_vars) | set(known_vars)
        other_vars = sorted(set(range(spec.dim)) - reserved)
        bounds = _slice_output_bounds(spec, partial_vars)
        return (
            dc3_spec, True, partial_vars, other_vars,
            len(partial_vars), bounds,
        )

    def _build_shim(
        self,
        bench: Benchmark,
        dc3_spec: BenchDC3Spec | None,
        partial_vars: list[int],
        other_vars: list[int],
        device: str,
    ) -> _DC3DataShim:
        newton_reg = dc3_spec.newton_reg if dc3_spec is not None else self.config.newton_reg
        warm_fn = dc3_spec.warm_start_fn if dc3_spec is not None else None
        linear = bool(dc3_spec.linear) if dc3_spec is not None else False
        strategy = (
            dc3_spec.completion_strategy if dc3_spec is not None else "generic_newton"
        )
        if self.config.completion_strategy_override:
            strategy = self.config.completion_strategy_override
            linear = strategy == "linear"
            newton_reg = self.config.newton_reg
        known_vars = list(dc3_spec.known_vars) if dc3_spec is not None else []
        known_values = list(dc3_spec.known_values) if dc3_spec is not None else []
        acopf_partition = (
            dc3_spec.meta.get("acopf_partition")
            if dc3_spec is not None else None
        )
        shim = _DC3DataShim(
            bench,
            partial_vars=partial_vars,
            other_vars=other_vars,
            linear=linear,
            newton_max_iter=self.config.newton_max_iter,
            newton_tol=self.config.newton_tol,
            newton_reg=newton_reg,
            newton_yf_damping=self.config.newton_yf_damping,
            newton_lm_damping=self.config.newton_lm_damping,
            warm_start_fn=warm_fn,
            warm_start_ctx={"bench": bench},
            known_vars=known_vars,
            known_values=known_values,
            completion_strategy=strategy,
            acopf_partition=acopf_partition,
        )
        shim.set_device(device)
        return shim

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

        # fp64 as in the paper: fp32 Jacobian inverses diverge on stiff ACOPF buses.
        _prev_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        try:
            return self._train_impl(
                bench, seed, logger, on_epoch_end, cfg, wall_start, spec, device,
            )
        finally:
            torch.set_default_dtype(_prev_dtype)

    def _train_impl(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Callable[[int, Module], None] | None,
        cfg,
        wall_start: float,
        spec,
        device: str,
    ) -> TrainResult:
        grad_steps, _grad_steps_all, total_loss = load_vendored()

        dc3_spec, use_compl, partial_vars, other_vars, nn_out_dim, bounds = (
            self._resolve(bench)
        )
        corr_mode = "partial" if use_compl else "full"

        # A fixed training pool selects the paper's DC3MLP backbone.
        use_paper_backbone = cfg.train_dataset_size > 0
        if use_paper_backbone:
            # Paper sizes; cfg.hidden / cfg.n_layers apply to CoordinationMLP only.
            model = DC3MLP(
                dim_zeta=spec.zeta_dim,
                dim_conditions=spec.condition_dim,
                dim_output=nn_out_dim,
                output_bounds=bounds,
                hidden=200,
                n_layers=2,
            ).to(device)
        else:
            model = CoordinationMLP(
                dim_zeta=spec.zeta_dim,
                dim_conditions=spec.condition_dim,
                dim_output=nn_out_dim,
                output_bounds=bounds,
                hidden=cfg.hidden,
                n_layers=cfg.n_layers,
                output_init_std=spec.model_hparams.get("output_init_std"),
            ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        shim = self._build_shim(bench, dc3_spec, partial_vars, other_vars, device)
        args = _dc3_args_dict(cfg, use_compl=use_compl, corr_mode=corr_mode)

        loss_trajectory: list[float] = []

        if use_paper_backbone:
            # Upstream's PFFunction accepts any completion, so disable the
            # divergence gates during training (restored for eval).
            shim._accept_floor_override = 1e10

            # Upstream objective normalization: divide obj by genbase.mean()**2.
            genbase = bench._data.get("genbase")
            if genbase is not None:
                shim._obj_scale = float(genbase.float().mean().item() ** 2)
            shim._in_loop_cap_override = None
            pool_query = bench.sample_queries(
                n=cfg.train_dataset_size, split="train", seed=int(cfg.seed),
            )
            # DC3 is deterministic in the conditions, so hold zeta at zero.
            pool_zeta = torch.zeros_like(pool_query.zeta).to(
                device=device, dtype=torch.float64,
            )
            pool_conditions = pool_query.conditions
            if pool_conditions is not None:
                pool_conditions = pool_conditions.to(
                    device=device, dtype=torch.float64,
                )
            n_pool = pool_zeta.shape[0]
            step_global = 0
            from pal.baselines.dc3._completion import CompletionDivergedError
            skipped_batches = 0

            for epoch in range(1, cfg.epochs + 1):
                _apply_jacobian_schedule(cfg, shim, bench, epoch)
                g = torch.Generator("cpu").manual_seed(
                    int(cfg.seed) * 10_000_000 + epoch,
                )
                perm = torch.randperm(n_pool, generator=g).to(device)
                for start in range(0, n_pool, cfg.batch_size):
                    idx = perm[start : start + cfg.batch_size]
                    zeta = pool_zeta.index_select(0, idx)
                    if pool_conditions is not None:
                        conditions = pool_conditions.index_select(0, idx)
                        X = conditions
                    else:
                        conditions = None
                        X = torch.zeros(zeta.shape[0], 0, device=device)
                    shim.bind_x(X)

                    model.train()
                    optimizer.zero_grad()
                    nn_out = model(zeta, conditions)
                    do_probe = cfg.measure_repair_mem and start == 0 and (
                        epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
                    )
                    with probe_peak_memory(enabled=do_probe) as probe:
                        try:
                            if use_compl:
                                Yhat = shim.complete_partial(X, nn_out)
                            else:
                                Yhat = nn_out
                        except CompletionDivergedError:
                            skipped_batches += 1
                            optimizer.zero_grad()
                            continue
                        Ynew = grad_steps(shim, X, Yhat, args)
                    # Upstream sums over the batch; the lr is tuned for that scale.
                    loss = total_loss(shim, X, Ynew, args).sum()

                    if not torch.isfinite(loss):
                        raise RuntimeError(
                            f"epoch {epoch}: non-finite DC3 loss "
                            f"(loss={float(loss.item())})"
                        )
                    log_grad_share = start == 0 and should_log_grad_share(
                        epoch, cfg.epochs
                    )
                    gn_con = (
                        constraint_grad_norm(
                            _dc3_constraint_loss(shim, X, Ynew, args).sum(),
                            model.parameters(),
                        )
                        if log_grad_share
                        else 0.0
                    )
                    # Pre = completion output, post = corrected point fed to the loss.
                    log_repair = start == 0 and should_log_repair_contraction(
                        epoch, cfg.epochs
                    )
                    c_pre_val = c_post_val = 0.0
                    if log_repair:
                        with torch.no_grad():
                            c_pre_val = mean_violation(
                                _dc3_violation(shim, X, Yhat.detach())
                            )
                            c_post_val = mean_violation(
                                _dc3_violation(shim, X, Ynew.detach())
                            )
                    loss.backward()
                    gn_tot = (
                        param_grad_norm(model.parameters())
                        if log_grad_share
                        else 0.0
                    )
                    grad_norm_val = 0.0
                    if cfg.grad_clip > 0:
                        gval = torch.nn.utils.clip_grad_norm_(
                            model.parameters(), cfg.grad_clip
                        )
                        grad_norm_val = float(
                            gval.item() if isinstance(gval, Tensor) else gval
                        )
                    optimizer.step()
                    mark_opt_step(bench)
                    step_global += 1
                    log_kwargs = dict(
                        loss=float(loss.item()),
                        grad_norm=grad_norm_val,
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
                    on_epoch_end(epoch, model)
        else:
            for epoch in range(1, cfg.epochs + 1):
                _apply_jacobian_schedule(cfg, shim, bench, epoch)
                model.train()
                optimizer.zero_grad()

                zeta = torch.randn(cfg.batch_size, spec.zeta_dim, device=device)
                conditions = sample_conditions(
                    bench, cfg.batch_size, device,
                    seed=cfg.seed * 10_000_000 + epoch,
                )
                if conditions is not None:
                    conditions = conditions.to(dtype=torch.float64)

                if conditions is not None:
                    X = conditions
                else:
                    X = torch.zeros(cfg.batch_size, 0, device=device)
                shim.bind_x(X)

                nn_out = model(zeta, conditions)
                do_probe = cfg.measure_repair_mem and (
                    epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
                )
                with probe_peak_memory(enabled=do_probe) as probe:
                    if use_compl:
                        Yhat = shim.complete_partial(X, nn_out)
                    else:
                        Yhat = nn_out
                    Ynew = grad_steps(shim, X, Yhat, args)
                loss = total_loss(shim, X, Ynew, args).mean()

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        f"epoch {epoch}: non-finite DC3 loss "
                        f"(loss={float(loss.item())})"
                    )

                log_grad_share = should_log_grad_share(epoch, cfg.epochs)
                gn_con = (
                    constraint_grad_norm(
                        _dc3_constraint_loss(shim, X, Ynew, args).mean(),
                        model.parameters(),
                    )
                    if log_grad_share
                    else 0.0
                )
                log_repair = should_log_repair_contraction(epoch, cfg.epochs)
                c_pre_val = c_post_val = 0.0
                if log_repair:
                    with torch.no_grad():
                        c_pre_val = mean_violation(
                            _dc3_violation(shim, X, Yhat.detach())
                        )
                        c_post_val = mean_violation(
                            _dc3_violation(shim, X, Ynew.detach())
                        )
                loss.backward()
                gn_tot = (
                    param_grad_norm(model.parameters()) if log_grad_share else 0.0
                )
                grad_norm_val = 0.0
                if cfg.grad_clip > 0:
                    g = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), cfg.grad_clip
                    )
                    grad_norm_val = float(g.item() if isinstance(g, Tensor) else g)
                optimizer.step()
                mark_opt_step(bench)

                step_metrics = {
                    "loss": float(loss.item()),
                    "grad_norm": grad_norm_val,
                }
                if cfg.measure_repair_mem:
                    step_metrics["repair_peak_mem_bytes"] = (
                        float(probe["peak_bytes"]) if do_probe else float("nan")
                    )
                if log_grad_share:
                    step_metrics[KEY_CON] = gn_con
                    step_metrics[KEY_TOT] = gn_tot
                if log_repair:
                    step_metrics[KEY_C_PRE] = c_pre_val
                    step_metrics[KEY_C_POST] = c_post_val
                logger.log_step(epoch, **step_metrics)
                loss_trajectory.append(float(loss.item()))

                if on_epoch_end is not None:
                    on_epoch_end(epoch, model)

        wall = time.monotonic() - wall_start

        shim.set_jacobian_mode("loop")
        set_meas = getattr(bench, "set_measurement", None)
        if callable(set_meas):
            set_meas(False)

        # Restore the tight completion gates for eval.
        if use_paper_backbone:
            shim._accept_floor_override = None
            shim._in_loop_cap_override = 1e3
            shim._obj_scale = 1.0  # report eval obj in original units

        if use_paper_backbone:
            eval_queries = bench.sample_queries(
                n=int(spec.n_eval_default), split="eval", seed=int(seed),
            ).to(device)
            eval_queries = Query(
                zeta=torch.zeros_like(eval_queries.zeta),
                conditions=eval_queries.conditions,
            )
        else:
            eval_queries = bench.eval_queries(seed).to(device)
        eval_queries = Query(
            zeta=eval_queries.zeta.to(dtype=torch.float64),
            conditions=(
                eval_queries.conditions.to(dtype=torch.float64)
                if eval_queries.conditions is not None and eval_queries.conditions.numel() > 0
                else eval_queries.conditions
            ),
        )
        with torch.no_grad():
            # DC3MLP stays in train mode: its BN running stats drift under the
            # sum-scaled updates and give pathological eval outputs.
            if not use_paper_backbone:
                model.eval()
            eval_conds = eval_conditions(eval_queries, spec.condition_dim)
            raw_eval = model(eval_queries.zeta, eval_conds).detach()
        if use_compl:
            # Completion must not be inside no_grad: newton_complete uses jacrev.
            X_eval = eval_queries.conditions if eval_conds is not None else torch.zeros(
                raw_eval.shape[0], 0, device=device,
            )
            shim.bind_x(X_eval)
            final_x = shim.complete_partial(X_eval, raw_eval).detach().cpu()
        else:
            final_x = raw_eval.cpu()

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
        cfg = self.config
        spec = bench.spec
        device = cfg.device

        grad_steps, grad_steps_all, _total_loss = load_vendored()

        dc3_spec, use_compl, partial_vars, other_vars, nn_out_dim, bounds = (
            self._resolve(bench)
        )
        corr_mode = "partial" if use_compl else "full"

        use_paper_backbone = cfg.train_dataset_size > 0
        if use_paper_backbone:
            model = DC3MLP(
                dim_zeta=spec.zeta_dim,
                dim_conditions=spec.condition_dim,
                dim_output=nn_out_dim,
                output_bounds=bounds,
                hidden=200,
                n_layers=2,
            ).to(device)
        else:
            model = CoordinationMLP(
                dim_zeta=spec.zeta_dim,
                dim_conditions=spec.condition_dim,
                dim_output=nn_out_dim,
                output_bounds=bounds,
                hidden=cfg.hidden,
                n_layers=cfg.n_layers,
                output_init_std=spec.model_hparams.get("output_init_std"),
            ).to(device)
        if train_result.model_state is None:
            raise RuntimeError(
                "DC3Solver.predict: missing model_state in train_result"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()

        shim = self._build_shim(bench, dc3_spec, partial_vars, other_vars, device)
        args = _dc3_args_dict(cfg, use_compl=use_compl, corr_mode=corr_mode)

        q = queries.to(device)
        conds = eval_conditions(q, spec.condition_dim)
        X = conds if conds is not None else torch.zeros(
            q.zeta.shape[0], 0, device=device,
        )
        shim.bind_x(X)

        with torch.no_grad():
            nn_out = model(q.zeta, conds)
        if use_compl:
            raw_y = shim.complete_partial(X, nn_out)
        else:
            raw_y = nn_out
        raw_y = raw_y.detach()

        # ``steps`` is the batch-wide count until all samples converge (or the cap).
        post_y, steps = grad_steps_all(shim, X, raw_y, args)
        post_y = post_y.detach()

        n_queries = len(queries)
        inf_iters = torch.full(
            (n_queries,), int(steps), dtype=torch.int32, device="cpu",
        )
        return PredictionOutputs(
            raw=raw_y.cpu(), post=post_y.cpu(), projection=None,
            inference_iters=inf_iters,
        )


def _merge_cfg(base: DC3Config, overrides: dict[str, Any]) -> DC3Config:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"DC3Solver.train: unknown hparam(s) {sorted(unknown)}")
    return DC3Config(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
