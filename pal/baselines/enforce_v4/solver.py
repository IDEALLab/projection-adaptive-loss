"""ENFORCE v4 baseline adapter around the vendored upstream `ENFORCE` module.

Inequalities use upstream's Fischer-Burmeister reformulation (one dual column per
inequality); only the first `spec.dim` output columns ever leave the adapter.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
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
from pal.baselines.enforce_v4.upstream.config import ENFORCEConfig
from pal.baselines.enforce_v4.upstream.fb_inequality_constraints import (
    FischerBurmeisterReformulation,
)
from pal.baselines.enforce_v4.upstream.model import ENFORCE, _ProjectionIFT
from pal.baselines.penalty import FixedPenalty
from pal.benchmarks.base import Benchmark, Query
from pal.constraints import Constraint, constraints_to_violation
from pal.model import box_squash
from pal.runner.probe import mark_opt_step
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.solvers.grad_share import (
    KEY_CON,
    KEY_TOT,
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
class EnforceV4Config:
    seed: int = 0
    epochs: int = 2000
    batch_size: int = 32
    lr: float = 1e-4
    grad_clip: float = 20.0
    device: str = "cpu"

    # `n_layers` counts hidden Linear layers like CoordinationMLP; upstream gets `n_layers - 1`.
    hidden: int = 512
    n_layers: int = 4

    eps_chol: float = 1e-8           # Tikhonov reg. of the gram matrix.
    eps: float = 1e-8                # FB smoothing inside the sqrt.
    training_tolerance: float = 1e-4  # AdaNP stop, mean |c|, during training.
    inference_tolerance: float = 1e-6  # AdaNP stop, max |c|, at inference.
    max_it: int = 100
    epoch_start_hard_constrained: int = 0
    ada_np_auto_activation: bool = True
    weighting_option: int = 5
    weight_loss_displacement: float = 0.5

    # True: IFT backward through the projection, constant memory in Newton depth.
    ift_backward: bool = False

    # True: skip the optimizer step on a non-finite loss or gradient instead of raising.
    skip_nonfinite_step: bool = False

    # Micro-batch size for gradient accumulation (one optimizer step per epoch).
    micro_batch: int | None = None

    # Hold zeta at zero during training (deterministic NN(x)).
    zeta_zero: bool = False

    measure_repair_mem: bool = False
    measure_repair_mem_every: int = 100


class _VizWrapper(nn.Module):
    """`(zeta, conditions) -> y[:, :dim]` view of the raw ENFORCE module."""

    def __init__(self, model: ENFORCE, dim: int, condition_dim: int):
        super().__init__()
        self.model = model
        self.dim = int(dim)
        self.condition_dim = int(condition_dim)

    def forward(self, zeta: Tensor, conditions: Tensor | None = None) -> Tensor:
        x = _model_input(zeta, conditions, self.condition_dim)
        return self.model.forward(x)[:, : self.dim]


def _model_input(
    zeta: Tensor, conditions: Tensor | None, condition_dim: int
) -> Tensor:
    """`x = [zeta | conditions]`; `x = zeta` for unconditional benches."""
    if condition_dim > 0 and conditions is not None:
        return torch.cat([zeta, conditions], dim=-1)
    return zeta


def _constraint_list_fn(
    bench: Benchmark,
) -> Callable[[Tensor, Tensor | None], list[Constraint]]:
    """`(y, conditions) -> list[Constraint]`, objective-free when possible."""
    if hasattr(bench, "constraint_list"):
        return bench.constraint_list  # type: ignore[return-value]
    forward_fn = make_forward_fn(bench)

    def _cons(y: Tensor, conditions: Tensor | None) -> list[Constraint]:
        return forward_fn(y, conditions)[1]

    return _cons


def _build_constraint_system(bench: Benchmark, cfg: EnforceV4Config):
    """Build `(c, fb)` with `c(x, y_ext) = concat(eq_residuals, fb_rows)`.

    Eq-only benches get `fb=None`.
    """
    spec = bench.spec
    zeta_dim = int(spec.zeta_dim)
    condition_dim = int(spec.condition_dim)
    dim = int(spec.dim)
    cons_fn = _constraint_list_fn(bench)

    def _conds_of(x: Tensor) -> Tensor | None:
        return x[:, zeta_dim:] if condition_dim > 0 else None

    ineq_idx = [k for k, t in enumerate(spec.constraint_types) if t == "ineq"]

    fb: FischerBurmeisterReformulation | None = None
    if ineq_idx:
        def _make_g(slot: int) -> Callable[[Tensor, Tensor], Tensor]:
            def g(x: Tensor, y: Tensor) -> Tensor:
                c_k = cons_fn(y, _conds_of(x))[slot]
                # PAL feasibility is `value <= -margin`.
                return c_k.value + c_k.margin
            return g

        fb = FischerBurmeisterReformulation(
            n_original_outputs=dim,
            inequalities=[_make_g(k) for k in ineq_idx],
            eps=cfg.eps,
        )

    def c(x: Tensor, y_ext: Tensor) -> Tensor:
        y = y_ext[:, :dim]
        cl = cons_fn(y, _conds_of(x))
        rows: list[Tensor] = []
        eq_vals = [k.value for k in cl if k.type == "eq"]
        if eq_vals:
            rows.append(torch.stack(eq_vals, dim=1))
        if fb is not None:
            g = torch.stack(
                [k.value + k.margin for k in cl if k.type == "ineq"], dim=1
            )
            lam = y_ext[:, dim:]
            # Vectorised equivalent of `fb(x, y_ext)`.
            rows.append(fb._fb(lam, -g))
        return torch.cat(rows, dim=1)

    return c, fb


class _BoxedENFORCE(ENFORCE):
    """Upstream ENFORCE with a box-squashed head on the design columns.

    The first ``spec.dim`` outputs are tanh-squashed into the design box, the
    Fischer-Burmeister dual columns stay raw. With ``hard_box`` the design
    columns are also clamped into the box after every projection step, since the
    constraint evaluator is only defined inside it.
    """

    def setup_boxed_head(
        self, lower: Tensor, upper: Tensor, hard_box: bool = False
    ) -> None:
        self.register_buffer("design_lower", lower)
        self.register_buffer("design_upper", upper)
        self._hard_box = bool(hard_box)
        self._last_raw_design: Tensor | None = None
        self._last_box_clamp_frac: float = 0.0

    def forward(self, x: Tensor) -> Tensor:  # noqa: D102 (mirrors upstream)
        x = self.input_layer(x)
        x = self.hidden_activation(x)
        for layer in self.hidden_layers:
            x = layer(x)
            x = self.hidden_activation(x)
        raw = self.output_layer(x)          # [BS, dim]  (design pre-activations)
        self._last_raw_design = raw.detach()
        # Stays 0.0 unless this chunk projects.
        self._last_box_clamp_frac = 0.0
        design = box_squash(raw, self.design_lower, self.design_upper)
        if self.fb is not None:
            # Append zero columns for FB dual variables (upstream contract).
            zeros = torch.zeros(
                design.shape[0], self.fb.n_ineq,
                device=design.device, dtype=design.dtype,
            )
            design = torch.cat([design, zeros], dim=1)
        return design

    def project(self, input, output):  # noqa: D102 (mirrors upstream)
        y = super().project(input, output)
        if not getattr(self, "_hard_box", False):
            return y
        # Output scaling is identity, so clamping y here clamps in unscaled space.
        assert bool((self.mean_output == 0).all()) and bool(
            (self.std_output == 1).all()
        ), "hard output box clamp assumes identity output scaling"
        # Clamp the design columns only, FB duals untouched.
        dim = int(self.design_lower.shape[0])
        lower = self.design_lower.to(device=y.device, dtype=y.dtype)
        upper = self.design_upper.to(device=y.device, dtype=y.dtype)
        design = y[:, :dim]
        clamped = torch.maximum(torch.minimum(design, upper), lower)
        self._last_box_clamp_frac = (
            float((clamped != design).float().mean().item())
            if design.numel()
            else 0.0
        )
        return torch.cat([clamped, y[:, dim:]], dim=1)


def _build_model(
    bench: Benchmark, cfg: EnforceV4Config
) -> tuple[ENFORCE, FischerBurmeisterReformulation | None]:
    """Construct the upstream module, then pin it to `cfg.device`."""
    spec = bench.spec
    c, fb = _build_constraint_system(bench, cfg)
    ni = int(spec.zeta_dim) + int(spec.condition_dim)
    n_ext = int(spec.dim) + (fb.n_ineq if fb is not None else 0)

    enf_cfg = ENFORCEConfig(
        input_neurons=ni,
        hidden_neurons=int(cfg.hidden),
        output_neurons=int(spec.dim),   # always spec.dim; fb sizes the head
        hidden_layers=max(int(cfg.n_layers) - 1, 0),
        training_tolerance=float(cfg.training_tolerance),
        inference_tolerance=float(cfg.inference_tolerance),
        max_it=int(cfg.max_it),
        epoch_start_hard_constrained=int(cfg.epoch_start_hard_constrained),
        ada_np_auto_activation=bool(cfg.ada_np_auto_activation),
        ift_backward=bool(cfg.ift_backward),
        regularise_gram=False,
        supervised=False,
        soft_constrained=False,
        weight_loss_displacement=float(cfg.weight_loss_displacement),
        weight_loss_soft=0.0,
        verbose=False,
        random_seed=int(cfg.seed),
    )
    model = _BoxedENFORCE(
        scaling_input=(torch.zeros(ni), torch.ones(ni)),
        scaling_output=(torch.zeros(n_ext), torch.ones(n_ext)),
        c=c,
        config=enf_cfg,
        fb=fb,
        constrained=True,
        weighting_option=int(cfg.weighting_option),
        ssl_loss=None,
        jac=None,
        eps_chol=float(cfg.eps_chol),
    )
    lo, hi = zip(*output_bounds_list(spec), strict=False)
    model.setup_boxed_head(
        torch.tensor(lo, dtype=torch.float32),
        torch.tensor(hi, dtype=torch.float32),
        hard_box=bool(getattr(spec, "hard_output_box", False)),
    )
    # Small head init keeps the squashed outputs near the box midpoint at step 1.
    std = spec.model_hparams.get("output_init_std")
    if std is None:
        std = (4.0 / float(cfg.hidden)) ** 0.5
    nn.init.normal_(model.output_layer.weight, std=float(std))
    nn.init.zeros_(model.output_layer.bias)
    # Undo upstream's CUDA auto-detect.
    model.device = cfg.device
    model.to(cfg.device)
    for attr in ("mean_input", "std_input", "mean_output", "std_output"):
        setattr(model, attr, getattr(model, attr).to(cfg.device))
    return model, fb


class _EpochAgg:
    """Accumulates per-micro-batch metrics into per-epoch scalars.

    ``loss`` sums the pre-scaled chunk losses, ``max`` takes the max over chunks
    and ``mean`` is the chunk-size-weighted mean (equal to the full-batch mean).
    """

    def __init__(self, batch_size: int):
        self._bs = int(batch_size)
        self._rows: list[dict[str, Any]] = []
        self.loss = 0.0

    def add(self, row: dict[str, Any]) -> None:
        self._rows.append(row)
        self.loss += float(row["loss_scaled"])

    def mean(self, key: str) -> float:
        total = sum(
            r[key] * r["cw"] for r in self._rows if r.get(key) is not None
        )
        return float(total / self._bs)

    def max(self, key: str) -> float:
        return float(max(r[key] for r in self._rows))


class EnforceV4Solver:
    """Self-supervised training around upstream ENFORCE v4's Newton projection."""

    name = "enforce_v4"

    def __init__(self, config: EnforceV4Config | None = None):
        self.config = config or EnforceV4Config()

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
        dim = int(spec.dim)
        K = spec.n_eq + spec.n_ineq

        model, fb = _build_model(bench, cfg)
        # Re-seed after upstream's own global seeding in `ENFORCE.__init__`.
        seed_everything(cfg.seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
        con_params = [p for p in model.parameters() if p.requires_grad]

        forward_fn = make_forward_fn(bench)
        penalty = FixedPenalty(
            n_constraints=K,
            constraint_types=list(spec.constraint_types),
            constraint_convention="standard",
        )

        # Objective normalisation for training loss and gate, eval stays in raw units.
        obj_scale = float(getattr(bench, "objective_scale", 1.0))

        def ssl_terms(y: Tensor, conditions: Tensor | None):
            """PAL's SSL setting: `(loss, objective, violations, penalty)`."""
            obj, cons = forward_fn(y, conditions)
            obj = obj / obj_scale
            viol = constraints_to_violation(cons)
            pen = penalty.compute_loss(viol)
            return obj.mean() + pen.mean(), obj, viol, pen

        def _micro_step(
            x_c: Tensor,
            conds_c: Tensor | None,
            proj_phase: bool,
            epoch: int,
            log_grad_share: bool = False,
            log_repair: bool = False,
            gn_con_acc: list[Tensor | None] | None = None,
        ) -> dict[str, Any]:
            """Forward, projection and scaled backward for one micro-batch.

            Gradients accumulate across chunks. ``ada_np``'s mean-mode stop reads
            the chunk mean ``|c|``, so Newton depth can differ from full batch.
            """
            cw = int(x_c.shape[0])
            yhat_c = model.forward(x_c)
            raw_design = model._last_raw_design
            head_sat = (
                float((raw_design.abs() > 3.0).float().mean().item())
                if raw_design is not None and raw_design.numel()
                else 0.0
            )
            proj_iters = 0
            if not proj_phase:
                # Soft step. `training_iter` is bumped per epoch by the caller.
                ytilde_c = yhat_c
            elif cfg.ift_backward:
                # IFT path: detached Newton sweep, gate uses a detached one-step preview.
                def _run_ift() -> tuple[Tensor, int]:
                    out = _ProjectionIFT.apply(
                        yhat_c, model, x_c,
                        "mean", cfg.training_tolerance, cfg.max_it,
                    )
                    return out, int(getattr(model, "_ift_proj_iter", 1))

                if cfg.ada_np_auto_activation:
                    with torch.enable_grad():
                        yhat_prev = yhat_c.detach().requires_grad_(True)
                        model.compute_dc_dy(x_c, yhat_prev)
                        ytilde_prev = model.project(x_c, yhat_prev).detach()
                    with torch.no_grad():
                        l_hat = ssl_terms(yhat_c[:, :dim], conds_c)[0]
                        l_tilde = ssl_terms(ytilde_prev[:, :dim], conds_c)[0]
                    if l_hat < l_tilde:
                        # Projection hurts -> skip it for this chunk.
                        ytilde_c = yhat_c
                        proj_iters = 0
                        model.start_projection = False
                    else:
                        ytilde_c, proj_iters = _run_ift()
                        model.start_projection = True
                else:
                    ytilde_c, proj_iters = _run_ift()
                    model.start_projection = True
            else:
                # Algorithm 2: one Newton step, then the per-chunk gate.
                model.compute_dc_dy(x_c, yhat_c)
                ytilde_c = model.project(x_c, yhat_c)
                proj_iters = 1
                if cfg.ada_np_auto_activation:
                    with torch.no_grad():
                        l_hat = ssl_terms(yhat_c[:, :dim], conds_c)[0]
                        l_tilde = ssl_terms(ytilde_c[:, :dim], conds_c)[0]
                    if l_hat < l_tilde:
                        ytilde_c = yhat_c
                        proj_iters = 0
                        model.start_projection = False
                    else:
                        ytilde_c, proj_iters = model.ada_np(
                            x_c, ytilde_c, tolerance_mode="mean",
                        )
                        model.start_projection = True
                else:
                    ytilde_c, proj_iters = model.ada_np(
                        x_c, ytilde_c, tolerance_mode="mean",
                    )
                    model.start_projection = True

            y_proj = ytilde_c[:, :dim]
            loss_ssl, objective, violations, penalty_loss = ssl_terms(
                y_proj, conds_c
            )
            # Displacement on STRIPPED outputs (duals excluded).
            displacement_loss = (yhat_c[:, :dim] - y_proj).pow(2).mean()
            loss = loss_ssl + cfg.weight_loss_displacement * displacement_loss

            loss_finite = bool(torch.isfinite(loss))
            if not loss_finite and not cfg.skip_nonfinite_step:
                raise RuntimeError(
                    f"epoch {epoch}: non-finite loss "
                    f"(obj={objective.mean().item():.3e}, "
                    f"pen={penalty_loss.mean().item():.3e})"
                )
            scale = cw / cfg.batch_size

            # Constraint-part gradient, accumulated over chunks for the grad-share probe.
            if log_grad_share and gn_con_acc is not None and loss_finite:
                pen_c = penalty_loss.mean() * scale
                if pen_c.requires_grad:
                    grads = torch.autograd.grad(
                        pen_c, con_params, retain_graph=True,
                        create_graph=False, allow_unused=True,
                    )
                    for i, g in enumerate(grads):
                        if g is None:
                            continue
                        g = g.detach()
                        gn_con_acc[i] = g if gn_con_acc[i] is None else gn_con_acc[i] + g

            # pre == post whenever the projection did not run (ytilde_c is yhat_c).
            c_pre_val = c_post_val = None
            if log_repair:
                c_post_val = mean_violation(violations)
                if ytilde_c is yhat_c:
                    c_pre_val = c_post_val
                else:
                    with torch.no_grad():
                        pre_cons = forward_fn(yhat_c[:, :dim].detach(), conds_c)[1]
                        c_pre_val = mean_violation(
                            constraints_to_violation(pre_cons)
                        )

            if loss_finite:
                (loss * scale).backward()

            return {
                "cw": cw,
                "loss_scaled": float(loss.item()) * scale,
                "objective": float(objective.mean().item()),
                "penalty": float(penalty_loss.mean().item()),
                "displacement": float(displacement_loss.item()),
                "max_violation": (
                    float(violations.max().item()) if violations.numel() else 0.0
                ),
                "proj_iters": int(proj_iters),
                "proj_active": 1.0 if proj_iters > 0 else 0.0,
                "head_sat_frac": head_sat,
                "box_clamp_frac": float(
                    getattr(model, "_last_box_clamp_frac", 0.0)
                ),
                "dual_abs_mean": (
                    float(ytilde_c[:, dim:].abs().mean().item())
                    if fb is not None
                    else None
                ),
                KEY_C_PRE: c_pre_val,
                KEY_C_POST: c_post_val,
            }

        viz_model = _VizWrapper(model, dim, int(spec.condition_dim))
        loss_trajectory: list[float] = []

        skipped_steps = 0
        warned_nonfinite_grad = False

        cuda_mem = str(device).startswith("cuda")

        for epoch in range(1, cfg.epochs + 1):
            model.train()
            model.epoch = epoch
            optimizer.zero_grad()

            # Per-epoch GPU peak over fwd+proj+bwd.
            if cuda_mem:
                torch.cuda.reset_peak_memory_stats(device)

            zeta = torch.randn(cfg.batch_size, spec.zeta_dim, device=device)
            if cfg.zeta_zero:
                zeta = torch.zeros_like(zeta)
            conditions = sample_conditions(
                bench, cfg.batch_size, device,
                seed=cfg.seed * 10_000_000 + epoch,
            )
            x = _model_input(zeta, conditions, int(spec.condition_dim))

            do_probe = cfg.measure_repair_mem and (
                epoch == 1 or epoch % cfg.measure_repair_mem_every == 0
            )
            proj_phase = epoch >= cfg.epoch_start_hard_constrained

            mb = (
                cfg.batch_size
                if cfg.micro_batch is None
                else max(1, min(int(cfg.micro_batch), cfg.batch_size))
            )

            # Inputs are sampled once per epoch and only sliced per chunk.
            agg = _EpochAgg(cfg.batch_size)
            log_grad_share = should_log_grad_share(epoch, cfg.epochs)
            log_repair = should_log_repair_contraction(epoch, cfg.epochs)
            gn_con_acc: list[Tensor | None] | None = (
                [None] * len(con_params) if log_grad_share else None
            )
            with probe_peak_memory(enabled=do_probe) as probe:
                for start in range(0, cfg.batch_size, mb):
                    stop = min(start + mb, cfg.batch_size)
                    conds_c = (
                        None if conditions is None else conditions[start:stop]
                    )
                    agg.add(_micro_step(
                        x[start:stop], conds_c, proj_phase, epoch,
                        log_grad_share, log_repair, gn_con_acc,
                    ))

            if not proj_phase:
                # One soft training iteration per epoch, not per chunk.
                model.training_iter += 1

            # Total-loss grad norm before clipping.
            gn_tot = param_grad_norm(model.parameters()) if log_grad_share else 0.0
            gn_con = (
                float(sum(float(g.pow(2).sum().item())
                          for g in (gn_con_acc or []) if g is not None)) ** 0.5
                if log_grad_share
                else 0.0
            )
            grad_norm = clip_grad_norm_(model.parameters(), cfg.grad_clip)
            grad_norm_val = float(
                grad_norm.item() if isinstance(grad_norm, Tensor) else grad_norm
            )
            # Non-finite guard: raise by default, or skip the step if configured.
            if math.isfinite(grad_norm_val) and math.isfinite(agg.loss):
                optimizer.step()
                mark_opt_step(bench)
                skipped_step = 0
            elif not cfg.skip_nonfinite_step:
                raise RuntimeError(
                    f"epoch {epoch}: non-finite gradient "
                    f"(grad_norm={grad_norm_val}, loss={agg.loss})"
                )
            else:
                optimizer.zero_grad(set_to_none=True)
                skipped_step = 1
                skipped_steps += 1
                if not warned_nonfinite_grad:
                    warned_nonfinite_grad = True
                    warnings.warn(
                        "EnforceV4Solver.train: non-finite gradient/loss at "
                        f"epoch {epoch} (grad_norm={grad_norm_val}, "
                        f"loss={agg.loss}); zeroing grads and skipping "
                        "optimizer.step() (weights unchanged this epoch, "
                        "skipped_step=1). Further skips are logged but silent.",
                        RuntimeWarning,
                        stacklevel=2,
                    )

            step_metrics = {
                "loss": agg.loss,
                "objective": agg.mean("objective"),
                "penalty": agg.mean("penalty"),
                "displacement": agg.mean("displacement"),
                "max_violation": agg.max("max_violation"),
                "grad_norm": grad_norm_val,
                "skipped_step": skipped_step,
                # Fraction of the batch that projected, and the deepest chunk's unroll.
                "proj_active": agg.mean("proj_active"),
                "proj_iters": agg.max("proj_iters"),
                "head_sat_frac": agg.mean("head_sat_frac"),
                "box_clamp_frac": agg.mean("box_clamp_frac"),
                # Peak allocated MB over this epoch, NaN off-CUDA.
                "gpu_peak_alloc_mb": (
                    float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)
                    if cuda_mem else float("nan")
                ),
            }
            if fb is not None:
                # Duals are logged as diagnostics only.
                step_metrics["dual_abs_mean"] = agg.mean("dual_abs_mean")
            if log_grad_share:
                step_metrics[KEY_CON] = gn_con
                step_metrics[KEY_TOT] = gn_tot
            if log_repair:
                step_metrics[KEY_C_PRE] = agg.mean(KEY_C_PRE)
                step_metrics[KEY_C_POST] = agg.mean(KEY_C_POST)
            if cfg.measure_repair_mem:
                step_metrics["repair_peak_mem_bytes"] = (
                    float(probe["peak_bytes"]) if do_probe else float("nan")
                )
            logger.log_step(epoch, **step_metrics)
            loss_trajectory.append(agg.loss)

            if on_epoch_end is not None:
                on_epoch_end(epoch, viz_model)

        wall = time.monotonic() - wall_start

        eval_queries = bench.eval_queries(seed).to(device)
        with torch.no_grad():
            model.eval()
            eval_conds = eval_conditions(eval_queries, spec.condition_dim)
            x_eval = _model_input(
                eval_queries.zeta, eval_conds, int(spec.condition_dim)
            )
            final_x = model.forward(x_eval)[:, :dim].detach().cpu()

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
        dim = int(spec.dim)

        model, _fb = _build_model(bench, cfg)
        if train_result.model_state is None:
            raise RuntimeError(
                "EnforceV4Solver.predict: missing model_state in train_result"
            )
        model.load_state_dict(train_result.model_state)
        model.eval()

        q = queries.to(device)
        conds = eval_conditions(q, spec.condition_dim)
        x = _model_input(q.zeta, conds, int(spec.condition_dim))

        # `compute_dc_dy` needs autograd, so only the backbone forward is no_grad.
        with torch.no_grad():
            y_ext = model.forward(x)
        raw = y_ext[:, :dim].detach()

        with torch.enable_grad():
            y0 = y_ext.detach().clone().requires_grad_(True)
            model.compute_dc_dy(x, y0)
            y1 = model.project(x, y0)
            y_post, iters = model.ada_np(
                x, y1,
                tolerance_mode="max",
                tolerance_value=cfg.inference_tolerance,
                max_iter=cfg.max_it,
            )
        post = y_post[:, :dim].detach()

        # Batch-wide early exit -> broadcast the batch's iteration count.
        inf_iters = torch.full(
            (len(q),), int(iters), dtype=torch.int32, device="cpu",
        )
        return PredictionOutputs(
            raw=raw, post=post, projection=None, inference_iters=inf_iters,
        )


def _merge_cfg(base: EnforceV4Config, overrides: dict[str, Any]) -> EnforceV4Config:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"EnforceV4Solver.train: unknown hparam(s) {sorted(unknown)}")
    return EnforceV4Config(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
