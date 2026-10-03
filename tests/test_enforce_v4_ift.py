"""IFT-backward, CLI flag, and non-finite-gradient guard tests for ENFORCE v4."""

from __future__ import annotations

import math

import pytest
import torch

import pal.baselines.enforce_v4.solver as solver_mod
from pal.baselines.enforce_v4 import EnforceV4Config, EnforceV4Solver
from pal.benchmarks import get as get_benchmark
from pal.runner.cli import _build_cfg, _parse_args

_BENCH = "s2_active_set_switch"
_BATCH = 16


class _RecordingLogger:
    """Captures the per-step scalar rows passed to ``log_step``."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def log_config(self, **cfg): pass

    def log_step(self, step, **scalars):
        self.rows.append({"step": step, **scalars})

    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **final): pass
    def finish(self, status="ok", error=None): pass


def test_ift_backward_training_runs_and_reports_proj_iters() -> None:
    """3 epochs, hard projection from epoch 0, IFT backward: finite + projects."""
    logger = _RecordingLogger()
    cfg = EnforceV4Config(
        seed=0, epochs=3, batch_size=_BATCH, lr=1e-4, device="cpu",
        epoch_start_hard_constrained=0, ift_backward=True,
    )
    result = EnforceV4Solver(cfg).train(
        get_benchmark(_BENCH), seed=0, logger=logger,
    )

    assert len(logger.rows) == 3
    for row in logger.rows:
        assert math.isfinite(row["loss"]), row
        assert row["skipped_step"] == 0, row
    # proj_iters = initial project + ada_np iters recorded in model._ift_proj_iter.
    assert max(row["proj_iters"] for row in logger.rows) >= 1
    assert torch.isfinite(result.final_x_on_eval).all()


def test_ift_backward_matches_unrolled_shapes() -> None:
    """IFT and unrolled paths both train to a finite eval tensor of same shape."""
    bench = get_benchmark(_BENCH)
    outs = {}
    for ift in (False, True):
        cfg = EnforceV4Config(
            seed=0, epochs=2, batch_size=_BATCH, lr=1e-4, device="cpu",
            epoch_start_hard_constrained=0, ift_backward=ift,
        )
        outs[ift] = EnforceV4Solver(cfg).train(
            get_benchmark(_BENCH), seed=0, logger=_RecordingLogger(),
        ).final_x_on_eval
    assert outs[False].shape == outs[True].shape == (
        len(bench.eval_queries(0)), int(bench.spec.dim),
    )
    assert torch.isfinite(outs[True]).all()


def test_cli_ift_backward_flag_sets_config() -> None:
    args = _parse_args(
        ["run", "--method", "enforce_v4", "--enforce-v4-ift-backward"]
    )
    cfg = _build_cfg("enforce_v4", args, seed=0)
    assert cfg.ift_backward is True


def test_cli_ift_backward_absent_defaults_false() -> None:
    args = _parse_args(["run", "--method", "enforce_v4"])
    cfg = _build_cfg("enforce_v4", args, seed=0)
    assert cfg.ift_backward is False


def test_cli_skip_nonfinite_flag_sets_config() -> None:
    args = _parse_args(
        ["run", "--method", "enforce_v4", "--enforce-v4-skip-nonfinite-step"]
    )
    cfg = _build_cfg("enforce_v4", args, seed=0)
    assert cfg.skip_nonfinite_step is True


def test_cli_skip_nonfinite_absent_defaults_false() -> None:
    args = _parse_args(["run", "--method", "enforce_v4"])
    cfg = _build_cfg("enforce_v4", args, seed=0)
    assert cfg.skip_nonfinite_step is False


def _patch_inject_nan_grad(monkeypatch, state: dict):
    """Monkeypatch clip_grad_norm_ to snapshot params and poison one gradient."""
    real_clip = solver_mod.clip_grad_norm_

    def fake_clip(parameters, max_norm, *args, **kwargs):
        plist = list(parameters)
        if "pre" not in state:
            state["pre"] = [p.detach().clone() for p in plist]
            for p in plist:
                if p.grad is not None:
                    with torch.no_grad():
                        p.grad.view(-1)[0] = float("nan")
                    break
        return real_clip(plist, max_norm, *args, **kwargs)

    monkeypatch.setattr(solver_mod, "clip_grad_norm_", fake_clip)


def test_nonfinite_grad_guard_skips_step(monkeypatch) -> None:
    """skip_nonfinite_step=True: NaN grad -> step skipped, params unchanged."""
    state: dict = {}
    _patch_inject_nan_grad(monkeypatch, state)

    def hook(epoch: int, viz_model) -> None:
        state["post"] = [p.detach().clone() for p in viz_model.model.parameters()]

    logger = _RecordingLogger()
    cfg = EnforceV4Config(
        seed=0, epochs=1, batch_size=_BATCH, lr=1e-4, device="cpu",
        epoch_start_hard_constrained=0, skip_nonfinite_step=True,
    )
    EnforceV4Solver(cfg).train(
        get_benchmark(_BENCH), seed=0, logger=logger, on_epoch_end=hook,
    )

    assert logger.rows[0]["skipped_step"] == 1
    assert "pre" in state and "post" in state
    for before, after in zip(state["pre"], state["post"], strict=True):
        assert torch.equal(before, after)


def test_nonfinite_grad_default_raises(monkeypatch) -> None:
    """Default skip_nonfinite_step=False: a NaN grad fails fast (RuntimeError)."""
    state: dict = {}
    _patch_inject_nan_grad(monkeypatch, state)

    cfg = EnforceV4Config(
        seed=0, epochs=1, batch_size=_BATCH, lr=1e-4, device="cpu",
        epoch_start_hard_constrained=0,  # skip_nonfinite_step defaults False
    )
    with pytest.raises(RuntimeError, match="non-finite gradient"):
        EnforceV4Solver(cfg).train(
            get_benchmark(_BENCH), seed=0, logger=_RecordingLogger(),
        )
