"""Repair-contraction instrumentation: cadence, keys, and non-perturbation."""

from __future__ import annotations

import math

import pytest

from pal.baselines import EnforceOrigConfig, EnforceOrigSolver
from pal.baselines.dc3 import DC3Config, DC3Solver
from pal.baselines.snarenet.solver import SnareNetConfig, SnareNetSolver
from pal.benchmarks import get as get_benchmark
from pal.method.loggap.solver import PALLogGapConfig, PALLogGapSolver
from pal.solvers.grad_share import GRAD_SHARE_EVERY
from pal.solvers.repair_contraction import (
    DISABLE_ENV,
    KEY_C_POST,
    KEY_C_PRE,
    mean_violation,
    should_log_repair_contraction,
)

BENCH = "s2_active_set_switch"
EPOCHS = 2 * GRAD_SHARE_EVERY  # two logged epochs, the second also final


class _RecordingLogger:
    """Captures every `log_step` payload keyed by epoch."""

    def __init__(self) -> None:
        self.steps: list[tuple[int, dict]] = []

    def log_config(self, **cfg): pass
    def log_step(self, step, **scalars): self.steps.append((step, dict(scalars)))
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **final): pass
    def finish(self, status="ok", error=None): pass

    def repair_epochs(self) -> list[int]:
        return [s for s, d in self.steps if KEY_C_PRE in d]

    def losses(self) -> list[float]:
        return [d["loss"] for _s, d in self.steps if "loss" in d]


def _solver(name: str):
    if name == "pal_loggap":
        return PALLogGapSolver(PALLogGapConfig(
            seed=0, epochs=EPOCHS, batch_size=8, device="cpu",
        ))
    if name == "dc3":
        return DC3Solver(DC3Config(
            seed=0, epochs=EPOCHS, batch_size=8, device="cpu", corr_lr=1e-2,
        ))
    if name == "enforce_orig":
        return EnforceOrigSolver(EnforceOrigConfig(
            seed=0, epochs=EPOCHS, batch_size=8, device="cpu",
            adanp_warmup_frac=0.0,
        ))
    if name == "snarenet":
        return SnareNetSolver(SnareNetConfig(
            seed=0, epochs=EPOCHS, batch_size=8, device="cpu",
            adaptive_relaxation=False,
        ))
    raise AssertionError(name)


def test_cadence_matches_grad_share() -> None:
    total = 25
    logged = [e for e in range(1, total + 1)
              if should_log_repair_contraction(e, total)]
    assert logged == [10, 20, 25]


def test_kill_switch(monkeypatch) -> None:
    monkeypatch.setenv(DISABLE_ENV, "1")
    assert not should_log_repair_contraction(10, 100)
    assert not should_log_repair_contraction(100, 100)


def test_mean_violation_empty_and_detached() -> None:
    import torch
    assert mean_violation(torch.zeros(4, 0)) == 0.0
    live = torch.tensor([[1.0, 3.0]], requires_grad=True)
    assert mean_violation(live) == pytest.approx(2.0)


@pytest.mark.parametrize(
    "method", ["pal_loggap", "dc3", "enforce_orig", "snarenet"]
)
def test_keys_land_on_cadence_and_contract(method: str) -> None:
    bench = get_benchmark(BENCH)
    logger = _RecordingLogger()
    _solver(method).train(bench, seed=0, logger=logger)

    assert logger.repair_epochs() == [GRAD_SHARE_EVERY, EPOCHS]
    for step, d in logger.steps:
        if KEY_C_PRE not in d:
            assert KEY_C_POST not in d
            continue
        pre, post = d[KEY_C_PRE], d[KEY_C_POST]
        assert isinstance(pre, float) and isinstance(post, float)
        assert math.isfinite(pre) and math.isfinite(post)
        assert pre >= 0.0 and post >= 0.0
        # Contraction is expected for this s2 fixture, not an algorithmic invariant.
        assert post <= pre + 1e-6, (method, step, pre, post)


@pytest.mark.parametrize(
    "method", ["pal_loggap", "dc3", "enforce_orig", "snarenet"]
)
def test_non_perturbing(method: str, monkeypatch) -> None:
    """Loss trajectory must be bit-identical with the probe on and off."""
    on = _RecordingLogger()
    _solver(method).train(get_benchmark(BENCH), seed=0, logger=on)

    monkeypatch.setenv(DISABLE_ENV, "1")
    off = _RecordingLogger()
    _solver(method).train(get_benchmark(BENCH), seed=0, logger=off)

    assert off.repair_epochs() == []
    a, b = on.losses(), off.losses()
    assert len(a) == len(b) > 0
    assert [x.hex() for x in a] == [y.hex() for y in b]


def test_snarenet_hook_registration(monkeypatch) -> None:
    """The backbone capture hook exists only when the probe can fire."""

    def hooks_during_training() -> list[int]:
        seen: list[int] = []

        def _on_epoch_end(_epoch, net) -> None:
            seen.append(len(net._base._forward_hooks))

        _solver("snarenet").train(
            get_benchmark(BENCH), seed=0, logger=_RecordingLogger(),
            on_epoch_end=_on_epoch_end,
        )
        return seen

    on = hooks_during_training()
    assert on and all(n == 1 for n in on)

    monkeypatch.setenv(DISABLE_ENV, "1")
    off = hooks_during_training()
    assert off and all(n == 0 for n in off)


def test_enforce_orig_warmup_pre_equals_post() -> None:
    """During AdaNP warmup the repair is a no-op, so pre == post exactly."""
    bench = get_benchmark(BENCH)
    logger = _RecordingLogger()
    EnforceOrigSolver(EnforceOrigConfig(
        seed=0, epochs=EPOCHS, batch_size=8, device="cpu",
        adanp_warmup_frac=0.5,
    )).train(bench, seed=0, logger=logger)

    warmup_epoch = int(EPOCHS * 0.5)
    assert logger.repair_epochs() == [GRAD_SHARE_EVERY, EPOCHS]

    contracted = False
    for step, d in logger.steps:
        if KEY_C_PRE not in d:
            continue
        pre, post = d[KEY_C_PRE], d[KEY_C_POST]
        if step <= warmup_epoch:
            assert pre == post, (step, pre, post)
        else:
            assert post <= pre + 1e-6, (step, pre, post)
            contracted = contracted or post < pre
    assert contracted, "no post-warmup epoch showed the projection contracting"


def test_snarenet_adaptive_relaxation_keys() -> None:
    """With adaptive relaxation on, the keys still land on schedule."""
    bench = get_benchmark(BENCH)
    logger = _RecordingLogger()
    SnareNetSolver(SnareNetConfig(
        seed=0, epochs=EPOCHS, batch_size=8, device="cpu",
        adaptive_relaxation=True, n_calibration_batches=2,
    )).train(bench, seed=0, logger=logger)

    assert logger.repair_epochs() == [GRAD_SHARE_EVERY, EPOCHS]
    for step, d in logger.steps:
        if KEY_C_PRE not in d:
            continue
        pre, post = d[KEY_C_PRE], d[KEY_C_POST]
        assert math.isfinite(pre) and math.isfinite(post), (step, pre, post)
        assert pre >= 0.0 and post >= 0.0, (step, pre, post)
