"""Per-step evaluator cost sits at the floor: 2 fwd and K+1 bwd per training step.

K+1 bwd = K VJPs building J from c_pre + one loss.backward(). Eval phases stay at 0.
"""

from __future__ import annotations

import pytest

from pal.benchmarks import get as get_benchmark
from pal.method.loggap import PALLogGapConfig, PALLogGapSolver
from pal.runner.probe import BenchProbe
from pal.solvers.grad_share import should_log_grad_share


class _NullLogger:
    def log_config(self, **kw): pass
    def log_step(self, step, **kw): pass
    def log_projection_trajectory(self, step, phase, traj): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **kw): pass
    def finish(self, status="ok", error=None): pass


_EPOCHS = 3
_BATCH = 8


@pytest.mark.parametrize("bench_id", ["s1_sphere_track", "s5_overdetermined"])
def test_pal_loggap_hits_floor(bench_id: str) -> None:
    raw = get_benchmark(bench_id)
    K = raw.spec.n_eq + raw.spec.n_ineq
    probe = BenchProbe(raw)
    probe.set_phase("train")

    cfg = PALLogGapConfig(
        seed=0, epochs=_EPOCHS, batch_size=_BATCH,
        eval_every=0, projection_log_every=0,
    )
    PALLogGapSolver(cfg).train(probe, seed=0, logger=_NullLogger())

    train = probe.snapshot()["train"]
    assert train["fwd_calls"] == 2 * _EPOCHS, (
        f"{bench_id}: expected 2 fwd/step x {_EPOCHS} epochs, got {train['fwd_calls']}"
    )
    n_grad_share_epochs = sum(
        should_log_grad_share(ep, _EPOCHS) for ep in range(1, _EPOCHS + 1)
    )
    expected_bwd = (K + 1) * _EPOCHS + n_grad_share_epochs
    assert train["bwd_calls"] == expected_bwd, (
        f"{bench_id}: expected K+1={K+1} bwd/step x {_EPOCHS} epochs "
        f"+ {n_grad_share_epochs} grad-share epochs, "
        f"got {train['bwd_calls']} (K={K})"
    )
    # opt_steps = epochs x steps_per_epoch, 1 step/epoch here.
    assert train["opt_steps"] == _EPOCHS, (
        f"{bench_id}: expected {_EPOCHS} opt_steps, got {train['opt_steps']}. "
        f"mark_opt_step() must fire once per optimizer.step()."
    )


@pytest.mark.parametrize(
    "bench_id,expected_bwd_per_step",
    [
        # Per step: K_eq + 10*(K_eq + 1) VJPs + ~1 outer backward.
        # rosenbrock_eq: K_eq=1, K_in=1 -> K_eq + 10*(K_eq + 1) + overhead ~ 22-25
        ("rosenbrock_eq", (20, 30)),
        # equality_dominated: K_eq=8, K_in=2 -> K_eq + 10*(K_eq + 1) + overhead ~ 98-110
        ("equality_dominated", (95, 115)),
    ],
)
def test_dc3_probe_hits_refactor_range(
    bench_id: str, expected_bwd_per_step: tuple[int, int]
) -> None:
    """DC3's K-loop VJPs must be visible to the probe."""
    from pal.baselines.dc3 import DC3Config, DC3Solver

    raw = get_benchmark(bench_id)
    probe = BenchProbe(raw)
    probe.set_phase("train")

    cfg = DC3Config(seed=0, epochs=_EPOCHS, batch_size=16, device="cpu")
    DC3Solver(cfg).train(probe, seed=0, logger=_NullLogger())

    train = probe.snapshot()["train"]
    bwd_per_step = train["bwd_calls"] / _EPOCHS
    lo, hi = expected_bwd_per_step
    assert lo <= bwd_per_step <= hi, (
        f"{bench_id}: expected bwd/step in [{lo}, {hi}], got {bwd_per_step:.2f} "
        f"(total bwd={train['bwd_calls']} over {_EPOCHS} epochs). "
        f"Pre-refactor would report ~0 because vmap(jacrev(...)) "
        f"hid K VJPs from probe hooks."
    )


@pytest.mark.parametrize(
    "bench_id,min_bwd_per_step",
    [
        # Up to newton_maxiter iters per step, K VJPs each: assert a floor, not a range.
        ("s5_overdetermined", 100),
        ("s1_sphere_track", 50),
    ],
)
def test_snarenet_probe_sees_newton_vjps(
    bench_id: str, min_bwd_per_step: int
) -> None:
    """SnareNet's Newton-repair Jacobian VJPs must be visible to the probe."""
    from pal.baselines.snarenet import SnareNetConfig, SnareNetSolver

    raw = get_benchmark(bench_id)
    probe = BenchProbe(raw)
    probe.set_phase("train")

    cfg = SnareNetConfig(
        seed=0, epochs=_EPOCHS, batch_size=16, device="cpu",
        adaptive_relaxation=False,  # AR's calibration pass triggers its own forwards
    )
    SnareNetSolver(cfg).train(probe, seed=0, logger=_NullLogger())

    train = probe.snapshot()["train"]
    bwd_per_step = train["bwd_calls"] / _EPOCHS
    assert bwd_per_step >= min_bwd_per_step, (
        f"{bench_id}: expected bwd/step >= {min_bwd_per_step}, "
        f"got {bwd_per_step:.2f}. Pre-refactor baseline was ~1/step "
        f"(vmap-hidden); the floor here asserts the K-loop refactor "
        f"is actually routing VJPs through eager autograd.grad."
    )


def test_dc3_sample_mode_populates_measurement_bucket() -> None:
    """Window=2, period=3, 6 epochs: epochs {1,2,4,5} are loop-mode, 4 sampled opt steps."""
    from pal.baselines.dc3 import DC3Config, DC3Solver

    raw = get_benchmark("rosenbrock_eq")
    probe = BenchProbe(raw)
    probe.set_phase("train")

    cfg = DC3Config(
        seed=0, epochs=6, batch_size=16, device="cpu",
        jacobian_mode="sample",
        measurement_window_epochs=2,
        measurement_period_epochs=3,
    )
    DC3Solver(cfg).train(probe, seed=0, logger=_NullLogger())

    train = probe.snapshot()["train"]
    assert train["measurement_opt_steps"] == 4, (
        f"expected 4 measurement opt_steps (window=2 x 2 periods of 3), "
        f"got {train['measurement_opt_steps']}"
    )
    assert train["measurement_fwd_calls"] > 0
    assert train["measurement_bwd_calls"] > 0
    # Sampled totals must never exceed the full totals.
    assert train["measurement_fwd_calls"] <= train["fwd_calls"]
    assert train["measurement_bwd_calls"] <= train["bwd_calls"]
    assert train["measurement_opt_steps"] <= train["opt_steps"]


def test_dc3_sample_mode_extrapolates_to_loop_mode_totals() -> None:
    """Sampled bwd_calls extrapolated to train_opt_steps match the loop-mode total."""
    from pal.baselines.dc3 import DC3Config, DC3Solver

    EPOCHS = 6
    common = dict(
        seed=0, epochs=EPOCHS, batch_size=16, device="cpu",
    )

    raw_loop = get_benchmark("equality_dominated")
    probe_loop = BenchProbe(raw_loop)
    probe_loop.set_phase("train")
    DC3Solver(DC3Config(jacobian_mode="loop", **common)).train(
        probe_loop, seed=0, logger=_NullLogger(),
    )
    loop_train = probe_loop.snapshot()["train"]

    raw_sample = get_benchmark("equality_dominated")
    probe_sample = BenchProbe(raw_sample)
    probe_sample.set_phase("train")
    DC3Solver(
        DC3Config(
            jacobian_mode="sample",
            measurement_window_epochs=2,
            measurement_period_epochs=3,
            **common,
        )
    ).train(probe_sample, seed=0, logger=_NullLogger())
    sample_train = probe_sample.snapshot()["train"]

    assert sample_train["measurement_opt_steps"] > 0
    bwd_per_step = (
        sample_train["measurement_bwd_calls"]
        / sample_train["measurement_opt_steps"]
    )
    extrapolated_bwd = bwd_per_step * sample_train["opt_steps"]

    # Per-step bwd is deterministic across modes; 10% covers Newton iter jitter.
    rel_err = abs(extrapolated_bwd - loop_train["bwd_calls"]) / loop_train["bwd_calls"]
    assert rel_err < 0.10, (
        f"extrapolation drift: extrapolated bwd={extrapolated_bwd:.1f} "
        f"vs loop-mode bwd={loop_train['bwd_calls']} (rel err {rel_err:.2%})"
    )


def test_eval_phases_do_not_pollute_train_counters() -> None:
    """Periodic eval runs inside the training loop; phase_ctx isolates it."""
    raw = get_benchmark("s1_sphere_track")
    probe = BenchProbe(raw)
    probe.set_phase("train")

    cfg = PALLogGapConfig(
        seed=0, epochs=4, batch_size=_BATCH,
        eval_every=2, eval_samples=8, projection_log_every=0,
    )
    PALLogGapSolver(cfg).train(probe, seed=0, logger=_NullLogger())

    snap = probe.snapshot()
    # 4 epochs x 2 fwd/step = 8; periodic eval sneaks past the probe.
    assert snap["train"]["fwd_calls"] == 8, snap
    assert snap["periodic_eval"]["fwd_calls"] == 0, snap
    assert snap["final_eval"]["fwd_calls"] == 0, snap
    assert snap["predict"]["fwd_calls"] == 0, snap
