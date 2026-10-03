"""ALM+Bolt-On smoke test: training matches plain ALM, projection does not hurt feasibility."""

from __future__ import annotations

import torch

from pal.baselines import ALMBoltOnConfig, ALMBoltOnSolver, ALMConfig, ALMSolver
from pal.benchmarks import get as get_benchmark
from pal.runner.probe import BenchProbe


class _NullLogger:
    def log_config(self, **kw): pass
    def log_step(self, step, **kw): pass
    def log_projection_trajectory(self, step, phase, traj): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **kw): pass
    def finish(self, status="ok", error=None): pass


def _tiny_alm_kwargs():
    return dict(
        epochs=10, batch_size=16, lr=1e-3,
        hidden=64, n_layers=2, seed=0, device="cpu",
    )


def test_alm_bolton_train_matches_plain_alm():
    """Same seed -> same trained weights (bolt-on training is byte-identical)."""
    bench_a = BenchProbe(get_benchmark("s1_sphere_track"))
    bench_b = BenchProbe(get_benchmark("s1_sphere_track"))

    plain = ALMSolver(ALMConfig(**_tiny_alm_kwargs()))
    bolton = ALMBoltOnSolver(ALMBoltOnConfig(**_tiny_alm_kwargs()))

    r_plain = plain.train(bench_a, seed=0, logger=_NullLogger())
    r_bolton = bolton.train(bench_b, seed=0, logger=_NullLogger())

    assert r_plain.model_state is not None
    assert r_bolton.model_state is not None
    assert set(r_plain.model_state) == set(r_bolton.model_state)
    for k in r_plain.model_state:
        torch.testing.assert_close(
            r_plain.model_state[k], r_bolton.model_state[k], rtol=0, atol=0,
            msg=lambda m, k=k: f"weight {k} drifted: {m}",
        )


def test_alm_bolton_predict_projector_improves_feasibility():
    """Post-projection output should be at least as feasible as raw."""
    bench = BenchProbe(get_benchmark("s1_sphere_track"))
    bolton = ALMBoltOnSolver(ALMBoltOnConfig(**_tiny_alm_kwargs()))
    result = bolton.train(bench, seed=0, logger=_NullLogger())

    queries = bench.eval_queries(seed=0, n=8)
    out = bolton.predict(bench, queries, result)

    assert out.raw is not None
    assert out.post is not None
    assert out.raw.shape == out.post.shape

    bench.set_phase("predict")
    with torch.no_grad():
        c_raw = bench.constraints(out.raw, queries.conditions)
        c_post = bench.constraints(out.post, queries.conditions)
    res_raw = c_raw.abs().max().item()
    res_post = c_post.abs().max().item()
    # Bolt-on should not make feasibility worse.
    assert res_post <= res_raw + 1e-6, (
        f"post residual ({res_post:.3e}) exceeds raw residual ({res_raw:.3e})"
    )


def test_alm_bolton_emits_nfe_counters():
    """End-to-end: counters survive train + predict path."""
    bench = BenchProbe(get_benchmark("s1_sphere_track"))
    bolton = ALMBoltOnSolver(ALMBoltOnConfig(**_tiny_alm_kwargs()))
    result = bolton.train(bench, seed=0, logger=_NullLogger())

    snapshot = bench.snapshot()
    train_counters = snapshot["train"]
    # 10 epochs x 1 opt step/epoch = 10 opt_steps.
    assert train_counters["opt_steps"] == 10
    assert train_counters["fwd_calls"] >= 10
    assert train_counters["bwd_calls"] >= 10
    # samples = calls x batch_size.
    assert train_counters["fwd_samples"] >= train_counters["fwd_calls"] * 16

    assert result.solver_name == "alm_bolton"
