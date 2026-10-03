"""Smoke test for the ALM baseline: the solver wires up and trains on a cheap benchmark."""

from __future__ import annotations

import math

from pal.baselines import ALMConfig, ALMSolver
from pal.benchmarks import get as get_benchmark


class _NullLogger:
    def log_config(self, **kw): pass

    def log_step(self, step, **kw): pass

    def log_projection_trajectory(self, step, phase, traj): pass

    def log_artifact(self, step, name, payload): pass

    def log_final(self, **kw): pass

    def finish(self, status="ok", error=None): pass


def test_alm_smoke_rosenbrock_eq():
    bench = get_benchmark("rosenbrock_eq")
    cfg = ALMConfig(epochs=5, batch_size=32, lr=1e-3, seed=0, device="cpu")
    solver = ALMSolver(cfg)
    result = solver.train(bench, seed=0, logger=_NullLogger())

    assert result.final_x_on_eval is not None
    assert result.final_x_on_eval.shape[-1] == bench.spec.dim
    assert len(result.train_loss_trajectory) == 5
    assert all(math.isfinite(v) for v in result.train_loss_trajectory)
