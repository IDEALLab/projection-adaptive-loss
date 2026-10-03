"""Final-eval pipeline tests (non-golden)."""

from __future__ import annotations

import numpy as np

from pal.benchmarks import get as get_benchmark
from pal.eval import run_final_eval
from pal.method import PALLogGapConfig, PALLogGapSolver


class _NullLogger:
    def log_config(self, **cfg): pass
    def log_step(self, step, **scalars): pass
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **final): pass
    def finish(self, status="ok", error=None): pass


def test_run_final_eval_ignores_distributed_flag_without_process_group() -> None:
    bench = get_benchmark("rosenbrock_eq")
    solver = PALLogGapSolver(PALLogGapConfig(
        seed=0,
        epochs=1,
        batch_size=8,
        lr=1e-4,
        device="cpu",
        eval_every=100,
        eval_samples=8,
        projection_log_every=0,
    ))
    train_result = solver.train(bench, seed=0, logger=_NullLogger())
    queries = bench.eval_queries(seed=0, n=4)

    eval_result = run_final_eval(
        bench=bench,
        solver=solver,
        train_result=train_result,
        queries=queries,
        distributed=True,
    )

    assert np.isfinite(eval_result.obj_mean_raw)
    assert np.isfinite(eval_result.viol_max_raw)
