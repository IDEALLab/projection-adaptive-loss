"""End-to-end smoke tests for learned solvers on rosenbrock_eq."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from pal.baselines import (
    EnforceOrigConfig,
    EnforceOrigSolver,
    EnforceV4Config,
    EnforceV4Solver,
)
from pal.baselines.dc3 import DC3Config, DC3Solver
from pal.baselines.fsnet import FSNetConfig, FSNetSolver
from pal.baselines.snarenet import SnareNetConfig, SnareNetSolver
from pal.benchmarks import get as get_benchmark
from pal.eval import run_final_eval


class _NullLogger:
    def log_config(self, **cfg): pass
    def log_step(self, step, **scalars): pass
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **final): pass
    def finish(self, status="ok", error=None): pass


def _run_smoke(solver, bench, seed: int = 0):
    train_result = solver.train(bench, seed=seed, logger=_NullLogger())
    assert train_result.final_x_on_eval is not None
    queries = bench.eval_queries(seed=seed, n=4)
    eval_result = run_final_eval(
        bench=bench,
        solver=solver,
        train_result=train_result,
        queries=queries,
    )
    assert np.isfinite(eval_result.obj_mean_raw)
    assert np.isfinite(eval_result.viol_max_raw)
    assert np.isfinite(eval_result.feasibility_raw)


def test_enforce_orig_smoke_rosenbrock_eq() -> None:
    bench = get_benchmark("rosenbrock_eq")
    solver = EnforceOrigSolver(EnforceOrigConfig(
        seed=0,
        epochs=2,
        batch_size=8,
        lr=1e-4,
        device="cpu",
    ))
    _run_smoke(solver, bench)


def _enforce_v4_cfg(**kw) -> EnforceV4Config:
    return EnforceV4Config(seed=0, epochs=2, batch_size=8, lr=1e-4, device="cpu", **kw)


def test_enforce_v4_smoke_s1_eq_only() -> None:
    """Eq-only bench: no FB reformulation, `output_neurons == spec.dim`."""
    _run_smoke(EnforceV4Solver(_enforce_v4_cfg()), get_benchmark("s1_sphere_track"))


def test_enforce_v4_fb_extended_output_is_stripped() -> None:
    """s2 = 1 eq + 10 ineq: the net runs at dim+10, callers must see dim."""
    bench = get_benchmark("s2_active_set_switch")
    dim = bench.spec.dim
    solver = EnforceV4Solver(_enforce_v4_cfg())
    train_result = solver.train(bench, seed=0, logger=_NullLogger())
    assert train_result.final_x_on_eval is not None
    assert train_result.final_x_on_eval.shape[1] == dim

    queries = bench.eval_queries(seed=0, n=4)
    out = solver.predict(bench, queries, train_result)
    assert out.raw.shape == (4, dim)
    assert out.post.shape == (4, dim)
    assert out.inference_iters is not None
    assert out.inference_iters.shape == (4,)
    # post is the post-cleanup output.
    assert not torch.allclose(out.raw, out.post)


def test_enforce_v4_unknown_hparam_raises() -> None:
    solver = EnforceV4Solver(_enforce_v4_cfg())
    with pytest.raises(TypeError, match="unknown hparam"):
        solver.train(
            get_benchmark("s1_sphere_track"), seed=0, logger=_NullLogger(),
            not_a_knob=1,
        )


def test_enforce_v4_s5_overdetermined_is_structural() -> None:
    """s5 (`n_eq=5 > dim=4`) is inapplicable to v4: the Gram matrix is singular."""
    solver = EnforceV4Solver(_enforce_v4_cfg())
    with pytest.raises(ValueError, match="Too many constraints"):
        solver.train(get_benchmark("s5_overdetermined"), seed=0, logger=_NullLogger())


def test_enforce_v4_fp64_default_dtype_is_honoured() -> None:
    """Under a float64 default dtype nothing in v4 silently falls back to fp32."""
    from pal.baselines.enforce_v4.upstream.fb_inequality_constraints import (
        FischerBurmeisterReformulation,
    )
    from pal.baselines.enforce_v4.upstream.model import ENFORCE

    seen: dict[str, torch.dtype] = {}
    orig_pt = ENFORCE.projection_tensors

    def _spy(self, B, v, W_inv=None):
        seen.setdefault("B", B.dtype)
        out = orig_pt(self, B, v, W_inv)
        seen.setdefault("B_star", out[0].dtype)
        seen.setdefault("mean_input", self.mean_input.dtype)
        seen.setdefault("std_output", self.std_output.dtype)
        return out

    prev = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    ENFORCE.projection_tensors = _spy
    try:
        solver = EnforceV4Solver(_enforce_v4_cfg(epoch_start_hard_constrained=1))
        solver.train(get_benchmark("s1_sphere_track"), seed=0, logger=_NullLogger())

        fb = FischerBurmeisterReformulation(
            n_original_outputs=2, inequalities=[lambda x, y: y[:, :1]],
        )
        ext = fb.extend_outputs(np.zeros((3, 2), dtype=np.float64))
        assert ext.dtype == np.float64
    finally:
        ENFORCE.projection_tensors = orig_pt
        torch.set_default_dtype(prev)

    assert seen, "projection never ran - test would not prove anything"
    assert seen["B"] is torch.float64
    assert seen["B_star"] is torch.float64
    assert seen["mean_input"] is torch.float64
    assert seen["std_output"] is torch.float64


def test_dc3_smoke_s5_overdetermined() -> None:
    """s5 is structurally DC3-inapplicable (n_eq=5 > dim=4) and must fail loudly."""
    bench = get_benchmark("s5_overdetermined")
    solver = DC3Solver(DC3Config(
        seed=0,
        epochs=2,
        batch_size=8,
        lr=1e-4,
        device="cpu",
    ))
    with pytest.raises(NotImplementedError, match="structurally inapplicable"):
        _run_smoke(solver, bench)


# Benchmark objective_scale must reach the training loss of every learned baseline.

def _loss_trajectory_with_scale(make_solver, scale: float) -> list[float]:
    bench = get_benchmark("s1_sphere_track")
    if scale != 1.0:
        # s1 declares no objective_scale; attach one the way e3 does.
        type(bench).objective_scale = property(lambda self: scale)
    try:
        tr = make_solver().train(bench, seed=0, logger=_NullLogger())
    finally:
        if scale != 1.0:
            delattr(type(bench), "objective_scale")
    return list(tr.train_loss_trajectory)


@pytest.mark.parametrize("make_solver", [
    lambda: EnforceV4Solver(_enforce_v4_cfg()),
    lambda: FSNetSolver(FSNetConfig(seed=0, epochs=2, steps_per_epoch=1, batch_size=8,
                                    hidden_dim=16, num_layers=2, device="cpu")),
    lambda: SnareNetSolver(SnareNetConfig(seed=0, epochs=2, batch_size=8, hidden_size=16,
                                          num_hidden_layers=2, n_calibration_batches=1,
                                          device="cpu")),
], ids=["enforce_v4", "fsnet", "snarenet"])
def test_objective_scale_enters_training_loss(make_solver) -> None:
    base = _loss_trajectory_with_scale(make_solver, 1.0)
    scaled = _loss_trajectory_with_scale(make_solver, 1e6)
    assert len(base) == len(scaled) == 2
    assert all(np.isfinite(base)) and all(np.isfinite(scaled))
    assert not np.allclose(base, scaled)
