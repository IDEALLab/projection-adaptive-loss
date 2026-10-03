"""Unit tests for `BenchProbe` counter wrapper."""

from __future__ import annotations

import pytest
import torch

from pal.benchmarks import get as get_benchmark
from pal.runner.probe import BenchProbe


@pytest.fixture()
def bench():
    return get_benchmark("rosenbrock_eq")


def test_forward_counts_calls_and_samples(bench):
    probe = BenchProbe(bench)
    x = torch.randn(8, bench.spec.dim)

    probe.forward(x)
    probe.forward(x)

    snap = probe.snapshot()
    assert snap["train"]["fwd_calls"] == 2
    assert snap["train"]["fwd_samples"] == 16


def test_eval_phases_sneak_past_counter(bench):
    """Only the train phase is counted; eval phases are no-ops."""
    probe = BenchProbe(bench)
    x = torch.randn(4, bench.spec.dim)

    probe.set_phase("train")
    probe.forward(x)
    probe.set_phase("final_eval")
    probe.forward(x)
    probe.forward(x)

    snap = probe.snapshot()
    assert snap["train"]["fwd_calls"] == 1
    assert snap["train"]["fwd_samples"] == 4
    assert snap["final_eval"]["fwd_calls"] == 0
    assert snap["final_eval"]["fwd_samples"] == 0


def test_phase_ctx_restores_previous(bench):
    probe = BenchProbe(bench)
    probe.set_phase("train")

    with probe.phase_ctx("periodic_eval"):
        assert probe.phase == "periodic_eval"
        probe.forward(torch.randn(2, bench.spec.dim))

    assert probe.phase == "train"
    snap = probe.snapshot()
    # periodic_eval sneaks past the counter; only train is tallied.
    assert snap["periodic_eval"]["fwd_calls"] == 0
    assert snap["train"]["fwd_calls"] == 0


def test_backward_hook_fires(bench):
    probe = BenchProbe(bench)
    x = torch.randn(5, bench.spec.dim, requires_grad=True)

    obj, _ = probe.forward(x)
    obj.sum().backward()

    snap = probe.snapshot()
    assert snap["train"]["bwd_calls"] == 1
    assert snap["train"]["bwd_samples"] == 5


def test_backward_hook_skipped_when_no_grad(bench):
    probe = BenchProbe(bench)
    x = torch.randn(3, bench.spec.dim)
    with torch.no_grad():
        probe.forward(x)

    snap = probe.snapshot()
    assert snap["train"]["fwd_calls"] == 1
    assert snap["train"]["bwd_calls"] == 0


def test_objective_and_constraint_list_both_counted(bench):
    """PAL's path uses .objective() + .constraint_list() separately."""
    probe = BenchProbe(bench)
    x = torch.randn(4, bench.spec.dim)

    probe.objective(x)
    probe.constraint_list(x)

    snap = probe.snapshot()
    assert snap["train"]["fwd_calls"] == 2
    assert snap["train"]["fwd_samples"] == 8


def test_spec_delegation(bench):
    probe = BenchProbe(bench)
    assert probe.spec.id == bench.spec.id
    assert probe.spec.dim == bench.spec.dim


def test_flat_snapshot_keys(bench):
    probe = BenchProbe(bench)
    flat = probe.flat_snapshot()
    per_phase = {
        "fwd_calls", "fwd_samples", "bwd_calls", "bwd_samples", "opt_steps",
        "measurement_fwd_calls", "measurement_fwd_samples",
        "measurement_bwd_calls", "measurement_bwd_samples",
        "measurement_opt_steps",
    }
    phases = ("train", "periodic_eval", "final_eval", "predict")
    expected = {f"{p}__{k}" for p in phases for k in per_phase}
    assert set(flat.keys()) == expected
    assert all(v == 0 for v in flat.values())


def test_invalid_phase_rejected(bench):
    probe = BenchProbe(bench)
    with pytest.raises(ValueError, match="unknown phase"):
        probe.set_phase("invalid")


def test_mark_opt_step_bumps_train_counter(bench):
    """`mark_opt_step` only increments the train phase."""
    probe = BenchProbe(bench)
    probe.set_phase("train")

    probe.mark_opt_step()
    probe.mark_opt_step()
    probe.mark_opt_step()

    snap = probe.snapshot()
    assert snap["train"]["opt_steps"] == 3

    probe.set_phase("final_eval")
    probe.mark_opt_step()  # no-op under eval phase
    snap = probe.snapshot()
    assert snap["train"]["opt_steps"] == 3
    assert snap["final_eval"]["opt_steps"] == 0


def test_mark_opt_step_free_function_on_plain_bench(bench):
    """Module-level helper is a no-op when bench lacks `mark_opt_step`."""
    from pal.runner.probe import mark_opt_step

    mark_opt_step(bench)

    probe = BenchProbe(bench)
    probe.set_phase("train")
    mark_opt_step(probe)
    mark_opt_step(probe)
    assert probe.snapshot()["train"]["opt_steps"] == 2


def test_probe_under_vmap_jacrev():
    """Probe undercounts under `vmap(jacrev(...))`: fwd < B and bwd < B x K."""
    bench = get_benchmark("s5_overdetermined")
    probe = BenchProbe(bench)
    B = 4
    D = bench.spec.dim
    K = bench.spec.n_eq + bench.spec.n_ineq
    COND = bench.spec.condition_dim
    assert K > 1, "test assumes a multi-constraint bench"

    y = torch.randn(B, D, requires_grad=True)
    conds = torch.rand(B, COND) + 0.5

    def _single(y_i: torch.Tensor, c_i: torch.Tensor) -> torch.Tensor:
        return probe.constraints(y_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

    J = torch.vmap(torch.func.jacrev(_single, argnums=0))(y, conds)
    assert J.shape == (B, K, D)

    snap = probe.snapshot()["train"]
    assert snap["fwd_calls"] < B
    assert snap["bwd_calls"] < B * K
    assert snap["fwd_samples"] < B
    assert snap["bwd_samples"] < B * K


def test_probe_under_checkpoint_single_forward(bench):
    """One checkpointed forward shows up as 2 probe fwd events."""
    probe = BenchProbe(bench)
    B = 5
    x = torch.randn(B, bench.spec.dim, requires_grad=True)

    def _forward(x_: torch.Tensor) -> torch.Tensor:
        return probe.objective(x_, None)

    obj = torch.utils.checkpoint.checkpoint(_forward, x, use_reentrant=False)
    obj.sum().backward()

    snap = probe.snapshot()
    assert snap["train"]["fwd_calls"] == 2
    assert snap["train"]["fwd_samples"] == 2 * B
    assert snap["train"]["bwd_calls"] == 1
    assert snap["train"]["bwd_samples"] == B


def test_probe_under_checkpoint_jacobian_build():
    """Checkpointing the K-loop Jacobian adds about one recompute per VJP."""
    bench = get_benchmark("s5_overdetermined")
    D = bench.spec.dim
    K = bench.spec.n_eq + bench.spec.n_ineq
    COND = bench.spec.condition_dim
    B = 4

    def _run(use_checkpoint: bool) -> dict[str, int]:
        probe = BenchProbe(bench)
        y = torch.randn(B, D, requires_grad=True)
        conds = torch.rand(B, COND) + 0.5

        def _jac_build(y_in: torch.Tensor, c_in: torch.Tensor) -> torch.Tensor:
            y_det = y_in.detach().requires_grad_(True)
            c = probe.constraints(y_det, c_in)
            J = torch.zeros(B, K, D)
            for k in range(K):
                g = torch.autograd.grad(
                    c[:, k].sum(), y_det, retain_graph=(k < K - 1)
                )[0]
                J[:, k, :] = g
            return J

        if use_checkpoint:
            J = torch.utils.checkpoint.checkpoint(
                _jac_build, y, conds, use_reentrant=False,
            )
        else:
            J = _jac_build(y, conds)

        # (y*y).sum() forces the outer backward, which triggers checkpoint recompute.
        loss = (J * J).sum() + (y * y).sum()
        loss.backward()
        return probe.snapshot()["train"]

    baseline = _run(use_checkpoint=False)
    ckpt = _run(use_checkpoint=True)

    assert baseline["fwd_calls"] == 1
    assert baseline["bwd_calls"] == K
    assert ckpt["fwd_calls"] >= baseline["fwd_calls"] + K


def test_measurement_flag_double_ticks_counters(bench):
    """When measurement is on, the sampled bucket mirrors the regular one."""
    probe = BenchProbe(bench)
    probe.set_phase("train")

    x = torch.randn(4, bench.spec.dim, requires_grad=True)

    probe.forward(x)
    snap = probe.snapshot()["train"]
    assert snap["fwd_calls"] == 1
    assert snap["measurement_fwd_calls"] == 0
    assert snap["measurement_fwd_samples"] == 0

    # Flag on, both buckets increment, bwd also double-ticks through the hook.
    x2 = torch.randn(3, bench.spec.dim, requires_grad=True)
    probe.set_measurement(True)
    obj, _ = probe.forward(x2)
    obj.sum().backward()

    snap = probe.snapshot()["train"]
    assert snap["fwd_calls"] == 2
    assert snap["fwd_samples"] == 4 + 3
    assert snap["bwd_calls"] == 1
    assert snap["bwd_samples"] == 3
    assert snap["measurement_fwd_calls"] == 1
    assert snap["measurement_fwd_samples"] == 3
    assert snap["measurement_bwd_calls"] == 1
    assert snap["measurement_bwd_samples"] == 3
    assert snap["measurement_opt_steps"] == 0

    probe.mark_opt_step()
    assert probe.snapshot()["train"]["measurement_opt_steps"] == 1


def test_measurement_ctx_restores_prior_state(bench):
    probe = BenchProbe(bench)
    probe.set_phase("train")

    with probe.measurement_ctx(True):
        probe.forward(torch.randn(2, bench.spec.dim))
        assert probe.snapshot()["train"]["measurement_fwd_calls"] == 1

    probe.forward(torch.randn(2, bench.spec.dim))
    snap = probe.snapshot()["train"]
    assert snap["fwd_calls"] == 2
    assert snap["measurement_fwd_calls"] == 1

    probe.set_measurement(True)
    with probe.measurement_ctx(False):
        probe.forward(torch.randn(2, bench.spec.dim))
    snap = probe.snapshot()["train"]
    assert snap["fwd_calls"] == 3
    # Ctx was False during that forward, so measurement stays at 1.
    assert snap["measurement_fwd_calls"] == 1

    probe.forward(torch.randn(2, bench.spec.dim))
    snap = probe.snapshot()["train"]
    assert snap["measurement_fwd_calls"] == 2


def test_measurement_flag_stays_zero_in_eval_phase(bench):
    """Eval phases sneak past the probe, including the measurement bucket."""
    probe = BenchProbe(bench)
    probe.set_phase("final_eval")
    probe.set_measurement(True)

    obj, _ = probe.forward(torch.randn(4, bench.spec.dim, requires_grad=True))
    obj.sum().backward()

    snap = probe.snapshot()
    assert snap["final_eval"]["fwd_calls"] == 0
    assert snap["final_eval"]["measurement_fwd_calls"] == 0
    assert snap["final_eval"]["measurement_bwd_calls"] == 0
