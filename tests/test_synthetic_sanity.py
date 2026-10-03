"""Sanity tests for the s1-s6 conditional synthetic benchmarks."""

from __future__ import annotations

import pytest
import torch

from pal.benchmarks.synthetic.s1_sphere_track import S1SphereTrack
from pal.benchmarks.synthetic.s2_active_set_switch import S2ActiveSetSwitch
from pal.benchmarks.synthetic.s3_illcond_tube import S3IllcondTube
from pal.benchmarks.synthetic.s4_qv_coupling import S4QvCoupling
from pal.benchmarks.synthetic.s5_overdetermined import S5Overdetermined
from pal.benchmarks.synthetic.s6_redundant_ineq import S6RedundantIneq

BENCHES = [
    ("s1", S1SphereTrack, 100),
    ("s2", S2ActiveSetSwitch, 100),
    ("s3", S3IllcondTube, 100),
    ("s4", S4QvCoupling, 100),
    ("s5", S5Overdetermined, 100),
    ("s6", S6RedundantIneq, 30),  # fewer samples for the 300-row bench
]


def _build(cls):
    return cls()


@pytest.mark.parametrize("name,cls,n_samples", BENCHES)
def test_feasibility_at_y_star(name, cls, n_samples):
    bench = _build(cls)
    q = bench.sample_queries(n_samples, "train", seed=42)
    y_star = bench.y_star(q.conditions)
    _, constraints = bench.forward(y_star, q.conditions)

    max_eq_violation = 0.0
    max_ineq_violation = 0.0
    for con in constraints:
        v = con.value
        if con.type == "eq":
            max_eq_violation = max(max_eq_violation, v.abs().max().item())
        else:
            max_ineq_violation = max(max_ineq_violation, v.clamp(min=0).max().item())

    assert max_eq_violation <= 1e-5, f"{name}: eq violation at y* = {max_eq_violation:.2e}"
    assert max_ineq_violation <= 1e-5, f"{name}: ineq violation at y* = {max_ineq_violation:.2e}"


@pytest.mark.parametrize("name,cls,n_samples", BENCHES)
def test_objective_finite_nonneg_at_y_star(name, cls, n_samples):
    """Objective at the feasible reference point is finite and non-negative."""
    bench = _build(cls)
    q = bench.sample_queries(n_samples, "train", seed=7)
    y_star = bench.y_star(q.conditions)
    obj = bench.objective(y_star, q.conditions)
    assert torch.isfinite(obj).all(), f"{name}: obj(y*) not finite"
    assert (obj >= 0).all(), f"{name}: obj(y*) negative, quadratic invariant broken"


@pytest.mark.parametrize("name,cls", [("s1", S1SphereTrack), ("s4", S4QvCoupling)])
def test_eq_jacobian_full_rank(name, cls):
    """s1 has 1 eq -> rank 1; s4 has 25 eqs on n=30 -> rank 25."""
    bench = _build(cls)
    expected_rank = bench.spec.n_eq
    q = bench.sample_queries(8, "train", seed=3)
    y_star = bench.y_star(q.conditions)

    for i in range(q.conditions.shape[0]):
        y_i = y_star[i:i+1].clone().requires_grad_(True)
        c_i = q.conditions[i:i+1]
        cs = bench.constraints(y_i, c_i)
        eq_vals = cs[:, : bench.spec.n_eq]
        J = torch.zeros(expected_rank, bench.spec.dim)
        for j in range(expected_rank):
            grads = torch.autograd.grad(eq_vals[0, j], y_i, retain_graph=True)[0]
            J[j] = grads[0]
        rank = torch.linalg.matrix_rank(J, tol=1e-6).item()
        assert rank == expected_rank, f"{name} c[{i}]: rank={rank}, expected={expected_rank}"


def test_s5_jacobian_rank_4_not_5():
    """s5 claim: 5 eqs on n=4, so J in R^{5x4} has rank <= 4. Confirms overdetermination."""
    bench = _build(S5Overdetermined)
    q = bench.sample_queries(8, "train", seed=3)
    y_star = bench.y_star(q.conditions)

    for i in range(q.conditions.shape[0]):
        y_i = y_star[i:i+1].clone().requires_grad_(True)
        c_i = q.conditions[i:i+1]
        cs = bench.constraints(y_i, c_i)
        J = torch.zeros(5, 4)
        for j in range(5):
            grads = torch.autograd.grad(cs[0, j], y_i, retain_graph=True)[0]
            J[j] = grads[0]
        rank = torch.linalg.matrix_rank(J, tol=1e-6).item()
        assert rank == 4, f"s5 c[{i}]: rank={rank}, expected=4 (less than n_eq=5)"


@pytest.mark.parametrize("name,cls,_n", BENCHES)
def test_spec_shapes(name, cls, _n):
    bench = _build(cls)
    spec = bench.spec
    assert len(spec.constraint_names) == spec.n_eq + spec.n_ineq
    assert len(spec.constraint_types) == spec.n_eq + spec.n_ineq
    eq_count = sum(1 for t in spec.constraint_types if t == "eq")
    ineq_count = sum(1 for t in spec.constraint_types if t == "ineq")
    assert eq_count == spec.n_eq
    assert ineq_count == spec.n_ineq


@pytest.mark.parametrize("name,cls,_n", BENCHES)
def test_sample_eval_shapes(name, cls, _n):
    bench = _build(cls)
    for seed in (0, 1, 2):
        q = bench.sample_queries(16, "train", seed=seed)
        assert q.zeta.shape == (16, bench.spec.zeta_dim)
        assert q.conditions.shape == (16, bench.spec.condition_dim)
        q_e = bench.eval_queries(seed=seed, n=8)
        assert q_e.zeta.shape == (8, bench.spec.zeta_dim)
        assert q_e.conditions.shape == (8, bench.spec.condition_dim)


def test_s6_projector_smoke():
    """Smoke: ensure forward on s6 at train batch size works without OOM/NaN."""

    bench = S6RedundantIneq()
    B = 64
    q = bench.sample_queries(B, "train", seed=0)
    lo, hi = bench.spec.output_bounds
    y = lo + (hi - lo) * torch.rand(B, bench.spec.dim)
    obj, cs = bench.forward(y, q.conditions)
    assert obj.shape == (B,)
    assert not obj.isnan().any()
    for con in cs:
        assert con.value.shape == (B,)
        assert not con.value.isnan().any()
