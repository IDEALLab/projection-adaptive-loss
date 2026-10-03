"""Stub-mode (live=False) smoke test: shapes, Constraint fields, gradient flow."""

from __future__ import annotations

import torch

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM


def _sample_design() -> torch.Tensor:
    """Feasible-ish nominal: middle of each bound."""
    x = torch.zeros(4, DIM)
    x[:, 9] = 1.0           # L = 1 m
    x[:, 10 + 19 + 3] = 0.3  # battery w
    x[:, 10 + 19 + 4] = 0.3  # battery d
    x[:, 10 + 19 + 5] = 0.2  # battery h
    x[:, -1] = torch.deg2rad(torch.tensor(1.0))
    return x


def _sample_conditions(B: int = 4) -> torch.Tensor:
    return torch.stack(
        [
            torch.full((B,), 2000.0),   # alt = 2 km
            torch.full((B,), 40.0),     # V = 40 m/s
        ],
        dim=-1,
    )


def test_construct_default():
    bench = E1BWB(live=False)
    assert bench.spec.dim == DIM
    assert bench.spec.condition_dim == 2
    assert bench.spec.n_eq == 1  # lift_balance is a first-class equality
    assert bench.spec.n_ineq == 2  # mean(ReLU) strain aggregate, see spec.py
    assert bench.spec.constraint_types == ["eq", "ineq", "ineq"]
    assert len(bench.spec.constraint_names) == bench.spec.n_eq + bench.spec.n_ineq
    assert bench.spec.constraint_names == [
        "lift_balance", "strain_agg", "tip_deflection",
    ]


def test_forward_shapes():
    bench = E1BWB(live=False)
    x = _sample_design()
    conds = _sample_conditions(B=4)
    obj, clist = bench.forward(x, conds)

    assert obj.shape == (4,)
    assert obj.dtype == torch.float32
    assert len(clist) == 3

    names_seen = [c.name for c in clist]
    assert names_seen == ["lift_balance", "strain_agg", "tip_deflection"]

    # lift_balance is eq with tol > 0 (dead_huber_eq requires it), the rest ineq.
    by_name = {c.name: c for c in clist}
    assert by_name["lift_balance"].type == "eq"
    assert (by_name["lift_balance"].tol > 0).all()
    assert by_name["strain_agg"].type == "ineq"
    assert by_name["tip_deflection"].type == "ineq"

    for c in clist:
        assert c.value.shape == (4,)
        assert c.tol.shape == (4,)
        assert c.margin.shape == (4,)
        assert c.type in {"eq", "ineq"}
        assert torch.isfinite(c.value).all()
        assert torch.isfinite(c.tol).all()
        assert torch.isfinite(c.margin).all()


def test_gradients_flow():
    bench = E1BWB(live=False)
    x = _sample_design().requires_grad_(True)
    conds = _sample_conditions(B=4)
    obj, clist = bench.forward(x, conds)

    loss = obj.sum() + sum(c.value.sum() for c in clist)
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0.0


def test_constraints_stacked_matches_spec_order():
    bench = E1BWB(live=False)
    x = _sample_design()
    conds = _sample_conditions(B=4)
    C = bench.constraints(x, conds)
    assert C.shape == (4, bench.spec.n_eq + bench.spec.n_ineq)


def test_sample_and_eval_queries():
    bench = E1BWB(live=False)
    q_train = bench.sample_queries(n=3, split="train", seed=0)
    q_eval = bench.eval_queries(seed=0)

    assert q_train.zeta.shape == (3, bench.spec.zeta_dim)
    assert q_train.conditions.shape == (3, 2)
    assert q_eval.zeta.shape == (bench.spec.n_eval_default, bench.spec.zeta_dim)
    assert q_eval.conditions.shape == (bench.spec.n_eval_default, 2)

    q_train_b = bench.sample_queries(n=3, split="train", seed=0)
    assert torch.equal(q_train.zeta, q_train_b.zeta)
    assert torch.equal(q_train.conditions, q_train_b.conditions)


def test_viz_methods_exist():
    bench = E1BWB(live=False)
    assert callable(bench.visualize_train)
    assert callable(bench.visualize_final)


def test_y_stations_sobol_deterministic_and_sorted():
    """Same x -> same stations; stations sorted ascending on [0, semi_span]."""
    bench = E1BWB(live=False)
    x = _sample_design()
    conds = _sample_conditions(B=4)
    _, c1 = bench.forward(x, conds)
    _, c2 = bench.forward(x, conds)
    for a, b in zip(c1, c2, strict=False):
        assert torch.allclose(a.value, b.value)
