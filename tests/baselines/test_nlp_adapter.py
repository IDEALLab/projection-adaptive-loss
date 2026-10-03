"""Unit tests for `pal.baselines.nlp_adapter.NLPView` + `zeta_seeded_rng`."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import Tensor

from pal.baselines.nlp_adapter import NLPView, zeta_restart_rng, zeta_seeded_rng
from pal.benchmarks.base import BenchmarkSpec
from pal.constraints import Constraint


class _MockBench:
    """Minimal Benchmark for NLPView tests.

    f(x) = sum(x^2), c_k(x) = sin(x[k % dim]). Counts forward calls.
    """

    def __init__(self, constraint_types: list[str], dim: int = 3) -> None:
        n_eq = sum(1 for t in constraint_types if t == "eq")
        n_ineq = sum(1 for t in constraint_types if t == "ineq")
        self.spec = BenchmarkSpec(
            id="mock",
            family="mock",
            variant=None,
            dim=dim,
            n_eq=n_eq,
            n_ineq=n_ineq,
            constraint_names=[f"c{i}" for i in range(len(constraint_types))],
            constraint_types=list(constraint_types),
            output_bounds=(
                torch.full((dim,), -5.0, dtype=torch.float64),
                torch.full((dim,), 5.0, dtype=torch.float64),
            ),
            condition_dim=0,
            zeta_dim=2,
            tolerance=1e-4,
            cost="cheap",
            recommended_device="cpu",
        )
        self.forward_count = 0

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        self.forward_count += 1
        obj = (x ** 2).sum(dim=-1)
        B = x.shape[0]
        dim = x.shape[-1]
        cons: list[Constraint] = []
        for k, t in enumerate(self.spec.constraint_types):
            val = torch.sin(x[..., k % dim])
            tol = torch.full((B,), 1e-3, dtype=x.dtype, device=x.device)
            margin = torch.full((B,), 1e-4, dtype=x.dtype, device=x.device)
            cons.append(
                Constraint(value=val, type=t, tol=tol, margin=margin, name=f"c{k}")
            )
        return obj, cons


def _make(
    types: list[str], dim: int = 3, cache_size: int = 4
) -> tuple[_MockBench, NLPView]:
    bench = _MockBench(types, dim=dim)
    view = NLPView(
        bench,
        conditions=None,
        cache_size=cache_size,
        torch_dtype=torch.float64,
    )
    return bench, view


def test_partitioning_mixed_order_preserved() -> None:
    _, view = _make(["ineq", "eq", "ineq", "eq"])
    assert view.n_constraints == 4
    np.testing.assert_array_equal(view.eq_idx, [1, 3])
    np.testing.assert_array_equal(view.ineq_idx, [0, 2])


def test_partitioning_all_ineq() -> None:
    _, view = _make(["ineq"] * 5)
    np.testing.assert_array_equal(view.eq_idx, [])
    np.testing.assert_array_equal(view.ineq_idx, [0, 1, 2, 3, 4])


def test_partitioning_all_eq() -> None:
    _, view = _make(["eq"] * 3)
    np.testing.assert_array_equal(view.eq_idx, [0, 1, 2])
    np.testing.assert_array_equal(view.ineq_idx, [])


def test_partitioning_no_constraints() -> None:
    _, view = _make([])
    assert view.n_constraints == 0
    np.testing.assert_array_equal(view.eq_idx, [])
    np.testing.assert_array_equal(view.ineq_idx, [])


@pytest.mark.parametrize(
    "precision,expected",
    [
        ("fp32", torch.float32),
        ("fp64", torch.float64),
        ("fp16", torch.float16),
        ("bf16", torch.bfloat16),
    ],
)
def test_torch_dtype_inferred_from_spec_precision(
    precision: str, expected: torch.dtype
) -> None:
    bench = _MockBench(["eq"])
    object.__setattr__(bench.spec, "precision", precision)
    view = NLPView(bench, conditions=None, torch_dtype=None)
    assert view._torch_dtype == expected


def test_f_grad_g_jac_share_one_forward() -> None:
    bench, view = _make(["ineq", "eq"])
    x = np.array([0.1, 0.2, 0.3])

    view.f(x)
    view.grad_f(x)
    view.g(x)
    view.jac_g(x)

    assert bench.forward_count == 1


def test_repeat_call_same_x_is_cache_hit() -> None:
    bench, view = _make(["ineq"])
    x = np.array([1.0, 2.0, 3.0])
    f1 = view.f(x)
    g1 = view.g(x)
    f2 = view.f(x)
    g2 = view.g(x)
    assert f1 == f2
    np.testing.assert_array_equal(g1, g2)
    assert bench.forward_count == 1


def test_lru_evicts_oldest_when_over_capacity() -> None:
    bench, view = _make(["ineq"], cache_size=4)
    xs = [np.array([float(i), 0.0, 0.0]) for i in range(5)]

    for x in xs:
        view.f(x)
    assert bench.forward_count == 5

    for x in xs[1:]:
        view.f(x)
    assert bench.forward_count == 5

    view.f(xs[0])
    assert bench.forward_count == 6


def test_lru_touches_entry_on_hit() -> None:
    bench, view = _make(["ineq"], cache_size=4)
    xs = [np.array([float(i), 0.0, 0.0]) for i in range(5)]

    for x in xs[:4]:
        view.f(x)
    assert bench.forward_count == 4

    view.f(xs[0])
    assert bench.forward_count == 4

    view.f(xs[4])
    assert bench.forward_count == 5

    view.f(xs[0])
    assert bench.forward_count == 5

    view.f(xs[1])
    assert bench.forward_count == 6


def test_returned_arrays_are_copies() -> None:
    _, view = _make(["ineq"], dim=3)
    x = np.array([0.1, 0.2, 0.3])
    g1 = view.g(x)
    g1[0] = 999.0
    g2 = view.g(x)
    assert g2[0] != 999.0


def test_grad_f_matches_finite_difference() -> None:
    _, view = _make(["ineq", "eq"], dim=4)
    x = np.array([0.3, -0.4, 0.7, 1.1])
    g_auto = view.grad_f(x)

    eps = 1e-6
    g_fd = np.zeros_like(x)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        g_fd[i] = (view.f(x + e) - view.f(x - e)) / (2 * eps)

    np.testing.assert_allclose(g_auto, g_fd, atol=1e-6, rtol=1e-6)


def test_jac_g_matches_finite_difference() -> None:
    _, view = _make(["ineq", "eq", "ineq"], dim=4)
    x = np.array([0.3, -0.4, 0.7, 1.1])
    jac = view.jac_g(x)
    assert jac.shape == (3, 4)

    eps = 1e-6
    jac_fd = np.zeros_like(jac)
    for i in range(len(x)):
        e = np.zeros_like(x)
        e[i] = eps
        jac_fd[:, i] = (view.g(x + e) - view.g(x - e)) / (2 * eps)

    np.testing.assert_allclose(jac, jac_fd, atol=1e-6, rtol=1e-6)


def test_bounds_mirror_spec_output_bounds() -> None:
    _, view = _make(["ineq"], dim=3)
    np.testing.assert_array_equal(view.lo, [-5.0, -5.0, -5.0])
    np.testing.assert_array_equal(view.hi, [5.0, 5.0, 5.0])


def test_zeta_seeded_rng_deterministic_for_same_inputs() -> None:
    zeta = torch.tensor([0.1, 0.2, 0.3])
    cond = torch.tensor([1.0, 2.0])
    a = zeta_seeded_rng(zeta, cond).random(10)
    b = zeta_seeded_rng(zeta, cond).random(10)
    np.testing.assert_array_equal(a, b)


def test_zeta_seeded_rng_differs_on_zeta_change() -> None:
    cond = torch.tensor([1.0, 2.0])
    a = zeta_seeded_rng(torch.tensor([0.1, 0.2, 0.3]), cond).random(10)
    b = zeta_seeded_rng(torch.tensor([0.4, 0.5, 0.6]), cond).random(10)
    assert not np.array_equal(a, b)


def test_zeta_seeded_rng_differs_on_conditions_change() -> None:
    zeta = torch.tensor([0.1, 0.2, 0.3])
    a = zeta_seeded_rng(zeta, torch.tensor([1.0, 2.0])).random(10)
    b = zeta_seeded_rng(zeta, torch.tensor([1.0, 3.0])).random(10)
    assert not np.array_equal(a, b)


def test_zeta_seeded_rng_handles_empty_conditions() -> None:
    zeta = torch.tensor([0.1, 0.2, 0.3])
    rng_none = zeta_seeded_rng(zeta, None).random(5)
    rng_empty = zeta_seeded_rng(zeta, torch.zeros(0)).random(5)
    np.testing.assert_array_equal(rng_none, rng_empty)


def test_zeta_restart_rng_deterministic_for_same_inputs() -> None:
    zeta = torch.tensor([0.1, 0.2, 0.3])
    cond = torch.tensor([1.0, 2.0])
    a = zeta_restart_rng(zeta, cond, restart_idx=5).random(10)
    b = zeta_restart_rng(zeta, cond, restart_idx=5).random(10)
    np.testing.assert_array_equal(a, b)


def test_zeta_restart_rng_differs_per_restart_idx() -> None:
    """Different restart indices give distinct draws."""
    zeta = torch.tensor([0.1, 0.2, 0.3])
    cond = torch.tensor([1.0, 2.0])
    draws = [zeta_restart_rng(zeta, cond, r).random(10) for r in range(20)]
    for i in range(len(draws)):
        for j in range(i + 1, len(draws)):
            assert not np.array_equal(draws[i], draws[j])


def test_zeta_restart_rng_standalone_reproduces_slot() -> None:
    """Restart 5 run standalone gives the same draw as slot 5 of a full loop."""
    zeta = torch.tensor([0.1, 0.2, 0.3])
    cond = torch.tensor([1.0, 2.0])
    in_process = [zeta_restart_rng(zeta, cond, r).random(10) for r in range(8)]
    standalone_5 = zeta_restart_rng(zeta, cond, 5).random(10)
    np.testing.assert_array_equal(in_process[5], standalone_5)


def test_rejects_wrong_shape() -> None:
    _, view = _make(["ineq"], dim=3)
    with pytest.raises(ValueError):
        view.f(np.array([0.1, 0.2]))
    with pytest.raises(ValueError):
        view.f(np.array([0.1, 0.2, 0.3, 0.4]))
