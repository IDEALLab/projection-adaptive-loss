"""Sanity and certification tests for the curvature_warp warp control, paired with curvature_sine.

The warp is `T(v) = v + a*sin(omega*b^T v)`; on the manifold the problem is the affine k0 problem.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest
import torch

from pal.benchmarks import get as get_benchmark
from pal.benchmarks import list_all as list_all_benchmarks
from pal.benchmarks.synthetic.curvature_hinge import CurvatureHinge
from pal.benchmarks.synthetic.curvature_sine import (
    KAPPA_BY_VARIANT as CURVATURE_SINE_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_sine import (
    CurvatureSine,
)
from pal.benchmarks.synthetic.curvature_warp import (
    KAPPA_BY_VARIANT,
    OMEGA_BY_VARIANT,
    VARIANTS,
    CurvatureWarp,
    unwarp,
    warp,
    warp_jacobian,
)

BENCH_IDS = [f"curvature_warp_{v}" for v in VARIANTS]

#: k0 ... k10: the points where a paired curvature_sine arm exists.
SHARED_VARIANTS = [v for v in VARIANTS if v in CURVATURE_SINE_KAPPA_BY_VARIANT]
#: k11 ... k13, the curvature_warp-only high-curvature extension (kappa = 1e4, 1e5, 1e6).
EXTENDED_VARIANTS = [v for v in VARIANTS if v not in CURVATURE_SINE_KAPPA_BY_VARIANT]

#: Largest omega with a paired curvature_sine variant, the reference for omega-scaled tolerances.
_SHARED_OMEGA_MAX = max(OMEGA_BY_VARIANT[v] for v in SHARED_VARIANTS)


def _phase_atol(bench, base: float) -> float:
    """`base` on k0 ... k10, scaled by omega past it (a phase-error allowance).

    float64 rounding of `b^T*` enters `sin(omega*b^T*)` as a phase error of omega*1e-16*|b^T*|.
    """
    return base * max(1.0, bench.omega / _SHARED_OMEGA_MAX)


def test_all_fourteen_variants_registered():
    assert VARIANTS == [f"k{i}" for i in range(14)]
    assert SHARED_VARIANTS == [f"k{i}" for i in range(11)]
    assert EXTENDED_VARIANTS == ["k11", "k12", "k13"]
    available = set(list_all_benchmarks())
    missing = [b for b in BENCH_IDS if b not in available]
    assert not missing, f"unregistered curvature_warp variants: {missing}"


@pytest.mark.parametrize("bench_id", BENCH_IDS)
def test_registry_returns_matching_variant(bench_id):
    bench = get_benchmark(bench_id)
    assert bench.spec.id == bench_id
    assert bench.spec.family == "curvature_warp"
    assert bench.spec.variant == bench_id.removeprefix("curvature_warp_")
    assert bench.kappa == KAPPA_BY_VARIANT[bench.spec.variant]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="unknown curvature_warp variant"):
        CurvatureWarp(variant="k99")


def test_kappa_grid_extends_curvature_sine_grid():
    """Same dial on the shared points, three decades further on k11 ... k13."""
    assert {v: KAPPA_BY_VARIANT[v] for v in SHARED_VARIANTS} == CURVATURE_SINE_KAPPA_BY_VARIANT
    assert [KAPPA_BY_VARIANT[v] for v in VARIANTS] == pytest.approx(
        [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0,
         1e4, 1e5, 1e6]
    )
    # the extension is strictly increasing and strictly above curvature_sine's top point
    assert [KAPPA_BY_VARIANT[v] for v in EXTENDED_VARIANTS] == pytest.approx(
        [1e4, 1e5, 1e6]
    )
    assert min(KAPPA_BY_VARIANT[v] for v in EXTENDED_VARIANTS) > max(
        CURVATURE_SINE_KAPPA_BY_VARIANT.values()
    )
    assert OMEGA_BY_VARIANT["k0"] == 0.0
    for v in VARIANTS[1:]:
        omega = OMEGA_BY_VARIANT[v]
        assert omega == pytest.approx(math.sqrt(KAPPA_BY_VARIANT[v]))
        assert omega * omega == pytest.approx(KAPPA_BY_VARIANT[v])
        assert CurvatureWarp(v).is_affine_anchor is False
    assert CurvatureWarp("k0").is_affine_anchor is True
    assert [OMEGA_BY_VARIANT[v] for v in EXTENDED_VARIANTS] == pytest.approx(
        [100.0, 316.22776601683796, 1000.0]
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_spec_shape(variant):
    spec = CurvatureWarp(variant).spec
    assert spec.dim == 8
    assert spec.condition_dim == 8
    assert spec.zeta_dim == 0
    assert spec.n_eq == 1
    assert spec.n_ineq == 0
    assert spec.constraint_types == ["eq"]
    assert len(spec.constraint_names) == 1
    assert spec.n_eval_default == 512
    lo, hi = spec.output_bounds
    assert lo.shape == (8,) and hi.shape == (8,)
    assert bool((lo < hi).all())


@pytest.mark.parametrize("variant", SHARED_VARIANTS)
def test_spec_matches_curvature_sine_where_the_arms_must_agree(variant):
    """Bounds, tolerances, batch and eval sizes are shared across the paired arms."""
    s33, s32 = CurvatureWarp(variant).spec, CurvatureSine(variant).spec
    assert torch.equal(s33.output_bounds[0], s32.output_bounds[0])
    assert torch.equal(s33.output_bounds[1], s32.output_bounds[1])
    assert s33.tolerance == s32.tolerance
    assert s33.tau == s32.tau
    assert s33.train_batch_size == s32.train_batch_size
    assert s33.n_eval_default == s32.n_eval_default
    assert s33.constraint_names == s32.constraint_names


@pytest.mark.parametrize("variant", VARIANTS)
def test_sample_eval_shapes(variant):
    bench = CurvatureWarp(variant)
    for seed in (0, 1, 2):
        q = bench.sample_queries(16, "train", seed=seed)
        assert q.zeta.shape == (16, 0)
        assert q.conditions.shape == (16, 8)
        assert q.conditions.min() >= -1.5 and q.conditions.max() <= 1.5
        assert bench.eval_queries(seed=seed, n=8).conditions.shape == (8, 8)
    assert bench.eval_queries(seed=0).conditions.shape == (512, 8)


def test_query_sets_and_directions_identical_across_variants():
    """Paired randomness within curvature_warp: kappa is the only thing that changes."""
    ref = CurvatureWarp("k0")
    ref_train = ref.sample_queries(64, "train", seed=5).conditions
    ref_eval = ref.eval_queries(seed=5, n=64).conditions
    for variant in VARIANTS[1:]:
        bench = CurvatureWarp(variant)
        assert torch.equal(
            bench.sample_queries(64, "train", seed=5).conditions, ref_train
        )
        assert torch.equal(bench.eval_queries(seed=5, n=64).conditions, ref_eval)
        assert torch.equal(bench._a, ref._a)
        assert torch.equal(bench._b, ref._b)


@pytest.mark.parametrize("variant", SHARED_VARIANTS)
@pytest.mark.parametrize("seed", [0, 1, 7])
def test_queries_and_directions_bit_identical_to_curvature_sine(variant, seed):
    """curvature_warp and curvature_sine share a, b and queries at every grid point and seed."""
    bw, bs = CurvatureWarp(variant), CurvatureSine(variant)
    assert torch.equal(bw._a, bs._a)
    assert torch.equal(bw._b, bs._b)
    for n in (64, 512):
        assert torch.equal(
            bw.eval_queries(seed=seed, n=n).conditions,
            bs.eval_queries(seed=seed, n=n).conditions,
        )
        assert torch.equal(
            bw.sample_queries(n, "train", seed=seed).conditions,
            bs.sample_queries(n, "train", seed=seed).conditions,
        )
    assert torch.equal(
        bw.eval_queries(seed=seed).conditions, bs.eval_queries(seed=seed).conditions
    )


def test_directions_shared_with_the_whole_dial_category():
    """All three curvature dials sit on the same (a, b): one geometry, three arms."""
    bw, bs, bh = CurvatureWarp("k4"), CurvatureSine("k4"), CurvatureHinge("k4")
    assert torch.equal(bw._a, bs._a) and torch.equal(bw._a, bh._a)
    assert torch.equal(bw._b, bs._b) and torch.equal(bw._b, bh._b)
    assert torch.equal(
        bw.eval_queries(seed=0, n=64).conditions,
        bh.eval_queries(seed=0, n=64).conditions,
    )


def test_directions_are_orthonormal():
    bench = CurvatureWarp("k4")
    a, b = bench._a, bench._b
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert float(a.norm()) == pytest.approx(1.0, abs=1e-15)
    assert float(b.norm()) == pytest.approx(1.0, abs=1e-15)
    assert abs(float(torch.dot(a, b))) <= 1e-15


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_projection_stays_inside_output_bounds(variant):
    """Data-derivation of the +/-4 box: the exact optimum never leaves it."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=0, n=2048).conditions
    y_star = bench.y_star(x)
    lo, hi = bench.spec.output_bounds
    assert bool(((y_star >= lo.double()) & (y_star <= hi.double())).all())
    # and the analytic envelope |y*| <= 1.5 + 2.5*max|a_i| is respected
    assert float(y_star.abs().max()) <= 1.5 + 2.5 * float(bench._a.abs().max())


def _sample_v(n: int, seed: int, scale: float = 4.0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    return scale * (2.0 * torch.rand(n, 8, generator=g, dtype=torch.float64) - 1.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_warp_inverse_is_exact_both_ways(variant):
    """`T^-1 o T = T o T^-1 = id` in closed form, no Newton solve anywhere."""
    bench = CurvatureWarp(variant)
    v = _sample_v(512, seed=31)
    # omega-scaled past k10: the round trip re-evaluates sin(omega*b^T*) on a rounded b^T*.
    atol = _phase_atol(bench, 1e-13)
    torch.testing.assert_close(bench.unwarp(bench.warp(v)), v, rtol=0.0, atol=atol)
    torch.testing.assert_close(bench.warp(bench.unwarp(v)), v, rtol=0.0, atol=atol)


@pytest.mark.parametrize("variant", VARIANTS)
def test_warp_preserves_the_b_coordinate(variant):
    """`b^T T(v) = b^T v` because `a perp b`, this is what makes `T^-1` closed form."""
    bench = CurvatureWarp(variant)
    v = _sample_v(512, seed=32)
    torch.testing.assert_close(bench.t_of(bench.warp(v)), bench.t_of(v))
    torch.testing.assert_close(bench.t_of(bench.unwarp(v)), bench.t_of(v))


@pytest.mark.parametrize("variant", VARIANTS)
def test_warp_moves_only_along_a(variant):
    """`T(v) - v` is parallel to `a` with magnitude `sin(omega*b^T v)`."""
    bench = CurvatureWarp(variant)
    v = _sample_v(256, seed=33)
    delta = bench.warp(v) - v
    coeff = delta @ bench._a
    torch.testing.assert_close(coeff, bench.p(bench.t_of(v)))
    torch.testing.assert_close(delta, coeff.unsqueeze(-1) * bench._a)
    assert float(coeff.abs().max()) <= 1.0 + 1e-12


@pytest.mark.parametrize("variant", ["k0", "k4", "k10", "k13"])
def test_module_level_warp_functions_match_the_bench_methods(variant):
    """The free functions are the public analysis API; keep them in step."""
    bench = CurvatureWarp(variant)
    v = _sample_v(64, seed=38)
    a, b = bench._a, bench._b
    torch.testing.assert_close(warp(v, a, b, bench.omega), bench.warp(v))
    torch.testing.assert_close(unwarp(v, a, b, bench.omega), bench.unwarp(v))
    torch.testing.assert_close(
        warp_jacobian(v, a, b, bench.omega), bench.warp_jacobian(v)
    )


def test_warp_is_the_identity_at_k0():
    bench = CurvatureWarp("k0")
    v = _sample_v(256, seed=34)
    torch.testing.assert_close(bench.warp(v), v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(bench.unwarp(v), v, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("variant", VARIANTS)
def test_warp_jacobian_matches_autograd_and_has_unit_determinant(variant):
    """`dT/dv = I + omega cos(omega*b^T v)*a b^T`; `det = 1` since `a perp b`."""
    bench = CurvatureWarp(variant)
    v = _sample_v(6, seed=35)
    J = bench.warp_jacobian(v)
    for i in range(v.shape[0]):
        J_auto = torch.autograd.functional.jacobian(
            lambda w: warp(w.unsqueeze(0), bench._a, bench._b, bench.omega).squeeze(0),
            v[i],
        )
        torch.testing.assert_close(J[i], J_auto, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(
        torch.linalg.det(J), torch.ones(v.shape[0], dtype=torch.float64)
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_manifold_is_the_image_of_the_hyperplane_under_the_warp(variant):
    """`{g = 0} = T({a^T z = 0})`, and `a^T T^-1(y) = g(y)` everywhere."""
    bench = CurvatureWarp(variant)
    a = bench._a
    z = _sample_v(512, seed=36)
    z = z - (z @ a).unsqueeze(-1) * a  # onto the hyperplane a.z = 0
    assert float((z @ a).abs().max()) <= 1e-14
    y = bench.warp(z)
    # omega-scaled past k10: g re-evaluates the sine at a rounded b^T y (phase error ~ omega).
    assert float(bench.g(y).abs().max()) <= _phase_atol(bench, 1e-13)
    # converse: the pullback of g is the linear functional a.z
    y_free = _sample_v(512, seed=37)
    torch.testing.assert_close(bench.unwarp(y_free) @ a, bench.g(y_free))


def _sample_y(bench, n: int, seed: int) -> torch.Tensor:
    lo, hi = bench.spec.output_bounds
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, bench.spec.dim, generator=g)


@pytest.mark.parametrize("variant", SHARED_VARIANTS)
def test_constraint_is_bit_identical_to_curvature_sine(variant):
    """Same manifold, not merely the same formula: values agree exactly."""
    curvature_warp, curvature_sine = CurvatureWarp(variant), CurvatureSine(variant)
    y = _sample_y(curvature_warp, 512, seed=41).double()
    assert torch.equal(curvature_warp.constraints(y, None), curvature_sine.constraints(y, None))
    assert torch.equal(curvature_warp.g(y), curvature_sine.g(y))
    assert torch.equal(curvature_warp.grad_g(y), curvature_sine.grad_g(y))
    assert torch.equal(curvature_warp.snap(y), curvature_sine.snap(y))
    t = y @ curvature_warp._b
    assert torch.equal(curvature_warp.curvature(t), curvature_sine.curvature(t))
    assert torch.equal(curvature_warp.fold(t), curvature_sine.fold(t))
    assert torch.equal(curvature_warp.p(t), curvature_sine.p(t))
    assert torch.equal(curvature_warp.p_prime(t), curvature_sine.p_prime(t))
    assert torch.equal(curvature_warp.p_second(t), curvature_sine.p_second(t))


@pytest.mark.parametrize("variant", VARIANTS)
def test_constraint_is_raw_g(variant):
    """Exposed value is exactly `a^T y - sin(omega*b^T y)`, no ||grad g|| normalization."""
    bench = CurvatureWarp(variant)
    y = _sample_y(bench, 64, seed=42).double()
    q = bench.sample_queries(64, "train", seed=42)
    expected = y @ bench._a - bench.p(y @ bench._b)

    got = bench.constraints(y, q.conditions.double())
    assert got.shape == (64, 1)
    torch.testing.assert_close(got[:, 0], expected)
    _, cons = bench.forward(y, q.conditions.double())
    assert len(cons) == 1 and cons[0].type == "eq" and cons[0].name == "sine_ripple"
    torch.testing.assert_close(cons[0].value, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_grad_g_matches_autograd(variant):
    bench = CurvatureWarp(variant)
    y = _sample_y(bench, 16, seed=43).double().requires_grad_(True)
    g_val = bench.constraints(y, None)[:, 0]
    (g_auto,) = torch.autograd.grad(g_val.sum(), y)
    torch.testing.assert_close(bench.grad_g(y.detach()), g_auto)
    assert bool(torch.isfinite(g_auto).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_grad_norm_in_one_to_sqrt_one_plus_kappa_band(variant):
    """As in curvature_sine, `||grad g||` lies in `[1, sqrt(1+kappa)]`."""
    bench = CurvatureWarp(variant)
    hi = math.sqrt(1.0 + bench.kappa)
    y = _sample_y(bench, 4096, seed=44).double()
    norms = bench.grad_g(y).norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-12
    assert float(norms.max()) <= hi + 1e-12


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_kappa_is_exactly_the_max_curvature(variant):
    """`K = kappa` at every crest, the shared manifold's defining property."""
    bench = CurvatureWarp(variant)
    omega, kappa = bench.omega, bench.kappa
    crests = (0.5 + torch.arange(-3, 4, dtype=torch.float64)) * math.pi / omega
    k_crest = bench.curvature(crests)
    torch.testing.assert_close(
        k_crest, torch.full_like(k_crest, kappa), rtol=1e-12, atol=0.0
    )
    t = torch.linspace(-4.0, 4.0, 200001, dtype=torch.float64)
    assert float(bench.curvature(t).max()) <= kappa * (1 + 1e-12)


def test_k0_anchor_is_exactly_linear():
    """k0: omega = 0 => p == 0, T = id, constraint Hessian identically zero."""
    bench = CurvatureWarp("k0")
    t = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    assert float(bench.p(t).abs().max()) == 0.0
    assert float(bench.curvature(t).abs().max()) == 0.0
    y = _sample_y(bench, 1, seed=45).double().requires_grad_(True)
    H = torch.autograd.functional.hessian(
        lambda z: bench.constraints(z, None).sum(), y
    )
    assert float(H.abs().max()) == pytest.approx(0.0, abs=1e-14)
    yd = _sample_y(bench, 32, seed=46).double()
    torch.testing.assert_close(bench.g(yd), yd @ bench._a)


@pytest.mark.parametrize("variant", VARIANTS)
def test_float32_training_path_stays_inside_the_tolerance(variant):
    """The float32 penalty is the rounded sine phase only, within tol 1e-3 (omega-scaled)."""
    bench = CurvatureWarp(variant)
    y32 = _sample_y(bench, 4096, seed=47)
    g32 = bench.constraints(y32, None)[:, 0]
    g64 = bench.constraints(y32.double(), None)[:, 0]
    assert float((g32.double() - g64).abs().max()) < _phase_atol(bench, 1e-4)


@pytest.mark.parametrize("variant", VARIANTS)
def test_objective_is_the_warped_squared_distance(variant):
    bench = CurvatureWarp(variant)
    q = bench.sample_queries(256, "train", seed=51)
    y = _sample_y(bench, 256, seed=51).double()
    x = q.conditions.double()
    expected = ((bench.unwarp(y) - bench.unwarp(x)) ** 2).sum(dim=-1)
    torch.testing.assert_close(bench.objective(y, x), expected)
    assert bool(torch.isfinite(expected).all()) and bool((expected >= 0).all())
    # zero exactly on the diagonal (T is injective)
    torch.testing.assert_close(
        bench.objective(x, x), torch.zeros(256, dtype=torch.float64)
    )


def test_objective_reduces_to_the_euclidean_one_at_k0():
    """k0: T = identity, so curvature_warp k0 IS curvature_sine k0 (shared anchor for both arms)."""
    curvature_warp, curvature_sine = CurvatureWarp("k0"), CurvatureSine("k0")
    x = curvature_warp.eval_queries(seed=0, n=256).conditions.double()
    y = _sample_y(curvature_warp, 256, seed=52).double()
    torch.testing.assert_close(curvature_warp.objective(y, x), curvature_sine.objective(y, x))
    torch.testing.assert_close(
        curvature_warp.reference_solution(x).y_star, curvature_sine.reference_solution(x).y_star
    )
    torch.testing.assert_close(
        curvature_warp.reference_solution(x).f_star, curvature_sine.reference_solution(x).f_star
    )


@pytest.mark.parametrize("variant", SHARED_VARIANTS[1:])
def test_objective_differs_from_curvature_sine_off_the_anchor(variant):
    """Guard against an accidental copy of curvature_sine: the objectives must not agree."""
    curvature_warp, curvature_sine = CurvatureWarp(variant), CurvatureSine(variant)
    x = curvature_warp.eval_queries(seed=0, n=256).conditions.double()
    y = _sample_y(curvature_warp, 256, seed=53).double()
    diff = (curvature_warp.objective(y, x) - curvature_sine.objective(y, x)).abs()
    assert float(diff.max()) > 1e-3


@pytest.mark.parametrize("variant", VARIANTS)
def test_objective_gradient_comes_from_autograd_and_is_finite(variant):
    """No hand-derived Jacobians: autograd through the closed-form `T^-1`."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=0, n=8).conditions.double()
    y = _sample_y(bench, 8, seed=54).double().requires_grad_(True)
    (grad,) = torch.autograd.grad(bench.objective(y, x).sum(), y)
    assert bool(torch.isfinite(grad).all())
    # central differences agree (float64, eps balanced against |p'''| = omega^3)
    eps = 5e-8
    fd = torch.zeros_like(grad)
    yd = y.detach()
    for j in range(bench.spec.dim):
        e = torch.zeros_like(yd)
        e[:, j] = eps
        fd[:, j] = (bench.objective(yd + e, x) - bench.objective(yd - e, x)) / (2 * eps)
    torch.testing.assert_close(grad, fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_restricted_problem_is_the_affine_k0_problem(variant):
    """For feasible `y = T(z)`, `f(y; x) = ||z - z_x||^2`: the k0 objective in `z` coordinates."""
    bench = CurvatureWarp(variant)
    a = bench._a
    x = bench.eval_queries(seed=2, n=256).conditions.double()
    z_x = bench.unwarp(x)
    z = _sample_v(256, seed=55)
    z = z - (z @ a).unsqueeze(-1) * a  # feasible after warping
    y = bench.warp(z)
    assert float(bench.g(y).abs().max()) <= _phase_atol(bench, 1e-13)
    torch.testing.assert_close(
        bench.objective(y, x),
        ((z - z_x) ** 2).sum(dim=-1),
        rtol=_phase_atol(bench, 1e-12),
        atol=_phase_atol(bench, 1e-12),
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_condition_box_straddles_manifold(variant):
    """Straddle guard, inherited: sign(g(x)) varies over the box at every kappa."""
    bench = CurvatureWarp(variant)
    q = bench.sample_queries(4096, "train", seed=56)
    g = bench.g(q.conditions)
    frac_pos = float((g > 0).double().mean())
    assert 0.3 < frac_pos < 0.7, f"{variant}: box does not straddle g=0 ({frac_pos=})"
    f_star = bench.reference_solution(q.conditions).f_star
    assert 0.01 < float(f_star.sqrt().mean()) < 10.0


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_solution_is_feasible_and_consistent(variant):
    """`g(y*) = 0` to float64 tolerance and `f(y*; x) = f*`, the spec identities."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=0, n=512).conditions.double()
    sol = bench.reference_solution(x)
    assert float(bench.g(sol.y_star).abs().max()) <= _phase_atol(bench, 1e-12)
    torch.testing.assert_close(
        bench.objective(sol.y_star, x),
        sol.f_star,
        rtol=1e-12,
        atol=_phase_atol(bench, 1e-13),
    )
    assert bool((sol.f_star >= 0).all())
    torch.testing.assert_close(sol.t_star, bench.t_of(sol.y_star))
    torch.testing.assert_close(sol.t_star, bench.t_of(x))  # the shear preserves b.x
    torch.testing.assert_close(bench.y_star(x), sol.y_star)
    assert bool(sol.from_stationary.all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_matches_the_closed_form_identities(variant):
    """`f* = g(x)^2` and `y* = x - g(x)*a = snap(x)` (unwound shear)."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=1, n=512).conditions.double()
    sol = bench.reference_solution(x)
    g_x = bench.g(x)
    torch.testing.assert_close(sol.f_star, g_x * g_x)
    rtol, atol = _phase_atol(bench, 1e-13), _phase_atol(bench, 1e-14)
    torch.testing.assert_close(
        sol.y_star, x - g_x.unsqueeze(-1) * bench._a, rtol=rtol, atol=atol
    )
    torch.testing.assert_close(sol.y_star, bench.snap(x), rtol=rtol, atol=atol)


@pytest.mark.parametrize("variant", VARIANTS)
def test_target_map_is_continuous_in_x(variant):
    """`x -> y*(x)` is smooth: bounded difference quotients at the sine selector-flip scale."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=4, n=1024).conditions.double()
    step = 1e-6
    dirs = _sample_v(1024, seed=57, scale=1.0)
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)
    y0 = bench.y_star(x)
    y1 = bench.y_star(x + step * dirs)
    lip = (y1 - y0).norm(dim=-1) / step
    # |y*(x) - y*(x')| <= (1 + |a| * ||grad g||) |x - x'| <= 1 + sqrt(1+kappa)
    assert float(lip.max()) <= 1.0 + math.sqrt(1.0 + bench.kappa) + 1e-3


def _manifold_chart(bench):
    """Orthonormal `[a, b, c1 ... c6]`; charts `{g = 0}` by `(t, s) in R^7`."""
    basis = torch.linalg.qr(
        torch.cat(
            [
                bench._a.unsqueeze(1),
                bench._b.unsqueeze(1),
                torch.eye(8, dtype=torch.float64),
            ],
            dim=1,
        )
    )[0][:, :8]
    # qr may flip signs; realign the first two columns with (a, b)
    a_col = basis[:, 0] * torch.sign(basis[:, 0] @ bench._a)
    b_col = basis[:, 1] * torch.sign(basis[:, 1] @ bench._b)
    torch.testing.assert_close(a_col, bench._a, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(b_col, bench._b, rtol=1e-12, atol=1e-12)
    return basis[:, 2:]  # the 6 directions orthogonal to both a and b


def _point_on_manifold(bench, t: torch.Tensor, s: torch.Tensor, c: torch.Tensor):
    """`y(t, s) = t*b + sin(omega t)*a + sum s_i c_i`, feasible for every `(t, s)`."""
    return (
        t.unsqueeze(-1) * bench._b
        + bench.p(t).unsqueeze(-1) * bench._a
        + s @ c.transpose(0, 1)
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_manifold_chart_is_feasible(variant):
    """Validates the chart itself before it is used to certify the solver."""
    bench = CurvatureWarp(variant)
    c = _manifold_chart(bench)
    g = torch.Generator().manual_seed(61)
    t = 8.0 * (torch.rand(4096, generator=g, dtype=torch.float64) - 0.5)
    s = 4.0 * (torch.rand(4096, 6, generator=g, dtype=torch.float64) - 0.5)
    y = _point_on_manifold(bench, t, s, c)
    assert float(bench.g(y).abs().max()) <= _phase_atol(bench, 1e-12)
    torch.testing.assert_close(bench.t_of(y), t)


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_beats_every_random_feasible_point(variant):
    """Independent global check: 20k random on-manifold points, none beats `f*`."""
    bench = CurvatureWarp(variant)
    c = _manifold_chart(bench)
    x = bench.eval_queries(seed=5, n=64).conditions.double()
    f_star = bench.reference_solution(x).f_star
    g = torch.Generator().manual_seed(62)
    best = torch.full_like(f_star, float("inf"))
    for _ in range(20):
        t = 8.0 * (torch.rand(64, generator=g, dtype=torch.float64) - 0.5)
        s = 4.0 * (torch.rand(64, 6, generator=g, dtype=torch.float64) - 0.5)
        y = _point_on_manifold(bench, t, s, c)
        best = torch.minimum(best, bench.objective(y, x))
    assert float((f_star - best).max()) <= 1e-12


@pytest.mark.parametrize("variant", ["k1", "k4", "k7", "k10", "k11", "k13"])
def test_reference_matches_dense_1d_scan_along_the_manifold_curve(variant):
    """The closed form is the minimum of a dense scan along `t -> t*b + a*sin(omega t)`."""
    bench = CurvatureWarp(variant)
    c = _manifold_chart(bench)
    x = bench.eval_queries(seed=6, n=32).conditions.double()
    sol = bench.reference_solution(x)
    s_opt = bench.unwarp(x) @ c  # optimal transverse coordinates
    u = bench.t_of(x)
    scan = torch.linspace(-3.0, 3.0, 60001, dtype=torch.float64)
    best = torch.full_like(sol.f_star, float("inf"))
    for i in range(x.shape[0]):
        t = u[i] + scan
        y = _point_on_manifold(
            bench, t, s_opt[i].unsqueeze(0).expand(t.shape[0], -1), c
        )
        best[i] = bench.objective(y, x[i].unsqueeze(0).expand_as(y)).min()
    assert float((best - sol.f_star).abs().max()) <= 1e-8


@pytest.mark.parametrize(
    "variant", ["k1", "k3", "k5", "k7", "k9", "k10", "k11", "k12", "k13"]
)
def test_reference_matches_scipy_multistart_minimization(variant):
    """Multistart L-BFGS-B over the full 7-parameter chart matches the closed form."""
    from scipy.optimize import minimize

    bench = CurvatureWarp(variant)
    c = _manifold_chart(bench)
    x = bench.eval_queries(seed=7, n=64).conditions.double()
    sol = bench.reference_solution(x)
    gen = torch.Generator().manual_seed(63)
    starts = torch.cat(
        [
            torch.zeros(1, 7, dtype=torch.float64),
            6.0 * (torch.rand(3, 7, generator=gen, dtype=torch.float64) - 0.5),
        ]
    )

    def make_fun(xi: torch.Tensor):
        def fun(theta):
            th = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
            y = _point_on_manifold(bench, th[:1], th[1:].unsqueeze(0), c)
            f = bench.objective(y, xi.unsqueeze(0))[0]
            (grad,) = torch.autograd.grad(f, th)
            return float(f.detach()), grad.numpy()

        return fun

    worst = 0.0
    for i in range(x.shape[0]):
        fun = make_fun(x[i])
        f_num = min(
            minimize(fun, s.numpy(), jac=True, method="L-BFGS-B",
                     options={"ftol": 1e-16, "gtol": 1e-12, "maxiter": 500}).fun
            for s in starts
        )
        f_ref = float(sol.f_star[i])
        assert f_num >= f_ref - 1e-10, (
            f"{variant} q{i}: multistart beat the closed form by {f_ref - f_num:.3e}"
        )
        worst = max(worst, abs(f_num - f_ref))
    assert worst <= 1e-8, f"{variant}: worst |f_num - f*| = {worst:.3e}"


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_no_fold_ambiguity_anywhere(variant):
    """The control's headline: exactly one local minimum, so no wrong-fold risk."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=0, n=512).conditions.double()
    sol = bench.reference_solution(x)
    assert int(sol.n_folds.min()) == 1 and int(sol.n_folds.max()) == 1
    assert bool(torch.isnan(sol.delta_fold).all())
    assert bool(torch.isnan(sol.t_second).all())
    assert bool(torch.isnan(sol.fold_second).all())
    assert bool((~sol.has_two_folds).all())
    torch.testing.assert_close(sol.fold_star, bench.fold(bench.t_of(x)))
    assert bool(torch.isfinite(sol.fold_star).all())


def test_k0_reference_bookkeeping_matches_curvature_sine():
    """k0 carries no folds at all, exactly as in curvature_sine."""
    bench = CurvatureWarp("k0")
    x = bench.eval_queries(seed=1, n=256).conditions.double()
    sol = bench.reference_solution(x)
    assert bool(torch.isnan(sol.fold_star).all())
    assert bool(torch.isnan(sol.delta_fold).all())
    assert int(sol.n_folds.max()) == 0
    assert bool((~sol.has_two_folds).all())


@pytest.mark.parametrize("variant", ["k7", "k10"])
def test_curvature_sine_does_have_the_ambiguity_that_curvature_warp_removes(variant):
    """Contrast test: the paired curvature_sine arm sees competing arches on the same queries."""
    curvature_warp, curvature_sine = CurvatureWarp(variant), CurvatureSine(variant)
    x = curvature_warp.eval_queries(seed=0, n=256).conditions
    assert int(curvature_sine.reference_solution(x).has_two_folds.sum()) > 0
    assert int(curvature_warp.reference_solution(x).has_two_folds.sum()) == 0


@pytest.mark.parametrize("variant", VARIANTS)
def test_snap_is_exactly_feasible_and_preserves_t(variant):
    bench = CurvatureWarp(variant)
    y = _sample_y(bench, 256, seed=71).double()
    y_snap = bench.snap(y)
    assert float(bench.g(y_snap).abs().max()) <= 1e-14 * (1.0 + bench.omega)
    torch.testing.assert_close(bench.t_of(y_snap), bench.t_of(y))
    torch.testing.assert_close(bench.fold(y_snap @ bench._b), bench.fold(y @ bench._b))


@pytest.mark.parametrize("variant", VARIANTS)
def test_snap_gap_is_nonnegative(variant):
    """The snap diagnostic can never beat the exact projection."""
    bench = CurvatureWarp(variant)
    y = _sample_y(bench, 512, seed=72).double()
    x = bench.eval_queries(seed=0, n=512).conditions.double()
    gap_snap = bench.objective(bench.snap(y), x) - bench.reference_solution(x).f_star
    assert float(gap_snap.min()) >= -1e-12


@pytest.mark.parametrize("variant", SHARED_VARIANTS)
def test_geometric_reference_is_curvature_sine_certified_euclidean_projection(variant):
    """The analyzer's `d` must be on curvature_sine's scale, not the warped one."""
    curvature_warp, curvature_sine = CurvatureWarp(variant), CurvatureSine(variant)
    assert curvature_warp._euclid is not None
    y = _sample_y(curvature_warp, 128, seed=73).double()
    geo, ref32 = curvature_warp.geometric_reference(y), curvature_sine.reference_solution(y)
    assert torch.equal(geo.f_star, ref32.f_star)
    assert torch.equal(geo.y_star, ref32.y_star)
    assert torch.equal(geo.t_star, ref32.t_star)
    torch.testing.assert_close(
        geo.f_star.sqrt(), (geo.y_star - y).norm(dim=-1), rtol=1e-9, atol=1e-11
    )


@pytest.mark.parametrize("variant", EXTENDED_VARIANTS)
def test_geometric_reference_runs_sine_solver_where_sine_has_no_variant(variant):
    """k11 ... k13: curvature_sine's solver at this omega gives a feasible Euclidean projection."""
    bench = CurvatureWarp(variant)
    assert bench._euclid is None
    y = _sample_y(bench, 128, seed=73).double()
    geo = bench.geometric_reference(y)
    assert float(bench.g(geo.y_star).abs().max()) <= _phase_atol(bench, 1e-12)
    torch.testing.assert_close(
        geo.f_star.sqrt(), (geo.y_star - y).norm(dim=-1), rtol=1e-9, atol=1e-11
    )
    # the snap along `a` is feasible, so the projection can never be worse
    assert bool((geo.f_star <= bench.g(y) ** 2 + 1e-12).all())
    assert bool(torch.isfinite(geo.t_star).all())


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_geometric_and_warped_references_disagree_off_the_anchor(variant):
    """They answer different questions; conflating them would misreport `d`."""
    bench = CurvatureWarp(variant)
    x = bench.eval_queries(seed=0, n=256).conditions.double()
    warped = bench.reference_solution(x).f_star
    euclid = bench.geometric_reference(x).f_star
    assert bool((euclid <= warped + 1e-12).all())  # Euclidean projection is closer
    assert float((warped - euclid).max()) > 1e-3


def _load_analyzer():
    name = "analyze_curvature_for_test"
    if name in sys.modules:
        return sys.modules[name]
    path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_curvature.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    # register before exec: dataclass field resolution looks the module up in sys.modules
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_analyzer_dispatches_all_three_dial_families():
    """One runs root holding all three curvature dial families must classify cleanly."""
    mod = _load_analyzer()
    assert mod._family_of("curvature_hinge_k4") == "curvature_hinge"
    assert mod._family_of("curvature_sine_k4") == "curvature_sine"
    assert mod._family_of("curvature_warp_k4") == "curvature_warp"
    assert mod._family_of("rosenbrock_eq") is None
    families = {"curvature_hinge", "curvature_sine", "curvature_warp"}
    assert set(mod.PREFIX_BY_FAMILY) == set(mod.KAPPA_BY_FAMILY) == families
    for family, grid in mod.KAPPA_BY_FAMILY.items():
        n_points = len(VARIANTS) if family == "curvature_warp" else 11
        assert [f"k{i}" for i in range(n_points)] == list(grid), family
    # curvature_warp carries curvature_sine's dial plus its own k11 ... k13 extension
    sine_grid = mod.KAPPA_BY_FAMILY["curvature_sine"]
    warp_grid = mod.KAPPA_BY_FAMILY["curvature_warp"]
    assert {v: k for v, k in warp_grid.items() if v in sine_grid} == sine_grid
    assert [v for v in warp_grid if v not in sine_grid] == EXTENDED_VARIANTS


def test_analyzer_renders_curvature_warp_tables_with_the_curvature_sine_column_layout():
    """Same columns as curvature_sine (side-by-side reading) plus the control's notes."""
    mod = _load_analyzer()
    row = {"variant": "k4", "kappa": 1.0, "n": 8, "feas": 1.0, "n_feas": 8,
           "gap_p50": 0.0, "gap_p90": 0.0, "snap_p50": 0.0, "snap_p90": 0.0,
           "snap_min": 0.0, "obj_check": 0.0, "affine_anchor": False,
           "omega": 1.0, "d_p50": 0.0, "d_p90": 0.0, "kd_p50": 0.0,
           "kd_p90": 0.0, "kd_p50_signed": 0.0, "kd_p90_signed": 0.0,
           "wrong_fold": float("nan"), "n_elig": 0, "n_ambig": 0,
           "n_via_snap": 0, "n_cap": 0}
    md33 = mod._render([dict(row)], "curvature_warp", 1e-4, 1e-2)
    md32 = mod._render([dict(row)], "curvature_sine", 1e-4, 1e-2)
    assert "### curvature_warp core" in md33 and "### curvature_warp strata bin counts" in md33
    assert "n_elig = 0" in md33 and "BY" in md33  # the degeneracy is explained
    heads33 = [ln for ln in md33.splitlines() if ln.startswith("| variant")]
    heads32 = [ln for ln in md32.splitlines() if ln.startswith("| variant")]
    assert heads33 == heads32
    assert "curvature_warp" not in md32  # curvature_sine rendering is untouched
