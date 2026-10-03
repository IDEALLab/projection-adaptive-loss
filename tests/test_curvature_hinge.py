"""Sanity and certification tests for the curvature_hinge curvature dial (hinge)."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks import get as get_benchmark
from pal.benchmarks import list_all as list_all_benchmarks
from pal.benchmarks.synthetic.curvature_hinge import (
    KAPPA_BY_VARIANT,
    R_BY_VARIANT,
    VARIANTS,
    CurvatureHinge,
    _f_1d,
    curvature_k,
    p_r,
    p_r_prime,
    p_r_second,
)

BENCH_IDS = [f"curvature_hinge_{v}" for v in VARIANTS]
SQRT2 = math.sqrt(2.0)


def test_all_eleven_variants_registered():
    assert VARIANTS == [f"k{i}" for i in range(11)]
    available = set(list_all_benchmarks())
    missing = [b for b in BENCH_IDS if b not in available]
    assert not missing, f"unregistered curvature_hinge variants: {missing}"


@pytest.mark.parametrize("bench_id", BENCH_IDS)
def test_registry_returns_matching_variant(bench_id):
    bench = get_benchmark(bench_id)
    assert bench.spec.id == bench_id
    assert bench.spec.family == "curvature_hinge"
    assert bench.spec.variant == bench_id.removeprefix("curvature_hinge_")
    assert bench.kappa == KAPPA_BY_VARIANT[bench.spec.variant]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="unknown curvature_hinge variant"):
        CurvatureHinge(variant="k99")


def test_kappa_grid_and_radius_match_plan():
    assert [KAPPA_BY_VARIANT[v] for v in VARIANTS] == pytest.approx(
        [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    )
    assert R_BY_VARIANT["k0"] == 0.0  # sentinel: affine anchor, no sqrt term
    for v in VARIANTS[1:]:
        assert R_BY_VARIANT[v] == pytest.approx(1.0 / (2.0 * KAPPA_BY_VARIANT[v]))
        assert CurvatureHinge(v).is_affine_anchor is False
    assert CurvatureHinge("k0").is_affine_anchor is True


@pytest.mark.parametrize("variant", VARIANTS)
def test_spec_shape(variant):
    spec = CurvatureHinge(variant).spec
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


@pytest.mark.parametrize("variant", VARIANTS)
def test_sample_eval_shapes(variant):
    bench = CurvatureHinge(variant)
    for seed in (0, 1, 2):
        q = bench.sample_queries(16, "train", seed=seed)
        assert q.zeta.shape == (16, 0)
        assert q.conditions.shape == (16, 8)
        assert q.conditions.min() >= -1.5 and q.conditions.max() <= 1.5
        assert bench.eval_queries(seed=seed, n=8).conditions.shape == (8, 8)
    assert bench.eval_queries(seed=0).conditions.shape == (512, 8)


def test_query_sets_and_directions_identical_across_variants():
    """Paired randomness: kappa is the only thing that changes across the sweep."""
    ref = CurvatureHinge("k0")
    ref_train = ref.sample_queries(64, "train", seed=5).conditions
    ref_eval = ref.eval_queries(seed=5, n=64).conditions
    for variant in VARIANTS[1:]:
        bench = CurvatureHinge(variant)
        assert torch.equal(
            bench.sample_queries(64, "train", seed=5).conditions, ref_train
        )
        assert torch.equal(bench.eval_queries(seed=5, n=64).conditions, ref_eval)
        assert torch.equal(bench._a, ref._a)
        assert torch.equal(bench._b, ref._b)


def test_directions_are_orthonormal():
    bench = CurvatureHinge("k4")
    a, b = bench._a, bench._b
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert float(a.norm()) == pytest.approx(1.0, abs=1e-15)
    assert float(b.norm()) == pytest.approx(1.0, abs=1e-15)
    assert abs(float(torch.dot(a, b))) <= 1e-15


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_hinge_is_centered_with_slopes_zero_to_one(variant):
    r = R_BY_VARIANT[variant]
    zero = torch.zeros(1, dtype=torch.float64)
    assert float(p_r(zero, r)) == 0.0
    assert float(p_r_prime(zero, r)) == pytest.approx(0.5, abs=1e-15)
    # max p'' = kappa, attained at t = 0
    t = torch.linspace(-40 * r, 40 * r, 20001, dtype=torch.float64)
    d2 = p_r_second(t, r)
    assert float(p_r_second(zero, r)) == pytest.approx(KAPPA_BY_VARIANT[variant])
    assert float(d2.max()) <= KAPPA_BY_VARIANT[variant] * (1 + 1e-12)
    # asymptotic slopes 0 (left arm) and 1 (right arm)
    far = torch.tensor([-1e6 * r, 1e6 * r], dtype=torch.float64)
    d1 = p_r_prime(far, r)
    assert float(d1[0]) < 1e-11
    assert float(d1[1]) > 1 - 1e-11
    assert bool(((p_r_prime(t, r) > 0) & (p_r_prime(t, r) < 1)).all())


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_max_curvature_is_0754_kappa_at_minus_0174_r(variant):
    """K_max ~ 0.754*kappa at t ~ -0.174*r, kappa itself is a scale, not a curvature."""
    bench = CurvatureHinge(variant)
    r = bench.r
    t = torch.linspace(-3 * r, 3 * r, 2000001, dtype=torch.float64)
    k = curvature_k(t, r)
    i = int(k.argmax())
    assert float(k[i]) / bench.kappa == pytest.approx(0.754, abs=2e-3)
    assert float(t[i]) / r == pytest.approx(-0.174, abs=2e-3)


def test_k0_anchor_is_exactly_affine():
    """k0: p(t) = t/2, p'' = 0, K = 0, constraint Hessian identically zero."""
    bench = CurvatureHinge("k0")
    t = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    torch.testing.assert_close(bench.p(t), 0.5 * t)
    torch.testing.assert_close(bench.p_prime(t), torch.full_like(t, 0.5))
    assert float(bench.p_second(t).abs().max()) == 0.0
    assert float(bench.curvature(t).abs().max()) == 0.0

    y = _sample_y(bench, 1, seed=15).double().requires_grad_(True)
    H = torch.autograd.functional.hessian(
        lambda z: bench.constraints(z, None).sum(), y
    )
    assert float(H.abs().max()) == pytest.approx(0.0, abs=1e-14)


def test_negative_arm_is_cancellation_free_at_k10():
    """Large |t|/r on the flat arm: p -> -r/2 + r^2/(4|t|), p' -> r^2/(4t^2)."""
    r = R_BY_VARIANT["k10"]
    t = -r * torch.tensor([1e2, 1e3, 1e4, 1e5, 1e6], dtype=torch.float64)
    excess = p_r(t, r) + 0.5 * r
    expected = r * r / (4.0 * t.abs())
    torch.testing.assert_close(excess, expected, rtol=1e-3, atol=0.0)
    torch.testing.assert_close(
        p_r_prime(t, r), r * r / (4.0 * t * t), rtol=1e-3, atol=0.0
    )
    assert bool((excess > 0).all())  # p stays strictly above its asymptote


def test_positive_arm_is_cancellation_free_at_k10():
    """Right arm: p -> t - r/2 + r^2/(4t), slope -> 1 from below."""
    r = R_BY_VARIANT["k10"]
    t = r * torch.tensor([1e2, 1e3, 1e4, 1e5, 1e6], dtype=torch.float64)
    excess = p_r(t, r) - (t - 0.5 * r)
    torch.testing.assert_close(excess, r * r / (4.0 * t), rtol=1e-3, atol=0.0)
    assert bool((p_r_prime(t, r) < 1.0).all())


def test_near_zero_series_both_arms_at_k10():
    """|t| << r: p ~ t/2 + t^2/(4r), p' ~ 1/2 + t/(2r) on both arms."""
    r = R_BY_VARIANT["k10"]
    t = r * torch.tensor(
        [-1e-2, -1e-3, -1e-4, 0.0, 1e-4, 1e-3, 1e-2], dtype=torch.float64
    )
    series = 0.5 * t + t * t / (4.0 * r)
    torch.testing.assert_close(p_r(t, r), series, rtol=0.0, atol=1e-6 * r)
    torch.testing.assert_close(
        p_r_prime(t, r), 0.5 + t / (2.0 * r), rtol=0.0, atol=1e-6
    )


def test_float32_training_path_keeps_stable_form_at_k10():
    """The float32 constraint path must use the rationalized negative arm."""
    r = R_BY_VARIANT["k10"]
    t32 = torch.tensor([-1.0, -10.0], dtype=torch.float32)
    stable32 = p_r_prime(t32, r)
    exact = p_r_prime(t32.double(), r)
    rel = ((stable32.double() - exact) / exact).abs().max()
    assert float(rel) < 1e-6, f"float32 stable p' rel err {float(rel):.2e}"
    naive = 0.5 * (1.0 + t32 / torch.sqrt(t32 * t32 + r * r))
    naive_rel = ((naive.double() - exact) / exact).abs()
    assert float(naive_rel.min()) > 1e-2, "naive float32 form is expected to die"
    assert float(naive[1]) == 0.0  # t = -10: no significant digits survive
    # p itself: float32 stable form stays far inside the 1e-4 feasibility tol
    assert float((p_r(t32, r).double() - p_r(t32.double(), r)).abs().max()) < 1e-8


def _sample_y(bench, n: int, seed: int) -> torch.Tensor:
    lo, hi = bench.spec.output_bounds
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, bench.spec.dim, generator=g)


@pytest.mark.parametrize("variant", VARIANTS)
def test_constraint_is_raw_g(variant):
    """Exposed value is exactly `a^T y - p_r(b^T y)`, no ||grad g|| normalization."""
    bench = CurvatureHinge(variant)
    y = _sample_y(bench, 64, seed=11).double()
    q = bench.sample_queries(64, "train", seed=11)
    expected = y @ bench._a - p_r(y @ bench._b, bench.r)

    got = bench.constraints(y, q.conditions.double())
    assert got.shape == (64, 1)
    torch.testing.assert_close(got[:, 0], expected)
    torch.testing.assert_close(bench.g(y), expected)
    _, cons = bench.forward(y, q.conditions.double())
    assert len(cons) == 1 and cons[0].type == "eq" and cons[0].name == "smoothed_hinge"
    torch.testing.assert_close(cons[0].value, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_grad_g_matches_autograd(variant):
    bench = CurvatureHinge(variant)
    y = _sample_y(bench, 16, seed=12).double().requires_grad_(True)
    g_val = bench.constraints(y, None)[:, 0]
    (g_auto,) = torch.autograd.grad(g_val.sum(), y)
    torch.testing.assert_close(bench.grad_g(y.detach()), g_auto)
    assert bool(torch.isfinite(g_auto).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_jacobian_matches_finite_differences(variant):
    bench = CurvatureHinge(variant)
    y = _sample_y(bench, 4, seed=13).double()
    # central differences in float64: eps ~ (ulp/|p'''|)^(1/3) balances
    # roundoff against truncation even at kappa = 1000 (|p'''| ~ 0.65/r^2)
    eps = 5e-8
    J_fd = torch.zeros(4, bench.spec.dim, dtype=torch.float64)
    for j in range(bench.spec.dim):
        e = torch.zeros_like(y)
        e[:, j] = eps
        J_fd[:, j] = (
            bench.constraints(y + e, None)[:, 0] - bench.constraints(y - e, None)[:, 0]
        ) / (2 * eps)
    torch.testing.assert_close(bench.grad_g(y), J_fd, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_grad_norm_in_one_to_sqrt2_band(variant):
    """Slope <= 1 => ||grad g|| = sqrt(1+p'^2) in [1, sqrt(2)]: no vanishing gradient, no growth."""
    bench = CurvatureHinge(variant)
    y = _sample_y(bench, 4096, seed=14).double()
    norms = bench.grad_g(y).norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-12
    assert float(norms.max()) <= SQRT2 + 1e-12


def test_grad_norm_dense_through_the_corner_at_k10():
    """Sweep b^T y densely across the k10 corner (scale r = 5e-4)."""
    bench = CurvatureHinge("k10")
    b = bench._b
    u = torch.cat(
        [
            torch.linspace(-40 * bench.r, 40 * bench.r, 40001, dtype=torch.float64),
            torch.linspace(-4.0, 4.0, 40001, dtype=torch.float64),
        ]
    )
    y = u.unsqueeze(-1) * b.unsqueeze(0)
    norms = bench.grad_g(y).norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-12
    assert float(norms.max()) <= SQRT2 + 1e-12
    d1 = bench.p_prime(u)
    assert float(d1.min()) < 1e-6 and float(d1.max()) > 1 - 1e-6


@pytest.mark.parametrize("variant", VARIANTS)
def test_objective_is_squared_distance_to_condition(variant):
    bench = CurvatureHinge(variant)
    q = bench.sample_queries(64, "train", seed=17)
    y = _sample_y(bench, 64, seed=17)
    obj = bench.objective(y, q.conditions)
    torch.testing.assert_close(obj, ((y - q.conditions) ** 2).sum(dim=-1))
    assert bool(torch.isfinite(obj).all()) and bool((obj >= 0).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_condition_box_straddles_manifold(variant):
    """Straddle guard: sign(g(x)) varies over the box at every variant."""
    bench = CurvatureHinge(variant)
    q = bench.sample_queries(8192, "train", seed=18)
    g = bench.g(q.conditions)
    frac_pos = float((g > 0).double().mean())
    assert 0.3 < frac_pos < 0.7, f"{variant}: box does not straddle g=0 ({frac_pos=})"
    dist = float((g.abs() / bench.grad_g(q.conditions).norm(dim=-1)).mean())
    assert 0.1 < dist < 10.0, f"{variant}: mean distance {dist:.3f} not O(1)"


def _brute_force_min(u: torch.Tensor, v: torch.Tensor, r: float, n: int = 200_001):
    """Dense envelope of `min F` over `[u-m, u+m]`, which holds a minimizer as F >= (t-u)^2."""
    m = (p_r(u, r) - v).abs()
    lo = u - m * 1.0000001 - 1e-12
    hi = u + m * 1.0000001 + 1e-12
    out = torch.empty_like(u)
    for i in range(u.shape[0]):
        t = torch.linspace(float(lo[i]), float(hi[i]), n, dtype=torch.float64)
        if r > 0.0:
            span = max(float(m[i]) + abs(float(u[i])), 10.0 * r)
            mags = torch.pow(
                torch.tensor(10.0, dtype=torch.float64),
                torch.linspace(math.log10(r) - 4.0, math.log10(span), 40_000),
            )
            overlay = torch.cat(
                [-mags.flip(0), torch.zeros(1, dtype=torch.float64), mags]
            ).clamp(min=float(lo[i]), max=float(hi[i]))
            t = torch.cat([t, overlay])
        out[i] = _f_1d(t, u[i], v[i], r).min()
    return out


def _assert_not_worse_than_envelope(bench, x, label: str):
    x = x.double()
    sol = bench.reference_solution(x)
    brute = _brute_force_min(x @ bench._b, x @ bench._a, bench.r)
    slack = sol.f_star - brute
    tol = 1e-12 * (1.0 + brute)
    assert bool((slack <= tol).all()), (
        f"{label} / {bench.variant}: solver worse than a 240k-point envelope by "
        f"{float((slack - tol).max()):.3e}"
    )
    assert bool(sol.from_stationary.all()), (
        f"{label} / {bench.variant}: winner was not a refined stationary point"
    )
    return sol


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_solution_is_feasible_and_consistent(variant):
    bench = CurvatureHinge(variant)
    x = bench.eval_queries(seed=0, n=128).conditions
    sol = bench.reference_solution(x)
    assert float(bench.g(sol.y_star).abs().max()) <= 1e-12
    torch.testing.assert_close(
        sol.f_star, ((sol.y_star - x.double()) ** 2).sum(dim=-1)
    )
    torch.testing.assert_close(sol.t_star, bench.t_of(sol.y_star))
    assert bool((sol.f_star >= 0).all())
    torch.testing.assert_close(bench.y_star(x), sol.y_star)
    lo, hi = bench.spec.output_bounds
    inside = (sol.y_star >= lo.double()) & (sol.y_star <= hi.double())
    assert bool(inside.all()), f"{variant}: reference projection escapes output_bounds"


def test_reference_k0_uses_the_closed_form_plane_projection():
    """k0: exact projection onto `a^T y = (b^T y)/2`; no arms, no Delta_arm."""
    bench = CurvatureHinge("k0")
    x = bench.eval_queries(seed=1, n=256).conditions.double()
    a, b = bench._a, bench._b
    n = a - 0.5 * b
    gx = x @ a - 0.5 * (x @ b)
    y_expect = x - (gx / n.dot(n)).unsqueeze(-1) * n
    sol = bench.reference_solution(x)
    torch.testing.assert_close(sol.y_star, y_expect)
    torch.testing.assert_close(sol.f_star, gx * gx / n.dot(n))
    assert bool(torch.isnan(sol.delta_arm).all())
    assert bool(torch.isnan(sol.t_second).all())
    assert bool((~sol.has_two_arms).all())


@pytest.mark.parametrize("variant", ["k1", "k4", "k7", "k10"])
def test_reference_matches_brute_force_envelope(variant):
    """Randomized batch vs a much denser independent scan of F."""
    bench = CurvatureHinge(variant)
    x = bench.eval_queries(seed=3, n=24).conditions
    sol = _assert_not_worse_than_envelope(bench, x, "random")
    brute = _brute_force_min(x.double() @ bench._b, x.double() @ bench._a, bench.r)
    assert float((brute - sol.f_star).max()) < 1e-8


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_reference_stress_near_medial_axis(variant):
    """Queries on the wedge's medial axis `u = (1 - sqrt(2))*v` (near-tied arms)."""
    bench = CurvatureHinge(variant)
    v = torch.linspace(0.05, 2.0, 32, dtype=torch.float64)
    jitter = torch.linspace(-0.02, 0.02, 32, dtype=torch.float64)
    u = (1.0 - math.sqrt(2.0)) * v + jitter * v
    x = u.unsqueeze(-1) * bench._b + v.unsqueeze(-1) * bench._a
    sol = _assert_not_worse_than_envelope(bench, x, "medial")
    if variant in ("k7", "k10"):
        assert int(sol.has_two_arms.sum()) >= 24
        arm_delta = sol.delta_arm[sol.has_two_arms]
        assert bool((arm_delta >= 0).all())
        assert float(arm_delta.min()) < 1e-3  # near-tie really is near


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_reference_stress_near_evolute(variant):
    """Queries at the centers of curvature (double-root / tangency regime)."""
    bench = CurvatureHinge(variant)
    r = bench.r
    t = r * torch.tensor(
        [-30.0, -10.0, -3.0, -1.0, -0.5, -0.174, 0.0, 0.174, 0.5, 1.0, 3.0, 10.0, 30.0],
        dtype=torch.float64,
    )
    foot = t.unsqueeze(-1) * bench._b + p_r(t, r).unsqueeze(-1) * bench._a
    normal = bench.grad_g(foot)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    radius = 1.0 / curvature_k(t, r)
    for s in (0.5, 0.95, 1.0, 1.05, 2.0):
        x = foot + (s * radius).unsqueeze(-1) * normal
        _assert_not_worse_than_envelope(bench, x, f"evolute s={s}")


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_delta_arm_is_nonnegative_and_costs_the_other_arm(variant):
    bench = CurvatureHinge(variant)
    x = bench.eval_queries(seed=0, n=256).conditions
    sol = bench.reference_solution(x)
    two = sol.has_two_arms
    assert bool((sol.delta_arm[two] >= 0).all())
    if bool(two.any()):
        # the runner-up is a different foot with objective exactly f* + Delta_arm
        t2 = sol.t_second[two]
        u = (x.double() @ bench._b)[two]
        v = (x.double() @ bench._a)[two]
        torch.testing.assert_close(
            _f_1d(t2, u, v, bench.r), sol.f_star[two] + sol.delta_arm[two]
        )
        assert bool(((t2 - sol.t_star[two]).abs() > 1e-9).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_snap_is_exactly_feasible_and_preserves_t(variant):
    """`y_snap = y - g(y)*a` is arm- and t-preserving and exactly on `{g=0}`."""
    bench = CurvatureHinge(variant)
    y = _sample_y(bench, 256, seed=21).double()
    y_snap = bench.snap(y)
    assert float(bench.g(y_snap).abs().max()) <= 1e-14
    torch.testing.assert_close(bench.t_of(y_snap), bench.t_of(y))
    x = bench.sample_queries(256, "train", seed=21).conditions.double()
    f_snap = ((y_snap - x) ** 2).sum(dim=-1)
    f_star = bench.reference_solution(x).f_star
    assert float((f_star - f_snap).max()) <= 1e-12
