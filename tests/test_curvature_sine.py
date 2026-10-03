"""Sanity and certification tests for the curvature_sine oscillation dial (sine ripple)."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks import get as get_benchmark
from pal.benchmarks import list_all as list_all_benchmarks
from pal.benchmarks.synthetic.curvature_hinge import CurvatureHinge
from pal.benchmarks.synthetic.curvature_sine import (
    KAPPA_BY_VARIANT,
    OMEGA_BY_VARIANT,
    VARIANTS,
    CurvatureSine,
    _f_1d,
    curvature_k,
    fold_index,
    p_sine,
    p_sine_prime,
    p_sine_second,
)

BENCH_IDS = [f"curvature_sine_{v}" for v in VARIANTS]


def test_all_eleven_variants_registered():
    assert VARIANTS == [f"k{i}" for i in range(11)]
    available = set(list_all_benchmarks())
    missing = [b for b in BENCH_IDS if b not in available]
    assert not missing, f"unregistered curvature_sine variants: {missing}"


@pytest.mark.parametrize("bench_id", BENCH_IDS)
def test_registry_returns_matching_variant(bench_id):
    bench = get_benchmark(bench_id)
    assert bench.spec.id == bench_id
    assert bench.spec.family == "curvature_sine"
    assert bench.spec.variant == bench_id.removeprefix("curvature_sine_")
    assert bench.kappa == KAPPA_BY_VARIANT[bench.spec.variant]


def test_unknown_variant_rejected():
    with pytest.raises(ValueError, match="unknown curvature_sine variant"):
        CurvatureSine(variant="k99")


def test_kappa_grid_and_omega_match_plan():
    assert [KAPPA_BY_VARIANT[v] for v in VARIANTS] == pytest.approx(
        [0.0, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0]
    )
    assert OMEGA_BY_VARIANT["k0"] == 0.0  # sentinel: linear anchor, sine dropped
    for v in VARIANTS[1:]:
        omega = OMEGA_BY_VARIANT[v]
        assert omega == pytest.approx(math.sqrt(KAPPA_BY_VARIANT[v]))
        assert omega * omega == pytest.approx(KAPPA_BY_VARIANT[v])  # kappa = omega^2
        assert CurvatureSine(v).is_affine_anchor is False
    assert OMEGA_BY_VARIANT["k10"] == pytest.approx(31.6227766, abs=1e-6)
    assert CurvatureSine("k0").is_affine_anchor is True


@pytest.mark.parametrize("variant", VARIANTS)
def test_spec_shape(variant):
    spec = CurvatureSine(variant).spec
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
    bench = CurvatureSine(variant)
    for seed in (0, 1, 2):
        q = bench.sample_queries(16, "train", seed=seed)
        assert q.zeta.shape == (16, 0)
        assert q.conditions.shape == (16, 8)
        assert q.conditions.min() >= -1.5 and q.conditions.max() <= 1.5
        assert bench.eval_queries(seed=seed, n=8).conditions.shape == (8, 8)
    assert bench.eval_queries(seed=0).conditions.shape == (512, 8)


def test_query_sets_and_directions_identical_across_variants():
    """Paired randomness: kappa is the only thing that changes across the sweep."""
    ref = CurvatureSine("k0")
    ref_train = ref.sample_queries(64, "train", seed=5).conditions
    ref_eval = ref.eval_queries(seed=5, n=64).conditions
    for variant in VARIANTS[1:]:
        bench = CurvatureSine(variant)
        assert torch.equal(
            bench.sample_queries(64, "train", seed=5).conditions, ref_train
        )
        assert torch.equal(bench.eval_queries(seed=5, n=64).conditions, ref_eval)
        assert torch.equal(bench._a, ref._a)
        assert torch.equal(bench._b, ref._b)


def test_directions_are_orthonormal():
    bench = CurvatureSine("k4")
    a, b = bench._a, bench._b
    assert a.dtype == torch.float64 and b.dtype == torch.float64
    assert float(a.norm()) == pytest.approx(1.0, abs=1e-15)
    assert float(b.norm()) == pytest.approx(1.0, abs=1e-15)
    assert abs(float(torch.dot(a, b))) <= 1e-15


def test_geometry_and_queries_shared_with_curvature_hinge_family():
    """Same (a, b) and same query sets as curvature_hinge: the hinge is a true control."""
    curvature_sine, curvature_hinge = CurvatureSine("k4"), CurvatureHinge("k4")
    assert torch.equal(curvature_sine._a, curvature_hinge._a)
    assert torch.equal(curvature_sine._b, curvature_hinge._b)
    assert torch.equal(
        curvature_sine.eval_queries(seed=0, n=64).conditions,
        curvature_hinge.eval_queries(seed=0, n=64).conditions,
    )
    assert torch.equal(
        curvature_sine.sample_queries(64, "train", seed=0).conditions,
        curvature_hinge.sample_queries(64, "train", seed=0).conditions,
    )


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_kappa_is_exactly_the_max_curvature(variant):
    """`K(t) = kappa` at every crest and `K <= kappa` everywhere, no fudge constant."""
    bench = CurvatureSine(variant)
    omega, kappa = bench.omega, bench.kappa
    crests = (
        (0.5 + torch.arange(-3, 4, dtype=torch.float64)) * math.pi / omega
    )  # sin(omega t) = +/-1
    k_crest = curvature_k(crests, omega)
    torch.testing.assert_close(
        k_crest, torch.full_like(k_crest, kappa), rtol=1e-12, atol=0.0
    )
    torch.testing.assert_close(
        p_sine(crests, omega).abs(), torch.ones_like(crests), rtol=0.0, atol=1e-12
    )
    t = torch.linspace(-4.0, 4.0, 400001, dtype=torch.float64)
    assert float(curvature_k(t, omega).max()) <= kappa * (1 + 1e-12)
    # |p''| max = kappa too (kappa = omega^2 = max |p''| AND max K)
    assert float(p_sine_second(t, omega).abs().max()) <= kappa * (1 + 1e-12)


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_curvature_vanishes_at_midline_crossings(variant):
    """Fold boundaries `omega t = k pi` are inflections: `K = 0`, `|p'| = omega` (max slope)."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    mid = torch.arange(-3, 4, dtype=torch.float64) * math.pi / omega
    assert float(curvature_k(mid, omega).abs().max()) < 1e-12 * bench.kappa + 1e-15
    torch.testing.assert_close(
        p_sine_prime(mid, omega).abs(), torch.full_like(mid, omega), rtol=1e-12, atol=0.0
    )


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_fold_index_is_one_arch_per_half_period(variant):
    """`fold = floor(omega t/pi)`: consecutive integers, one extremum of `p` per fold."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    for k in range(-3, 4):
        lo = k * math.pi / omega
        hi = (k + 1) * math.pi / omega
        t = torch.linspace(lo + 1e-9, hi - 1e-9, 1001, dtype=torch.float64)
        folds = fold_index(t, omega)
        assert bool((folds == k).all()), f"fold {k} mislabeled"
        # exactly one extremum (|p| = 1) strictly inside, at the mid point
        crest = 0.5 * (lo + hi)
        assert float(p_sine(torch.tensor([crest], dtype=torch.float64), omega).abs()) == (
            pytest.approx(1.0, abs=1e-12)
        )
        assert float(fold_index(torch.tensor([crest], dtype=torch.float64), omega)) == k
    assert math.isnan(float(bench.fold(torch.tensor([float("nan")]))))


def test_k0_anchor_is_exactly_linear():
    """k0: omega = 0 => p == 0, K == 0, constraint Hessian identically zero."""
    bench = CurvatureSine("k0")
    t = torch.linspace(-5.0, 5.0, 101, dtype=torch.float64)
    assert float(bench.p(t).abs().max()) == 0.0
    assert float(bench.p_prime(t).abs().max()) == 0.0
    assert float(bench.p_second(t).abs().max()) == 0.0
    assert float(bench.curvature(t).abs().max()) == 0.0

    y = _sample_y(bench, 1, seed=15).double().requires_grad_(True)
    H = torch.autograd.functional.hessian(
        lambda z: bench.constraints(z, None).sum(), y
    )
    assert float(H.abs().max()) == pytest.approx(0.0, abs=1e-14)
    # g = a.y exactly (sine dropped, not merely small)
    yd = _sample_y(bench, 32, seed=16).double()
    torch.testing.assert_close(bench.g(yd), yd @ bench._a)


def test_sine_identities_hold_at_large_t_at_k10():
    """Benign numerics: `p^2 + (p'/omega)^2 = 1` and `p'' = -omega^2 p` at |t| >> 1."""
    omega = OMEGA_BY_VARIANT["k10"]
    t = torch.cat(
        [
            torch.linspace(-1e6, 1e6, 20001, dtype=torch.float64),
            torch.tensor([-1e6, -1e3, -1.0, 0.0, 1.0, 1e3, 1e6], dtype=torch.float64),
        ]
    )
    p, d1, d2 = p_sine(t, omega), p_sine_prime(t, omega), p_sine_second(t, omega)
    one = p * p + (d1 / omega) ** 2
    torch.testing.assert_close(one, torch.ones_like(one), rtol=0.0, atol=1e-14)
    torch.testing.assert_close(d2, -(omega**2) * p, rtol=1e-14, atol=1e-12)
    assert float(p.abs().max()) <= 1.0
    assert float(d1.abs().max()) <= omega * (1 + 1e-15)
    k = curvature_k(t, omega)
    assert float(k.min()) >= 0.0 and float(k.max()) <= 1000.0 * (1 + 1e-12)


def test_float32_training_path_stays_far_inside_the_tolerance_at_k10():
    """float32 costs an O(omega*|t|*eps) phase error, 1e-5 at worst against a 1e-3 tol."""
    bench = CurvatureSine("k10")
    y32 = _sample_y(bench, 4096, seed=19)
    g32 = bench.constraints(y32, None)[:, 0]
    g64 = bench.constraints(y32.double(), None)[:, 0]
    err = float((g32.double() - g64).abs().max())
    assert err < 1e-4, f"float32 g deviation {err:.2e}"
    assert err < 1e-3  # the exposed Constraint tolerance


def _sample_y(bench, n: int, seed: int) -> torch.Tensor:
    lo, hi = bench.spec.output_bounds
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(n, bench.spec.dim, generator=g)


@pytest.mark.parametrize("variant", VARIANTS)
def test_constraint_is_raw_g(variant):
    """Exposed value is exactly `a^T y - sin(omega*b^T y)`, no ||grad g|| normalization."""
    bench = CurvatureSine(variant)
    y = _sample_y(bench, 64, seed=11).double()
    q = bench.sample_queries(64, "train", seed=11)
    expected = y @ bench._a - p_sine(y @ bench._b, bench.omega)

    got = bench.constraints(y, q.conditions.double())
    assert got.shape == (64, 1)
    torch.testing.assert_close(got[:, 0], expected)
    torch.testing.assert_close(bench.g(y), expected)
    _, cons = bench.forward(y, q.conditions.double())
    assert len(cons) == 1 and cons[0].type == "eq" and cons[0].name == "sine_ripple"
    torch.testing.assert_close(cons[0].value, expected)


@pytest.mark.parametrize("variant", VARIANTS)
def test_grad_g_matches_autograd(variant):
    bench = CurvatureSine(variant)
    y = _sample_y(bench, 16, seed=12).double().requires_grad_(True)
    g_val = bench.constraints(y, None)[:, 0]
    (g_auto,) = torch.autograd.grad(g_val.sum(), y)
    torch.testing.assert_close(bench.grad_g(y.detach()), g_auto)
    assert bool(torch.isfinite(g_auto).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_jacobian_matches_finite_differences(variant):
    bench = CurvatureSine(variant)
    y = _sample_y(bench, 4, seed=13).double()
    # central differences in float64: eps ~ (ulp/|p'''|)^(1/3) balances roundoff
    # against truncation even at kappa = 1000 (|p'''| = omega^3 ~ 3.2e4)
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
def test_grad_norm_in_one_to_sqrt_one_plus_kappa_band(variant):
    """`||grad g|| = sqrt(1 + omega^2 cos^2(omega t)) in [1, sqrt(1+kappa)]`, the stiffness band."""
    bench = CurvatureSine(variant)
    hi = math.sqrt(1.0 + bench.kappa)
    y = _sample_y(bench, 4096, seed=14).double()
    norms = bench.grad_g(y).norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-12
    assert float(norms.max()) <= hi + 1e-12


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_grad_norm_hits_both_ends_of_the_band(variant):
    """Equality checks: `||grad g|| = 1` exactly at crests, `sqrt(1+kappa)` at the midline."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    b = bench._b
    crests = (0.5 + torch.arange(-3, 4, dtype=torch.float64)) * math.pi / omega
    mids = torch.arange(-3, 4, dtype=torch.float64) * math.pi / omega
    for t, expect in ((crests, 1.0), (mids, math.sqrt(1.0 + bench.kappa))):
        y = t.unsqueeze(-1) * b.unsqueeze(0)  # b^T y = t, a^T y = 0
        norms = bench.grad_g(y).norm(dim=-1)
        torch.testing.assert_close(
            norms, torch.full_like(norms, expect), rtol=1e-11, atol=1e-11
        )


def test_grad_norm_dense_across_the_ripple_at_k10():
    """Sweep b^T y densely across ~17 oscillations at k10 (omega ~ 31.6)."""
    bench = CurvatureSine("k10")
    u = torch.linspace(-4.0, 4.0, 400001, dtype=torch.float64)
    y = u.unsqueeze(-1) * bench._b.unsqueeze(0)
    norms = bench.grad_g(y).norm(dim=-1)
    assert float(norms.min()) >= 1.0 - 1e-12
    assert float(norms.max()) <= math.sqrt(1001.0) + 1e-12
    assert float(norms.min()) < 1.0 + 1e-3
    assert float(norms.max()) > math.sqrt(1001.0) - 1e-3
    p = bench.p(u)
    crossings = int((p[1:] * p[:-1] < 0).sum())
    assert crossings > 70, crossings  # 8 * omega / pi ~ 80 half-periods
    assert float(p.abs().max()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("variant", VARIANTS)
def test_objective_is_squared_distance_to_condition(variant):
    bench = CurvatureSine(variant)
    q = bench.sample_queries(64, "train", seed=17)
    y = _sample_y(bench, 64, seed=17)
    obj = bench.objective(y, q.conditions)
    torch.testing.assert_close(obj, ((y - q.conditions) ** 2).sum(dim=-1))
    assert bool(torch.isfinite(obj).all()) and bool((obj >= 0).all())


@pytest.mark.parametrize("variant", VARIANTS)
def test_condition_box_straddles_manifold(variant):
    """Straddle guard: sign(g(x)) varies over the box at every variant."""
    bench = CurvatureSine(variant)
    q = bench.sample_queries(4096, "train", seed=18)
    g = bench.g(q.conditions)
    frac_pos = float((g > 0).double().mean())
    assert 0.3 < frac_pos < 0.7, f"{variant}: box does not straddle g=0 ({frac_pos=})"
    # exact distance stays O(1) down to the 1/omega shrinkage the sine implies
    d = bench.reference_solution(q.conditions).f_star.sqrt()
    assert 0.01 < float(d.mean()) < 10.0, f"{variant}: mean d {float(d.mean()):.3f}"


def _brute_force_min(u: torch.Tensor, v: torch.Tensor, omega: float, n: int = 400_001):
    """Dense envelope of `min F` over `[u-m, u+m]`, which holds a minimizer as F >= (t-u)^2."""
    m = (p_sine(u, omega) - v).abs()
    lo = u - m * 1.0000001 - 1e-12
    hi = u + m * 1.0000001 + 1e-12
    out = torch.empty_like(u)
    for i in range(u.shape[0]):
        t = torch.linspace(float(lo[i]), float(hi[i]), n, dtype=torch.float64)
        out[i] = _f_1d(t, u[i], v[i], omega).min()
    return out


def _assert_not_worse_than_envelope(bench, x, label: str):
    x = x.double()
    sol = bench.reference_solution(x)
    brute = _brute_force_min(x @ bench._b, x @ bench._a, bench.omega)
    slack = sol.f_star - brute
    tol = 1e-12 * (1.0 + brute)
    assert bool((slack <= tol).all()), (
        f"{label} / {bench.variant}: solver worse than a 400k-point envelope by "
        f"{float((slack - tol).max()):.3e}"
    )
    assert bool(sol.from_stationary.all()), (
        f"{label} / {bench.variant}: winner was not a refined stationary point"
    )
    return sol


@pytest.mark.parametrize("variant", VARIANTS)
def test_reference_solution_is_feasible_and_consistent(variant):
    bench = CurvatureSine(variant)
    x = bench.eval_queries(seed=0, n=128).conditions
    sol = bench.reference_solution(x)
    assert float(bench.g(sol.y_star).abs().max()) <= 1e-11
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
    """k0: exact projection onto `a^T y = 0`; no folds, no Delta_fold."""
    bench = CurvatureSine("k0")
    x = bench.eval_queries(seed=1, n=256).conditions.double()
    a, b = bench._a, bench._b
    v = x @ a
    torch.testing.assert_close(bench.g(x), v)
    sol = bench.reference_solution(x)
    torch.testing.assert_close(sol.y_star, x - v.unsqueeze(-1) * a)
    torch.testing.assert_close(sol.f_star, v * v)
    torch.testing.assert_close(sol.t_star, x @ b)
    assert bool(torch.isnan(sol.delta_fold).all())
    assert bool(torch.isnan(sol.t_second).all())
    assert bool(torch.isnan(sol.fold_star).all())
    assert bool((~sol.has_two_folds).all())
    assert int(sol.n_folds.max()) == 0


@pytest.mark.parametrize("variant", ["k1", "k4", "k7", "k10"])
def test_reference_matches_brute_force_envelope(variant):
    """Randomized batch vs a much denser independent scan of F."""
    bench = CurvatureSine(variant)
    x = bench.eval_queries(seed=3, n=24).conditions
    sol = _assert_not_worse_than_envelope(bench, x, "random")
    # the envelope's own discretization error is O(spacing^2 * F''), ~1e-6 at k10
    brute = _brute_force_min(x.double() @ bench._b, x.double() @ bench._a, bench.omega)
    assert float((brute - sol.f_star).max()) < 1e-5


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_reference_stress_crest_adjacent(variant):
    """Crest-adjacent queries on the crest evolute (tangency regime)."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    t = torch.cat(
        [
            (0.5 + torch.arange(-2, 3, dtype=torch.float64)) * math.pi / omega,
            (0.5 + torch.arange(-2, 3, dtype=torch.float64) + 0.12)
            * math.pi
            / omega,
        ]
    )
    foot = t.unsqueeze(-1) * bench._b + p_sine(t, omega).unsqueeze(-1) * bench._a
    normal = bench.grad_g(foot)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    side = torch.sign(p_sine_second(t, omega)).unsqueeze(-1)
    radius = 1.0 / curvature_k(t, omega)
    for s in (0.5, 0.95, 1.0, 1.05, 2.0, 10.0):
        x = foot + (s * radius).unsqueeze(-1) * side * normal
        _assert_not_worse_than_envelope(bench, x, f"crest evolute s={s}")


@pytest.mark.parametrize("variant", ["k4", "k7", "k10"])
def test_reference_stress_midline(variant):
    """Midline stress: queries on the `K = 0` / max-slope fold boundaries."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    t = torch.arange(-3, 4, dtype=torch.float64) * math.pi / omega
    foot = t.unsqueeze(-1) * bench._b  # p = 0 on the midline
    normal = bench.grad_g(foot)
    normal = normal / normal.norm(dim=-1, keepdim=True)
    for off in (-2.0, -0.5, -1e-3, 1e-3, 0.5, 2.0):
        _assert_not_worse_than_envelope(
            bench, foot + off * normal, f"midline off={off}"
        )


@pytest.mark.parametrize("variant", ["k7", "k10"])
def test_reference_stress_between_arches(variant):
    """Above a trough, two flanking arches tie: the curvature_sine medial-axis analog."""
    bench = CurvatureSine(variant)
    omega = bench.omega
    # trough at omega t = -pi/2: points above it are equidistant from two crests
    t0 = -0.5 * math.pi / omega
    v = torch.linspace(0.2, 2.0, 33, dtype=torch.float64)
    # odd count => jitter = 0 is included, i.e. the EXACT tie is in the batch
    jitter = torch.linspace(-0.4, 0.4, 33, dtype=torch.float64) * math.pi / omega
    u = t0 + jitter
    x = u.unsqueeze(-1) * bench._b + v.unsqueeze(-1) * bench._a
    sol = _assert_not_worse_than_envelope(bench, x, "between arches")
    assert int(sol.has_two_folds.sum()) >= 24
    delta = sol.delta_fold[sol.has_two_folds]
    assert bool((delta >= 0).all())
    assert float(delta.min()) < 1e-6  # the exact tie really is tied


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_delta_fold_is_nonnegative_and_costs_the_other_fold(variant):
    bench = CurvatureSine(variant)
    x = bench.eval_queries(seed=0, n=256).conditions
    sol = bench.reference_solution(x)
    two = sol.has_two_folds
    assert bool((sol.delta_fold[two] >= 0).all())
    if bool(two.any()):
        # the runner-up sits in a different fold, objective exactly f* + Delta_fold
        t2 = sol.t_second[two]
        u = (x.double() @ bench._b)[two]
        v = (x.double() @ bench._a)[two]
        torch.testing.assert_close(
            _f_1d(t2, u, v, bench.omega), sol.f_star[two] + sol.delta_fold[two]
        )
        assert bool((sol.fold_second[two] != sol.fold_star[two]).all())
        assert bool(((t2 - sol.t_star[two]).abs() > 1e-9).all())
    assert bool(torch.isnan(sol.delta_fold[~two]).all())
    assert bool(torch.isnan(sol.t_second[~two]).all())
    assert bool(torch.isnan(sol.fold_second[~two]).all())


@pytest.mark.parametrize("variant", VARIANTS[1:])
def test_n_folds_counts_distinct_fold_candidates(variant):
    """`n_folds` >= 1, and `>= 2` exactly when the Delta_fold ambiguity exists."""
    bench = CurvatureSine(variant)
    x = bench.eval_queries(seed=0, n=256).conditions
    sol = bench.reference_solution(x)
    assert int(sol.n_folds.min()) >= 1
    torch.testing.assert_close(
        (sol.n_folds >= 2), sol.has_two_folds, rtol=0.0, atol=0.0
    )
    assert bool(
        (sol.fold_star[sol.has_two_folds] != sol.fold_second[sol.has_two_folds]).all()
    )


def test_fold_candidate_count_grows_with_kappa():
    """More oscillations => more competing arches (the k10 headline claim)."""
    means = [
        float(
            CurvatureSine(v)
            .reference_solution(CurvatureSine(v).eval_queries(seed=0, n=128).conditions)
            .n_folds.double()
            .mean()
        )
        for v in ("k4", "k6", "k8", "k10")
    ]
    assert means == sorted(means), means
    assert means[0] < 1.5 < means[-1]


@pytest.mark.parametrize("variant", VARIANTS)
def test_snap_is_exactly_feasible_and_preserves_t(variant):
    """`y_snap = y - g(y)*a` is fold- and t-preserving and exactly on `{g=0}`."""
    bench = CurvatureSine(variant)
    y = _sample_y(bench, 256, seed=21).double()
    y_snap = bench.snap(y)
    # exact in real arithmetic; the ulp-level change in the recomputed `b^T y`
    # is amplified by `p' = O(omega)`, hence the omega-aware bound
    assert float(bench.g(y_snap).abs().max()) <= 1e-14 * (1.0 + bench.omega)
    torch.testing.assert_close(bench.t_of(y_snap), bench.t_of(y))
    torch.testing.assert_close(bench.fold(y_snap @ bench._b), bench.fold(y @ bench._b))
    x = bench.sample_queries(256, "train", seed=21).conditions.double()
    f_snap = ((y_snap - x) ** 2).sum(dim=-1)
    f_star = bench.reference_solution(x).f_star
    assert float((f_star - f_snap).max()) <= 1e-12
