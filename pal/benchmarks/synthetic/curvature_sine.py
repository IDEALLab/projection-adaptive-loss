"""Curvature sine dial: projection onto `a^T y = sin(omega*b^T y)` with curvature kappa = omega^2.

kappa is exactly the max curvature of the zero set (attained at every crest).
`k0` is the linear anchor omega = 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

_DIM = 8
_COND_DIM = 8
_N_EQ = 1

# Target box: straddles {g = 0} at O(1) distance.
_X_LO = -1.5
_X_HI = 1.5

# Family-wide and equal to curvature_hinge: the two families share a, b and every query set.
_EVAL_SALT = 3101
_CTOR_SEED = 3101

#: Variant suffix -> curvature kappa = omega^2 = max curvature (`k0` is the linear anchor).
KAPPA_BY_VARIANT: dict[str, float] = {
    "k0": 0.0,
    "k1": 0.03,
    "k2": 0.1,
    "k3": 0.3,
    "k4": 1.0,
    "k5": 3.0,
    "k6": 10.0,
    "k7": 30.0,
    "k8": 100.0,
    "k9": 300.0,
    "k10": 1000.0,
}

#: Variant suffix -> angular frequency `omega = sqrt(kappa)` (`omega = 0` for the k0 anchor).
OMEGA_BY_VARIANT: dict[str, float] = {
    k: math.sqrt(kappa) for k, kappa in KAPPA_BY_VARIANT.items()
}

VARIANTS: list[str] = list(KAPPA_BY_VARIANT)

_GRID_MIN = 257  # linear nodes across the certified enclosure (floor)
_GRID_MAX = 8193  # hard cap; k10 needs only a few hundred
_NODES_PER_HALF_PERIOD = 8.0  # spec asks for >= ~6 per half period
_BISECT_ITERS = 100
_GOLDEN_ITERS = 80
_MAX_TANGENTS = 32  # smallest-|F'| dips kept for the tangency hunt
_DEDUP_REL = 1e-7
_DEDUP_ABS = 1e-12


def p_sine(t: Tensor, omega: float) -> Tensor:
    """Ripple profile `p_omega(t) = sin(omega t)`; `omega = 0` gives the k0 anchor `p == 0`."""
    if omega == 0.0:
        return torch.zeros_like(t)
    return torch.sin(omega * t)


def p_sine_prime(t: Tensor, omega: float) -> Tensor:
    """`p_omega'(t) = omega cos(omega t)`, zero at the crests, `+/-omega` at the midline."""
    if omega == 0.0:
        return torch.zeros_like(t)
    return omega * torch.cos(omega * t)


def p_sine_second(t: Tensor, omega: float) -> Tensor:
    """`p_omega''(t) = -omega^2 sin(omega t)`; `|p''| = omega^2 = kappa` exactly at the crests."""
    if omega == 0.0:
        return torch.zeros_like(t)
    return -(omega * omega) * torch.sin(omega * t)


def curvature_k(t: Tensor, omega: float) -> Tensor:
    """Extrinsic graph curvature `K(t) = |p''|/(1+p'^2)^{3/2}` of `{g = 0}`.

    Unsigned: exactly `kappa` at every crest and `0` at every midline crossing.
    """
    d1 = p_sine_prime(t, omega)
    d2 = p_sine_second(t, omega)
    return d2.abs() / (1.0 + d1 * d1).pow(1.5)


def fold_index(t: Tensor, omega: float) -> Tensor:
    """Fold (arch) label `floor(omega t/pi)`: one crest or trough per fold, float64-valued."""
    if omega == 0.0:
        return torch.zeros_like(t)
    return torch.floor(omega * t / math.pi)


def _make_directions() -> tuple[Tensor, Tensor]:
    """Deterministic float64 orthonormal pair `(a, b)`, bit-identical to curvature_hinge's."""
    g = torch.Generator().manual_seed(_CTOR_SEED)
    raw = torch.randn(2, _DIM, generator=g, dtype=torch.float64)
    a = raw[0] / raw[0].norm()
    b = raw[1] - (raw[1] @ a) * a
    b = b / b.norm()
    b = b - (b @ a) * a  # second pass: orthogonality to float64 noise floor
    b = b / b.norm()
    cos_ab = float(torch.dot(a, b).abs())
    if cos_ab > 1e-15:
        raise ValueError(f"Gram-Schmidt failed: |a^T b| = {cos_ab:.3e}")
    return a, b


@dataclass(frozen=True)
class ReferenceSolution:
    """Certified per-query solution of `min ||y - x||^2 s.t. g(y) = 0`.

    All tensors are float64 with leading batch dim `B`.

    Attributes:
        y_star: `[B, dim]` exact constrained minimizer.
        f_star: `[B]` optimal objective `||y* - x||^2` (`= d^2`).
        t_star: `[B]` `b^T y*`, the 1D coordinate of the projection foot.
        fold_star: `[B]` fold index `floor(omega t*/pi)` of the foot (`NaN` for k0).
        delta_fold: `[B]` `F_second - F_best` over local minima in distinct
            folds, `NaN` when fewer than two folds carry a local minimum.
        t_second: `[B]` foot coordinate of the best other-fold local minimum,
            `NaN` when `delta_fold` is `NaN`.
        fold_second: `[B]` its fold index, `NaN` when `delta_fold` is `NaN`.
        n_folds: `[B]` number of distinct folds carrying a local minimum.
        from_stationary: `[B]` bool, the winner is a refined stationary point.
    """

    y_star: Tensor
    f_star: Tensor
    t_star: Tensor
    fold_star: Tensor
    delta_fold: Tensor
    t_second: Tensor
    fold_second: Tensor
    n_folds: Tensor
    from_stationary: Tensor

    @property
    def has_two_folds(self) -> Tensor:
        """`[B]` bool: local minima exist in two distinct folds (ambiguity)."""
        return ~torch.isnan(self.delta_fold)


def _f_1d(t: Tensor, u: Tensor, v: Tensor, omega: float) -> Tensor:
    """`F(t) = (t-u)^2 + (p_omega(t)-v)^2`, the exact reduced objective."""
    du = t - u
    dv = p_sine(t, omega) - v
    return du * du + dv * dv


def _f_1d_prime(t: Tensor, u: Tensor, v: Tensor, omega: float) -> Tensor:
    return 2.0 * (t - u) + 2.0 * (p_sine(t, omega) - v) * p_sine_prime(t, omega)


def _f_1d_second(t: Tensor, u: Tensor, v: Tensor, omega: float) -> Tensor:
    d1 = p_sine_prime(t, omega)
    return 2.0 + 2.0 * d1 * d1 + 2.0 * (p_sine(t, omega) - v) * p_sine_second(t, omega)


def _enclosure(u: Tensor, v: Tensor, omega: float) -> tuple[Tensor, Tensor]:
    """Certified finite search interval `[u - m, u + m]`, `m = |p_omega(u) - v|`.

    `F(t) >= (t-u)^2` and `F(u) = m^2`, so the interval contains a global minimizer.
    """
    m = (p_sine(u, omega) - v).abs()
    pad = 1e-12 * (1.0 + u.abs()) + 1e-12 * m
    half = m + pad
    return u - half, u + half


def _grid_size(lo: Tensor, hi: Tensor, omega: float) -> int:
    """Node count giving `>= _NODES_PER_HALF_PERIOD` nodes per half period.

    Half period `= pi/omega`; the widest enclosure in the batch sets the count.
    """
    width = float((hi - lo).max())
    need = int(math.ceil(_NODES_PER_HALF_PERIOD * omega * width / math.pi)) + 1
    return max(_GRID_MIN, min(_GRID_MAX, need))


def _candidate_grid(lo: Tensor, hi: Tensor, omega: float) -> Tensor:
    """Uniform scale-aware partition of the certified enclosure, `[B, n]`."""
    n = _grid_size(lo, hi, omega)
    frac = torch.linspace(0.0, 1.0, n, device=lo.device, dtype=lo.dtype)
    return lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * frac.unsqueeze(0)


def _topk_masked(score: Tensor, mask: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """`k` smallest-`score` entries where `mask`; returns `(indices, valid)`."""
    big = torch.where(mask, score, torch.full_like(score, float("inf")))
    vals, idx = big.sort(dim=1)
    return idx[:, :k], torch.isfinite(vals[:, :k])


def _bisect(lo: Tensor, hi: Tensor, u: Tensor, v: Tensor, omega: float) -> Tensor:
    """Vectorized bisection of `F'` on brackets `[lo, hi]` (float64 refinement)."""
    f_lo = _f_1d_prime(lo, u, v, omega)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        f_mid = _f_1d_prime(mid, u, v, omega)
        same = (f_mid > 0) == (f_lo > 0)
        lo = torch.where(same, mid, lo)
        f_lo = torch.where(same, f_mid, f_lo)
        hi = torch.where(same, hi, mid)
    return 0.5 * (lo + hi)


def _golden_min(
    lo: Tensor,
    hi: Tensor,
    u: Tensor,
    v: Tensor,
    omega: float,
    *,
    on_abs_prime: bool,
) -> Tensor:
    """Golden-section minimization of `|F'|` (tangency hunt) or `F` (fallback)."""

    def obj(t: Tensor) -> Tensor:
        if on_abs_prime:
            return _f_1d_prime(t, u, v, omega).abs()
        return _f_1d(t, u, v, omega)

    inv_phi = (math.sqrt(5.0) - 1.0) / 2.0
    c = hi - inv_phi * (hi - lo)
    d = lo + inv_phi * (hi - lo)
    f_c, f_d = obj(c), obj(d)
    for _ in range(_GOLDEN_ITERS):
        take_left = f_c <= f_d
        hi = torch.where(take_left, d, hi)
        lo = torch.where(take_left, lo, c)
        c = hi - inv_phi * (hi - lo)
        d = lo + inv_phi * (hi - lo)
        f_c, f_d = obj(c), obj(d)
    return 0.5 * (lo + hi)


def _solve_1d(
    u: Tensor, v: Tensor, omega: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Global 1D solve of `F(t) = (t-u)^2 + (p_omega(t)-v)^2`.

    Returns `(t_star, f_star, delta_fold, t_second, n_folds, from_stationary)`.
    Bracket capacity is derived from the partition (one local minimum per fold).
    """
    lo, hi = _enclosure(u, v, omega)
    grid = _candidate_grid(lo, hi, omega)
    u1, v1 = u.unsqueeze(1), v.unsqueeze(1)
    fp = _f_1d_prime(grid, u1, v1, omega)
    n_int = grid.shape[1] - 1
    inf = torch.tensor(float("inf"), dtype=u.dtype, device=u.device)

    # sign-changing brackets of F'
    sign_change = (fp[:, :-1] * fp[:, 1:]) < 0
    n_brackets = max(1, min(n_int, int(sign_change.sum(dim=1).max())))
    order = torch.arange(n_int, device=u.device).unsqueeze(0).expand_as(sign_change)
    b_idx, b_valid = _topk_masked(order.to(u.dtype), sign_change, n_brackets)
    b_idx = b_idx.clamp(max=n_int - 1)
    roots = _bisect(grid.gather(1, b_idx), grid.gather(1, b_idx + 1), u1, v1, omega)

    # tangent / double roots: |F'| dips to a local min without sign change
    absfp = fp.abs()
    is_dip = (absfp[:, 1:-1] <= absfp[:, :-2]) & (absfp[:, 1:-1] <= absfp[:, 2:])
    t_idx, t_valid = _topk_masked(absfp[:, 1:-1], is_dip, _MAX_TANGENTS)
    tangents = _golden_min(
        grid.gather(1, t_idx), grid.gather(1, t_idx + 2), u1, v1, omega, on_abs_prime=True
    )
    scale = 1.0 + u.abs() + v.abs()
    tol_stat = 1e-9 * scale.unsqueeze(1)
    t_valid = t_valid & (_f_1d_prime(tangents, u1, v1, omega).abs() <= tol_stat)

    # Candidates: stationary points, enclosure bounds, and a polished best grid node.
    stat = torch.cat([roots, tangents], dim=1)
    stat_valid = torch.cat([b_valid, t_valid], dim=1)
    f_stat = torch.where(stat_valid, _f_1d(stat, u1, v1, omega), inf)

    f_grid = _f_1d(grid, u1, v1, omega)
    g_best = f_grid.argmin(dim=1, keepdim=True)
    g_lo = grid.gather(1, (g_best - 1).clamp(min=0))
    g_hi = grid.gather(1, (g_best + 1).clamp(max=grid.shape[1] - 1))
    polished = _golden_min(g_lo, g_hi, u1, v1, omega, on_abs_prime=False)
    extra = torch.cat([lo.unsqueeze(1), hi.unsqueeze(1), polished], dim=1)
    f_extra = _f_1d(extra, u1, v1, omega)

    all_t = torch.cat([stat, extra], dim=1)
    all_f = torch.cat([f_stat, f_extra], dim=1)
    best = all_f.argmin(dim=1, keepdim=True)
    t_star = all_t.gather(1, best).squeeze(1)
    f_star = all_f.gather(1, best).squeeze(1)
    f_stat_best = f_stat.min(dim=1).values
    from_stationary = f_stat_best <= f_star + 1e-12 * (1.0 + f_star)

    # local minima -> fold count + Delta_fold (cost of the best other arch)
    is_min = stat_valid & (_f_1d_second(stat, u1, v1, omega) > 0)
    folds = fold_index(stat, omega)
    f_min = torch.where(is_min, f_stat, inf)

    fold_masked = torch.where(is_min, folds, torch.full_like(folds, float("inf")))
    fold_sorted_asc = fold_masked.sort(dim=1).values
    is_new = torch.cat(
        [
            torch.ones_like(fold_sorted_asc[:, :1], dtype=torch.bool),
            fold_sorted_asc[:, 1:] != fold_sorted_asc[:, :-1],
        ],
        dim=1,
    ) & torch.isfinite(fold_sorted_asc)
    n_folds = is_new.sum(dim=1)

    order_f = f_min.argsort(dim=1)
    t_sorted = stat.gather(1, order_f)
    f_sorted = f_min.gather(1, order_f)
    fold_ord = folds.gather(1, order_f)
    t_first, fold_first = t_sorted[:, :1], fold_ord[:, :1]
    distinct = (fold_ord != fold_first) & (
        (t_sorted - t_first).abs() > (_DEDUP_ABS + _DEDUP_REL * t_first.abs())
    )
    f_other = torch.where(distinct, f_sorted, inf)
    second = f_other.argmin(dim=1, keepdim=True)
    f_second = f_other.gather(1, second).squeeze(1)
    t_second = t_sorted.gather(1, second).squeeze(1)
    nan = torch.full_like(f_star, float("nan"))
    two_folds = torch.isfinite(f_second) & torch.isfinite(f_sorted[:, 0])
    delta_fold = torch.where(two_folds, f_second - f_sorted[:, 0], nan)
    t_second = torch.where(two_folds, t_second, nan)
    return t_star, f_star, delta_fold, t_second, n_folds, from_stationary


def _make_spec(variant: str) -> BenchmarkSpec:
    kappa = KAPPA_BY_VARIANT[variant]
    omega = OMEGA_BY_VARIANT[variant]
    bounds_lo = torch.full((_DIM,), -4.0)
    bounds_hi = torch.full((_DIM,), 4.0)
    return BenchmarkSpec(
        id=f"curvature_sine_{variant}",
        family="curvature_sine",
        variant=variant,
        dim=_DIM,
        n_eq=_N_EQ,
        n_ineq=0,
        constraint_names=["sine_ripple"],
        constraint_types=["eq"],
        output_bounds=(bounds_lo, bounds_hi),
        condition_dim=_COND_DIM,
        zeta_dim=0,
        tolerance=1e-4,
        tau=1e-4,
        cost="cheap",
        recommended_device="cpu",
        train_batch_size=256,
        n_eval_default=512,
        notes=(
            f"oscillation dial: kappa={kappa:g}, omega={omega:g}; "
            f"g(y)=a.y - sin(omega*b.y), A=1, raw g"
            + (" (k0: linear anchor g=a.y)" if omega == 0.0 else "")
        ),
    )


class CurvatureSine:
    """Conditional projection onto a fixed-amplitude sine-ripple hypersurface.

    Args:
        variant: One of `KAPPA_BY_VARIANT` (`"k0"` ... `"k10"`).
    """

    def __init__(self, variant: str = "k4") -> None:
        if variant not in KAPPA_BY_VARIANT:
            raise ValueError(
                f"unknown curvature_sine variant '{variant}'; "
                f"expected one of {sorted(KAPPA_BY_VARIANT)}"
            )
        self.variant = variant
        self.kappa = KAPPA_BY_VARIANT[variant]
        self.omega = OMEGA_BY_VARIANT[variant]
        #: True for the k0 linear anchor (no crests, no folds, no ambiguity).
        self.is_affine_anchor = self.omega == 0.0
        self.spec = _make_spec(variant)
        self._a, self._b = _make_directions()

    def _dirs(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._a.to(device=like.device, dtype=like.dtype),
            self._b.to(device=like.device, dtype=like.dtype),
        )

    def p(self, t: Tensor) -> Tensor:
        """`p_omega(t) = sin(omega t)` in float64."""
        return p_sine(t.double(), self.omega)

    def p_prime(self, t: Tensor) -> Tensor:
        """`p_omega'(t)` in float64."""
        return p_sine_prime(t.double(), self.omega)

    def p_second(self, t: Tensor) -> Tensor:
        """`p_omega''(t)` in float64."""
        return p_sine_second(t.double(), self.omega)

    def curvature(self, t: Tensor) -> Tensor:
        """Local extrinsic curvature `K(t)` of `{g = 0}`, float64 (`= kappa` at crests)."""
        return curvature_k(t.double(), self.omega)

    def fold(self, t: Tensor) -> Tensor:
        """Fold (arch) index `floor(omega t/pi)` in float64; all zeros for the k0 anchor."""
        return fold_index(t.double(), self.omega)

    def g(self, y: Tensor) -> Tensor:
        """`[B]` raw equality value `a^T y - sin(omega*b^T y)` in float64."""
        return self._g_native(y.double())

    def grad_g(self, y: Tensor) -> Tensor:
        """`[B, dim]` closed-form `grad g = a - omega cos(omega*b^T y)*b` in float64.

        `||grad g|| = sqrt(1 + p_omega'^2) in [1, sqrt(1+kappa)]` since `a perp b` are orthonormal.
        """
        yd = y.double()
        a, b = self._dirs(yd)
        return a.expand_as(yd) - p_sine_prime(yd @ b, self.omega).unsqueeze(-1) * b

    def t_of(self, y: Tensor) -> Tensor:
        """`b^T y`, the 1D ripple coordinate, in float64."""
        yd = y.double()
        return yd @ self._dirs(yd)[1]

    def _g_native(self, y: Tensor) -> Tensor:
        a, b = self._dirs(y)
        return y @ a - p_sine(y @ b, self.omega)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        diff = x - conditions
        obj = (diff * diff).sum(dim=-1)
        raw_eq = self._g_native(x)
        B = x.shape[0]
        device = x.device
        return obj, [
            Constraint(
                value=raw_eq,
                type="eq",
                tol=torch.full((B,), 1e-3, device=device),
                margin=torch.zeros(B, device=device),
                name="sine_ripple",
            ),
        ]

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        """Raw `g`, deliberately not normalized by `||grad g||`."""
        return self._g_native(x).unsqueeze(-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def _sample_x(self, n: int, g: torch.Generator) -> Tensor:
        return _X_LO + (_X_HI - _X_LO) * torch.rand(n, _COND_DIM, generator=g)

    def sample_queries(
        self, n: int, split: Literal["train", "eval"], seed: int
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        c = self._sample_x(n, g)
        zeta = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=c)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + _EVAL_SALT)
        c = self._sample_x(n, g)
        zeta = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=c)

    def reference_solution(self, x: Tensor) -> ReferenceSolution:
        """Certified exact projection of every row of `x` onto `{g = 0}`.

        With `a perp b` orthonormal the problem reduces to the 1D
        `min_t F(t) = (t-u)^2 + (p_omega(t)-v)^2`, `u = b^T x`, `v = a^T x`. k0 uses the
        closed-form plane projection.
        """
        xd = x.double()
        a, b = self._dirs(xd)
        u = xd @ b
        v = xd @ a
        if self.is_affine_anchor:
            # normal n = a (unit), g(x) = v: y* = x - v*a, t* = u (a perp b).
            y_star = xd - v.unsqueeze(-1) * a
            f_star = v * v
            nan = torch.full_like(f_star, float("nan"))
            return ReferenceSolution(
                y_star=y_star,
                f_star=f_star,
                t_star=u,
                fold_star=nan,
                delta_fold=nan,
                t_second=nan,
                fold_second=nan,
                n_folds=torch.zeros_like(f_star, dtype=torch.int64),
                from_stationary=torch.ones_like(f_star, dtype=torch.bool),
            )
        t_star, f_star, delta_fold, t_second, n_folds, from_stat = _solve_1d(
            u, v, self.omega
        )
        y_star = (
            xd
            + (t_star - u).unsqueeze(-1) * b
            + (p_sine(t_star, self.omega) - v).unsqueeze(-1) * a
        )
        return ReferenceSolution(
            y_star=y_star,
            f_star=f_star,
            t_star=t_star,
            fold_star=fold_index(t_star, self.omega),
            delta_fold=delta_fold,
            t_second=t_second,
            fold_second=fold_index(t_second, self.omega),
            n_folds=n_folds,
            from_stationary=from_stat,
        )

    def y_star(self, conditions: Tensor) -> Tensor:
        """Exact constrained minimizer (the certified reference projection)."""
        return self.reference_solution(conditions).y_star

    def snap(self, y: Tensor) -> Tensor:
        """`y_snap = y - g(y)*a`, exactly feasible (diagnostic only)."""
        yd = y.double()
        a, _ = self._dirs(yd)
        return yd - self.g(yd).unsqueeze(-1) * a

    def check_env(self) -> None:
        return None

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None
