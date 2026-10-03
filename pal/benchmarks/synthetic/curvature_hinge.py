"""Curvature hinge dial: projection onto `a^T y = p_r(b^T y)` with a centered smoothed hinge.

`p_r(t) = (t + sqrt(t^2 + r^2) - r) / 2` with curvature scale kappa = 1/(2r) = max p''.
`k0` is the affine anchor p(t) = t/2.
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

# Family-wide, not per-variant: identical query sets across the kappa sweep.
_EVAL_SALT = 3101
_CTOR_SEED = 3101

#: Variant suffix -> curvature scale kappa = 1/(2r) = max p'' (`k0` is the affine anchor kappa = 0).
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

#: Variant suffix -> hinge radius `r = 1/(2kappa)`; `r = 0.0` is the sentinel for `p(t) = t/2`.
R_BY_VARIANT: dict[str, float] = {
    k: (0.0 if kappa == 0.0 else 1.0 / (2.0 * kappa))
    for k, kappa in KAPPA_BY_VARIANT.items()
}

VARIANTS: list[str] = list(KAPPA_BY_VARIANT)

_GRID_LIN = 257  # linear nodes across the certified enclosure
_GRID_LOG = 241  # log-spaced nodes per asymptotic arm (anchored at t = 0)
_GRID_CAP = 65  # linear nodes across the corner cap, in units of r
_CAP_HALF_WIDTHS = 8.0  # cap grid spans t/r in [-8, 8]
_BISECT_ITERS = 100
_GOLDEN_ITERS = 80
_MAX_BRACKETS = 8  # F' has <= 3 roots here; 8 is slack, not a design bound
_MAX_TANGENTS = 6
_DEDUP_REL = 1e-7
_DEDUP_ABS = 1e-12


def p_r(t: Tensor, r: float) -> Tensor:
    """Centered smoothed hinge `p_r(t) = (t + sqrt(t^2+r^2) - r)/2`.

    `r = 0` selects the affine anchor `p(t) = t/2`. For `t < 0` the rationalized
    form `r^2/(sqrt(t^2+r^2) - t)` avoids cancellation.
    """
    if r == 0.0:
        return 0.5 * t
    t_pos = t.clamp(min=0.0)
    t_neg = t.clamp(max=0.0)
    s_pos = torch.sqrt(t_pos * t_pos + r * r)
    s_neg = torch.sqrt(t_neg * t_neg + r * r)
    pos = 0.5 * (t_pos + s_pos - r)
    neg = 0.5 * (r * r / (s_neg - t_neg) - r)
    return torch.where(t >= 0, pos, neg)


def p_r_prime(t: Tensor, r: float) -> Tensor:
    """`p_r'(t) = (1 + t/sqrt(t^2+r^2))/2 in (0, 1)`; `= 1/2` for the k0 anchor.

    For `t < 0` uses the cancellation-free form `r^2 / (2*s*(s - t))`, `s = sqrt(t^2+r^2)`.
    """
    if r == 0.0:
        return torch.full_like(t, 0.5)
    t_pos = t.clamp(min=0.0)
    t_neg = t.clamp(max=0.0)
    s_pos = torch.sqrt(t_pos * t_pos + r * r)
    s_neg = torch.sqrt(t_neg * t_neg + r * r)
    pos = 0.5 * (1.0 + t_pos / s_pos)
    neg = 0.5 * (r * r) / (s_neg * (s_neg - t_neg))
    return torch.where(t >= 0, pos, neg)


def p_r_second(t: Tensor, r: float) -> Tensor:
    """`p_r''(t) = r^2 / (2 (t^2+r^2)^{3/2})`; max `= 1/(2r) = kappa` at `t = 0`."""
    if r == 0.0:
        return torch.zeros_like(t)
    s2 = t * t + r * r
    return 0.5 * r * r / (s2 * torch.sqrt(s2))


def curvature_k(t: Tensor, r: float) -> Tensor:
    """Extrinsic graph curvature `K(t) = p''/(1+p'^2)^{3/2}`, peaking at `~ 0.754*kappa`."""
    d1 = p_r_prime(t, r)
    d2 = p_r_second(t, r)
    return d2 / (1.0 + d1 * d1).pow(1.5)


def _make_directions() -> tuple[Tensor, Tensor]:
    """Deterministic float64 orthonormal pair `(a, b)` via two Gram-Schmidt passes."""
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
        delta_arm: `[B]` `F_second - F_best` over distinct local minima,
            `NaN` when only one local minimum exists.
        t_second: `[B]` foot coordinate of the runner-up local minimum,
            `NaN` when `delta_arm` is `NaN`.
        from_stationary: `[B]` bool, the winner is a refined stationary point.
    """

    y_star: Tensor
    f_star: Tensor
    t_star: Tensor
    delta_arm: Tensor
    t_second: Tensor
    from_stationary: Tensor

    @property
    def has_two_arms(self) -> Tensor:
        """`[B]` bool: two distinct local minima exist (two-arm ambiguity)."""
        return ~torch.isnan(self.delta_arm)


def _f_1d(t: Tensor, u: Tensor, v: Tensor, r: float) -> Tensor:
    """`F(t) = (t-u)^2 + (p_r(t)-v)^2`, the exact reduced objective."""
    du = t - u
    dv = p_r(t, r) - v
    return du * du + dv * dv


def _f_1d_prime(t: Tensor, u: Tensor, v: Tensor, r: float) -> Tensor:
    return 2.0 * (t - u) + 2.0 * (p_r(t, r) - v) * p_r_prime(t, r)


def _f_1d_second(t: Tensor, u: Tensor, v: Tensor, r: float) -> Tensor:
    d1 = p_r_prime(t, r)
    return 2.0 + 2.0 * d1 * d1 + 2.0 * (p_r(t, r) - v) * p_r_second(t, r)


def _enclosure(u: Tensor, v: Tensor, r: float) -> tuple[Tensor, Tensor]:
    """Certified finite search interval `[u - m, u + m]`, `m = |p_r(u) - v|`.

    `F(t) >= (t-u)^2` and `F(u) = m^2`, so the interval contains a global minimizer.
    """
    m = (p_r(u, r) - v).abs()
    pad = 1e-12 * (1.0 + u.abs()) + 1e-12 * m
    half = m + pad
    return u - half, u + half


def _candidate_grid(lo: Tensor, hi: Tensor, r: float) -> Tensor:
    """Scale-aware partition: linear over the enclosure + log-spaced arms + cap.

    Nodes at `t = +/-r*10^k` keep the O(r) corner cap resolved at small r.
    """
    B = lo.shape[0]
    dev, dt = lo.device, lo.dtype
    frac = torch.linspace(0.0, 1.0, _GRID_LIN, device=dev, dtype=dt)
    grid = lo.unsqueeze(1) + (hi - lo).unsqueeze(1) * frac.unsqueeze(0)
    if r > 0.0:
        t_max = float(torch.maximum(lo.abs().max(), hi.abs().max()).item())
        t_max = max(t_max, 10.0 * r)
        expo = torch.linspace(
            math.log10(r) - 3.0, math.log10(t_max), _GRID_LOG, device=dev, dtype=dt
        )
        mags = torch.pow(torch.tensor(10.0, device=dev, dtype=dt), expo)
        zero = torch.zeros(1, device=dev, dtype=dt)
        cap = r * torch.linspace(
            -_CAP_HALF_WIDTHS, _CAP_HALF_WIDTHS, _GRID_CAP, device=dev, dtype=dt
        )
        fixed = torch.cat([-mags.flip(0), zero, mags, cap])
        grid = torch.cat([grid, fixed.unsqueeze(0).expand(B, -1)], dim=1)
    grid = grid.clamp(min=lo.unsqueeze(1), max=hi.unsqueeze(1))
    return grid.sort(dim=1).values


def _topk_masked(score: Tensor, mask: Tensor, k: int) -> tuple[Tensor, Tensor]:
    """`k` smallest-`score` entries where `mask`; returns `(indices, valid)`."""
    big = torch.where(mask, score, torch.full_like(score, float("inf")))
    vals, idx = big.sort(dim=1)
    return idx[:, :k], torch.isfinite(vals[:, :k])


def _bisect(
    lo: Tensor, hi: Tensor, u: Tensor, v: Tensor, r: float
) -> Tensor:
    """Vectorized bisection of `F'` on brackets `[lo, hi]` (float64 refinement)."""
    f_lo = _f_1d_prime(lo, u, v, r)
    for _ in range(_BISECT_ITERS):
        mid = 0.5 * (lo + hi)
        f_mid = _f_1d_prime(mid, u, v, r)
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
    r: float,
    *,
    on_abs_prime: bool,
) -> Tensor:
    """Golden-section minimization of `|F'|` (tangency hunt) or `F` (fallback)."""

    def obj(t: Tensor) -> Tensor:
        if on_abs_prime:
            return _f_1d_prime(t, u, v, r).abs()
        return _f_1d(t, u, v, r)

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
    u: Tensor, v: Tensor, r: float
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Global 1D solve of `F(t) = (t-u)^2 + (p_r(t)-v)^2`.

    Returns `(t_star, f_star, delta_arm, t_second, from_stationary)`.
    """
    lo, hi = _enclosure(u, v, r)
    grid = _candidate_grid(lo, hi, r)
    u1, v1 = u.unsqueeze(1), v.unsqueeze(1)
    fp = _f_1d_prime(grid, u1, v1, r)
    n_int = grid.shape[1] - 1
    inf = torch.tensor(float("inf"), dtype=u.dtype, device=u.device)

    # sign-changing brackets of F'
    sign_change = (fp[:, :-1] * fp[:, 1:]) < 0
    order = torch.arange(n_int, device=u.device).unsqueeze(0).expand_as(sign_change)
    b_idx, b_valid = _topk_masked(order.to(u.dtype), sign_change, _MAX_BRACKETS)
    b_idx = b_idx.clamp(max=n_int - 1)
    roots = _bisect(
        grid.gather(1, b_idx),
        grid.gather(1, b_idx + 1),
        u1,
        v1,
        r,
    )

    # tangent / double roots: |F'| dips to a local min without sign change
    absfp = fp.abs()
    is_dip = (absfp[:, 1:-1] <= absfp[:, :-2]) & (absfp[:, 1:-1] <= absfp[:, 2:])
    t_idx, t_valid = _topk_masked(absfp[:, 1:-1], is_dip, _MAX_TANGENTS)
    tangents = _golden_min(
        grid.gather(1, t_idx),
        grid.gather(1, t_idx + 2),
        u1,
        v1,
        r,
        on_abs_prime=True,
    )
    scale = 1.0 + u.abs() + v.abs()
    tol_stat = 1e-9 * scale.unsqueeze(1)
    t_valid = t_valid & (_f_1d_prime(tangents, u1, v1, r).abs() <= tol_stat)

    # Candidates: stationary points, enclosure bounds, and a polished best grid node.
    stat = torch.cat([roots, tangents], dim=1)
    stat_valid = torch.cat([b_valid, t_valid], dim=1)
    f_stat = torch.where(stat_valid, _f_1d(stat, u1, v1, r), inf)

    f_grid = _f_1d(grid, u1, v1, r)
    g_best = f_grid.argmin(dim=1, keepdim=True)
    g_lo = grid.gather(1, (g_best - 1).clamp(min=0))
    g_hi = grid.gather(1, (g_best + 1).clamp(max=grid.shape[1] - 1))
    polished = _golden_min(
        g_lo, g_hi, u1, v1, r, on_abs_prime=False
    )
    extra = torch.cat([lo.unsqueeze(1), hi.unsqueeze(1), polished], dim=1)
    f_extra = _f_1d(extra, u1, v1, r)

    all_t = torch.cat([stat, extra], dim=1)
    all_f = torch.cat([f_stat, f_extra], dim=1)
    best = all_f.argmin(dim=1, keepdim=True)
    t_star = all_t.gather(1, best).squeeze(1)
    f_star = all_f.gather(1, best).squeeze(1)
    f_stat_best = f_stat.min(dim=1).values
    from_stationary = f_stat_best <= f_star + 1e-12 * (1.0 + f_star)

    # distinct local minima -> Delta_arm (cost of the other arm)
    is_min = stat_valid & (
        _f_1d_second(stat, u1, v1, r) > 0
    )
    f_min = torch.where(is_min, f_stat, inf)
    order_f = f_min.argsort(dim=1)
    t_sorted = stat.gather(1, order_f)
    f_sorted = f_min.gather(1, order_f)
    t_first = t_sorted[:, :1]
    distinct = (t_sorted - t_first).abs() > (_DEDUP_ABS + _DEDUP_REL * t_first.abs())
    f_other = torch.where(distinct, f_sorted, inf)
    second = f_other.argmin(dim=1, keepdim=True)
    f_second = f_other.gather(1, second).squeeze(1)
    t_second = t_sorted.gather(1, second).squeeze(1)
    nan = torch.full_like(f_star, float("nan"))
    two_arms = torch.isfinite(f_second) & torch.isfinite(f_sorted[:, 0])
    delta_arm = torch.where(two_arms, f_second - f_sorted[:, 0], nan)
    t_second = torch.where(two_arms, t_second, nan)
    return t_star, f_star, delta_arm, t_second, from_stationary


def _make_spec(variant: str) -> BenchmarkSpec:
    kappa = KAPPA_BY_VARIANT[variant]
    r = R_BY_VARIANT[variant]
    bounds_lo = torch.full((_DIM,), -3.0)
    bounds_hi = torch.full((_DIM,), 3.0)
    return BenchmarkSpec(
        id=f"curvature_hinge_{variant}",
        family="curvature_hinge",
        variant=variant,
        dim=_DIM,
        n_eq=_N_EQ,
        n_ineq=0,
        constraint_names=["smoothed_hinge"],
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
            f"curvature dial: kappa={kappa:g}, r={r:g}; "
            f"g(y)=a.y - p_r(b.y), p_r(t)=(t+sqrt(t^2+r^2)-r)/2, raw g"
            + (" (k0: affine anchor p(t)=t/2)" if r == 0.0 else "")
        ),
    )


class CurvatureHinge:
    """Conditional projection onto a centered smoothed-hinge hypersurface.

    Args:
        variant: One of `KAPPA_BY_VARIANT` (`"k0"` ... `"k10"`).
    """

    def __init__(self, variant: str = "k4") -> None:
        if variant not in KAPPA_BY_VARIANT:
            raise ValueError(
                f"unknown curvature_hinge variant '{variant}'; "
                f"expected one of {sorted(KAPPA_BY_VARIANT)}"
            )
        self.variant = variant
        self.kappa = KAPPA_BY_VARIANT[variant]
        self.r = R_BY_VARIANT[variant]
        #: True for the k0 affine anchor (no corner, no arms, no medial axis).
        self.is_affine_anchor = self.r == 0.0
        self.spec = _make_spec(variant)
        self._a, self._b = _make_directions()

    def _dirs(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._a.to(device=like.device, dtype=like.dtype),
            self._b.to(device=like.device, dtype=like.dtype),
        )

    def p(self, t: Tensor) -> Tensor:
        """`p_r(t)` in float64."""
        return p_r(t.double(), self.r)

    def p_prime(self, t: Tensor) -> Tensor:
        """`p_r'(t)` in float64."""
        return p_r_prime(t.double(), self.r)

    def p_second(self, t: Tensor) -> Tensor:
        """`p_r''(t)` in float64."""
        return p_r_second(t.double(), self.r)

    def curvature(self, t: Tensor) -> Tensor:
        """Local extrinsic curvature `K(t)` of `{g = 0}`, float64."""
        return curvature_k(t.double(), self.r)

    def g(self, y: Tensor) -> Tensor:
        """`[B]` raw equality value `a^T y - p_r(b^T y)` in float64."""
        return self._g_native(y.double())

    def grad_g(self, y: Tensor) -> Tensor:
        """`[B, dim]` closed-form `grad g = a - p_r'(b^T y)*b` in float64.

        `||grad g|| = sqrt(1 + p_r'^2) in [1, sqrt(2)]` because `a perp b` are orthonormal.
        """
        yd = y.double()
        a, b = self._dirs(yd)
        return a.expand_as(yd) - p_r_prime(yd @ b, self.r).unsqueeze(-1) * b

    def t_of(self, y: Tensor) -> Tensor:
        """`b^T y`, the 1D hinge coordinate, in float64."""
        yd = y.double()
        return yd @ self._dirs(yd)[1]

    def _g_native(self, y: Tensor) -> Tensor:
        a, b = self._dirs(y)
        return y @ a - p_r(y @ b, self.r)

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
                name="smoothed_hinge",
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
        `min_t F(t) = (t-u)^2 + (p_r(t)-v)^2`, `u = b^T x`, `v = a^T x`. k0 uses the
        closed-form plane projection.
        """
        xd = x.double()
        a, b = self._dirs(xd)
        u = xd @ b
        v = xd @ a
        if self.is_affine_anchor:
            # normal n = a - b/2, ||n||^2 = 5/4; g(x) = v - u/2.
            gx = v - 0.5 * u
            n = a - 0.5 * b
            nn = 1.25
            y_star = xd - (gx / nn).unsqueeze(-1) * n
            f_star = gx * gx / nn
            t_star = u + 0.4 * gx
            nan = torch.full_like(f_star, float("nan"))
            return ReferenceSolution(
                y_star=y_star,
                f_star=f_star,
                t_star=t_star,
                delta_arm=nan,
                t_second=nan,
                from_stationary=torch.ones_like(f_star, dtype=torch.bool),
            )
        t_star, f_star, delta_arm, t_second, from_stat = _solve_1d(u, v, self.r)
        y_star = (
            xd
            + (t_star - u).unsqueeze(-1) * b
            + (p_r(t_star, self.r) - v).unsqueeze(-1) * a
        )
        return ReferenceSolution(
            y_star=y_star,
            f_star=f_star,
            t_star=t_star,
            delta_arm=delta_arm,
            t_second=t_second,
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
