"""Curvature warp dial (paper curvature sweep): curvature_sine's manifold with a warped objective.

Objective `||T^-1(y) - T^-1(x)||^2` with `T(v) = v + a*sin(omega*b^T v)`.

The restricted problem is the affine k0 problem at every omega, with `y* = x - g(x)*a`.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query
from pal.benchmarks.synthetic.curvature_sine import (
    KAPPA_BY_VARIANT as _CURVATURE_SINE_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_sine import (
    CurvatureSine,
    ReferenceSolution,
    _solve_1d,
    curvature_k,
    fold_index,
    p_sine,
    p_sine_prime,
    p_sine_second,
)
from pal.constraints import Constraint

_DIM = 8
_COND_DIM = 8
_N_EQ = 1

# Target box: identical to curvature_hinge/curvature_sine (straddles {g = 0} at O(1) distance).
_X_LO = -1.5
_X_HI = 1.5

# Family-wide and equal to curvature_hinge/curvature_sine: bit-identical queries
# at every grid point.
_EVAL_SALT = 3101
_CTOR_SEED = 3101

#: Variant suffix -> curvature kappa = omega^2; k0...k10 equal curvature_sine's grid,
#: k11...k13 extend it.
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
    "k11": 10000.0,
    "k12": 100000.0,
    "k13": 1000000.0,
}

#: Variant suffix -> angular frequency `omega = sqrt(kappa)` (`omega = 0` for the k0 anchor).
OMEGA_BY_VARIANT: dict[str, float] = {
    k: math.sqrt(kappa) for k, kappa in KAPPA_BY_VARIANT.items()
}

VARIANTS: list[str] = list(KAPPA_BY_VARIANT)


def warp(v: Tensor, a: Tensor, b: Tensor, omega: float) -> Tensor:
    """`T(v) = v + a*sin(omega*b^T v)`, a shear along `a` driven by `b^T v`."""
    return v + p_sine(v @ b, omega).unsqueeze(-1) * a


def unwarp(v: Tensor, a: Tensor, b: Tensor, omega: float) -> Tensor:
    """`T^-1(v) = v - a*sin(omega*b^T v)`, exact because `b^T T(v) = b^T v` for `a perp b`."""
    return v - p_sine(v @ b, omega).unsqueeze(-1) * a


def warp_jacobian(v: Tensor, a: Tensor, b: Tensor, omega: float) -> Tensor:
    """`dT/dv = I + omega*cos(omega*b^T v)*a b^T`, `[B, dim, dim]` (analysis only, `det == 1`)."""
    eye = torch.eye(v.shape[-1], dtype=v.dtype, device=v.device).expand(
        v.shape[0], -1, -1
    )
    outer = a.unsqueeze(-1) * b.unsqueeze(0)
    return eye + p_sine_prime(v @ b, omega).reshape(-1, 1, 1) * outer


def _make_directions() -> tuple[Tensor, Tensor]:
    """Deterministic float64 orthonormal pair `(a, b)`, bit-identical to the other dials."""
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


def _euclidean_reference(
    y: Tensor, a: Tensor, b: Tensor, omega: float
) -> ReferenceSolution:
    """curvature_sine's Euclidean projection onto `{g = 0}` at an `omega` outside its grid.

    At `omega = 1000` the 8193-node cap gives ~ 4 nodes per half period, so on k13
    this is a resolved upper bound on the distance, not a certified projection.
    """
    yd = y.double()
    u = yd @ b
    v = yd @ a
    t_star, f_star, delta_fold, t_second, n_folds, from_stat = _solve_1d(u, v, omega)
    y_star = (
        yd
        + (t_star - u).unsqueeze(-1) * b
        + (p_sine(t_star, omega) - v).unsqueeze(-1) * a
    )
    return ReferenceSolution(
        y_star=y_star,
        f_star=f_star,
        t_star=t_star,
        fold_star=fold_index(t_star, omega),
        delta_fold=delta_fold,
        t_second=t_second,
        fold_second=fold_index(t_second, omega),
        n_folds=n_folds,
        from_stationary=from_stat,
    )


def _make_spec(variant: str) -> BenchmarkSpec:
    kappa = KAPPA_BY_VARIANT[variant]
    omega = OMEGA_BY_VARIANT[variant]
    # Same box as curvature_sine: max |y*_i| over k0...k13 is 2.87.
    bounds_lo = torch.full((_DIM,), -4.0)
    bounds_hi = torch.full((_DIM,), 4.0)
    return BenchmarkSpec(
        id=f"curvature_warp_{variant}",
        family="curvature_warp",
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
            f"warp control: kappa={kappa:g}, omega={omega:g}; "
            f"g(y)=a.y - sin(omega*b.y) (same manifold as curvature_sine), "
            f"f(y;x)=|T^-1(y)-T^-1(x)|^2 with T(v)=v+a*sin(omega*b.v)"
            + (" (k0: omega=0, T=identity, identical to curvature_sine k0)" if omega == 0.0 else "")
        ),
    )


class CurvatureWarp:
    """curvature_sine's manifold with an objective warped to remove the fold ambiguity.

    Args:
        variant: One of `KAPPA_BY_VARIANT` (`"k0"` ... `"k13"`).
    """

    def __init__(self, variant: str = "k4") -> None:
        if variant not in KAPPA_BY_VARIANT:
            raise ValueError(
                f"unknown curvature_warp variant '{variant}'; "
                f"expected one of {sorted(KAPPA_BY_VARIANT)}"
            )
        self.variant = variant
        self.kappa = KAPPA_BY_VARIANT[variant]
        self.omega = OMEGA_BY_VARIANT[variant]
        #: True for the k0 anchor (omega = 0, T = identity, so k0 equals curvature_sine k0).
        self.is_affine_anchor = self.omega == 0.0
        self.spec = _make_spec(variant)
        self._a, self._b = _make_directions()
        # curvature_sine companion for Euclidean distances to the shared manifold
        # (none on k11...k13).
        self._euclid = (
            CurvatureSine(variant) if variant in _CURVATURE_SINE_KAPPA_BY_VARIANT else None
        )

    def _dirs(self, like: Tensor) -> tuple[Tensor, Tensor]:
        return (
            self._a.to(device=like.device, dtype=like.dtype),
            self._b.to(device=like.device, dtype=like.dtype),
        )

    def p(self, t: Tensor) -> Tensor:
        """`p_omega(t) = sin(omega t)` in float64 (the shared manifold profile)."""
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
        """`[B]` raw equality value `a^T y - sin(omega*b^T y)` in float64 (= `a^T T^-1(y)`)."""
        return self._g_native(y.double())

    def grad_g(self, y: Tensor) -> Tensor:
        """`[B, dim]` closed-form `grad g = a - omega cos(omega*b^T y)*b` in float64."""
        yd = y.double()
        a, b = self._dirs(yd)
        return a.expand_as(yd) - p_sine_prime(yd @ b, self.omega).unsqueeze(-1) * b

    def t_of(self, y: Tensor) -> Tensor:
        """`b^T y`, the 1D ripple coordinate (a `T`-invariant), in float64."""
        yd = y.double()
        return yd @ self._dirs(yd)[1]

    def warp(self, v: Tensor) -> Tensor:
        """`T(v) = v + a*sin(omega*b^T v)` in float64."""
        vd = v.double()
        return warp(vd, *self._dirs(vd), self.omega)

    def unwarp(self, v: Tensor) -> Tensor:
        """`T^-1(v) = v - a*sin(omega*b^T v)` in float64."""
        vd = v.double()
        return unwarp(vd, *self._dirs(vd), self.omega)

    def warp_jacobian(self, v: Tensor) -> Tensor:
        """`[B, dim, dim]` `dT/dv` in float64 (diagnostic; `det == 1`)."""
        vd = v.double()
        return warp_jacobian(vd, *self._dirs(vd), self.omega)

    def _g_native(self, y: Tensor) -> Tensor:
        a, b = self._dirs(y)
        return y @ a - p_sine(y @ b, self.omega)

    def _unwarp_native(self, v: Tensor) -> Tensor:
        return unwarp(v, *self._dirs(v), self.omega)

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        assert conditions is not None and conditions.shape[-1] == _COND_DIM
        # f(y; x) = ||T^-1(y) - T^-1(x)||^2
        diff = self._unwarp_native(x) - self._unwarp_native(conditions)
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
        """Raw `g`, identical to curvature_sine, deliberately not normalized by `||grad g||`."""
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
        """Exact per-query solution of `min f(y; x)` s.t. `g(y) = 0`, closed form.

        With `z = T^-1(y)`: `z* = z_x - (a^T z_x)*a`, `y* = T(z*)`, `f* = g(x)^2`.
        There is exactly one local minimum, so `n_folds = 1` and `Delta_fold = NaN`.
        """
        xd = x.double()
        a, b = self._dirs(xd)
        z_x = unwarp(xd, a, b, self.omega)
        alpha = z_x @ a  # = a^T x - sin(omega*b^T x) = g(x)
        z_star = z_x - alpha.unsqueeze(-1) * a
        y_star = warp(z_star, a, b, self.omega)
        f_star = alpha * alpha
        t_star = z_star @ b  # = b^T x = b^T y*
        nan = torch.full_like(f_star, float("nan"))
        if self.is_affine_anchor:
            # omega = 0 => T = identity: identical to curvature_sine's k0 anchor, no folds.
            n_folds = torch.zeros_like(f_star, dtype=torch.int64)
            fold_star = nan
        else:
            n_folds = torch.ones_like(f_star, dtype=torch.int64)
            fold_star = fold_index(t_star, self.omega)
        return ReferenceSolution(
            y_star=y_star,
            f_star=f_star,
            t_star=t_star,
            fold_star=fold_star,
            delta_fold=nan,
            t_second=nan,
            fold_second=nan,
            n_folds=n_folds,
            from_stationary=torch.ones_like(f_star, dtype=torch.bool),
        )

    def y_star(self, conditions: Tensor) -> Tensor:
        """Exact constrained minimizer `y* = T(T^-1(x) - (a^T T^-1(x))*a) = x - g(x)*a`."""
        return self.reference_solution(conditions).y_star

    def geometric_reference(self, y: Tensor) -> ReferenceSolution:
        """curvature_sine's certified EUCLIDEAN projection of `y` onto the shared `{g = 0}`."""
        if self._euclid is None:
            yd = y.double()
            return _euclidean_reference(yd, *self._dirs(yd), self.omega)
        return self._euclid.reference_solution(y)

    def snap(self, y: Tensor) -> Tensor:
        """`y_snap = y - g(y)*a`, exactly feasible and `t`-preserving (diagnostic only)."""
        yd = y.double()
        a, _ = self._dirs(yd)
        return yd - self.g(yd).unsqueeze(-1) * a

    def check_env(self) -> None:
        return None

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None) -> None:
        return None
