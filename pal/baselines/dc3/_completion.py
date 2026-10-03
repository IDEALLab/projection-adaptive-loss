"""Generic Newton completion for DC3 partial-variable mode.

Given an NN that predicts ``y_partial = Z`` of shape ``[B, ydim - n_eq]``,
solve ``h(x, [y_partial, y_other]) = 0`` for ``y_other`` so the eq manifold
is satisfied, either in closed form (affine eq) or by batched Newton.
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.func import jacrev, vmap


class CompletionDivergedError(RuntimeError):
    """Raised when Newton completion blows up or returns NaN (messages start "Newton diverged")."""


def _scatter_per_sample(
    y_partial_single: Tensor, y_other_single: Tensor,
    pv_t: Tensor, ov_t: Tensor, ydim: int,
) -> Tensor:
    """Single-sample scatter into a length-``ydim`` vector (vmap-safe)."""
    y_full = torch.zeros(
        ydim, dtype=y_other_single.dtype, device=y_other_single.device
    )
    y_full = y_full.index_copy(0, pv_t, y_partial_single)
    y_full = y_full.index_copy(0, ov_t, y_other_single)
    return y_full


def _scatter_batched(
    Z: Tensor, Y_other: Tensor, pv_t: Tensor, ov_t: Tensor, ydim: int,
) -> Tensor:
    """Build ``[B, ydim]`` from ``[B, n_partial] + [B, n_other]``."""
    return vmap(
        lambda zp, yo: _scatter_per_sample(zp, yo, pv_t, ov_t, ydim)
    )(Z, Y_other)


def newton_complete(
    eq_resid_batched_fn: Callable[[Tensor, Tensor], Tensor],
    X: Tensor,
    Z: Tensor,
    partial_vars: Sequence[int],
    other_vars: Sequence[int],
    ydim: int,
    *,
    linear: bool = False,
    max_iter: int = 20,
    tol: float = 1e-6,
    reg: float = 1e-8,
    yf_damping: float = 0.0,
    lm_damping: float = 0.0,
    warm_start_fn: Callable[..., Tensor] | None = None,
    warm_start_ctx: dict | None = None,
    jacobian_mode: str = "loop",
    eq_resid_per_sample_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Solve ``h(x, y)=0`` for ``y_other`` given ``y_partial = Z``.

    Args:
        eq_resid_batched_fn: ``(Y[B, ydim], X[B, cond_dim]) -> [B, n_eq]``.
        X: Conditions ``[B, cond_dim]`` (``[B, 0]`` for unconditional benches).
        Z: Partial-vars output ``[B, n_partial]`` from the NN.
        partial_vars: y-vector indices the NN predicts. ``len = n_partial``.
        other_vars: y-vector indices to complete via Newton. ``len = n_eq``.
        ydim: Total y-vector dimension.
        linear: If True, take the closed-form path.
        max_iter, tol, reg: Newton knobs.
        yf_damping: Yamashita-Fukushima coefficient ``c``: ``mu_b = reg + c * ||h_b||^2``
            in ``(J^T J + mu I) delta = J^T h``. ``0.0`` = plain Newton ``(J + reg I) delta = h``.
        lm_damping: Constant damping added to ``mu``; either knob > 0 selects the LM form.
        warm_start_fn: Optional ``(y_partial, partial, other, ctx) -> [B, n_other]``
            callable for physical Newton init (e.g. ACOPF ``vm=1.0``).
        warm_start_ctx: Opaque dict forwarded to ``warm_start_fn``.
        jacobian_mode: ``"loop"`` builds J from ``n_eq`` explicit VJPs (visible to
            the probe hooks), ``"vmap"`` uses ``vmap(jacrev(eq_resid_per_sample_fn))``.
        eq_resid_per_sample_fn: ``(y_full[ydim], x[cond_dim]) -> [n_eq]``,
            required only when ``jacobian_mode="vmap"``.

    Returns:
        Full ``y`` ``[B, ydim]`` with eq satisfied to ``tol`` (or raises).

    Raises:
        CompletionDivergedError: ``||h||_inf > 1e3`` or NaN at any iter.
    """
    B = Z.shape[0]
    n_partial = len(partial_vars)
    n_other = len(other_vars)
    if n_partial + n_other != ydim:
        raise ValueError(
            f"partition mismatch: |partial|={n_partial} + |other|={n_other} "
            f"!= ydim={ydim}"
        )

    device = Z.device
    dtype = Z.dtype
    pv_t = torch.as_tensor(list(partial_vars), dtype=torch.long, device=device)
    ov_t = torch.as_tensor(list(other_vars), dtype=torch.long, device=device)

    if jacobian_mode not in ("loop", "vmap"):
        raise ValueError(
            f"unknown jacobian_mode {jacobian_mode!r}; expected 'loop' or 'vmap'"
        )
    if jacobian_mode == "vmap" and eq_resid_per_sample_fn is None:
        raise ValueError(
            "jacobian_mode='vmap' requires eq_resid_per_sample_fn"
        )

    if linear:
        return _linear_complete(
            eq_resid_batched_fn, X, Z, pv_t, ov_t, ydim, reg=reg,
            jacobian_mode=jacobian_mode,
            eq_resid_per_sample_fn=eq_resid_per_sample_fn,
        )

    if warm_start_fn is not None:
        y_other = warm_start_fn(
            Z, list(partial_vars), list(other_vars), warm_start_ctx or {},
        ).to(device=device, dtype=dtype)
    else:
        y_other = torch.zeros(B, n_other, device=device, dtype=dtype)

    eye = torch.eye(n_other, device=device, dtype=dtype).expand(B, -1, -1)

    for it in range(max_iter):
        if jacobian_mode == "loop":
            # One batched forward, then n_eq VJPs on its graph (vmap tracers bypass hooks).
            y_other_live = y_other.detach().requires_grad_(True)
            Y_live = _scatter_batched(Z, y_other_live, pv_t, ov_t, ydim)
            h_live = eq_resid_batched_fn(Y_live, X)  # [B, n_eq]
            h = h_live.detach()

            if not torch.isfinite(h).all():
                raise CompletionDivergedError(
                    f"Newton diverged at iter {it}: non-finite residual"
                )
            h_inf = float(h.abs().max().item())
            if h_inf > 1e3:
                raise CompletionDivergedError(
                    f"Newton diverged at iter {it}: ||h||_inf={h_inf:.3e}"
                )
            if h_inf < tol:
                break

            n_eq_actual = h_live.shape[1]
            J = torch.zeros(B, n_eq_actual, n_other, device=device, dtype=dtype)
            for k in range(n_eq_actual):
                g = torch.autograd.grad(
                    h_live[:, k].sum(), y_other_live,
                    retain_graph=(k < n_eq_actual - 1),
                )[0]
                J[:, k, :] = g
            J = J.detach()
        else:
            Y_det = _scatter_batched(
                Z.detach(), y_other.detach(), pv_t, ov_t, ydim,
            )
            h_det = eq_resid_batched_fn(Y_det, X.detach())
            h = h_det.detach()

            if not torch.isfinite(h).all():
                raise CompletionDivergedError(
                    f"Newton diverged at iter {it}: non-finite residual"
                )
            h_inf = float(h.abs().max().item())
            if h_inf > 1e3:
                raise CompletionDivergedError(
                    f"Newton diverged at iter {it}: ||h||_inf={h_inf:.3e}"
                )
            if h_inf < tol:
                break

            partial_list = list(partial_vars)
            other_list = list(other_vars)

            def _resid_other(y_other_single, z_single, x_single):
                y_full = _scatter_per_sample(
                    z_single, y_other_single, pv_t, ov_t, ydim,
                )
                return eq_resid_per_sample_fn(y_full, x_single)

            J = vmap(
                jacrev(_resid_other, argnums=0), in_dims=(0, 0, 0),
            )(y_other.detach(), Z.detach(), X.detach())
            J = J.detach()

        if yf_damping > 0.0 or lm_damping > 0.0:
            # Damp J^T J, not J: mu on the indefinite J can cancel a negative eigenvalue.
            mu = reg + lm_damping + yf_damping * (h * h).sum(dim=-1)  # [B]
            JtJ = J.transpose(1, 2) @ J
            Jth = (J.transpose(1, 2) @ h.unsqueeze(-1))
            delta = torch.linalg.solve(
                JtJ + mu.view(-1, 1, 1) * eye, Jth
            ).squeeze(-1)  # [B, n_other]
        else:
            delta = torch.linalg.solve(
                J + reg * eye, h.unsqueeze(-1)
            ).squeeze(-1)  # [B, n_other]

        if not torch.isfinite(delta).all():
            raise CompletionDivergedError(
                f"Newton diverged at iter {it}: non-finite Newton step"
            )

        y_other = y_other - delta

    return _scatter_batched(Z, y_other, pv_t, ov_t, ydim)


def _linear_complete(
    eq_resid_batched_fn: Callable[[Tensor, Tensor], Tensor],
    X: Tensor,
    Z: Tensor,
    pv_t: Tensor,
    ov_t: Tensor,
    ydim: int,
    reg: float,
    jacobian_mode: str = "loop",
    eq_resid_per_sample_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Closed-form completion for affine ``h(x, y) = J*y - b(x)``.

    ``J = dh/dy`` at ``y=0`` (assumed x-independent), ``b(x) = -h(x, 0)``, and
    ``y_other = J_other^{-1} (b(x) - J_partial * Z)`` per sample.
    """
    B = Z.shape[0]
    device = Z.device
    dtype = Z.dtype

    if jacobian_mode == "loop":
        y_zero_live_batched = torch.zeros(
            B, ydim, device=device, dtype=dtype, requires_grad=True,
        )
        h0_batched_live = eq_resid_batched_fn(y_zero_live_batched, X)  # [B, n_eq]
        n_eq_actual = h0_batched_live.shape[1]
        J_per_batch = torch.zeros(B, n_eq_actual, ydim, device=device, dtype=dtype)
        for k in range(n_eq_actual):
            g = torch.autograd.grad(
                h0_batched_live[:, k].sum(), y_zero_live_batched,
                retain_graph=(k < n_eq_actual - 1),
            )[0]
            J_per_batch[:, k, :] = g
        J = J_per_batch[0].detach()  # affine -> rows are identical
        b_x = -h0_batched_live.detach()
    else:
        if eq_resid_per_sample_fn is None:
            raise ValueError(
                "jacobian_mode='vmap' requires eq_resid_per_sample_fn"
            )
        y_zero = torch.zeros(ydim, device=device, dtype=dtype)
        J = jacrev(eq_resid_per_sample_fn, argnums=0)(y_zero, X[0]).detach()
        h0_batched = eq_resid_batched_fn(
            torch.zeros(B, ydim, device=device, dtype=dtype), X,
        ).detach()
        b_x = -h0_batched

    J_partial = J.index_select(1, pv_t)  # [n_eq, n_partial]
    J_other = J.index_select(1, ov_t)    # [n_eq, n_other], square

    rhs = b_x - Z @ J_partial.T  # [B, n_eq]

    eye = torch.eye(J_other.shape[0], device=device, dtype=dtype)
    Y_other = torch.linalg.solve(
        (J_other + reg * eye).expand(B, -1, -1),
        rhs.unsqueeze(-1),
    ).squeeze(-1)  # [B, n_other]

    return _scatter_batched(Z, Y_other, pv_t, ov_t, ydim)
