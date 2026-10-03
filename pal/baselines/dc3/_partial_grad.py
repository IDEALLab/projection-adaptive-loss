"""Implicit-function gradient for DC3's partial-correction step.

Returns a full ``[B, ydim]`` vector: the other slots carry the induced motion
that keeps ``h(x, y) = 0`` as ``y_partial`` moves (upstream ``utils.py:896-908``):

    eq_jac        = dh/dy at current Y        shape [B, n_eq, ydim]
    dynz_dz       = -J_eq_other^-1 * J_eq_partial   [B, n_other, n_partial]
    direct_grad   = d ||ineq_dist||^2 / dy            [B, ydim]
    indirect      = dynz_dz.T * direct_grad[other] [B, n_partial]
    full_partial  = indirect + direct_grad[partial][B, n_partial]
    induced_other = dynz_dz * full_partial         [B, n_other]
    out[partial]  = full_partial
    out[other]    = induced_other

The ``direct_grad`` term is the autograd of ``||clamp(g, 0)||^2`` w.r.t.
``y``; by the chain rule through ``clamp`` it equals
``2 * J_ineq^T * clamp(g, 0)`` (i.e. DC3's ``ineq_grad``).
"""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torch import Tensor
from torch.func import grad as func_grad, jacrev, vmap


def ineq_partial_grad_generic(
    eq_resid_batched_fn: Callable[[Tensor, Tensor], Tensor],
    ineq_dist_batched_fn: Callable[[Tensor, Tensor], Tensor],
    X: Tensor,
    Y: Tensor,
    partial_vars: Sequence[int],
    other_vars: Sequence[int],
    ydim: int,
    *,
    reg: float = 1e-8,
    known_vars: Sequence[int] = (),
    jacobian_mode: str = "loop",
    eq_resid_per_sample_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
    ineq_dist_per_sample_fn: Callable[[Tensor, Tensor], Tensor] | None = None,
) -> Tensor:
    """Return ``[B, ydim]`` full-vector DC3 partial-correction gradient.

    Args:
        eq_resid_batched_fn: ``(Y[B, ydim], X[B, cond_dim]) -> [B, n_eq]``.
        ineq_dist_batched_fn: ``(Y[B, ydim], X[B, cond_dim]) -> [B, n_ineq]``
            clamped >= 0.
        X: ``[B, cond_dim]`` conditions.
        Y: ``[B, ydim]`` current iterate, assumed on the eq manifold.
        partial_vars: y-vector indices for NN-predicted entries.
        other_vars: y-vector indices completed via eq solve (``len == n_eq``).
        ydim: Total y dim.
        reg: Tikhonov for ``J_eq_other`` solve.
        known_vars: Optional pinned y-vector indices; the gradient is zero there.

    Returns:
        Full gradient ``[B, ydim]`` over partial and other slots.
    """
    B = Y.shape[0]
    n_partial = len(partial_vars)
    n_other = len(other_vars)
    n_known = len(known_vars)
    if n_partial + n_other + n_known != ydim:
        raise ValueError(
            f"partition mismatch: |partial|={n_partial} + |other|={n_other} "
            f"+ |known|={n_known} != ydim={ydim}"
        )

    if jacobian_mode not in ("loop", "vmap"):
        raise ValueError(
            f"unknown jacobian_mode {jacobian_mode!r}; expected 'loop' or 'vmap'"
        )

    device = Y.device
    dtype = Y.dtype
    pv_t = torch.as_tensor(list(partial_vars), dtype=torch.long, device=device)
    ov_t = torch.as_tensor(list(other_vars), dtype=torch.long, device=device)

    if jacobian_mode == "loop":
        # enable_grad: upstream grad_steps_all runs under no_grad.
        with torch.enable_grad():
            Y_live_eq = Y.detach().requires_grad_(True)
            h_live = eq_resid_batched_fn(Y_live_eq, X)  # [B, n_eq]
            n_eq_actual = h_live.shape[1]
            eq_jac = torch.zeros(B, n_eq_actual, ydim, device=device, dtype=dtype)
            for k in range(n_eq_actual):
                g = torch.autograd.grad(
                    h_live[:, k].sum(), Y_live_eq,
                    retain_graph=(k < n_eq_actual - 1),
                )[0]
                eq_jac[:, k, :] = g
            eq_jac = eq_jac.detach()

            # Grad of the summed loss equals the per-sample grads (samples independent).
            Y_live_ineq = Y.detach().requires_grad_(True)
            ineq_vals = ineq_dist_batched_fn(Y_live_ineq, X)  # [B, n_ineq]
            if ineq_vals.shape[1] == 0:
                direct_grad = torch.zeros(B, ydim, device=device, dtype=dtype)
            else:
                loss_scalar = ineq_vals.pow(2).sum()
                direct_grad = torch.autograd.grad(
                    loss_scalar, Y_live_ineq,
                )[0].detach()
    else:
        if eq_resid_per_sample_fn is None or ineq_dist_per_sample_fn is None:
            raise ValueError(
                "jacobian_mode='vmap' requires eq_resid_per_sample_fn and "
                "ineq_dist_per_sample_fn"
            )
        eq_jac = vmap(
            jacrev(eq_resid_per_sample_fn, argnums=0),
            in_dims=(0, 0),
        )(Y.detach(), X.detach()).detach()
        n_eq_actual = eq_jac.shape[1]

        def _ineq_loss_single(y_single, x_single):
            return ineq_dist_per_sample_fn(y_single, x_single).pow(2).sum()

        probe_ineq = ineq_dist_per_sample_fn(Y[0].detach(), X[0].detach())
        if probe_ineq.shape[0] == 0:
            direct_grad = torch.zeros(B, ydim, device=device, dtype=dtype)
        else:
            direct_grad = vmap(
                func_grad(_ineq_loss_single, argnums=0),
                in_dims=(0, 0),
            )(Y.detach(), X.detach()).detach()

    eq_jac_other = eq_jac.index_select(2, ov_t)    # [B, n_eq, n_other]
    eq_jac_partial = eq_jac.index_select(2, pv_t)  # [B, n_eq, n_partial]

    # dynz_dz = -(J_other)^{-1} * J_partial
    eye = torch.eye(n_other, device=device, dtype=dtype).expand(B, -1, -1)
    dynz_dz = -torch.linalg.solve(
        eq_jac_other + reg * eye, eq_jac_partial
    )  # [B, n_other, n_partial]

    grad_other = direct_grad.index_select(1, ov_t)      # [B, n_other]
    grad_partial = direct_grad.index_select(1, pv_t)    # [B, n_partial]

    indirect = dynz_dz.transpose(1, 2).bmm(
        grad_other.unsqueeze(-1)
    ).squeeze(-1)  # [B, n_partial]
    full_partial = indirect + grad_partial  # [B, n_partial]

    induced_other = dynz_dz.bmm(
        full_partial.unsqueeze(-1)
    ).squeeze(-1)  # [B, n_other]

    out = torch.zeros(B, ydim, device=device, dtype=dtype)
    out.index_copy_(1, pv_t, full_partial)
    out.index_copy_(1, ov_t, induced_other)
    return out
