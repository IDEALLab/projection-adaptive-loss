"""``ineq_partial_grad_generic`` vs the upstream DC3 closed form on a linear toy.

Other slots must carry the induced tangent motion, not zeros.
"""

from __future__ import annotations

import unittest

import torch

from pal.baselines.dc3._partial_grad import ineq_partial_grad_generic


def _upstream_linear_ref(
    A: torch.Tensor,  # [n_eq, ydim]
    G: torch.Tensor,  # [n_ineq, ydim]
    h: torch.Tensor,  # [n_ineq]
    X: torch.Tensor,  # [B, n_eq]
    Y: torch.Tensor,  # [B, ydim]
    partial_vars: list[int],
    other_vars: list[int],
) -> torch.Tensor:
    """Transcription of DC3 utils.py:237-244 (SimpleProblem.ineq_partial_grad).

    Eq form: ``A * y = x`` (X serves as eq RHS). Ineq: ``G * y <= h``.
    """
    A_partial = A[:, partial_vars]
    A_other = A[:, other_vars]
    A_other_inv = torch.linalg.inv(A_other)
    ydim = A.shape[1]
    B = X.shape[0]

    G_effective = G[:, partial_vars] - G[:, other_vars] @ (A_other_inv @ A_partial)
    h_effective = h - (X @ A_other_inv.T) @ G[:, other_vars].T
    grad = 2 * torch.clamp(
        Y[:, partial_vars] @ G_effective.T - h_effective, min=0.0,
    ) @ G_effective
    out = torch.zeros(B, ydim)
    out[:, partial_vars] = grad
    out[:, other_vars] = -(grad @ A_partial.T) @ A_other_inv.T
    return out


class P13Regression(unittest.TestCase):
    def test_matches_upstream_linear_form(self):
        """Both partial and other slots match the upstream reference."""
        torch.manual_seed(0)
        ydim, n_eq, n_ineq, B = 6, 2, 4, 10
        partial_vars = [0, 1, 3, 5]
        other_vars = [2, 4]
        assert len(partial_vars) + len(other_vars) == ydim
        assert len(other_vars) == n_eq

        A = torch.randn(n_eq, ydim)
        A[:, other_vars] = torch.eye(n_eq) + 0.1 * torch.randn(n_eq, n_eq)
        G = torch.randn(n_ineq, ydim) * 0.5
        h = torch.randn(n_ineq) * 0.1

        X = torch.randn(B, n_eq)

        # Build Y that is ON the eq manifold: y_other = A_other^{-1}(X - A_partial Z)
        Z = torch.randn(B, len(partial_vars))
        A_partial = A[:, partial_vars]
        A_other_inv = torch.linalg.inv(A[:, other_vars])
        Y = torch.zeros(B, ydim)
        Y[:, partial_vars] = Z
        Y[:, other_vars] = (X - Z @ A_partial.T) @ A_other_inv.T

        self.assertLess(
            ((Y @ A.T) - X).abs().max().item(), 1e-5,
            "test setup broke: Y not on eq manifold",
        )

        def eq_resid_batched_fn(Y_in, X_in):
            return Y_in @ A.T - X_in  # [B, n_eq]

        def ineq_dist_batched_fn(Y_in, X_in):
            return torch.clamp(Y_in @ G.T - h, min=0.0)  # [B, n_ineq]

        out = ineq_partial_grad_generic(
            eq_resid_batched_fn, ineq_dist_batched_fn,
            X, Y, partial_vars, other_vars, ydim, reg=0.0,
        )
        ref = _upstream_linear_ref(A, G, h, X, Y, partial_vars, other_vars)

        self.assertEqual(out.shape, ref.shape)

        torch.testing.assert_close(
            out[:, partial_vars], ref[:, partial_vars],
            atol=1e-5, rtol=1e-5,
            msg="partial-slot gradient disagrees with upstream linear form",
        )

        # Other slots carry the induced motion, not zero.
        torch.testing.assert_close(
            out[:, other_vars], ref[:, other_vars],
            atol=1e-5, rtol=1e-5,
            msg="other-slot induced motion disagrees with upstream",
        )

        self.assertGreater(
            out[:, other_vars].abs().max().item(), 1e-4,
            "other-slot induced motion must be nonzero",
        )


class VmapParity(unittest.TestCase):
    """``jacobian_mode="vmap"`` matches ``"loop"`` on the same inputs."""

    def test_vmap_matches_loop(self):
        torch.manual_seed(3)
        ydim, n_eq, n_ineq, B = 6, 2, 4, 10
        partial_vars = [0, 1, 3, 5]
        other_vars = [2, 4]

        A = torch.randn(n_eq, ydim)
        A[:, other_vars] = torch.eye(n_eq) + 0.1 * torch.randn(n_eq, n_eq)
        G = torch.randn(n_ineq, ydim) * 0.5
        h = torch.randn(n_ineq) * 0.1

        X = torch.randn(B, n_eq)
        Z = torch.randn(B, len(partial_vars))
        A_partial = A[:, partial_vars]
        A_other_inv = torch.linalg.inv(A[:, other_vars])
        Y = torch.zeros(B, ydim)
        Y[:, partial_vars] = Z
        Y[:, other_vars] = (X - Z @ A_partial.T) @ A_other_inv.T

        def eq_resid_batched_fn(Y_in, X_in):
            return Y_in @ A.T - X_in

        def ineq_dist_batched_fn(Y_in, X_in):
            return torch.clamp(Y_in @ G.T - h, min=0.0)

        def eq_resid_per_sample_fn(y, x):
            return y @ A.T - x

        def ineq_dist_per_sample_fn(y, x):
            return torch.clamp(y @ G.T - h, min=0.0)

        out_loop = ineq_partial_grad_generic(
            eq_resid_batched_fn, ineq_dist_batched_fn,
            X, Y, partial_vars, other_vars, ydim, reg=0.0,
            jacobian_mode="loop",
        )
        out_vmap = ineq_partial_grad_generic(
            eq_resid_batched_fn, ineq_dist_batched_fn,
            X, Y, partial_vars, other_vars, ydim, reg=0.0,
            jacobian_mode="vmap",
            eq_resid_per_sample_fn=eq_resid_per_sample_fn,
            ineq_dist_per_sample_fn=ineq_dist_per_sample_fn,
        )
        torch.testing.assert_close(out_loop, out_vmap, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
