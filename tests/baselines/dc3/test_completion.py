"""Unit tests for ``pal.baselines.dc3._completion.newton_complete``."""

from __future__ import annotations

import unittest

import torch

from pal.baselines.dc3._completion import (
    CompletionDivergedError,
    newton_complete,
)


class LinearClosedForm(unittest.TestCase):
    """Eq: ``A * y = b(x)`` with random ``A`` of shape ``[n_eq, ydim]``."""

    def test_matches_direct_solve(self):
        torch.manual_seed(0)
        ydim, n_eq, B, cond_dim = 5, 2, 8, 3
        partial_vars = [0, 2, 3]
        other_vars = [1, 4]

        A = torch.randn(n_eq, ydim)
        A[:, other_vars] = torch.eye(n_eq) + 0.1 * torch.randn(n_eq, n_eq)
        b_coef = torch.randn(n_eq, cond_dim)

        def eq_resid_batched_fn(Y, X):
            return Y @ A.T - X @ b_coef.T

        X = torch.randn(B, cond_dim)
        Z = torch.randn(B, len(partial_vars))

        Y = newton_complete(
            eq_resid_batched_fn, X, Z,
            partial_vars, other_vars, ydim,
            linear=True,
        )
        self.assertEqual(Y.shape, (B, ydim))

        for b in range(B):
            res = A @ Y[b] - b_coef @ X[b]
            self.assertLess(res.abs().max().item(), 1e-5)

        A_partial = A[:, partial_vars]
        A_other = A[:, other_vars]
        for b in range(B):
            rhs = b_coef @ X[b] - A_partial @ Z[b]
            y_other_expected = torch.linalg.solve(A_other, rhs)
            torch.testing.assert_close(
                Y[b, other_vars], y_other_expected, atol=1e-5, rtol=1e-5,
            )


class NonlinearNewton(unittest.TestCase):
    """Eq: smooth nonlinear system ``y_other_i = sin(y_partial_i)``."""

    def test_converges_in_few_iters(self):
        torch.manual_seed(1)
        ydim, _n_eq, B = 4, 2, 8
        partial_vars = [0, 1]
        other_vars = [2, 3]

        # Eq residual: y[2] - sin(y[0]) = 0, y[3] - sin(y[1]) = 0
        def eq_resid_batched_fn(Y, X):
            return torch.stack(
                [Y[:, 2] - torch.sin(Y[:, 0]), Y[:, 3] - torch.sin(Y[:, 1])],
                dim=-1,
            )

        X = torch.zeros(B, 0)
        Z = torch.randn(B, 2) * 0.5  # small so sin is contractive

        Y = newton_complete(
            eq_resid_batched_fn, X, Z,
            partial_vars, other_vars, ydim,
            linear=False, max_iter=10, tol=1e-8,
        )
        for b in range(B):
            self.assertLess((Y[b, 2] - torch.sin(Y[b, 0])).abs().item(), 1e-7)
            self.assertLess((Y[b, 3] - torch.sin(Y[b, 1])).abs().item(), 1e-7)
        torch.testing.assert_close(Y[:, partial_vars], Z, atol=1e-7, rtol=1e-7)


class WarmStart(unittest.TestCase):
    def test_warm_start_used(self):
        torch.manual_seed(2)
        ydim, B = 3, 4
        partial_vars = [0]
        other_vars = [1, 2]

        # Warm start at the solution of y[1] = y[2] = 1: Newton stops immediately.
        def eq_resid_batched_fn(Y, X):
            return torch.stack([Y[:, 1] - 1.0, Y[:, 2] - 1.0], dim=-1)

        called = {"hit": False}

        def warm(Z, partial, other, ctx):
            called["hit"] = True
            self.assertEqual(partial, [0])
            self.assertEqual(other, [1, 2])
            self.assertIs(ctx["bench"], "stub")
            return torch.ones(Z.shape[0], 2)

        X = torch.zeros(B, 0)
        Z = torch.randn(B, 1)
        Y = newton_complete(
            eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
            warm_start_fn=warm, warm_start_ctx={"bench": "stub"},
        )
        self.assertTrue(called["hit"])
        torch.testing.assert_close(
            Y[:, [1, 2]], torch.ones(B, 2), atol=1e-7, rtol=1e-7,
        )


class VmapParity(unittest.TestCase):
    """``jacobian_mode="vmap"`` matches ``"loop"`` up to fp32 reduction noise."""

    def test_newton_matches_loop(self):
        torch.manual_seed(7)
        ydim, _n_eq, B = 4, 2, 8
        partial_vars = [0, 1]
        other_vars = [2, 3]

        def eq_resid_batched_fn(Y, X):
            return torch.stack(
                [Y[:, 2] - torch.sin(Y[:, 0]), Y[:, 3] - torch.sin(Y[:, 1])],
                dim=-1,
            )

        def eq_resid_per_sample_fn(y, x):
            return torch.stack(
                [y[2] - torch.sin(y[0]), y[3] - torch.sin(y[1])],
            )

        X = torch.zeros(B, 0)
        Z = torch.randn(B, 2) * 0.5

        Y_loop = newton_complete(
            eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
            max_iter=10, tol=1e-8, jacobian_mode="loop",
        )
        Y_vmap = newton_complete(
            eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
            max_iter=10, tol=1e-8, jacobian_mode="vmap",
            eq_resid_per_sample_fn=eq_resid_per_sample_fn,
        )
        torch.testing.assert_close(Y_loop, Y_vmap, atol=1e-6, rtol=1e-6)

    def test_linear_matches_loop(self):
        torch.manual_seed(8)
        ydim, n_eq, B, cond_dim = 5, 2, 8, 3
        partial_vars = [0, 2, 3]
        other_vars = [1, 4]

        A = torch.randn(n_eq, ydim)
        A[:, other_vars] = torch.eye(n_eq) + 0.1 * torch.randn(n_eq, n_eq)
        b_coef = torch.randn(n_eq, cond_dim)

        def eq_resid_batched_fn(Y, X):
            return Y @ A.T - X @ b_coef.T

        def eq_resid_per_sample_fn(y, x):
            return y @ A.T - x @ b_coef.T

        X = torch.randn(B, cond_dim)
        Z = torch.randn(B, len(partial_vars))

        Y_loop = newton_complete(
            eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
            linear=True, jacobian_mode="loop",
        )
        Y_vmap = newton_complete(
            eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
            linear=True, jacobian_mode="vmap",
            eq_resid_per_sample_fn=eq_resid_per_sample_fn,
        )
        torch.testing.assert_close(Y_loop, Y_vmap, atol=1e-5, rtol=1e-5)


class Divergence(unittest.TestCase):
    def test_raises_on_blowup(self):
        ydim, B = 3, 2
        partial_vars = [0]
        other_vars = [1, 2]

        # Eq with no solution at finite y: y[1] = exp(y[1]). Newton blows up.
        def eq_resid_batched_fn(Y, X):
            return torch.stack(
                [Y[:, 1] - torch.exp(Y[:, 1]), Y[:, 2] - torch.exp(Y[:, 2])],
                dim=-1,
            )

        X = torch.zeros(B, 0)
        Z = torch.randn(B, 1)
        with self.assertRaises(CompletionDivergedError) as ctx:
            newton_complete(
                eq_resid_batched_fn, X, Z, partial_vars, other_vars, ydim,
                max_iter=20,
            )
        self.assertIn("Newton diverged", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
