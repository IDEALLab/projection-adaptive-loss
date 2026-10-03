"""Parity of e3 ``_safe_ops`` with upstream ml4opf, and vmap compatibility.

ml4opf's ``@torch.jit.script`` primitives fail under ``vmap(grad(...))``.
"""

from __future__ import annotations

import unittest

import torch


class SafeOpsParityTests(unittest.TestCase):
    def setUp(self):
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        torch.manual_seed(0)
        self.bench = E3ACOPF(case="ieee30")
        self.viol = self.bench._violation
        self.data = self.bench._data

        lo, hi = self.bench.spec.output_bounds
        B = 4
        u = torch.rand(B, lo.shape[0])
        self.x = lo + u * (hi - lo)
        q = self.bench.sample_queries(n=B, split="train", seed=0)
        self.conditions = q.conditions

    def _unpack(self):
        pg, qg, vm, va = self.bench._unpack_output(self.x)
        pd, qd = self.bench._unpack_conditions(self.conditions)
        return pg, qg, vm, va, pd, qd

    def test_objective_parity(self):
        from pal.benchmarks.engineering.e3_acopf import _safe_ops

        pg, *_ = self._unpack()
        ref = self.viol.objective(pg)
        got = _safe_ops.objective(pg, self.data["c0"], self.data["c1"], self.data["c2"])
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-5)

    def test_calc_violations_parity(self):
        from pal.benchmarks.engineering.e3_acopf import _safe_ops

        pg, qg, vm, va, pd, qd = self._unpack()
        ref = self.viol.calc_violations(
            pd, qd, pg, qg, vm, va=va, reduction="none", clamp=False,
        )
        got = _safe_ops.calc_violations(pd, qd, pg, qg, vm, va, data=self.data)

        self.assertEqual(set(got.keys()), set(ref.keys()))
        for name in ref:
            torch.testing.assert_close(
                got[name], ref[name], rtol=1e-5, atol=1e-5,
                msg=f"mismatch in {name!r}",
            )

    def test_bench_forward_parity(self):
        """Full ``E3ACOPF.forward`` must produce finite obj + Constraints."""
        obj, cons = self.bench.forward(self.x, self.conditions)
        self.assertEqual(obj.shape, (self.x.shape[0],))
        self.assertTrue(torch.isfinite(obj).all())
        self.assertEqual(len(cons), self.bench.spec.n_eq + self.bench.spec.n_ineq)
        for c in cons:
            self.assertTrue(torch.isfinite(c.value).all())


class SafeOpsVmapTests(unittest.TestCase):
    """forward/constraints/objective compose with vmap(grad) and vmap(jacrev)."""

    def setUp(self):
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        torch.manual_seed(1)
        self.bench = E3ACOPF(case="ieee30")

        lo, hi = self.bench.spec.output_bounds
        B = 3
        u = torch.rand(B, lo.shape[0])
        self.x = lo + u * (hi - lo)
        q = self.bench.sample_queries(n=B, split="train", seed=0)
        self.conditions = q.conditions

    def test_vmap_objective(self):
        """Per-sample objective via vmap, simplest vmap path."""
        from torch.func import vmap

        def single_obj(x_i, c_i):
            return self.bench.objective(x_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

        batched = vmap(single_obj)(self.x, self.conditions)
        ref = self.bench.objective(self.x, self.conditions)
        self.assertEqual(batched.shape, (self.x.shape[0],))
        torch.testing.assert_close(batched, ref, rtol=1e-5, atol=1e-5)

    def test_vmap_jacrev_constraints(self):
        """Per-sample Jacobian via vmap(jacrev(...)), the DC3/FSNet path."""
        from torch.func import jacrev, vmap

        def single_cons(x_i, c_i):
            return self.bench.constraints(x_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

        J = vmap(jacrev(single_cons))(self.x, self.conditions)
        n_con = self.bench.spec.n_eq + self.bench.spec.n_ineq
        self.assertEqual(J.shape, (self.x.shape[0], n_con, self.bench.spec.dim))
        self.assertTrue(torch.isfinite(J).all())

    def test_vmap_grad_objective(self):
        """Per-sample gradient via vmap(grad(...)), the SnareNet/FSNet path."""
        from torch.func import grad, vmap

        def single_obj(x_i, c_i):
            return self.bench.objective(x_i.unsqueeze(0), c_i.unsqueeze(0)).squeeze(0)

        g = vmap(grad(single_obj))(self.x, self.conditions)
        self.assertEqual(g.shape, self.x.shape)
        self.assertTrue(torch.isfinite(g).all())


if __name__ == "__main__":
    unittest.main()
