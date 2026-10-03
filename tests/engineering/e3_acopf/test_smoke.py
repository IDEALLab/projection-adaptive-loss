"""Smoke test on IEEE-30 and IEEE-57: check_env and one finite forward()."""

from __future__ import annotations

import unittest

import torch


class SmokeTests(unittest.TestCase):
    def _run(self, case: str) -> None:
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        bench = E3ACOPF(case=case)
        bench.check_env()

        q = bench.sample_queries(n=2, split="train", seed=0)
        self.assertEqual(q.zeta.shape, (2, bench.spec.zeta_dim))
        self.assertEqual(q.conditions.shape, (2, bench.spec.condition_dim))

        lo, hi = bench.spec.output_bounds
        x = 0.5 * (lo + hi).unsqueeze(0).expand(2, -1).contiguous()
        obj, cons = bench.forward(x, q.conditions)

        self.assertEqual(obj.shape, (2,))
        self.assertTrue(torch.isfinite(obj).all())
        self.assertEqual(len(cons), bench.spec.n_eq + bench.spec.n_ineq)
        for c in cons:
            self.assertIn(c.type, ("eq", "ineq"))
            self.assertEqual(c.value.shape, (2,))
            self.assertTrue(torch.isfinite(c.value).all())

    def test_check_env_and_forward_ieee30(self):
        self._run("ieee30")

    def test_check_env_and_forward_ieee57(self):
        self._run("ieee57")


if __name__ == "__main__":
    unittest.main()
