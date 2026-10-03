"""Smoke test: check_env + one forward() returns finite obj + 1 Constraint."""

from __future__ import annotations

import unittest

import torch


class SmokeTests(unittest.TestCase):
    def test_check_env_and_forward(self):
        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout()
        bench.check_env()

        lo, hi = bench.spec.output_bounds
        x = 0.5 * (lo + hi).unsqueeze(0)
        obj, cons = bench.forward(x)

        self.assertEqual(obj.shape, (1,))
        self.assertTrue(torch.isfinite(obj).all())
        self.assertEqual(len(cons), 1)
        self.assertEqual(cons[0].type, "ineq")
        self.assertEqual(cons[0].name, "overlap")
        self.assertTrue(torch.isfinite(cons[0].value).all())


if __name__ == "__main__":
    unittest.main()
