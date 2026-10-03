"""pal Benchmark protocol conformance + registry presence for E4."""

from __future__ import annotations

import unittest


class ProtocolConformanceTests(unittest.TestCase):
    def test_isinstance_benchmark(self):
        from pal.benchmarks.base import Benchmark
        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout()
        self.assertIsInstance(bench, Benchmark)

    def test_spec_shape(self):
        from pal.benchmarks.base import BenchmarkSpec
        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout()  # default 30 blocks
        spec = bench.spec
        self.assertIsInstance(spec, BenchmarkSpec)
        self.assertEqual(spec.id, "e4/chip_layout")
        self.assertEqual(spec.family, "e4")
        self.assertEqual(spec.dim, 3 * 30)
        self.assertEqual(spec.n_eq, 0)
        self.assertEqual(spec.n_ineq, 1)
        self.assertEqual(spec.constraint_names, ["overlap"])
        self.assertEqual(spec.constraint_types, ["ineq"])
        self.assertEqual(spec.condition_dim, 0)
        self.assertEqual(spec.zeta_dim, 8)

    def test_registered(self):
        from pal.benchmarks import registry

        self.assertIn("e4/chip_layout", registry.list_all())


if __name__ == "__main__":
    unittest.main()
