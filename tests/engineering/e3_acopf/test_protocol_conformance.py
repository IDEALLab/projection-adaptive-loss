"""pal Benchmark protocol conformance + registry presence for E3."""

from __future__ import annotations

import unittest


class ProtocolConformanceTests(unittest.TestCase):
    def test_isinstance_benchmark(self):
        from pal.benchmarks.base import Benchmark
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        bench = E3ACOPF(case="ieee30")
        self.assertIsInstance(bench, Benchmark)

    def test_spec_shape_ieee30(self):
        from pal.benchmarks.base import BenchmarkSpec
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        bench = E3ACOPF(case="ieee30")
        spec = bench.spec
        self.assertIsInstance(spec, BenchmarkSpec)
        self.assertEqual(spec.id, "e3/acopf_ieee30")
        self.assertEqual(spec.family, "e3")
        # IEEE-30: 6 gen + 30 bus -> dim = 2*6 + 2*30 = 72; 60 eq, 248 ineq
        self.assertEqual(spec.dim, 72)
        self.assertEqual(spec.n_eq, 60)
        self.assertGreater(spec.n_ineq, 0)
        self.assertEqual(len(spec.constraint_types), spec.n_eq + spec.n_ineq)
        self.assertEqual(len(spec.constraint_names), spec.n_eq + spec.n_ineq)
        self.assertEqual(spec.condition_dim, 2 * 21)  # 21 loads on IEEE-30
        self.assertEqual(spec.zeta_dim, 8)

    def test_spec_shape_ieee57(self):
        from pal.benchmarks.base import BenchmarkSpec
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF

        bench = E3ACOPF(case="ieee57")
        spec = bench.spec
        self.assertIsInstance(spec, BenchmarkSpec)
        self.assertEqual(spec.id, "e3/acopf_ieee57")
        self.assertEqual(spec.family, "e3")
        # IEEE-57: 7 gen + 57 bus -> dim = 2*7 + 2*57 = 128; 114 eq.
        self.assertEqual(spec.dim, 128)
        self.assertEqual(spec.n_eq, 114)
        self.assertGreater(spec.n_ineq, 0)
        self.assertEqual(len(spec.constraint_types), spec.n_eq + spec.n_ineq)
        self.assertEqual(len(spec.constraint_names), spec.n_eq + spec.n_ineq)
        self.assertEqual(spec.condition_dim, 2 * 42)  # 42 loads on IEEE-57
        self.assertEqual(spec.zeta_dim, 8)

    def test_registered(self):
        from pal.benchmarks import registry

        names = registry.list_all()
        self.assertIn("e3/acopf_ieee30", names)
        self.assertIn("e3/acopf_ieee57", names)
        self.assertIn("e3/acopf_ieee118", names)


if __name__ == "__main__":
    unittest.main()
