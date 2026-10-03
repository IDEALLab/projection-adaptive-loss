"""pal Benchmark protocol conformance and registry presence for E1 BWB."""

from __future__ import annotations

import unittest

from pal.benchmarks.engineering.e1_bwb.spec import (
    ZETA_DIM,
    make_spec,
)


class ProtocolConformanceTests(unittest.TestCase):
    def test_isinstance_benchmark(self):
        """Runtime-checkable Protocol validates attribute + method presence."""
        from pal.benchmarks.base import Benchmark
        from pal.benchmarks.engineering.e1_bwb.benchmark import E1BWB

        # Bypass __init__ so the real surrogates are not loaded.
        stub = object.__new__(E1BWB)
        stub.spec = make_spec()
        self.assertIsInstance(stub, Benchmark)

    def test_spec_shape_matches_protocol(self):
        from pal.benchmarks.base import BenchmarkSpec

        spec = make_spec()
        self.assertIsInstance(spec, BenchmarkSpec)
        self.assertEqual(spec.id, "e1/bwb")
        self.assertEqual(spec.family, "e1")
        self.assertEqual(spec.variant, "bwb")
        self.assertEqual(spec.dim, 36)
        self.assertEqual(spec.n_eq, 1)  # lift_balance is a first-class equality
        self.assertEqual(spec.n_ineq, 2)  # mean(ReLU) strain aggregate
        self.assertEqual(spec.constraint_types, ["eq", "ineq", "ineq"])
        self.assertEqual(
            spec.constraint_names,
            ["lift_balance", "strain_agg", "tip_deflection"],
        )
        self.assertEqual(spec.condition_dim, 2)
        self.assertEqual(spec.zeta_dim, ZETA_DIM)
        lo, hi = spec.output_bounds
        self.assertEqual(lo.shape, (36,))
        self.assertEqual(hi.shape, (36,))

    def test_registered(self):
        from pal.benchmarks import registry

        self.assertIn("e1/bwb", registry.list_all())


if __name__ == "__main__":
    unittest.main()
