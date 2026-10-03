"""pal Benchmark protocol conformance + registry presence for E2."""

from __future__ import annotations

import unittest


class ProtocolConformanceTests(unittest.TestCase):
    def test_isinstance_benchmark(self):
        """Runtime-checkable Protocol validates attribute + method presence."""
        from pal.benchmarks.base import Benchmark
        from pal.benchmarks.engineering.e2_urban_wind.benchmark import (
            E2UrbanWind,
            _make_spec,
        )

        # Bypass __init__ so the real WinDiNet surrogate is not loaded.
        stub = object.__new__(E2UrbanWind)
        stub.spec = _make_spec()
        self.assertIsInstance(stub, Benchmark)

    def test_spec_shape_matches_protocol(self):
        from pal.benchmarks.base import BenchmarkSpec
        from pal.benchmarks.engineering.e2_urban_wind.benchmark import _make_spec

        spec = _make_spec()
        self.assertIsInstance(spec, BenchmarkSpec)
        self.assertEqual(spec.id, "e2/urban_wind")
        self.assertEqual(spec.family, "e2")
        self.assertEqual(spec.dim, 50)
        self.assertEqual(spec.n_eq, 1)  # coverage is a first-class equality
        self.assertEqual(spec.n_ineq, 4)
        self.assertEqual(
            spec.constraint_types,
            ["ineq", "ineq", "ineq", "ineq", "eq"],
        )
        self.assertEqual(
            spec.constraint_names,
            ["site", "clearance", "height", "danger", "coverage"],
        )
        self.assertEqual(spec.condition_dim, 0)
        self.assertEqual(spec.zeta_dim, 16)
        lo, hi = spec.output_bounds
        self.assertEqual(lo.shape, (50,))
        self.assertEqual(hi.shape, (50,))

    def test_registered(self):
        from pal.benchmarks import registry

        self.assertIn("e2/urban_wind", registry.list_all())


if __name__ == "__main__":
    unittest.main()
