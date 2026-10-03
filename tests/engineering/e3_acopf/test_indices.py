"""Pinned bus/gen partition indices of E3ACOPF, used to build the DC3 partition."""

from __future__ import annotations

import unittest


class IEEE30Indices(unittest.TestCase):
    """Pinned indices for IEEE-30: slack bus 0, 5 PV buses (1, 21, 26, 22, 12), 24 PQ, 6 gens."""

    @classmethod
    def setUpClass(cls):
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF
        cls.bench = E3ACOPF(case="ieee30")

    def test_dimensions(self):
        b = self.bench
        self.assertEqual(b._n_bus, 30)
        self.assertEqual(b._n_gen, 6)
        self.assertEqual(b.n_ext_grid, 1)

    def test_y_layout_offsets(self):
        b = self.bench
        self.assertEqual(b.pg_start_yidx, 0)
        self.assertEqual(b.qg_start_yidx, 6)
        self.assertEqual(b.vm_start_yidx, 12)
        self.assertEqual(b.va_start_yidx, 42)
        self.assertEqual(b.spec.dim, 72)  # 2*6 + 2*30
        self.assertEqual(b.va_start_yidx + b._n_bus, b.spec.dim)

    def test_bus_partition_disjoint_and_complete(self):
        b = self.bench
        slack = set(b.slack_bus_idx)
        pv = set(b.pv_bus_idx)
        spv = set(b.spv_bus_idx)

        self.assertEqual(len(slack), 1)
        self.assertEqual(len(pv), 5)
        self.assertEqual(spv, slack | pv)
        self.assertEqual(slack & pv, set())
        self.assertEqual(len(spv), 6)
        # PQ = all buses minus spv -> 24
        pq = set(range(b._n_bus)) - spv
        self.assertEqual(len(pq), 24)

    def test_gen_partition(self):
        b = self.bench
        self.assertEqual(b.slack_gen_idx, [0])
        self.assertEqual(b.pv_gen_idx, [1, 2, 3, 4, 5])
        self.assertEqual(
            sorted(b.slack_gen_idx + b.pv_gen_idx), list(range(b._n_gen))
        )

    def test_dc3_partition_size(self):
        """|partial_vars| = ydim - n_eq for ACOPF (Newton-completable)."""
        b = self.bench
        n_partial = (
            len(b.pv_gen_idx)        # pg at non-slack gens
            + len(b.spv_bus_idx)     # vm at all gen-attached buses
            + len(b.slack_bus_idx)   # va at slack
        )
        self.assertEqual(n_partial, b.spec.dim - b.spec.n_eq)
        # IEEE-30: 5 + 6 + 1 = 12 = 72 - 60
        self.assertEqual(n_partial, 12)


class IEEE57Indices(unittest.TestCase):
    """Pinned indices for IEEE-57: slack bus 0, 6 PV buses (1, 2, 5, 7, 8, 11), 50 PQ, 7 gens."""

    @classmethod
    def setUpClass(cls):
        from pal.benchmarks.engineering.e3_acopf import E3ACOPF
        cls.bench = E3ACOPF(case="ieee57")

    def test_dimensions(self):
        b = self.bench
        self.assertEqual(b._n_bus, 57)
        self.assertEqual(b._n_gen, 7)
        self.assertEqual(b.n_ext_grid, 1)

    def test_y_layout_offsets(self):
        b = self.bench
        self.assertEqual(b.pg_start_yidx, 0)
        self.assertEqual(b.qg_start_yidx, 7)
        self.assertEqual(b.vm_start_yidx, 14)
        self.assertEqual(b.va_start_yidx, 71)
        self.assertEqual(b.spec.dim, 128)  # 2*7 + 2*57
        self.assertEqual(b.va_start_yidx + b._n_bus, b.spec.dim)

    def test_bus_partition_disjoint_and_complete(self):
        b = self.bench
        slack = set(b.slack_bus_idx)
        pv = set(b.pv_bus_idx)
        spv = set(b.spv_bus_idx)

        self.assertEqual(len(slack), 1)
        self.assertEqual(len(pv), 6)
        self.assertEqual(spv, slack | pv)
        self.assertEqual(slack & pv, set())
        self.assertEqual(len(spv), 7)
        # PQ = all buses minus spv -> 50
        pq = set(range(b._n_bus)) - spv
        self.assertEqual(len(pq), 50)

    def test_gen_partition(self):
        b = self.bench
        self.assertEqual(b.slack_gen_idx, [0])
        self.assertEqual(b.pv_gen_idx, [1, 2, 3, 4, 5, 6])
        self.assertEqual(
            sorted(b.slack_gen_idx + b.pv_gen_idx), list(range(b._n_gen))
        )

    def test_dc3_partition_size(self):
        """|partial_vars| = ydim - n_eq for ACOPF (Newton-completable)."""
        b = self.bench
        n_partial = (
            len(b.pv_gen_idx)        # pg at non-slack gens
            + len(b.spv_bus_idx)     # vm at all gen-attached buses
            + len(b.slack_bus_idx)   # va at slack
        )
        self.assertEqual(n_partial, b.spec.dim - b.spec.n_eq)
        # IEEE-57: 6 + 7 + 1 = 14 = 128 - 114
        self.assertEqual(n_partial, 14)


if __name__ == "__main__":
    unittest.main()
