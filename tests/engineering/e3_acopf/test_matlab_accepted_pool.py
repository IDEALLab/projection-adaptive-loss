"""Tests for the MATLAB-accepted pool: generation, save/load, fingerprints, shipped pools."""

from __future__ import annotations

import importlib.resources
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch


class MatlabAcceptedPoolGenerationTests(unittest.TestCase):
    """Fast end-to-end smoke of the generator (target=10 on ieee30)."""

    @classmethod
    def setUpClass(cls):
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            generate_matlab_accepted_pool,
        )
        cls.pool = generate_matlab_accepted_pool(
            "ieee30",
            target_accepted=10,
            train_size=6,
            eval_size=4,
            verbose=False,
        )

    def test_sample_count_and_shapes(self):
        pool = self.pool
        n = pool["metadata"]["n_accepted"]
        self.assertGreaterEqual(n, 10)
        self.assertEqual(pool["pd"].shape[0], n)
        self.assertEqual(pool["qd"].shape[0], n)
        self.assertEqual(pool["pg"].shape[0], n)
        self.assertEqual(pool["qg"].shape[0], n)
        self.assertEqual(pool["vm"].shape[0], n)
        self.assertEqual(pool["va"].shape[0], n)
        fp = pool["metadata"]["case_fingerprint"]
        self.assertEqual(pool["pd"].shape[1], fp["n_load"])
        self.assertEqual(pool["qd"].shape[1], fp["n_load"])
        self.assertEqual(pool["pg"].shape[1], fp["n_gen"])
        self.assertEqual(pool["qg"].shape[1], fp["n_gen"])
        self.assertEqual(pool["vm"].shape[1], fp["n_bus"])
        self.assertEqual(pool["va"].shape[1], fp["n_bus"])

    def test_tensors_finite_and_float32(self):
        for key in ("pd", "qd", "pg", "qg", "vm", "va"):
            t = self.pool[key]
            self.assertEqual(t.dtype, torch.float32, key)
            self.assertTrue(torch.isfinite(t).all().item(), key)

    def test_metadata_shape(self):
        md = self.pool["metadata"]
        self.assertEqual(md["case_name"], "ieee30")
        self.assertFalse(md["thermal_limits_during_solve"])
        self.assertEqual(md["format_version"], 2)
        self.assertEqual(md["solve_vm_bounds"], (0.5, 1.5))
        self.assertEqual(md["split_sizes"], {"train": 6, "eval": 4})
        self.assertEqual(md["split_indices"]["train"].shape, (6,))
        self.assertEqual(md["split_indices"]["eval"].shape, (4,))
        overlap = set(md["split_indices"]["train"].tolist()) & set(
            md["split_indices"]["eval"].tolist()
        )
        self.assertEqual(overlap, set())
        for k in ("MaxChangeLoad", "CorrCoeff", "MIN_PF", "MAX_PF",
                  "sampler_seed", "split_seed"):
            self.assertIn(k, md["matlab_params"])

    def test_violation_diagnostic_keys(self):
        diag = self.pool["metadata"]["violation_diagnostic"]
        for g in ("p_balance", "q_balance",
                  "vm_lower", "vm_upper",
                  "pg_lower", "pg_upper",
                  "qg_lower", "qg_upper",
                  "thrm_1", "thrm_2",
                  "dva_lower", "dva_upper"):
            self.assertIn(g, diag)
            for k in ("any_violated_ratio", "max_residual", "mean_residual"):
                self.assertIn(k, diag[g])

    def test_determinism_under_same_seeds(self):
        """Two generator runs with the same seeds produce bit-identical pools."""
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            generate_matlab_accepted_pool,
        )
        second = generate_matlab_accepted_pool(
            "ieee30", target_accepted=10, train_size=6, eval_size=4, verbose=False,
        )
        torch.testing.assert_close(self.pool["pd"], second["pd"])
        torch.testing.assert_close(self.pool["qd"], second["qd"])
        torch.testing.assert_close(self.pool["pg"], second["pg"])


class MatlabAcceptedPoolRoundTripTests(unittest.TestCase):
    """save_pool / load_pool round-trip + fingerprint enforcement."""

    @classmethod
    def setUpClass(cls):
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            generate_matlab_accepted_pool,
        )
        cls.pool = generate_matlab_accepted_pool(
            "ieee30",
            target_accepted=10,
            train_size=6,
            eval_size=4,
            verbose=False,
        )

    def test_save_load_via_explicit_path(self):
        from pal.benchmarks.engineering.e3_acopf.grid_adapter import (
            pandapower_to_ml4opf,
        )
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            load_pool,
            save_pool,
        )
        with TemporaryDirectory() as d:
            p = Path(d) / "matlab_accepted_pool_ieee30.pt"
            save_pool(self.pool, p)
            live = pandapower_to_ml4opf("ieee30")
            reloaded = load_pool("ieee30", live_data=live, path=p)
        torch.testing.assert_close(reloaded["pd"], self.pool["pd"])
        self.assertEqual(
            reloaded["metadata"]["case_fingerprint"],
            self.pool["metadata"]["case_fingerprint"],
        )

    def test_fingerprint_mismatch_raises(self):
        from pal.benchmarks.engineering.e3_acopf.grid_adapter import (
            pandapower_to_ml4opf,
        )
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            PoolCaseMismatch,
            load_pool,
            save_pool,
        )
        with TemporaryDirectory() as d:
            p = Path(d) / "matlab_accepted_pool_ieee30.pt"
            save_pool(self.pool, p)
            live = pandapower_to_ml4opf("ieee30")
            # Tamper with pd_nominal, any change should flip the sha256.
            live["pd_nominal"] = live["pd_nominal"] + 0.001
            with self.assertRaises(PoolCaseMismatch):
                load_pool("ieee30", live_data=live, path=p)


class MatlabAcceptedPoolPackagingTests(unittest.TestCase):
    """The shipped pools are in package data and load cleanly for all three cases."""

    def _check_case(self, case: str) -> None:
        from pal.benchmarks.engineering.e3_acopf.grid_adapter import (
            pandapower_to_ml4opf,
        )
        from pal.benchmarks.engineering.e3_acopf.matlab_accepted_pool import (
            _pool_filename,
            load_pool,
        )
        # Direct resource lookup catches missing package data.
        pkg_files = importlib.resources.files(
            "pal.benchmarks.engineering.e3_acopf.data"
        )
        res = pkg_files / _pool_filename(case)
        self.assertTrue(
            res.is_file(),
            f"{_pool_filename(case)} missing from package data; "
            "regenerate via scripts/gen_e3_pools.py and check "
            "the pyproject.toml package-data stanza",
        )

        live = pandapower_to_ml4opf(case)
        pool = load_pool(case, live_data=live)
        md = pool["metadata"]
        self.assertEqual(md["case_name"], case)
        self.assertEqual(md["format_version"], 2)
        self.assertGreaterEqual(
            md["n_accepted"],
            md["split_sizes"]["train"] + md["split_sizes"]["eval"],
        )

    def test_ieee30_packaged(self):
        self._check_case("ieee30")

    def test_ieee57_packaged(self):
        self._check_case("ieee57")

    def test_ieee118_packaged(self):
        self._check_case("ieee118")


if __name__ == "__main__":
    unittest.main()
