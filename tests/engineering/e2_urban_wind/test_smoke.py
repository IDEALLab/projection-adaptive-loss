"""One forward pass with ``FakeSurrogate`` gives a finite objective and 5 Constraints."""

from __future__ import annotations

import unittest

import torch
from urban_benchmark import CITY_YAML_PATH, UrbanWindBenchmark  # noqa: E402


class FakeSurrogate:
    def __call__(self, building_mask, inlet_u, inlet_v):
        B, H, W = building_mask.shape
        F = 16
        fluid = 1.0 - building_mask
        u = torch.full(
            (B, F, H, W), float(inlet_u.mean().detach().cpu()),
            device=building_mask.device, dtype=building_mask.dtype,
        ) + 8.0 * fluid.unsqueeze(1)
        v = torch.zeros_like(u) + float(inlet_v.mean().detach().cpu())
        return u, v


class SmokeTests(unittest.TestCase):
    def _make_bench(self):
        """Build E2UrbanWind without loading the real surrogate."""
        from pal.benchmarks.engineering.e2_urban_wind.benchmark import (
            _CONSTRAINT_NAMES,
            _DEFAULT_CONSTRAINT_SCALES,
            E2UrbanWind,
            _kept_constraint_names,
            _make_spec,
        )

        bench = object.__new__(E2UrbanWind)
        bench.spec = _make_spec()
        bench.device = "cpu"
        bench._diffusion_dir = None
        bench._bench = UrbanWindBenchmark(
            yaml_path=CITY_YAML_PATH,
            device="cpu",
            surrogate=FakeSurrogate(),
        )
        bench._batch_diag_printed = False
        bench._last_result = None
        bench.objective_scale = 1.0
        bench._kept_names = _kept_constraint_names()
        bench._kept_indices = [_CONSTRAINT_NAMES.index(n) for n in bench._kept_names]
        bench._constraint_scale_vec = torch.tensor(
            [_DEFAULT_CONSTRAINT_SCALES[n] for n in bench._kept_names],
            dtype=torch.float32,
        )
        bench._constraint_scales = {n: _DEFAULT_CONSTRAINT_SCALES[n] for n in bench._kept_names}
        return bench

    def test_forward_returns_finite(self):
        bench = self._make_bench()
        raw = bench._bench.make_initial_raw_params(batch_size=1, device="cpu")
        x = torch.cat([raw["cx"], raw["cy"], raw["w"], raw["d"], raw["h"]], dim=1)
        obj, cons = bench.forward(x)

        self.assertEqual(obj.shape, (1,))
        self.assertTrue(torch.isfinite(obj).all())
        self.assertEqual(len(cons), 5)
        by_name = {c.name: c for c in cons}
        # coverage is a first-class equality (tol > 0); the rest stay ineq.
        self.assertEqual(by_name["coverage"].type, "eq")
        self.assertTrue((by_name["coverage"].tol > 0).all())
        for name in ("site", "clearance", "height", "danger"):
            self.assertEqual(by_name[name].type, "ineq")
        for c in cons:
            self.assertEqual(c.value.shape, (1,))
            self.assertTrue(torch.isfinite(c.value).all())

    def test_check_env_runs_after_construction(self):
        bench = self._make_bench()
        self.assertIsNone(bench.check_env())


if __name__ == "__main__":
    unittest.main()
