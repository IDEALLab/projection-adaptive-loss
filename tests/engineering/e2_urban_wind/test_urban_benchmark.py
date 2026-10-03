#!/usr/bin/env python3
"""Lightweight tests for the urban benchmark core."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from urban_benchmark import CITY_YAML_PATH, UrbanWindBenchmark, inverse_softplus  # noqa: E402


class FakeSurrogate:
    """Cheap stand-in for WinDiNet for geometry/backprop tests."""

    def __call__(self, building_mask, inlet_u, inlet_v):
        batch_size, height, width = building_mask.shape
        frames = 16
        fluid = 1.0 - building_mask
        base = torch.full(
            (batch_size, frames, height, width),
            float(inlet_u.mean().detach().cpu()),
            device=building_mask.device,
            dtype=building_mask.dtype,
        )
        u = base + 8.0 * fluid.unsqueeze(1)
        v = torch.zeros_like(u) + float(inlet_v.mean().detach().cpu())
        return u, v


class UrbanBenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = UrbanWindBenchmark(
            yaml_path=CITY_YAML_PATH,
            device="cpu",
            surrogate=FakeSurrogate(),
            tail_frames=10,
        )

    def _make_raw(self):
        raw = self.benchmark.make_initial_raw_params(device="cpu")
        return {name: value.clone().detach().requires_grad_(True) for name, value in raw.items()}

    def test_decode_positive_extents(self):
        raw = self._make_raw()
        raw["w"].data.fill_(-10.0)
        raw["d"].data.fill_(-12.0)
        raw["h"].data.fill_(-15.0)
        decoded = self.benchmark.decode_raw_params(raw)
        self.assertTrue(torch.all(decoded.w > 0.0))
        self.assertTrue(torch.all(decoded.d > 0.0))
        self.assertTrue(torch.all(decoded.h > 0.0))
        self.assertTrue(torch.allclose(decoded.cz, decoded.h * 0.5))

    def test_site_constraint_rises_when_box_leaves_domain(self):
        raw = self._make_raw()
        baseline = self.benchmark.evaluate_raw_params(raw)
        raw["cx"].data[0, 0] = 0.0
        shifted = self.benchmark.evaluate_raw_params(raw)
        self.assertGreater(
            shifted.constraints["site"][0].item(),
            baseline.constraints["site"][0].item(),
        )

    def test_clearance_constraint_rises_on_collision(self):
        raw = self._make_raw()
        baseline = self.benchmark.evaluate_raw_params(raw)
        raw["cx"].data[0, 0] = raw["cx"].data[0, 1]
        raw["cy"].data[0, 0] = raw["cy"].data[0, 1]
        collided = self.benchmark.evaluate_raw_params(raw)
        self.assertGreater(
            collided.constraints["clearance"][0].item(),
            baseline.constraints["clearance"][0].item(),
        )

    def test_height_constraint_rises_above_cap(self):
        raw = self._make_raw()
        baseline = self.benchmark.evaluate_raw_params(raw)
        raw["h"].data[0, 0] = inverse_softplus(torch.tensor(299.0))
        violated = self.benchmark.evaluate_raw_params(raw)
        self.assertGreater(
            violated.constraints["height"][0].item(),
            baseline.constraints["height"][0].item(),
        )

    def test_danger_constraint_rises_with_faster_tail_mean(self):
        raw = self._make_raw()
        decoded = self.benchmark.decode_raw_params(raw)
        _, building_mask = self.benchmark.rasterize(decoded)

        safe_u = torch.full((1, 16, 256, 256), 10.0)
        fast_u = torch.full((1, 16, 256, 256), 20.0)
        v = torch.zeros_like(safe_u)

        _, _, speed_safe = self.benchmark.compute_tail_means(safe_u, v)
        _, _, speed_fast = self.benchmark.compute_tail_means(fast_u, v)
        c_safe, _ = self.benchmark.danger_constraint(speed_safe, building_mask)
        c_fast, _ = self.benchmark.danger_constraint(speed_fast, building_mask)
        self.assertGreater(c_fast[0].item(), c_safe[0].item())

    def test_coverage_constraint_rises_when_footprints_grow(self):
        raw = self._make_raw()
        baseline = self.benchmark.evaluate_raw_params(raw)
        self.assertGreaterEqual(baseline.coverage_ratio[0].item(), 0.0)
        # Blow widths/depths up well beyond the 0.5 cap (softplus is ~identity here).
        raw["w"].data.fill_(200.0)
        raw["d"].data.fill_(200.0)
        inflated = self.benchmark.evaluate_raw_params(raw)
        self.assertGreater(
            inflated.coverage_ratio[0].item(),
            baseline.coverage_ratio[0].item(),
        )
        self.assertGreater(inflated.constraints["coverage"][0].item(), 0.0)

    def test_coverage_constraint_is_signed_below_target(self):
        """Coverage is an equality: a footprint below target gives a negative residual."""
        raw = self._make_raw()
        raw["w"].data.fill_(-20.0)
        raw["d"].data.fill_(-20.0)
        result = self.benchmark.evaluate_raw_params(raw)
        ratio = result.coverage_ratio[0].item()
        cov = result.constraints["coverage"][0].item()
        self.assertLess(ratio, self.benchmark.max_coverage)
        self.assertLess(cov, 0.0)
        self.assertAlmostEqual(cov, ratio - self.benchmark.max_coverage, places=5)

    def test_backward_is_finite(self):
        raw = self._make_raw()
        result = self.benchmark.evaluate_raw_params(raw)
        loss = result.objective.mean() + result.constraint_tensor().sum()
        loss.backward()
        for tensor in raw.values():
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main()
