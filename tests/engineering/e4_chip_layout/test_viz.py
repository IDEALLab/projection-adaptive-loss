"""Tests for e4 visualize_train (matplotlib) and visualize_final (PyVista)."""

from __future__ import annotations

import math
import unittest

import numpy as np
import torch


def _grid_design(bench) -> torch.Tensor:
    """A deterministic non-pathological decision vector: grid layout + square widths."""
    N = bench.n_blocks
    canvas = bench.canvas_size
    g = int(math.ceil(math.sqrt(N)))
    cell = canvas / g
    pos = torch.zeros(1, 2 * N)
    for i in range(N):
        r, c = divmod(i, g)
        pos[0, 2 * i + 0] = (c + 0.5) * cell
        pos[0, 2 * i + 1] = (r + 0.5) * cell
    target_w = bench.area_target.sqrt()
    widths_raw = torch.log(torch.exp(target_w - 0.5) - 1.0).unsqueeze(0)
    x = torch.cat([pos, widths_raw], dim=1)
    lo, hi = bench.spec.output_bounds
    return torch.maximum(torch.minimum(x, hi.unsqueeze(0)), lo.unsqueeze(0))


def _stacked_design(bench) -> torch.Tensor:
    """All blocks piled on the canvas center, maximally infeasible."""
    N = bench.n_blocks
    c = bench.canvas_size / 2
    pos = torch.full((1, 2 * N), c)
    target_w = bench.area_target.sqrt()
    widths_raw = torch.log(torch.exp(target_w - 0.5) - 1.0).unsqueeze(0)
    x = torch.cat([pos, widths_raw], dim=1)
    lo, hi = bench.spec.output_bounds
    return torch.maximum(torch.minimum(x, hi.unsqueeze(0)), lo.unsqueeze(0))


class VisualizeTrainTests(unittest.TestCase):
    def test_returns_matplotlib_figure_with_expected_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout(n_blocks=12, seed=1)
        fig = bench.visualize_train(_grid_design(bench))
        self.assertIsInstance(fig, Figure)
        self.assertEqual(len(fig.axes), 3)

    def test_accepts_1d_input(self):
        import matplotlib
        matplotlib.use("Agg")
        from matplotlib.figure import Figure

        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout(n_blocks=8, seed=2)
        x = _grid_design(bench).squeeze(0)
        fig = bench.visualize_train(x)
        self.assertIsInstance(fig, Figure)

    def test_red_edge_set_matches_overlap_set(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch

        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout
        from pal.benchmarks.engineering.e4_chip_layout.viz import (
            _extract_layout,
            _pairwise_overlap,
        )

        bench = E4ChipLayout(n_blocks=10, seed=3)
        x = _stacked_design(bench)
        g = _extract_layout(bench, x)
        ia, ib, vol = _pairwise_overlap(g["cx"], g["cy"], g["w"], g["h"])
        expected = len(set(ia[vol > 0].tolist()) | set(ib[vol > 0].tolist()))
        self.assertGreater(expected, 0)

        fig = bench.visualize_train(x)
        ax_main = fig.axes[0]
        red_rgb = mcolors.to_rgba("#cc2233")[:3]
        red_count = sum(
            1
            for p in ax_main.patches
            if isinstance(p, FancyBboxPatch)
            and np.allclose(p.get_edgecolor()[:3], red_rgb, atol=1e-3)
        )
        self.assertEqual(red_count, expected)

    def test_no_red_edges_when_no_overlap(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.colors as mcolors
        from matplotlib.patches import FancyBboxPatch

        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout(n_blocks=4, seed=7)
        # Small uniform areas give blocks that cannot collide on a 2x2 grid.
        bench.area_target = torch.full_like(bench.area_target, 4.0)
        fig = bench.visualize_train(_grid_design(bench))

        ax_main = fig.axes[0]
        red_rgb = mcolors.to_rgba("#cc2233")[:3]
        for p in ax_main.patches:
            if isinstance(p, FancyBboxPatch):
                self.assertFalse(
                    np.allclose(p.get_edgecolor()[:3], red_rgb, atol=1e-3)
                )


class VisualizeFinalTests(unittest.TestCase):
    def test_returns_ndarray_or_fallback(self):
        from pal.benchmarks.engineering.e4_chip_layout import E4ChipLayout

        bench = E4ChipLayout(n_blocks=12, seed=5)
        payload = bench.visualize_final(_grid_design(bench))
        self.assertIsInstance(payload, dict)
        self.assertIn("chip_3d", payload)
        img = payload["chip_3d"]
        if isinstance(img, np.ndarray):
            self.assertEqual(img.ndim, 3)
            self.assertIn(img.shape[2], (3, 4))
            self.assertEqual(img.dtype, np.uint8)
        else:
            from matplotlib.figure import Figure
            self.assertIsInstance(img, Figure)


if __name__ == "__main__":
    unittest.main()
