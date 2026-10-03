"""Offline-render path tests for ``E2UrbanWind.visualize_final``."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import torch
from urban_benchmark import CITY_YAML_PATH, UrbanWindBenchmark  # noqa: E402


class FakeSurrogate:
    """Same no-HF stub used in test_smoke.py."""

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


def _make_bench():
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


def _make_x(bench):
    raw = bench._bench.make_initial_raw_params(batch_size=1, device="cpu")
    return torch.cat(
        [raw["cx"], raw["cy"], raw["w"], raw["d"], raw["h"]], dim=1,
    )


class SaveFinalDataTests(unittest.TestCase):
    def test_dump_has_schema_and_required_keys(self):
        from pal.benchmarks.engineering.e2_urban_wind.benchmark import (
            _N_BUILDINGS,
            VIZ_FINAL_DATA_SCHEMA,
        )

        bench = _make_bench()
        x = _make_x(bench)
        bench.forward(x)  # populate _last_result

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "viz_final_data.pt"
            bench.save_final_data(out, x, conditions=None)
            self.assertTrue(out.exists())

            data = torch.load(out, weights_only=False, map_location="cpu")

        self.assertEqual(data["schema"], VIZ_FINAL_DATA_SCHEMA)
        for key in (
            "x", "conditions",
            "tail_mean_speed", "tail_mean_u", "tail_mean_v",
            "building_mask",
            "decoded_cx", "decoded_cy", "decoded_w", "decoded_d", "decoded_h",
            "hard_danger_fraction", "constraint_vec",
            "total_volume", "program", "metadata",
        ):
            self.assertIn(key, data, f"missing key: {key}")

        self.assertEqual(data["tail_mean_speed"].shape, (256, 256))
        self.assertEqual(data["building_mask"].shape, (256, 256))
        self.assertEqual(data["decoded_cx"].shape, (_N_BUILDINGS,))
        self.assertEqual(data["constraint_vec"].shape, (5,))
        self.assertEqual(data["metadata"]["benchmark_id"], "e2/urban_wind")
        for pk in ("config", "params", "batch_size"):
            self.assertIn(pk, data["program"])


class EnvTogglePathTests(unittest.TestCase):
    def test_env_var_makes_visualize_final_dump_and_return_none(self):
        bench = _make_bench()
        x = _make_x(bench)
        bench.forward(x)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env_dump.pt"
            old_flag = os.environ.get("PAL_VIZ_FINAL_DUMP")
            old_path = os.environ.get("PAL_VIZ_FINAL_DUMP_PATH")
            os.environ["PAL_VIZ_FINAL_DUMP"] = "1"
            os.environ["PAL_VIZ_FINAL_DUMP_PATH"] = str(out)
            try:
                result = bench.visualize_final(x)
            finally:
                if old_flag is None:
                    os.environ.pop("PAL_VIZ_FINAL_DUMP", None)
                else:
                    os.environ["PAL_VIZ_FINAL_DUMP"] = old_flag
                if old_path is None:
                    os.environ.pop("PAL_VIZ_FINAL_DUMP_PATH", None)
                else:
                    os.environ["PAL_VIZ_FINAL_DUMP_PATH"] = old_path

            self.assertIsNone(result)
            self.assertTrue(out.exists())


class RenderPathTests(unittest.TestCase):
    def test_render_from_dump_returns_figures(self):
        import pytest
        pytest.importorskip("pyvista")
        pytest.importorskip("vtk")

        from pal.benchmarks.engineering.e2_urban_wind.benchmark import (
            _render_final_pyvista,
        )

        bench = _make_bench()
        x = _make_x(bench)
        bench.forward(x)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "viz_final_data.pt"
            bench.save_final_data(out, x, conditions=None)
            data = torch.load(out, weights_only=False, map_location="cpu")

        figs = _render_final_pyvista(
            program_config=data["program"]["config"],
            program_params=data["program"]["params"],
            program_batch_size=data["program"]["batch_size"],
            tail_mean_speed=data["tail_mean_speed"],
            tail_mean_u=data["tail_mean_u"],
            tail_mean_v=data["tail_mean_v"],
            building_mask=data["building_mask"],
            decoded_cx=data["decoded_cx"],
            decoded_cy=data["decoded_cy"],
            decoded_w=data["decoded_w"],
            decoded_d=data["decoded_d"],
            decoded_h=data["decoded_h"],
            hard_danger_fraction=data["hard_danger_fraction"],
            constraint_vec=data["constraint_vec"],
            total_volume=data["total_volume"],
        )

        import matplotlib.figure as mfig
        self.assertIsNotNone(figs)
        self.assertIn("3d_wind", figs)
        self.assertIn("3d_comfort", figs)
        self.assertIsInstance(figs["3d_wind"], mfig.Figure)
        self.assertIsInstance(figs["3d_comfort"], mfig.Figure)


if __name__ == "__main__":
    unittest.main()
