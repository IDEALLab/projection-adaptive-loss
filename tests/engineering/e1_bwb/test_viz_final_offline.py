"""Offline-render path tests for ``E1BWB.visualize_final`` (need live surrogates and pyvista)."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import pytest
import torch

pytest.importorskip("pyvista")
pytest.importorskip("geometry")

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM  # noqa: E402


def _sample_design() -> torch.Tensor:
    x = torch.zeros(1, DIM)
    x[:, 9] = 1.0
    x[:, 10 + 19 + 3] = 0.3
    x[:, 10 + 19 + 4] = 0.3
    x[:, 10 + 19 + 5] = 0.2
    x[:, -1] = torch.deg2rad(torch.tensor(1.0))
    return x


def _sample_conditions() -> torch.Tensor:
    return torch.tensor([[2000.0, 40.0]])


def _make_live_bench() -> E1BWB:
    try:
        return E1BWB(live=True)
    except Exception as exc:
        pytest.skip(f"e1 live surrogates unavailable: {exc}")


class CollectHeroDataTests(unittest.TestCase):
    def test_collect_returns_pickleable_dict(self):
        from pal.benchmarks.engineering.e1_bwb.viz import collect_hero_data

        bench = _make_live_bench()
        x = _sample_design()
        conds = _sample_conditions()

        data = collect_hero_data(bench, x, conds, resolution=32)

        self.assertIn("meshes", data)
        self.assertIn("bwb", data["meshes"])
        self.assertIn("spars", data["meshes"])
        self.assertIn("ribs", data["meshes"])
        self.assertIn("battery_bounds", data)
        self.assertIn("cp_scalar", data)
        self.assertIn("deflection", data)
        self.assertIn("stats", data)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "collect.pt"
            torch.save(data, out)
            loaded = torch.load(out, weights_only=False, map_location="cpu")

        self.assertEqual(
            loaded["meshes"]["bwb"]["points"].shape[1], 3,
        )


class SaveFinalDataTests(unittest.TestCase):
    def test_dump_has_schema_and_required_keys(self):
        from pal.benchmarks.engineering.e1_bwb.benchmark import (
            VIZ_FINAL_DATA_SCHEMA,
        )
        from pal.benchmarks.engineering.e1_bwb.viz import collect_hero_data

        bench = _make_live_bench()
        x = _sample_design()
        conds = _sample_conditions()

        # Hand-roll the dump at low res to skip meshing at res=256.
        payload = collect_hero_data(bench, x, conds, resolution=32)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "viz_final_data.pt"
            full = {
                "schema": VIZ_FINAL_DATA_SCHEMA,
                "x": x.detach().cpu(),
                "conditions": conds.detach().cpu(),
                **payload,
                "metadata": {
                    "benchmark_id": "e1/bwb",
                    "git_sha": "",
                    "timestamp": "test",
                },
            }
            torch.save(full, out)
            self.assertTrue(out.exists())
            data = torch.load(out, weights_only=False, map_location="cpu")

        self.assertEqual(data["schema"], VIZ_FINAL_DATA_SCHEMA)
        for key in (
            "x", "conditions", "meshes", "battery_bounds",
            "cp_scalar", "deflection", "stats", "metadata",
        ):
            self.assertIn(key, data)
        self.assertEqual(data["metadata"]["benchmark_id"], "e1/bwb")
        self.assertIn("R_km", data["stats"])
        self.assertIn("name_to_val", data["stats"])


class EnvTogglePathTests(unittest.TestCase):
    def test_env_var_makes_visualize_final_dump_and_return_none(self):
        bench = _make_live_bench()
        x = _sample_design()
        conds = _sample_conditions()

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "env_dump.pt"
            overrides = {
                "PAL_VIZ_FINAL_DUMP": "1",
                "PAL_VIZ_FINAL_DUMP_PATH": str(out),
                "PAL_VIZ_FINAL_RESOLUTION": "32",
            }
            saved = {k: os.environ.get(k) for k in overrides}
            os.environ.update(overrides)
            try:
                result = bench.visualize_final(x, conds)
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

            self.assertIsNone(result)
            self.assertTrue(out.exists())


class RenderPathTests(unittest.TestCase):
    def test_render_from_dump_returns_figure(self):
        from pal.benchmarks.engineering.e1_bwb.viz import (
            collect_hero_data,
            render_hero_from_data,
        )

        bench = _make_live_bench()
        x = _sample_design()
        conds = _sample_conditions()

        data = collect_hero_data(bench, x, conds, resolution=32)

        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "viz_final_data.pt"
            torch.save(data, out)
            loaded = torch.load(out, weights_only=False, map_location="cpu")

        fig = render_hero_from_data(loaded)

        import matplotlib.figure as mfig
        self.assertIsNotNone(fig)
        self.assertIsInstance(fig, mfig.Figure)


if __name__ == "__main__":
    unittest.main()
