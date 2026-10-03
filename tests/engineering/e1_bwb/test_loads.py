"""Smoke tests for FiLM-based compute_loads (skipped without the checkpoints)."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks.engineering.e1_bwb import DIM
from pal.benchmarks.engineering.e1_bwb._artifacts import ensure_artifact_path
from pal.benchmarks.engineering.e1_bwb.interfaces import Loads
from pal.benchmarks.engineering.e1_bwb.loads import FiLMLoads, build_bwb_program
from pal.benchmarks.engineering.e1_bwb.x_layout import decode_x


def _all_artefacts_present() -> bool:
    for name in ("a_aero_norm_stats", "film_weights", "film_norm_stats", "bwb_sdf_weights"):
        try:
            ensure_artifact_path(name)
        except (FileNotFoundError, KeyError, RuntimeError):
            return False
    return True


pytestmark = pytest.mark.skipif(
    not _all_artefacts_present(),
    reason="BWB SDF + FiLM + A_aero artifacts required"
    " (shipped in pal/benchmarks/engineering/e1_bwb/_artifacts_data/)",
)


def _sample_design(B: int = 2, symmetric: bool = True) -> torch.Tensor:
    """Mid-bounds nominal. `symmetric=True` means all batch rows share x."""
    x = torch.zeros(B, DIM)
    x[:, 9] = 1.0
    x[:, 10 + 19 + 3] = 0.3
    x[:, 10 + 19 + 4] = 0.3
    x[:, 10 + 19 + 5] = 0.2
    x[:, -1] = math.radians(1.5)
    if not symmetric:
        x[1, :9] += 0.2
    return x


def _sample_conditions(B: int = 2) -> torch.Tensor:
    return torch.stack(
        [torch.full((B,), 2000.0), torch.full((B,), 40.0)], dim=-1,
    )


def _sobol_stations(B: int, N: int, semi_span: torch.Tensor) -> torch.Tensor:
    """Cheap, deterministic spanwise stations in [0.01*semi, 0.99*semi]."""
    t = torch.linspace(0.01, 0.99, N)
    return semi_span.unsqueeze(-1) * t


@pytest.fixture(scope="module")
def loads_module():
    return FiLMLoads.load_default(
        device="cpu", isocontour_base_res=4, isocontour_levels=3,
    )


def test_compute_loads_shapes_and_finite(loads_module):
    B, N = 2, 3
    x = _sample_design(B=B, symmetric=True)
    conds = _sample_conditions(B=B)
    dec = decode_x(x)
    prog = build_bwb_program(dec, device="cpu")

    semi_span = 0.5 * dec.L.squeeze(-1)
    y_stations = _sobol_stations(B, N, semi_span)
    loads = loads_module(prog, dec, conds, y_stations)

    assert isinstance(loads, Loads)
    for t_ in (loads.q_z, loads.m, loads.x_cp):
        assert t_.shape == (B, N)
        assert torch.isfinite(t_).all()


def test_compute_loads_symmetric_batches_match(loads_module):
    """Same x in both batch rows -> identical q_z, m, x_cp per matching station."""
    B, N = 2, 3
    x = _sample_design(B=B, symmetric=True)
    conds = _sample_conditions(B=B)
    dec = decode_x(x)
    prog = build_bwb_program(dec, device="cpu")
    semi_span = 0.5 * dec.L.squeeze(-1)
    y_stations = _sobol_stations(B, N, semi_span)

    loads = loads_module(prog, dec, conds, y_stations)
    assert torch.allclose(loads.q_z[0], loads.q_z[1], atol=1e-4, rtol=1e-4)
    assert torch.allclose(loads.m[0], loads.m[1], atol=1e-4, rtol=1e-4)
    assert torch.allclose(loads.x_cp[0], loads.x_cp[1], atol=1e-4, rtol=1e-4)


def test_compute_loads_x_cp_in_plausible_range(loads_module):
    """Physical x_cp at L=1 lies in a sensible chordwise band."""
    B, N = 1, 3
    x = _sample_design(B=B)
    conds = _sample_conditions(B=B)
    dec = decode_x(x)
    prog = build_bwb_program(dec, device="cpu")
    semi_span = 0.5 * dec.L.squeeze(-1)
    y_stations = _sobol_stations(B, N, semi_span)

    loads = loads_module(prog, dec, conds, y_stations)
    # At L=1 m, physical x-extent of the BWB unit frame is [0, ~1.16] m.
    assert (loads.x_cp > -2.0).all() and (loads.x_cp < 3.0).all()


def test_compute_loads_gradient_flows_through_x(loads_module):
    """Grad from `loads.q_z.sum()` back to `x` is finite and non-trivial."""
    B, N = 1, 2
    x = _sample_design(B=B).requires_grad_(True)
    conds = _sample_conditions(B=B)
    dec = decode_x(x)
    prog = build_bwb_program(dec, device="cpu")
    semi_span = 0.5 * dec.L.squeeze(-1)
    y_stations = _sobol_stations(B, N, semi_span)

    loads = loads_module(prog, dec, conds, y_stations)
    loads.q_z.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().max() > 0.0


def test_shape_roundtrip_conversions():
    """`x.shape in [-1, 1]^9` <-> raw mm/deg <-> SDF ratios <-> FiLM 10-dim."""
    from pal.benchmarks.engineering.e1_bwb.loads import (
        _shape_raw_to_film_10,
        _shape_raw_to_sdf_ratio,
        _unnormalise_x_shape,
    )

    x_shape = torch.zeros(3, 9)  # mid-range
    x_shape[1] = -1.0
    x_shape[2] = 1.0
    raw = _unnormalise_x_shape(x_shape)
    # Row 1 should equal shape_min; row 2 shape_max.
    from pal.benchmarks.engineering.e1_bwb.loads import _load_aaero_shape_ranges
    lo, hi = _load_aaero_shape_ranges()
    assert torch.allclose(raw[1], lo, atol=1e-5)
    assert torch.allclose(raw[2], hi, atol=1e-5)

    ratios = _shape_raw_to_sdf_ratio(raw)
    # B1-C4 columns (indices 0-5) must be mm / 1000.
    assert torch.allclose(ratios[:, :6], raw[:, :6] / 1000.0)
    assert torch.equal(ratios[:, 6:], raw[:, 6:])

    shape_10 = _shape_raw_to_film_10(raw)
    assert shape_10.shape == (3, 10)
    # C1 slot at index 3 must be 1000 mm.
    assert torch.allclose(shape_10[:, 3], torch.full((3,), 1000.0))
    assert torch.equal(shape_10[:, :3], raw[:, :3])
    assert torch.equal(shape_10[:, 4:7], raw[:, 3:6])
