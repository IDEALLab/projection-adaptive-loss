"""Smoke tests for the A_aero surrogate."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pal.benchmarks.engineering.e1_bwb import atmosphere
from pal.benchmarks.engineering.e1_bwb.a_aero import AAeroSurrogate
from pal.benchmarks.engineering.e1_bwb.interfaces import ComputeAero
from pal.benchmarks.engineering.e1_bwb.x_layout import DIM, decode_x, default_bounds

ALT_RANGE = (0.0, 8000.0)
V_RANGE = (20.0, 80.0)


def _random_design(B: int, seed: int = 0) -> torch.Tensor:
    lo, hi = default_bounds()
    g = torch.Generator().manual_seed(seed)
    u = torch.rand((B, DIM), generator=g)
    return lo + u * (hi - lo)


def _random_conditions(B: int, seed: int = 1) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    u = torch.rand((B, 2), generator=g)
    alt = ALT_RANGE[0] + u[:, 0] * (ALT_RANGE[1] - ALT_RANGE[0])
    V = V_RANGE[0] + u[:, 1] * (V_RANGE[1] - V_RANGE[0])
    return torch.stack([alt, V], dim=-1)


@pytest.fixture(scope="module")
def model() -> AAeroSurrogate:
    return AAeroSurrogate.load_default()


def test_protocol_conformance(model: AAeroSurrogate) -> None:
    assert isinstance(model, ComputeAero)


def test_forward_shapes_and_dtype(model: AAeroSurrogate) -> None:
    B = 4
    x_vec = _random_design(B)
    x = decode_x(x_vec)
    conds = _random_conditions(B)
    CL, CD, CM = model(x, conds)
    assert CL.shape == (B,)
    assert CD.shape == (B,)
    assert CM.shape == (B,)
    for t in (CL, CD, CM):
        assert t.dtype == torch.float32
        assert torch.isfinite(t).all()


def test_grad_flows_through_x_and_conditions(model: AAeroSurrogate) -> None:
    B = 4
    x_vec = _random_design(B).requires_grad_(True)
    conds = _random_conditions(B).requires_grad_(True)
    x = decode_x(x_vec)
    CL, CD, CM = model(x, conds)
    (CL.sum() + CD.sum() + CM.sum()).backward()
    assert x_vec.grad is not None
    assert conds.grad is not None
    assert x_vec.grad.abs().sum() > 0
    assert conds.grad.abs().sum() > 0


def test_determinism_in_eval_mode(model: AAeroSurrogate) -> None:
    x_vec = _random_design(8, seed=42)
    conds = _random_conditions(8, seed=43)
    x = decode_x(x_vec)
    out1 = tuple(t.detach().clone() for t in model(x, conds))
    out2 = tuple(t.detach().clone() for t in model(x, conds))
    for a, b in zip(out1, out2, strict=False):
        assert torch.equal(a, b)


def test_outputs_physically_plausible(model: AAeroSurrogate) -> None:
    """CL/CD/CM on random designs inside the sampling envelope stay in DeCoDe's rough range."""
    x = decode_x(_random_design(64, seed=5))
    conds = _random_conditions(64, seed=6)
    CL, CD, CM = model(x, conds)
    # DeCoDe spans roughly CL in [-0.6, 0.8], CD in [1e-3, 0.4], CM in [-0.2, 0.15].
    assert CL.min().item() > -1.2 and CL.max().item() < 1.2
    assert (CD > 1e-5).all() and CD.max().item() < 1.0
    assert CM.abs().max().item() < 0.5


def test_mach_derivation_matches_training_convention(model: AAeroSurrogate) -> None:
    """Varying V at fixed alt must move Ma -> outputs, not just pass through alt."""
    x = decode_x(_random_design(4, seed=7))
    alt = torch.full((4,), 2000.0)
    V_lo = torch.full((4,), 25.0)
    V_hi = torch.full((4,), 75.0)
    c_lo = torch.stack([alt, V_lo], dim=-1)
    c_hi = torch.stack([alt, V_hi], dim=-1)
    Ma_lo = atmosphere.mach(V_lo, alt)
    Ma_hi = atmosphere.mach(V_hi, alt)
    assert (Ma_hi > Ma_lo).all()
    out_lo = model(x, c_lo)
    out_hi = model(x, c_hi)
    diff = sum((a - b).abs().sum().item() for a, b in zip(out_lo, out_hi, strict=False))
    assert diff > 1e-6, "outputs must respond to V (through Ma), not only alt"


def test_checkpoint_r2_recorded_and_meets_plan() -> None:
    """The saved checkpoint logs should record R^2 at or above the target thresholds."""
    from pal.benchmarks.engineering.e1_bwb._artifacts import ensure_artifact_path

    ckpt_path = Path(ensure_artifact_path("a_aero_weights"))
    bundle = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    r2 = bundle.get("r2_test")
    assert r2 is not None, "checkpoint missing r2_test metadata"
    CL_r2, CD_r2, CM_r2 = r2
    assert CL_r2 > 0.95, f"CL R^2={CL_r2:.4f} below plan target 0.95"
    assert CD_r2 > 0.90, f"CD R^2={CD_r2:.4f} below plan target 0.90"
    assert CM_r2 > 0.85, f"CM R^2={CM_r2:.4f} below plan target 0.85"
