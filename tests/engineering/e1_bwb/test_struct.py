"""Smoke tests for ``StructSurrogate`` (skipped without the checkpoint)."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks.engineering.e1_bwb import DIM
from pal.benchmarks.engineering.e1_bwb._artifacts import ensure_artifact_path
from pal.benchmarks.engineering.e1_bwb.interfaces import StructProps
from pal.benchmarks.engineering.e1_bwb.struct import StructSurrogate
from pal.benchmarks.engineering.e1_bwb.x_layout import decode_x


def _struct_weights_available() -> bool:
    try:
        ensure_artifact_path("struct_weights")
        return True
    except (FileNotFoundError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _struct_weights_available(),
    reason="struct_weights artifact required"
    " (shipped in pal/benchmarks/engineering/e1_bwb/_artifacts_data/)",
)


def _sample_design(B: int = 2) -> torch.Tensor:
    """Mid-bounds nominal with zero struct latent."""
    x = torch.zeros(B, DIM)
    x[:, 9] = 1.0                     # L = 1 m
    x[:, 10 + 19 + 3] = 0.3           # battery w
    x[:, 10 + 19 + 4] = 0.3           # battery d
    x[:, 10 + 19 + 5] = 0.2           # battery h
    x[:, -1] = math.radians(1.5)
    return x


def _stations(B: int, N: int, semi_span: torch.Tensor) -> torch.Tensor:
    t = torch.linspace(0.05, 0.95, N)
    return semi_span.unsqueeze(-1) * t


@pytest.fixture(scope="module")
def struct_module():
    return StructSurrogate.load_default(device="cpu")


def test_protocol_conformance_and_shapes(struct_module):
    B, N = 2, 4
    x = _sample_design(B=B)
    dec = decode_x(x)
    y = _stations(B, N, 0.5 * dec.L.squeeze(-1))
    props = struct_module(dec, y)

    assert isinstance(props, StructProps)
    for name in ("I_uu", "I_vv", "J", "A", "u_cg", "v_cg", "Q_max"):
        t_ = getattr(props, name)
        assert t_.shape == (B, N), f"{name} shape mismatch"
        assert torch.isfinite(t_).all(), f"{name} has non-finite entries"


def test_nondim_outputs_physically_plausible(struct_module):
    """Unit-scale section props at L=1 land in the O(1e-8 .. 1e-2) band."""
    B, N = 1, 3
    x = _sample_design(B=B)
    dec = decode_x(x)
    y = _stations(B, N, 0.5 * dec.L.squeeze(-1))
    props = struct_module(dec, y)

    A = props.A
    assert (A > 0).all()
    assert (A < 1.0).all(), f"A too large (unit scale?): {A.tolist()}"

    # Second moments are O(1e-8 .. 1e-4) at unit scale.
    for name in ("I_uu", "I_vv", "J"):
        t_ = getattr(props, name)
        assert (t_ > 0).all()
        assert (t_ < 1.0).all()

    assert (props.Q_max > 0).all()


def test_determinism_in_eval_mode(struct_module):
    B, N = 1, 3
    x = _sample_design(B=B)
    dec = decode_x(x)
    y = _stations(B, N, 0.5 * dec.L.squeeze(-1))
    a = struct_module(dec, y)
    b = struct_module(dec, y)
    for name in ("I_uu", "I_vv", "J", "A", "u_cg", "v_cg", "Q_max"):
        assert torch.equal(getattr(a, name), getattr(b, name))


def test_grad_flows_through_struct_shape_and_L(struct_module):
    """Back-prop from a scalar loss through every `x` sub-field."""
    B, N = 1, 2
    x = _sample_design(B=B).requires_grad_(True)
    dec = decode_x(x)
    y = _stations(B, N, 0.5 * dec.L.squeeze(-1))
    props = struct_module(dec, y)
    loss = props.I_uu.sum() + props.A.sum() + props.u_cg.sum()
    loss.backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad[:, :9].abs().sum() > 0, "no grad through x.shape"
    assert x.grad[:, 9].abs().sum() > 0, "no grad through x.L"
    assert x.grad[:, 10:29].abs().sum() > 0, "no grad through x.struct"


def test_batched_stations_match_single_station_loop(struct_module):
    """Flattened `[B*N]` forward must equal a per-station loop (within fp32)."""
    B, N = 2, 3
    x = _sample_design(B=B)
    dec = decode_x(x)
    y = _stations(B, N, 0.5 * dec.L.squeeze(-1))

    batched = struct_module(dec, y)
    for s in range(N):
        y_s = y[:, s : s + 1]
        ps = struct_module(dec, y_s)
        for name in ("I_uu", "I_vv", "J", "A", "u_cg", "v_cg", "Q_max"):
            assert torch.allclose(
                getattr(batched, name)[:, s : s + 1], getattr(ps, name),
                atol=1e-5, rtol=1e-5,
            ), f"{name} mismatch at station {s}"


def test_checkpoint_swap_mechanism():
    """Passing an explicit path must use it instead of the artifact default."""
    ckpt_path = ensure_artifact_path("struct_weights")
    surrogate = StructSurrogate.load_default(checkpoint=ckpt_path, device="cpu")
    assert isinstance(surrogate, StructSurrogate)

    with pytest.raises(FileNotFoundError):
        StructSurrogate.load_default(
            checkpoint="/nonexistent/path/nope.pt", device="cpu",
        )
