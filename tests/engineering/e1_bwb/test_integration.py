"""End-to-end test of ``E1BWB(live=True)``: finite outputs, gradients, stub fallback."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM
from pal.benchmarks.engineering.e1_bwb._artifacts import check_runtime_artifacts
from pal.benchmarks.engineering.e1_bwb.loads import _BWB_YAML


def _all_runtime_artifacts_available() -> bool:
    if not _BWB_YAML.exists():
        return False
    try:
        check_runtime_artifacts()
        return True
    except (FileNotFoundError, KeyError, RuntimeError):
        return False


pytestmark = pytest.mark.skipif(
    not _all_runtime_artifacts_available(),
    reason="e1/bwb integration tests require the e1/bwb artifacts"
    " (shipped in pal/benchmarks/engineering/e1_bwb/_artifacts_data/)",
)


def _sample_design(B: int = 2) -> torch.Tensor:
    """Mid-bounds nominal, shape=0, L=1, struct=0, battery small+centred."""
    x = torch.zeros(B, DIM)
    x[:, 9] = 1.0                     # L = 1 m
    x[:, 10 + 19 + 3] = 0.25          # battery w
    x[:, 10 + 19 + 4] = 0.25          # battery d
    x[:, 10 + 19 + 5] = 0.1           # battery h
    x[:, -1] = math.radians(1.5)      # alpha_cr = 1.5 deg
    return x


def _sample_conditions(B: int = 2) -> torch.Tensor:
    return torch.stack(
        [
            torch.full((B,), 2000.0),   # alt = 2 km
            torch.full((B,), 40.0),     # V = 40 m/s
        ],
        dim=-1,
    )


@pytest.fixture(scope="module")
def bench_live() -> E1BWB:
    return E1BWB(live=True, n_stations=5)


def test_live_forward_finite(bench_live):
    x = _sample_design(B=2)
    conds = _sample_conditions(B=2)
    obj, clist = bench_live.forward(x, conds)

    assert obj.shape == (2,)
    assert torch.isfinite(obj).all()

    # Lift balance is a normalised (L - W)/W ratio; should be O(1) at most.
    names = [c.name for c in clist]
    assert names[0] == "lift_balance"
    lb = clist[0].value
    assert torch.isfinite(lb).all()

    for c in clist[1:]:
        assert c.value.shape == (2,)
        assert torch.isfinite(c.value).all()


def test_live_gradients_reach_every_subfield(bench_live):
    x = _sample_design(B=2).requires_grad_(True)
    conds = _sample_conditions(B=2)
    obj, clist = bench_live.forward(x, conds)
    loss = obj.sum() + clist[0].value.sum()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    # Shape (9) + L (1) + struct (19) + battery (6) + alpha (1) all move.
    assert x.grad[:, 0:9].abs().sum() > 0, "no grad through x.shape"
    assert x.grad[:, 9].abs().sum() > 0, "no grad through x.L"
    assert x.grad[:, 10:29].abs().sum() > 0, "no grad through x.struct"
    assert x.grad[:, 29:35].abs().sum() > 0, "no grad through x.battery"
    assert x.grad[:, -1].abs().sum() > 0, "no grad through x.alpha_cr"


def test_check_env_passes_when_artefacts_present(bench_live):
    bench_live.check_env()


def test_stub_mode_still_runs():
    """Sanity, `live=False` gives the scaffold stub pipeline."""
    bench = E1BWB(live=False)
    x = _sample_design(B=2)
    conds = _sample_conditions(B=2)
    obj, clist = bench.forward(x, conds)
    assert torch.isfinite(obj).all()
    assert all(torch.isfinite(c.value).all() for c in clist)
