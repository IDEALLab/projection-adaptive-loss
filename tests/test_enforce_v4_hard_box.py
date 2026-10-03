"""Hard-output-box clamp tests for the ENFORCE v4 adapter.

Design columns are clamped into the box after each Newton step, FB duals untouched.
"""

from __future__ import annotations

from unittest import mock

import pytest
import torch

from pal.baselines.enforce_v4.solver import _BoxedENFORCE
from pal.baselines.enforce_v4.upstream.model import ENFORCE

# 3 design cols + 2 FB dual cols; cols 0 and 2 straddle the box, col 1 is inside.
_LOWER = torch.tensor([0.5, -1.0, 2.0])
_UPPER = torch.tensor([2.0, 1.0, 5.0])
_DIM = 3
_N_INEQ = 2
_N_EXT = _DIM + _N_INEQ

_Y_OUT = torch.tensor([[-5.0, 0.0, 9.0, 7.0, -3.0]])


def _make_boxed(hard_box: bool) -> _BoxedENFORCE:
    """A `_BoxedENFORCE` head set up like `_build_model`, without the MLP."""
    inst = _BoxedENFORCE.__new__(_BoxedENFORCE)
    torch.nn.Module.__init__(inst)
    inst.setup_boxed_head(_LOWER.clone(), _UPPER.clone(), hard_box=hard_box)
    # Identity output scaling, as `_build_model` constructs it.
    inst.mean_output = torch.zeros(_N_EXT)
    inst.std_output = torch.ones(_N_EXT)
    return inst


def _call_project(inst: _BoxedENFORCE, y: torch.Tensor) -> torch.Tensor:
    """Invoke `inst.project` with upstream `ENFORCE.project` returning `y`."""
    with mock.patch.object(ENFORCE, "project", lambda self, i, o: y):
        return inst.project(torch.zeros(1, 1), torch.zeros(1, _N_EXT))


def test_hard_box_clamps_design_leaves_duals() -> None:
    inst = _make_boxed(hard_box=True)
    out = _call_project(inst, _Y_OUT)

    expected_design = torch.tensor([[0.5, 0.0, 5.0]])
    assert torch.equal(out[:, :_DIM], expected_design)
    assert torch.equal(out[:, _DIM:], _Y_OUT[:, _DIM:])
    # box_clamp_frac = 2 of the 3 design entries were actually moved.
    assert inst._last_box_clamp_frac == pytest.approx(2 / 3)


def test_hard_box_no_clamp_when_inside() -> None:
    inst = _make_boxed(hard_box=True)
    y_inside = torch.tensor([[1.0, 0.0, 3.0, 7.0, -3.0]])
    out = _call_project(inst, y_inside)

    assert torch.equal(out, y_inside)
    assert inst._last_box_clamp_frac == 0.0


def test_soft_box_returns_upstream_unchanged() -> None:
    inst = _make_boxed(hard_box=False)
    out = _call_project(inst, _Y_OUT)

    # Identical to upstream's result (same object, no clamp).
    assert out is _Y_OUT
    assert torch.equal(out, _Y_OUT)
    assert inst._last_box_clamp_frac == 0.0
