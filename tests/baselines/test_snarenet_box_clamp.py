"""SnareNet repair-layer output-box clamp.

With a box every Newton iterate stays inside it. Without one the loop matches upstream.
"""

from __future__ import annotations

import torch

from pal.baselines.snarenet.solver import _SnareNetRepairLayer


class _Cfg:
    """Minimal cfg duck type: `SnareNetRepairLayer` reads `cfg.model[...]`."""

    def __init__(self) -> None:
        self.model = {"newton_maxiter": 10, "rtol": 1e-8, "lambd": 0.0}


class _ToyData:
    """One equality ``g(y) = y[..., 0] == 0``, ydim=2, recording every evaluated ``y``.

    `box_mode`: "box" returns a tuple, "none" returns None, "absent" has no method.
    """

    def __init__(self, box_mode: str, box=None) -> None:
        self.nineq = 0
        self.neq = 1
        self.ydim = 2
        self._box_mode = box_mode
        self._box = box
        self.g_calls: list[torch.Tensor] = []
        if box_mode == "box":
            self.get_output_box = lambda: self._box  # type: ignore[assignment]
        elif box_mode == "none":
            self.get_output_box = lambda: None  # type: ignore[assignment]
        # "absent": no attribute, exercises the getattr(..., None) fallback.

    def get_lower_upper_bounds(self, *unused):
        return torch.zeros(1), torch.zeros(1)

    def get_g(self, *unused):
        def g(y):
            self.g_calls.append(y.detach().clone())
            return y[..., :1]  # [B, 1]

        return g

    def get_jacobian(self, *unused):
        def j(y):
            B = y.shape[0]
            jm = torch.zeros(B, 1, 2, dtype=y.dtype, device=y.device)
            jm[:, 0, 0] = 1.0
            return jm

        return j


_LO = torch.tensor([0.5, -10.0])
_HI = torch.tensor([10.0, 10.0])
_OUTPUT = torch.tensor([[-1.0, 3.0]])  # y0 below the box lower bound (0.5)


def test_repair_clamps_every_iterate_into_box() -> None:
    data = _ToyData("box", box=(_LO, _HI))
    layer = _SnareNetRepairLayer(data, _Cfg())

    out = layer.repair(_OUTPUT.clone())

    assert torch.all(out >= _LO - 1e-6), out
    assert torch.all(out <= _HI + 1e-6), out
    assert data.g_calls, "g was never called"
    for y in data.g_calls:
        assert torch.all(y >= _LO - 1e-6), y
        assert torch.all(y <= _HI + 1e-6), y
    # Newton pulls y0 -> 0, the clamp holds it at the lower bound 0.5.
    assert abs(out[0, 0].item() - 0.5) < 1e-5, out


def test_box_none_and_absent_are_bit_identical_to_upstream() -> None:
    data_none = _ToyData("none")
    data_absent = _ToyData("absent")
    out_none = _SnareNetRepairLayer(data_none, _Cfg()).repair(_OUTPUT.clone())
    out_absent = _SnareNetRepairLayer(data_absent, _Cfg()).repair(_OUTPUT.clone())

    assert torch.equal(out_none, out_absent)
    # No-box paths reach the unclamped fixed point y0 -> 0 (not 0.5).
    assert abs(out_none[0, 0].item()) < 1e-5, out_none

    out_box = _SnareNetRepairLayer(
        _ToyData("box", box=(_LO, _HI)), _Cfg()
    ).repair(_OUTPUT.clone())
    assert not torch.allclose(out_box, out_none)
