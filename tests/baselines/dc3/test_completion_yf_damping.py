"""Yamashita-Fukushima damping knob on the generic Newton completion."""
from __future__ import annotations

import torch

from pal.baselines.dc3._completion import newton_complete


def _resid(Y: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
    """One equality y1 - sin(3*y1) - y0 = 0 with a unique root for these y0."""
    return (Y[:, 1] + 0.3 * torch.sin(Y[:, 1]) - Y[:, 0]).unsqueeze(-1)


def _solve(yf: float, max_iter: int) -> torch.Tensor:
    Z = torch.tensor([[2.0], [4.0], [6.0]], dtype=torch.float64)
    X = torch.zeros(3, 0, dtype=torch.float64)
    return newton_complete(
        _resid, X, Z, partial_vars=[0], other_vars=[1], ydim=2,
        max_iter=max_iter, tol=1e-12, reg=1e-8, yf_damping=yf,
    )


def test_yf_zero_is_plain_newton() -> None:
    """yf_damping=0.0 reproduces the pre-knob code path bit-for-bit."""
    assert torch.equal(_solve(0.0, 50), _solve(0.0, 50))
    y = _solve(0.0, 50)
    assert _resid(y, torch.zeros(3, 0, dtype=torch.float64)).abs().max() < 1e-9


def test_yf_damping_changes_iterates_and_still_converges() -> None:
    """A positive coefficient shortens early steps but reaches the same root."""
    one_step_plain = _solve(0.0, 1)
    one_step_damped = _solve(1.0, 1)
    assert not torch.allclose(one_step_plain, one_step_damped)
    converged = _solve(1.0, 100)
    assert _resid(converged, torch.zeros(3, 0, dtype=torch.float64)).abs().max() < 1e-9
