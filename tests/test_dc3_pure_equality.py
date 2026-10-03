"""DC3 `grad_steps_all` on a bench with no inequality constraints (`[B, 0]` ineq)."""

from __future__ import annotations

import torch

from pal.baselines.dc3._upstream_loader import load_vendored

ARGS = {
    "useTestCorr": True,
    "useCompl": False,
    "corrMode": "full",
    "corrLr": 1e-1,
    "corrEps": 1e-4,
    "corrTestMaxSteps": 5,
    "corrMomentum": 0.0,
    "softWeightEqFrac": 0.5,
}


class _PureEqData:
    """Minimal DC3 `data` surface for a single linear equality `sum(y) = 0`."""

    neq = 1
    nineq = 0

    def __init__(self, scale: float = 1.0):
        self._scale = scale

    def eq_resid(self, X, Y):
        return self._scale * Y.sum(dim=1, keepdim=True)  # [B, 1]

    def ineq_dist(self, X, Y):
        return Y.new_zeros(Y.shape[0], 0)  # [B, 0], the crashing input

    def eq_grad(self, X, Y):
        return 2.0 * self._scale**2 * Y.sum(dim=1, keepdim=True).expand_as(Y)

    def ineq_grad(self, X, Y):
        return torch.zeros_like(Y)


def _run(scale: float):
    _grad_steps, grad_steps_all, _total_loss = load_vendored()
    data = _PureEqData(scale=scale)
    X = torch.zeros(4, 0)
    Y = torch.randn(4, 3, generator=torch.Generator().manual_seed(0))
    return grad_steps_all(data, X, Y, ARGS)


def test_grad_steps_all_empty_ineq_unconverged_eq():
    """EQ residual starts above `corrEps`: loop runs, ineq term must not crash."""
    Y_new, steps = _run(scale=1.0)
    assert torch.isfinite(Y_new).all()
    assert steps >= 1


def test_grad_steps_all_empty_ineq_converged_eq():
    """EQ residual already below `corrEps`: the `or` chain reaches the empty ineq max."""
    Y_new, steps = _run(scale=1e-12)
    assert torch.isfinite(Y_new).all()
    # Empty ineq set adds no violation, so the loop exits after the first iteration.
    assert steps == 1


def test_grad_steps_all_empty_ineq_and_empty_eq():
    """Both constraint sets empty: neither `torch.max` is reached."""
    _grad_steps, grad_steps_all, _total_loss = load_vendored()

    class _Empty(_PureEqData):
        neq = 0
        nineq = 0

        def eq_resid(self, X, Y):
            return Y.new_zeros(Y.shape[0], 0)

    X = torch.zeros(2, 0)
    Y = torch.ones(2, 3)
    Y_new, steps = grad_steps_all(_Empty(), X, Y, ARGS)
    assert steps == 1
    assert torch.isfinite(Y_new).all()
