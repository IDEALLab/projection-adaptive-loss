"""Micro-batch gradient equivalence for the ENFORCE v4 adapter.

Chunk losses scaled by chunk_size / batch_size must reproduce the full-batch gradient.
"""

from __future__ import annotations

import torch

from pal.baselines.enforce_v4 import EnforceV4Config, EnforceV4Solver
from pal.benchmarks import get as get_benchmark

_BENCH = "s2_active_set_switch"  # 1 eq + 10 ineq: exercises the FB projection.
_BATCH = 16
_RTOL = 1e-4
_ATOL = 1e-6


class _NullLogger:
    def log_config(self, **cfg): pass
    def log_step(self, step, **scalars): pass
    def log_projection_trajectory(self, step, phase, trajectory): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **final): pass
    def finish(self, status="ok", error=None): pass


def _epoch1_grads(micro_batch: int | None, epoch_start_hard: int) -> torch.Tensor:
    """Gradient accumulated over epoch 1, flattened across all parameters."""
    grabbed: dict[str, torch.Tensor] = {}

    def hook(epoch: int, viz_model) -> None:
        if epoch == 1:
            grabbed["g"] = torch.cat([
                p.grad.detach().flatten().clone()
                for p in viz_model.model.parameters()
                if p.grad is not None
            ])

    cfg = EnforceV4Config(
        seed=0, epochs=1, batch_size=_BATCH, lr=1e-4, device="cpu",
        epoch_start_hard_constrained=epoch_start_hard, micro_batch=micro_batch,
    )
    EnforceV4Solver(cfg).train(
        get_benchmark(_BENCH), seed=0, logger=_NullLogger(), on_epoch_end=hook,
    )
    return grabbed["g"]


def _assert_grad_equivalent(epoch_start_hard: int) -> None:
    full = _epoch1_grads(None, epoch_start_hard)
    assert full.norm() > 0  # sanity: the epoch produced a real gradient.
    # 5 leaves a ragged last chunk; 100 clamps to the full batch (bit-exact None path).
    for micro_batch in (4, 5, 1, 100):
        g = _epoch1_grads(micro_batch, epoch_start_hard)
        max_abs = (g - full).abs().max().item()
        assert torch.allclose(g, full, rtol=_RTOL, atol=_ATOL), (
            f"micro_batch={micro_batch} grad mismatch at "
            f"epoch_start_hard={epoch_start_hard}: max_abs={max_abs:.3e}"
        )


def test_micro_batch_equivalent_warmup_phase() -> None:
    """Warm-up soft step (no projection): per-sample independent -> exact."""
    _assert_grad_equivalent(epoch_start_hard=999)


def test_micro_batch_equivalent_projection_phase() -> None:
    """Projection step (Newton + FB + auto-activation gate) from epoch 0."""
    _assert_grad_equivalent(epoch_start_hard=0)


def test_micro_batch_none_is_full_batch() -> None:
    """`micro_batch=None` (default) and a batch-spanning value are identical."""
    base = _epoch1_grads(None, epoch_start_hard=0)
    clamped = _epoch1_grads(_BATCH, epoch_start_hard=0)
    assert torch.equal(base, clamped)
