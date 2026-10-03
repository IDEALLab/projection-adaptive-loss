"""Adaptive relaxation for SnareNet's repair layer, ported from `SnareNet/utils/utils.py:28-114`.

The initial slack is calibrated on `n_calibration_batches` sampled condition draws
instead of one pass over a fixed training set.
"""

from __future__ import annotations

from collections.abc import Callable

import torch


class AdaptiveRelaxation:
    """Gradually tighten constraint slack from measured max violation -> 0."""

    def __init__(
        self,
        start_epoch: int,
        decay_epochs: int,
        device: str | torch.device,
        decay_fn: str = "linear",
    ):
        self.start_epoch = start_epoch
        self.decay_epochs = decay_epochs
        self.device = device
        self.eps_initial: torch.Tensor | None = None
        self.initialized = False

        if decay_fn == "linear":
            self.decay_fn: Callable = self._linear_decay
        elif decay_fn == "harmonic":
            self.decay_fn = self._harmonic_decay
        elif decay_fn == "linear_harmonic":
            self.decay_fn = self._linear_harmonic_decay
        else:
            raise ValueError(f"unknown decay function: {decay_fn}")

    def get_init_eps(
        self,
        shim,
        net,
        bench,
        seed: int,
        n_batches: int,
        batch_size: int,
        zeta_zero: bool = False,
    ) -> None:
        """N-batch sampled scan of max constraint violation per slot."""
        if self.initialized:
            return
        self.initialized = True

        n_constraints = shim.neq + shim.nineq
        self.eps_initial = torch.full(
            (n_constraints,), float("-inf"), device=self.device,
        )

        if not hasattr(net, "set_repair"):
            raise AttributeError(
                "network must expose `set_repair` to use adaptive relaxation"
            )
        original_repair_state = net._if_repair
        net.set_repair(False)
        try:
            with torch.no_grad():
                for i in range(n_batches):
                    q = bench.sample_queries(
                        batch_size, split="train", seed=seed + i,
                    )
                    bench_x = q.conditions.to(device=self.device)
                    zeta = q.zeta.to(device=self.device)
                    if zeta_zero:
                        zeta = torch.zeros_like(zeta)
                    x = torch.cat([zeta, bench_x], dim=-1)
                    shim.update_x(x, bench_x)
                    y_nn = net(x)
                    # Repair is off here, so clamp into the box first to avoid NaN eps.
                    _get_box = getattr(shim, "get_output_box", None)
                    box = _get_box() if _get_box is not None else None
                    if box is not None:
                        y_nn = torch.clamp(
                            y_nn,
                            min=box[0].to(device=y_nn.device, dtype=y_nn.dtype),
                            max=box[1].to(device=y_nn.device, dtype=y_nn.dtype),
                        )
                    resid = shim.get_resid(x, y_nn)         # [B, K]
                    max_per_slot = resid.max(dim=0).values  # [K]
                    self.eps_initial = torch.maximum(
                        max_per_slot, self.eps_initial,
                    )
        finally:
            net.set_repair(original_repair_state)

    def _linear_decay(self, step, total_steps, initial_value):
        return initial_value * (1 - step / total_steps)

    def _harmonic_decay(self, step, total_steps, initial_value):
        return initial_value / (step + 1)

    def _linear_harmonic_decay(self, step, total_steps, initial_value):
        mid_step = total_steps / 2
        if step < mid_step:
            return initial_value * (1 - step / (2 * total_steps))
        adjusted_step = step - mid_step + 1
        return (initial_value / 2) / adjusted_step

    def get_eps(self, epoch: int) -> torch.Tensor:
        if not self.initialized or epoch < self.start_epoch:
            return torch.tensor(0.0, device=self.device)
        decay_step = epoch - self.start_epoch
        if decay_step >= self.decay_epochs:
            return torch.tensor(0.0, device=self.device)
        eps = self.decay_fn(decay_step, self.decay_epochs, self.eps_initial)
        return torch.clamp(eps, min=0)
