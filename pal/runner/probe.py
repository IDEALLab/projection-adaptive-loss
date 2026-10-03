"""Per-run instrumentation: counts bench forward/backward passes during training.

`BenchProbe` wraps a `Benchmark`, counts calls to its solver-facing entry points,
and hooks the input tensor so each backward that reaches the evaluator counts
once. Only the training phase is counted. Under `torch.vmap` / `jacrev` the probe
undercounts the Jacobian work.
"""

from __future__ import annotations

import weakref
from contextlib import contextmanager
from typing import Any

from torch import Tensor

from pal.benchmarks.base import Benchmark

_PHASES = ("train", "periodic_eval", "final_eval", "predict")


def mark_opt_step(bench: Any) -> None:
    """Bump `bench`'s outer optimizer-step counter if it's a probe.

    No-op when `bench` is a bare `Benchmark` without a probe wrapper.
    """
    fn = getattr(bench, "mark_opt_step", None)
    if callable(fn):
        fn()


def _zero_counters() -> dict[str, dict[str, int]]:
    return {
        p: {
            "fwd_calls": 0,
            "fwd_samples": 0,
            "bwd_calls": 0,
            "bwd_samples": 0,
            "opt_steps": 0,
            # Measurement-window counters, bumped only while the measurement flag is set.
            "measurement_fwd_calls": 0,
            "measurement_fwd_samples": 0,
            "measurement_bwd_calls": 0,
            "measurement_bwd_samples": 0,
            "measurement_opt_steps": 0,
        }
        for p in _PHASES
    }


class BenchProbe:
    """Transparent counter wrapper around a `Benchmark`.

    Usage:

        probe = BenchProbe(bench)
        probe.set_phase("train")
        solver.train(probe, ...)
        probe.set_phase("final_eval")
        run_final_eval(probe, ...)
        counts = probe.snapshot()   # dict[phase][metric] -> int
    """

    def __init__(self, bench: Benchmark):
        self._bench = bench
        self._phase = "train"
        self._counters = _zero_counters()
        # Ids of already-hooked inputs, so a reused input is counted once.
        self._hooked_ids: set[int] = set()
        self._measurement: bool = False

    @property
    def spec(self):
        return self._bench.spec

    @property
    def phase(self) -> str:
        return self._phase

    def set_phase(self, phase: str) -> None:
        if phase not in _PHASES:
            raise ValueError(f"unknown phase '{phase}'; expected one of {_PHASES}")
        self._phase = phase

    @contextmanager
    def phase_ctx(self, phase: str):
        """Temporarily switch phase, restoring the previous phase on exit.

        Used for nested stages, e.g. periodic eval inside `solver.train()`.
        """
        prev = self._phase
        self.set_phase(phase)
        try:
            yield
        finally:
            self._phase = prev

    def snapshot(self) -> dict[str, dict[str, int]]:
        """Return a deep copy of the current counter dict."""
        return {p: dict(v) for p, v in self._counters.items()}

    def flat_snapshot(self) -> dict[str, int]:
        """Flatten counters to `"{phase}__{metric}"` keys for JSON logging."""
        out: dict[str, int] = {}
        for phase, metrics in self._counters.items():
            for k, v in metrics.items():
                out[f"{phase}__{k}"] = v
        return out

    def mark_opt_step(self) -> None:
        """Bump the outer optimizer-step counter for the current phase.

        Solvers call this after each `optimizer.step()`. Train phase only.
        """
        if self._phase != "train":
            return
        self._counters[self._phase]["opt_steps"] += 1
        if self._measurement:
            self._counters[self._phase]["measurement_opt_steps"] += 1

    def set_measurement(self, flag: bool) -> None:
        """Enable/disable the measurement-window double-tick.

        Solvers flip this per-epoch during `jacobian_mode="sample"` windows.
        """
        self._measurement = bool(flag)

    @contextmanager
    def measurement_ctx(self, flag: bool = True):
        """Temporarily toggle the measurement flag; restore on exit."""
        prev = self._measurement
        self._measurement = bool(flag)
        try:
            yield
        finally:
            self._measurement = prev

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bench, name)

    def forward(self, x: Tensor, conditions: Tensor | None = None):
        B = self._tally_fwd(x)
        obj, cons = self._bench.forward(x, conditions)
        self._hook_bwd(x, B)
        return obj, cons

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        B = self._tally_fwd(x)
        obj = self._bench.objective(x, conditions)
        self._hook_bwd(x, B)
        return obj

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        B = self._tally_fwd(x)
        c = self._bench.constraints(x, conditions)
        self._hook_bwd(x, B)
        return c

    def constraint_list(self, x: Tensor, conditions: Tensor | None = None):
        B = self._tally_fwd(x)
        clist = self._bench.constraint_list(x, conditions)
        self._hook_bwd(x, B)
        return clist

    def _tally_fwd(self, x: Tensor) -> int:
        B = int(x.shape[0]) if x.dim() >= 2 else 1
        if self._phase != "train":
            return B
        c = self._counters[self._phase]
        c["fwd_calls"] += 1
        c["fwd_samples"] += B
        if self._measurement:
            c["measurement_fwd_calls"] += 1
            c["measurement_fwd_samples"] += B
        return B

    def _hook_bwd(self, tensor: Tensor, B: int) -> None:
        if not isinstance(tensor, Tensor):
            return
        if not tensor.requires_grad:
            return
        if self._phase != "train":
            return
        tid = id(tensor)
        if tid in self._hooked_ids:
            return
        self._hooked_ids.add(tid)
        try:
            weakref.finalize(tensor, self._hooked_ids.discard, tid)
        except TypeError:
            # Some tensor subclasses resist weakref.
            pass
        phase = self._phase
        measurement = self._measurement
        counters = self._counters

        def _hook(grad: Tensor) -> None:
            counters[phase]["bwd_calls"] += 1
            counters[phase]["bwd_samples"] += B
            if measurement:
                counters[phase]["measurement_bwd_calls"] += 1
                counters[phase]["measurement_bwd_samples"] += B
            return None

        try:
            tensor.register_hook(_hook)
        except RuntimeError:
            pass
