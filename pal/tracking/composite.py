"""Fan-out logger, forwards every event to multiple sinks."""

from __future__ import annotations

from typing import Any, Literal

from pal.tracking.base import Logger


class CompositeLogger:
    """Forwards every `log_*` call to each child sink in order.

    If one child raises, the exception propagates: losing a sink silently is
    worse than failing the run.
    """

    def __init__(self, sinks: list[Logger]):
        self.sinks = list(sinks)

    def log_config(self, **cfg: Any) -> None:
        for s in self.sinks:
            s.log_config(**cfg)

    def log_step(self, step: int, **scalars: float) -> None:
        for s in self.sinks:
            s.log_step(step, **scalars)

    def log_projection_trajectory(
        self, step: int, phase: str, trajectory: list[Any]
    ) -> None:
        for s in self.sinks:
            s.log_projection_trajectory(step, phase, trajectory)

    def log_artifact(self, step: int, name: str, payload: Any) -> None:
        for s in self.sinks:
            s.log_artifact(step, name, payload)

    def log_final(self, **final: Any) -> None:
        for s in self.sinks:
            s.log_final(**final)

    def finish(
        self,
        status: Literal["ok", "failed"] = "ok",
        error: str | None = None,
    ) -> None:
        for s in self.sinks:
            s.finish(status=status, error=error)
