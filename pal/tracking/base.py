"""Logger protocol, the key interface every sink implements."""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable


@runtime_checkable
class Logger(Protocol):
    """Structured sink for runner + solver telemetry.

    `log_step` takes scalars only, `log_projection_trajectory` a list of
    `ProjectionStep`, and `log_artifact` non-scalar payloads (figures, tensors).
    """

    def log_config(self, **cfg: Any) -> None: ...

    def log_step(self, step: int, **scalars: float) -> None: ...

    def log_projection_trajectory(
        self, step: int, phase: str, trajectory: list[Any]
    ) -> None: ...

    def log_artifact(self, step: int, name: str, payload: Any) -> None: ...

    def log_final(self, **final: Any) -> None: ...

    def finish(
        self,
        status: Literal["ok", "failed"] = "ok",
        error: str | None = None,
    ) -> None: ...
