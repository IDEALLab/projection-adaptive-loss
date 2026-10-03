"""CompositeLogger fan-out tests."""

from __future__ import annotations

from pal.tracking.composite import CompositeLogger


class _Recorder:
    def __init__(self) -> None:
        self.events: list[tuple] = []

    def log_config(self, **cfg):
        self.events.append(("config", cfg))

    def log_step(self, step, **scalars):
        self.events.append(("step", step, scalars))

    def log_projection_trajectory(self, step, phase, trajectory):
        self.events.append(("trajectory", step, phase, list(trajectory)))

    def log_artifact(self, step, name, payload):
        self.events.append(("artifact", step, name, payload))

    def log_final(self, **final):
        self.events.append(("final", final))

    def finish(self, status="ok", error=None):
        self.events.append(("finish", status, error))


def test_composite_fans_out_all_events_in_order() -> None:
    a, b = _Recorder(), _Recorder()
    log = CompositeLogger([a, b])

    log.log_config(method="pal_loggap")
    log.log_step(0, loss=1.0)
    log.log_projection_trajectory(0, "train", [])
    log.log_artifact(0, "viz", {"k": 1})
    log.log_final(obj_mean=0.5)
    log.finish(status="ok")

    expected = [
        ("config", {"method": "pal_loggap"}),
        ("step", 0, {"loss": 1.0}),
        ("trajectory", 0, "train", []),
        ("artifact", 0, "viz", {"k": 1}),
        ("final", {"obj_mean": 0.5}),
        ("finish", "ok", None),
    ]
    assert a.events == expected
    assert b.events == expected


def test_composite_propagates_exception_from_sink() -> None:
    import pytest

    class _Boom:
        def log_step(self, step, **scalars):
            raise RuntimeError("sink-down")

        def log_config(self, **cfg): ...
        def log_projection_trajectory(self, *a, **k): ...
        def log_artifact(self, *a, **k): ...
        def log_final(self, **k): ...
        def finish(self, status="ok", error=None): ...

    log = CompositeLogger([_Boom()])
    with pytest.raises(RuntimeError, match="sink-down"):
        log.log_step(0, loss=1.0)


def test_composite_accepts_logger_protocol_instance() -> None:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from pal.tracking.base import Logger
    from pal.tracking.step_logger import JSONLLogger

    with TemporaryDirectory() as td:
        jsonl = JSONLLogger(Path(td))
        assert isinstance(jsonl, Logger)

        log = CompositeLogger([jsonl])
        log.log_step(0, loss=1.0)
        log.finish()
