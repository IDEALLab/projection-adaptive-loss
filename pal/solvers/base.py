"""Shared solver interface: `TrainResult`, `PredictionOutputs`, `Solver` protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from torch import Tensor

from pal.benchmarks.base import Benchmark, Query
from pal.projection.trace import ProjectionResult


@dataclass
class TrainResult:
    """Output of `Solver.train()`.

    Learned solvers fill `model_state`; classical ones fill `cached_predictions`.
    """

    solver_name: str
    train_wall_time_s: float
    n_restarts: int = 1
    model_state: dict[str, Any] | None = None
    train_loss_trajectory: list[float] | None = None
    cached_predictions: dict[str, Tensor] | None = None
    final_x_on_eval: Tensor | None = None
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionOutputs:
    """Both stages of a learned-solver prediction.

    `raw` is the bare NN output and `post` the post-repair output.
    `inference_iters` holds per-query repair-step counts `[N_queries]`, or
    `None` for solvers without an inference repair loop.
    """

    raw: Tensor
    post: Tensor
    projection: ProjectionResult | None = None
    inference_iters: Tensor | None = None


@runtime_checkable
class Solver(Protocol):
    """All solvers expose a `name` and `train` / `predict`.

    `predict()` returns `PredictionOutputs` with both stages.
    """

    name: str

    def train(self, bench: Benchmark, seed: int, logger, **hp) -> TrainResult: ...

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger=None,
    ) -> PredictionOutputs: ...
