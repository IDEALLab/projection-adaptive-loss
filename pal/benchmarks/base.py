"""Benchmark protocol, spec dataclass, and Query abstraction.

Constraints are positional per `spec.constraint_types`: eq feasible at `c == 0`, ineq at `c <= 0`.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, Protocol, runtime_checkable

import torch
from torch import Tensor

from pal.artifacts.core import ArtifactRef


@dataclass(frozen=True)
class BenchmarkSpec:
    """Static metadata for a benchmark (tolerance, bounds, artifacts, cost).

    Args:
        id: Canonical id. Families use `"<family>/<variant>"`
            (e.g. `"e3/acopf_ieee30"`); singletons are just `"rosenbrock_eq"`.
        family: Family prefix (e.g. `"rosenbrock_eq"`, `"e3"`).
        variant: Variant suffix for families, `None` for singletons.
        dim: Decision variable dimension.
        n_eq: Number of equality constraints.
        n_ineq: Number of inequality constraints.
        constraint_names: Length `n_eq + n_ineq`, positional with `constraint_types`.
        constraint_types: Length `n_eq + n_ineq`, each entry `"eq"` or `"ineq"`.
            Records the exact constraint ordering of `constraint_list`.
        output_bounds: `(lo, hi)` tensors of shape `[dim]`.
        condition_dim: 0 for unconditional benchmarks.
        zeta_dim: Latent-input dim; drives unconditional variance.
        tolerance: Per-benchmark feasibility threshold (overrides framework default).
        tau: Per-benchmark log-gap target for multiplier methods, independent of
            `tolerance`. `None` falls back to the per-method `cfg.tau` default.
        cost: Rough per-seed compute budget.
        recommended_device: Where the benchmark is meant to run.
        artifacts: HF-hosted files required at runtime.
        precision: Default precision for training and eval. `"fp64"` makes the
            runner solve and evaluate in float64.
        recommended_batch_per_gpu: Hardware-keyed batch sizes for the launcher.
        train_batch_size: Per-bench training batch override for all learned
            methods. An explicit `--batch-size` on the CLI still wins.
        hard_output_box: If True, the projector clamps every projected iterate
            into `output_bounds` (for evaluators undefined outside the box).
        solver_hparams: Per-solver hparam overrides keyed by solver name,
            applied before explicit CLI flags.
        model_hparams: Per-bench overrides for the shared `CoordinationMLP`
            (currently `output_init_std`).
        notes: Free-form (e.g. `"requires float64, jax CPU"`).
    """

    id: str
    family: str
    variant: str | None
    dim: int
    n_eq: int
    n_ineq: int
    constraint_names: list[str]
    constraint_types: list[str]
    output_bounds: tuple[Tensor, Tensor]
    condition_dim: int
    zeta_dim: int
    tolerance: float
    cost: Literal["cheap", "mid", "expensive"]
    recommended_device: Literal["cpu", "gpu"]
    tau: float | None = None
    artifacts: list[ArtifactRef] = field(default_factory=list)
    precision: Literal["fp32", "bf16", "fp16", "fp64"] = "fp32"
    recommended_batch_per_gpu: dict[str, int] = field(default_factory=dict)
    train_batch_size: int | None = None
    n_eval_default: int = 64
    hard_output_box: bool = False
    solver_hparams: dict[str, dict[str, Any]] = field(default_factory=dict)
    model_hparams: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def __post_init__(self) -> None:
        lo, hi = self.output_bounds
        if lo.shape != (self.dim,) or hi.shape != (self.dim,):
            raise ValueError(
                f"output_bounds must each be shape [{self.dim}]; "
                f"got {tuple(lo.shape)} and {tuple(hi.shape)}"
            )
        K = self.n_eq + self.n_ineq
        if len(self.constraint_names) != K:
            raise ValueError(
                f"constraint_names length {len(self.constraint_names)} != "
                f"n_eq + n_ineq = {K}"
            )
        if len(self.constraint_types) != K:
            raise ValueError(
                f"constraint_types length {len(self.constraint_types)} != "
                f"n_eq + n_ineq = {K}"
            )
        bad = [t for t in self.constraint_types if t not in ("eq", "ineq")]
        if bad:
            raise ValueError(f"constraint_types entries must be 'eq'|'ineq'; got {bad}")
        if sum(1 for t in self.constraint_types if t == "eq") != self.n_eq:
            raise ValueError(
                "constraint_types eq-count disagrees with n_eq "
                f"({self.constraint_types} vs n_eq={self.n_eq})"
            )

    def to_json_dict(self) -> dict[str, Any]:
        """Lossy-but-portable JSON representation for `config.json`."""
        return {
            "id": self.id,
            "family": self.family,
            "variant": self.variant,
            "dim": self.dim,
            "n_eq": self.n_eq,
            "n_ineq": self.n_ineq,
            "constraint_names": list(self.constraint_names),
            "constraint_types": list(self.constraint_types),
            "output_bounds_lo": self.output_bounds[0].tolist(),
            "output_bounds_hi": self.output_bounds[1].tolist(),
            "condition_dim": self.condition_dim,
            "zeta_dim": self.zeta_dim,
            "tolerance": self.tolerance,
            "tau": self.tau,
            "cost": self.cost,
            "recommended_device": self.recommended_device,
            "artifacts": [asdict(a) for a in self.artifacts],
            "precision": self.precision,
            "recommended_batch_per_gpu": dict(self.recommended_batch_per_gpu),
            "train_batch_size": self.train_batch_size,
            "n_eval_default": self.n_eval_default,
            "hard_output_box": self.hard_output_box,
            "solver_hparams": {k: dict(v) for k, v in self.solver_hparams.items()},
            "model_hparams": dict(self.model_hparams),
            "notes": self.notes,
        }


@dataclass(frozen=True)
class Query:
    """Full learned-model input: latent `zeta` plus `conditions` (`[N, 0]` if unconditional)."""

    zeta: Tensor
    conditions: Tensor

    def __len__(self) -> int:
        return int(self.zeta.shape[0])

    def __post_init__(self) -> None:
        if self.zeta.shape[0] != self.conditions.shape[0]:
            raise ValueError(
                f"zeta and conditions must share leading dim; "
                f"got zeta={tuple(self.zeta.shape)}, "
                f"conditions={tuple(self.conditions.shape)}"
            )

    def to(self, device: torch.device | str) -> Query:
        return Query(self.zeta.to(device), self.conditions.to(device))


@runtime_checkable
class Benchmark(Protocol):
    """Duck-typed benchmark interface.

    Baselines need `forward(x, conditions) -> (obj, list[Constraint])` computing both in one pass.
    """

    spec: BenchmarkSpec

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor: ...

    def constraints(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Tensor: ...

    def sample_queries(
        self,
        n: int,
        split: Literal["train", "eval"],
        seed: int,
    ) -> Query: ...

    def eval_queries(self, seed: int, n: int | None = None) -> Query: ...
    """Fixed evaluation set. `n=None` uses `BenchmarkSpec.n_eval_default`."""

    def check_env(self) -> None: ...

    def visualize_train(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Any | None: ...
    """Cheap per-epoch viz (matplotlib only). Return `None` if unsupported."""

    def visualize_final(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> Any | None: ...
    """High-polish local-only viz: a dict of panels, a single `Figure`, or `None`."""
