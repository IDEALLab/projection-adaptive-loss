"""Frozen eval points loaded from `pal/configs/eval_points/<bench>.json`.

The JSON holds `benchmark`, `frozen_at`, `rationale`, a list of
`{"zeta": [...], "condition": [...]}` points and an `ipopt` options dict.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from pal.benchmarks.base import BenchmarkSpec, Query


@dataclass(frozen=True)
class FrozenPoints:
    """In-memory representation of a frozen eval-points JSON file."""

    benchmark: str
    frozen_at: str
    rationale: str
    zeta: Tensor          # [N, zeta_dim]
    conditions: Tensor    # [N, condition_dim]
    ipopt: dict[str, Any]

    def __len__(self) -> int:
        return int(self.zeta.shape[0])

    def to_query(self, spec: BenchmarkSpec) -> Query:
        """Validate against the bench spec and produce a `Query`."""
        if spec.id != self.benchmark and spec.family != self.benchmark:
            raise ValueError(
                f"frozen points declare benchmark={self.benchmark!r}, "
                f"but spec.id={spec.id!r} (family={spec.family!r})"
            )
        if self.zeta.shape[1] != spec.zeta_dim:
            raise ValueError(
                f"zeta dim mismatch: JSON has {self.zeta.shape[1]}, "
                f"spec.zeta_dim={spec.zeta_dim}"
            )
        if self.conditions.shape[1] != spec.condition_dim:
            raise ValueError(
                f"conditions dim mismatch: JSON has {self.conditions.shape[1]}, "
                f"spec.condition_dim={spec.condition_dim}"
            )
        return Query(zeta=self.zeta, conditions=self.conditions)


def load_frozen_points(path: str | Path) -> FrozenPoints:
    """Load a frozen-points JSON file. Empty `points` is allowed (stub)."""
    p = Path(path)
    with p.open() as f:
        raw = json.load(f)

    points = raw.get("points", [])
    zeta_dim = len(points[0]["zeta"]) if points else 0
    cond_dim = len(points[0]["condition"]) if points else 0

    for i, pt in enumerate(points):
        if len(pt["zeta"]) != zeta_dim:
            raise ValueError(
                f"point {i} zeta has length {len(pt['zeta'])}, "
                f"expected {zeta_dim}"
            )
        if len(pt["condition"]) != cond_dim:
            raise ValueError(
                f"point {i} condition has length {len(pt['condition'])}, "
                f"expected {cond_dim}"
            )

    if points:
        zeta = torch.tensor(
            [pt["zeta"] for pt in points], dtype=torch.get_default_dtype(),
        )
        conditions = torch.tensor(
            [pt["condition"] for pt in points], dtype=torch.get_default_dtype(),
        )
    else:
        zeta = torch.zeros((0, zeta_dim), dtype=torch.get_default_dtype())
        conditions = torch.zeros((0, cond_dim), dtype=torch.get_default_dtype())

    return FrozenPoints(
        benchmark=str(raw.get("benchmark", "")),
        frozen_at=str(raw.get("frozen_at", "")),
        rationale=str(raw.get("rationale", "")),
        zeta=zeta,
        conditions=conditions,
        ipopt=dict(raw.get("ipopt", {})),
    )
