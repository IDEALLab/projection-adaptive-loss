"""GPU VRAM tier registry loader + resolver.

The tier of a (bench, method) row is the max of the bench and method tiers
(S < M < L), unless an override pins the pair.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml

Tier = Literal["S", "M", "L"]

_TIER_ORDER: dict[Tier, int] = {"S": 0, "M": 1, "L": 2}
_DEFAULT_PRESETS_PATH = Path(__file__).with_name("gpu_presets.yaml")


@dataclass(frozen=True)
class TierRegistry:
    """Parsed tier registry.

    Args:
        benchmarks: bench_id -> tier.
        methods: method_name -> tier.
        overrides: (bench_id, method) -> tier (takes precedence over max rule).
        default_tier: fallback when a bench or method is not registered.
    """

    benchmarks: dict[str, Tier]
    methods: dict[str, Tier]
    overrides: dict[tuple[str, str], Tier] = field(default_factory=dict)
    default_tier: Tier = "M"

    def tier_for_bench(self, bench_id: str) -> Tier:
        return self.benchmarks.get(bench_id, self.default_tier)

    def tier_for_method(self, method: str) -> Tier:
        return self.methods.get(method, self.default_tier)


def _validate_tier(value: object, where: str) -> Tier:
    if value not in _TIER_ORDER:
        raise ValueError(
            f"invalid tier {value!r} at {where}; expected one of {list(_TIER_ORDER)}"
        )
    return value  # type: ignore[return-value]


def load_tier_registry(path: Path | str | None = None) -> TierRegistry:
    """Load the tier registry from YAML. Path defaults to `gpu_presets.yaml`."""
    p = Path(path) if path is not None else _DEFAULT_PRESETS_PATH
    raw = yaml.safe_load(p.read_text()) or {}

    default_tier = _validate_tier(
        (raw.get("defaults") or {}).get("tier", "M"), "defaults.tier"
    )

    benchmarks: dict[str, Tier] = {}
    for bid, tier in (raw.get("benchmarks") or {}).items():
        benchmarks[bid] = _validate_tier(tier, f"benchmarks.{bid}")

    methods: dict[str, Tier] = {}
    for m, tier in (raw.get("methods") or {}).items():
        methods[m] = _validate_tier(tier, f"methods.{m}")

    overrides: dict[tuple[str, str], Tier] = {}
    for i, entry in enumerate(raw.get("overrides") or []):
        if not isinstance(entry, dict):
            raise ValueError(f"overrides[{i}] must be a mapping, got {type(entry)}")
        try:
            bench = entry["bench"]
            method = entry["method"]
            tier = entry["tier"]
        except KeyError as e:
            raise ValueError(f"overrides[{i}] missing key {e.args[0]!r}") from None
        overrides[(str(bench), str(method))] = _validate_tier(
            tier, f"overrides[{i}].tier"
        )

    return TierRegistry(
        benchmarks=benchmarks,
        methods=methods,
        overrides=overrides,
        default_tier=default_tier,
    )


def resolve_tier(
    bench_id: str, method: str, registry: TierRegistry | None = None
) -> Tier:
    """Tier for one (bench, method) pair."""
    reg = registry if registry is not None else load_tier_registry()
    if (bench_id, method) in reg.overrides:
        return reg.overrides[(bench_id, method)]
    bt = reg.tier_for_bench(bench_id)
    mt = reg.tier_for_method(method)
    return bt if _TIER_ORDER[bt] >= _TIER_ORDER[mt] else mt
