"""All 6 e1/bwb runtime artifacts ship in-repo under ``_artifacts_data/``."""

from __future__ import annotations

from pal.benchmarks.engineering.e1_bwb._artifacts import (
    _BENCH_ID,
    check_runtime_artifacts,
)


def test_check_runtime_artifacts_passes():
    """All 6 in-repo artifacts resolve cleanly, no raise."""
    check_runtime_artifacts()


def test_registry_has_e1_entry():
    from pal.artifacts.registry import REGISTRY

    assert _BENCH_ID in REGISTRY
    assert all(ref.location == "in_repo" for ref in REGISTRY[_BENCH_ID])
