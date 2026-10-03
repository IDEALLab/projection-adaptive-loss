"""Tests for the GPU tier registry and `pal sweep plan` tier partitioning."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pal.sweep import (
    load_tier_registry,
    resolve_tier,
)


def test_default_registry_loads():
    reg = load_tier_registry()
    assert reg.default_tier in ("S", "M", "L")
    assert reg.benchmarks, "benchmarks section must be populated"
    assert reg.methods, "methods section must be populated"


def test_default_registry_known_tiers():
    """Core tier assignments: e1/bwb is L (peaks ~78.5 GB on A100-80GB)."""
    reg = load_tier_registry()
    assert reg.tier_for_bench("rosenbrock_eq") == "S"
    assert reg.tier_for_bench("e1/bwb") == "L"
    assert reg.tier_for_bench("e2/urban_wind") == "L"
    assert reg.tier_for_method("alm") == "S"
    assert reg.tier_for_method("fsnet") == "M"


def test_unknown_falls_back_to_default():
    reg = load_tier_registry()
    assert reg.tier_for_bench("b999_nonexistent") == reg.default_tier
    assert reg.tier_for_method("__not_a_method__") == reg.default_tier


def test_resolve_max_rule_takes_higher_tier():
    reg = load_tier_registry()
    # synthetic (S) x fsnet (M) -> M
    assert resolve_tier("rosenbrock_eq", "fsnet", reg) == "M"
    # e1 (L) x alm (S) -> L
    assert resolve_tier("e1/bwb", "alm", reg) == "L"
    # e2 (L) x anything -> L
    assert resolve_tier("e2/urban_wind", "alm", reg) == "L"
    assert resolve_tier("e2/urban_wind", "fsnet", reg) == "L"
    # synthetic (S) x alm (S) -> S
    assert resolve_tier("rosenbrock_eq", "alm", reg) == "S"


def test_override_takes_precedence(tmp_path: Path):
    yaml_text = """\
defaults:
  tier: M
benchmarks:
  b_test: S
methods:
  m_test: S
overrides:
  - bench: b_test
    method: m_test
    tier: L
"""
    p = tmp_path / "presets.yaml"
    p.write_text(yaml_text)
    reg = load_tier_registry(p)
    # Max rule would say S, override forces L.
    assert resolve_tier("b_test", "m_test", reg) == "L"
    assert resolve_tier("b_test", "other", reg) == "M"  # other defaults to M


def test_invalid_tier_rejected(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text("benchmarks:\n  b_test: XXL\n")
    with pytest.raises(ValueError, match="invalid tier"):
        load_tier_registry(p)


def test_override_missing_key_rejected(tmp_path: Path):
    p = tmp_path / "bad.yaml"
    p.write_text(
        "overrides:\n  - bench: b_test\n    tier: S\n"  # missing 'method'
    )
    with pytest.raises(ValueError, match="method"):
        load_tier_registry(p)


def test_sweep_plan_writes_per_tier_files(tmp_path: Path):
    """`pal sweep plan` partitions gpu rows into jobs_gpu_{s,m,l}.jsonl with tier counts."""
    import shutil
    import uuid

    from pal.runner.sweep import _cmd_plan, _parse_args

    sweep_name = f"tier_test_{uuid.uuid4().hex[:8]}"

    args = _parse_args([
        "plan",
        "--methods", "alm,fsnet",
        "--seeds", "0",
        "--skip-benches", ",".join([
            # Keep only rosenbrock_eq (S, cpu), e1/bwb (L, gpu), e2/urban_wind (L, gpu).
            "two_basins", "equality_dominated",
            "s1_sphere_track", "s2_active_set_switch", "s3_illcond_tube",
            "s4_qv_coupling", "s5_overdetermined", "s6_redundant_ineq",
            *[f"curvature_hinge_k{i}" for i in range(11)],
            *[f"curvature_sine_k{i}" for i in range(11)],
            *[f"curvature_warp_k{i}" for i in range(14)],
            "e3/acopf_ieee30", "e3/acopf_ieee57", "e3/acopf_ieee118",
            "e4/chip_layout",
        ]),
        "--name", sweep_name,
    ])
    rc = _cmd_plan(args)
    assert rc == 0

    pal_root = Path(__file__).resolve().parents[1]
    sweep_dir = pal_root / "runs" / sweep_name
    try:
        manifest = json.loads((sweep_dir / "manifest.json").read_text())

        assert manifest["schema_version"] == 3

        cpu_tiers = manifest["n_rows_cpu_by_tier"]
        assert cpu_tiers == {"cheap": 2, "std": 0}  # rosenbrock_eq x {alm, fsnet} x 1 seed

        tiers = manifest["n_rows_gpu_by_tier"]
        # e1/bwb (L) x {alm, fsnet} -> L, L            -> 2 L rows
        # e2/urban_wind (L) x {alm, fsnet} -> L, L    -> 2 L rows
        assert tiers["S"] == 0
        assert tiers["M"] == 0
        assert tiers["L"] == 4

        for t, n in tiers.items():
            p = sweep_dir / f"jobs_gpu_{t.lower()}.jsonl"
            assert p.exists()
            lines = [ln for ln in p.read_text().splitlines() if ln.strip()]
            assert len(lines) == n
            for ln in lines:
                row = json.loads(ln)
                assert row["gpu_tier"] == t
    finally:
        shutil.rmtree(sweep_dir, ignore_errors=True)
