"""create_run_dir unit tests."""

from __future__ import annotations

import json
from pathlib import Path

from pal.tracking.run_dir import create_run_dir, write_config_json


def test_create_run_dir_creates_unique_directory(tmp_path: Path) -> None:
    d1 = create_run_dir("pal_loggap", "rosenbrock_eq", 0, runs_root=tmp_path)
    d2 = create_run_dir("pal_loggap", "rosenbrock_eq", 0, runs_root=tmp_path)
    assert d1.is_dir()
    assert d2.is_dir()
    assert d1 != d2
    assert "pal_loggap" in d1.name and "rosenbrock_eq" in d1.name and "seed0" in d1.name


def test_create_run_dir_sanitizes_family_variant_ids(tmp_path: Path) -> None:
    d = create_run_dir("pal_loggap", "e3/acopf_ieee30", 2, runs_root=tmp_path)
    assert "e3-acopf_ieee30" in d.name
    assert "/" not in d.name


def test_create_run_dir_with_restart_shard_suffix(tmp_path: Path) -> None:
    d = create_run_dir(
        "ipopt", "e3/acopf_ieee57", 1, runs_root=tmp_path, restart_shard=(3, 20)
    )
    assert "_seed1_r3_of20_" in d.name
    # Sibling shards never collide on dir name (uuid8 differs).
    d2 = create_run_dir(
        "ipopt", "e3/acopf_ieee57", 1, runs_root=tmp_path, restart_shard=(3, 20)
    )
    assert d != d2


def test_write_config_json_handles_nested_data(tmp_path: Path) -> None:
    d = create_run_dir("pal_loggap", "rosenbrock_eq", 0, runs_root=tmp_path)
    write_config_json(d, {"method": "pal_loggap", "hp": {"lr": 1e-3}})
    cfg = json.loads((d / "config.json").read_text())
    assert cfg == {"hp": {"lr": 1e-3}, "method": "pal_loggap"}
