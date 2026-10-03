"""Tests for `pal/runner/run_config.py`, YAML run-config expander."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pal.runner.run_config import (
    expand_config_to_argv,
    extract_config_from_argv,
)


def _write(path: Path, body: str) -> Path:
    path.write_text(body)
    return path


def test_minimal_config_emits_required_flags(tmp_path):
    cfg = _write(
        tmp_path / "min.yaml",
        """
methods: [pal_loggap, alm]
benchmark: e4/chip_layout
seeds: [0, 1, 2]
""",
    )
    argv = expand_config_to_argv(cfg)
    assert argv == [
        "--method", "pal_loggap,alm",
        "--benchmarks", "e4/chip_layout",
        "--seeds", "0,1,2",
    ]


def test_scalar_and_bool_flags(tmp_path):
    cfg = _write(
        tmp_path / "full.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
seeds: 0
epochs: 2000
batch_size: 32
lr: 1.0e-4
multi_start: 20
device: cuda
eval_points: pal/configs/eval_points/e4.json
measure_repair_mem: true
wandb: false
""",
    )
    argv = expand_config_to_argv(cfg)
    # Order: methods, benchmarks, seeds, scalars, then booleans.
    assert "--epochs" in argv
    assert argv[argv.index("--epochs") + 1] == "2000"
    assert "--batch-size" in argv and argv[argv.index("--batch-size") + 1] == "32"
    assert "--lr" in argv
    assert "--multi-start" in argv and argv[argv.index("--multi-start") + 1] == "20"
    assert "--device" in argv and argv[argv.index("--device") + 1] == "cuda"
    assert "--measure-repair-mem" in argv
    assert "--wandb" not in argv  # false -> omitted


def test_env_block_writes_os_environ(tmp_path, monkeypatch):
    monkeypatch.delenv("E1_ALT_HI", raising=False)
    cfg = _write(
        tmp_path / "env.yaml",
        """
methods: [pal_loggap]
benchmark: e1/bwb
seeds: 0
env:
  E1_ALT_HI: 4000.0
  BWB_FAST_SDF: 1
""",
    )
    expand_config_to_argv(cfg)
    assert os.environ["E1_ALT_HI"] == "4000.0"
    assert os.environ["BWB_FAST_SDF"] == "1"


def test_unknown_keys_raise(tmp_path):
    cfg = _write(
        tmp_path / "bad.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
seeds: 0
epoch: 2000   # typo!
""",
    )
    with pytest.raises(ValueError, match="unknown keys"):
        expand_config_to_argv(cfg)


def test_benchmark_xor_benchmarks(tmp_path):
    cfg = _write(
        tmp_path / "both.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
benchmarks: [e1/bwb]
seeds: 0
""",
    )
    with pytest.raises(ValueError, match="benchmark.*benchmarks"):
        expand_config_to_argv(cfg)


def test_benchmarks_list_joins_with_comma(tmp_path):
    cfg = _write(
        tmp_path / "multi.yaml",
        """
methods: [pal_loggap, alm]
benchmarks: [e1/bwb, e2/urban_wind]
seeds: 0
""",
    )
    argv = expand_config_to_argv(cfg)
    assert "--benchmarks" in argv
    assert argv[argv.index("--benchmarks") + 1] == "e1/bwb,e2/urban_wind"


def test_extract_config_from_argv_with_config(tmp_path):
    cfg = _write(
        tmp_path / "x.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
seeds: 0
epochs: 2000
""",
    )
    argv = ["run", "--config", str(cfg), "--epochs", "100"]
    cleaned, yaml_argv = extract_config_from_argv(argv)
    assert cleaned == ["run", "--epochs", "100"]
    assert "--epochs" in yaml_argv
    assert yaml_argv[yaml_argv.index("--epochs") + 1] == "2000"


def test_extract_config_from_argv_no_config():
    argv = ["run", "--method", "pal_loggap", "--benchmarks", "e4"]
    cleaned, yaml_argv = extract_config_from_argv(argv)
    assert cleaned == argv
    assert yaml_argv == []


def test_missing_config_path_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        expand_config_to_argv(tmp_path / "does_not_exist.yaml")


def test_empty_yaml_is_empty_argv(tmp_path):
    cfg = _write(tmp_path / "empty.yaml", "")
    assert expand_config_to_argv(cfg) == []


def _mock_parser():
    """Minimal argparse parser mimicking the `run` subcommand's override surface."""
    import argparse
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="verb", required=True)
    run = sub.add_parser("run")
    run.add_argument("--config", default=None)
    run.add_argument("--method", default=None)
    run.add_argument("--benchmarks", default=None)
    run.add_argument("--seeds", default="0")
    run.add_argument("--epochs", type=int, default=42)
    run.add_argument("--multi-start", type=int, default=None)
    run.add_argument("--measure-repair-mem", action="store_true")
    return p


def _dispatch(argv: list[str]):
    """Mimic main()'s --config interception, then parse with the mock parser."""
    raw = list(argv)
    if "--config" in raw:
        cleaned, yaml_argv = extract_config_from_argv(raw)
        if cleaned and cleaned[0] in {"run", "eval"}:
            raw = [cleaned[0]] + yaml_argv + cleaned[1:]
        else:
            raw = yaml_argv + cleaned
    return _mock_parser().parse_args(raw)


def test_yaml_supplies_defaults_when_cli_silent(tmp_path):
    cfg = _write(
        tmp_path / "x.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
seeds: [0, 1, 2]
epochs: 2000
multi_start: 20
measure_repair_mem: true
""",
    )
    args = _dispatch(["run", "--config", str(cfg)])
    assert args.method == "pal_loggap"
    assert args.benchmarks == "e4/chip_layout"
    assert args.seeds == "0,1,2"
    assert args.epochs == 2000
    assert args.multi_start == 20
    assert args.measure_repair_mem is True


def test_explicit_cli_wins_over_yaml(tmp_path):
    """Critical regression: --epochs from CLI must override --epochs from YAML."""
    cfg = _write(
        tmp_path / "x.yaml",
        """
methods: [pal_loggap]
benchmark: e4/chip_layout
seeds: 0
epochs: 2000
""",
    )
    args = _dispatch(["run", "--config", str(cfg), "--epochs", "100"])
    assert args.epochs == 100
    assert args.method == "pal_loggap"
    assert args.seeds == "0"


def test_cli_can_add_flags_yaml_omitted(tmp_path):
    cfg = _write(
        tmp_path / "x.yaml",
        """
methods: [ipopt]
benchmark: e3/acopf_ieee57
seeds: 0
""",
    )
    args = _dispatch(
        ["run", "--config", str(cfg), "--multi-start", "10"]
    )
    assert args.multi_start == 10  # only CLI provides this


def test_all_shipped_run_yamls_parse_via_loader():
    """Smoke-load every config under pal/configs/runs/*.yaml."""
    import glob
    repo = Path(__file__).resolve().parents[1]
    yamls = sorted(glob.glob(str(repo / "pal" / "configs" / "runs" / "*.yaml")))
    assert len(yamls) >= 9, f"expected >=9 run YAMLs, found {len(yamls)}"
    for y in yamls:
        argv = expand_config_to_argv(y)
        assert "--method" in argv, f"{y}: missing methods"
        assert "--benchmarks" in argv, f"{y}: missing benchmark(s)"
        assert "--seeds" in argv, f"{y}: missing seeds"
