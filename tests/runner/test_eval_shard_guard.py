"""`pal eval` must refuse to run on an IPOPT restart-shard run dir."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pal.runner.cli import _eval_one


def _write_shard_run(run_dir: Path, *, restart_shard: list[int] | None) -> None:
    run_dir.mkdir(parents=True)
    hparams: dict = {"seed": 0, "device": "cpu", "multi_start": 20}
    if restart_shard is not None:
        hparams["restart_shard"] = restart_shard
    config = {
        "method": "ipopt",
        "benchmark_id": "rosenbrock_eq",
        "seed": 0,
        "hparams": hparams,
    }
    (run_dir / "config.json").write_text(json.dumps(config))


def test_eval_refuses_sharded_run_dir(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    name = "20260430T120000Z_ipopt_rosenbrock_eq_seed0_r3_of5_deadbeef"
    _write_shard_run(runs_root / name, restart_shard=[3, 5])

    args = argparse.Namespace(device="cpu", n_eval=None, inference_trajectory_max=0)
    with pytest.raises(ValueError, match="restart_shard"):
        _eval_one(name, args, runs_root)


def test_eval_runs_on_unsharded_run_dir_does_not_raise_for_shard_reason(
    tmp_path: Path,
) -> None:
    """The guard fires only when restart_shard is set."""
    runs_root = tmp_path / "runs"
    name = "20260430T120000Z_ipopt_rosenbrock_eq_seed0_deadbeef"
    _write_shard_run(runs_root / name, restart_shard=None)

    args = argparse.Namespace(device="cpu", n_eval=None, inference_trajectory_max=0)
    try:
        _eval_one(name, args, runs_root)
    except ValueError as exc:
        assert "restart_shard" not in str(exc), (
            "shard guard should not fire on an unsharded run"
        )
    except Exception:
        pass
