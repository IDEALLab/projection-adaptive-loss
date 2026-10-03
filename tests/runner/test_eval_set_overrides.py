"""`pal eval --set`: override application, precedence, and provenance."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from pal.runner.cli import (
    _build_cfg_from_hparams,
    _eval_one,
    _parse_args,
    _resolve_eval_cfg_overrides,
)

RUN_NAME = "20260430T120000Z_alm_e3_acopf_seed0_deadbeef"


def _write_source_run(
    runs_root: Path,
    *,
    protocol: str = "paper-faithful",
    bench_id: str = "e3/acopf_case57",
) -> Path:
    """Minimal trained-run dir: enough to rehydrate a config, not to run inference."""
    run_dir = runs_root / RUN_NAME
    run_dir.mkdir(parents=True)
    config = {
        "method": "alm",
        "benchmark_id": bench_id,
        "seed": 0,
        "protocol": protocol,
        "hparams": {"seed": 0, "device": "cpu", "epochs": 7, "lr": 0.5},
    }
    (run_dir / "config.json").write_text(json.dumps(config))
    (run_dir / "final.json").write_text(json.dumps({"train_wall_time_s": 12.5}))
    (run_dir / "model.pt").write_bytes(b"not-a-real-checkpoint")
    return run_dir


def _eval_args(**kw) -> argparse.Namespace:
    base = dict(
        device="cpu",
        n_eval=None,
        inference_trajectory_max=0,
        inference_trajectory_downsample=1,
        method_override="alm_bolton",
        set_overrides=[],
        viz_final=False,
        viz_n=1,
    )
    base.update(kw)
    return argparse.Namespace(**base)


def _sibling_config(runs_root: Path) -> dict:
    path = runs_root / f"{RUN_NAME}__as_alm_bolton" / "config.json"
    return json.loads(path.read_text())


def _run_eval_expecting_failure(runs_root: Path, args: argparse.Namespace) -> None:
    """`_eval_one` dies on the stub model.pt after provenance is written."""
    with pytest.raises(Exception):  # noqa: B017, any post-write-back failure
        _eval_one(RUN_NAME, args, runs_root)


def test_eval_subparser_accepts_repeatable_set():
    args = _parse_args([
        "eval", "--run-id", RUN_NAME,
        "--method-override", "alm_bolton",
        "--set", "proj_lambda_min=1e-2",
        "--set", "epochs=3",
    ])
    assert args.set_overrides == ["proj_lambda_min=1e-2", "epochs=3"]


def test_eval_subparser_set_defaults_empty():
    args = _parse_args(["eval", "--run-id", RUN_NAME])
    assert args.set_overrides == []


def test_unknown_key_raises():
    cfg = _build_cfg_from_hparams("alm_bolton", {})
    with pytest.raises(ValueError, match="not a field of"):
        _resolve_eval_cfg_overrides(cfg, ["definitely_not_a_field=1"], {})


def test_malformed_pair_raises():
    cfg = _build_cfg_from_hparams("alm_bolton", {})
    with pytest.raises(ValueError, match="KEY=VALUE"):
        _resolve_eval_cfg_overrides(cfg, ["proj_lambda_min"], {})


def test_known_key_applied_and_reported():
    cfg = _build_cfg_from_hparams("alm_bolton", {"epochs": 7})
    applied = _resolve_eval_cfg_overrides(cfg, ["epochs=3", "lr=0.25"], {})
    assert cfg.epochs == 3
    assert cfg.lr == 0.25
    assert applied == {"epochs": 3, "lr": 0.25}


def test_set_beats_forced_value():
    cfg = _build_cfg_from_hparams("alm_bolton", {})
    applied = _resolve_eval_cfg_overrides(
        cfg, ["proj_lambda_min=1e-2"], {"proj_lambda_min": 1e-4},
    )
    assert cfg.proj_lambda_min == 1e-2
    assert applied == {"proj_lambda_min": 1e-2}


def test_forced_value_reported_when_no_set():
    cfg = _build_cfg_from_hparams("alm_bolton", {})
    applied = _resolve_eval_cfg_overrides(cfg, [], {"proj_lambda_min": 1e-4})
    assert cfg.proj_lambda_min == 1e-4
    assert applied == {"proj_lambda_min": 1e-4}


def test_write_back_lands_in_sibling_config(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    _run_eval_expecting_failure(
        runs_root, _eval_args(set_overrides=["epochs=3", "lr=0.25"]),
    )

    cfg = _sibling_config(runs_root)
    assert cfg["eval_set_overrides"] == ["epochs=3", "lr=0.25"]
    assert cfg["hparams"]["epochs"] == 3
    assert cfg["hparams"]["lr"] == 0.25
    # Untouched source hparams survive, as does the override provenance.
    assert cfg["hparams"]["seed"] == 0
    assert cfg["method"] == "alm_bolton"
    assert cfg["override_source_run_id"] == RUN_NAME
    assert cfg["override_source_method"] == "alm"


def test_set_beats_e3_proj_lambda_min_in_config_and_cfg(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    _run_eval_expecting_failure(
        runs_root, _eval_args(set_overrides=["proj_lambda_min=1e-2"]),
    )

    cfg = _sibling_config(runs_root)
    assert cfg["hparams"]["proj_lambda_min"] == 1e-2
    assert cfg["eval_set_overrides"] == ["proj_lambda_min=1e-2"]


def test_e3_special_case_written_back_when_no_set(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    _run_eval_expecting_failure(runs_root, _eval_args())

    cfg = _sibling_config(runs_root)
    assert cfg["hparams"]["proj_lambda_min"] == 1e-4
    assert cfg["eval_set_overrides"] == []


def test_no_set_leaves_hparams_untouched(tmp_path: Path):
    """A `--set`-free override eval changes nothing but the empty `eval_set_overrides` key."""
    runs_root = tmp_path / "runs"
    src = _write_source_run(runs_root, protocol="synthetic")
    src_hparams = json.loads((src / "config.json").read_text())["hparams"]
    _run_eval_expecting_failure(runs_root, _eval_args())

    cfg = _sibling_config(runs_root)
    assert cfg["hparams"] == src_hparams
    assert cfg["eval_set_overrides"] == []


def test_unknown_key_raises_through_eval_one(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    with pytest.raises(ValueError, match="not a field of"):
        _eval_one(RUN_NAME, _eval_args(set_overrides=["nope=1"]), runs_root)


def test_set_without_method_override_raises(tmp_path: Path):
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    args = _eval_args(method_override=None, set_overrides=["epochs=3"])
    with pytest.raises(ValueError, match="requires --method-override"):
        _eval_one(RUN_NAME, args, runs_root)


def test_set_with_noop_method_override_raises(tmp_path: Path):
    """`--method-override alm` on an alm run normalizes to None: same refusal."""
    runs_root = tmp_path / "runs"
    _write_source_run(runs_root)
    args = _eval_args(method_override="alm", set_overrides=["epochs=3"])
    with pytest.raises(ValueError, match="requires --method-override"):
        _eval_one(RUN_NAME, args, runs_root)
