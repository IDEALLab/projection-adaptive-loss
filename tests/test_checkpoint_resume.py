"""Epoch-level checkpoint + resume for pal_loggap and alm.

A 3+3 epoch resumed run must match a straight 6-epoch run bit-for-bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from pal.baselines import ALMConfig, ALMSolver
from pal.benchmarks import get as get_benchmark
from pal.method import PALLogGapConfig, PALLogGapSolver

_BENCH = "s1_sphere_track"


class _NullLogger:
    def log_config(self, **kw): pass
    def log_step(self, step, **kw): pass
    def log_projection_trajectory(self, step, phase, traj): pass
    def log_artifact(self, step, name, payload): pass
    def log_final(self, **kw): pass
    def finish(self, status="ok", error=None): pass


def _load_ckpt(d: Path) -> dict:
    return torch.load(d / "checkpoint_latest.pt", weights_only=False)


def _assert_state_matches(a: dict, b: dict, keys: list[str]) -> None:
    for sd_key in keys:
        sa, sb = a[sd_key], b[sd_key]
        for k in sa:
            assert torch.allclose(sa[k], sb[k], atol=1e-12, rtol=0), (
                f"{sd_key}[{k}] mismatch: max |Delta| = "
                f"{(sa[k] - sb[k]).abs().max().item():.3e}"
            )


def _make_solver(kind: str, epochs: int, bench: str, tds: int):
    if kind == "loggap":
        cfg = PALLogGapConfig(
            epochs=epochs, batch_size=16, seed=0, device="cpu",
            eval_every=0, train_dataset_size=tds,
        )
        return PALLogGapSolver(cfg)
    cfg = ALMConfig(
        epochs=epochs, batch_size=16, lr=1e-3, seed=0, device="cpu",
        train_dataset_size=tds,
    )
    return ALMSolver(cfg)


# (kind, multiplier-state keys): loggap -> MetricLogGap trackers, alm -> ALMState.
_METHODS = [
    ("loggap", ["lg", "lg_disp"]),
    ("alm", ["alm_state"]),
]
_SAMPLING_CASES = [
    ("s1_sphere_track", 32),   # fixed-pool path
    ("rosenbrock_eq", 0),   # default path, zeta_dim > 0
]


@pytest.mark.parametrize("kind,state_keys", _METHODS)
@pytest.mark.parametrize("bench,tds", _SAMPLING_CASES)
def test_resume_matches_straight(
    tmp_path: Path, kind: str, state_keys: list[str], bench: str, tds: int
) -> None:
    d_straight = tmp_path / "straight"
    d_straight.mkdir()
    r_straight = _make_solver(kind, 6, bench, tds).train(
        get_benchmark(bench), seed=0, logger=_NullLogger(),
        checkpoint_every=6, checkpoint_dir=d_straight,
    )

    d_part = tmp_path / "part"
    d_part.mkdir()
    _make_solver(kind, 3, bench, tds).train(
        get_benchmark(bench), seed=0, logger=_NullLogger(),
        checkpoint_every=3, checkpoint_dir=d_part,
    )
    ckpt3 = _load_ckpt(d_part)
    assert ckpt3["epoch"] == 3

    d_resume = tmp_path / "resume"
    d_resume.mkdir()
    r_resume = _make_solver(kind, 6, bench, tds).train(
        get_benchmark(bench), seed=0, logger=_NullLogger(),
        checkpoint_every=6, checkpoint_dir=d_resume, resume_state=ckpt3,
    )

    ms_a, ms_b = r_straight.model_state, r_resume.model_state
    for k in ms_a:
        assert torch.allclose(ms_a[k], ms_b[k], atol=1e-12, rtol=0)
    _assert_state_matches(_load_ckpt(d_straight), _load_ckpt(d_resume), state_keys)


def _run_cli(tmp_path: Path, epochs: int) -> Path:
    from pal.runner.cli import main

    rc = main([
        "run",
        "--method", "pal_loggap",
        "--benchmarks", _BENCH,
        "--seeds", "0",
        "--epochs", str(epochs),
        "--batch-size", "16",
        "--eval-every", "0",
        "--device", "cpu",
        "--auto-resume",
        "--checkpoint-every", "2",
        "--no-final-eval",
        "--runs-root", str(tmp_path),
    ])
    assert rc == 0
    return tmp_path / f"{_BENCH}__pal_loggap__seed0"


def test_cli_auto_resume_reuses_dir_and_appends(tmp_path: Path) -> None:
    run_dir = _run_cli(tmp_path, epochs=2)
    assert run_dir.is_dir(), "deterministic run dir not created"
    ckpt = _load_ckpt(run_dir)
    assert (run_dir / "checkpoint_latest.pt").exists()
    assert ckpt["epoch"] == 2  # written on epoch 2 (final)

    metrics = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics) == 2
    first_line = metrics[0]
    first_rec = json.loads(first_line)
    assert first_rec["step"] == 1

    # Same flags, more epochs: reuse the dir, resume, append to metrics.jsonl.
    run_dir2 = _run_cli(tmp_path, epochs=4)
    assert run_dir2 == run_dir, "auto-resume did not reuse the deterministic dir"

    ckpt2 = _load_ckpt(run_dir)
    assert ckpt2["epoch"] == 4

    metrics2 = (run_dir / "metrics.jsonl").read_text().splitlines()
    assert len(metrics2) == 4, f"expected 4 lines, got {len(metrics2)}"
    assert metrics2[0] == first_line, "earlier metrics.jsonl line was mutated"
    assert json.loads(metrics2[2])["step"] == 3
    assert json.loads(metrics2[3])["step"] == 4
