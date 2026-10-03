"""Manifest/argv `precision` contract for the breaking-point campaign driver."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.bp_campaign import driver, run_task

_FAKE_WINNER = {
    "rel_path": "results/fake/confirm_winner.json",
    "sha256": "0" * 64,
    "trial_index": 15,
    "config": {"lr": "1e-4"},
}

_ARM = driver.Arm(
    label="pal_loggap_tau1e-4",
    method="pal_loggap",
    winner="pal_loggap",
    trial_index=15,
    extra_set={"tau": "1e-4"},
)


def _build_task(precision: str = driver.DEFAULT_PRECISION, seed: int = 0,
                 campaign_root: Path | None = None) -> dict:
    return driver.build_task(
        _ARM, _FAKE_WINNER, "k0", seed,
        campaign_root or Path("/scratch/bp-campaign-fp64"),
        precision,
    )


def test_build_task_appends_precision_flag_to_command():
    task = _build_task(precision="fp64")
    assert task["precision"] == "fp64"
    assert task["command"][-2:] == ["--precision", "fp64"]


@pytest.mark.parametrize("precision", driver.PRECISION_CHOICES)
def test_build_task_precision_roundtrips_for_every_choice(precision):
    task = _build_task(precision=precision)
    assert task["precision"] == precision
    assert task["command"][-2:] == ["--precision", precision]


def test_default_precision_is_fp64_for_this_campaign():
    assert driver.DEFAULT_PRECISION == "fp64"
    task = _build_task()
    assert task["precision"] == "fp64"


def test_build_manifest_records_precision_in_protocol_and_per_task(monkeypatch):
    monkeypatch.setattr(driver, "load_winner", lambda winner, trial_index: _FAKE_WINNER)
    root = Path("/scratch/bp-campaign-fp64")
    manifest = driver.build_manifest(root, (_ARM,), ("k0",), (0, 1), precision="fp32")
    assert manifest["protocol"]["precision"] == "fp32"
    assert len(manifest["tasks"]) == 2
    assert all(t["precision"] == "fp32" for t in manifest["tasks"])
    assert all(t["command"][-2:] == ["--precision", "fp32"] for t in manifest["tasks"])


def test_default_campaign_root_is_fp64_scratch_path():
    assert driver.DEFAULT_CAMPAIGN_ROOT.name == "bp-campaign-fp64"


def test_generate_cli_default_precision_is_fp64():
    args = driver.build_parser().parse_args(["generate"])
    assert args.precision == "fp64"


def test_generate_cli_rejects_unknown_precision():
    with pytest.raises(SystemExit):
        driver.build_parser().parse_args(["generate", "--precision", "int8"])


def test_generate_cli_accepts_every_precision_choice():
    for precision in driver.PRECISION_CHOICES:
        args = driver.build_parser().parse_args(["generate", "--precision", precision])
        assert args.precision == precision


def test_verify_command_matches_a_freshly_built_task():
    task = _build_task(precision="fp64")
    cmd = run_task.verify_command(task)
    assert cmd[1:] == task["command"][1:]
    assert "--precision" in cmd
    assert cmd[cmd.index("--precision") + 1] == "fp64"


def test_verify_command_missing_precision_field_raises_clear_error():
    """An old manifest without `precision` must fail loudly, not default silently."""
    task = _build_task(precision="fp64")
    del task["precision"]
    with pytest.raises(driver.DriverError, match="precision"):
        run_task.verify_command(task)


def test_verify_command_detects_precision_drift_from_recorded_command():
    """If `precision` and `command` disagree, verify_command must hard-fail."""
    task = _build_task(precision="fp32")
    task["precision"] = "fp64"  # drifted from the recorded command
    with pytest.raises(driver.DriverError, match="drifted"):
        run_task.verify_command(task)


def test_generate_cpus_override_applies_to_one_arm(tmp_path, monkeypatch) -> None:
    """`--cpus fsnet=16` changes only fsnet's -c / BENCH_THREADS and is recorded."""
    monkeypatch.setattr(driver, "CPUS_OVERRIDE", {})
    root = tmp_path / "root"
    rc = driver.main(["--campaign-root", str(root), "generate", "--variants", "k0",
                      "--seeds", "0", "--cpus", "fsnet=16"])
    assert rc == 0
    fsnet = (root / "sbatch" / "bp_fsnet_k0.sbatch").read_text()
    alm = (root / "sbatch" / "bp_alm_k0.sbatch").read_text()
    assert "#SBATCH -c 16" in fsnet and "BENCH_THREADS=16" in fsnet
    assert f"#SBATCH -c {driver.CPUS_PER_TASK}" in alm
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["protocol"]["cpus_per_task"]["fsnet"] == 16
    assert manifest["protocol"]["cpus_per_task"]["alm"] == driver.CPUS_PER_TASK


def test_generate_cpus_override_rejects_unknown_arm(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(driver, "CPUS_OVERRIDE", {})
    rc = driver.main(["--campaign-root", str(tmp_path / "r"), "generate",
                      "--variants", "k0", "--seeds", "0", "--cpus", "nope=4"])
    assert rc == 2
