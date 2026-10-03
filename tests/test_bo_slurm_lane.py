from __future__ import annotations

import json
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts.bo import slurm_lane


def _cells(root: Path, count: int = 2) -> list[dict[str, object]]:
    return [
        {
            "method": "alm" if index == 0 else "pal_loggap",
            "bench": "s1_sphere_track",
            "seed": 100 + index,
            "trial_dir": root / "trials" / f"cell-{index}",
            "set_overrides": {"lr": "1e-4", "epochs": "5"},
            "timeout_s": 600,
        }
        for index in range(count)
    ]


def _fake_run(stdout: str = "12345\n") -> CompletedProcess[str]:
    return CompletedProcess([], 0, stdout=stdout, stderr="")


def test_manifest_round_trip_and_single_array_submission(tmp_path, monkeypatch):
    seen: list[list[str]] = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        return _fake_run("12345;cluster\n")

    monkeypatch.setattr(slurm_lane.subprocess, "run", fake_run)
    jobid = slurm_lane.submit_cells(
        _cells(tmp_path),
        tmp_path,
        "abc123",
        {"preset": "cpu_lane", "account": "my_account", "time": "00:15:00"},
    )

    assert jobid == "12345"
    manifest_path = slurm_lane.manifest_for_job(jobid)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["git_sha"] == "abc123"
    assert manifest["cells"][1] == {
        "array_index": 1,
        "attempt": 1,
        "bench": "s1_sphere_track",
        "method": "pal_loggap",
        "seed": 101,
        "set_overrides": {"epochs": "5", "lr": "1e-4"},
        "source_index": 1,
        "timeout_s": 600.0,
        "trial_dir": str((tmp_path / "trials" / "cell-1").resolve()),
    }
    assert len(seen) == 1
    assert seen[0][0:3] == ["sbatch", "--parsable", "--array=0-1"]
    assert "--mem-per-cpu=4096M" in seen[0]
    assert "--time=00:15:00" in seen[0]
    script = Path(manifest["sbatch_script"]).read_text()
    assert "#SBATCH --mem-per-cpu=4096M" in script
    assert "cell_executor.py" not in script  # dispatcher task entrypoint reads the manifest


def test_resource_caps_and_gpu_exclusion(tmp_path, monkeypatch):
    calls: list[list[str]] = []
    monkeypatch.setattr(
        slurm_lane.subprocess,
        "run",
        lambda argv, **kwargs: calls.append(argv) or _fake_run(),
    )

    slurm_lane.submit_cells(_cells(tmp_path, 1), tmp_path, "sha", "gpu_lane")
    argv = calls[0]
    assert "--array=0-0" in argv
    assert "--mem-per-cpu=4096M" in argv
    assert "--time=04:00:00" in argv
    assert "--gpus=<GPU_TYPE>:1" in argv
    assert "--exclude=gpu-node-046" in argv

    with pytest.raises(slurm_lane.SlurmLaneError, match="4096"):
        slurm_lane.submit_cells(
            _cells(tmp_path, 1),
            tmp_path,
            "sha",
            {"mem_per_cpu_mb": 4097},
        )
    with pytest.raises(slurm_lane.SlurmLaneError, match="24:00:00"):
        slurm_lane.submit_cells(
            _cells(tmp_path, 1),
            tmp_path,
            "sha",
            {"time": "24:00:01"},
        )


def test_poll_combines_results_queue_accounting_and_vanished(
    tmp_path, monkeypatch
):
    manifest = {
        "manifest_path": str(tmp_path / "manifest.json"),
        "cells": [
            {"array_index": index, "trial_dir": str(tmp_path / f"cell-{index}")}
            for index in range(5)
        ],
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest))
    (tmp_path / "cell-0").mkdir()
    (tmp_path / "cell-0" / "result.json").write_text(
        json.dumps({"status": "ok"})
    )
    slurm_lane.register_manifest("99", manifest["manifest_path"])

    outputs = iter(
        [
            _fake_run("99_1|RUNNING\n99_2|PENDING\n"),
            _fake_run(
                "99_0|COMPLETED\n99_1|RUNNING\n99_2|PENDING\n"
                "99_3|CANCELLED by 123\n"
            ),
        ]
    )
    monkeypatch.setattr(
        slurm_lane.subprocess, "run", lambda argv, **kwargs: next(outputs)
    )

    assert slurm_lane.poll_states("99") == {
        0: "ok",
        1: "running",
        2: "pending",
        3: "infra_failure",
        4: "infra_failure",  # absent from both commands: vanished
    }


def test_completed_without_result_is_infra_failure(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "cells": [{"array_index": 0, "trial_dir": str(tmp_path / "cell")}],
            }
        )
    )
    slurm_lane.register_manifest("88", manifest_path)
    monkeypatch.setattr(
        slurm_lane.subprocess,
        "run",
        lambda argv, **kwargs: _fake_run(
            "" if argv[0] == "squeue" else "88_0|COMPLETED\n"
        ),
    )
    assert slurm_lane.poll_states("88") == {0: "infra_failure"}


def test_retry_is_bounded_and_compacts_only_failed_indices(tmp_path, monkeypatch):
    submissions: list[list[str]] = []

    def fake_run(argv, **kwargs):
        if argv[0] == "sbatch":
            submissions.append(argv)
            return _fake_run("777\n")
        if argv[0] == "squeue":
            return _fake_run("")
        return _fake_run("42_0|FAILED\n42_1|FAILED\n42_2|COMPLETED\n")

    monkeypatch.setattr(slurm_lane.subprocess, "run", fake_run)
    cells = _cells(tmp_path, 3)
    cells[0]["attempt"] = 1
    cells[1]["attempt"] = 2
    cells[2]["attempt"] = 1
    for index in (0, 1):
        Path(cells[index]["trial_dir"]).mkdir(parents=True)
    Path(cells[2]["trial_dir"]).mkdir(parents=True)
    (Path(cells[2]["trial_dir"]) / "result.json").write_text(
        json.dumps({"status": "ok"})
    )
    manifest = {
        "schema_version": 1,
        "manifest_path": str(tmp_path / "original.json"),
        "campaign_root": str(tmp_path),
        "git_sha": "sha",
        "resources": slurm_lane._normalise_resources(
            {"preset": "cpu_lane", "time": "00:15:00"}
        ),
        "cells": slurm_lane._normalise_cells(cells, tmp_path.resolve()),
    }
    Path(manifest["manifest_path"]).write_text(json.dumps(manifest))

    retry_job = slurm_lane.retry_failed(
        manifest["manifest_path"], "42", max_attempts=2
    )
    assert retry_job == "777"
    assert submissions[0][2] == "--array=0-0"
    retried = json.loads(slurm_lane.manifest_for_job("777").read_text())
    assert [cell["source_index"] for cell in retried["cells"]] == [0]
    assert retried["cells"][0]["array_index"] == 0
    assert retried["cells"][0]["attempt"] == 2

    assert (
        slurm_lane.retry_failed(
            slurm_lane.manifest_for_job("777"), "777", max_attempts=2
        )
        is None
    )


def test_task_exec_uses_manifest_cell_and_explicit_python(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    python_bin = "/scratch/venv/bin/python"
    manifest_path.write_text(
        json.dumps(
            {
                "git_sha": "abc",
                "resources": {"python_bin": python_bin},
                "cells": [
                    {
                        "method": "alm",
                        "bench": "s1_sphere_track",
                        "seed": 100,
                        "trial_dir": str(tmp_path / "trial"),
                        "timeout_s": 20,
                        "attempt": 2,
                        "set_overrides": {"epochs": "5"},
                    }
                ],
            }
        )
    )
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(
        slurm_lane.subprocess, "check_output", lambda *args, **kwargs: "abc\n"
    )
    seen = []
    monkeypatch.setattr(slurm_lane.os, "execv", lambda *args: seen.append(args))

    slurm_lane._run_task(manifest_path)

    executable, argv = seen[0]
    assert executable == python_bin
    assert argv[0:2] == [python_bin, str(slurm_lane._CELL_EXECUTOR)]
    assert argv[argv.index("--attempt") + 1] == "2"
    assert argv[argv.index("--python-bin") + 1] == python_bin
    assert argv[-2:] == ["--set", "epochs=5"]


def test_retry_scheduler_calls_are_squeue_then_sacct(tmp_path, monkeypatch):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "campaign_root": str(tmp_path),
                "git_sha": "sha",
                "resources": slurm_lane._normalise_resources(None),
                "cells": [
                    {
                        "array_index": 0,
                        "source_index": 0,
                        "attempt": 2,
                        "trial_dir": str(tmp_path / "trial"),
                    }
                ],
            }
        )
    )
    calls = []

    def fake(argv, **kwargs):
        calls.append(argv)
        return _fake_run("")

    monkeypatch.setattr(slurm_lane.subprocess, "run", fake)
    assert slurm_lane.retry_failed(manifest_path, "5", 2) is None
    assert calls[0][0] == "squeue"
    assert "-r" in calls[0]
    assert calls[1][0] == "sacct"
