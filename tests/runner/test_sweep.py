"""Unit tests for `pal sweep` CLI planning, run-row skip logic, aggregator."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _invoke(argv: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, "-m", "pal.runner.cli", "sweep", *argv]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)


def test_plan_writes_jobs_files(tmp_path: Path):
    sweep_name = "sweep_test_plan"
    import pal  # ensure pal package is importable
    pal_root = Path(pal.__file__).resolve().parent.parent
    sweep_dir = pal_root / "runs" / sweep_name
    if sweep_dir.exists():
        import shutil
        shutil.rmtree(sweep_dir)

    aux_synthetic = ["rosenbrock_eq", "two_basins", "equality_dominated"]
    skip_benches = [*aux_synthetic, "e2/urban_wind"]
    try:
        r = _invoke([
            "plan",
            "--methods", "pal_loggap,alm",
            "--seeds", "0,1",
            "--skip-benches", ",".join(skip_benches),
            "--name", sweep_name,
        ])
        assert r.returncode == 0, r.stderr
        assert sweep_dir.exists()
        manifest = json.loads((sweep_dir / "manifest.json").read_text())
        assert manifest["methods"] == ["pal_loggap", "alm"]
        assert manifest["seeds"] == [0, 1]
        assert "e2/urban_wind" in manifest["skip_benches"]

        # 43 cpu benches x 2 methods x 2 seeds = 172 rows: 42 cheap (168) + 1 std (4).
        cpu_cheap_rows = (sweep_dir / "jobs_cpu_cheap.jsonl").read_text().strip().split("\n")
        cpu_std_rows = (sweep_dir / "jobs_cpu_std.jsonl").read_text().strip().split("\n")
        assert len(cpu_cheap_rows) == 168
        assert len(cpu_std_rows) == 4
        assert manifest["n_rows_cpu_by_tier"] == {"cheap": 168, "std": 4}
        assert json.loads(cpu_cheap_rows[0])["cpu_tier"] == "cheap"
        assert json.loads(cpu_std_rows[0])["cpu_tier"] == "std"

        # GPU tiers: e1 is L (4 rows), e3/acopf_* are M (3 x 4 = 12 rows).
        gpu_s_text = (sweep_dir / "jobs_gpu_s.jsonl").read_text().strip()
        gpu_m_rows = (sweep_dir / "jobs_gpu_m.jsonl").read_text().strip().split("\n")
        gpu_l_rows = (sweep_dir / "jobs_gpu_l.jsonl").read_text().strip().split("\n")
        assert gpu_s_text == ""
        assert len(gpu_m_rows) == 12
        assert len(gpu_l_rows) == 4
        assert manifest["n_rows_gpu_by_tier"] == {"S": 0, "M": 12, "L": 4}

        first_l = json.loads(gpu_l_rows[0])
        assert first_l["bench_id"] == "e1/bwb"
        assert first_l["gpu_tier"] == "L"
        first_m = json.loads(gpu_m_rows[0])
        assert first_m["bench_id"].startswith("e3/acopf_")
        assert first_m["gpu_tier"] == "M"
    finally:
        if sweep_dir.exists():
            import shutil
            shutil.rmtree(sweep_dir)


def test_run_row_skips_existing(tmp_path: Path):
    """run-row sees a prior final.json with `feasibility_post` -> skips."""
    sweep_dir = tmp_path / "sweep"
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(parents=True)

    jobs_path = sweep_dir / "jobs_cpu.jsonl"
    jobs_path.write_text(json.dumps({
        "method": "pal_loggap", "bench_id": "rosenbrock_eq", "seed": 42,
    }) + "\n")

    prior = runs_root / "20260419T000000Z_pal_loggap_rosenbrock_eq_seed42_abcdef12"
    prior.mkdir()
    (prior / "final.json").write_text(json.dumps({
        "feasibility_post": 1.0,
        "obj_mean_post": 0.5,
    }))

    r = _invoke([
        "run-row",
        "--jobs-file", str(jobs_path),
        "--row", "0",
    ])
    assert r.returncode == 0
    assert "SKIP (prior ok)" in r.stdout


def test_run_row_skips_prior_failed(tmp_path: Path):
    """Prior failed run (status=='failed') -> skipped by default."""
    sweep_dir = tmp_path / "sweep"
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(parents=True)

    jobs_path = sweep_dir / "jobs_cpu.jsonl"
    jobs_path.write_text(json.dumps({
        "method": "pal_loggap", "bench_id": "rosenbrock_eq", "seed": 7,
    }) + "\n")

    prior = runs_root / "20260419T000000Z_pal_loggap_rosenbrock_eq_seed7_failedhex"
    prior.mkdir()
    (prior / "final.json").write_text(json.dumps({
        "status": "failed",
        "error": "OOM on purpose",
    }))

    r = _invoke([
        "run-row",
        "--jobs-file", str(jobs_path),
        "--row", "0",
    ])
    assert r.returncode == 0
    assert "SKIP (prior failed" in r.stdout


def test_aggregate_empty_sweep(tmp_path: Path):
    sweep_dir = tmp_path / "sweep_agg"
    (sweep_dir / "runs").mkdir(parents=True)
    manifest = {
        "methods": ["pal_loggap"],
        "seeds": [0],
        "bench_ids": ["rosenbrock_eq"],
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest))

    r = _invoke(["aggregate", "--sweep-dir", str(sweep_dir)])
    assert r.returncode == 0
    out = (sweep_dir / "results.md").read_text()
    assert "rosenbrock_eq" in out
    assert "| x |" in out


def test_aggregate_with_one_row(tmp_path: Path):
    sweep_dir = tmp_path / "sweep_one"
    runs_root = sweep_dir / "runs"
    runs_root.mkdir(parents=True)

    manifest = {
        "methods": ["pal_loggap"],
        "seeds": [0],
        "bench_ids": ["rosenbrock_eq"],
    }
    (sweep_dir / "manifest.json").write_text(json.dumps(manifest))

    run_dir = runs_root / "20260419T000000Z_pal_loggap_rosenbrock_eq_seed0_deadbeef"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(json.dumps({
        "method": "pal_loggap", "benchmark_id": "rosenbrock_eq", "seed": 0,
    }))
    (run_dir / "final.json").write_text(json.dumps({
        "feasibility_post": 0.95,
        "obj_mean_post": 1.23,
        "train_wall_time_s": 12.5,
        "train__fwd_samples": 50000,
        "train__bwd_samples": 40000,
        "periodic_eval__fwd_samples": 128,
        "final_eval__fwd_samples": 64,
    }))

    r = _invoke(["aggregate", "--sweep-dir", str(sweep_dir)])
    assert r.returncode == 0, r.stderr
    out = (sweep_dir / "results.md").read_text()
    assert "95%" in out
    assert "50000" in out
