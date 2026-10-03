"""Tests for scripts/render_engineering.py on a synthetic gap parquet."""

from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

_REPO = Path(__file__).resolve().parents[1]
_SCRIPT_RENDER = _REPO / "scripts" / "render_engineering.py"
_SCRIPT_AGG = _REPO / "scripts" / "aggregate_engineering.py"


def _make_gap_df(rows: list[dict]) -> pl.DataFrame:
    """Build a gap parquet DF with the canonical schema."""
    cols = [
        "bench", "method", "seed", "obj_post", "obj_ipopt", "gap_rel",
        "feas_post", "viol_max_post",
        "repair_peak_mem_bytes_mean", "repair_peak_mem_bytes_std",
    ]
    return pl.DataFrame({c: [r.get(c) for r in rows] for c in cols})


def _write_synth(tmp_path: Path) -> Path:
    """3 methods x 3 seeds; pal beats ipopt by ~1%, alm by ~5%, dc3 worse."""
    rows = []
    bench = "b00_test/v1"
    for seed in (0, 1, 2):
        ipopt_obj = 100.0 + 0.1 * seed
        rows.append(dict(
            bench=bench, method="ipopt", seed=seed,
            obj_post=ipopt_obj, obj_ipopt=ipopt_obj, gap_rel=0.0,
            feas_post=1.0, viol_max_post=1e-6,
            repair_peak_mem_bytes_mean=math.nan,
            repair_peak_mem_bytes_std=math.nan,
        ))
        pal_obj = ipopt_obj * 1.01
        rows.append(dict(
            bench=bench, method="pal_loggap", seed=seed,
            obj_post=pal_obj, obj_ipopt=ipopt_obj,
            gap_rel=(pal_obj - ipopt_obj) / abs(ipopt_obj),
            feas_post=0.95 if seed != 1 else 0.90,
            viol_max_post=8.0e-5 + 1e-5 * seed,
            repair_peak_mem_bytes_mean=0.5 * (1024 ** 3),
            repair_peak_mem_bytes_std=0.05 * (1024 ** 3),
        ))
        alm_obj = ipopt_obj * 1.05
        rows.append(dict(
            bench=bench, method="alm", seed=seed,
            obj_post=alm_obj, obj_ipopt=ipopt_obj,
            gap_rel=(alm_obj - ipopt_obj) / abs(ipopt_obj),
            feas_post=0.80,
            viol_max_post=5.0e-4,
            repair_peak_mem_bytes_mean=0.0,  # ALM emits 0 for schema parity
            repair_peak_mem_bytes_std=math.nan,
        ))
    df = _make_gap_df(rows)
    out = tmp_path / "gap.parquet"
    df.write_parquet(out)
    return out


def _run(*argv: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, *argv],
        capture_output=True,
        text=True,
        cwd=_REPO,
    )


def test_render_writes_md_and_tex(tmp_path):
    parquet = _write_synth(tmp_path)
    out_dir = tmp_path / "out"
    proc = _run(
        str(_SCRIPT_RENDER),
        "--bench", "b00_test/v1",
        "--gap-parquet", str(parquet),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr

    md = (out_dir / "b00_test_v1.md").read_text()
    tex = (out_dir / "b00_test_v1.tex").read_text()

    assert "| `pal_loggap` |" in md
    assert "| `alm` |" in md
    assert "| `ipopt` |" in md
    assert md.find("pal_loggap") < md.find("alm") < md.find("ipopt")
    # 3 seeds with constant gap -> std=0.0000; mean=+0.0100 (PAL) / +0.0500 (ALM).
    assert "+0.0100" in md
    assert "+0.0500" in md
    assert "0 (ref)" in md
    # Mem column: PAL ~ 0.5 GiB; ALM 0 GiB (always-on schema); IPOPT blank.
    assert "0.50" in md
    assert "93%" in md or "94%" in md  # pal mean across {0.95, 0.90, 0.95}

    assert "\\begin{tabular}" in tex
    assert "\\end{tabular}" in tex
    assert "$\\pm$" in tex
    assert "\\%" in tex


def test_render_handles_single_seed_dropping_std(tmp_path):
    rows = [
        dict(bench="b00/v1", method="pal_loggap", seed=0,
             obj_post=101.0, obj_ipopt=100.0, gap_rel=0.01,
             feas_post=1.0, viol_max_post=1e-5,
             repair_peak_mem_bytes_mean=1.0 * (1024 ** 3),
             repair_peak_mem_bytes_std=math.nan),
    ]
    parquet = tmp_path / "gap.parquet"
    _make_gap_df(rows).write_parquet(parquet)

    out_dir = tmp_path / "out"
    proc = _run(
        str(_SCRIPT_RENDER),
        "--bench", "b00/v1",
        "--gap-parquet", str(parquet),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr
    md = (out_dir / "b00_v1.md").read_text()
    # Single seed -> std is NaN -> cell shows just the mean, no `+/- std` fragment.
    assert "+0.0100" in md
    assert "+0.0100 +/-" not in md


def test_render_rejects_multi_bench(tmp_path):
    parquet = tmp_path / "gap.parquet"
    _make_gap_df([]).write_parquet(parquet)

    proc = _run(
        str(_SCRIPT_RENDER),
        "--bench", "e3,e4",
        "--gap-parquet", str(parquet),
        "--out-dir", str(tmp_path / "out"),
    )
    assert proc.returncode == 2
    assert "single id" in proc.stderr

    proc = _run(
        str(_SCRIPT_RENDER),
        "--bench", "all",
        "--gap-parquet", str(parquet),
        "--out-dir", str(tmp_path / "out"),
    )
    assert proc.returncode == 2


def test_render_empty_parquet(tmp_path):
    parquet = tmp_path / "gap.parquet"
    _make_gap_df([]).write_parquet(parquet)

    out_dir = tmp_path / "out"
    proc = _run(
        str(_SCRIPT_RENDER),
        "--bench", "b00/v1",
        "--gap-parquet", str(parquet),
        "--out-dir", str(out_dir),
    )
    assert proc.returncode == 0, proc.stderr
    md = (out_dir / "b00_v1.md").read_text()
    assert "empty" in md.lower()


def test_aggregate_rejects_multi_bench(tmp_path):
    proc = _run(
        str(_SCRIPT_AGG),
        "--bench", "e3,e4",
        "--runs-root", str(tmp_path),
    )
    assert proc.returncode == 2
    assert "single id" in proc.stderr


def test_aggregate_rejects_family_prefix(tmp_path):
    """`e3` (no slash) is a family prefix, not a bench id."""
    proc = _run(
        str(_SCRIPT_AGG),
        "--bench", "e3",
        "--runs-root", str(tmp_path),
    )
    assert proc.returncode == 2
    assert "family prefix" in proc.stderr


@pytest.mark.parametrize("bench", ["rosenbrock_eq", "e4/chip_layout"])
def test_aggregate_accepts_well_formed_bench(tmp_path, bench):
    """Synth (no slash, b0X) and engineering (slash) ids are both accepted."""
    runs = tmp_path / "runs"
    runs.mkdir()
    out = tmp_path / "gap.parquet"
    proc = _run(
        str(_SCRIPT_AGG),
        "--bench", bench,
        "--runs-root", str(runs),
        "--out", str(out),
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()
