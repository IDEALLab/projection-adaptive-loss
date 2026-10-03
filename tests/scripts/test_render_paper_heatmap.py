"""Tests for the paper cost-table heatmap renderer."""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl
import pytest

from scripts.render_paper_heatmap import (
    BENCH_ORDER,
    METHOD_ORDER,
    _per_run_nfe,
    render,
)


def _make_run_row(
    method: str,
    bench: str,
    seed: int,
    *,
    feasibility_post: float | None = 0.95,
    obj_mean_post: float | None = 0.10,
    train_fwd_calls: int | None = 400,
    train_bwd_calls: int | None = 400,
    train_fwd_samples: int | None = 3200,
    train_bwd_samples: int | None = 3200,
    train_opt_steps: int | None = 200,
    wall_start: str = "2026-04-24T00:00:00",
    status: str = "ok",
) -> dict:
    return {
        "run_id": f"{method}_{bench}_seed{seed}",
        "method": method,
        "benchmark_id": bench,
        "seed": seed,
        "status": status,
        "feasibility_post": feasibility_post,
        "obj_mean_post": obj_mean_post,
        "train_fwd_calls": train_fwd_calls,
        "train_bwd_calls": train_bwd_calls,
        "train_fwd_samples": train_fwd_samples,
        "train_bwd_samples": train_bwd_samples,
        "train_opt_steps": train_opt_steps,
        "wall_start": wall_start,
    }


def _make_parquet(tmp_path: Path, rows: list[dict]) -> Path:
    df = pl.from_dicts(rows, infer_schema_length=None)
    out = tmp_path / "runs.parquet"
    df.write_parquet(out)
    return out


def test_per_run_nfe_formula():
    row = {
        "train_fwd_calls": 400,
        "train_bwd_calls": 400,
        "train_fwd_samples": 3200,
        "train_bwd_samples": 3200,
        "train_opt_steps": 200,
    }
    fwd, bwd = _per_run_nfe(row)
    # nfe_fwd = train_fwd_calls / train_opt_steps = 400/200 = 2.0
    assert fwd == pytest.approx(2.0)
    assert bwd == pytest.approx(2.0)


def test_per_run_nfe_returns_none_on_zero_opt_steps():
    row = {
        "train_fwd_calls": 400,
        "train_bwd_calls": 400,
        "train_fwd_samples": 3200,
        "train_bwd_samples": 3200,
        "train_opt_steps": 0,
    }
    fwd, bwd = _per_run_nfe(row)
    assert fwd is None
    assert bwd is None


def test_per_run_nfe_returns_none_on_missing():
    row = {
        "train_fwd_calls": None,
        "train_bwd_calls": None,
        "train_fwd_samples": None,
        "train_bwd_samples": None,
        "train_opt_steps": None,
    }
    assert _per_run_nfe(row) == (None, None)


def _full_parquet(tmp_path: Path) -> Path:
    """Synthesize 7 methods x 6 benches x 3 seeds: PAL best, ENFORCE worst on NFE, std=0."""
    rows: list[dict] = []
    method_keys = [m for (m, _) in METHOD_ORDER]
    bench_keys = [b for (b, _) in BENCH_ORDER]

    method_cost = {
        "alm": 3.0,
        "alm_bolton": 1.0,
        "enforce_orig": 11.0,
        "dc3": 8.0,
        "fsnet": 5.0,
        "snarenet": 6.0,
        "pal_loggap": 2.0,
    }
    method_feas = {
        "alm": 0.80,
        "alm_bolton": 0.85,
        "enforce_orig": 0.90,
        "dc3": 0.88,
        "fsnet": 0.92,
        "snarenet": 0.94,
        "pal_loggap": 0.98,
    }
    method_obj = {
        "alm": 0.45,
        "alm_bolton": 0.40,
        "enforce_orig": 0.20,
        "dc3": 0.25,
        "fsnet": 0.18,
        "snarenet": 0.12,
        "pal_loggap": 0.05,
    }

    for method in method_keys:
        cost = method_cost[method]
        feas = method_feas[method]
        obj = method_obj[method]
        for bench in bench_keys:
            for seed in (0, 1, 2):
                # General: nfe = cost / 8 (avg_B=8).
                fwd_calls = int(cost * 200)
                rows.append(_make_run_row(
                    method, bench, seed,
                    feasibility_post=feas,
                    obj_mean_post=obj,
                    train_fwd_calls=fwd_calls,
                    train_bwd_calls=fwd_calls,
                    train_fwd_samples=fwd_calls * 8,
                    train_bwd_samples=fwd_calls * 8,
                    train_opt_steps=200,
                ))
    return _make_parquet(tmp_path, rows)


def test_render_full_table_emits_all_cells(tmp_path):
    parquet = _full_parquet(tmp_path)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()

    # 7 methods x 6 benches x 4 metrics, minus 4 excluded dc3xs5 cells,
    # plus 4 metrics x 7 methods Mean-row cells = 192 \cc calls.
    cc_calls = re.findall(r"\\cc\{(-?\d+)\}", text)
    assert len(cc_calls) == 7 * 6 * 4 - 4 + 4 * 7, (
        f"expected 192 \\cc calls, got {len(cc_calls)}"
    )

    for n_str in cc_calls:
        n = int(n_str)
        assert 0 <= n <= 100, f"goodness {n} out of range"

    # Bold-best lives only on the Mean row: at least one \best per metric block.
    best_rows = re.findall(r"\\best\{", text)
    assert len(best_rows) >= 4, f"expected >=4 \\best markers, got {len(best_rows)}"


def test_render_best_marker_count_matches_rows(tmp_path):
    """No ties: exactly one `\\best` per metric Mean row, 4 in total."""
    parquet = _full_parquet(tmp_path)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()
    n_best = len(re.findall(r"\\best\{", text))
    assert n_best == 4, f"expected 4 \\best markers, got {n_best}"


def test_missing_seeds_render_dash(tmp_path):
    """Drop one cell entirely, that cell renders an em dash, other cells unaffected."""
    parquet = _full_parquet(tmp_path)
    df = pl.read_parquet(parquet)
    df = df.filter(
        ~((pl.col("method") == "alm_bolton") & (pl.col("benchmark_id") == "s1_sphere_track"))
    )
    pruned = tmp_path / "pruned.parquet"
    df.write_parquet(pruned)

    out = tmp_path / "table.tex"
    render(pruned, out, standalone_path=None, preamble_path=None)
    text = out.read_text()
    assert text.count("---") >= 4


def test_zero_opt_steps_renders_dash(tmp_path):
    """train_opt_steps=0 -> NFE cell renders an em dash (no division by zero)."""
    rows = []
    method_keys = [m for (m, _) in METHOD_ORDER]
    bench_keys = [b for (b, _) in BENCH_ORDER]
    for method in method_keys:
        for bench in bench_keys:
            for seed in (0, 1, 2):
                opt_steps = 0 if (method == "alm_bolton" and bench == "s1_sphere_track") else 200
                rows.append(_make_run_row(
                    method, bench, seed,
                    train_opt_steps=opt_steps,
                ))
    parquet = _make_parquet(tmp_path, rows)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()
    # Only the broken pair's NFE cells (fwd + bwd) render an em dash.
    assert text.count("---") >= 2


def test_tie_break_bolds_both_cells(tmp_path):
    """Two methods produce identically-formatted NFE -> both get \\best."""
    rows = []
    method_keys = [m for (m, _) in METHOD_ORDER]
    bench_keys = [b for (b, _) in BENCH_ORDER]
    for method in method_keys:
        for bench in bench_keys:
            for seed in (0, 1, 2):
                # alm_bolton and pal_loggap will produce identical .1f-rounded NFE.
                if method in ("alm_bolton", "pal_loggap"):
                    fwd_calls = 200  # nfe = 1/8 = 0.125 -> "0.1"
                else:
                    fwd_calls = int({
                        "alm": 3.0, "enforce_orig": 11.0, "dc3": 8.0,
                        "fsnet": 5.0, "snarenet": 6.0,
                    }[method] * 200)
                rows.append(_make_run_row(
                    method, bench, seed,
                    train_fwd_calls=fwd_calls,
                    train_bwd_calls=fwd_calls,
                    train_fwd_samples=fwd_calls * 8,
                    train_bwd_samples=fwd_calls * 8,
                    train_opt_steps=200,
                ))
    parquet = _make_parquet(tmp_path, rows)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()

    # NFE_fwd is the third metric block: look for a line with at least 2 \best{}.
    rows_with_two_best = [
        line for line in text.splitlines() if line.count(r"\best{") >= 2
    ]
    assert rows_with_two_best, (
        "expected at least one row with 2+ \\best markers (tie-break case);\n"
        + text
    )


def test_divergence_annotations_and_worst_case_imputation(tmp_path):
    """Divergence-as-worst: k/N superscripts on partial cells, a dash for dc3xs5."""
    rows = []
    for s in range(3):
        rows.append(_make_run_row("pal_loggap", "s1_sphere_track", s))
    for s in range(2):
        rows.append(_make_run_row("pal_loggap", "s2_active_set_switch", s))
    rows.append(_make_run_row(
        "pal_loggap", "s2_active_set_switch", 2,
        status="failed", feasibility_post=None, obj_mean_post=None,
        train_fwd_calls=None, train_bwd_calls=None,
        train_fwd_samples=None, train_bwd_samples=None, train_opt_steps=None,
    ))
    rows.append(_make_run_row("pal_loggap", "s3_illcond_tube", 0))
    for s in (1, 2):
        rows.append(_make_run_row(
            "pal_loggap", "s3_illcond_tube", s,
            status="failed", feasibility_post=None, obj_mean_post=None,
            train_fwd_calls=None, train_bwd_calls=None,
            train_fwd_samples=None, train_bwd_samples=None, train_opt_steps=None,
        ))
    for m, _ in METHOD_ORDER:
        if m == "pal_loggap":
            continue
        for b, _ in BENCH_ORDER[:3]:
            for s in range(3):
                rows.append(_make_run_row(m, b, s))

    parquet = _make_parquet(tmp_path, rows)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()

    # Partially-diverged cells keep a (worst-case imputed) value, no bare em dash.
    assert r"$^{1/3}$" in text, "expected 1/3 marker on the 2/3-ok cell"
    assert r"$^{2/3}$" in text, "expected 2/3 marker on the 1/3-ok cell"
    assert "---" + r"$^{\S}$" in text, "expected structural dash on dc3xs5"
    assert "seeds diverged" in text
    assert r"$^{\ddag}$" in text or r"\ddag" in text


def test_no_footnote_when_all_cells_clean(tmp_path):
    """All-success matrix -> no dagger footnote emitted."""
    rows = []
    for m, _ in METHOD_ORDER:
        for b, _ in BENCH_ORDER:
            for s in range(3):
                rows.append(_make_run_row(m, b, s))
    parquet = _make_parquet(tmp_path, rows)
    out = tmp_path / "table.tex"
    render(parquet, out, standalone_path=None, preamble_path=None)
    text = out.read_text()
    assert r"$^{\dag}$" not in text
    assert r"\footnotesize $\dag$" not in text


def test_dedupe_keeps_latest_wall_start(tmp_path):
    """Two runs with same (method, bench, seed), latest wins."""
    rows = [
        _make_run_row(
            "pal_loggap", "s1_sphere_track", 0,
            obj_mean_post=99.0, wall_start="2026-04-20T00:00:00",
        ),
        _make_run_row(
            "pal_loggap", "s1_sphere_track", 0,
            obj_mean_post=0.5, wall_start="2026-04-24T00:00:00",
        ),
    ]
    parquet = _make_parquet(tmp_path, rows)
    df = pl.read_parquet(parquet).filter(pl.col("status") == "ok")
    from scripts.render_paper_heatmap import _dedupe
    df = _dedupe(df)
    assert df.height == 1
    assert df["obj_mean_post"][0] == pytest.approx(0.5)
