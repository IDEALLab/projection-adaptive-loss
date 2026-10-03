"""End-to-end test for aggregate_engineering's IPOPT restart-shard merge."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
from aggregate_engineering import build_gap  # noqa: E402


def _write_run(
    runs_root: Path,
    *,
    ts: str,
    method: str,
    bench: str,
    seed: int,
    shard: tuple[int, int] | None,
    final: dict,
    config_extras: dict | None = None,
) -> Path:
    bench_slug = bench.replace("/", "-")
    shard_suffix = f"_r{shard[0]}_of{shard[1]}" if shard else ""
    name = f"{ts}_{method}_{bench_slug}_seed{seed}{shard_suffix}_abcd1234"
    d = runs_root / name
    d.mkdir(parents=True)
    config = {
        "method": method,
        "benchmark_id": bench,
        "seed": seed,
        "wall_start": ts,
    }
    if config_extras:
        config.update(config_extras)
    (d / "config.json").write_text(json.dumps(config))
    (d / "final.json").write_text(json.dumps(final))
    (d / "status.json").write_text(json.dumps({"status": "ok"}))
    return d


def _ipopt_final_for_shard(restart_records: list[dict]) -> dict:
    """Minimal final.json with one query carrying the given restarts."""
    feasible = [r for r in restart_records if r["feasible"]]
    if feasible:
        winner = min(feasible, key=lambda r: r["obj"])
    else:
        winner = min(restart_records, key=lambda r: r["max_violation"])
    return {
        "obj_mean_post": winner["obj"],
        "viol_max_post": winner["max_violation"],
        "feasibility_post": float(winner["feasible"]),
        "n_queries": 1,
        "n_restarts": len(restart_records),
        "per_query": [
            {
                "obj": winner["obj"],
                "max_violation": winner["max_violation"],
                "feasible": winner["feasible"],
                "status": winner["status"],
                "n_feasible_restarts": sum(1 for r in restart_records if r["feasible"]),
                "restarts": restart_records,
            }
        ],
    }


def test_build_gap_merges_ipopt_restart_shards(tmp_path: Path) -> None:
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bench = "e3/acopf_ieee57"

    # Shard 1/2 holds the global best (0.3); a latest-wall_start dedupe would lose it.
    shard_0 = [
        {"restart_idx": 0, "x": [0.5, 0.5], "obj": 1.0, "max_violation": 0.0,
         "feasible": True, "status": 0},
        {"restart_idx": 2, "x": [0.7, 0.7], "obj": 5.0, "max_violation": 0.0,
         "feasible": True, "status": 0},
    ]
    shard_1 = [
        {"restart_idx": 1, "x": [0.3, 0.3], "obj": 0.3, "max_violation": 0.0,
         "feasible": True, "status": 0},
        {"restart_idx": 3, "x": [0.6, 0.6], "obj": 2.0, "max_violation": 0.0,
         "feasible": True, "status": 0},
    ]

    _write_run(
        runs_root,
        ts="20260430T100000Z",
        method="ipopt",
        bench=bench,
        seed=0,
        shard=(0, 2),
        final=_ipopt_final_for_shard(shard_0),
    )
    _write_run(
        runs_root,
        ts="20260430T101000Z",
        method="ipopt",
        bench=bench,
        seed=0,
        shard=(1, 2),
        final=_ipopt_final_for_shard(shard_1),
    )
    _write_run(
        runs_root,
        ts="20260430T102000Z",
        method="pal_loggap",
        bench=bench,
        seed=0,
        shard=None,
        final={
            "obj_mean_post": 0.8,
            "viol_max_post": 0.0,
            "feasibility_post": 1.0,
            "n_queries": 1,
            "n_restarts": 1,
        },
    )

    df = build_gap(runs_root, bench)
    ipopt_rows = df.filter(df["method"] == "ipopt")
    assert ipopt_rows.height == 1
    # Merged row carries the GLOBAL best obj (0.3), not either shard's local best.
    assert ipopt_rows["obj_post"][0] == pytest.approx(0.3)
    assert ipopt_rows["obj_ipopt"][0] == pytest.approx(0.3)
    pal_row = df.filter(df["method"] == "pal_loggap")
    assert pal_row.height == 1
    expected_gap = (0.8 - 0.3) / 0.3
    assert pal_row["gap_rel"][0] == pytest.approx(expected_gap)


def test_build_gap_rejects_partial_shard_set_by_default(tmp_path: Path) -> None:
    """Refuse IPOPT multi-start rows built from k<R shards unless explicitly allowed."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bench = "e3/acopf_ieee57"

    # Only 2 of 5 shards present.
    for r in range(2):
        _write_run(
            runs_root,
            ts=f"20260430T10000{r}Z",
            method="ipopt",
            bench=bench,
            seed=0,
            shard=(r, 5),
            final=_ipopt_final_for_shard(
                [{"restart_idx": r, "x": [0.0, 0.0], "obj": float(r),
                  "max_violation": 0.0, "feasible": True, "status": 0}]
            ),
        )

    with pytest.raises(ValueError, match="incomplete"):
        build_gap(runs_root, bench)


def test_build_gap_allow_partial_shards_flags_row(tmp_path: Path) -> None:
    """With --allow-partial-shards the row merges but carries partial_shards=True."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bench = "e3/acopf_ieee57"

    for r in range(2):
        _write_run(
            runs_root,
            ts=f"20260430T10000{r}Z",
            method="ipopt",
            bench=bench,
            seed=0,
            shard=(r, 5),
            final=_ipopt_final_for_shard(
                [{"restart_idx": r, "x": [0.0, 0.0], "obj": float(r),
                  "max_violation": 0.0, "feasible": True, "status": 0}]
            ),
        )

    df = build_gap(runs_root, bench, allow_partial_shards=True)
    assert df.height == 1
    assert df["obj_post"][0] == pytest.approx(0.0)


def test_build_gap_rejects_inconsistent_shard_R(tmp_path: Path) -> None:
    """Mixed R values across shards in the same runs_root indicate stale runs."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bench = "e3/acopf_ieee57"

    _write_run(
        runs_root,
        ts="20260430T100000Z",
        method="ipopt",
        bench=bench,
        seed=0,
        shard=(0, 2),
        final=_ipopt_final_for_shard(
            [{"restart_idx": 0, "x": [0.0, 0.0], "obj": 1.0, "max_violation": 0.0,
              "feasible": True, "status": 0}]
        ),
    )
    _write_run(
        runs_root,
        ts="20260430T100100Z",
        method="ipopt",
        bench=bench,
        seed=0,
        shard=(0, 5),  # different R, stale run
        final=_ipopt_final_for_shard(
            [{"restart_idx": 0, "x": [0.0, 0.0], "obj": 0.5, "max_violation": 0.0,
              "feasible": True, "status": 0}]
        ),
    )

    with pytest.raises(ValueError, match="inconsistent"):
        build_gap(runs_root, bench)


def test_build_gap_passes_through_unsharded_ipopt_unchanged(tmp_path: Path) -> None:
    """No restart_shard in dir name -> row passes through the merge step untouched."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    bench = "e3/acopf_ieee57"

    restarts = [
        {"restart_idx": 0, "x": [0.5, 0.5], "obj": 1.5, "max_violation": 0.0,
         "feasible": True, "status": 0},
    ]
    _write_run(
        runs_root,
        ts="20260430T100000Z",
        method="ipopt",
        bench=bench,
        seed=0,
        shard=None,
        final=_ipopt_final_for_shard(restarts),
    )

    df = build_gap(runs_root, bench)
    assert df.height == 1
    assert df["obj_post"][0] == pytest.approx(1.5)
