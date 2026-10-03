"""Tests for `pal run --shard I/N` SLURM-array dispatch."""

from __future__ import annotations

import pytest

from pal.runner.shard import select_shard


def _grid(methods, benches, seeds):
    return [(m, b, s) for m in methods for b in benches for s in seeds]


def test_select_shard_picks_first_tuple():
    grid = _grid(["pal_loggap", "alm"], ["e4/chip_layout"], [0, 1, 2])
    out = select_shard(grid, "0/6")
    assert out == [("pal_loggap", "e4/chip_layout", 0)]


def test_select_shard_picks_last_tuple():
    grid = _grid(["pal_loggap", "alm"], ["e4/chip_layout"], [0, 1, 2])
    out = select_shard(grid, "5/6")
    assert out == [("alm", "e4/chip_layout", 2)]


def test_select_shard_method_major_indexing():
    """Grid order is method-major, then bench, then seed."""
    methods = ["m1", "m2", "m3"]
    benches = ["b1", "b2"]
    seeds = [0, 1]
    grid = _grid(methods, benches, seeds)
    assert grid[0] == ("m1", "b1", 0)
    assert grid[1] == ("m1", "b1", 1)
    assert grid[2] == ("m1", "b2", 0)
    assert grid[3] == ("m1", "b2", 1)
    assert grid[4] == ("m2", "b1", 0)
    assert grid[11] == ("m3", "b2", 1)
    out = select_shard(grid, "7/12")
    assert out == [("m2", "b2", 1)]


def test_select_shard_total_mismatch_fails():
    grid = _grid(["pal_loggap"], ["e4/chip_layout"], [0, 1, 2])
    with pytest.raises(SystemExit, match="grid has 3 tuples"):
        select_shard(grid, "0/24")


def test_select_shard_index_out_of_range():
    grid = _grid(["pal_loggap"], ["e4/chip_layout"], [0, 1, 2])
    with pytest.raises(SystemExit, match="out of range"):
        select_shard(grid, "5/3")


def test_select_shard_negative_index():
    grid = _grid(["pal_loggap"], ["e4/chip_layout"], [0, 1, 2])
    with pytest.raises(SystemExit, match="out of range"):
        select_shard(grid, "-1/3")


def test_select_shard_malformed_no_slash():
    grid = _grid(["pal_loggap"], ["e4/chip_layout"], [0])
    with pytest.raises(SystemExit, match="expects 'I/N'"):
        select_shard(grid, "0")


def test_select_shard_non_integer():
    grid = _grid(["pal_loggap"], ["e4/chip_layout"], [0])
    with pytest.raises(SystemExit, match="must be integers"):
        select_shard(grid, "a/b")


def test_select_shard_engineering_grid_sizes():
    """Grid sizes baked into the engineering sbatches."""
    cases = [
        # (n_methods, n_benches, n_seeds, expected_total)
        (5, 1, 3, 15),   # e1_gpu, e2, e3, e4
        (1, 1, 3, 3),    # bNNN_ipopt grid size (sliced by --shard), restart-shards live on a separate axis
        (7, 2, 1, 14),   # memory_bs1
    ]
    for nm, nb, ns, total in cases:
        methods = [f"m{i}" for i in range(nm)]
        benches = [f"b{i}" for i in range(nb)]
        seeds = list(range(ns))
        grid = _grid(methods, benches, seeds)
        assert len(grid) == total
        out = select_shard(grid, f"0/{total}")
        assert len(out) == 1
        out_last = select_shard(grid, f"{total - 1}/{total}")
        assert len(out_last) == 1
