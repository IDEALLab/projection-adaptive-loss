"""SLURM array dispatch helpers for `pal run --shard I/N`.

The grid is `(method, bench, seed)` in method-major order and each SLURM array
task selects exactly one tuple.
"""

from __future__ import annotations


def select_shard(
    grid: list[tuple[str, str, int]], shard_spec: str,
) -> list[tuple[str, str, int]]:
    """Validate `--shard I/N` and return [grid[I]].

    `N` must equal `len(grid)` so a stale SLURM array size fails loudly.
    """
    if "/" not in shard_spec:
        raise SystemExit(
            f"--shard expects 'I/N' (e.g. '3/24'), got: {shard_spec!r}"
        )
    idx_s, total_s = shard_spec.split("/", 1)
    try:
        idx = int(idx_s)
        total = int(total_s)
    except ValueError as e:
        raise SystemExit(
            f"--shard {shard_spec!r}: I and N must be integers"
        ) from e
    actual = len(grid)
    if total != actual:
        raise SystemExit(
            f"--shard {shard_spec}: N={total} but the (method x bench x seed) "
            f"grid has {actual} tuples. Update `#SBATCH --array=0-{actual - 1}` "
            f"to match the YAML grid (methods x benches x seeds)."
        )
    if idx < 0 or idx >= total:
        raise SystemExit(
            f"--shard {shard_spec}: index {idx} out of range [0, {total})"
        )
    return [grid[idx]]
