"""Checkpoint resolution for the vendored WinDiNet slice.

Delegates to `pal.artifacts.ensure_artifact`, which reads each entry from
the central registry (`pal.artifacts.registry.REGISTRY`). The three
WinDiNet weights (`dit`, `scalar_embedding`, `vae_decoder`) exceed the
in-repo size limit and live on Hugging Face at `pal.artifacts.HF_REPO_ID`.

Raw filesystem paths are also accepted for local override, tests / debugging.

Mirrors the surface of `pal/benchmarks/engineering/e1_bwb/_artifacts.py`
so both benchmarks expose the same three helpers:
`_BENCH_ID`, `ensure_checkpoint` (this file's public name, kept for
WinDiNet-fidelity), and `check_runtime_artifacts`.
"""

from __future__ import annotations

from pathlib import Path

_BENCH_ID = "e2/urban_wind"
_LOGICAL_NAMES = ("dit", "scalar_embedding", "vae_decoder")


def ensure_checkpoint(name_or_path: str | Path) -> str:
    """Resolve a logical checkpoint name, or pass through an existing path.

    Args:
        name_or_path: one of `dit`, `scalar_embedding`, `vae_decoder`, or a
            path-like pointing to an on-disk file.

    Returns:
        Absolute string path to the resolved checkpoint. Raises
        FileNotFoundError (unregistered or not downloaded) or
        KeyError (name doesn't exist in the central registry).
    """
    p = Path(name_or_path)
    if p.exists():
        return str(p.resolve())

    from pal.artifacts import ensure_artifact, ref_by_name

    if str(name_or_path) in _LOGICAL_NAMES:
        ref = ref_by_name(_BENCH_ID, str(name_or_path))
        return str(ensure_artifact(_BENCH_ID, ref))

    raise FileNotFoundError(
        f"e2/urban_wind: unknown checkpoint or missing path: {name_or_path!r}\n"
        f"  valid names: {', '.join(_LOGICAL_NAMES)}"
    )


def check_runtime_artifacts() -> None:
    """Verify every WinDiNet checkpoint the runtime path needs is on disk.

    Tries to resolve each of the three registered checkpoints and collects
    missing / tampered entries into a single error message so the user
    doesn't fix them one at a time.
    """
    from pal.artifacts import ensure_artifact, ref_by_name

    missing: list[tuple[str, str]] = []
    for name in _LOGICAL_NAMES:
        try:
            ref = ref_by_name(_BENCH_ID, name)
            ensure_artifact(_BENCH_ID, ref)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            missing.append((name, str(exc).replace("\n", " ").strip()))

    if missing:
        lines = [f"  - {name}: {msg}" for name, msg in missing]
        raise FileNotFoundError(
            "e2/urban_wind: missing runtime checkpoints (run "
            "`python scripts/download_artifacts.py --benchmark e2` to fetch "
            "them from HF):\n" + "\n".join(lines)
        )
