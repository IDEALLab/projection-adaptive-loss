"""Artifact resolution for the E1 BWB benchmark.

Logical names resolve through `pal.artifacts`, raw filesystem paths pass through.
"""

from __future__ import annotations

from pathlib import Path

_BENCH_ID = "e1/bwb"
_LOGICAL_NAMES = (
    "a_aero_weights",
    "a_aero_norm_stats",
    "film_weights",
    "film_norm_stats",
    "struct_weights",
    "bwb_sdf_weights",
    "struct_training_data",
)


def ensure_artifact_path(name_or_path: str | Path) -> str:
    """Resolve a logical artifact name, or pass through an existing path.

    Args:
        name_or_path: one of the logical names above, or a path-like pointing
            to an on-disk file (useful for unit tests + local overrides).

    Returns:
        Absolute string path to the resolved artifact. Raises
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
        f"e1/bwb: unknown artifact or missing path: {name_or_path!r}\n"
        f"  valid names: {', '.join(_LOGICAL_NAMES)}"
    )


def check_runtime_artifacts() -> None:
    """Verify every runtime artifact is on disk, reporting all missing ones at once."""
    from pal.artifacts import ensure_artifact, ref_by_name

    runtime_names = [n for n in _LOGICAL_NAMES if n != "struct_training_data"]
    missing: list[tuple[str, str]] = []
    for name in runtime_names:
        try:
            ref = ref_by_name(_BENCH_ID, name)
            ensure_artifact(_BENCH_ID, ref)
        except (FileNotFoundError, KeyError, RuntimeError) as exc:
            missing.append((name, str(exc).replace("\n", " ").strip()))

    if missing:
        lines = [f"  - {name}: {msg}" for name, msg in missing]
        raise FileNotFoundError(
            "e1/bwb: missing runtime artifacts (these ship in-repo under "
            "pal/benchmarks/engineering/e1_bwb/_artifacts_data/, check your "
            "working tree):\n" + "\n".join(lines)
        )
