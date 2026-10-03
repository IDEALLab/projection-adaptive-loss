"""The e1 logic modules import cleanly without absolute paths or sys.path hacks."""

from __future__ import annotations

import pytest

from pal.benchmarks.engineering.e1_bwb import (
    a_aero,
    beam,
    loads,
    struct,
)
from pal.benchmarks.engineering.e1_bwb._artifacts import (
    _LOGICAL_NAMES,
    ensure_artifact_path,
)


def test_modules_import():
    assert hasattr(a_aero, "AAeroSurrogate")
    assert hasattr(loads, "FiLMLoads")
    assert hasattr(loads, "build_bwb_program")
    assert hasattr(struct, "StructSurrogate")
    assert hasattr(beam, "compute_stress")


def test_logical_names_complete():
    """Every loader's logical name shows up in _LOGICAL_NAMES."""
    expected = {
        "a_aero_weights",
        "a_aero_norm_stats",
        "film_weights",
        "film_norm_stats",
        "struct_weights",
        "bwb_sdf_weights",
        "struct_training_data",
    }
    assert set(_LOGICAL_NAMES) == expected


def test_ensure_artifact_path_unknown_name_raises_filenotfound():
    """Bad name with no on-disk fallback fails with a useful message."""
    with pytest.raises(FileNotFoundError, match="unknown artifact"):
        ensure_artifact_path("does_not_exist_anywhere")


def test_registry_entries_registered():
    """REGISTRY has a e1/bwb entry with every runtime name, all in-repo."""
    from pal.artifacts.registry import REGISTRY, ref_by_name

    assert "e1/bwb" in REGISTRY
    runtime_names = set(_LOGICAL_NAMES) - {"struct_training_data"}
    for name in runtime_names:
        ref = ref_by_name("e1/bwb", name)
        assert ref.name == name
        assert ref.location == "in_repo"
        assert ref.repo_path.startswith(
            "pal/benchmarks/engineering/e1_bwb/_artifacts_data/"
        )


def test_bwb_yaml_is_package_resource():
    """The BWB YAML ships with the package (no absolute path)."""
    assert loads._BWB_YAML.exists(), f"YAML missing: {loads._BWB_YAML}"


def test_no_absolute_user_paths():
    """Sanity: no hardcoded `/Users/...` in the ported runtime modules."""
    import inspect
    for mod in (a_aero, loads, struct, beam):
        src = inspect.getsource(mod)
        assert "/Users/" not in src, f"{mod.__name__} still has a /Users/ path"
