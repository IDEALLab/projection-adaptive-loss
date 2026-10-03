"""Location-branching tests for `pal.artifacts.ensure_artifact` (in_repo vs hf)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pal.artifacts import ensure_artifact, ref_by_name
from pal.artifacts.core import (
    IN_REPO_SIZE_LIMIT_BYTES,
    ArtifactRef,
    download_artifact,
    get_repo_root,
)
from pal.artifacts.registry import REGISTRY


def test_every_in_repo_artifact_exists_and_fits_the_limit():
    """Walk REGISTRY; every in_repo ref must resolve and be under the cap."""
    root = get_repo_root()
    for bench_id, refs in REGISTRY.items():
        for ref in refs:
            if ref.location != "in_repo":
                continue
            assert ref.repo_path, (
                f"{bench_id}:{ref.name} is in_repo but has no repo_path"
            )
            path = root / ref.repo_path
            assert path.exists(), f"{bench_id}:{ref.name} missing at {path}"
            size = path.stat().st_size
            assert size <= IN_REPO_SIZE_LIMIT_BYTES, (
                f"{bench_id}:{ref.name} is {size} bytes, exceeds "
                f"IN_REPO_SIZE_LIMIT_BYTES={IN_REPO_SIZE_LIMIT_BYTES}. "
                f"Move to HF."
            )


def test_every_hf_artifact_has_revision_and_filename():
    for bench_id, refs in REGISTRY.items():
        for ref in refs:
            if ref.location != "hf":
                continue
            assert ref.hf_filename, f"{bench_id}:{ref.name} missing hf_filename"
            assert ref.hf_revision, f"{bench_id}:{ref.name} missing hf_revision"
            assert ref.hf_revision != "local-dev", (
                f"{bench_id}:{ref.name} still pinned to local-dev sentinel"
            )


def test_in_repo_ensure_artifact_returns_absolute_path(tmp_path, monkeypatch):
    """PAL_DATA_DIR is irrelevant for in_repo, path comes from repo root."""
    monkeypatch.setenv("PAL_DATA_DIR", str(tmp_path))
    ref = ref_by_name("e1/bwb", "a_aero_weights")
    path = ensure_artifact("e1/bwb", ref)
    assert path.is_absolute()
    assert path.exists()
    assert path == get_repo_root() / ref.repo_path


def test_in_repo_missing_file_raises_filenotfound(tmp_path):
    bogus = ArtifactRef(
        name="ghost",
        location="in_repo",
        repo_path="does/not/exist.bin",
    )
    with pytest.raises(FileNotFoundError, match="Missing in-repo artifact"):
        ensure_artifact("e1/bwb", bogus)


def test_in_repo_empty_repo_path_raises_valueerror():
    bogus = ArtifactRef(name="ghost", location="in_repo")
    with pytest.raises(ValueError, match="missing repo_path"):
        ensure_artifact("e1/bwb", bogus)


def test_download_refuses_in_repo():
    ref = ref_by_name("e1/bwb", "a_aero_weights")
    with pytest.raises(RuntimeError, match="refusing to download in-repo"):
        download_artifact("e1/bwb", ref)


def _write_sidecar(target: Path, revision: str) -> None:
    st = target.stat()
    meta_dir = target.parent / ".pal_meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    (meta_dir / f"{target.name}.json").write_text(
        json.dumps(
            {
                "revision": revision,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    )


def test_hf_branch_enforces_sidecar(tmp_path, monkeypatch):
    """Stage a fake HF artifact, confirm the sidecar + revision gate works."""
    monkeypatch.setenv("PAL_DATA_DIR", str(tmp_path))
    ref = ArtifactRef(
        name="fake_hf",
        hf_filename="b999/fake.bin",
        hf_revision="deadbeef",
        local_subdir="slice",
        location="hf",
    )
    target = tmp_path / "b999/urban" / "slice" / "fake.bin"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"payload")

    # No sidecar -> RuntimeError (revision comparison fails).
    with pytest.raises(RuntimeError, match="Revision mismatch"):
        ensure_artifact("b999/urban", ref)

    _write_sidecar(target, revision="deadbeef")
    path = ensure_artifact("b999/urban", ref)
    assert path == target

    _write_sidecar(target, revision="cafef00d")
    with pytest.raises(RuntimeError, match="Revision mismatch"):
        ensure_artifact("b999/urban", ref)


def test_unknown_location_raises():
    ref = ArtifactRef(name="x", location="s3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unknown artifact location"):
        ensure_artifact("b000/test", ref)


def test_e1_ensure_artifact_path_resolves_all_names():
    from pal.benchmarks.engineering.e1_bwb._artifacts import (
        _LOGICAL_NAMES,
        ensure_artifact_path,
    )

    runtime_names = [n for n in _LOGICAL_NAMES if n != "struct_training_data"]
    for name in runtime_names:
        path = Path(ensure_artifact_path(name))
        assert path.exists(), f"{name} -> {path} missing"


def test_e1_check_runtime_artifacts_passes():
    from pal.benchmarks.engineering.e1_bwb._artifacts import (
        check_runtime_artifacts,
    )

    check_runtime_artifacts()  # must not raise


def test_e2_ensure_checkpoint_passes_through_existing_path(tmp_path):
    from pal.benchmarks.engineering.e2_urban_wind._vendor.windinet.checkpoints import (
        ensure_checkpoint,
    )

    fake = tmp_path / "fake.pt"
    fake.write_bytes(b"x")
    assert ensure_checkpoint(str(fake)) == str(fake.resolve())


def test_e2_ensure_checkpoint_unknown_name_raises():
    from pal.benchmarks.engineering.e2_urban_wind._vendor.windinet.checkpoints import (
        ensure_checkpoint,
    )

    with pytest.raises(FileNotFoundError, match="unknown checkpoint"):
        ensure_checkpoint("not_a_real_name")
