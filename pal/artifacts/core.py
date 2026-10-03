from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Hugging Face repo holding the large artifacts (the public WinDiNet weights used by E2).
# Override with PAL_HF_REPO_ID.
HF_REPO_ID = os.environ.get("PAL_HF_REPO_ID", "rabischof/windinet")

# Files at or below this size ship in the git repo; larger ones live on Hugging Face.
IN_REPO_SIZE_LIMIT_BYTES = 10 * 1024 * 1024

_REPO_ROOT = Path(__file__).resolve().parents[2]


Location = Literal["in_repo", "hf"]


@dataclass(frozen=True)
class ArtifactRef:
    """Pointer to a single PAL artifact.

    `"hf"` artifacts are fetched from `HF_REPO_ID` at a pinned revision into
    `$PAL_DATA_DIR/<bench_id>/<local_subdir>`; `"in_repo"` artifacts live at
    `repo_path` relative to the repo root.
    """

    name: str
    hf_filename: str = ""
    hf_revision: str = ""
    local_subdir: str = ""
    location: Location = "hf"
    repo_path: str = ""


def get_data_dir() -> Path:
    return Path(os.environ.get("PAL_DATA_DIR", Path.home() / ".pal_data"))


def get_repo_root() -> Path:
    return _REPO_ROOT


def _meta_path(target: Path) -> Path:
    return target.parent / ".pal_meta" / f"{target.name}.json"


def _read_meta(meta: Path) -> dict:
    return json.loads(meta.read_text()) if meta.exists() else {}


def _ensure_in_repo(bench_id: str, ref: ArtifactRef) -> Path:
    if not ref.repo_path:
        raise ValueError(
            f"in_repo artifact {bench_id}:{ref.name} is missing repo_path"
        )
    target = _REPO_ROOT / ref.repo_path
    if not target.exists():
        raise FileNotFoundError(
            f"Missing in-repo artifact for {bench_id}:{ref.name}\n"
            f"  expected at: {target}\n"
            f"  this file should be tracked in git, check your working tree"
        )
    return target


def _ensure_hf(bench_id: str, ref: ArtifactRef) -> Path:
    target = get_data_dir() / bench_id / ref.local_subdir / Path(ref.hf_filename).name
    meta_path = _meta_path(target)
    if not target.exists():
        raise FileNotFoundError(
            f"Missing artifact for {bench_id}: {ref.hf_filename}\n"
            f"  expected at: {target}\n"
            f"  fix: python scripts/download_artifacts.py --benchmark {bench_id}\n"
            f"  (downloads from the public Hugging Face repo HF_REPO_ID)"
        )
    meta = _read_meta(meta_path)
    st = target.stat()
    if meta.get("revision") != ref.hf_revision:
        raise RuntimeError(
            f"Revision mismatch for {bench_id}:{ref.hf_filename}\n"
            f"  pinned revision: {ref.hf_revision}\n"
            f"  sidecar revision: {meta.get('revision', '(no metadata)')}\n"
            f"  fix: python scripts/download_artifacts.py --benchmark {bench_id} --force"
        )
    if meta.get("size") != st.st_size or meta.get("mtime_ns") != st.st_mtime_ns:
        raise RuntimeError(
            f"Artifact tampered or re-written outside the loader for {bench_id}:{ref.hf_filename}\n"
            f"  sidecar: size={meta.get('size')} mtime_ns={meta.get('mtime_ns')}\n"
            f"  on-disk: size={st.st_size} mtime_ns={st.st_mtime_ns}\n"
            f"  fix: python scripts/download_artifacts.py --benchmark {bench_id} --force"
        )
    return target


def ensure_artifact(bench_id: str, ref: ArtifactRef) -> Path:
    if ref.location == "in_repo":
        return _ensure_in_repo(bench_id, ref)
    if ref.location == "hf":
        return _ensure_hf(bench_id, ref)
    raise ValueError(f"unknown artifact location {ref.location!r} for {bench_id}:{ref.name}")


def download_artifact(bench_id: str, ref: ArtifactRef) -> Path:
    if ref.location == "in_repo":
        raise RuntimeError(
            f"refusing to download in-repo artifact {bench_id}:{ref.name}, "
            f"it should already be present at {_REPO_ROOT / ref.repo_path}"
        )
    from huggingface_hub import hf_hub_download

    target_dir = get_data_dir() / bench_id / ref.local_subdir
    target_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=ref.hf_filename,
            revision=ref.hf_revision,
            local_dir=target_dir,
        )
    )
    # Flatten repo-side subdirs so ensure_artifact finds the file by basename only.
    path = target_dir / Path(ref.hf_filename).name
    if downloaded.resolve() != path.resolve():
        if path.exists():
            path.unlink()
        shutil.move(str(downloaded), str(path))
    st = path.stat()
    meta_path = _meta_path(path)
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(
        json.dumps(
            {
                "revision": ref.hf_revision,
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    )
    return path
