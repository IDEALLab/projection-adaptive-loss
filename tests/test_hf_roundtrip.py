import json
import os

import pytest

from pal.artifacts import core
from pal.artifacts.registry import REGISTRY


@pytest.mark.hf
def test_hf_roundtrip_smallest_artifact(tmp_path, monkeypatch):
    """Download the smallest artifact, verify sidecar + ensure_artifact roundtrip."""
    if "HF_TOKEN" not in os.environ:
        pytest.skip("HF_TOKEN not set; skipping HF smoke test")
    monkeypatch.setenv("PAL_DATA_DIR", str(tmp_path))

    ref = REGISTRY["e2/urban_wind"][0]
    assert ref.hf_filename.endswith("scalar_embedding.safetensors")

    path = core.download_artifact("e2/urban_wind", ref)
    assert path.exists()
    assert path.stat().st_size > 10_000_000

    meta = core._meta_path(path)
    assert meta.exists()
    m = json.loads(meta.read_text())
    assert m["revision"] == ref.hf_revision
    assert m["size"] == path.stat().st_size
    assert m["mtime_ns"] == path.stat().st_mtime_ns

    path2 = core.ensure_artifact("e2/urban_wind", ref)
    assert path2 == path

    meta.write_text(json.dumps({**m, "revision": "deadbeef"}))
    with pytest.raises(RuntimeError, match="Revision mismatch"):
        core.ensure_artifact("e2/urban_wind", ref)

    core.download_artifact("e2/urban_wind", ref)
    path.write_bytes(path.read_bytes() + b"x")
    with pytest.raises(RuntimeError, match="tampered"):
        core.ensure_artifact("e2/urban_wind", ref)
