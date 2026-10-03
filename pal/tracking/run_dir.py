"""Run directory creation + config.json capture."""

from __future__ import annotations

import json
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _default_runs_root() -> Path:
    return Path.cwd() / "runs"


def create_run_dir(
    method: str,
    bench_id: str,
    seed: int,
    runs_root: Path | None = None,
    restart_shard: tuple[int, int] | None = None,
    deterministic: bool = False,
) -> Path:
    """Create a fresh run directory and return its path.

    Naming: `runs/<UTC timestamp>_<method>_<bench-sanitized>_seed<seed>_<uuid8>/`.
    `/` in bench ids becomes `-`. `restart_shard=(r, R)` adds `_r{r}_of{R}`.
    With `deterministic=True` the name is `<bench-slug>__<method>__seed<seed>`
    and an existing dir is reused, so resumed jobs find their checkpoint.
    """
    root = runs_root or _default_runs_root()
    bench_slug = bench_id.replace("/", "-")
    shard_suffix = ""
    if restart_shard is not None:
        r, R = restart_shard
        shard_suffix = f"_r{r}_of{R}"
    if deterministic:
        run_id = f"{bench_slug}__{method}__seed{seed}{shard_suffix}"
        path = root / run_id
        path.mkdir(parents=True, exist_ok=True)
        return path
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_id = (
        f"{stamp}_{method}_{bench_slug}_seed{seed}{shard_suffix}_{uuid.uuid4().hex[:8]}"
    )
    path = root / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def atomic_torch_save(obj: Any, path: Path) -> None:
    """`torch.save(obj, path)` that survives a mid-write kill.

    Writes to a sibling tmp file, then renames over `path` atomically.
    """
    import torch

    path = Path(path)
    tmp = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def write_config_json(run_dir: Path, config: dict[str, Any]) -> None:
    """Serialize the run config to `<run_dir>/config.json`.

    The caller is responsible for converting non-JSON-serializable fields
    (tensors, dataclasses). For benchmark specs use `spec.to_json_dict()`.
    """
    with (run_dir / "config.json").open("w") as f:
        json.dump(config, f, indent=2, sort_keys=True, default=str)
