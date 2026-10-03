#!/usr/bin/env python3
"""Append-only JSONL event ledger + atomic Ax-snapshot I/O for the BO controller.

One JSON object per event line, never rewritten. Every record carries ts, event, method,
campaign_root, git_sha and toy. Snapshots are written atomically after every ask and tell.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

EVENT_CAMPAIGN_START = "campaign_start"
EVENT_ASK = "ask"
EVENT_GUARD = "guard"
EVENT_SUBMISSION = "submission"
EVENT_CELL_DONE = "cell_done"
EVENT_RETRY = "retry"
EVENT_TRIAL_DONE = "trial_done"
EVENT_ABANDON = "abandon"
EVENT_CONFIRM = "confirm"
EVENT_SLURM_SUBMIT = "slurm_submit"


def _now() -> str:
    return datetime.now(UTC).isoformat()


class Ledger:
    """Append-only JSONL writer. Flushes + fsyncs each record for durability."""

    def __init__(self, path: str | Path, *, method: str, campaign_root: str | Path,
                 git_sha: str, toy: bool) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._common = {
            "method": method,
            "campaign_root": str(campaign_root),
            "git_sha": git_sha,
            "toy": bool(toy),
        }

    def append(self, event: str, **payload: Any) -> dict[str, Any]:
        record = {"ts": _now(), "event": event, **self._common, **payload}
        line = json.dumps(record, sort_keys=True)
        with self.path.open("a") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())
        return record

    def read_all(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        records: list[dict[str, Any]] = []
        with self.path.open() as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records


def write_snapshot(path: str | Path, snapshot: dict[str, Any]) -> None:
    """Atomically write an Ax JSON snapshot (tmp + fsync + os.replace)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(snapshot, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_snapshot(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)
