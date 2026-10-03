#!/usr/bin/env python3
"""Executor abstraction for BO cells: a local process pool and a Slurm array lane.

Each cell runs ``cell_executor.run_cell``, whose ``result.json`` on disk is the source of truth.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
from concurrent.futures import Future, ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from scripts.bo import ledger as ledger_mod
from scripts.bo import slurm_lane

_REPO_ROOT = Path(__file__).resolve().parents[2]

# Slurm task states that mean "not yet terminal" (poll_states lowercases these).
_ACTIVE_LANE_STATES = {
    "pending",
    "running",
    "completing",
    "configuring",
    "suspended",
}
# Terminal per-cell result statuses (scoreable data, never retried).
_TERMINAL_RESULT_STATUSES = {"ok", "diverged", "structurally_excluded"}


@dataclass(frozen=True)
class CellSpec:
    """One executable cell. ``set_overrides`` are LOGICAL keys (``lr`` included);
    the executor maps ``lr`` to ``--lr`` / ``--set learning_rate``."""

    method: str
    bench: str
    seed: int
    trial_dir: str
    set_overrides: dict[str, str] = field(default_factory=dict)
    timeout_s: float = 3600.0
    attempt: int = 1
    # alm_bolton only: root of the frozen ALM run dirs (one per bench/seed).
    frozen_alm_dir: str | None = None

    @property
    def cell_id(self) -> str:
        return f"{self.method}/{self.bench}/seed{self.seed}"

    @property
    def result_path(self) -> Path:
        return Path(self.trial_dir) / "result.json"


class Executor(Protocol):
    """Minimal surface the controller needs. ``submit_cell`` returns a handle
    exposing ``done()`` / ``result()`` (a ``concurrent.futures.Future`` for the
    local pool). ``result()`` yields the ``result.json`` payload dict."""

    def submit_cell(self, cell: CellSpec) -> Future: ...
    def shutdown(self, wait: bool = True) -> None: ...


def _pool_initializer(repo_root: str) -> None:
    # Re-assert the checkout root so `import pal` resolves to this checkout under spawn.
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    os.chdir(repo_root)


def _run_cell_worker(
    method: str,
    bench: str,
    seed: int,
    trial_dir: str,
    set_overrides: dict[str, str],
    timeout_s: float,
    attempt: int,
    python_bin: str,
    frozen_alm_dir: str | None = None,
) -> dict[str, Any]:
    # Imported here so the parent does not import cell_executor under spawn.
    from scripts.bo.cell_executor import run_cell

    return run_cell(
        method=method,
        bench=bench,
        seed=seed,
        trial_dir=Path(trial_dir),
        set_overrides=dict(set_overrides),
        timeout_s=timeout_s,
        attempt=attempt,
        python_bin=python_bin,
        frozen_alm_dir=frozen_alm_dir,
    )


def default_pool_size() -> int:
    return max(1, min(6, (os.cpu_count() or 2) - 2))


class LocalPoolExecutor:
    """Concurrent process pool over ``run_cell``. Cap defaults to
    ``min(6, cores-2)``; cells of multiple in-flight trials share the pool."""

    def __init__(self, max_workers: int | None = None, python_bin: str | None = None,
                 repo_root: str | Path = _REPO_ROOT) -> None:
        self.max_workers = max_workers or default_pool_size()
        self.python_bin = python_bin or sys.executable
        self._repo_root = str(Path(repo_root))
        self._pool = ProcessPoolExecutor(
            max_workers=self.max_workers,
            initializer=_pool_initializer,
            initargs=(self._repo_root,),
        )

    def submit_cell(self, cell: CellSpec) -> Future:
        return self._pool.submit(
            _run_cell_worker,
            cell.method,
            cell.bench,
            cell.seed,
            cell.trial_dir,
            dict(cell.set_overrides),
            cell.timeout_s,
            cell.attempt,
            self.python_bin,
            cell.frozen_alm_dir,
        )

    def shutdown(self, wait: bool = True) -> None:
        self._pool.shutdown(wait=wait)

    def __enter__(self) -> LocalPoolExecutor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=False)


def _read_result_json(path: Path) -> dict[str, Any] | None:
    """Read a cell ``result.json`` (or return None if absent/unreadable)."""
    if not path.exists():
        return None
    try:
        with path.open() as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _read_fresh_result_json(path: Path, expected_attempt: int) -> dict[str, Any] | None:
    """Like :func:`_read_result_json`, but a payload written by an EARLIER
    attempt (``fingerprint.attempt`` < ``expected_attempt``) counts as absent.

    Retry arrays reuse the trial_dir, so a stale failed payload must not end a retry early.
    """
    payload = _read_result_json(path)
    if payload is None:
        return None
    try:
        attempt = int((payload.get("fingerprint") or {}).get("attempt", 1))
    except (TypeError, ValueError):
        attempt = 1
    return None if attempt < expected_attempt else payload


class _SlurmCellHandle:
    """Future-like handle for one cell driven by a Slurm array batch.

    ``result()`` never raises: a raising result would make the controller mis-retry a cell.
    """

    def __init__(self, executor: SlurmLaneExecutor) -> None:
        self._exec = executor
        self._done = False
        self._payload: dict[str, Any] | None = None

    def _set(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self._done = True

    def done(self) -> bool:
        self._exec._maybe_poll()
        return self._done

    def result(self, timeout: float | None = None) -> dict[str, Any]:
        # Block-poll (rate-limited) so blocking callers such as confirm also work.
        while not self._done:
            self._exec._maybe_poll()
            if not self._done:
                time.sleep(min(self._exec.poll_interval, 1.0) or 0.01)
        assert self._payload is not None
        return self._payload


@dataclass
class _SlurmBatch:
    """One trial's cells -> one Slurm array (possibly followed by retry arrays)."""

    trial_root: str
    cells: dict[tuple[str, int], CellSpec] = field(default_factory=dict)
    handles: dict[tuple[str, int], _SlurmCellHandle] = field(default_factory=dict)
    job_id: str | None = None
    manifest_path: str | None = None
    submit_time: float = 0.0
    attempt_round: int = 0
    submitted: bool = False
    finalized: bool = False


class SlurmLaneExecutor:
    """Cluster lane executor: batches a trial's cells into ONE Slurm array.

    Retries fire only once the originating array is fully terminal. On resume, prior
    arrays are re-registered from the ledger instead of resubmitted.
    """

    def __init__(
        self,
        *,
        campaign_root: str | Path,
        sha: str,
        resources: str | Mapping[str, Any] | None = None,
        max_attempts: int = 3,
        poll_interval: float = 30.0,
        grace_s: float = 120.0,
        ledger: Any | None = None,
    ) -> None:
        self.campaign_root = Path(campaign_root).expanduser().resolve()
        self.sha = str(sha)
        self.resources = resources
        self.max_attempts = int(max_attempts)
        self.poll_interval = float(poll_interval)
        self.grace_s = float(grace_s)
        self._ledger = ledger
        self._batches: dict[str, _SlurmBatch] = {}
        self._last_poll = float("-inf")
        # trial_root -> (job_id, manifest_path, attempt_round) from prior life.
        self._known: dict[str, tuple[str, str, int]] = {}
        self._load_known_from_ledger()

    def _load_known_from_ledger(self) -> None:
        if self._ledger is None:
            return
        try:
            recs = self._ledger.read_all()
        except Exception:  # noqa: BLE001 - resume must never crash on a bad ledger
            return
        for rec in recs:
            if rec.get("event") != ledger_mod.EVENT_SLURM_SUBMIT:
                continue
            tr = rec.get("trial_root")
            jid = rec.get("job_id")
            mp = rec.get("manifest_path")
            if not (tr and jid and mp):
                continue
            rnd = int(rec.get("attempt_round", 0))
            prev = self._known.get(tr)
            if prev is None or rnd >= prev[2]:
                self._known[tr] = (str(jid), str(mp), rnd)

    def _record_submit(self, batch: _SlurmBatch) -> None:
        self._known[batch.trial_root] = (
            str(batch.job_id), str(batch.manifest_path), batch.attempt_round)
        if self._ledger is None:
            return
        try:
            self._ledger.append(
                ledger_mod.EVENT_SLURM_SUBMIT,
                trial_root=batch.trial_root,
                job_id=batch.job_id,
                manifest_path=batch.manifest_path,
                attempt_round=batch.attempt_round,
                cells=[{"bench": b, "seed": s} for (b, s) in sorted(batch.cells)],
            )
        except Exception:  # noqa: BLE001 - observability must never crash the run
            pass

    def submit_cell(self, cell: CellSpec) -> Future:
        trial_root = str(Path(cell.trial_dir).parent)
        batch = self._batches.get(trial_root)
        if batch is None:
            batch = _SlurmBatch(trial_root=trial_root)
            self._batches[trial_root] = batch
        key = (cell.bench, cell.seed)
        batch.cells[key] = cell
        handle = _SlurmCellHandle(self)
        batch.handles[key] = handle
        return handle  # type: ignore[return-value]

    @staticmethod
    def _cell_dict(cell: CellSpec) -> dict[str, Any]:
        return {
            "method": cell.method,
            "bench": cell.bench,
            "seed": cell.seed,
            "trial_dir": cell.trial_dir,
            "set_overrides": dict(cell.set_overrides),
            "timeout_s": cell.timeout_s,
            "attempt": cell.attempt,
            "frozen_alm_dir": cell.frozen_alm_dir,
        }

    def flush(self) -> None:
        """Dispatch (or, on resume, re-register) every not-yet-submitted batch."""
        for batch in list(self._batches.values()):
            if batch.submitted or batch.finalized:
                continue
            self._submit_batch(batch)

    def _submit_batch(self, batch: _SlurmBatch) -> None:
        known = self._known.get(batch.trial_root)
        if known is not None:
            # Resume: re-register the prior array instead of resubmitting.
            job_id, manifest_path, rnd = known
            slurm_lane.register_manifest(job_id, manifest_path)
            batch.job_id = job_id
            batch.manifest_path = manifest_path
            batch.attempt_round = rnd
            batch.submit_time = 0.0  # old array: grace has long elapsed
            batch.submitted = True
            return
        cell_dicts = [self._cell_dict(c) for c in batch.cells.values()]
        job_id = slurm_lane.submit_cells(
            cell_dicts, self.campaign_root, self.sha, self.resources)
        batch.job_id = job_id
        batch.manifest_path = str(slurm_lane.manifest_for_job(job_id))
        batch.submit_time = time.time()
        batch.submitted = True
        self._record_submit(batch)

    def _maybe_poll(self) -> None:
        now = time.monotonic()
        if now - self._last_poll >= self.poll_interval:
            self._poll_all()

    def _poll_all(self, force: bool = False) -> None:
        self._last_poll = time.monotonic()
        self.flush()  # safety net; a no-op once the controller has flushed
        for batch in list(self._batches.values()):
            if batch.finalized or not batch.submitted:
                continue
            self._poll_batch(batch)

    def _grace_states(self, batch: _SlurmBatch) -> dict[int, str]:
        """poll_states, but during the grace window a no-result + no-row
        ``infra_failure`` is downgraded to ``pending`` (sacct/squeue lag)."""
        states = slurm_lane.poll_states(str(batch.job_id))
        if time.time() - batch.submit_time >= self.grace_s:
            return states
        try:
            manifest = slurm_lane._load_manifest(batch.manifest_path)
        except (OSError, json.JSONDecodeError, KeyError):
            return states
        idx_dir = {int(c["array_index"]): str(c["trial_dir"])
                   for c in manifest.get("cells", [])}
        expected_attempt = batch.attempt_round + 1
        for idx, st in list(states.items()):
            if st != "infra_failure":
                continue
            trial_dir = idx_dir.get(idx)
            if trial_dir and _read_fresh_result_json(
                    Path(trial_dir) / "result.json", expected_attempt) is None:
                states[idx] = "pending"  # no result for THIS attempt yet
        return states

    def _poll_batch(self, batch: _SlurmBatch) -> None:
        states = self._grace_states(batch)
        active = any(st in _ACTIVE_LANE_STATES for st in states.values())

        # Surface terminal cells now; this never triggers a retry.
        unresolved: list[tuple[tuple[str, int], CellSpec, dict[str, Any] | None]] = []
        expected_attempt = batch.attempt_round + 1
        for key, cell in batch.cells.items():
            handle = batch.handles[key]
            if handle._done:
                continue
            payload = _read_fresh_result_json(
                Path(cell.trial_dir) / "result.json", expected_attempt)
            if payload is not None and payload.get("status") in _TERMINAL_RESULT_STATUSES:
                handle._set(payload)
            else:
                unresolved.append((key, cell, payload))

        if not unresolved:
            batch.finalized = True
            return
        if active:
            return  # array not fully terminal -> never retry (avoids result race)

        # Array fully terminal with unresolved cells: retry only now.
        new_job = slurm_lane.retry_failed(
            batch.manifest_path, str(batch.job_id), self.max_attempts)
        if new_job is not None:
            batch.job_id = new_job
            batch.manifest_path = str(slurm_lane.manifest_for_job(new_job))
            batch.submit_time = time.time()
            batch.attempt_round += 1
            self._record_submit(batch)
            return  # poll the retry array on the next cycle

        # Retries exhausted: surface a terminal infra failure so the trial is abandoned.
        for key, cell, payload in unresolved:
            handle = batch.handles[key]
            handle._set(self._terminal_infra(cell, payload))
        batch.finalized = True

    def _terminal_infra(
        self, cell: CellSpec, payload: dict[str, Any] | None
    ) -> dict[str, Any]:
        if payload is not None and payload.get("status") == "infra_failure":
            out = dict(payload)
            fp = dict(out.get("fingerprint") or {})
            fp["attempt"] = max(int(fp.get("attempt", 1)), self.max_attempts)
            out["fingerprint"] = fp
            return out
        return {
            "status": "infra_failure",
            "reason": "slurm cell produced no scoreable result after retries",
            "fingerprint": {
                "method": cell.method, "bench": cell.bench, "seed": cell.seed,
                "requested_overrides": dict(cell.set_overrides),
                "attempt": self.max_attempts, "run_dir": None,
            },
            "returncode": None, "stderr_tail": None, "final": None,
        }

    def shutdown(self, wait: bool = True) -> None:
        # No-op: leave arrays running so a resume can adopt them.
        return

    def __enter__(self) -> SlurmLaneExecutor:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown(wait=False)
