#!/usr/bin/env python3
"""Mocked tests for SlurmLaneExecutor (scripts/bo/executor.py).

``FakeSlurm`` fabricates sbatch/squeue/sacct; result.json files are written by hand.
"""

from __future__ import annotations

import json
import shlex
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from scripts.bo import controller as ctrl_mod
from scripts.bo import ledger as ledger_mod
from scripts.bo import slurm_lane
from scripts.bo.controller import CampaignConfig, Controller
from scripts.bo.executor import CellSpec, SlurmLaneExecutor


class FakeSlurm:
    """Fabricates sbatch/squeue/sacct: ``squeue``/``sacct`` map jobid -> {index: STATE},
    ``on_sbatch`` (optional) receives the parsed manifest."""

    def __init__(self, on_sbatch=None):
        self.job_counter = 9000
        self.sbatch_calls: list[list[str]] = []
        self.squeue: dict[str, dict[int, str]] = {}
        self.sacct: dict[str, dict[int, str]] = {}
        self._on_sbatch = on_sbatch

    def run(self, argv, **kwargs):
        cmd = argv[0]
        if cmd == "sbatch":
            self.sbatch_calls.append(argv)
            jid = str(self.job_counter)
            self.job_counter += 1
            if self._on_sbatch is not None:
                self._on_sbatch(self._manifest_from_script(argv[-1]))
            return CompletedProcess(argv, 0, stdout=jid + "\n", stderr="")
        if cmd in ("squeue", "sacct"):
            jid = argv[argv.index("-j") + 1]
            table = self.squeue if cmd == "squeue" else self.sacct
            rows = "".join(f"{jid}_{i}|{st}\n" for i, st in table.get(jid, {}).items())
            return CompletedProcess(argv, 0, stdout=rows, stderr="")
        return CompletedProcess(argv, 0, stdout="", stderr="")

    @staticmethod
    def _manifest_from_script(script_path: str) -> dict:
        text = Path(script_path).read_text()
        for line in text.splitlines():
            if "run-task" in line:
                return json.loads(Path(shlex.split(line)[-1]).read_text())
        raise AssertionError("no run-task line in sbatch script")


@pytest.fixture(autouse=True)
def _clean_submissions():
    slurm_lane._SUBMISSIONS.clear()
    yield
    slurm_lane._SUBMISSIONS.clear()


def _write_result(trial_dir: Path, status: str, *, attempt: int = 1,
                  overrides: dict | None = None) -> None:
    trial_dir.mkdir(parents=True, exist_ok=True)
    final = None
    if status in ("ok", "diverged"):
        final = {"obj_mean_post": 0.0, "feasibility_post": 1.0,
                 "viol_max_post": 0.0, "n_queries": 64, "tolerance": 1e-4}
    payload = {
        "status": status,
        "reason": None if status == "ok" else status,
        "fingerprint": {"method": "alm", "bench": trial_dir.name.split("_seed")[0],
                        "seed": 100, "requested_overrides": overrides or {},
                        "attempt": attempt, "run_dir": str(trial_dir / f"run_a{attempt}")},
        "returncode": 0 if status == "ok" else 1,
        "stderr_tail": None, "final": final,
    }
    (trial_dir / f"run_a{attempt}").mkdir(exist_ok=True)
    (trial_dir / "result.json").write_text(json.dumps(payload, sort_keys=True))


def _cells(root: Path, benches, seed=100, overrides=None) -> list[CellSpec]:
    overrides = overrides or {"lr": "1e-4"}
    return [
        CellSpec(method="alm", bench=b, seed=seed,
                 trial_dir=str(root / "alm" / "trial_0" / f"{b}_seed{seed}"),
                 set_overrides=dict(overrides), timeout_s=600.0, attempt=1)
        for b in benches
    ]


def test_submit_poll_and_surface(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    ex = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha",
                           resources={"preset": "cpu_lane", "time": "04:00:00"},
                           poll_interval=0.0, grace_s=0.0)

    cells = _cells(tmp_path, ("s1_sphere_track", "s2_active_set_switch",
                              "s3_illcond_tube"))
    handles = [ex.submit_cell(c) for c in cells]
    ex.flush()
    assert len(fs.sbatch_calls) == 1                      # exactly ONE array
    assert "--array=0-2" in fs.sbatch_calls[0]

    # Array running, no results yet -> nothing surfaced.
    jid = list(slurm_lane._SUBMISSIONS)[0]
    fs.squeue[jid] = {0: "RUNNING", 1: "RUNNING", 2: "RUNNING"}
    assert not any(h.done() for h in handles)

    for c in cells:
        _write_result(Path(c.trial_dir), "ok", overrides=c.set_overrides)
    assert all(h.done() for h in handles)
    assert all(h.result()["status"] == "ok" for h in handles)
    assert len(fs.sbatch_calls) == 1


def test_retry_only_after_array_terminal(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    ex = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                           max_attempts=3, poll_interval=0.0, grace_s=0.0)

    c0, c1 = _cells(tmp_path, ("s1_sphere_track", "s2_active_set_switch"))
    h0 = ex.submit_cell(c0)
    h1 = ex.submit_cell(c1)
    ex.flush()
    jid = list(slurm_lane._SUBMISSIONS)[0]

    # c0 done ok, c1 still running -> array not terminal -> no retry.
    _write_result(Path(c0.trial_dir), "ok", overrides=c0.set_overrides)
    fs.squeue[jid] = {1: "RUNNING"}
    assert h0.done() and not h1.done()
    assert len(fs.sbatch_calls) == 1                      # no retry while active

    # c1 now infra, array fully terminal -> retry submits exactly one compact array.
    _write_result(Path(c1.trial_dir), "infra_failure", attempt=1,
                  overrides=c1.set_overrides)
    fs.squeue[jid] = {}
    assert not h1.done()                                  # retry submitted, still pending
    assert len(fs.sbatch_calls) == 2
    assert fs.sbatch_calls[1][2] == "--array=0-0"         # only the one failed cell

    _write_result(Path(c1.trial_dir), "ok", attempt=2, overrides=c1.set_overrides)
    assert h1.done() and h1.result()["status"] == "ok"
    assert len(fs.sbatch_calls) == 2


def test_retry_exhaustion_surfaces_terminal_infra(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    ex = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                           max_attempts=2, poll_interval=0.0, grace_s=0.0)
    (c0,) = _cells(tmp_path, ("s1_sphere_track",))
    h0 = ex.submit_cell(c0)
    ex.flush()
    _write_result(Path(c0.trial_dir), "infra_failure", attempt=1, overrides=c0.set_overrides)
    assert not h0.done()
    assert len(fs.sbatch_calls) == 2
    # attempt 2 infra, at max -> no further retry, surfaced terminal.
    _write_result(Path(c0.trial_dir), "infra_failure", attempt=2, overrides=c0.set_overrides)
    assert h0.done()
    payload = h0.result()
    assert payload["status"] == "infra_failure"
    assert payload["fingerprint"]["attempt"] >= ex.max_attempts   # abandon, not resubmit
    assert len(fs.sbatch_calls) == 2


def test_resume_reregisters_and_does_not_resubmit(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    led = ledger_mod.Ledger(tmp_path / "alm" / "ledger.jsonl", method="alm",
                            campaign_root=tmp_path, git_sha="sha", toy=False)

    # Life 1: cell0 completes ok before the crash.
    c0, c1 = _cells(tmp_path, ("s1_sphere_track", "s2_active_set_switch"))
    ex1 = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                            poll_interval=0.0, grace_s=0.0, ledger=led)
    ex1.submit_cell(c0)
    ex1.submit_cell(c1)
    ex1.flush()
    assert len(fs.sbatch_calls) == 1
    _write_result(Path(c0.trial_dir), "ok", overrides=c0.set_overrides)
    c0_bytes = (Path(c0.trial_dir) / "result.json").read_bytes()

    # Simulate controller restart: the in-process jobid->manifest map is lost.
    slurm_lane._SUBMISSIONS.clear()

    # Resume: reuse cell0, resubmit cell1; the executor must re-register, not resubmit.
    ex2 = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                            poll_interval=0.0, grace_s=0.0, ledger=led)
    assert ex2._known                                     # rebuilt from the ledger
    h1 = ex2.submit_cell(c1)                              # only the missing cell
    ex2.flush()
    assert len(fs.sbatch_calls) == 1                      # no new sbatch on resume
    assert slurm_lane._SUBMISSIONS                        # manifest re-registered

    assert (Path(c0.trial_dir) / "result.json").read_bytes() == c0_bytes
    _write_result(Path(c1.trial_dir), "ok", overrides=c1.set_overrides)
    assert h1.done() and h1.result()["status"] == "ok"


def test_grace_window_prevents_premature_infra(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    ex = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                           max_attempts=3, poll_interval=0.0, grace_s=10_000.0)
    (c0,) = _cells(tmp_path, ("s1_sphere_track",))
    h0 = ex.submit_cell(c0)
    ex.flush()
    trial_root = str(Path(c0.trial_dir).parent)

    # No result.json and no squeue/sacct row yet: within grace this is not terminal infra.
    assert not h0.done()
    assert len(fs.sbatch_calls) == 1

    # Once grace elapses, the same no-row + no-result state IS infra -> retry.
    ex._batches[trial_root].submit_time = 0.0             # expire the grace window
    assert not h0.done()
    assert len(fs.sbatch_calls) == 2                      # retry now fired


def test_stale_result_not_reread_while_retry_pending(tmp_path, monkeypatch):
    fs = FakeSlurm()
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    ex = SlurmLaneExecutor(campaign_root=tmp_path, sha="sha", resources="cpu_lane",
                           max_attempts=3, poll_interval=0.0, grace_s=10_000.0)
    (c0,) = _cells(tmp_path, ("s1_sphere_track",))
    h0 = ex.submit_cell(c0)
    ex.flush()
    jid0 = list(slurm_lane._SUBMISSIONS)[0]
    trial_root = str(Path(c0.trial_dir).parent)

    _write_result(Path(c0.trial_dir), "infra_failure", attempt=1, overrides=c0.set_overrides)
    fs.sacct[jid0] = {0: "OUT_OF_MEMORY"}
    assert not h0.done()
    assert len(fs.sbatch_calls) == 2
    assert ex._batches[trial_root].attempt_round == 1

    # Retry invisible to squeue/sacct, stale attempt-1 result.json on disk: stay pending.
    for _ in range(3):
        assert not h0.done()
    assert len(fs.sbatch_calls) == 2
    jid1 = list(slurm_lane._SUBMISSIONS)[-1]
    assert slurm_lane.poll_states(jid1) == {0: "infra_failure"}  # raw classification...
    assert ex._grace_states(ex._batches[trial_root]) == {0: "pending"}  # ...downgraded

    # Even with the grace window expired, a RUNNING retry row keeps the cell active.
    ex._batches[trial_root].submit_time = 0.0
    fs.squeue[jid1] = {0: "RUNNING"}
    assert not h0.done()
    assert len(fs.sbatch_calls) == 2

    fs.squeue[jid1] = {}
    _write_result(Path(c0.trial_dir), "ok", attempt=2, overrides=c0.set_overrides)
    assert h0.done() and h0.result()["status"] == "ok"
    assert h0.result()["fingerprint"]["attempt"] == 2
    assert len(fs.sbatch_calls) == 2


def _slurm_cfg(tmp_path) -> CampaignConfig:
    return CampaignConfig(
        method="alm", campaign_root=tmp_path.resolve(), sobol=2, bo=0,
        max_in_flight=2, max_attempts=3, ax_seed=7, toy=True,
        seeds=(100, 101), benches=("s1_sphere_track", "s2_active_set_switch"),
        timeout_s=60.0, pool_size=None,
    )


def test_controller_end_to_end_slurm_lane(tmp_path, monkeypatch):
    pytest.importorskip("ax")
    # Fake sbatch 'runs' the cells: write an ok result.json for every manifest cell.
    def on_sbatch(manifest):
        for cell in manifest["cells"]:
            td = Path(cell["trial_dir"])
            td.mkdir(parents=True, exist_ok=True)
            (td / "run").mkdir(exist_ok=True)
            (td / "result.json").write_text(json.dumps({
                "status": "ok", "reason": None,
                "fingerprint": {"method": cell["method"], "bench": cell["bench"],
                                "seed": cell["seed"],
                                "requested_overrides": cell["set_overrides"],
                                "attempt": cell["attempt"], "run_dir": str(td / "run")},
                "returncode": 0, "stderr_tail": None,
                "final": {"obj_mean_post": 0.0, "feasibility_post": 1.0,
                          "viol_max_post": 0.0, "n_queries": 64, "tolerance": 1e-4},
            }, sort_keys=True))

    fs = FakeSlurm(on_sbatch=on_sbatch)
    monkeypatch.setattr(slurm_lane.subprocess, "run", fs.run)
    monkeypatch.setattr(ctrl_mod, "WandbRun", lambda *a, **k: _NoWandb())

    cfg = _slurm_cfg(tmp_path)
    ctrl = Controller(cfg, executor_kind="slurm",
                      slurm_resources={"preset": "cpu_lane", "time": "04:00:00"},
                      poll_interval=0.0, wandb_online=False)
    assert isinstance(ctrl.executor, SlurmLaneExecutor)
    ctrl.executor.grace_s = 0.0                            # results are immediate

    best = ctrl.run(resume=False)
    assert best is not None
    assert len(ctrl.completed) == cfg.budget               # 2 sobol trials scored
    submits = [r for r in ctrl.ledger.read_all()
               if r.get("event") == ledger_mod.EVENT_SLURM_SUBMIT]
    assert len(submits) == cfg.budget
    assert all(s.get("job_id") and s.get("manifest_path") for s in submits)


class _NoWandb:
    def log(self, *a, **k):
        pass

    def finish(self):
        pass
