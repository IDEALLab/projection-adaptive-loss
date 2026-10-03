#!/usr/bin/env python3
"""Tests for the Ax BO controller (scripts/bo/controller.py) and its helpers."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("ax")

from pal.eval.table1_constants import EPS1, EPS2  # noqa: E402
from pal.eval.table1_metrics import BOObjective  # noqa: E402
from scripts.bo import controller as ctrl_mod  # noqa: E402
from scripts.bo import ledger as ledger_mod  # noqa: E402
from scripts.bo.controller import CampaignConfig, Controller  # noqa: E402
from scripts.bo.executor import CellSpec  # noqa: E402
from scripts.bo.scoring import ScoringError, score_trial  # noqa: E402
from scripts.bo.search_spaces import (  # noqa: E402
    alpha_from_u,
    apply_snarenet_guard,
    u_from_alpha,
)


def make_result(cell: CellSpec, *, status="ok", obj=0.0, feas=1.0, viol=0.0,
                n_queries=64, attempt=None) -> dict:
    attempt = cell.attempt if attempt is None else attempt
    final = None
    if status in ("ok", "diverged"):
        final = {
            "obj_mean_post": obj, "feasibility_post": feas, "viol_max_post": viol,
            "n_queries": n_queries, "tolerance": 1e-4,
        }
    return {
        "status": status,
        "reason": None if status == "ok" else status,
        "fingerprint": {
            "git_sha": "testsha", "pal_file": "test",
            "method": cell.method, "bench": cell.bench, "seed": cell.seed,
            "requested_overrides": dict(cell.set_overrides),
            "command": ["fake"], "attempt": attempt, "wall_time_s": 0.01,
            "run_dir": str(Path(cell.trial_dir) / f"run_a{attempt}"),
        },
        "returncode": 0 if status == "ok" else 1,
        "stderr_tail": None,
        "final": final,
    }


def write_result_to_disk(cell: CellSpec, payload: dict) -> None:
    p = cell.result_path
    p.parent.mkdir(parents=True, exist_ok=True)
    # also create the referenced run dir so run-dir counting is meaningful
    rd = payload.get("fingerprint", {}).get("run_dir")
    if rd:
        Path(rd).mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, sort_keys=True))


class FakeExecutor:
    """Executor that fabricates results via ``responder(cell) -> dict`` and counts calls."""

    def __init__(self, responder=None):
        self.responder = responder or (lambda cell: make_result(cell))
        self.calls: list[CellSpec] = []
        self.max_workers = 4

    def submit_cell(self, cell: CellSpec) -> Future:
        self.calls.append(cell)
        payload = self.responder(cell)
        write_result_to_disk(cell, payload)
        fut: Future = Future()
        fut.set_result(payload)
        return fut

    def shutdown(self, wait: bool = True) -> None:
        pass


def make_cfg(tmp_path: Path, method="alm", *, sobol=2, bo=2, seeds=(100, 101),
             benches=("s1_sphere_track", "s2_active_set_switch"), max_in_flight=3,
             max_attempts=3, toy=True) -> CampaignConfig:
    return CampaignConfig(
        method=method, campaign_root=tmp_path, sobol=sobol, bo=bo,
        max_in_flight=max_in_flight, max_attempts=max_attempts, ax_seed=7, toy=toy,
        seeds=tuple(seeds), benches=tuple(benches), timeout_s=60.0, pool_size=None,
    )


def new_controller(cfg, executor):
    c = Controller.__new__(Controller)
    ctrl_mod.assert_ax_version()
    c.cfg = cfg
    c.git_sha = "testsha"
    from scripts.bo.search_spaces import METHOD_SPACES
    c.space = METHOD_SPACES[cfg.method]
    cfg.method_root.mkdir(parents=True, exist_ok=True)
    c.ledger = ledger_mod.Ledger(cfg.ledger_path, method=cfg.method,
                                 campaign_root=cfg.campaign_root, git_sha="testsha", toy=cfg.toy)
    c._own_executor = False
    c.executor = executor

    class _NoWandb:
        def log(self, *a, **k): pass
        def finish(self): pass
    c.wandb = _NoWandb()
    c.completed = {}
    c.abandoned = set()
    c.in_flight = {}
    c.ask_count = 0
    c.ax_client = None
    return c


def test_alpha_u_roundtrip():
    for u in (-3.0, -2.5, -2.0, -1.3, -1.0):
        assert u_from_alpha(alpha_from_u(u)) == pytest.approx(u, abs=1e-12)
    assert alpha_from_u(-3.0) == pytest.approx(0.999)
    assert alpha_from_u(-1.0) == pytest.approx(0.9)


def test_snarenet_guard_ok_clip_reject():
    ok = apply_snarenet_guard({"soft_epochs": 1000, "decay_epochs": 1000})
    assert ok.action == "ok"  # boundary 2000 is inclusive
    clipped = apply_snarenet_guard({"soft_epochs": 1600, "decay_epochs": 1000})
    assert clipped.action == "clipped" and clipped.params["decay_epochs"] == 250
    reject = apply_snarenet_guard({"soft_epochs": 1950, "decay_epochs": 1000})
    assert reject.action == "reject"


def test_scoring_all_feasible_golden(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", seeds=(100,))
    ov = {"lr": "1e-4", "gamma": "1e-2", "alpha": "0.99", "eps": "1e-8", "epochs": "5"}
    results = {}
    for bench in cfg.benches:
        cell = CellSpec("alm", bench, 100, str(tmp_path / bench), ov)
        results[(bench, 100)] = make_result(cell, status="ok", obj=0.0, feas=1.0, viol=0.0)
    ts = score_trial("alm", ov, (100,), results, benches=cfg.benches)
    assert ts.objective.l1 == pytest.approx(1.0)
    assert ts.objective.l2 == pytest.approx(0.0)
    assert ts.objective.l3 == pytest.approx(0.0)
    assert ts.objective.scalar == pytest.approx(1.0)
    assert ts.objective.diverged is False


def test_scoring_one_bench_diverged_golden(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", seeds=(100,))
    ov = {"lr": "1e-4", "gamma": "1e-2", "alpha": "0.99", "eps": "1e-8", "epochs": "5"}
    results = {}
    # s1 feasible, s2 fully diverged -> worst-cased inside the metric.
    c21 = CellSpec("alm", "s1_sphere_track", 100, str(tmp_path / "s1"), ov)
    results[("s1_sphere_track", 100)] = make_result(c21, status="ok", obj=0.0, feas=1.0, viol=0.0)
    c22 = CellSpec("alm", "s2_active_set_switch", 100, str(tmp_path / "s2"), ov)
    results[("s2_active_set_switch", 100)] = make_result(c22, status="diverged")
    ts = score_trial("alm", ov, (100,), results, benches=cfg.benches)
    assert ts.objective.l1 == pytest.approx(0.5)  # (1 + 0)/2
    assert ts.objective.l2 == pytest.approx(0.5)  # (0 + 1)/2  (diverged obj at ceiling)
    assert ts.objective.l3 == pytest.approx(0.5)  # (0 + 1)/2
    assert ts.objective.diverged is True


def test_fingerprint_mismatch_hard_errors(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", seeds=(100,))
    ov = {"lr": "1e-4", "gamma": "1e-2", "alpha": "0.99", "eps": "1e-8", "epochs": "5"}
    results = {}
    for bench in cfg.benches:
        cell = CellSpec("alm", bench, 100, str(tmp_path / bench), ov)
        r = make_result(cell)
        results[(bench, 100)] = r
    results[("s2_active_set_switch", 100)]["fingerprint"]["requested_overrides"]["gamma"] = "9.9"
    with pytest.raises(ScoringError, match="fingerprint mismatch"):
        score_trial("alm", ov, (100,), results, benches=cfg.benches)


def test_scoring_diverged_final_none_worst_cases_seed(tmp_path):
    """A crash-diverged cell (final=None) is worst-cased at the seed level like an eval NaN."""
    cfg = make_cfg(tmp_path, method="alm", seeds=(100,))
    ov = {"lr": "1e-4", "gamma": "1e-2", "alpha": "0.99", "eps": "1e-8", "epochs": "5"}
    results = {}
    c21 = CellSpec("alm", "s1_sphere_track", 100, str(tmp_path / "s1"), ov)
    results[("s1_sphere_track", 100)] = make_result(c21, status="ok", obj=0.0, feas=1.0, viol=0.0)
    c22 = CellSpec("alm", "s2_active_set_switch", 100, str(tmp_path / "s2"), ov)
    crash = make_result(c22, status="diverged")
    crash["final"] = None  # crash-diverged: no metrics at all
    results[("s2_active_set_switch", 100)] = crash
    ts = score_trial("alm", ov, (100,), results, benches=cfg.benches)
    # Identical tuple to the eval-time-NaN golden: s2 worst-cased.
    assert ts.objective.l1 == pytest.approx(0.5)
    assert ts.objective.l2 == pytest.approx(0.5)
    assert ts.objective.l3 == pytest.approx(0.5)
    assert ts.objective.diverged is True


def test_cell_set_mismatch_hard_errors(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", seeds=(100,))
    ov = {"lr": "1e-4", "gamma": "1e-2", "alpha": "0.99", "eps": "1e-8", "epochs": "5"}
    cell = CellSpec("alm", "s1_sphere_track", 100, str(tmp_path / "s1"), ov)
    results = {("s1_sphere_track", 100): make_result(cell)}
    with pytest.raises(ScoringError, match="cell set mismatch"):
        score_trial("alm", ov, (100,), results, benches=cfg.benches)


def test_trial_fanout_alm_18_cells():
    cfg = make_cfg(Path("/tmp/x"), method="alm", seeds=(100, 101, 102),
                   benches=None if False else (
                       "s1_sphere_track", "s2_active_set_switch", "s3_illcond_tube",
                       "s4_qv_coupling", "s5_overdetermined", "s6_redundant_ineq"),
                   toy=False)
    c = new_controller(cfg, FakeExecutor())
    cells = c._make_cells(0, {"lr": "1e-4"})
    assert len(cells) == 18
    assert ("s5_overdetermined", 100) in cells


def test_trial_fanout_dc3_15_cells():
    from pal.eval.table1_constants import TABLE1_BENCHES
    cfg = make_cfg(Path("/tmp/x"), method="dc3", seeds=(100, 101, 102),
                   benches=TABLE1_BENCHES, toy=False)
    c = new_controller(cfg, FakeExecutor())
    cells = c._make_cells(0, {"lr": "1e-4"})
    assert len(cells) == 15  # 6 benches - s5 (structural) = 5 benches x 3 seeds
    assert not any(b == "s5_overdetermined" for (b, _s) in cells)


def test_lex_winner_differs_from_best_scalar(tmp_path):
    cfg = make_cfg(tmp_path, method="alm")
    c = new_controller(cfg, FakeExecutor())

    def obj(l1, l2, l3):
        scalar = l1 - EPS1 * l2 - EPS2 * l3
        return BOObjective(l1=l1, l2=l2, l3=l3, scalar=scalar, n_applicable_benches=2,
                           n_seeds=2, n_queries=64, n_diverged_cells=0,
                           n_diverged_seeds=0, diverged=False)
    # Same L1. A: lower L2 but max L3. B: slightly higher L2 but zero L3.
    # Lex prefers A (L2 has priority); scalar prefers B (eps2*Delta L3 > eps1*Delta L2).
    a = obj(0.5, 0.0, 1.0)
    b = obj(0.5, 1e-4, 0.0)
    c.completed = {0: (a, {"tag": "A"}), 1: (b, {"tag": "B"})}
    best = c._best_so_far()
    assert best[0] == 0 and best[2]["tag"] == "A"          # lex winner
    assert max((0, 1), key=lambda i: c.completed[i][0].scalar) == 1  # argmax-S would pick B


def test_ledger_append_and_atomic_snapshot(tmp_path):
    led = ledger_mod.Ledger(tmp_path / "l.jsonl", method="alm",
                            campaign_root=tmp_path, git_sha="sha", toy=True)
    led.append(ledger_mod.EVENT_ASK, trial_index=0, config={"lr": "1e-4"})
    led.append(ledger_mod.EVENT_TRIAL_DONE, trial_index=0, outcome="scored",
               l1=1.0, l2=0.0, l3=0.0, scalar=1.0, diverged=False, config={})
    recs = led.read_all()
    assert [r["event"] for r in recs] == [ledger_mod.EVENT_ASK, ledger_mod.EVENT_TRIAL_DONE]
    assert all(r["method"] == "alm" and r["toy"] is True for r in recs)

    snap_path = tmp_path / "snap.json"
    ledger_mod.write_snapshot(snap_path, {"a": 1})
    assert not snap_path.with_suffix(".json.tmp").exists()  # atomic: no tmp left
    assert ledger_mod.read_snapshot(snap_path) == {"a": 1}


def test_full_fake_campaign_completes(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", sobol=2, bo=2)

    # Score by a hidden knob so BO has signal: prefer higher gamma.
    def responder(cell: CellSpec):
        gamma = float(cell.set_overrides.get("gamma", 1e-2))
        feas = min(1.0, 0.5 + gamma)
        return make_result(cell, status="ok", obj=0.0, feas=feas, viol=0.0)

    c = new_controller(cfg, FakeExecutor(responder))
    best = c.run(resume=False)
    assert best is not None
    assert len(c.completed) == cfg.budget  # 4 trials scored
    n_results = len(list(tmp_path.rglob("result.json")))
    assert n_results == cfg.budget * len(cfg.benches) * len(cfg.seeds)
    recs = c.ledger.read_all()
    scored = [r for r in recs if r.get("event") == "trial_done" and r.get("outcome") == "scored"]
    assert len(scored) == cfg.budget


def test_infra_retry_then_abandon(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", sobol=1, bo=0, seeds=(100,),
                   benches=("s1_sphere_track", "s2_active_set_switch"), max_attempts=3)

    def responder(cell: CellSpec):
        if cell.bench == "s2_active_set_switch":
            return make_result(cell, status="infra_failure", attempt=cell.attempt)
        return make_result(cell, status="ok")

    fake = FakeExecutor(responder)
    c = new_controller(cfg, fake)
    best = c.run(resume=False)
    assert best is None  # no trial scored (never a fake-worst number)
    assert 0 in c.abandoned
    recs = c.ledger.read_all()
    # attempt 1 + 2 retries -> attempt 3 exhausts; scope to ti=0.
    retries0 = [r for r in recs if r.get("event") == "retry"
                and r.get("bench") == "s2_active_set_switch" and r.get("trial_index") == 0]
    assert len(retries0) == cfg.max_attempts - 1
    abandon0 = [r for r in recs if r.get("event") == "abandon" and r.get("trial_index") == 0]
    assert len(abandon0) == 1
    td0 = [r for r in recs if r.get("event") == "trial_done" and r.get("trial_index") == 0]
    assert td0 and td0[0]["outcome"] == "abandoned"


def test_resume_reuses_completed_cells(tmp_path):
    cfg = make_cfg(tmp_path, method="alm", sobol=2, bo=0)

    c1 = new_controller(cfg, FakeExecutor())
    c1._init_ax(resume=False)
    params, ti = c1.ax_client.get_next_trial()
    c1._snapshot()
    overrides = c1._overrides_for(params)
    cells = c1._make_cells(ti, overrides)
    keys = sorted(cells)
    seeded = keys[:-1]           # all but the last cell completed pre-crash
    missing = keys[-1]
    for k in seeded:
        write_result_to_disk(cells[k], make_result(cells[k], status="ok"))
    c1.ledger.append(ledger_mod.EVENT_ASK, trial_index=ti, phase="sobol",
                     ax_params=params, config=overrides)

    seeded_bytes = {k: cells[k].result_path.read_bytes() for k in seeded}

    fake2 = FakeExecutor()
    c2 = new_controller(cfg, fake2)
    c2.cfg = cfg
    best = c2.run(resume=True)

    # The reconciled trial's pre-seeded cells were reused, never re-run.
    submitted_ti = {(cell.bench, cell.seed) for cell in fake2.calls if f"/trial_{ti}/" in cell.trial_dir}
    assert missing in submitted_ti                       # the missing cell ran
    for k in seeded:
        assert k not in submitted_ti                     # completed cells reused
        assert cells[k].result_path.read_bytes() == seeded_bytes[k]  # byte-identical
    assert best is not None


def _run_dir_count(cell_dir: Path) -> int:
    if not cell_dir.exists():
        return 0
    return sum(1 for p in cell_dir.iterdir() if p.is_dir())


@pytest.mark.slow
def test_gate_kill_resume(tmp_path):
    campaign = tmp_path / "campaign"
    env = dict(os.environ)
    cmd = [sys.executable, "-m", "scripts.bo.controller", "run",
           "--method", "alm", "--campaign-root", str(campaign), "--toy",
           "--max-in-flight", "2", "--no-wandb-online"]

    # Launch in its own process group so we can kill the whole tree.
    proc = subprocess.Popen(cmd, cwd=str(_REPO_ROOT), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    deadline = time.time() + 240
    first_results: list[Path] = []
    try:
        while time.time() < deadline:
            if proc.poll() is not None:
                break  # finished too fast; still valid but degenerate
            found = list(campaign.rglob("result.json"))
            oks = [p for p in found if json.loads(p.read_text()).get("status") in ("ok", "diverged")]
            if oks:
                first_results = oks
                break
            time.sleep(0.5)
        assert first_results, "no cell completed within the window"
        time.sleep(1.0)
        pre_kill = [p for p in campaign.rglob("result.json")
                    if json.loads(p.read_text()).get("status") in ("ok", "diverged")]
        pre_kill_bytes = {p: p.read_bytes() for p in pre_kill}
        pre_kill_rundirs = {p: _run_dir_count(p.parent) for p in pre_kill}
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait(timeout=30)

    assert pre_kill, "expected at least one completed cell before the kill"

    resume_cmd = [sys.executable, "-m", "scripts.bo.controller", "resume",
                  "--method", "alm", "--campaign-root", str(campaign), "--toy",
                  "--max-in-flight", "2", "--no-wandb-online"]
    out = subprocess.run(resume_cmd, cwd=str(_REPO_ROOT), env=env,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, f"resume failed: {out.stderr[-2000:]}"
    winner = json.loads(out.stdout.strip().splitlines()[-1])
    assert "winner_trial_index" in winner

    # Every pre-kill cell was reused: result.json byte-identical, run-dir count unchanged.
    for p, b in pre_kill_bytes.items():
        assert p.read_bytes() == b, f"completed cell {p} was rewritten on resume"
        assert _run_dir_count(p.parent) == pre_kill_rundirs[p], \
            f"completed cell {p.parent} gained a run dir (re-ran) on resume"

    recs = [
        json.loads(line)
        for line in (campaign / "alm" / "ledger.jsonl").read_text().splitlines()
        if line.strip()
    ]
    done = [r for r in recs if r.get("event") == "trial_done"]
    assert len(done) == 4
    for cell_dir in {p.parent for p in campaign.rglob("result.json")}:
        assert len(list(cell_dir.glob("result.json"))) == 1
