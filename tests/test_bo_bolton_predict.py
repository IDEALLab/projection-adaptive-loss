#!/usr/bin/env python3
"""Tests for the alm_bolton second-stage predict-only cell path and controller wiring."""
from __future__ import annotations

import json
import shutil
import sys
from concurrent.futures import Future
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

pytest.importorskip("ax")

from scripts.bo import cell_executor as ce  # noqa: E402
from scripts.bo import controller as ctrl_mod  # noqa: E402
from scripts.bo.controller import CampaignConfig, Controller, build_config  # noqa: E402
from scripts.bo.executor import CellSpec  # noqa: E402
from scripts.bo import ledger as ledger_mod  # noqa: E402


# ALM winner hparams a frozen run records (== asdict(ALMConfig)); grad_clip 20.0.
FROZEN_HPARAMS = {
    "gamma": 0.01, "alpha": 0.99, "eps": 1e-8, "mu_init": 1.0, "lambda_init": 1.0,
    "epochs": 40, "batch_size": 32, "lr": 1e-4, "grad_clip": 20.0,
    "hidden": 512, "n_layers": 4, "seed": 100, "device": "cpu",
    "predict_batch_size": None,
}


def _write_frozen_run(frozen_root: Path, bench: str, seed: int, *,
                      grad_clip: float = 20.0, fingerprint: str = "fp0") -> Path:
    rd = frozen_root / f"{bench}_seed{seed}"
    rd.mkdir(parents=True, exist_ok=True)
    hp = dict(FROZEN_HPARAMS); hp["grad_clip"] = grad_clip; hp["seed"] = seed
    (rd / "config.json").write_text(json.dumps({
        "method": "alm", "benchmark_id": bench, "seed": seed,
        "protocol": "synthetic", "eval_queries_fingerprint": fingerprint,
        "hparams": hp,
    }))
    (rd / "model.pt").write_bytes(b"\x00fake-weights")
    (rd / "final.json").write_text(json.dumps({
        "obj_mean_post": 0.05, "feasibility_post": 0.0, "viol_max_post": 2.5,
        "train_wall_time_s": 1.23, "n_queries": 64, "tolerance": 1e-4,
    }))
    return rd


def test_build_bolton_hparams_types_and_merges():
    hp = ce.build_bolton_hparams(FROZEN_HPARAMS, {"proj_delta": "1e-2", "proj_max_iters": "5"})
    assert hp["proj_delta"] == 0.01 and isinstance(hp["proj_delta"], float)
    assert hp["proj_max_iters"] == 5 and isinstance(hp["proj_max_iters"], int)
    assert hp["grad_clip"] == 20.0 and hp["epochs"] == 40 and hp["lr"] == 1e-4


def test_build_bolton_hparams_rejects_non_projector_key():
    with pytest.raises(ce.CellExecutorError, match="non-projector override"):
        ce.build_bolton_hparams(FROZEN_HPARAMS, {"grad_clip": "0.0"})


def test_assert_parity_passes_by_construction():
    hp = ce.build_bolton_hparams(FROZEN_HPARAMS, {"proj_delta": "1e-3", "proj_max_iters": "10"})
    ce.assert_bolton_matches_frozen_alm(hp, FROZEN_HPARAMS, bench="s1_sphere_track", seed=100)


def test_assert_parity_fails_on_grad_clip_drift():
    hp = ce.build_bolton_hparams(FROZEN_HPARAMS, {"proj_delta": "1e-3", "proj_max_iters": "10"})
    hp["grad_clip"] = 0.0  # the exact yaml 0.0-vs-20.0 bug class
    with pytest.raises(ce.CellExecutorError, match="prelaunch assert FAILED"):
        ce.assert_bolton_matches_frozen_alm(hp, FROZEN_HPARAMS, bench="s1_sphere_track", seed=100)


def test_assert_parity_fails_when_grad_clip_absent():
    frozen = {k: v for k, v in FROZEN_HPARAMS.items() if k != "grad_clip"}
    hp = ce.build_bolton_hparams(frozen, {"proj_delta": "1e-3", "proj_max_iters": "10"})
    with pytest.raises(ce.CellExecutorError, match="no.*grad_clip"):
        ce.assert_bolton_matches_frozen_alm(hp, frozen, bench="s1_sphere_track", seed=100)


def test_resolve_frozen_run_dir_layout(tmp_path):
    rd = ce.resolve_frozen_run_dir(tmp_path, "s1_sphere_track", 101)
    assert rd == tmp_path / "s1_sphere_track_seed101"


def test_stage_bolton_run_dir(tmp_path):
    frozen_root = tmp_path / "frozen"
    frozen_run = _write_frozen_run(frozen_root, "s1_sphere_track", 100)
    frozen_config = json.loads((frozen_run / "config.json").read_text())
    bolton_hp = ce.build_bolton_hparams(frozen_config["hparams"],
                                        {"proj_delta": "1e-2", "proj_max_iters": "5"})
    staged = tmp_path / "trial" / "predict_run"
    ce.stage_bolton_run_dir(frozen_run, staged, bolton_hp, frozen_config)

    cfg = json.loads((staged / "config.json").read_text())
    assert cfg["method"] == "alm_bolton"
    assert cfg["hparams"]["proj_delta"] == 0.01 and cfg["hparams"]["proj_max_iters"] == 5
    assert cfg["hparams"]["grad_clip"] == 20.0  # ALM winner inherited
    assert cfg["bolton_frozen_from"] == str(frozen_run)
    assert cfg["eval_queries_fingerprint"] == "fp0"  # carried over
    assert (staged / "model.pt").read_bytes() == b"\x00fake-weights"
    # final.json COPIED (not the frozen inode) so `pal eval` can overwrite it.
    assert (staged / "final.json").stat().st_ino != (frozen_run / "final.json").stat().st_ino


def _mock_eval(monkeypatch, *, returncode=0, final=None, write_final=True):
    """Patch subprocess.run + resolve_pal_file so no real eval/torch runs."""
    monkeypatch.setattr(ce, "resolve_pal_file", lambda *a, **k: "fakepal")

    class _CP:
        def __init__(self, rc): self.returncode, self.stdout, self.stderr = rc, "", ""

    def fake_run(cmd, **kwargs):
        runs_root = Path(cmd[cmd.index("--runs-root") + 1])
        run_id = cmd[cmd.index("--run-id") + 1]
        staged = runs_root / run_id
        if write_final:
            payload = final if final is not None else {
                "obj_mean_post": 16.78, "feasibility_post": 1.0,
                "viol_max_raw": 2.53, "viol_max_post": 4.7e-7,
                "n_queries": 64, "tolerance": 1e-4, "train_wall_time_s": 1.23,
                "inf_iters_median": 4.0, "inf_iters_max": 4.0,
                "inf_iters_n_converged": 64, "inf_iters_max_allowed": 10,
            }
            (staged / "final.json").write_text(json.dumps(payload))
        return _CP(returncode)

    monkeypatch.setattr(ce.subprocess, "run", fake_run)


def test_run_predict_cell_ok(tmp_path, monkeypatch):
    frozen_root = tmp_path / "frozen"
    _write_frozen_run(frozen_root, "s1_sphere_track", 100)
    _mock_eval(monkeypatch)
    r = ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=frozen_root,
                    set_overrides={"proj_delta": "1e-3", "proj_max_iters": "10"})
    assert r["status"] == "ok"
    # fingerprint must carry only the projector knobs (scoring matches on this).
    assert r["fingerprint"]["requested_overrides"] == {"proj_delta": "1e-3", "proj_max_iters": "10"}
    assert r["fingerprint"]["method"] == "alm_bolton"
    assert r["final"]["feasibility_post"] == 1.0
    assert r["final"]["inf_iters_max"] == 4.0
    assert json.loads((tmp_path / "t" / "result.json").read_text())["status"] == "ok"


def test_run_predict_cell_diverged_on_nan(tmp_path, monkeypatch):
    frozen_root = tmp_path / "frozen"
    _write_frozen_run(frozen_root, "s1_sphere_track", 100)
    _mock_eval(monkeypatch, final={
        "obj_mean_post": float("nan"), "feasibility_post": 0.0,
        "viol_max_post": float("inf"), "n_queries": 64, "tolerance": 1e-4,
    })
    r = ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=frozen_root,
                    set_overrides={"proj_delta": "1e-3", "proj_max_iters": "10"})
    assert r["status"] == "diverged"


def test_run_predict_cell_infra_on_nonzero_rc(tmp_path, monkeypatch):
    frozen_root = tmp_path / "frozen"
    _write_frozen_run(frozen_root, "s1_sphere_track", 100)
    _mock_eval(monkeypatch, returncode=1, write_final=False)
    r = ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=frozen_root,
                    set_overrides={"proj_delta": "1e-3", "proj_max_iters": "10"})
    assert r["status"] == "infra_failure" and "returncode=1" in r["reason"]


def test_run_predict_cell_infra_on_missing_frozen(tmp_path, monkeypatch):
    _mock_eval(monkeypatch)  # eval never reached
    r = ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=tmp_path / "nope",
                    set_overrides={"proj_delta": "1e-3", "proj_max_iters": "10"})
    assert r["status"] == "infra_failure" and "artifacts incomplete" in r["reason"]


def test_run_predict_cell_asserts_before_eval(tmp_path, monkeypatch):
    # grad_clip 0.0 in the frozen run -> parity assert fires, eval is never invoked.
    frozen_root = tmp_path / "frozen"
    _write_frozen_run(frozen_root, "s1_sphere_track", 100, grad_clip=0.0)
    # Poison bolton hparams path: mock build to force a drift the assert catches.
    orig = ce.build_bolton_hparams
    monkeypatch.setattr(ce, "build_bolton_hparams",
                        lambda fh, ov: {**orig(fh, ov), "grad_clip": 999.0})
    _mock_eval(monkeypatch)
    with pytest.raises(ce.CellExecutorError, match="prelaunch assert FAILED"):
        ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=frozen_root,
                    set_overrides={"proj_delta": "1e-3", "proj_max_iters": "10"})


def test_run_cell_frozen_dir_rejects_non_bolton(tmp_path):
    with pytest.raises(ce.CellExecutorError, match="only valid for method=alm_bolton"):
        ce.run_cell(method="alm", bench="s1_sphere_track", seed=100,
                    trial_dir=tmp_path / "t", frozen_alm_dir=tmp_path,
                    set_overrides={})


class _Args:
    def __init__(self, **kw):
        self.method = kw.get("method", "alm_bolton")
        self.campaign_root = kw["campaign_root"]
        self.trials = kw.get("trials")
        self.sobol = kw.get("sobol")
        self.toy = kw.get("toy", True)
        self.frozen_alm_dir = kw.get("frozen_alm_dir")
        for k in ("max_in_flight", "max_attempts", "ax_seed", "timeout_s", "pool_size"):
            setattr(self, k, kw.get(k))


def test_build_config_requires_frozen_dir(tmp_path):
    with pytest.raises(SystemExit, match="requires --frozen-alm-dir"):
        build_config(_Args(campaign_root=str(tmp_path), frozen_alm_dir=None))


def test_build_config_rejects_frozen_dir_for_other_method(tmp_path):
    with pytest.raises(SystemExit, match="only valid for method=alm_bolton"):
        build_config(_Args(method="alm", campaign_root=str(tmp_path),
                           frozen_alm_dir=str(tmp_path)))


def _bolton_cfg(tmp_path, frozen_root, *, seeds=(100, 101),
                benches=("s1_sphere_track", "s2_active_set_switch")) -> CampaignConfig:
    return CampaignConfig(
        method="alm_bolton", campaign_root=tmp_path / "camp", sobol=2, bo=2,
        max_in_flight=3, max_attempts=3, ax_seed=7, toy=True,
        seeds=tuple(seeds), benches=tuple(benches), timeout_s=60.0,
        pool_size=None, frozen_alm_dir=str(frozen_root),
    )


def test_overrides_skip_toy_extras_for_bolton(tmp_path):
    frozen = tmp_path / "frozen"
    cfg = _bolton_cfg(tmp_path, frozen)
    c = _new_controller(cfg, _FakeBoltonExecutor())
    ov = c._overrides_for({"proj_delta": 1e-3, "proj_max_iters": 10})
    assert set(ov) == {"proj_delta", "proj_max_iters"}  # no epochs=5 toy extra
    cells = c._make_cells(0, ov)
    assert all(cell.frozen_alm_dir == str(frozen) for cell in cells.values())


def test_prelaunch_check_missing_artifacts(tmp_path):
    frozen = tmp_path / "frozen"  # empty
    cfg = _bolton_cfg(tmp_path, frozen)
    c = _new_controller(cfg, _FakeBoltonExecutor())
    with pytest.raises(SystemExit, match="frozen ALM winner artifacts missing"):
        c._prelaunch_check_frozen_alm(cfg.seeds)


def test_full_fake_bolton_campaign(tmp_path):
    frozen = tmp_path / "frozen"
    benches = ("s1_sphere_track", "s2_active_set_switch")
    seeds = (100, 101)
    for b in benches:
        for s in seeds:
            _write_frozen_run(frozen, b, s)
    cfg = _bolton_cfg(tmp_path, frozen, seeds=seeds, benches=benches)
    c = _new_controller(cfg, _FakeBoltonExecutor())
    best = c.run(resume=False)
    assert best is not None
    ti, obj, ov = best
    assert set(ov) == {"proj_delta", "proj_max_iters"}
    # 4 trials scored (sobol 2 + bo 2), each 2x2 = 4 cells.
    scored = [r for r in ledger_mod.Ledger(cfg.ledger_path, method="alm_bolton",
              campaign_root=cfg.campaign_root, git_sha="x", toy=True).read_all()
              if r.get("event") == ledger_mod.EVENT_TRIAL_DONE and r.get("outcome") == "scored"]
    assert len(scored) == 4


class _FakeBoltonExecutor:
    """Writes a bolton-shaped result.json (requested_overrides = proj knobs)."""

    def __init__(self):
        self.max_workers = 4

    def submit_cell(self, cell: CellSpec) -> Future:
        assert cell.frozen_alm_dir is not None, "bolton cells must carry frozen_alm_dir"
        payload = {
            "status": "ok", "reason": None,
            "fingerprint": {
                "git_sha": "x", "pal_file": "fake", "method": cell.method,
                "bench": cell.bench, "seed": cell.seed,
                "requested_overrides": dict(cell.set_overrides),
                "command": ["fake"], "attempt": cell.attempt, "wall_time_s": 0.01,
                "run_dir": str(Path(cell.trial_dir) / "predict_run"),
            },
            "returncode": 0, "stderr_tail": None,
            "final": {"obj_mean_post": 1.0, "feasibility_post": 1.0,
                      "viol_max_post": 0.0, "n_queries": 64, "tolerance": 1e-4},
        }
        p = cell.result_path
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, sort_keys=True))
        fut: Future = Future(); fut.set_result(payload)
        return fut

    def shutdown(self, wait: bool = True) -> None:
        pass


def _new_controller(cfg, executor):
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


def test_slurm_manifest_carries_frozen_alm_dir(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    from scripts.bo import slurm_lane

    monkeypatch.setattr(slurm_lane.subprocess, "run",
                        lambda argv, **k: CompletedProcess([], 0, stdout="777\n", stderr=""))
    (tmp_path / "trials" / "bolton").mkdir(parents=True)
    jobid = slurm_lane.submit_cells(
        [{"method": "alm_bolton", "bench": "s1_sphere_track", "seed": 100,
          "trial_dir": tmp_path / "trials" / "bolton",
          "set_overrides": {"proj_delta": "1e-3", "proj_max_iters": "10"},
          "timeout_s": 600, "frozen_alm_dir": str(tmp_path / "frozen")}],
        tmp_path, "sha", {"preset": "cpu_lane", "time": "00:15:00"})
    manifest = json.loads(slurm_lane.manifest_for_job(jobid).read_text())
    assert manifest["cells"][0]["frozen_alm_dir"] == str(tmp_path / "frozen")


def test_slurm_manifest_omits_frozen_alm_dir_for_training_methods(tmp_path, monkeypatch):
    from subprocess import CompletedProcess

    from scripts.bo import slurm_lane

    monkeypatch.setattr(slurm_lane.subprocess, "run",
                        lambda argv, **k: CompletedProcess([], 0, stdout="778\n", stderr=""))
    (tmp_path / "trials" / "alm").mkdir(parents=True)
    jobid = slurm_lane.submit_cells(
        [{"method": "alm", "bench": "s1_sphere_track", "seed": 100,
          "trial_dir": tmp_path / "trials" / "alm",
          "set_overrides": {"lr": "1e-4"}, "timeout_s": 600}],
        tmp_path, "sha", {"preset": "cpu_lane", "time": "00:15:00"})
    manifest = json.loads(slurm_lane.manifest_for_job(jobid).read_text())
    assert "frozen_alm_dir" not in manifest["cells"][0]  # schema byte-identical


@pytest.mark.slow
def test_real_alm_train_then_bolton_predict(tmp_path):
    py = sys.executable
    train_dir = tmp_path / "almtrain"
    res = ce.run_cell(method="alm", bench="s1_sphere_track", seed=100,
                      trial_dir=train_dir, set_overrides={"epochs": "40"},
                      timeout_s=600, python_bin=py)
    assert res["status"] == "ok", res.get("stderr_tail")
    run_dir = Path(res["fingerprint"]["run_dir"])
    assert json.loads((run_dir / "config.json").read_text())["hparams"]["grad_clip"] == 20.0

    frozen = tmp_path / "frozen"
    shutil.copytree(run_dir, frozen / "s1_sphere_track_seed100")

    def predict(delta):
        r = ce.run_cell(method="alm_bolton", bench="s1_sphere_track", seed=100,
                        trial_dir=tmp_path / f"p_{delta}", frozen_alm_dir=frozen,
                        set_overrides={"proj_delta": delta, "proj_max_iters": "10"},
                        timeout_s=600, python_bin=py)
        assert r["status"] == "ok", r.get("stderr_tail")
        return r

    r1 = predict("1e-3")
    r2 = predict("1e-1")
    f1, f2 = r1["final"], r2["final"]
    # projector loads + runs: inference_iters populated, feasibility repaired.
    assert f1["inf_iters_max"] is not None
    assert f1["viol_max_post"] < f1["viol_max_raw"]  # projection improves feasibility
    assert f1["feasibility_post"] >= f1["feasibility_raw"]
    assert f1["viol_max_post"] != f2["viol_max_post"]
    assert r1["fingerprint"]["requested_overrides"] == {"proj_delta": "1e-3", "proj_max_iters": "10"}
