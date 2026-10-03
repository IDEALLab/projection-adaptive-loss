#!/usr/bin/env python3
"""Ax Service API ask-tell BO controller for method tuning.

A trial is one config evaluated on every (bench, tuning seed) cell. All state lives on disk
(cell result.json, Ax snapshot, JSONL ledger), so ``resume`` never re-runs a completed cell.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import ax
except ImportError:  # ax-platform ships in the optional `bo` extra
    ax = None

from pal.eval.table1_constants import TABLE1_BENCHES
from pal.eval.table1_metrics import BOObjective, applicable_benches, select_best_trial

from scripts.bo import ledger as ledger_mod
from scripts.bo.executor import (
    CellSpec,
    Executor,
    LocalPoolExecutor,
    SlurmLaneExecutor,
)
from scripts.bo.scoring import ScoringError, read_result, score_trial
from scripts.bo.search_spaces import (
    METHOD_SPACES,
    apply_snarenet_guard,
    dc3_effective_soft_weight,
    default_budget,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_REQUIREMENTS = _REPO_ROOT / "scripts" / "bo" / "requirements-bo.txt"

# Not "S": Ax decodes objective names through sympy on resume, where S is a builtin.
OBJECTIVE_NAME = "score"

TUNING_SEEDS = (100, 101, 102)
TOY_SEEDS = (100, 101)
CONFIRM_SEEDS = (103, 104, 105, 106, 107)
TOY_BENCHES = ("s1_sphere_track", "s2_active_set_switch")


def _pinned_ax_version() -> str:
    text = _REQUIREMENTS.read_text()
    m = re.search(r"^\s*ax-platform==([\w.]+)", text, re.MULTILINE)
    if not m:
        raise RuntimeError(f"could not find ax-platform pin in {_REQUIREMENTS}")
    return m.group(1)


def assert_ax_version() -> None:
    pinned = _pinned_ax_version()
    if ax is None:
        raise RuntimeError(
            "ax-platform is not installed. Install the `bo` extra "
            "(pip install -e '.[bo]') or scripts/bo/requirements-bo.txt."
        )
    if ax.__version__ != pinned:
        raise RuntimeError(
            f"installed ax-platform {ax.__version__} != pinned {pinned} "
            f"(scripts/bo/requirements-bo.txt). Reinstall the pin or bump the file."
        )


@dataclass(frozen=True)
class CampaignConfig:
    method: str
    campaign_root: Path
    sobol: int
    bo: int
    max_in_flight: int
    max_attempts: int
    ax_seed: int
    toy: bool
    seeds: tuple[int, ...]
    benches: tuple[str, ...]
    timeout_s: float
    pool_size: int | None
    # alm_bolton only: root of the frozen ALM winner run dirs (<root>/<bench>_seed<seed>/).
    frozen_alm_dir: str | None = None

    @property
    def budget(self) -> int:
        return self.sobol + self.bo

    @property
    def method_root(self) -> Path:
        return self.campaign_root / self.method

    @property
    def snapshot_path(self) -> Path:
        return self.method_root / "ax_snapshot.json"

    @property
    def ledger_path(self) -> Path:
        return self.method_root / "ledger.jsonl"

    @property
    def meta_path(self) -> Path:
        return self.method_root / "campaign_meta.json"

    def trial_dir(self, ti: int, bench: str, seed: int, *, subtree: str = "") -> Path:
        base = self.method_root / subtree if subtree else self.method_root
        return base / f"trial_{ti}" / f"{bench}_seed{seed}"

    def toy_extras(self) -> dict[str, str]:
        return {"epochs": "5"} if self.toy else {}


def build_config(args: argparse.Namespace) -> CampaignConfig:
    method = args.method
    if method not in METHOD_SPACES:
        raise SystemExit(f"unknown method {method!r}; known: {sorted(METHOD_SPACES)}")
    toy = bool(getattr(args, "toy", False))
    d_sobol, d_bo = default_budget(method)
    if toy:
        sobol = 2
        total = 4
    else:
        total = getattr(args, "trials", None) or (d_sobol + d_bo)
        sobol = getattr(args, "sobol", None) or d_sobol
    bo = total - sobol
    if bo < 0:
        raise SystemExit(f"--trials ({total}) must be >= --sobol ({sobol})")
    seeds = TOY_SEEDS if toy else TUNING_SEEDS
    benches = TOY_BENCHES if toy else TABLE1_BENCHES
    frozen_alm_dir = getattr(args, "frozen_alm_dir", None)
    if method == "alm_bolton" and not frozen_alm_dir:
        raise SystemExit(
            "method=alm_bolton is predict-only and requires --frozen-alm-dir "
            "pointing at the frozen ALM winner run dirs "
            "(<root>/<bench>_seed<seed>/{config.json,model.pt,final.json})."
        )
    if frozen_alm_dir and method != "alm_bolton":
        raise SystemExit("--frozen-alm-dir is only valid for method=alm_bolton")
    return CampaignConfig(
        method=method,
        campaign_root=Path(args.campaign_root).expanduser().resolve(),
        sobol=sobol,
        bo=bo,
        max_in_flight=getattr(args, "max_in_flight", None) or 3,
        max_attempts=getattr(args, "max_attempts", None) or 3,
        ax_seed=getattr(args, "ax_seed", None) if getattr(args, "ax_seed", None) is not None else 0,
        toy=toy,
        seeds=tuple(seeds),
        benches=tuple(benches),
        timeout_s=getattr(args, "timeout_s", None) or 3600.0,
        pool_size=getattr(args, "pool_size", None),
        frozen_alm_dir=(str(Path(frozen_alm_dir).expanduser().resolve())
                        if frozen_alm_dir else None),
    )


def _git_sha() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# W&B dirs live on scratch (home is quota-limited).
_SCRATCH = os.environ.get("SCRATCH", "/tmp")
WANDB_DIR = os.path.join(_SCRATCH, "wandb")
WANDB_CACHE_DIR = os.path.join(_SCRATCH, "wandb_cache")
WANDB_ARTIFACT_DIR = os.path.join(_SCRATCH, "wandb_artifacts")
WANDB_ENTITY = os.environ.get("WANDB_ENTITY")


class WandbRun:
    def __init__(self, cfg: CampaignConfig, project: str, online: bool,
                 entity: str | None = WANDB_ENTITY) -> None:
        self._run = None
        try:
            import os
            os.environ.setdefault("WANDB_SILENT", "true")
            os.environ.setdefault("WANDB_CACHE_DIR", WANDB_CACHE_DIR)
            os.environ.setdefault("WANDB_ARTIFACT_DIR", WANDB_ARTIFACT_DIR)
            os.makedirs(WANDB_DIR, exist_ok=True)
            import wandb  # type: ignore
            self._wandb = wandb
            self._run = wandb.init(
                entity=entity,
                project=project,
                name=f"{cfg.method}{'-toy' if cfg.toy else ''}",
                mode="online" if online else "offline",
                dir=WANDB_DIR,
                config={
                    "method": cfg.method, "sobol": cfg.sobol, "bo": cfg.bo,
                    "seeds": list(cfg.seeds), "benches": list(cfg.benches),
                    "ax_seed": cfg.ax_seed, "toy": cfg.toy,
                },
                reinit=True,
            )
        except Exception as e:  # noqa: BLE001
            sys.stderr.write(f"[wandb] disabled ({type(e).__name__}: {e})\n")
            self._run = None

    def log(self, data: dict[str, Any], step: int | None = None) -> None:
        if self._run is None:
            return
        try:
            self._wandb.log(data, step=step)
        except Exception:  # noqa: BLE001
            pass

    def finish(self) -> None:
        if self._run is None:
            return
        try:
            self._run.finish()
        except Exception:  # noqa: BLE001
            pass


@dataclass
class TrialRun:
    ti: int
    overrides: dict[str, str]           # full logical config (incl. toy epochs)
    phase: str                          # "sobol" | "bo"
    cells: dict[tuple[str, int], CellSpec]
    pending: dict[tuple[str, int], Future] = field(default_factory=dict)
    results: dict[tuple[str, int], dict] = field(default_factory=dict)  # scoreable
    infra_exhausted: bool = False

    @property
    def all_keys(self) -> set[tuple[str, int]]:
        return set(self.cells.keys())

    @property
    def done_collecting(self) -> bool:
        return not self.pending and (self.infra_exhausted or self.results.keys() == self.cells.keys())


class Controller:
    def __init__(self, cfg: CampaignConfig, *, executor: Executor | None = None,
                 executor_kind: str = "local",
                 slurm_resources: str | dict[str, Any] | None = None,
                 poll_interval: float = 30.0,
                 wandb_project: str = "pal-bo", wandb_online: bool = True,
                 wandb_entity: str | None = WANDB_ENTITY) -> None:
        assert_ax_version()
        self.cfg = cfg
        self.git_sha = _git_sha()
        self.space = METHOD_SPACES[cfg.method]
        cfg.method_root.mkdir(parents=True, exist_ok=True)
        self.ledger = ledger_mod.Ledger(
            cfg.ledger_path, method=cfg.method, campaign_root=cfg.campaign_root,
            git_sha=self.git_sha, toy=cfg.toy,
        )
        self._own_executor = executor is None
        # An explicit executor (tests) wins; the Slurm lane needs the ledger built above.
        if executor is not None:
            self.executor = executor
        elif executor_kind == "slurm":
            self.executor = SlurmLaneExecutor(
                campaign_root=cfg.campaign_root, sha=self.git_sha,
                resources=slurm_resources, max_attempts=cfg.max_attempts,
                poll_interval=poll_interval, ledger=self.ledger,
            )
        else:
            self.executor = LocalPoolExecutor(max_workers=cfg.pool_size)
        self.wandb = WandbRun(cfg, wandb_project, wandb_online, entity=wandb_entity)

        # completed[ti] = (BOObjective, overrides); abandoned = set of ti.
        self.completed: dict[int, tuple[BOObjective, dict[str, str]]] = {}
        self.abandoned: set[int] = set()
        self.in_flight: dict[int, TrialRun] = {}
        self.ask_count = 0  # number of get_next_trial calls issued this campaign
        self.ax_client = None  # set in _init_ax

    def _generation_strategy(self):
        from ax.generation_strategy.generation_node import GenerationStep, Generators
        from ax.generation_strategy.generation_strategy import GenerationStrategy
        return GenerationStrategy(nodes=[
            GenerationStep(
                generator=Generators.SOBOL, num_trials=self.cfg.sobol,
                min_trials_observed=1, max_parallelism=max(1, self.cfg.sobol),
            ),
            GenerationStep(
                generator=Generators.BOTORCH_MODULAR, num_trials=-1,
                max_parallelism=self.cfg.max_in_flight,
            ),
        ])

    def _init_ax(self, resume: bool) -> None:
        from ax.service.ax_client import AxClient
        from ax.service.utils.instantiation import ObjectiveProperties

        snap = ledger_mod.read_snapshot(self.cfg.snapshot_path) if resume else None
        if snap is not None:
            self.ax_client = AxClient.from_json_snapshot(snap)
        else:
            self.ax_client = AxClient(
                generation_strategy=self._generation_strategy(),
                random_seed=self.cfg.ax_seed, verbose_logging=False,
            )
            self.ax_client.create_experiment(
                name=f"bo_{self.cfg.method}",
                parameters=self.space.ax_parameters,
                objectives={OBJECTIVE_NAME: ObjectiveProperties(minimize=False)},
            )
            self._write_meta()
            self.ledger.append(
                ledger_mod.EVENT_CAMPAIGN_START,
                sobol=self.cfg.sobol, bo=self.cfg.bo, budget=self.cfg.budget,
                seeds=list(self.cfg.seeds), benches=list(self.cfg.benches),
                max_in_flight=self.cfg.max_in_flight, ax_seed=self.cfg.ax_seed,
                pool_size=getattr(self.executor, "max_workers", None),
            )

    def _write_meta(self) -> None:
        ledger_mod.write_snapshot(self.cfg.meta_path, {
            "method": self.cfg.method, "toy": self.cfg.toy,
            "sobol": self.cfg.sobol, "bo": self.cfg.bo,
            "seeds": list(self.cfg.seeds), "benches": list(self.cfg.benches),
            "max_in_flight": self.cfg.max_in_flight, "ax_seed": self.cfg.ax_seed,
        })

    def _snapshot(self) -> None:
        ledger_mod.write_snapshot(self.cfg.snapshot_path, self.ax_client.to_json_snapshot())

    def _overrides_for(self, params: dict[str, Any]) -> dict[str, str]:
        ov = self.space.to_overrides(params)
        # alm_bolton is predict-only: never apply the toy training epochs.
        if self.cfg.method != "alm_bolton":
            ov.update(self.cfg.toy_extras())
        return ov

    def _make_cells(self, ti: int, overrides: dict[str, str]) -> dict[tuple[str, int], CellSpec]:
        cells: dict[tuple[str, int], CellSpec] = {}
        for bench in applicable_benches(self.cfg.method, self.cfg.benches):
            for seed in self.cfg.seeds:
                td = self.cfg.trial_dir(ti, bench, seed)
                cells[(bench, seed)] = CellSpec(
                    method=self.cfg.method, bench=bench, seed=seed,
                    trial_dir=str(td), set_overrides=dict(overrides),
                    timeout_s=self.cfg.timeout_s, attempt=1,
                    frozen_alm_dir=self.cfg.frozen_alm_dir,
                )
        return cells

    def _flush_executor(self) -> None:
        """Signal batching executors (Slurm lane) that a trial's cells are all
        queued and its array may be dispatched. No-op for the local pool."""
        flush = getattr(self.executor, "flush", None)
        if callable(flush):
            flush()

    def _submit_or_reuse(self, run: TrialRun, key: tuple[str, int]) -> None:
        """Reuse a matching scoreable result.json; else (re)submit the cell.

        Reuse guarantees a completed cell is not re-run on resume (the gate).
        Infra results on disk are retried (attempt read from their fingerprint).
        """
        cell = run.cells[key]
        existing = read_result(cell.result_path)
        if existing is not None:
            status = existing.get("status")
            fp = existing.get("fingerprint", {}) or {}
            req = {str(k): str(v) for k, v in (fp.get("requested_overrides") or {}).items()}
            matches = req == {str(k): str(v) for k, v in cell.set_overrides.items()}
            if status in ("ok", "diverged") and matches:
                run.results[key] = existing            # reuse, do not re-run
                return
            if status == "infra_failure" and matches:
                attempt = int(fp.get("attempt", 1))
                if attempt >= self.cfg.max_attempts:
                    run.infra_exhausted = True
                    return
                cell = CellSpec(**{**cell.__dict__, "attempt": attempt + 1})
                run.cells[key] = cell
                self.ledger.append(
                    ledger_mod.EVENT_RETRY, trial_index=run.ti, bench=key[0], seed=key[1],
                    attempt_from=attempt, attempt_to=attempt + 1, reason="infra_failure",
                )
        run.pending[key] = self.executor.submit_cell(cell)

    def _phase(self) -> str:
        return "sobol" if self.ask_count < self.cfg.sobol else "bo"

    def _sobol_gate_open(self) -> bool:
        """BO may only begin once all Sobol trials (indices < sobol) are terminal."""
        for ti in range(self.cfg.sobol):
            if ti in self.completed or ti in self.abandoned:
                continue
            if ti in self.in_flight:
                return False
            # not yet asked -> still in Sobol phase; gate not relevant here
        return True

    def _ask_one(self) -> bool:
        """Ask Ax for one trial, build+submit its cells. Returns True if asked."""
        from ax.exceptions.core import DataRequiredError
        from ax.exceptions.generation_strategy import MaxParallelismReachedException

        phase = self._phase()
        if phase == "bo" and not self._sobol_gate_open():
            return False
        try:
            params, ti = self.ax_client.get_next_trial()
        except (DataRequiredError, MaxParallelismReachedException):
            return False
        self.ask_count += 1
        self._snapshot()  # after every ask

        # snarenet validity guard (reject-and-re-ask or clip).
        params, guarded = self._apply_guard(ti, params)
        if guarded == "reject":
            self.ax_client.abandon_trial(ti, reason="snarenet_validity_reject")
            self._snapshot()
            self.abandoned.add(ti)
            return True  # counts as an ask; loop will ask a replacement

        overrides = self._overrides_for(params)
        self.ledger.append(
            ledger_mod.EVENT_ASK, trial_index=ti, phase=phase,
            ax_params=params, config=overrides,
        )
        cells = self._make_cells(ti, overrides)
        run = TrialRun(ti=ti, overrides=overrides, phase=phase, cells=cells)

        sub_payload: dict[str, Any] = {
            "trial_index": ti, "config": overrides,
            "config_fingerprint": overrides,
            "cells": [{"bench": b, "seed": s, "trial_dir": c.trial_dir}
                      for (b, s), c in cells.items()],
        }
        if self.cfg.method == "dc3":
            eff = {}
            for bench in applicable_benches(self.cfg.method, self.cfg.benches):
                e, use_compl = dc3_effective_soft_weight(float(overrides["soft_weight"]), bench)
                eff[bench] = {"effective": e, "use_compl": use_compl}
            sub_payload["dc3_soft_weight"] = {"requested": float(overrides["soft_weight"]), "per_bench": eff}
        self.ledger.append(ledger_mod.EVENT_SUBMISSION, **sub_payload)

        for key in cells:
            self._submit_or_reuse(run, key)
        self._flush_executor()  # dispatch this trial's Slurm array (no-op: local)
        self.in_flight[ti] = run
        return True

    def _apply_guard(self, ti: int, params: dict[str, Any]) -> tuple[dict[str, Any], str]:
        if self.cfg.method != "snarenet":
            return params, "ok"
        gr = apply_snarenet_guard(params)
        if gr.action != "ok":
            self.ledger.append(
                ledger_mod.EVENT_GUARD, trial_index=ti, action=gr.action, note=gr.note,
                params_in=dict(params), params_out=dict(gr.params),
            )
        return gr.params, gr.action

    def _harvest(self, run: TrialRun) -> None:
        """Move finished futures into results/infra; schedule retries."""
        done_keys = [k for k, fut in run.pending.items() if fut.done()]
        for key in done_keys:
            fut = run.pending.pop(key)
            try:
                result = fut.result()
            except Exception as e:  # noqa: BLE001 - worker crash == infra
                result = {"status": "infra_failure", "reason": f"worker: {e}",
                          "fingerprint": {"attempt": run.cells[key].attempt}}
            status = result.get("status")
            fp = result.get("fingerprint", {}) or {}
            self.ledger.append(
                ledger_mod.EVENT_CELL_DONE, trial_index=run.ti, bench=key[0], seed=key[1],
                attempt=int(fp.get("attempt", run.cells[key].attempt)), status=status,
                run_dir=fp.get("run_dir"), result_path=str(run.cells[key].result_path),
            )
            if status in ("ok", "diverged"):
                run.results[key] = result
            elif status == "structurally_excluded":
                # Should not occur (we never generate dc3 x s5), but treat as data.
                run.results[key] = result
            else:  # infra_failure
                attempt = int(fp.get("attempt", run.cells[key].attempt))
                if attempt >= self.cfg.max_attempts:
                    run.infra_exhausted = True
                else:
                    new_cell = CellSpec(**{**run.cells[key].__dict__, "attempt": attempt + 1})
                    run.cells[key] = new_cell
                    self.ledger.append(
                        ledger_mod.EVENT_RETRY, trial_index=run.ti, bench=key[0], seed=key[1],
                        attempt_from=attempt, attempt_to=attempt + 1, reason="infra_failure",
                    )
                    run.pending[key] = self.executor.submit_cell(new_cell)

    def _finalize(self, run: TrialRun) -> None:
        ti = run.ti
        if run.infra_exhausted:
            infra_cells = [{"bench": b, "seed": s} for (b, s) in run.all_keys
                           if (b, s) not in run.results]
            self.ax_client.abandon_trial(ti, reason="infra_failure_exhausted")
            self._snapshot()
            self.abandoned.add(ti)
            self.ledger.append(ledger_mod.EVENT_ABANDON, trial_index=ti,
                               reason="infra_failure_exhausted", cells_infra=infra_cells)
            self.ledger.append(ledger_mod.EVENT_TRIAL_DONE, trial_index=ti,
                               outcome="abandoned", config=run.overrides)
            del self.in_flight[ti]
            return

        try:
            ts = score_trial(self.cfg.method, run.overrides, self.cfg.seeds,
                             run.results, benches=self.cfg.benches)
        except ScoringError:
            raise  # invariant violation -> hard error, by design

        obj = ts.objective
        self.ax_client.complete_trial(ti, raw_data={OBJECTIVE_NAME: (float(obj.scalar), 0.0)})
        self._snapshot()  # after every tell
        self.completed[ti] = (obj, run.overrides)
        self.ledger.append(
            ledger_mod.EVENT_TRIAL_DONE, trial_index=ti, outcome="scored",
            l1=obj.l1, l2=obj.l2, l3=obj.l3, scalar=obj.scalar,
            diverged=obj.diverged or ts.worst, config=run.overrides,
        )
        self._log_wandb(ti, ts)
        del self.in_flight[ti]
        self._write_status()

    def _log_wandb(self, ti: int, ts) -> None:
        obj = ts.objective
        data: dict[str, Any] = {
            "trial": ti, "score": obj.scalar, "l1": obj.l1, "l2": obj.l2, "l3": obj.l3,
            "n_diverged_seeds": obj.n_diverged_seeds,
        }
        for bench, m in ts.per_bench.items():
            data[f"feas/{bench}"] = m["feas_bo"]
            data[f"obj/{bench}"] = m["obj_bo"]
        best = self._best_so_far()
        if best is not None:
            data["best_l1"] = best[1].l1
            data["best_score"] = best[1].scalar
        self.wandb.log(data, step=ti)

    def _best_so_far(self) -> tuple[int, BOObjective, dict[str, str]] | None:
        if not self.completed:
            return None
        items = sorted(self.completed.items())
        objs = [obj for _, (obj, _) in items]
        idx = select_best_trial(objs)
        ti, (obj, ov) = items[idx]
        return ti, obj, ov

    def _reconcile_from_disk(self) -> None:
        """Rebuild completed/abandoned from the ledger, then re-attach every
        still-RUNNING Ax trial and reconcile its cells against disk."""
        for rec in self.ledger.read_all():
            if rec.get("event") != ledger_mod.EVENT_TRIAL_DONE:
                continue
            ti = int(rec["trial_index"])
            if rec.get("outcome") == "abandoned":
                self.abandoned.add(ti)
            elif rec.get("outcome") == "scored":
                obj = BOObjective(
                    l1=rec["l1"], l2=rec["l2"], l3=rec["l3"], scalar=rec["scalar"],
                    n_applicable_benches=len(applicable_benches(self.cfg.method, self.cfg.benches)),
                    n_seeds=len(self.cfg.seeds), n_queries=0,
                    n_diverged_cells=0, n_diverged_seeds=0, diverged=bool(rec.get("diverged")),
                )
                self.completed[ti] = (obj, rec.get("config", {}))

        # ask_count = trials Ax has generated, so the Sobol/BO phase resumes correctly.
        exp = self.ax_client.experiment
        self.ask_count = len(exp.trials)

        # Reattach RUNNING trials (asked, not told) and reconcile their cells.
        for ti, trial in exp.trials.items():
            if ti in self.completed or ti in self.abandoned:
                continue
            status_name = trial.status.name
            if status_name not in ("RUNNING", "CANDIDATE", "STAGED"):
                continue
            params = dict(trial.arm.parameters) if trial.arm is not None else {}
            params, guarded = (params, "ok")
            if self.cfg.method == "snarenet":
                gr = apply_snarenet_guard(params)
                params, guarded = gr.params, gr.action
            if guarded == "reject":
                self.ax_client.abandon_trial(ti, reason="snarenet_validity_reject")
                self.abandoned.add(ti)
                continue
            overrides = self._overrides_for(params)
            cells = self._make_cells(ti, overrides)
            phase = "sobol" if ti < self.cfg.sobol else "bo"
            run = TrialRun(ti=ti, overrides=overrides, phase=phase, cells=cells)
            for key in cells:
                self._submit_or_reuse(run, key)
            self._flush_executor()  # re-register/dispatch this trial's array
            self.in_flight[ti] = run

    def _prelaunch_check_frozen_alm(self, seeds: tuple[int, ...]) -> None:
        """Fail fast (before any Ax ask) if the frozen ALM winner artifacts for
        every applicable (bench, seed) cell are not laid out on disk.
        """
        if self.cfg.method != "alm_bolton":
            return
        root = self.cfg.frozen_alm_dir
        if not root:
            raise SystemExit("alm_bolton requires frozen_alm_dir (see build_config)")
        from scripts.bo.cell_executor import resolve_frozen_run_dir
        missing: list[str] = []
        for bench in applicable_benches(self.cfg.method, self.cfg.benches):
            for seed in seeds:
                rd = resolve_frozen_run_dir(root, bench, seed)
                if not all((rd / f).exists() for f in ("config.json", "model.pt", "final.json")):
                    missing.append(str(rd))
        if missing:
            raise SystemExit(
                "frozen ALM winner artifacts missing/incomplete for "
                f"{len(missing)} cell(s); each needs "
                "config.json+model.pt+final.json:\n  " + "\n  ".join(missing))

    def run(self, resume: bool = False) -> tuple[int, BOObjective, dict[str, str]] | None:
        self._prelaunch_check_frozen_alm(self.cfg.seeds)
        self._init_ax(resume=resume)
        if resume:
            self._reconcile_from_disk()

        asks_cap = self.cfg.budget * 2  # generous slack for infra-abandoned replacements
        try:
            while len(self.completed) < self.cfg.budget:
                phase = self._phase()
                cap = self.cfg.sobol if phase == "sobol" else self.cfg.max_in_flight
                asked_this_round = False
                while (len(self.in_flight) < cap and self.ask_count < asks_cap
                       and len(self.completed) + len(self.in_flight) < self.cfg.budget):
                    if not self._ask_one():
                        break
                    asked_this_round = True

                if not self.in_flight:
                    if self.ask_count >= asks_cap:
                        sys.stderr.write(
                            f"[controller] ask cap {asks_cap} reached with "
                            f"{len(self.completed)}/{self.cfg.budget} scored; stopping.\n")
                        break
                    if not asked_this_round:
                        break  # nothing to do and nothing asked -> done/stuck

                progressed = False
                for run in list(self.in_flight.values()):
                    self._harvest(run)
                for run in list(self.in_flight.values()):
                    if run.done_collecting:
                        self._finalize(run)
                        progressed = True
                if not progressed and self.in_flight:
                    time.sleep(0.2)
        finally:
            if self._own_executor:
                self.executor.shutdown(wait=False)
            self.wandb.finish()
        self._write_status()
        return self._best_so_far()

    def _write_status(self) -> None:
        try:
            write_campaign_status(self.cfg.campaign_root)
        except Exception:  # noqa: BLE001 - observability must never crash the run
            pass


def _read_method_ledger(method_root: Path) -> list[dict[str, Any]]:
    lp = method_root / "ledger.jsonl"
    if not lp.exists():
        return []
    out = []
    with lp.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _method_summary(method_root: Path) -> dict[str, Any] | None:
    recs = _read_method_ledger(method_root)
    if not recs:
        return None
    start = next((r for r in recs if r.get("event") == ledger_mod.EVENT_CAMPAIGN_START), None)
    budget = start.get("budget") if start else None
    scored: list[dict[str, Any]] = []
    abandoned = 0
    for r in recs:
        if r.get("event") != ledger_mod.EVENT_TRIAL_DONE:
            continue
        if r.get("outcome") == "scored":
            scored.append(r)
        elif r.get("outcome") == "abandoned":
            abandoned += 1
    method = recs[0].get("method")
    if not scored:
        return {"method": method, "done": 0, "budget": budget, "abandoned": abandoned,
                "best": None, "top3": []}
    ranked = sorted(scored, key=lambda r: (-r["l1"], r["l2"], r["l3"]))
    best = ranked[0]
    top3 = ranked[:3]
    return {"method": method, "done": len(scored), "budget": budget, "abandoned": abandoned,
            "best": best, "top3": top3}


def write_campaign_status(campaign_root: str | Path) -> Path:
    campaign_root = Path(campaign_root)
    campaign_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for method_root in sorted(p for p in campaign_root.iterdir() if p.is_dir()):
        summ = _method_summary(method_root)
        if summ is not None:
            rows.append(summ)
    lines = ["# BO tuning status", "",
             f"_generated {ledger_mod._now()}_", "",
             "| method | trials | abandoned | best (L1, L2, L3) | best S |",
             "| --- | --- | --- | --- | --- |"]
    for s in rows:
        b = s["best"]
        best_tuple = f"({b['l1']:.4f}, {b['l2']:.4f}, {b['l3']:.4f})" if b else "-"
        best_s = f"{b['scalar']:.5f}" if b else "-"
        budget = s["budget"] if s["budget"] is not None else "?"
        lines.append(f"| {s['method']} | {s['done']}/{budget} | {s['abandoned']} | {best_tuple} | {best_s} |")
    lines.append("")
    for s in rows:
        if not s["top3"]:
            continue
        lines.append(f"## {s['method']}: top 3 configs")
        for i, t in enumerate(s["top3"], 1):
            lines.append(f"{i}. tuple=({t['l1']:.4f}, {t['l2']:.4f}, {t['l3']:.4f}) "
                         f"S={t['scalar']:.5f} config=`{json.dumps(t.get('config', {}), sort_keys=True)}`")
        lines.append("")
    out = campaign_root / "status.md"
    out.write_text("\n".join(lines))
    return out


def run_confirm(cfg: CampaignConfig, *, executor: Executor | None = None) -> dict[str, Any]:
    """Re-evaluate the top-3 tuple-ranked configs on fresh seeds 103-107 and
    pick the final winner by the tuple over the widened seed set 100-107.

    Seeds 0-9 (the reported eval seeds) are not touched here or in tuning.
    """
    assert min(cfg.seeds) >= 100 and min(CONFIRM_SEEDS) >= 100, "report seeds 0-9 must be untouched"
    if cfg.method == "alm_bolton":
        # Predict-only confirm also needs frozen ALM checkpoints for the confirm seeds.
        if not cfg.frozen_alm_dir:
            raise SystemExit("alm_bolton confirm requires --frozen-alm-dir")
        from scripts.bo.cell_executor import resolve_frozen_run_dir
        missing = [
            str(resolve_frozen_run_dir(cfg.frozen_alm_dir, b, s))
            for b in applicable_benches(cfg.method, cfg.benches)
            for s in CONFIRM_SEEDS
            if not all((resolve_frozen_run_dir(cfg.frozen_alm_dir, b, s) / f).exists()
                       for f in ("config.json", "model.pt", "final.json"))
        ]
        if missing:
            raise SystemExit(
                "alm_bolton confirm: frozen ALM checkpoints for confirm seeds "
                f"{list(CONFIRM_SEEDS)} missing/incomplete:\n  " + "\n  ".join(missing))

    recs = _read_method_ledger(cfg.method_root)
    scored = [r for r in recs if r.get("event") == ledger_mod.EVENT_TRIAL_DONE
              and r.get("outcome") == "scored"]
    if not scored:
        raise SystemExit(f"no scored trials in {cfg.ledger_path}; run the campaign first")
    ranked = sorted(scored, key=lambda r: (-r["l1"], r["l2"], r["l3"]))
    top3 = ranked[:3]

    own = executor is None
    ex = executor or LocalPoolExecutor(max_workers=cfg.pool_size)
    led = ledger_mod.Ledger(cfg.ledger_path, method=cfg.method,
                            campaign_root=cfg.campaign_root, git_sha=_git_sha(), toy=cfg.toy)
    widened_seeds = tuple(sorted(set(cfg.seeds) | set(CONFIRM_SEEDS)))
    results_per_cfg: list[tuple[dict, BOObjective]] = []
    try:
        # Submit all ranks' missing cells, then flush once so the ranks run in parallel.
        ranks: list[dict[str, Any]] = []
        for rank, trec in enumerate(top3):
            ti = int(trec["trial_index"])
            overrides = trec.get("config", {})
            led.append(ledger_mod.EVENT_CONFIRM, phase="start", rank=rank, trial_index=ti,
                       config=overrides, confirm_seeds=list(CONFIRM_SEEDS))
            futures: dict[tuple[str, int], Future] = {}
            for bench in applicable_benches(cfg.method, cfg.benches):
                for seed in CONFIRM_SEEDS:
                    td = cfg.trial_dir(ti, bench, seed, subtree="confirm")
                    cell = CellSpec(method=cfg.method, bench=bench, seed=seed,
                                    trial_dir=str(td), set_overrides=dict(overrides),
                                    timeout_s=cfg.timeout_s,
                                    frozen_alm_dir=cfg.frozen_alm_dir)
                    existing = read_result(cell.result_path)
                    if existing and existing.get("status") in ("ok", "diverged"):
                        continue
                    futures[(bench, seed)] = ex.submit_cell(cell)
            ranks.append({"rank": rank, "trec": trec, "ti": ti,
                          "overrides": overrides, "futures": futures})
        _flush = getattr(ex, "flush", None)
        if callable(_flush):
            _flush()  # dispatch all ranks' confirm arrays at once (no-op: local pool)

        for r in ranks:
            for fut in r["futures"].values():
                fut.result()

        # Score ranks in tuple-rank order.
        for r in ranks:
            rank, trec, ti, overrides = r["rank"], r["trec"], r["ti"], r["overrides"]
            # Gather widened result set: tuning cells (100-102) + confirm (103-107).
            gathered: dict[tuple[str, int], dict] = {}
            for bench in applicable_benches(cfg.method, cfg.benches):
                for seed in cfg.seeds:
                    res = read_result(cfg.trial_dir(ti, bench, seed).joinpath("result.json"))
                    if res is not None:
                        gathered[(bench, seed)] = res
                for seed in CONFIRM_SEEDS:
                    res = read_result(cfg.trial_dir(ti, bench, seed, subtree="confirm").joinpath("result.json"))
                    if res is not None:
                        gathered[(bench, seed)] = res
            ts = score_trial(cfg.method, overrides, widened_seeds, gathered, benches=cfg.benches)
            results_per_cfg.append((trec, ts.objective))
            led.append(ledger_mod.EVENT_CONFIRM, phase="scored", rank=rank, trial_index=ti,
                       l1=ts.objective.l1, l2=ts.objective.l2, l3=ts.objective.l3,
                       scalar=ts.objective.scalar, seeds=list(widened_seeds), config=overrides)
    finally:
        if own:
            ex.shutdown(wait=False)

    winner_idx = select_best_trial([o for _, o in results_per_cfg])
    win_trec, win_obj = results_per_cfg[winner_idx]
    winner = {
        "method": cfg.method,
        "winner_trial_index": int(win_trec["trial_index"]),
        "config": win_trec.get("config", {}),
        "widened_seeds": list(widened_seeds),
        "tuple": [win_obj.l1, win_obj.l2, win_obj.l3],
        "scalar": win_obj.scalar,
    }
    led.append(ledger_mod.EVENT_CONFIRM, phase="winner", **winner)
    (cfg.method_root / "confirm_winner.json").write_text(json.dumps(winner, indent=2, sort_keys=True))
    return winner


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--method", required=True, choices=sorted(METHOD_SPACES))
    p.add_argument("--campaign-root", required=True)
    p.add_argument("--trials", type=int, default=None)
    p.add_argument("--sobol", type=int, default=None)
    p.add_argument("--max-in-flight", type=int, default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--ax-seed", type=int, default=None)
    p.add_argument("--timeout-s", type=float, default=None)
    p.add_argument("--pool-size", type=int, default=None)
    p.add_argument("--toy", action="store_true")
    p.add_argument(
        "--frozen-alm-dir", default=None,
        help="alm_bolton second-stage predict-only (required for that method): "
             "root holding the frozen ALM winner run dirs, one per cell at "
             "<root>/<bench>_seed<seed>/{config.json,model.pt,final.json}. Each "
             "trial loads these checkpoints and predicts with candidate "
             "projector knobs (proj_delta, proj_max_iters); no training.")
    p.add_argument("--executor", choices=("local", "slurm"), default="local",
                   help="cell executor: local process pool or an x86-cluster Slurm array lane")
    p.add_argument("--slurm-resources", default=None,
                   help="Slurm resources as JSON ('{\"preset\":\"cpu_lane\",...}') "
                        "or comma-separated k=v (preset=cpu_lane,time=04:00:00,"
                        "python_bin=/path/to/python); passed to slurm_lane.")
    p.add_argument("--poll-interval", type=float, default=30.0,
                   help="Slurm poll interval in seconds (executor=slurm only)")
    p.add_argument("--wandb-project", default="pal-bo")
    p.add_argument("--wandb-entity", default=WANDB_ENTITY)
    p.add_argument("--wandb-online", action=argparse.BooleanOptionalAction,
                   default=True, help="live W&B (default on); --no-wandb-online for offline")


def _parse_slurm_resources(spec: str | None) -> str | dict[str, Any] | None:
    """Parse --slurm-resources: JSON object, or comma-separated k=v pairs.

    Supports the slurm_lane resources dict (preset/time/python_bin/cpus_per_task/
    mem_per_cpu_mb/...). Returns None when unset.
    """
    if not spec:
        return None
    spec = spec.strip()
    if spec.startswith("{"):
        return json.loads(spec)
    out: dict[str, Any] = {}
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        key, sep, val = tok.partition("=")
        if not sep:
            raise SystemExit(f"--slurm-resources token must be KEY=VALUE: {tok!r}")
        out[key.strip()] = val.strip()
    for key in ("cpus_per_task", "mem_per_cpu_mb"):
        if key in out:
            out[key] = int(out[key])
    return out


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    for name in ("run", "resume", "confirm"):
        sp = sub.add_parser(name)
        _add_common(sp)
    st = sub.add_parser("status")
    st.add_argument("--campaign-root", required=True)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.command == "status":
        out = write_campaign_status(args.campaign_root)
        sys.stdout.write(out.read_text())
        return 0

    cfg = build_config(args)
    if args.command in ("run", "resume"):
        ctrl = Controller(
            cfg,
            executor_kind=args.executor,
            slurm_resources=_parse_slurm_resources(args.slurm_resources),
            poll_interval=args.poll_interval,
            wandb_project=args.wandb_project,
            wandb_online=args.wandb_online,
            wandb_entity=args.wandb_entity,
        )
        best = ctrl.run(resume=(args.command == "resume"))
        if best is None:
            sys.stdout.write(json.dumps({"status": "no_completed_trials"}) + "\n")
            return 1
        ti, obj, ov = best
        sys.stdout.write(json.dumps({
            "winner_trial_index": ti, "tuple": [obj.l1, obj.l2, obj.l3],
            "scalar": obj.scalar, "config": ov,
        }, sort_keys=True) + "\n")
        return 0
    if args.command == "confirm":
        # Build the Slurm lane for confirm exactly as Controller.__init__ does.
        ex = None
        if args.executor == "slurm":
            led = ledger_mod.Ledger(
                cfg.ledger_path, method=cfg.method, campaign_root=cfg.campaign_root,
                git_sha=_git_sha(), toy=cfg.toy,
            )
            ex = SlurmLaneExecutor(
                campaign_root=cfg.campaign_root, sha=_git_sha(),
                resources=_parse_slurm_resources(args.slurm_resources),
                max_attempts=cfg.max_attempts,
                poll_interval=args.poll_interval, ledger=led,
            )
        try:
            winner = run_confirm(cfg, executor=ex)
        finally:
            # run_confirm only shuts down executors it owns; we own this one.
            if ex is not None:
                ex.shutdown(wait=False)
        sys.stdout.write(json.dumps(winner, sort_keys=True) + "\n")
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
