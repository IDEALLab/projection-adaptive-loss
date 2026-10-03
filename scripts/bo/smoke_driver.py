#!/usr/bin/env python3
"""Slurm lane smoke driver: submit 4 cells, poll to terminal, verify results.

Usage: python smoke_driver.py <campaign_root>
Prints jobid, then poll loop, then verification summary. Exits nonzero on gate fail.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expandvars("$SCRATCH/pal-harness/scripts/bo"))
import slurm_lane

SHA = "23881c2ec51974abdc1f5d62f6a1c74cf0a09f20"
VENV_PY = os.path.expandvars("$SCRATCH/pal-venv/bin/python")


def make_cells(root: Path):
    cells = []
    for method in ("alm", "pal_loggap"):
        for seed in (100, 101):
            cells.append({
                "method": method,
                "bench": "s1_sphere_track",
                "seed": seed,
                "set_overrides": {"epochs": "5"},
                "timeout_s": 600,
                "trial_dir": str(root / method / f"s1_sphere_track_seed{seed}"),
            })
    return cells


def submit(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    jobid = slurm_lane.submit_cells(
        make_cells(root),
        campaign_root=root,
        sha=SHA,
        resources={
            "preset": "cpu_lane",
            "time": "00:15:00",
            "python_bin": VENV_PY,
        },
    )
    print(f"SUBMITTED jobid={jobid} manifest={slurm_lane.manifest_for_job(jobid)}", flush=True)
    return jobid


def poll_to_terminal(jobid: str, interval: int = 20, max_wait: int = 3600):
    active = {"pending", "running", "configuring", "completing", "suspended"}
    start = time.time()
    while True:
        states = slurm_lane.poll_states(jobid)
        print(f"t+{time.time()-start:6.0f}s states={states}", flush=True)
        if not any(v in active for v in states.values()):
            return states
        if time.time() - start > max_wait:
            print("POLL TIMEOUT", flush=True)
            return states
        time.sleep(interval)


def verify(root: Path):
    ok = True
    for method in ("alm", "pal_loggap"):
        for seed in (100, 101):
            rp = root / method / f"s1_sphere_track_seed{seed}" / "result.json"
            if not rp.exists():
                print(f"MISSING {rp}", flush=True)
                ok = False
                continue
            r = json.loads(rp.read_text())
            fp = r.get("fingerprint", {})
            checks = {
                "status_ok": r.get("status") == "ok",
                "sha_match": fp.get("git_sha") == SHA,
                "pal_under_scratch": str(fp.get("pal_file", "")).startswith(os.path.expandvars("$SCRATCH/")),
            }
            print(f"{method} seed{seed}: {checks} wall={fp.get('wall_time_s'):.1f}s", flush=True)
            if not all(checks.values()):
                ok = False
                print(f"  detail: status={r.get('status')} reason={r.get('reason')} pal_file={fp.get('pal_file')}", flush=True)
    return ok


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    jobid = submit(root)
    states = poll_to_terminal(jobid)
    good = verify(root)
    print(f"GATE {'PASS' if good else 'FAIL'} jobid={jobid} final_states={states}", flush=True)
    sys.exit(0 if good else 1)
