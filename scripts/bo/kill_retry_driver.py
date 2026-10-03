#!/usr/bin/env python3
"""Slurm lane kill/retry drill: kill one array task, poll to terminal, retry it."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.expandvars("$SCRATCH/pal-harness/scripts/bo"))
import slurm_lane
from smoke_driver import SHA, VENV_PY, make_cells, poll_to_terminal

sys.path.insert(0, str(Path(__file__).parent))

KILL_INDEX = 1

root = Path(os.path.expandvars("$SCRATCH/bo-smoke-kill")).resolve()
root.mkdir(parents=True, exist_ok=True)
jobid = slurm_lane.submit_cells(
    make_cells(root),
    campaign_root=root,
    sha=SHA,
    resources={"preset": "cpu_lane", "time": "00:15:00", "python_bin": VENV_PY},
)
manifest_path = str(slurm_lane.manifest_for_job(jobid))
print(f"SUBMITTED jobid={jobid} manifest={manifest_path}", flush=True)

# Give Slurm a moment, then kill index KILL_INDEX.
time.sleep(5)
states = slurm_lane.poll_states(jobid)
print(f"pre-kill states={states}", flush=True)
target = f"{jobid}_{KILL_INDEX}"
subprocess.run(["scancel", target], check=True)
print(f"SCANCELLED {target}", flush=True)

final_states = poll_to_terminal(jobid)
print(f"POST-KILL poll_states: {final_states}", flush=True)
expect = {i: ("infra_failure" if i == KILL_INDEX else "ok") for i in range(4)}
print(f"kill-phase check: {'PASS' if final_states == expect else 'FAIL, expected ' + str(expect)}", flush=True)

retry_jobid = slurm_lane.retry_failed(manifest_path, jobid, max_attempts=2)
print(f"RETRY jobid={retry_jobid} manifest={slurm_lane.manifest_for_job(retry_jobid)}", flush=True)
retry_manifest = json.loads(Path(slurm_lane.manifest_for_job(retry_jobid)).read_text())
cells = retry_manifest["cells"]
print(f"retry array size={len(cells)} attempt={cells[0]['attempt']} cell={cells[0]['method']} seed{cells[0]['seed']}", flush=True)

retry_states = poll_to_terminal(retry_jobid)
print(f"retry final states={retry_states}", flush=True)

print(f"original job poll after retry: {slurm_lane.poll_states(jobid)}", flush=True)
rp = Path(cells[0]["trial_dir"]) / "result.json"
r = json.loads(rp.read_text())
print(f"retried result: status={r['status']} attempt={r['fingerprint']['attempt']} sha_ok={r['fingerprint']['git_sha']==SHA}", flush=True)
print("DRILL DONE", flush=True)
