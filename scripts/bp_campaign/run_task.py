#!/usr/bin/env python3
"""Per-array-task entrypoint for the breaking-point campaign.

One Slurm array task = one (arm, variant, seed) training run. Skips if a run dir
with `final.json` exists, verifies the winner hash and argv against the manifest,
runs `scripts/bench_run.py`, and writes `winner_provenance.json` into the run dir.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bo.cell_executor import build_command  # noqa: E402
from scripts.bp_campaign.driver import (  # noqa: E402
    DEVICE,
    EPOCHS,
    N_EVAL,
    DriverError,
    pal_run_dirs,
    sha256_file,
    write_provenance,
)


def find_task(manifest: dict, task_id: str) -> dict:
    """Return the task spec with this id.

    Args:
        manifest: Loaded campaign manifest.
        task_id: `<arm>/<variant>/seed<seed>`.

    Returns:
        The task spec dict.

    Raises:
        DriverError: If no task carries that id.
    """
    for task in manifest["tasks"]:
        if task["task_id"] == task_id:
            return task
    raise DriverError(f"task id {task_id!r} not in manifest")


def verify_command(task: dict) -> list[str]:
    """Rebuild the training argv and check it against the manifest.

    Args:
        task: Task spec from the manifest.

    Returns:
        The argv to execute (with this interpreter as argv[0]).

    Raises:
        DriverError: If the rebuilt argv differs from the recorded one, or if
            the task has no `precision` field.
    """
    if "precision" not in task:
        raise DriverError(
            f"task {task['task_id']!r} has no 'precision' field; "
            "regenerate the manifest with `driver.py generate`."
        )
    cmd = build_command(
        python_bin=sys.executable,
        method=task["method"],
        bench=task["bench_id"],
        seed=int(task["seed"]),
        trial_dir=Path(task["runs_root"]),
        set_overrides=dict(task["set_overrides"]),
        extra_args=["--epochs", str(EPOCHS), "--n-eval", str(N_EVAL),
                    "--device", DEVICE, "--precision", str(task["precision"])],
    )
    recorded = list(task["command"])
    if cmd[1:] != recorded[1:]:
        raise DriverError(
            "manifest command drifted from the code that would run now:\n"
            f"  manifest: {recorded[1:]}\n  rebuilt : {cmd[1:]}\n"
            "re-run `driver.py generate`."
        )
    return cmd


def verify_winner(task: dict) -> str:
    """Re-hash the winner file at run time and check it against the manifest.

    Args:
        task: Task spec from the manifest.

    Returns:
        The freshly computed sha256.

    Raises:
        DriverError: On a missing winner file or a hash mismatch.
    """
    path = REPO / task["winner_path"]
    if not path.exists():
        raise DriverError(f"winner file missing at run time: {path}")
    digest = sha256_file(path)
    if digest != task["winner_sha256"]:
        raise DriverError(
            f"winner hash changed since generate: {task['winner_path']} "
            f"is {digest}, manifest says {task['winner_sha256']}"
        )
    return digest


def main(argv: list[str] | None = None) -> int:
    """Run one campaign task.

    Args:
        argv: Argument vector (defaults to `sys.argv[1:]`).

    Returns:
        Process exit code (the training subprocess's rc, or 0 on skip).
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", required=True, help="path to campaign manifest.json")
    ap.add_argument("--task-id", required=True, help="<arm>/<variant>/seed<seed>")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the command and exit without training")
    args = ap.parse_args(argv)

    try:
        manifest = json.loads(Path(args.manifest).read_text())
        task = find_task(manifest, args.task_id)
        runs_root = Path(task["runs_root"])

        for existing in pal_run_dirs(runs_root):
            if (existing / "final.json").exists():
                # `final.json` marks completion; the verdict is in `status.json`.
                status = "?"
                try:
                    status = json.loads(
                        (existing / "status.json").read_text()).get("status", "?")
                except Exception:
                    pass
                print(f"[task] SKIP {args.task_id}: {existing.name}/final.json "
                      f"already present (status={status}); relaunch is idempotent.")
                return 0

        digest = verify_winner(task)
        cmd = verify_command(task)
        if args.dry_run:
            print(" ".join(cmd))
            return 0

        runs_root.mkdir(parents=True, exist_ok=True)
        before = {d.name for d in pal_run_dirs(runs_root)}
        if before:
            print(f"[task] WARNING: {len(before)} incomplete run dir(s) already in "
                  f"{runs_root}: {sorted(before)} (leftover from a crashed task; "
                  f"the analyzer treats duplicates as a hard error, clean up "
                  f"before analysis).")
        print(f"[task] {args.task_id}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode

        created = [d for d in pal_run_dirs(runs_root) if d.name not in before]
        if len(created) != 1:
            print(f"[task] ERROR: expected exactly 1 new run dir under {runs_root}, "
                  f"found {[d.name for d in created]}", file=sys.stderr)
            return rc or 1
        run_dir = created[0]
        sidecar = write_provenance(
            run_dir,
            winner_path=task["winner_path"],
            winner_sha256=digest,
            winner_trial_index=int(task["winner_trial_index"]),
            arm_label=task["arm_label"],
            variant=task["variant"],
            seed=int(task["seed"]),
        )
        print(f"[task] run dir: {run_dir}")
        print(f"[task] provenance: {sidecar}")
        return rc
    except DriverError as exc:
        print(f"[task] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
