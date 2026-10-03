#!/usr/bin/env python3
"""Breaking-point campaign driver: 640 cpu training runs as Slurm job arrays.

Subcommands: generate, submit, smoke, status, bolton-eval, flatten.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.bo.cell_executor import build_command  # noqa: E402

#: `submit` refuses unless HEAD == origin/<BRANCH> and the tree is clean.
BRANCH = os.environ.get("PAL_CAMPAIGN_BRANCH", "main")

DEFAULT_CAMPAIGN_ROOT = Path(os.environ.get("SCRATCH", "/tmp")) / "bp-campaign-fp64"
WINNERS_DIR = REPO / "results" / "2026-07-26_bo_tuned_table" / "winners"
WINNERS_MANIFEST = WINNERS_DIR / "MANIFEST.sha256"
#: Campaign-specific fixed configurations (take precedence over `WINNERS_DIR`).
CONFIGS_DIR = REPO / "results" / "2026-09-23_bp_campaign_fp64" / "configs"
CONFIGS_MANIFEST = CONFIGS_DIR / "MANIFEST.sha256"
CAMPAIGN_CONFIGS: frozenset[str] = frozenset({"enforce_v4"})
RUN_TASK = REPO / "scripts" / "bp_campaign" / "run_task.py"

#: kappa grid {0,1,10,1e2,1e3,1e4,1e5,1e6}.
VARIANTS: tuple[str, ...] = ("k0", "k4", "k6", "k8", "k10", "k11", "k12", "k13")
SEEDS: tuple[int, ...] = tuple(range(10))
BENCH_PREFIX = "curvature_warp_"

#: snarenet and enforce_v4 schedules are tuned against epochs=2000.
EPOCHS = 2000
N_EVAL = 512
DEVICE = "cpu"

#: Training dtype. Stored per task in manifest.json and re-read by run_task.py.
PRECISION_CHOICES: tuple[str, ...] = ("fp32", "bf16", "fp16", "fp64")
DEFAULT_PRECISION = "fp64"

#: Slurm resources per task (cpu-only campaign).
CPUS_PER_TASK = 4
MEM_PER_CPU = "4G"
#: Per-arm core override (`generate --cpus ARM=N`).
CPUS_OVERRIDE: dict[str, int] = {}


def cpus_for(arm_label: str) -> int:
    """Cores (and BENCH_THREADS) reserved for one arm's tasks."""
    return CPUS_OVERRIDE.get(arm_label, CPUS_PER_TASK)

WALLTIMES: dict[str, str] = {
    "pal_loggap_tau1e-4": "04:00:00",
    "pal_loggap_tau1e-2": "04:00:00",
    "pal_loggap_tau1": "04:00:00",
    "pal_loggap_tau1e2": "04:00:00",
    "pal_loggap_tau1e4": "04:00:00",
    "alm": "04:00:00",
    "dc3": "04:00:00",
    "dc3_newton": "04:00:00",
    "dc3_newton_yf": "04:00:00",
    "dc3_newton_lm": "04:00:00",
    "enforce_v4": "04:00:00",
    "fsnet": "24:00:00",
    "snarenet": "04:00:00",
}


@dataclass(frozen=True)
class Arm:
    """One training arm of the campaign.

    Attributes:
        label: Directory/job-name label, e.g. `pal_loggap_tau1e-2`.
        method: pal method string passed to `--method`.
        winner: Winner name resolved by `load_winner` (`CONFIGS_DIR` or `WINNERS_DIR`).
        trial_index: Expected `winner_trial_index`.
        extra_set: Ablation-only `--set` overrides layered ON TOP of the frozen
            winner config (the three PAL tau rows; nothing else).
    """

    label: str
    method: str
    winner: str
    trial_index: int
    extra_set: dict[str, str]


#: Training arms. alm's second table row comes from the `bolton-eval` pass.
ARMS: tuple[Arm, ...] = (
    # Explicit tau equals the spec default but makes config.json record hparams.tau.
    Arm("pal_loggap_tau1e-4", "pal_loggap", "pal_loggap", 15, {"tau": "1e-4"}),
    Arm("pal_loggap_tau1e-2", "pal_loggap", "pal_loggap", 15, {"tau": "1e-2"}),
    Arm("pal_loggap_tau1", "pal_loggap", "pal_loggap", 15, {"tau": "1.0"}),
    # Loose log-gap targets (own campaign root).
    Arm("pal_loggap_tau1e2", "pal_loggap", "pal_loggap", 15, {"tau": "1e2"}),
    Arm("pal_loggap_tau1e4", "pal_loggap", "pal_loggap", 15, {"tau": "1e4"}),
    Arm("alm", "alm", "alm", 17, {}),
    Arm("dc3", "dc3", "dc3", 3, {}),
    # Diagnostic arms: force the nonlinear Newton completion on curvature_warp.
    Arm("dc3_newton", "dc3", "dc3", 3,
        {"completion_strategy_override": "generic_newton"}),
    # Levenberg-Marquardt (J^T J + mu I) with Yamashita-Fukushima mu = reg + c||h||^2.
    Arm("dc3_newton_yf", "dc3", "dc3", 3,
        {"completion_strategy_override": "generic_newton", "newton_yf_damping": "1.0",
         "newton_reg": "0.1", "newton_max_iter": "100"}),
    # Constant LM damping, no ||h||^2 term.
    Arm("dc3_newton_lm", "dc3", "dc3", 3,
        {"completion_strategy_override": "generic_newton", "newton_lm_damping": "0.1",
         "newton_reg": "1e-8", "newton_max_iter": "100"}),
    Arm("enforce_v4", "enforce_v4", "enforce_v4", 20, {}),
    Arm("fsnet", "fsnet", "fsnet", 5, {}),
    Arm("snarenet", "snarenet", "snarenet", 22, {}),
)
ARMS_BY_LABEL: dict[str, Arm] = {a.label: a for a in ARMS}

#: Predict-only bolton pass over alm checkpoints (values bit-exact from the winner file).
BOLTON_WINNER = "alm_bolton"
BOLTON_TRIAL_INDEX = 10
BOLTON_EXPECTED: dict[str, float] = {"proj_delta": 1e-5, "proj_max_iters": 20}


class DriverError(RuntimeError):
    """Raised for campaign-level usage / integrity errors."""


def sha256_file(path: Path) -> str:
    """Return the hex sha256 of a file's bytes.

    Args:
        path: File to hash.

    Returns:
        Lower-case hex digest.
    """
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_winners_manifest(manifest_path: Path = WINNERS_MANIFEST) -> dict[str, str]:
    """Parse a `MANIFEST.sha256` into {relative path: sha256}.

    Args:
        manifest_path: Manifest file, `winners/MANIFEST.sha256` by default.

    Returns:
        Mapping like `{"alm/confirm_winner.json": "<sha256>"}`.
    """
    entries: dict[str, str] = {}
    for line in manifest_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, rel = line.partition("  ")
        entries[rel.strip()] = digest.strip()
    return entries


def load_winner(winner: str, expected_trial_index: int) -> dict:
    """Load a frozen winner file and verify its hash and trial index.

    Names in `CAMPAIGN_CONFIGS` load `CONFIGS_DIR/<winner>.json` (checked against
    `CONFIGS_MANIFEST`), all others `WINNERS_DIR/<winner>/confirm_winner.json`.

    Args:
        winner: Winner name (method-ish), e.g. `snarenet`.
        expected_trial_index: Trial id the campaign design pins for this arm.

    Returns:
        Dict with keys `path` (Path), `rel_path` (repo-relative str), `sha256`,
        `trial_index`, `config` (the bit-exact string-valued config dict).

    Raises:
        DriverError: On a missing/misindexed winner or a sha256 mismatch
            against `MANIFEST.sha256`.
    """
    if winner in CAMPAIGN_CONFIGS:
        rel_in_manifest = f"{winner}.json"
        path, manifest_path = CONFIGS_DIR / rel_in_manifest, CONFIGS_MANIFEST
    else:
        rel_in_manifest = f"{winner}/confirm_winner.json"
        path, manifest_path = WINNERS_DIR / rel_in_manifest, WINNERS_MANIFEST
    if not path.exists():
        raise DriverError(f"missing winner file: {path}")
    manifest = read_winners_manifest(manifest_path)
    if rel_in_manifest not in manifest:
        raise DriverError(f"{rel_in_manifest} absent from {manifest_path}")
    digest = sha256_file(path)
    if digest != manifest[rel_in_manifest]:
        raise DriverError(
            f"winner hash mismatch for {rel_in_manifest}: file={digest} "
            f"manifest={manifest[rel_in_manifest]}, the frozen winner changed, "
            f"refusing to build the campaign"
        )
    data = json.loads(path.read_text())
    trial = int(data["winner_trial_index"])
    if trial != expected_trial_index:
        raise DriverError(
            f"{rel_in_manifest}: winner_trial_index={trial}, campaign design "
            f"pins t{expected_trial_index} (doc sec. 1 arm table)"
        )
    config = data["config"]
    if not isinstance(config, dict):
        raise DriverError(f"{rel_in_manifest}: 'config' is not an object")
    return {
        "path": path,
        "rel_path": str(path.relative_to(REPO)),
        "sha256": digest,
        "trial_index": trial,
        # Values stay strings; the runner casts them via the dataclass field types.
        "config": {str(k): str(v) for k, v in config.items()},
    }


def task_id(arm_label: str, variant: str, seed: int) -> str:
    """Return the stable task identifier `<arm>/<variant>/seed<seed>`."""
    return f"{arm_label}/{variant}/seed{seed}"


def build_task(
    arm: Arm,
    winner: dict,
    variant: str,
    seed: int,
    campaign_root: Path,
    precision: str = DEFAULT_PRECISION,
) -> dict:
    """Build one task spec (including the exact training argv).

    Args:
        arm: Campaign arm.
        winner: Result of `load_winner` for this arm.
        variant: curvature_warp variant suffix, e.g. `k10`.
        seed: Training seed.
        campaign_root: Campaign root directory.
        precision: Training dtype passed as `--precision <value>`; recorded in the task.

    Returns:
        JSON-serializable task spec dict.
    """
    bench_id = f"{BENCH_PREFIX}{variant}"
    runs_root = campaign_root / "runs" / arm.label / f"{variant}_seed{seed}"
    # Winner keys first (bit-exact), tau ablation layered on top.
    set_overrides = dict(winner["config"])
    set_overrides.update(arm.extra_set)
    command = build_command(
        python_bin="python",
        method=arm.method,
        bench=bench_id,
        seed=seed,
        trial_dir=runs_root,
        set_overrides=set_overrides,
        extra_args=["--epochs", str(EPOCHS), "--n-eval", str(N_EVAL),
                    "--device", DEVICE, "--precision", precision],
    )
    return {
        "task_id": task_id(arm.label, variant, seed),
        "arm_label": arm.label,
        "method": arm.method,
        "variant": variant,
        "bench_id": bench_id,
        "seed": seed,
        "runs_root": str(runs_root),
        "winner_path": winner["rel_path"],
        "winner_sha256": winner["sha256"],
        "winner_trial_index": winner["trial_index"],
        "set_overrides": set_overrides,
        "precision": precision,
        "command": command,
    }


def build_manifest(
    campaign_root: Path,
    arms: tuple[Arm, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    precision: str = DEFAULT_PRECISION,
) -> dict:
    """Build the full campaign manifest.

    Args:
        campaign_root: Campaign root directory.
        arms: Arms to include.
        variants: Variant suffixes to include.
        seeds: Seeds to include.
        precision: Training dtype for every task.

    Returns:
        Manifest dict with `arms` metadata and the flat `tasks` list.
    """
    winners = {a.label: load_winner(a.winner, a.trial_index) for a in arms}
    tasks = [
        build_task(arm, winners[arm.label], variant, seed, campaign_root, precision)
        for arm in arms
        for variant in variants
        for seed in seeds
    ]
    return {
        "campaign": "breaking_point_curvature_warp",
        "doc": "results/2026-09-23_bp_campaign_fp64/README.md",
        "generated_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "repo": str(REPO),
        "branch": BRANCH,
        "git_sha": git_sha(),
        "campaign_root": str(campaign_root),
        "protocol": {"epochs": EPOCHS, "n_eval": N_EVAL, "device": DEVICE,
                     "precision": precision,
                     "train_dataset_size_override": None},
        "variants": list(variants),
        "seeds": list(seeds),
        "arms": {
            a.label: {
                "method": a.method,
                "winner_path": winners[a.label]["rel_path"],
                "winner_sha256": winners[a.label]["sha256"],
                "winner_trial_index": winners[a.label]["trial_index"],
                "extra_set": a.extra_set,
                "walltime": WALLTIMES[a.label],
            }
            for a in arms
        },
        "tasks": tasks,
    }


def array_spec(seeds: tuple[int, ...]) -> str:
    """Return the `--array` spec for a seed list (array index == seed)."""
    if list(seeds) == list(range(min(seeds), max(seeds) + 1)):
        return f"{min(seeds)}-{max(seeds)}" if len(seeds) > 1 else str(seeds[0])
    return ",".join(str(s) for s in seeds)


def sbatch_text(
    arm: Arm,
    variant: str,
    seeds: tuple[int, ...],
    campaign_root: Path,
    precision: str = DEFAULT_PRECISION,
) -> str:
    """Render the sbatch script for one (arm x variant) array.

    Args:
        arm: Campaign arm.
        variant: curvature_warp variant suffix.
        seeds: Seeds; the array index IS the seed.
        campaign_root: Campaign root directory.
        precision: Training dtype, echoed in the header comment only.

    Returns:
        Full sbatch script text.
    """
    job_name = f"bp_{arm.label}_{variant}"
    logs = campaign_root / "logs"
    manifest = campaign_root / "manifest.json"
    cpus = cpus_for(arm.label)
    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array={array_spec(seeds)}
#SBATCH -c {cpus}
#SBATCH --mem-per-cpu={MEM_PER_CPU}
#SBATCH --time={WALLTIMES[arm.label]}
#SBATCH -o {logs}/%x_%A_%a.out
#SBATCH -e {logs}/%x_%A_%a.err
# precision: {precision}
set -eo pipefail
export PYTHONUNBUFFERED=1
# Match the thread pin to the core reservation (bench_run.py defaults to 1).
export BENCH_THREADS={cpus}

# House rule: no silent venv default. A pal-main job fed to the old pal-env CLI
# fails 640 times over (2026-05-01 incident).
if [ -z "${{PAL_VENV:-}}" ]; then
  echo "PAL_VENV must be set; submit with:" >&2
  echo "  sbatch --export=ALL,PAL_VENV=\\$VENV/bin/activate {job_name}.sbatch" >&2
  exit 1
fi

# Load the site modules providing gcc and python_cuda here.
source "$PAL_VENV"
cd {REPO}

python {RUN_TASK} \\
  --manifest {manifest} \\
  --task-id "{arm.label}/{variant}/seed${{SLURM_ARRAY_TASK_ID}}"
"""


def write_scripts(
    campaign_root: Path,
    arms: tuple[Arm, ...],
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    precision: str = DEFAULT_PRECISION,
) -> list[Path]:
    """Write one sbatch script per (arm x variant); return the written paths."""
    sbatch_dir = campaign_root / "sbatch"
    sbatch_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for arm in arms:
        for variant in variants:
            path = sbatch_dir / f"bp_{arm.label}_{variant}.sbatch"
            path.write_text(sbatch_text(arm, variant, seeds, campaign_root, precision))
            paths.append(path)
    return paths


def git_sha(rev: str = "HEAD") -> str:
    """Return the resolved git sha for `rev`, or `"unknown"` if it fails."""
    try:
        out = subprocess.check_output(
            ["git", "-C", str(REPO), "rev-parse", rev], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def repo_guard_problems(porcelain: str, head: str, remote_head: str) -> list[str]:
    """Pure guard predicate over already-collected git state.

    Args:
        porcelain: Output of `git status --porcelain`.
        head: `git rev-parse HEAD`.
        remote_head: `git rev-parse origin/<BRANCH>` (or `"unknown"`).

    Returns:
        List of human-readable problems; empty means safe to submit.
    """
    problems: list[str] = []
    if porcelain.strip():
        n = len(porcelain.strip().splitlines())
        problems.append(f"working tree dirty ({n} changed path(s)), commit or stash first")
    if head == "unknown":
        problems.append("cannot resolve HEAD")
    if remote_head == "unknown":
        problems.append(f"cannot resolve origin/{BRANCH}, push the branch first")
    elif head != remote_head:
        problems.append(
            f"HEAD ({head[:12]}) != origin/{BRANCH} ({remote_head[:12]}), "
            f"push before launching (doc sec. 6 item 6)"
        )
    return problems


def check_repo() -> list[str]:
    """Collect git state and return `repo_guard_problems` for this checkout."""
    try:
        porcelain = subprocess.check_output(
            ["git", "-C", str(REPO), "status", "--porcelain"]
        ).decode()
    except Exception as exc:  # pragma: no cover - git must exist on the x86 cluster
        return [f"git status failed: {exc}"]
    return repo_guard_problems(porcelain, git_sha("HEAD"), git_sha(f"origin/{BRANCH}"))


def pal_run_dirs(runs_root: Path) -> list[Path]:
    """Return the pal run dirs directly under `runs_root`.

    `--method-override` siblings (`*__as_<method>/`) are excluded: they are eval
    products, not training runs.

    Args:
        runs_root: Per-task runs root.

    Returns:
        Sorted list of run dirs (may be empty).
    """
    if not runs_root.is_dir():
        return []
    return sorted(
        d for d in runs_root.iterdir()
        if d.is_dir() and "__as_" not in d.name and (d / "config.json").exists()
    )


def finished_run_dir(runs_root: Path) -> Path | None:
    """Return the first run dir under `runs_root` carrying `final.json`."""
    for d in pal_run_dirs(runs_root):
        if (d / "final.json").exists():
            return d
    return None


def load_manifest(campaign_root: Path) -> dict:
    """Load `<campaign_root>/manifest.json` or raise `DriverError`."""
    path = campaign_root / "manifest.json"
    if not path.exists():
        raise DriverError(f"no manifest at {path}, run `generate` first")
    return json.loads(path.read_text())


def cmd_generate(args: argparse.Namespace) -> int:
    """Write sbatch scripts + manifest.json. Submits nothing."""
    root = Path(args.campaign_root)
    arms, variants, seeds = _filters(args)
    if args.walltime_override:
        for label in WALLTIMES:
            WALLTIMES[label] = args.walltime_override
    for spec in args.cpus or []:
        label, _, n = spec.partition("=")
        if label not in ARMS_BY_LABEL or not n.isdigit():
            raise DriverError(f"--cpus expects ARM=N with a known arm, got {spec!r}")
        CPUS_OVERRIDE[label] = int(n)
    manifest = build_manifest(root, arms, variants, seeds, args.precision)
    manifest["protocol"]["cpus_per_task"] = {a.label: cpus_for(a.label) for a in arms}
    (root / "logs").mkdir(parents=True, exist_ok=True)
    (root / "runs").mkdir(parents=True, exist_ok=True)
    paths = write_scripts(root, arms, variants, seeds, args.precision)
    manifest["sbatch_scripts"] = [str(p) for p in paths]
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[generate] campaign root: {root}")
    print(f"[generate] precision: {args.precision}")
    print(f"[generate] tasks: {len(manifest['tasks'])} "
          f"({len(arms)} arms x {len(variants)} variants x {len(seeds)} seeds)")
    print(f"[generate] sbatch scripts: {len(paths)} -> {root / 'sbatch'}")
    print(f"[generate] manifest: {manifest_path}")
    return 0


def _submit_one(script: Path, dry_run: bool, array: str | None = None) -> str | None:
    """sbatch one script (or print it under `--dry-run`); return the job id.

    Args:
        script: Generated sbatch script.
        dry_run: Print instead of submit.
        array: Optional `--array` spec that OVERRIDES the script's own
            `#SBATCH --array` directive (command-line wins in Slurm). Used by
            `--seeds` / `smoke` so a seed subset does not need a regenerate.

    Returns:
        The sbatch job id, or None under `--dry-run`.
    """
    pal_venv = os.environ.get("PAL_VENV")
    if not pal_venv:
        raise SystemExit(
            "PAL_VENV must be set in the submitting shell; e.g.\n"
            "  PAL_VENV=$VENV/bin/activate python scripts/bp_campaign/driver.py submit ..."
        )
    cmd = (
        ["sbatch", f"--export=ALL,PAL_VENV={pal_venv}"]
        + (["--array", array] if array else [])
        + [str(script)]
    )
    if dry_run:
        print("[dry-run] " + " ".join(cmd))
        return None
    out = subprocess.check_output(cmd).decode().strip()
    print(f"[submit] {script.name}: {out}")
    return out.rsplit(" ", 1)[-1]


def cmd_submit(args: argparse.Namespace) -> int:
    """Submit the generated arrays (guarded on a clean, pushed repo)."""
    root = Path(args.campaign_root)
    manifest = load_manifest(root)
    arms, variants, seeds = _filters(args)

    for arm in arms:
        load_winner(arm.winner, arm.trial_index)

    problems = check_repo()
    if problems and not args.dry_run:
        print("[submit] REFUSING to submit:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 2
    if manifest["git_sha"] != git_sha("HEAD"):
        print(f"[submit] WARNING: manifest was generated at "
              f"{manifest['git_sha'][:12]}, HEAD is {git_sha('HEAD')[:12]}; "
              f"re-run `generate` if the commands changed.", file=sys.stderr)

    # A command-line `--array` overrides the script's own directive.
    array = array_spec(seeds) if list(seeds) != list(manifest["seeds"]) else None
    n = 0
    for arm in arms:
        for variant in variants:
            script = root / "sbatch" / f"bp_{arm.label}_{variant}.sbatch"
            if not script.exists():
                raise DriverError(f"missing sbatch script {script}, run `generate`")
            _submit_one(script, args.dry_run, array)
            n += 1
    print(f"[submit] {'would submit' if args.dry_run else 'submitted'} {n} arrays "
          f"x {len(seeds)} tasks = {n * len(seeds)} runs")
    return 0


def cmd_smoke(args: argparse.Namespace) -> int:
    """Submit seed 0 x k0 x all 8 arms (8 single-task arrays)."""
    args.arm = None
    args.variants = "k0"
    args.seeds = "0"
    return cmd_submit(args)


def cmd_status(args: argparse.Namespace) -> int:
    """Report squeue state + finished/missing runs against the manifest."""
    root = Path(args.campaign_root)
    manifest = load_manifest(root)
    tasks = manifest["tasks"]

    try:
        out = subprocess.check_output(
            ["squeue", "-u", getpass.getuser(), "-h", "-o", "%j %t"]
        ).decode()
        states: dict[str, int] = {}
        for line in out.splitlines():
            name, _, state = line.strip().partition(" ")
            if name.startswith("bp_"):
                states[state.strip()] = states.get(state.strip(), 0) + 1
        print("[status] squeue (bp_* jobs): "
              + (", ".join(f"{k}={v}" for k, v in sorted(states.items())) or "none"))
    except Exception as exc:
        print(f"[status] squeue unavailable: {exc}")

    per_arm: dict[str, list[int]] = {}
    missing: list[str] = []
    for task in tasks:
        done = finished_run_dir(Path(task["runs_root"])) is not None
        counts = per_arm.setdefault(task["arm_label"], [0, 0])
        counts[1] += 1
        if done:
            counts[0] += 1
        else:
            missing.append(task["task_id"])
    total_done = sum(c[0] for c in per_arm.values())
    print(f"[status] finished runs: {total_done}/{len(tasks)}")
    for label in sorted(per_arm):
        done, total = per_arm[label]
        print(f"  {label:<20} {done:>3}/{total}")
    if missing:
        head = missing[: args.list_missing]
        print(f"[status] missing ({len(missing)}), first {len(head)}:")
        for t in head:
            print(f"  {t}")
    return 0


def cmd_bolton_eval(args: argparse.Namespace) -> int:
    """Run the predict-only alm_bolton eval pass over finished alm runs.

    Each pass writes a sibling `<run_id>__as_alm_bolton/` run dir and its provenance sidecar.

    No `--n-eval` is passed, so the eval-query fingerprint guard stays armed.
    """
    root = Path(args.campaign_root)
    manifest = load_manifest(root)
    winner = load_winner(BOLTON_WINNER, BOLTON_TRIAL_INDEX)
    bolton_set = winner["config"]
    if set(bolton_set) != set(BOLTON_EXPECTED):
        raise DriverError(
            f"alm_bolton winner keys {sorted(bolton_set)} != pinned "
            f"{sorted(BOLTON_EXPECTED)} (doc sec. 3f t10 knobs)"
        )
    for key, expected in BOLTON_EXPECTED.items():
        if float(bolton_set[key]) != expected:
            raise DriverError(
                f"alm_bolton winner {key}={bolton_set[key]!r} != doc sec. 3f "
                f"value {expected!r}"
            )

    alm_tasks = [t for t in manifest["tasks"] if t["arm_label"] == "alm"]
    n_done = n_skip = n_fail = 0
    for task in alm_tasks:
        runs_root = Path(task["runs_root"])
        src = finished_run_dir(runs_root)
        if src is None:
            print(f"[bolton] SKIP {task['task_id']}: no finished alm run under {runs_root}")
            n_skip += 1
            continue
        sibling = src.parent / f"{src.name}__as_{BOLTON_WINNER}"
        if (sibling / "final.json").exists():
            print(f"[bolton] SKIP {task['task_id']}: already done ({sibling.name}/final.json)")
            n_skip += 1
            continue
        cmd = [
            sys.executable, str(REPO / "scripts" / "bench_run.py"), "eval",
            "--run-id", src.name,
            "--runs-root", str(runs_root),
            "--method-override", BOLTON_WINNER,
        ]
        for key in sorted(bolton_set):
            cmd += ["--set", f"{key}={bolton_set[key]}"]
        if args.dry_run:
            print("[dry-run] " + " ".join(cmd))
            continue
        print(f"[bolton] {task['task_id']}: {' '.join(cmd)}", flush=True)
        rc = subprocess.run(cmd, cwd=str(REPO)).returncode
        if rc != 0:
            print(f"[bolton] FAILED {task['task_id']} (rc={rc})", file=sys.stderr)
            n_fail += 1
            continue
        write_provenance(
            sibling,
            winner_path=winner["rel_path"],
            winner_sha256=sha256_file(winner["path"]),
            winner_trial_index=winner["trial_index"],
            arm_label=BOLTON_WINNER,
            variant=task["variant"],
            seed=task["seed"],
        )
        n_done += 1
    print(f"[bolton] done={n_done} skipped={n_skip} failed={n_fail} "
          f"of {len(alm_tasks)} alm tasks")
    return 1 if n_fail else 0


def write_provenance(
    run_dir: Path,
    *,
    winner_path: str,
    winner_sha256: str,
    winner_trial_index: int,
    arm_label: str,
    variant: str,
    seed: int,
) -> Path:
    """Write `winner_provenance.json` into a run dir.

    Args:
        run_dir: The pal run dir the sidecar belongs to.
        winner_path: Repo-relative path of the consumed `confirm_winner.json`.
        winner_sha256: sha256 of that file, computed at run time.
        winner_trial_index: The winner's BO trial id.
        arm_label: Campaign arm label.
        variant: curvature_warp variant suffix.
        seed: Training seed.

    Returns:
        Path of the written sidecar.
    """
    path = run_dir / "winner_provenance.json"
    path.write_text(json.dumps({
        "winner_path": winner_path,
        "winner_sha256": winner_sha256,
        "winner_trial_index": winner_trial_index,
        "arm_label": arm_label,
        "variant": variant,
        "seed": seed,
    }, indent=2) + "\n")
    return path


def _filters(args: argparse.Namespace) -> tuple[tuple[Arm, ...], tuple[str, ...], tuple[int, ...]]:
    """Resolve `--arm` / `--variants` / `--seeds` into concrete selections.

    Args:
        args: Parsed namespace.

    Returns:
        (arms, variants, seeds) tuples.

    Raises:
        DriverError: On an unknown arm label or variant.
    """
    arm_filter = getattr(args, "arm", None)
    if arm_filter:
        labels = [s.strip() for s in arm_filter.split(",") if s.strip()]
        unknown = [label for label in labels if label not in ARMS_BY_LABEL]
        if unknown:
            raise DriverError(
                f"unknown arm(s) {unknown}; valid: {sorted(ARMS_BY_LABEL)}")
        arms = tuple(ARMS_BY_LABEL[label] for label in labels)
    else:
        arms = ARMS

    variants_arg = getattr(args, "variants", None)
    if variants_arg:
        variants = tuple(s.strip() for s in variants_arg.split(",") if s.strip())
        unknown_v = [v for v in variants if v not in VARIANTS]
        if unknown_v:
            raise DriverError(f"unknown variant(s) {unknown_v}; valid: {list(VARIANTS)}")
    else:
        variants = VARIANTS

    seeds_arg = getattr(args, "seeds", None)
    seeds = (tuple(int(s) for s in seeds_arg.split(",") if s.strip())
             if seeds_arg else SEEDS)
    return arms, variants, seeds


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--arm", default=None,
                   help=f"comma-separated arm labels (default all): {sorted(ARMS_BY_LABEL)}")
    p.add_argument("--variants", default=None,
                   help=f"comma-separated variant suffixes (default all): {list(VARIANTS)}")
    p.add_argument("--seeds", default=None,
                   help="comma-separated seeds (default 0-9); array index == seed")


def cmd_flatten(args: argparse.Namespace) -> int:
    """Symlink every run dir under runs/ into runs_flat/ (analyzer view).

    Args:
        args: Parsed CLI namespace (uses `campaign_root`).

    Returns:
        Process exit code (0 on success).
    """
    root = Path(args.campaign_root)
    flat = root / "runs_flat"
    flat.mkdir(parents=True, exist_ok=True)
    n = 0
    for cfg in sorted((root / "runs").glob("*/*/*/config.json")):
        run_dir = cfg.parent.resolve()
        link = flat / run_dir.name
        if link.is_symlink() and link.resolve() != run_dir:
            raise DriverError(
                f"run-dir name collision in runs_flat: {run_dir.name} already "
                f"links to {link.resolve()}"
            )
        if not link.is_symlink():
            link.symlink_to(run_dir)
        n += 1
    print(f"[flatten] {n} run dirs linked into {flat}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the driver's argument parser."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--campaign-root", default=str(DEFAULT_CAMPAIGN_ROOT),
                   help=f"campaign root (default {DEFAULT_CAMPAIGN_ROOT})")
    sub = p.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="write sbatch scripts + manifest.json")
    _add_filters(gen)
    gen.add_argument("--walltime-override", default=None,
                     help="use this --time for every arm instead of WALLTIMES")
    gen.add_argument("--cpus", action="append", default=None, metavar="ARM=N",
                     help="cores (= BENCH_THREADS) for one arm's tasks; "
                          f"repeatable; default {CPUS_PER_TASK} for every arm")
    gen.add_argument("--precision", default=DEFAULT_PRECISION,
                     choices=PRECISION_CHOICES,
                     help=f"training dtype, passed as --precision <value> to "
                          f"`pal run` (default {DEFAULT_PRECISION} for the "
                          f"fp64 rerun); recorded in manifest.json "
                          f"(protocol.precision + per-task)")
    gen.set_defaults(func=cmd_generate)

    sub_submit = sub.add_parser("submit", help="sbatch the generated arrays")
    _add_filters(sub_submit)
    sub_submit.add_argument("--dry-run", action="store_true",
                            help="print the sbatch commands instead of running them")
    sub_submit.set_defaults(func=cmd_submit)

    smoke = sub.add_parser("smoke", help="submit seed 0 x k0 x all 8 arms")
    smoke.add_argument("--dry-run", action="store_true")
    smoke.set_defaults(func=cmd_smoke)

    st = sub.add_parser("status", help="squeue + finished/missing run counts")
    st.add_argument("--list-missing", type=int, default=20,
                    help="how many missing task ids to print (default 20)")
    st.set_defaults(func=cmd_status)

    bolton = sub.add_parser("bolton-eval",
                            help="alm_bolton predict-only eval over finished alm runs")
    bolton.add_argument("--dry-run", action="store_true")
    bolton.set_defaults(func=cmd_bolton_eval)

    flat = sub.add_parser(
        "flatten",
        help="symlink every run dir into <root>/runs_flat/ for the analyzer "
             "(analyze_curvature.py --campaign scans runs_root/*/config.json, one "
             "level deep; __as_alm_bolton siblings land next to their source "
             "run again so the grad-share borrow keeps working)")
    flat.set_defaults(func=cmd_flatten)
    return p


def main(argv: list[str] | None = None) -> int:
    """Entry point.

    Args:
        argv: Argument vector (defaults to `sys.argv[1:]`).

    Returns:
        Process exit code.
    """
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except DriverError as exc:
        print(f"[driver] ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
