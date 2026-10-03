#!/usr/bin/env python3
"""Submit a benchmark campaign to the GH200 cluster from a TOML manifest.

One job per (benchmark, method, seed), plus an optional dependent render job per group.
Usage: python scripts/submit_paper_campaign_gpu.py --manifest <toml> [--groups A,B] [--submit]
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import tomllib
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class JobSpec:
    group_name: str
    run_group: str
    wrapper: str
    benchmark: str
    method: str
    seed: int
    command: list[str]


def _load_manifest(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        data = tomllib.load(f)
    if "campaign" not in data or "groups" not in data:
        raise SystemExit("[campaign] manifest must define [campaign] and at least one [[groups]] entry")
    return data


def _slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum():
            out.append(ch.lower())
        else:
            out.append("-")
    slug = "".join(out).strip("-")
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "job"


def _join_csv(values: list[Any]) -> str:
    return ",".join(str(v) for v in values)


def _build_export_items(
    group: dict[str, Any],
    campaign: dict[str, Any],
    benchmark: str,
    method: str,
    seed: int,
) -> list[str]:
    run_group = str(group.get("run_group") or group["name"])
    runs_root_base = str(campaign["runs_root_base"]).rstrip("/")
    runs_root = group.get("runs_root") or f"{runs_root_base}/{run_group}"
    items = {
        "BENCHMARK": benchmark,
        "METHODS": method,
        "SEEDS": str(seed),
        "RUNS_ROOT": str(runs_root),
        "EPOCHS": str(group.get("epochs", campaign.get("default_epochs", 200))),
        "PAL_DEVICE": str(group.get("device", "cpu")),
        "ENABLE_WANDB": "1" if group.get("enable_wandb", campaign.get("default_enable_wandb", False)) else "0",
        "WANDB_PROJECT": str(group.get("wandb_project", campaign.get("default_wandb_project", "pal"))),
    }
    if "wandb_entity" in group:
        items["WANDB_ENTITY"] = str(group.get("wandb_entity", ""))
    elif campaign.get("default_wandb_entity", "") != "":
        items["WANDB_ENTITY"] = str(campaign["default_wandb_entity"])
    if group.get("edf_environment"):
        items["EDF_ENVIRONMENT"] = str(group["edf_environment"])
    if group.get("hf_token_file"):
        items["HF_TOKEN_FILE"] = str(group["hf_token_file"])
    if group.get("wandb_token_file"):
        items["WANDB_TOKEN_FILE"] = str(group["wandb_token_file"])
    if group.get("pal_extra_args"):
        items["PAL_EXTRA_ARGS"] = str(group["pal_extra_args"])
    return [f"{key}={value}" for key, value in items.items()]


def _build_sbatch_command(
    group: dict[str, Any],
    campaign: dict[str, Any],
    benchmark: str,
    method: str,
    seed: int,
) -> list[str]:
    wrapper = str(group["wrapper"])
    job_name = group.get(
        "job_name",
        f"pal-paper-{_slug(group['name'])}-{_slug(method)}-{_slug(benchmark)}-s{seed}",
    )
    cmd = ["sbatch", "--parsable", f"--job-name={job_name}"]
    for key in ("account", "partition", "nodes", "ntasks", "cpus_per_task", "gpus", "mem", "time"):
        manifest_key = key
        if manifest_key in group:
            cli_key = manifest_key.replace("_", "-")
            cmd.append(f"--{cli_key}={group[manifest_key]}")
    export_items = ["ALL", *_build_export_items(group, campaign, benchmark, method, seed)]
    cmd.append(f"--export={','.join(export_items)}")
    cmd.append(str(REPO_ROOT / wrapper))
    return cmd


def _expand_jobs(data: dict[str, Any], selected_groups: set[str] | None) -> tuple[list[JobSpec], list[str]]:
    campaign = data["campaign"]
    jobs: list[JobSpec] = []
    notes: list[str] = []
    default_methods = list(campaign.get("default_methods", ["pal"]))
    default_seeds = list(campaign.get("default_seeds", [0]))
    default_submit_unit = str(campaign.get("default_submit_unit", "run"))

    for group in data["groups"]:
        group_name = str(group["name"])
        if selected_groups and group_name not in selected_groups:
            continue
        if group.get("enabled", True) is False:
            notes.append(f"[skip] {group_name}: disabled in manifest")
            continue
        submission = str(group.get("submission", "sbatch"))
        if submission != "sbatch":
            notes.append(f"[manual] {group_name}: {group.get('notes', 'manual submission required')}")
            continue

        benchmarks = list(group.get("benchmarks", []))
        if not benchmarks:
            notes.append(f"[skip] {group_name}: no benchmarks listed")
            continue
        methods = list(group.get("methods", default_methods))
        seeds = [int(v) for v in group.get("seeds", default_seeds)]
        submit_unit = str(group.get("submit_unit", default_submit_unit))
        if submit_unit != "run":
            raise SystemExit(f"[campaign] unsupported submit_unit={submit_unit!r} in group {group_name}")

        run_group = str(group.get("run_group") or group_name)
        for benchmark in benchmarks:
            for method in methods:
                for seed in seeds:
                    cmd = _build_sbatch_command(group, campaign, str(benchmark), str(method), int(seed))
                    jobs.append(
                        JobSpec(
                            group_name=group_name,
                            run_group=run_group,
                            wrapper=str(group["wrapper"]),
                            benchmark=str(benchmark),
                            method=str(method),
                            seed=int(seed),
                            command=cmd,
                        )
                    )
    return jobs, notes


def _render_command(group_name: str, run_group: str, runs_root_base: str, deps: list[str]) -> list[str]:
    runs_root = f"{runs_root_base.rstrip('/')}/{run_group}"
    cmd = ["sbatch", "--parsable", f"--job-name=pal-render-{_slug(group_name)}"]
    if deps:
        cmd.append(f"--dependency=afterany:{':'.join(deps)}")
    export_items = [
        "ALL",
        f"RUN_GROUP={run_group}",
        f"RUNS_ROOT={runs_root}",
        f"RUNS_PARQUET={runs_root}/runs.parquet",
        f"RESULTS_MD={runs_root}/results.md",
        f"PAPER_RESULTS_DIR={runs_root}/paper_results",
    ]
    cmd.append(f"--export={','.join(export_items)}")
    cmd.append(str(REPO_ROOT / "scripts/render_run_summary_gpu.slurm"))
    return cmd


def _run_command(cmd: list[str], submit: bool) -> str | None:
    printable = shlex.join(cmd)
    print(printable)
    if not submit:
        return None
    result = subprocess.run(cmd, check=True, text=True, capture_output=True)
    job_id = result.stdout.strip().splitlines()[-1]
    print(f"  -> {job_id}")
    return job_id


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="submit paper benchmark campaigns to the GH200 cluster")
    p.add_argument(
        "--manifest",
        default="slurm/campaigns/paper_gpu.toml",
        help="path to the campaign TOML manifest",
    )
    p.add_argument(
        "--groups",
        default=None,
        help="comma-separated subset of group names to submit (default: all enabled groups)",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="actually call sbatch; default is a dry-run that prints the commands",
    )
    args = p.parse_args(argv)

    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    if not manifest_path.exists():
        print(f"[campaign] manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    selected_groups = None
    if args.groups:
        selected_groups = {part.strip() for part in args.groups.split(",") if part.strip()}

    data = _load_manifest(manifest_path)
    campaign = data["campaign"]
    jobs, notes = _expand_jobs(data, selected_groups)

    if not jobs and not notes:
        print("[campaign] nothing to do")
        return 0

    print(f"[campaign] {campaign['name']}")
    print(f"[campaign] manifest: {manifest_path}")
    print(f"[campaign] mode: {'submit' if args.submit else 'dry-run'}")
    print(f"[campaign] default W&B project: {campaign.get('default_wandb_project', 'pal')}")
    print()

    for note in notes:
        print(note)
    if notes:
        print()

    jobs_by_group: dict[str, list[JobSpec]] = defaultdict(list)
    for job in jobs:
        jobs_by_group[job.group_name].append(job)

    render_after = bool(campaign.get("render_after_group", True))
    submitted_job_ids: dict[str, list[str]] = defaultdict(list)

    for group_name in sorted(jobs_by_group):
        group_jobs = jobs_by_group[group_name]
        run_group = group_jobs[0].run_group
        print(f"[group] {group_name} ({len(group_jobs)} jobs) -> {run_group}")
        for job in group_jobs:
            print(
                f"  - {job.benchmark} / {job.method} / seed={job.seed} "
                f"via {job.wrapper}"
            )
            job_id = _run_command(job.command, submit=args.submit)
            if job_id:
                submitted_job_ids[group_name].append(job_id)
        print()

        if render_after:
            render_cmd = _render_command(
                group_name=group_name,
                run_group=run_group,
                runs_root_base=str(campaign["runs_root_base"]),
                deps=submitted_job_ids.get(group_name, []),
            )
            print(f"[render] {group_name}")
            _run_command(render_cmd, submit=args.submit)
            print()

    print("[campaign] complete")
    if not args.submit:
        print("[campaign] dry-run only; rerun with --submit on the GH200 cluster to enqueue jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
