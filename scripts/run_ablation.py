"""Launch an ablation study (scenarios x benchmarks x seeds) from a YAML descriptor.

Runs with a prior ok final.json are skipped unless --force. `--plan-out jobs.jsonl` writes the
pending rows for scripts/run_shard.py instead of running them.

Usage: python scripts/run_ablation.py --ablation <yaml> [--dry-run | --force | --plan-out F]
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_ablation_yaml(path: Path) -> dict[str, Any]:
    with path.open() as f:
        ab = yaml.safe_load(f)
    required = ("ablation_name", "method", "benchmarks", "seeds", "anchor",
                "base_hparams", "scenarios")
    missing = [k for k in required if k not in ab]
    if missing:
        raise SystemExit(f"{path}: missing required keys: {missing}")
    scn_labels = [s["label"] for s in ab["scenarios"]]
    if ab["anchor"] not in scn_labels:
        raise SystemExit(
            f"{path}: anchor '{ab['anchor']}' not in scenarios labels {scn_labels}"
        )
    return ab


def _effective_hparams(base: dict, overrides: dict | None) -> dict:
    """Merge `base (+) overrides`. Overrides win on conflict."""
    out = dict(base)
    if overrides:
        out.update(overrides)
    return out


def _fmt_set(key: str, value: Any) -> str:
    """Format a value as the CLI --set expects. Bools -> lowercase true/false."""
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


def _find_matching_run(
    runs_root: Path,
    method: str,
    bench_id: str,
    seed: int,
    ablation_name: str,
    scenario_label: str,
) -> Path | None:
    """Return the latest run dir whose config.json matches the ablation tuple, or None."""
    if not runs_root.exists():
        return None
    bench_slug = bench_id.replace("/", "-")
    seed_token = f"seed{seed}_"
    method_token = f"_{method}_"
    candidates: list[Path] = []
    for d in runs_root.iterdir():
        if not d.is_dir():
            continue
        name = d.name
        if method_token not in name or seed_token not in name or bench_slug not in name:
            continue
        cfg_path = d / "config.json"
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except Exception:
            continue
        hp = cfg.get("hparams") or {}
        if (
            cfg.get("method") == method
            and cfg.get("benchmark_id") == bench_id
            and int(cfg.get("seed", -1)) == seed
            and hp.get("ablation_name") == ablation_name
            and hp.get("scenario_label") == scenario_label
        ):
            candidates.append(d)
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def _prior_is_ok(run_dir: Path) -> tuple[bool, bool]:
    """Return (is_ok, is_failed) based on final.json + status.json."""
    final_path = run_dir / "final.json"
    status_path = run_dir / "status.json"
    is_ok = False
    is_failed = False
    if final_path.exists():
        try:
            final = json.loads(final_path.read_text())
            if "feasibility_post" in final or str(final.get("status", "")).lower() == "ok":
                is_ok = True
        except Exception:
            pass
    if status_path.exists():
        try:
            status = json.loads(status_path.read_text())
            if str(status.get("status", "")).lower() == "failed":
                is_failed = True
            if str(status.get("status", "")).lower() == "ok":
                is_ok = True
        except Exception:
            pass
    return is_ok, is_failed


def _row_slug(idx: int, label: str, bench_id: str, seed: int) -> str:
    """Filesystem-safe slug for per-row log filenames (slashes become '_')."""
    raw = f"{label}_{bench_id}_seed{seed}"
    return f"{idx:04d}_" + raw.replace("/", "_").replace(" ", "_")


def _build_run_cmd(
    method: str,
    bench_id: str,
    seed: int,
    hparams: dict,
    runs_root: Path,
    device: str,
    extra_flags: list[str],
) -> list[str]:
    """Compose the `pal run` command for one (scenario, bench, seed)."""
    cmd = [
        sys.executable, "-m", "pal.runner.cli", "run",
        "--method", method,
        "--benchmarks", bench_id,
        "--seeds", str(seed),
        "--device", device,
        "--runs-root", str(runs_root),
    ]
    # All hparams go through --set; the CLI handles casting and unknown keys.
    for key, value in hparams.items():
        cmd += ["--set", _fmt_set(key, value)]
    cmd += extra_flags
    return cmd


def _plan_mode(
    *,
    args: argparse.Namespace,
    scenarios: list[dict],
    benchmarks: list[str],
    seeds: list[int],
    base_hparams: dict,
    ablation_name: str,
    method: str,
    runs_root: Path,
    extra_flags: list[str],
    total: int,
) -> int:
    """Emit pending rows as JSONL to args.plan_out; returns 0 on success.

    Each row has idx, id_human, slug, cmd, allow_skip and skip_meta. Prior-ok rows are dropped
    unless --force, prior-failed rows unless --retry-failed or --force.
    """
    plan_path: Path = args.plan_out
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    allow_skip = not args.force
    n_planned = n_skip_ok = n_skip_fail = 0
    idx = -1
    with plan_path.open("w") as f:
        for scn in scenarios:
            label = scn["label"]
            overrides = scn.get("overrides") or {}
            hparams = _effective_hparams(base_hparams, overrides)
            hparams["ablation_name"] = ablation_name
            hparams["scenario_label"] = label
            for bench_id in benchmarks:
                for seed in seeds:
                    idx += 1
                    prior = _find_matching_run(
                        runs_root, method, bench_id, seed, ablation_name, label,
                    )
                    if prior is not None:
                        is_ok, is_failed = _prior_is_ok(prior)
                        if is_ok and not args.force:
                            n_skip_ok += 1
                            continue
                        if is_failed and not args.retry_failed and not args.force:
                            n_skip_fail += 1
                            continue
                    cmd = _build_run_cmd(
                        method=method, bench_id=bench_id, seed=seed,
                        hparams=hparams, runs_root=runs_root,
                        device=args.device, extra_flags=extra_flags,
                    )
                    row = {
                        "idx": idx,
                        "id_human": f"{label}/{bench_id}/seed{seed}",
                        "slug": _row_slug(idx, label, bench_id, seed),
                        "cmd": cmd,
                        "allow_skip": allow_skip,
                        "skip_meta": {
                            "method": method,
                            "bench_id": bench_id,
                            "seed": seed,
                            "ablation_name": ablation_name,
                            "scenario_label": label,
                        },
                    }
                    f.write(json.dumps(row) + "\n")
                    n_planned += 1
    print(
        f"[ablation] plan: planned={n_planned} skipped_ok={n_skip_ok} "
        f"skipped_fail={n_skip_fail} total={total} -> {plan_path}",
        file=sys.stderr,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="launch an ablation study from a YAML descriptor")
    p.add_argument("--ablation", required=True, type=Path, help="path to ablation YAML")
    p.add_argument("--runs-root", default="runs",
                   help="where to write run dirs (default: ./runs)")
    p.add_argument("--device", default="cpu", help="'auto'|'cpu'|'cuda'|'mps'")
    p.add_argument("--dry-run", action="store_true",
                   help="print commands without running")
    p.add_argument("--force", action="store_true",
                   help="re-run even if a prior ok run exists")
    p.add_argument("--retry-failed", action="store_true",
                   help="re-run rows whose prior run ended in status=failed")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project", default="pal")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--stop-on-fail", action="store_true",
                   help="exit on the first failing run rather than continuing")
    p.add_argument("--plan-out", type=Path, default=None,
                   help="emit pending rows as JSONL (one per line) instead of running; "
                        "consumed by scripts/run_shard.py")
    p.add_argument("--benchmarks", default=None,
                   help="comma-separated subset of benchmarks to run (overrides YAML's full list); "
                        "all entries must appear in the YAML benchmarks")
    args = p.parse_args(argv)

    ab = _load_ablation_yaml(args.ablation)
    method = ab["method"]
    benchmarks = list(ab["benchmarks"])
    if args.benchmarks:
        requested = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
        unknown = [b for b in requested if b not in benchmarks]
        if unknown:
            raise SystemExit(
                f"--benchmarks: {unknown} not in YAML benchmarks {benchmarks}"
            )
        benchmarks = requested
    seeds = [int(s) for s in ab["seeds"]]
    base_hparams = dict(ab["base_hparams"])
    scenarios = ab["scenarios"]
    ablation_name = ab["ablation_name"]

    runs_root = Path(args.runs_root).resolve()
    runs_root.mkdir(parents=True, exist_ok=True)

    extra_flags: list[str] = []
    if args.wandb:
        extra_flags += ["--wandb", "--wandb-project", args.wandb_project]
        if args.wandb_entity:
            extra_flags += ["--wandb-entity", args.wandb_entity]

    total = len(scenarios) * len(benchmarks) * len(seeds)
    print(f"[ablation] {ablation_name}: {len(scenarios)} scenarios x "
          f"{len(benchmarks)} benchmarks x {len(seeds)} seeds = {total} rows",
          file=sys.stderr)
    print(f"[ablation] runs root: {runs_root}", file=sys.stderr)

    if args.plan_out is not None:
        return _plan_mode(
            args=args, scenarios=scenarios, benchmarks=benchmarks, seeds=seeds,
            base_hparams=base_hparams, ablation_name=ablation_name, method=method,
            runs_root=runs_root, extra_flags=extra_flags, total=total,
        )

    n_done = n_skip_ok = n_skip_fail = n_ran = n_failed = 0
    for scn in scenarios:
        label = scn["label"]
        overrides = scn.get("overrides") or {}
        hparams = _effective_hparams(base_hparams, overrides)
        hparams["ablation_name"] = ablation_name
        hparams["scenario_label"] = label
        for bench_id in benchmarks:
            for seed in seeds:
                n_done += 1
                prior = _find_matching_run(
                    runs_root, method, bench_id, seed, ablation_name, label,
                )
                if prior is not None:
                    is_ok, is_failed = _prior_is_ok(prior)
                    if is_ok and not args.force:
                        print(f"[{n_done}/{total}] SKIP ok: {label} / {bench_id} / seed={seed} "
                              f"({prior.name})")
                        n_skip_ok += 1
                        continue
                    if is_failed and not args.retry_failed and not args.force:
                        print(f"[{n_done}/{total}] SKIP failed (use --retry-failed): "
                              f"{label} / {bench_id} / seed={seed} ({prior.name})")
                        n_skip_fail += 1
                        continue
                cmd = _build_run_cmd(
                    method=method, bench_id=bench_id, seed=seed,
                    hparams=hparams, runs_root=runs_root,
                    device=args.device, extra_flags=extra_flags,
                )
                print(f"[{n_done}/{total}] RUN  {label} / {bench_id} / seed={seed}")
                if args.dry_run:
                    print("       " + " ".join(cmd))
                    continue
                rc = subprocess.call(cmd)
                if rc != 0:
                    n_failed += 1
                    print(f"[{n_done}/{total}] FAIL rc={rc}: {label} / {bench_id} / seed={seed}")
                    if args.stop_on_fail:
                        print("[ablation] --stop-on-fail set; aborting.")
                        return rc
                else:
                    n_ran += 1

    print(f"[ablation] done: ran={n_ran} skip_ok={n_skip_ok} skip_fail={n_skip_fail} "
          f"fail={n_failed} total={total}")
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
