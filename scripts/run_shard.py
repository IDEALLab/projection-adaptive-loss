"""Worker-pool executor for one shard (`rows[SHARD_I::SHARD_N]`) of a planned ablation sweep.

Threads are set via OMP/MKL/OPENBLAS env vars on each `pal run` subprocess, not via torch.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
from run_ablation import _find_matching_run, _prior_is_ok  # noqa: E402


def _child_env(threads: int) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _run_one(
    row: dict,
    runs_root_str: str,
    log_dir_str: str,
    child_env: dict[str, str],
    shard_i: int,
) -> dict:
    runs_root = Path(runs_root_str)
    log_dir = Path(log_dir_str)
    idx = row["idx"]
    slug = row["slug"]
    id_human = row["id_human"]

    if row.get("allow_skip", True):
        meta = row["skip_meta"]
        prior = _find_matching_run(
            runs_root,
            method=meta["method"],
            bench_id=meta["bench_id"],
            seed=int(meta["seed"]),
            ablation_name=meta["ablation_name"],
            scenario_label=meta["scenario_label"],
        )
        if prior is not None:
            is_ok, _is_failed = _prior_is_ok(prior)
            if is_ok:
                return {
                    "idx": idx, "id": id_human, "slug": slug,
                    "rc": 0, "wall_s": 0.0, "skipped": "ok",
                }

    out_p = log_dir / f"shard{shard_i}_row_{slug}.out"
    err_p = log_dir / f"shard{shard_i}_row_{slug}.err"
    t0 = time.time()
    try:
        with out_p.open("w") as fo, err_p.open("w") as fe:
            proc = subprocess.run(
                row["cmd"], stdout=fo, stderr=fe,
                env=child_env, check=False,
            )
        rc = proc.returncode
    except Exception as exc:
        err_p.write_text(f"worker exception: {exc!r}\n")
        rc = 255
    return {
        "idx": idx, "id": id_human, "slug": slug,
        "rc": rc, "wall_s": round(time.time() - t0, 2), "skipped": None,
    }


def _parse_shard(spec: str) -> tuple[int, int]:
    try:
        i_str, n_str = spec.split("/", 1)
        i, n = int(i_str), int(n_str)
    except Exception as exc:
        raise SystemExit(f"--shard must be 'I/N' (got {spec!r}): {exc}") from exc
    if n <= 0 or not (0 <= i < n):
        raise SystemExit(f"--shard I/N invalid: i={i} n={n}")
    return i, n


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="run one shard of an ablation plan")
    p.add_argument("--jobs", required=True, type=Path, help="jobs.jsonl from --plan-out")
    p.add_argument("--shard", required=True, help="I/N (zero-indexed)")
    p.add_argument("--workers", required=True, type=int)
    p.add_argument("--threads", required=True, type=int, help="threads per worker subprocess")
    p.add_argument("--runs-root", required=True, type=Path)
    p.add_argument("--log-dir", required=True, type=Path)
    args = p.parse_args(argv)

    shard_i, shard_n = _parse_shard(args.shard)
    if not args.jobs.exists():
        raise SystemExit(f"jobs file not found: {args.jobs}")
    args.log_dir.mkdir(parents=True, exist_ok=True)
    args.runs_root.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []
    with args.jobs.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            all_rows.append(json.loads(line))

    my_rows = all_rows[shard_i::shard_n]
    planned_path = args.log_dir / f"shard{shard_i}_planned.txt"
    planned_path.write_text("\n".join(r["slug"] for r in my_rows) + "\n")

    if not my_rows:
        print(f"[shard {shard_i}/{shard_n}] empty slice; nothing to do")
        return 0

    child_env = _child_env(args.threads)
    runs_root_str = str(args.runs_root.resolve())
    log_dir_str = str(args.log_dir.resolve())

    print(
        f"[shard {shard_i}/{shard_n}] rows={len(my_rows)} workers={args.workers} "
        f"threads/worker={args.threads} runs_root={runs_root_str}",
        flush=True,
    )

    shard_log_path = args.log_dir / f"shard{shard_i}.log"
    n_ok = n_skipped = n_failed = 0
    t_start = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            shard_log_path.open("w") as logf:
        futs = {
            pool.submit(
                _run_one, row, runs_root_str, log_dir_str, child_env, shard_i,
            ): row
            for row in my_rows
        }
        for fut_i, fut in enumerate(as_completed(futs), start=1):
            result = fut.result()
            logf.write(json.dumps(result) + "\n")
            logf.flush()
            if result["skipped"] == "ok":
                n_skipped += 1
                tag = "SKIP"
            elif result["rc"] == 0:
                n_ok += 1
                tag = "OK  "
            else:
                n_failed += 1
                tag = "FAIL"
            print(
                f"[shard {shard_i}/{shard_n}] {fut_i}/{len(my_rows)} {tag} "
                f"rc={result['rc']} wall={result['wall_s']}s  {result['id']}",
                flush=True,
            )

    elapsed = round(time.time() - t_start, 1)
    print(
        f"[shard {shard_i}/{shard_n}] done: ok={n_ok} skipped={n_skipped} "
        f"failed={n_failed} total={len(my_rows)} elapsed={elapsed}s",
        flush=True,
    )
    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
