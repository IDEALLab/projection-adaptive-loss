"""GPU-memory runner: one single-GPU `pal run` subprocess per combo, peak memory sampled at 1 Hz.

One process per combo because the PyTorch caching pool only resets at process exit.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import Queue

import yaml


def _sample_peak(gpu_id: int, peakfile: Path, stop_evt: threading.Event) -> None:
    while not stop_evt.is_set():
        try:
            out = subprocess.run(
                [
                    "nvidia-smi",
                    "-i", str(gpu_id),
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True, text=True, timeout=2,
            )
            used_str = out.stdout.strip()
            if used_str.isdigit():
                used = int(used_str)
                cur_str = peakfile.read_text().strip() if peakfile.exists() else "0"
                cur = int(cur_str) if cur_str.isdigit() else 0
                if used > cur:
                    peakfile.write_text(str(used))
        except Exception:
            pass
        stop_evt.wait(1.0)


def _run_combo(
    method: str,
    bench: str,
    seed: int,
    gpu_id: int,
    cfg: dict,
    runs_root: Path,
) -> tuple[str, str, int, int, int, int]:
    bench_safe = bench.replace("/", "_")
    log = runs_root / f"_combo_{method}_{bench_safe}_seed{seed}.log"
    peakfile = runs_root / f"_peak_{method}_{bench_safe}_seed{seed}.txt"
    peakfile.write_text("0")

    stop_evt = threading.Event()
    sampler = threading.Thread(
        target=_sample_peak, args=(gpu_id, peakfile, stop_evt), daemon=True,
    )
    sampler.start()

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    if method == "snarenet":
        env["PAL_MEM_PROBE_DISABLE_BN_BS_CHECK"] = "1"

    cmd = [
        "pal", "run",
        "--method", method,
        "--benchmarks", bench,
        "--seeds", str(seed),
        "--epochs", str(cfg["epochs"]),
        "--batch-size", str(cfg["batch_size"]),
        "--lr", str(cfg["lr"]),
        "--device", cfg["device"],
        "--no-final-eval",
        "--runs-root", str(runs_root),
    ]
    if cfg.get("n_eval") is not None:
        # Optionally shrink the final eval so the peak reflects training.
        cmd += ["--n-eval", str(cfg["n_eval"])]

    start = time.monotonic()
    with log.open("w") as f:
        rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
    elapsed = int(time.monotonic() - start)

    stop_evt.set()
    sampler.join(timeout=2)

    peak_str = peakfile.read_text().strip()
    peak_mb = int(peak_str) if peak_str.isdigit() else 0
    return method, bench, seed, peak_mb, elapsed, rc


def _worker(
    gpu_id: int,
    q: Queue,
    cfg: dict,
    runs_root: Path,
    results_lock: threading.Lock,
    results_file: Path,
) -> None:
    while True:
        item = q.get()
        if item is None:
            q.task_done()
            return
        method, bench, seed = item
        try:
            row = _run_combo(method, bench, seed, gpu_id, cfg, runs_root)
            with results_lock, results_file.open("a", newline="") as f:
                csv.writer(f).writerow(row)
            status = "ok" if row[5] == 0 else f"FAIL rc={row[5]}"
            print(
                f"  [gpu{gpu_id}] {method:<12} {bench:<22} seed={seed} "
                f"{status:<11} peak={row[3]}MB elapsed={row[4]}s",
                flush=True,
            )
        except Exception as e:
            print(
                f"  [gpu{gpu_id}] {method} x {bench} seed={seed} EXCEPTION: {e}",
                flush=True,
            )
        finally:
            q.task_done()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--runs-root", type=Path, required=True)
    p.add_argument("--n-gpus", type=int, default=4)
    args = p.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    args.runs_root.mkdir(parents=True, exist_ok=True)

    combos = [
        (m, b, s)
        for m in cfg["methods"]
        for b in cfg["benchmarks"]
        for s in cfg["seeds"]
    ]
    print(
        f"[memory experiment] {len(combos)} combos across {args.n_gpus} GPUs",
        flush=True,
    )
    print(f"  methods={cfg['methods']}", flush=True)
    print(f"  benches={cfg['benchmarks']}", flush=True)
    print(f"  seeds={cfg['seeds']}", flush=True)
    print(
        f"  BS={cfg['batch_size']} epochs={cfg['epochs']} device={cfg['device']}",
        flush=True,
    )

    results_file = args.runs_root / "_results.csv"
    with results_file.open("w", newline="") as f:
        csv.writer(f).writerow(
            ["method", "bench", "seed", "peak_gpu_mb", "elapsed_s", "rc"]
        )

    q: Queue = Queue()
    for combo in combos:
        q.put(combo)
    for _ in range(args.n_gpus):
        q.put(None)

    results_lock = threading.Lock()
    threads = []
    for gpu_id in range(args.n_gpus):
        t = threading.Thread(
            target=_worker,
            args=(gpu_id, q, cfg, args.runs_root, results_lock, results_file),
        )
        t.start()
        threads.append(t)
    for t in threads:
        t.join()

    print(f"\n[memory experiment] done. results at {results_file}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
