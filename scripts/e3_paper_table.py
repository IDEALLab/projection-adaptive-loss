"""e3/acopf_ieee57 table with the DC3 metric set, recomputed from each run's checkpoint.

Violations are `eq -> |h|`, `ineq -> max(g, 0)`; per-method numbers are mean +/- std over seeds.

Usage:
  python scripts/e3_paper_table.py RUN_ROOT [--tol 1e-3] [--device cpu] [--out FILE.md]
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

import torch

PAL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAL_ROOT))

from pal.benchmarks import registry as bench_registry  # noqa: E402
from pal.eval.final_eval import _violations  # noqa: E402
from pal.runner.cli import (  # noqa: E402
    _apply_set_overrides,
    _build_cfg_from_hparams,
    _build_eval_queries,
    _build_solver,
)
from pal.runner.probe import BenchProbe  # noqa: E402
from pal.solvers.base import TrainResult  # noqa: E402

METRICS = ("feasibility", "viol_max", "max_eq", "mean_eq", "max_ineq", "mean_ineq", "obj")


def _per_seed_metrics(
    run_dir: Path, *, tol: float, device: str, set_overrides: list[str] | None = None,
) -> dict:
    cfg_json = json.loads((run_dir / "config.json").read_text())
    method = cfg_json["method"]
    bench_id = cfg_json["benchmark_id"]
    seed = int(cfg_json["seed"])
    hparams = cfg_json.get("hparams") or {}
    protocol = cfg_json.get("protocol", "synthetic")

    cfg = _build_cfg_from_hparams(method, hparams)
    cfg.device = device
    # Inference-time overrides, applied to the rehydrated config only.
    if set_overrides:
        _apply_set_overrides(cfg, set_overrides)
    # alm_bolton shares the projector, so give it the same 1e-4 LM damping floor.
    if (
        method == "alm_bolton"
        and bench_id.startswith("e3/")
        and protocol == "paper-faithful"
        and hasattr(cfg, "proj_lambda_min")
    ):
        cfg.proj_lambda_min = 1e-4

    solver = _build_solver(method, cfg, bench_id=bench_id)
    model_state = torch.load(
        run_dir / "model.pt", map_location=device, weights_only=True
    )
    train_result = TrainResult(
        solver_name=method, train_wall_time_s=0.0, n_restarts=1,
        model_state=model_state,
    )

    bench = BenchProbe(bench_registry.get(bench_id, device=device))
    paper_faithful = protocol == "paper-faithful" and bench_id.startswith("e3/")
    queries = _build_eval_queries(bench, None, seed, paper_faithful=paper_faithful)

    bench.set_phase("predict")
    outputs = solver.predict(bench, queries, train_result, logger=None)
    post = outputs.post.detach()

    spec = bench.spec
    conds = None
    if spec.condition_dim > 0:
        conds = queries.conditions.to(post.device)

    c = bench.constraints(post, conds).detach()
    v = _violations(c, spec.constraint_types).double()  # [B, K]

    ctypes = spec.constraint_types
    eq_mask = torch.tensor([t == "eq" for t in ctypes], device=v.device)
    ineq_mask = ~eq_mask
    v_eq = v[:, eq_mask]
    v_ineq = v[:, ineq_mask]

    per_query_max = v.max(dim=-1).values
    feasible = (per_query_max < tol)

    def _mean(t: torch.Tensor) -> float:
        return float(t.mean().item()) if t.numel() else 0.0

    obj = bench.objective(post, conds).detach().double()

    return {
        "method": method,
        "seed": seed,
        "n_queries": int(post.shape[0]),
        "feasibility": float(feasible.double().mean().item()),
        # Per-query max over all eq + ineq rows, averaged over queries.
        "viol_max": _mean(per_query_max),
        "per_query": {
            "viol_max": per_query_max.tolist(),
            "max_eq": v_eq.max(dim=-1).values.tolist() if v_eq.numel() else [],
            "max_ineq": v_ineq.max(dim=-1).values.tolist() if v_ineq.numel() else [],
            "obj": obj.tolist(),
            "feasible": feasible.tolist(),
        },
        "max_eq": _mean(v_eq.max(dim=-1).values) if v_eq.numel() else 0.0,
        "mean_eq": _mean(v_eq.mean(dim=-1)) if v_eq.numel() else 0.0,
        "max_ineq": _mean(v_ineq.max(dim=-1).values) if v_ineq.numel() else 0.0,
        "mean_ineq": _mean(v_ineq.mean(dim=-1)) if v_ineq.numel() else 0.0,
        "obj": _mean(obj),
    }


def _fmt(mean: float, std: float, metric: str) -> str:
    if not math.isfinite(mean):
        return "-"
    if metric == "feasibility":
        return f"{mean:6.3f} ({std:.3f})"
    return f"{mean:.3e} ({std:.1e})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run_root", type=Path)
    ap.add_argument("--tol", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=None, help="write a markdown table here")
    ap.add_argument(
        "--include-failed", action="store_true",
        help="also rehydrate runs whose status.json says failed (use when the "
             "failure was predict-time and training completed; model.pt required)",
    )
    ap.add_argument(
        "--seeds", type=str, default=None,
        help="comma-separated seed filter (e.g. '3' for one array task)",
    )
    ap.add_argument(
        "--json-out", type=Path, default=None,
        help="write the per-seed metric rows as JSON (for a later --from-json merge)",
    )
    ap.add_argument(
        "--from-json", type=str, default=None,
        help="glob of --json-out files: skip recomputation and aggregate those rows",
    )
    ap.add_argument(
        "--per-query-out", type=Path, default=None,
        help="write one CSV row per (method, seed, query) with per-query violations",
    )
    ap.add_argument(
        "--set", dest="set_overrides", action="append", default=[], metavar="KEY=VALUE",
        help="inference-time config override applied after rehydration (repeatable)",
    )
    args = ap.parse_args()

    run_dirs = sorted(
        d for d in args.run_root.iterdir()
        if d.is_dir()
        and (d / "config.json").exists()
        and (d / "model.pt").exists()
        and not d.name.startswith("_cancelled")
    )
    # Failed runs still carry model.pt; skip them and report as diverged seeds.
    failed = []
    kept = []
    for d in run_dirs:
        sp = d / "status.json"
        run_status = json.loads(sp.read_text()).get("status") if sp.exists() else None
        (failed if (run_status == "failed" and not args.include_failed) else kept).append(d)
    run_dirs = kept
    for d in failed:
        print(f"# skipped (status=failed): {d.name}", flush=True)
    if args.seeds:
        wanted = {int(x) for x in args.seeds.split(",")}
        run_dirs = [
            d for d in run_dirs
            if int(json.loads((d / "config.json").read_text())["seed"]) in wanted
        ]
    if args.from_json is None and not run_dirs:
        sys.exit(f"no usable run dirs under {args.run_root}")

    by_method: dict[str, list[dict]] = defaultdict(list)
    ov = f", overrides={args.set_overrides}" if args.set_overrides else ""
    print(f"# e3 paper table  (tol={args.tol:g}, device={args.device}{ov})\n", flush=True)
    print("per-seed:", flush=True)
    if args.from_json is not None:
        import glob as _glob
        loaded = []
        for jp in sorted(_glob.glob(args.from_json)):
            loaded.extend(json.loads(Path(jp).read_text()))
        print(f"# merged {len(loaded)} per-seed rows from {args.from_json}", flush=True)
        rows_iter = [(None, r) for r in loaded]
    else:
        rows_iter = [(d, None) for d in run_dirs]
    for d, pre in rows_iter:
        if pre is not None:
            row = pre
        else:
            try:
                row = _per_seed_metrics(d, tol=args.tol, device=args.device, set_overrides=args.set_overrides)
            except Exception as e:  # noqa: BLE001
                print(f"  SKIP {d.name}: {type(e).__name__}: {e}", flush=True)
                continue
        by_method[row["method"]].append(row)
        print(
            f"  {row['method']:12} seed={row['seed']:<3} "
            f"feas={row['feasibility']:.3f}  "
            f"max_eq={row['max_eq']:.3e}  mean_eq={row['mean_eq']:.3e}  "
            f"max_ineq={row['max_ineq']:.3e}  mean_ineq={row['mean_ineq']:.3e}  "
            f"obj={row['obj']:.4f}",
            flush=True,
        )

    lines = [
        "| Method | Seeds | Feasibility | Viol max | Max eq | Mean eq | Max ineq | Mean ineq | Obj |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    print("\nper-method (mean (std) over seeds):", flush=True)
    for method in sorted(by_method):
        rows = by_method[method]
        # Non-finite seeds count as diverged: feasibility 0, excluded from the averages.
        n_div = sum(1 for r in rows if not math.isfinite(r["max_eq"]))
        agg = {}
        for m in METRICS:
            vals = [
                r[m] for r in rows
                if m == "feasibility" or (math.isfinite(r["max_eq"]) and math.isfinite(r[m]))
            ]
            mean = st.mean(vals) if vals else float("nan")
            std = st.stdev(vals) if len(vals) > 1 else 0.0
            agg[m] = (mean, std)
        seeds_label = f"{len(rows)}" + (f" ({n_div} div.)" if n_div else "")
        print(
            f"  {method:12} n={seeds_label:<12} "
            f"feas={_fmt(*agg['feasibility'], 'feasibility')}  "
            f"viol_max={_fmt(*agg['viol_max'], 'viol_max')}  "
            f"max_eq={_fmt(*agg['max_eq'], 'max_eq')}  "
            f"mean_eq={_fmt(*agg['mean_eq'], 'mean_eq')}  "
            f"max_ineq={_fmt(*agg['max_ineq'], 'max_ineq')}  "
            f"mean_ineq={_fmt(*agg['mean_ineq'], 'mean_ineq')}",
            flush=True,
        )
        lines.append(
            f"| {method} | {seeds_label} | "
            f"{_fmt(*agg['feasibility'], 'feasibility')} | "
            f"{_fmt(*agg['viol_max'], 'viol_max')} | "
            f"{_fmt(*agg['max_eq'], 'max_eq')} | "
            f"{_fmt(*agg['mean_eq'], 'mean_eq')} | "
            f"{_fmt(*agg['max_ineq'], 'max_ineq')} | "
            f"{_fmt(*agg['mean_ineq'], 'mean_ineq')} | "
            f"{_fmt(*agg['obj'], 'obj')} |"
        )

    if args.json_out is not None:
        rows_all = [r for rows in by_method.values() for r in rows]
        args.json_out.write_text(json.dumps(rows_all))
        print(f"wrote {args.json_out}", flush=True)

    if args.per_query_out is not None:
        import csv as _csv
        with args.per_query_out.open("w", newline="") as f:
            w = _csv.writer(f)
            w.writerow(["method", "seed", "query_idx", "viol_max", "max_eq", "max_ineq", "obj", "feasible"])
            for method in sorted(by_method):
                for r in by_method[method]:
                    pq = r["per_query"]
                    for i in range(len(pq["viol_max"])):
                        w.writerow([method, r["seed"], i, pq["viol_max"][i],
                                    pq["max_eq"][i] if pq["max_eq"] else "",
                                    pq["max_ineq"][i] if pq["max_ineq"] else "",
                                    pq["obj"][i], int(pq["feasible"][i])])
        print(f"wrote {args.per_query_out}", flush=True)

    if args.out is not None:
        args.out.write_text("\n".join(lines) + "\n")
        print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
