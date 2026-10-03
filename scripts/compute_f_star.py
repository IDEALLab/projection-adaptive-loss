#!/usr/bin/env python3
"""Compute benchmark-level f* references for paper tables.

Solves the first eval query per benchmark with a classical solver and writes
`paper_tables/f_star.json`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pal.baselines.nlp_adapter import NLPView
from pal.baselines.slsqp.solver import (
    SLSQPConfig,
)
from pal.baselines.slsqp.solver import (
    _solve_one_query as _solve_one_query_slsqp,
)
from pal.baselines.slsqp.solver import (
    _torch_dtype_from_spec as _torch_dtype_from_spec_slsqp,
)
from pal.benchmarks import get as get_benchmark
from pal.benchmarks import list_all as list_benchmarks

_AUTO_SOLVER_ORDER = ("ipopt", "slsqp")


def _parse_benchmarks(raw: str | None) -> list[str]:
    if raw is None:
        return list_benchmarks()
    return [part.strip() for part in raw.split(",") if part.strip()]


def _load_ipopt_helpers() -> tuple[Any, Any, Any]:
    try:
        from pal.baselines.ipopt.solver import (  # noqa: PLC0415
            IPOPTConfig,
        )
        from pal.baselines.ipopt.solver import (
            _solve_one_query as _solve_one_query_ipopt,
        )
        from pal.baselines.ipopt.solver import (
            _torch_dtype_from_spec as _torch_dtype_from_spec_ipopt,
        )
    except Exception as exc:  # noqa: BLE001 - surface optional dependency errors cleanly
        raise RuntimeError(f"IPOPT unavailable: {exc}") from exc
    return IPOPTConfig, _solve_one_query_ipopt, _torch_dtype_from_spec_ipopt


def _base_context(bench_id: str, *, seed: int, device: str) -> tuple[Any, Any, Any, Any]:
    bench = get_benchmark(bench_id, device=device)
    spec = bench.spec
    query = bench.eval_queries(seed=seed, n=1)
    zeta = query.zeta[0]
    conditions = query.conditions[0] if spec.condition_dim > 0 else None
    return bench, spec, zeta, conditions


def _solve_with_slsqp(
    bench_id: str,
    *,
    seed: int,
    multi_start: int,
    max_iter: int,
    tol: float,
    device: str,
) -> dict[str, Any]:
    bench, spec, zeta, conditions = _base_context(bench_id, seed=seed, device=device)
    view = NLPView(
        bench,
        conditions=conditions,
        torch_dtype=_torch_dtype_from_spec_slsqp(spec),
        device=device,
    )
    cfg = SLSQPConfig(
        seed=seed,
        maxiter=max_iter,
        ftol=tol,
        multi_start=multi_start,
        device=device,
    )
    best = _solve_one_query_slsqp(view, zeta, conditions, cfg, float(spec.tolerance))
    diag = best["diag"]
    return {
        "f_star": float(diag["obj"]),
        "feasible": bool(diag["feasible"]),
        "max_violation": float(diag["max_violation"]),
        "status": int(diag["status"]),
        "solver": "slsqp",
        "query_seed": int(seed),
        "query_index": 0,
        "multi_start": int(multi_start),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "computed_at": datetime.now(UTC).isoformat(),
    }


def _solve_with_ipopt(
    bench_id: str,
    *,
    seed: int,
    multi_start: int,
    max_iter: int,
    tol: float,
    device: str,
) -> dict[str, Any]:
    IPOPTConfig, _solve_one_query_ipopt, _torch_dtype_from_spec_ipopt = _load_ipopt_helpers()
    bench, spec, zeta, conditions = _base_context(bench_id, seed=seed, device=device)
    view = NLPView(
        bench,
        conditions=conditions,
        torch_dtype=_torch_dtype_from_spec_ipopt(spec),
        device=device,
    )
    cfg = IPOPTConfig(
        seed=seed,
        max_iter=max_iter,
        tol=tol,
        multi_start=multi_start,
        device=device,
    )
    best = _solve_one_query_ipopt(view, zeta, conditions, cfg, float(spec.tolerance))
    diag = best["diag"]
    return {
        "f_star": float(diag["obj"]),
        "feasible": bool(diag["feasible"]),
        "max_violation": float(diag["max_violation"]),
        "status": int(diag["status"]),
        "solver": "ipopt",
        "query_seed": int(seed),
        "query_index": 0,
        "multi_start": int(multi_start),
        "max_iter": int(max_iter),
        "tol": float(tol),
        "computed_at": datetime.now(UTC).isoformat(),
    }


def _pick_best_result(results: list[dict[str, Any]]) -> dict[str, Any]:
    feasible = [r for r in results if r["feasible"]]
    if feasible:
        return min(feasible, key=lambda r: float(r["f_star"]))
    return min(results, key=lambda r: (float(r["max_violation"]), float(r["f_star"])))


def _solve_f_star(
    bench_id: str,
    *,
    solver: str,
    seed: int,
    multi_start: int,
    max_iter: int,
    tol: float,
    device: str,
) -> dict[str, Any]:
    solver_order = list(_AUTO_SOLVER_ORDER if solver == "auto" else (solver,))
    successes: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []

    for solver_name in solver_order:
        try:
            if solver_name == "slsqp":
                result = _solve_with_slsqp(
                    bench_id,
                    seed=seed,
                    multi_start=multi_start,
                    max_iter=max_iter,
                    tol=tol,
                    device=device,
                )
            elif solver_name == "ipopt":
                result = _solve_with_ipopt(
                    bench_id,
                    seed=seed,
                    multi_start=multi_start,
                    max_iter=max_iter,
                    tol=tol,
                    device=device,
                )
            else:
                raise ValueError(f"unsupported solver: {solver_name}")
        except Exception as exc:  # noqa: BLE001 - report and continue in auto mode
            attempts.append({"solver": solver_name, "ok": False, "error": str(exc)})
            if solver != "auto":
                raise
            print(f"[f*] {bench_id}: {solver_name} unavailable/failed ({exc}); trying next solver")
            continue

        successes.append(result)
        attempts.append(
            {
                "solver": solver_name,
                "ok": True,
                "feasible": bool(result["feasible"]),
                "max_violation": float(result["max_violation"]),
                "f_star": float(result["f_star"]),
                "status": int(result["status"]),
            }
        )
        print(
            f"[f*] {bench_id}: {solver_name} -> f*={result['f_star']:+.6e} "
            f"max_viol={result['max_violation']:.3e} "
            f"{'feasible' if result['feasible'] else 'infeasible'}"
        )

    if not successes:
        reasons = "; ".join(f"{a['solver']}: {a.get('error', 'failed')}" for a in attempts)
        raise RuntimeError(f"no solver succeeded for {bench_id}: {reasons}")

    chosen = dict(_pick_best_result(successes))
    chosen["attempted_solvers"] = attempts
    return chosen


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="compute benchmark-level f* references")
    p.add_argument("--benchmarks", default=None, help="comma-separated benchmark ids (default: all)")
    p.add_argument(
        "--solver",
        choices=["auto", "slsqp", "ipopt"],
        default="auto",
        help="reference solver to use (default: auto = IPOPT then SLSQP fallback)",
    )
    p.add_argument("--seed", type=int, default=0, help="eval query seed for canonical query selection")
    p.add_argument("--multi-start", type=int, default=10, help="classical restarts per benchmark")
    p.add_argument("--max-iter", type=int, default=500, help="solver max_iter / maxiter")
    p.add_argument("--tol", type=float, default=1e-9, help="solver tol / ftol")
    p.add_argument("--device", default="cpu", help="solver device (default: cpu)")
    p.add_argument(
        "--out",
        default=str(Path(__file__).resolve().parents[1] / "paper_tables" / "f_star.json"),
        help="output JSON path",
    )
    args = p.parse_args(argv)

    benchmarks = _parse_benchmarks(args.benchmarks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {}
    for bench_id in benchmarks:
        print(f"[f*] solving {bench_id}")
        results[bench_id] = _solve_f_star(
            bench_id,
            solver=args.solver,
            seed=args.seed,
            multi_start=args.multi_start,
            max_iter=args.max_iter,
            tol=args.tol,
            device=args.device,
        )
        status = "feasible" if results[bench_id]["feasible"] else "infeasible"
        print(
            f"[f*] {bench_id}: f*={results[bench_id]['f_star']:+.6e} "
            f"max_viol={results[bench_id]['max_violation']:.3e} {status}"
        )

    out_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(f"[f*] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
