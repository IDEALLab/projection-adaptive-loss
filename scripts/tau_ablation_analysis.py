"""Read-only helper for the curvature_warp probe: loss-share (`share`) and eval (`eval`) modes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))


def _rows(run_dir: Path) -> list[dict]:
    out = []
    with (run_dir / "metrics.jsonl").open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _variant(run_dir: Path) -> str:
    cfg = json.loads((run_dir / "config.json").read_text())
    return str(cfg["benchmark_spec"]["variant"])


def _discover(root: Path, variants: set[str] | None) -> list[Path]:
    dirs = [d for d in sorted(root.iterdir()) if (d / "config.json").exists()]
    if variants:
        dirs = [d for d in dirs if _variant(d) in variants]
    return dirs


def cmd_share(args: argparse.Namespace) -> None:
    root = Path(args.runs_root)
    want = set(args.variants.split(",")) if args.variants else None
    hdr = (
        f"{'variant':>7} {'ep':>5} {'loss':>11} {'obj':>11} {'constr':>11} "
        f"{'disp':>11} {'c%':>7} {'d%':>7} {'w_c':>10} {'w_d':>10} "
        f"{'c_pre':>10} {'c_post':>10} {'tau':>9}"
    )
    print(hdr)
    print("-" * len(hdr))
    for d in _discover(root, want):
        v = _variant(d)
        rows = _rows(d)
        rows = [r for r in rows if "loss/objective" in r]
        by_step = {r["step"]: r for r in rows}
        marks = [s for s in args.epochs if s in by_step]
        for s in marks:
            r = by_step[s]
            tot = abs(r["loss/objective"]) + r["loss/constraint"] + r["loss/displacement"]
            cpct = 100.0 * r["loss/constraint"] / tot if tot else float("nan")
            dpct = 100.0 * r["loss/displacement"] / tot if tot else float("nan")
            print(
                f"{v:>7} {s:>5} {r['loss']:>11.4e} {r['loss/objective']:>11.4e} "
                f"{r['loss/constraint']:>11.4e} {r['loss/displacement']:>11.4e} "
                f"{cpct:>7.3f} {dpct:>7.3f} {r['residual/w_0']:>10.3e} "
                f"{r['displacement/w_0']:>10.3e} {r['residual/c_pre_0_mean']:>10.3e} "
                f"{r['residual/c_post_0_mean']:>10.3e} "
                f"{r['residual/tau_effective']:>9.1e}"
            )
        # weight trajectory summary
        w = [r["residual/w_0"] for r in rows]
        wd = [r["displacement/w_0"] for r in rows]
        cpo = [r["residual/c_post_0_mean"] for r in rows]
        n_above = sum(1 for x in w if x > 1.0000001e-6)
        print(
            f"{v:>7}  traj: w_c min={min(w):.3e} max={max(w):.3e} final={w[-1]:.3e} "
            f"steps_above_floor={n_above}/{len(w)} | "
            f"w_d max={max(wd):.3e} final={wd[-1]:.3e} | "
            f"c_post max={max(cpo):.3e} final={cpo[-1]:.3e}"
        )
        print()


def cmd_eval(args: argparse.Namespace) -> None:
    """Rehydrate model.pt, re-predict, and report feasibility + gap quantiles.

    `eval_rows.parquet` has no y, so the gap is recomputed against `bench.reference_solution`.
    """
    import pandas as pd
    import torch

    from pal.benchmarks import get as get_benchmark
    from pal.runner.cli import _build_cfg_from_hparams, _build_solver
    from pal.solvers.base import TrainResult

    root = Path(args.runs_root)
    want = set(args.variants.split(",")) if args.variants else None
    print(f"{'variant':>7} {'tau':>9} {'n':>5} {'feas':>7} {'gap p50':>11} "
          f"{'gap p90':>11} {'gap mean':>11} {'obj_chk':>9}")
    rows_out = []
    for d in _discover(root, want):
        cfg = json.loads((d / "config.json").read_text())
        bench_id = cfg["benchmark_id"]
        v = cfg["benchmark_spec"]["variant"]
        tau_h = (cfg.get("hparams") or {}).get("tau")
        tau = tau_h if tau_h is not None else cfg["benchmark_spec"]["tau"]
        method = cfg["method"]
        seed = int(cfg["seed"])
        n_eval = int(cfg["n_eval_effective"])

        solver_cfg = _build_cfg_from_hparams(method, cfg.get("hparams") or {})
        solver_cfg.device = "cpu"
        solver = _build_solver(method, solver_cfg, bench_id=bench_id)
        bench = get_benchmark(bench_id, device="cpu")
        queries = bench.eval_queries(seed, n=n_eval)
        train_result = TrainResult(
            solver_name=method,
            train_wall_time_s=0.0,
            n_restarts=1,
            model_state=torch.load(
                d / "model.pt", map_location="cpu", weights_only=True
            ),
        )
        pred = solver.predict(bench, queries, train_result, logger=None)
        rows = pd.read_parquet(d / "eval_rows.parquet").sort_values("query_idx")

        x = queries.conditions.detach().double()
        y_post = pred.post.detach().double()
        f_star = bench.reference_solution(x).f_star
        obj_post = bench.objective(y_post, x)
        obj_chk = float(
            (obj_post - torch.tensor(rows["obj_post"].to_numpy())).abs().max()
        )
        feas = torch.tensor(rows["feasible_post"].to_numpy(), dtype=torch.bool)
        gap = (obj_post - f_star)[feas]
        rec = {
            "run_dir": d.name,
            "variant": v,
            "tau": float(tau),
            "n": int(len(rows)),
            "n_feasible": int(feas.sum()),
            "feas": float(feas.double().mean()),
            "p50": float(torch.quantile(gap, 0.5)) if gap.numel() else float("nan"),
            "p90": float(torch.quantile(gap, 0.9)) if gap.numel() else float("nan"),
            "mean": float(gap.mean()) if gap.numel() else float("nan"),
            "obj_check": obj_chk,
        }
        rows_out.append(rec)
        print(f"{v:>7} {rec['tau']:>9.1e} {rec['n']:>5} {rec['feas']:>7.3f} "
              f"{rec['p50']:>11.4e} {rec['p90']:>11.4e} {rec['mean']:>11.4e} "
              f"{obj_chk:>9.1e}")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows_out, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("share")
    s.add_argument("--runs-root", required=True)
    s.add_argument("--variants", default=None)
    s.add_argument("--epochs", type=int, nargs="+",
                   default=[1, 100, 500, 1000, 1500, 2000])
    s.set_defaults(fn=cmd_share)
    e = sub.add_parser("eval")
    e.add_argument("--runs-root", required=True)
    e.add_argument("--variants", default=None)
    e.add_argument("--json-out", default=None)
    e.set_defaults(fn=cmd_eval)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
