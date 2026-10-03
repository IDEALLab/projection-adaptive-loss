"""e2 inference eval: run the projector for `max_iters` steps, write `eval_diag.json`.

Usage:
    E2_CLEARANCE_PER_PAIR=1 python scripts/eval_e2.py \
        --run-dir /path/to/run \
        --n-queries 64 --max-iters 10 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch

from pal.benchmarks.engineering.e2_urban_wind import E2UrbanWind
from pal.method.solver import PALConfig, _build_projector, _make_constraint_fn
from pal.model import CoordinationMLP


def _cfg_from_hparams(hp: dict) -> PALConfig:
    cfg = PALConfig()
    for k, v in hp.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--n-queries", type=int, default=8)
    p.add_argument("--max-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if os.environ.get("E2_CLEARANCE_PER_PAIR") != "1":
        raise RuntimeError("set E2_CLEARANCE_PER_PAIR=1 (matches training run)")

    run_dir = args.run_dir
    cfg_json = json.loads((run_dir / "config.json").read_text())
    hp = cfg_json["hparams"]
    cfg = _cfg_from_hparams(hp)
    cfg.device = args.device

    # Plain `alm` has no inference projection, so only iter 0 is reported.
    trained_method = cfg_json.get("method")
    if trained_method == "alm" and args.max_iters > 0:
        print("[diag] method=alm: forcing max_iters=0 (no inference projection)",
              flush=True)
        args.max_iters = 0

    print(f"[diag] run_dir={run_dir}", flush=True)
    print(f"[diag] proj_method={cfg.proj_method} proj_delta={cfg.proj_delta} "
          f"proj_lambda_min={hp.get('proj_lambda_min')} eps_active={cfg.eps_active}",
          flush=True)
    print(f"[diag] hidden={cfg.hidden} n_layers={cfg.n_layers}", flush=True)

    bench = E2UrbanWind(device=args.device)
    spec = bench.spec
    print(f"[diag] n_constraints={spec.n_eq + spec.n_ineq} "
          f"tolerance={spec.tolerance}", flush=True)

    queries = bench.eval_queries(seed=args.seed, n=args.n_queries)
    zeta = queries.zeta.to(args.device)
    cond = (queries.conditions.to(args.device)
            if spec.condition_dim > 0 else None)
    print(f"[diag] n_queries={len(queries)} zeta_shape={tuple(zeta.shape)}",
          flush=True)

    output_bounds_list = [
        (float(lo), float(hi))
        for lo, hi in zip(spec.output_bounds[0].tolist(),
                          spec.output_bounds[1].tolist(), strict=False)
    ]
    model = CoordinationMLP(
        dim_zeta=spec.zeta_dim,
        dim_conditions=spec.condition_dim,
        dim_output=spec.dim,
        output_bounds=output_bounds_list,
        hidden=cfg.hidden,
        n_layers=cfg.n_layers,
        output_init_std=spec.model_hparams.get("output_init_std"),
    ).to(args.device)
    state = torch.load(run_dir / "model.pt", map_location=args.device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    with torch.no_grad():
        raw = model(zeta, cond).detach()
    print(f"[diag] raw shape={tuple(raw.shape)}", flush=True)

    projector = _build_projector(cfg, spec, args.device)
    proj_lambda_min = hp.get("proj_lambda_min")
    if proj_lambda_min is not None:
        projector.lambda_min = float(proj_lambda_min)
    constraint_fn = _make_constraint_fn(bench)

    types = list(spec.constraint_types)
    names = list(spec.constraint_names)
    is_eq = torch.tensor([t == "eq" for t in types])
    out_path = args.out or (run_dir / "eval_diag.json")

    def snapshot(
        it_idx: int,
        c_vals: torch.Tensor,
        J: torch.Tensor | None,
        active: torch.Tensor | None,
        step_norm_per_query: list[float] | None,
    ) -> dict:
        c = c_vals.detach().cpu()
        v = torch.where(is_eq, c.abs(), c.clamp(min=0))
        per_query_max = v.max(dim=-1).values
        per_query_argmax = v.argmax(dim=-1)  # offending pair per query
        per_pair_max = v.max(dim=0).values
        top_vals, top_idx = torch.topk(per_pair_max, k=min(5, per_pair_max.numel()))

        # Jacobian-row norm of each query's worst constraint (a vanished gradient shows here).
        per_query_topJ_norm: list[float] = []
        if J is not None:
            J_cpu = J.detach().cpu()  # [B, K, D]
            for b, k in enumerate(per_query_argmax.tolist()):
                per_query_topJ_norm.append(
                    float(J_cpu[b, int(k)].norm().item())
                )

        per_query_active: list[int] = []
        if active is not None:
            per_query_active = [int(x) for x in active.detach().cpu().sum(dim=-1).tolist()]

        return {
            "iter": int(it_idx),
            "viol_max_per_query": [float(x) for x in per_query_max.tolist()],
            "viol_max": float(per_query_max.max().item()),
            "viol_mean": float(v.mean().item()),
            "feasible_count": int((per_query_max < spec.tolerance).sum().item()),
            "top_constraints": [
                {"name": names[int(i)], "viol_max": float(val)}
                for i, val in zip(top_idx.tolist(), top_vals.tolist(), strict=False)
            ],
            "per_query_offender": [
                names[int(k)] for k in per_query_argmax.tolist()
            ],
            "per_query_topJ_norm": per_query_topJ_norm,
            "per_query_active": per_query_active,
            "step_norm_per_query": step_norm_per_query or [],
        }

    def flush(iters_data: list[dict], wall_so_far: float, done: bool) -> None:
        out_path.write_text(json.dumps({
            "run_dir": str(run_dir),
            "n_queries": int(len(queries)),
            "max_iters": int(args.max_iters),
            "tolerance": float(spec.tolerance),
            "wall_time_s": float(wall_so_far),
            "completed": bool(done),
            "per_iter": iters_data,
        }, indent=2))

    iters_data: list[dict] = []
    t0 = time.monotonic()
    y_cur = raw.clone()

    c_values, J = projector._jacobian_at_detached_y(y_cur, constraint_fn, cond)
    active = projector._build_active_mask(c_values)
    snap = snapshot(0, c_values, J, active, step_norm_per_query=None)
    iters_data.append(snap)
    flush(iters_data, time.monotonic() - t0, done=False)
    top1 = snap["top_constraints"][0]
    print(f"[diag] iter   0  viol_max={snap['viol_max']:.4e}  "
          f"feas={snap['feasible_count']}/{len(queries)}  "
          f"top={top1['name']}={top1['viol_max']:.3e}  "
          f"wall={time.monotonic() - t0:.1f}s", flush=True)

    for i in range(args.max_iters):
        active = projector._build_active_mask(c_values)
        y_prev = y_cur.detach().clone()
        y_cur, _ = projector._project_step_inner(y_cur, c_values, J, active)
        y_cur = projector._clamp_to_box(y_cur)
        step_norm = (y_cur - y_prev).detach().norm(dim=-1).cpu().tolist()
        c_values, J = projector._jacobian_at_detached_y(y_cur, constraint_fn, cond)
        active_post = projector._build_active_mask(c_values)
        wall_now = time.monotonic() - t0
        snap = snapshot(
            i + 1, c_values, J, active_post,
            step_norm_per_query=[float(s) for s in step_norm],
        )
        iters_data.append(snap)
        flush(iters_data, wall_now, done=False)
        top1 = snap["top_constraints"][0]
        worst_q = int(torch.tensor(snap["viol_max_per_query"]).argmax().item())
        worst_J = (snap["per_query_topJ_norm"][worst_q]
                   if snap["per_query_topJ_norm"] else float("nan"))
        worst_step = (snap["step_norm_per_query"][worst_q]
                      if snap["step_norm_per_query"] else float("nan"))
        worst_active = (snap["per_query_active"][worst_q]
                        if snap["per_query_active"] else -1)
        print(f"[diag] iter {i + 1:>3}  viol_max={snap['viol_max']:.4e}  "
              f"feas={snap['feasible_count']}/{len(queries)}  "
              f"top={top1['name']}={top1['viol_max']:.3e}  "
              f"q{worst_q}: ||J||={worst_J:.3e} ||Delta y||={worst_step:.3e} "
              f"act={worst_active}  "
              f"wall={wall_now:.1f}s", flush=True)

    flush(iters_data, time.monotonic() - t0, done=True)
    print(f"\n[diag] wrote {out_path}", flush=True)
    last = iters_data[-1]
    print(f"\n[diag] final per-query viol_max (iter {last['iter']}):")
    for q_idx, vmax in enumerate(last["viol_max_per_query"]):
        feas = "FEAS" if vmax < spec.tolerance else "INFEAS"
        print(f"  q{q_idx}: viol_max={vmax:.4e}  [{feas}]")


if __name__ == "__main__":
    main()
