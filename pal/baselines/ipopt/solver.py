"""IPOPT classical baseline via `cyipopt.Problem` (native API).

Unlike `minimize_ipopt` (which inherits scipy's `g >= 0` ineq convention),
`cyipopt.Problem` accepts two-sided bounds `cl <= g(x) <= cu`, so signed values
pass through unchanged (ineq: `cl = -inf, cu = 0`, eq: `cl = cu = 0`). Each eval
query is solved independently with multi-start and Deb-dominance selection.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import cyipopt
import numpy as np
import torch
from torch import Tensor

from pal.baselines._shared import seed_everything
from pal.baselines.ipopt.trace import SolveTracer, trace_dir_from_env
from pal.baselines.nlp_adapter import (
    NLPView,
    max_violation,
    select_best_by_dominance,
    zeta_restart_rng,
)
from pal.benchmarks.base import Benchmark, Query
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.tracking.base import Logger

# Print one [ipopt-iter] line every N IPOPT iterations (0 disables).
_ITER_LOG_EVERY = int(os.environ.get("PAL_IPOPT_ITER_LOG_EVERY", "10"))


def _resolve_eval_queries(bench: Benchmark, seed: int) -> Query:
    """Query set solved in ``train``: ``bench.eval_queries(seed)``, or the frozen
    points in ``PAL_IPOPT_EVAL_POINTS`` if set (also used to shard by query).
    """
    pts = os.environ.get("PAL_IPOPT_EVAL_POINTS")
    if not pts:
        return bench.eval_queries(seed)
    from pal.eval.frozen_points import load_frozen_points

    return load_frozen_points(pts).to_query(bench.spec)


# IPOPT status codes (ipopt/src/Interfaces/IpReturnCodes_inc.h).
_IPOPT_STATUS = {
    0: "ok",
    1: "ok_acceptable",
    2: "infeasible_detected",
    3: "step_too_small",
    4: "diverging",
    5: "user_stop",
    6: "feasible_point_found",
    -1: "max_iter",
    -2: "restoration_failed",
    -3: "step_compute_error",
    -4: "max_cpu_time",
    -10: "not_enough_dof",
    -11: "invalid_problem",
    -12: "invalid_option",
    -13: "invalid_number",
    -100: "unrecoverable_exception",
    -101: "non_ipopt_exception",
    -102: "insufficient_memory",
    -199: "internal_error",
}


@dataclass
class IPOPTConfig:
    seed: int = 0
    max_iter: int = 500
    tol: float = 1e-8
    mu_strategy: str = "adaptive"
    hessian_approximation: str = "limited-memory"
    print_level: int = 0
    multi_start: int = 1
    device: str = "cpu"
    # (r, R) runs only restarts r, r+R, r+2R, ...; None runs all.
    restart_shard: tuple[int, int] | None = None


class IPOPTSolver:
    name = "ipopt"
    restart_shardable = True

    def __init__(self, config: IPOPTConfig | None = None) -> None:
        self.config = config or IPOPTConfig()

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Any = None,  # unused
        **hp: Any,
    ) -> TrainResult:
        cfg = _merge_cfg(self.config, dict(hp, seed=seed))
        seed_everything(cfg.seed)
        t0 = time.monotonic()

        spec = bench.spec
        tol = float(spec.tolerance)
        torch_dtype = _torch_dtype_from_spec(spec)

        queries = _resolve_eval_queries(bench, seed)
        n = len(queries)
        solutions = torch.empty((n, spec.dim), dtype=torch_dtype)
        per_query: list[dict[str, Any]] = []
        feasible_total = 0

        _print_run_header(bench, n, cfg)
        for i in range(n):
            zeta_i = queries.zeta[i]
            cond_i = (
                queries.conditions[i] if spec.condition_dim > 0 else None
            )
            view = NLPView(
                bench,
                conditions=cond_i,
                torch_dtype=torch_dtype,
                device=cfg.device,
            )
            _print_query_header(i, n, t0)
            t_q = time.perf_counter()
            best = _solve_one_query(view, zeta_i, cond_i, cfg, tol, query_idx=i)
            t_q = time.perf_counter() - t_q
            solutions[i] = torch.from_numpy(best["x"]).to(dtype=torch_dtype)
            per_query.append(best["diag"])
            feasible_total += int(best["diag"]["feasible"])
            _print_query_footer(i, n, t_q, t0, best["diag"], feasible_total)

            logger.log_step(
                i,
                query_idx=i,
                obj=float(best["diag"]["obj"]),
                max_violation=float(best["diag"]["max_violation"]),
                feasible=int(best["diag"]["feasible"]),
                n_feasible_restarts=int(best["diag"]["n_feasible_restarts"]),
                ipopt_status=int(best["diag"]["status"]),
            )

        wall = time.monotonic() - t0
        # n_restarts counts only the restarts run in this shard.
        local_restarts = (
            len(list(range(cfg.multi_start))[cfg.restart_shard[0]::cfg.restart_shard[1]])
            if cfg.restart_shard is not None
            else cfg.multi_start
        )
        extras: dict[str, Any] = {
            "per_query": per_query,
            "feasible_fraction": feasible_total / max(n, 1),
            "solve_mode": "classical",
            "multi_start_total": cfg.multi_start,
        }
        if cfg.restart_shard is not None:
            extras["restart_shard"] = list(cfg.restart_shard)
        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=local_restarts,
            final_x_on_eval=solutions,
            extras=extras,
        )

    def predict(
        self,
        bench: Benchmark,
        queries: Query,
        train_result: TrainResult,
        logger: Logger | None = None,
    ) -> PredictionOutputs:
        spec = bench.spec
        torch_dtype = _torch_dtype_from_spec(spec)
        n = len(queries)

        cached = train_result.final_x_on_eval
        if cached is not None and cached.shape[0] == n:
            raw = cached.to(dtype=torch_dtype)
            return PredictionOutputs(raw=raw, post=raw, projection=None)

        cfg = self.config
        tol = float(spec.tolerance)
        solutions = torch.empty((n, spec.dim), dtype=torch_dtype)
        t0 = time.monotonic()
        _print_run_header(bench, n, cfg, mode="predict")
        for i in range(n):
            zeta_i = queries.zeta[i]
            cond_i = (
                queries.conditions[i] if spec.condition_dim > 0 else None
            )
            view = NLPView(
                bench,
                conditions=cond_i,
                torch_dtype=torch_dtype,
                device=cfg.device,
            )
            _print_query_header(i, n, t0)
            t_q = time.perf_counter()
            best = _solve_one_query(view, zeta_i, cond_i, cfg, tol, query_idx=i)
            t_q = time.perf_counter() - t_q
            solutions[i] = torch.from_numpy(best["x"]).to(dtype=torch_dtype)
            _print_query_footer(i, n, t_q, t0, best["diag"], None)
        return PredictionOutputs(raw=solutions, post=solutions, projection=None)


class _IPOPTCallbacks:
    """Routes cyipopt callbacks to NLPView (dense Jacobian, row-major flattened)."""

    def __init__(
        self,
        view: NLPView,
        query_idx: int = 0,
        restart: int = 0,
        tracer: SolveTracer | None = None,
    ) -> None:
        self.view = view
        self.query_idx = query_idx
        self.restart = restart
        self.tracer = tracer
        self.t_start = time.perf_counter()
        self.cb_counts = {"obj": 0, "grad": 0, "cons": 0, "jac": 0}

    def _per_constraint_viol(self, g: np.ndarray) -> np.ndarray:
        """pal-signed per-constraint violation vector (eq: |h|, ineq: max(g,0))."""
        viol = np.zeros_like(g)
        if self.view.eq_idx.size:
            viol[self.view.eq_idx] = np.abs(g[self.view.eq_idx])
        if self.view.ineq_idx.size:
            viol[self.view.ineq_idx] = np.maximum(g[self.view.ineq_idx], 0.0)
        return viol

    def objective(self, x: np.ndarray) -> float:
        self.cb_counts["obj"] += 1
        return self.view.f(x)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        self.cb_counts["grad"] += 1
        return self.view.grad_f(x)

    def constraints(self, x: np.ndarray) -> np.ndarray:
        self.cb_counts["cons"] += 1
        if self.view.n_constraints == 0:
            return np.zeros(0)
        return self.view.g(x)

    def jacobian(self, x: np.ndarray) -> np.ndarray:
        self.cb_counts["jac"] += 1
        if self.view.n_constraints == 0:
            return np.zeros(0)
        jac = self.view.jac_g(x)
        if self.tracer is not None:
            # Cache hits on the same x, so no extra forward.
            grad = self.view.grad_f(x)
            g = self.view.g(x)
            obj = self.view.f(x)
            self.tracer.record_deriv(grad, jac, g)
            viol = self._per_constraint_viol(g)
            self.tracer.maybe_update_best(
                x, obj, g, viol, float(viol.max()) if viol.size else 0.0
            )
        return jac.ravel()

    def intermediate(
        self,
        alg_mod: int,
        iter_count: int,
        obj_value: float,
        inf_pr: float,
        inf_du: float,
        mu: float,
        d_norm: float,
        regularization_size: float,
        alpha_du: float,
        alpha_pr: float,
        ls_trials: int,
    ) -> bool:
        """Per-iteration IPOPT callback; returning True keeps IPOPT going."""
        if self.tracer is not None:
            self.tracer.record_iter(
                alg_mod=alg_mod,
                iter_count=iter_count,
                obj_value=obj_value,
                inf_pr=inf_pr,
                inf_du=inf_du,
                mu=mu,
                d_norm=d_norm,
                regularization_size=regularization_size,
                alpha_du=alpha_du,
                alpha_pr=alpha_pr,
                ls_trials=ls_trials,
            )
        if _ITER_LOG_EVERY > 0 and (iter_count % _ITER_LOG_EVERY == 0):
            elapsed = time.perf_counter() - self.t_start
            phase = "R" if alg_mod == 1 else "N"
            print(
                f"[ipopt-iter] q={self.query_idx} r={self.restart} "
                f"iter={iter_count} phase={phase} obj={obj_value:.6e} "
                f"inf_pr={inf_pr:.2e} inf_du={inf_du:.2e} "
                f"mu={mu:.1e} ||d||={d_norm:.2e} a_pr={alpha_pr:.2e} "
                f"ls={ls_trials} elapsed={elapsed:.1f}s",
                flush=True,
            )
        return True


def _effective_ipopt_options(cfg: IPOPTConfig) -> dict[str, Any]:
    """IPOPT options applied: ``IPOPTConfig`` plus env-gated extras.

    ``PAL_IPOPT_SCALING`` sets nlp_scaling_method, ``PAL_IPOPT_ACCEPTABLE=1``
    enables acceptable_tol / acceptable_iter.
    """
    opts: dict[str, Any] = {
        "print_level": int(cfg.print_level),
        "max_iter": int(cfg.max_iter),
        "tol": float(cfg.tol),
        "mu_strategy": cfg.mu_strategy,
        "hessian_approximation": cfg.hessian_approximation,
    }
    scaling = os.environ.get("PAL_IPOPT_SCALING")
    if scaling:
        opts["nlp_scaling_method"] = scaling
    if os.environ.get("PAL_IPOPT_ACCEPTABLE") == "1":
        opts["acceptable_tol"] = float(
            os.environ.get("PAL_IPOPT_ACCEPTABLE_TOL", "1e-4")
        )
        opts["acceptable_iter"] = int(
            os.environ.get("PAL_IPOPT_ACCEPTABLE_ITER", "15")
        )
    return opts


def _decode_status_msg(msg: Any) -> str:
    """cyipopt returns `status_msg` as bytes on most builds; normalize to str."""
    if isinstance(msg, bytes):
        return msg.decode("utf-8", errors="replace")
    return str(msg)


def _parse_slice(spec: str) -> slice:
    a, b = spec.split(":")
    return slice(int(a), int(b))


def _init_mode() -> str:
    return os.environ.get("PAL_IPOPT_INIT_MODE", "box").strip().lower()


def _warmstart_x0(view: NLPView) -> np.ndarray:
    """Warm start from the bench's ``make_initial_raw_params`` layout.

    Flat order matches the spec's variable layout ``[cx, cy, w_raw, d_raw, h_raw]``.
    """
    bench = getattr(view, "bench", None)
    inner = getattr(bench, "_bench", bench)
    make = getattr(inner, "make_initial_raw_params", None)
    if make is None:
        raise ValueError(
            "PAL_IPOPT_INIT_MODE=warmstart requires the bench to expose "
            "make_initial_raw_params (e2/urban_wind does)."
        )
    raw = make(batch_size=1, device="cpu")
    parts = [
        np.asarray(raw[k].detach().cpu().reshape(-1), dtype=np.float64)
        for k in ("cx", "cy", "w", "d", "h")
    ]
    x0 = np.concatenate(parts)
    if x0.shape != view.lo.shape:
        raise ValueError(
            f"warmstart x0 shape {x0.shape} != view box {view.lo.shape}"
        )
    return x0


def _sample_x0(
    view: NLPView, zeta: Tensor, conditions: Tensor | None, restart_idx: int
) -> np.ndarray:
    """Draw the restart's initial point per PAL_IPOPT_INIT_MODE.

    - ``box`` (default): uniform over the full variable box.
    - ``site``: uniform, with position slots restricted to the legal site range.
    - ``warmstart``: the bench's initial layout plus Gaussian jitter, clipped to box.
    """
    rng = zeta_restart_rng(zeta, conditions, restart_idx)
    mode = _init_mode()
    if mode == "box":
        return rng.uniform(view.lo, view.hi)
    if mode == "site":
        x0 = rng.uniform(view.lo, view.hi)
        lo_site = float(os.environ.get("PAL_IPOPT_SITE_LO", "175.0"))
        hi_site = float(os.environ.get("PAL_IPOPT_SITE_HI", "925.0"))
        default_slots = f"0:{2 * view.dim // 5}"  # e2: cx,cy of [cx,cy,w,d,h]
        sl = _parse_slice(
            os.environ.get("PAL_IPOPT_SITE_POS_SLOTS", default_slots)
        )
        seg_lo = np.maximum(view.lo[sl], lo_site)
        seg_hi = np.minimum(view.hi[sl], hi_site)
        x0[sl] = rng.uniform(seg_lo, seg_hi)
        return x0
    if mode == "warmstart":
        base = _warmstart_x0(view)
        sigma = float(os.environ.get("PAL_IPOPT_WARMSTART_SIGMA", "5.0"))
        if restart_idx > 0 and sigma > 0:
            base = base + rng.normal(0.0, sigma, size=base.shape)
        return np.clip(base, view.lo, view.hi)
    raise ValueError(f"unknown PAL_IPOPT_INIT_MODE={mode!r}")


def _make_constraint_bounds(view: NLPView) -> tuple[np.ndarray, np.ndarray]:
    """Build IPOPT's `cl, cu`: eq `cl = cu = 0`, ineq (`g <= 0`) `cl = -inf, cu = 0`."""
    K = view.n_constraints
    cl = np.zeros(K)
    cu = np.zeros(K)
    types = view.spec.constraint_types
    for k, t in enumerate(types):
        if t == "eq":
            cl[k] = 0.0
            cu[k] = 0.0
        else:
            cl[k] = -np.inf
            cu[k] = 0.0
    return cl, cu


def _solve_one_query(
    view: NLPView,
    zeta: Tensor,
    conditions: Tensor | None,
    cfg: IPOPTConfig,
    tol: float,
    query_idx: int = 0,
) -> dict[str, Any]:
    cl, cu = _make_constraint_bounds(view)
    candidates: list[dict[str, Any]] = []
    n_feasible = 0
    trace_dir = trace_dir_from_env()
    ipopt_options = _effective_ipopt_options(cfg)

    total_restarts = max(1, cfg.multi_start)
    if cfg.restart_shard is not None:
        r, R = cfg.restart_shard
        restart_indices = list(range(total_restarts))[r::R]
    else:
        restart_indices = list(range(total_restarts))

    for restart_idx in restart_indices:
        # Per-restart RNG keying makes each restart reproducible on its own.
        x0 = _sample_x0(view, zeta, conditions, restart_idx)
        status = -1
        status_msg = ""
        view.reset_diag()
        tracer = (
            SolveTracer(
                trace_dir,
                query_idx=query_idx,
                restart_idx=restart_idx,
                constraint_names=list(view.spec.constraint_names),
                constraint_types=list(view.spec.constraint_types),
            )
            if trace_dir
            else None
        )
        callbacks = _IPOPTCallbacks(
            view, query_idx=query_idx, restart=restart_idx, tracer=tracer
        )
        print(
            f"[ipopt-restart] q={query_idx} r={restart_idx}/{cfg.multi_start} "
            f"x0_norm={float(np.linalg.norm(x0)):.3e} "
            f"x0_in_box={int(np.all((x0 >= view.lo) & (x0 <= view.hi)))}",
            flush=True,
        )
        t_solve = time.perf_counter()
        try:
            # Fresh Problem per restart: cyipopt state is not reliably resettable.
            nlp = cyipopt.Problem(
                n=view.dim,
                m=view.n_constraints,
                problem_obj=callbacks,
                lb=view.lo.tolist(),
                ub=view.hi.tolist(),
                cl=cl.tolist(),
                cu=cu.tolist(),
            )
            for _opt_k, _opt_v in ipopt_options.items():
                nlp.add_option(_opt_k, _opt_v)
            x, info = nlp.solve(x0)
            x = np.asarray(x, dtype=np.float64)
            status = int(info["status"])
            status_msg = _decode_status_msg(info.get("status_msg", ""))
        except Exception as exc:  # noqa: BLE001
            x = np.clip(x0, view.lo, view.hi)
            status_msg = f"exception: {type(exc).__name__}: {exc}"
        t_solve = time.perf_counter() - t_solve
        _print_diag(view, callbacks, t_solve, status, restart_idx)

        x = np.clip(x, view.lo, view.hi)
        g_val = view.g(x) if view.n_constraints else np.zeros(0)
        obj = float(view.f(x))
        max_viol = max_violation(view, g_val)
        feasible = bool(max_viol <= tol and np.isfinite(obj))
        if feasible:
            n_feasible += 1
        if tracer is not None:
            viol_vec = callbacks._per_constraint_viol(g_val) if g_val.size else g_val
            tracer.finalize(
                status=status,
                status_msg=status_msg,
                n_iter=tracer.last_iter,
                wall_s=t_solve,
                x0=x0,
                x_final=x,
                obj_final=obj,
                g_final=g_val,
                viol_final=viol_vec,
                max_viol_final=max_viol,
                paper_tolerance=float(view.spec.tolerance),
                timing_buckets=dict(view.diag),
                cb_counts=dict(callbacks.cb_counts),
                ipopt_options=ipopt_options,
                init_mode=_init_mode(),
            )
        candidates.append(
            {
                "x": x,
                "obj": obj,
                "max_violation": max_viol,
                "feasible": feasible,
                "status": status,
                "restart_idx": restart_idx,
            }
        )

    best = select_best_by_dominance(candidates)
    diag = {
        "obj": best["obj"],
        "max_violation": best["max_violation"],
        "feasible": best["feasible"],
        "status": best["status"],
        "n_feasible_restarts": n_feasible,
        "restarts": [_restart_record(c) for c in candidates],
    }
    return {"x": best["x"], "diag": diag}


def _restart_record(candidate: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe snapshot of a single multistart candidate (non-finite values become None)."""
    obj = float(candidate["obj"])
    viol = float(candidate["max_violation"])
    return {
        "restart_idx": int(candidate["restart_idx"]),
        "x": np.asarray(candidate["x"], dtype=np.float64).tolist(),
        "obj": obj if np.isfinite(obj) else None,
        "max_violation": viol if np.isfinite(viol) else None,
        "feasible": bool(candidate["feasible"]),
        "status": int(candidate["status"]),
    }


def _print_diag(
    view: NLPView,
    callbacks: _IPOPTCallbacks,
    t_solve: float,
    status: int,
    restart: int,
) -> None:
    """Per-restart timing breakdown."""
    d = view.diag
    cb = callbacks.cb_counts
    n_miss = max(d["n_cache_miss"], 1)
    t_torch = d["t_forward"] + d["t_grad_obj"] + d["t_jac_loop"] + d["t_to_numpy"]
    t_other = max(t_solve - t_torch, 0.0)
    print(
        f"[ipopt-diag] restart={restart} status={status} "
        f"t_solve={t_solve:.3f}s n_torch_evals={d['n_cache_miss']} "
        f"jac_rows={d['jac_rows']} "
        f"cb(obj/grad/cons/jac)={cb['obj']}/{cb['grad']}/{cb['cons']}/{cb['jac']} "
        f"view_calls={d['n_eval_calls']} cache_hits={d['n_cache_hits']}",
        flush=True,
    )
    print(
        f"[ipopt-diag]   torch_total={t_torch:.3f}s "
        f"forward={d['t_forward']:.3f}s grad_obj={d['t_grad_obj']:.3f}s "
        f"jac_loop={d['t_jac_loop']:.3f}s to_numpy={d['t_to_numpy']:.3f}s "
        f"non_torch={t_other:.3f}s",
        flush=True,
    )
    if d["n_cache_miss"] > 0:
        print(
            f"[ipopt-diag]   per_miss(ms): forward={1e3*d['t_forward']/n_miss:.2f} "
            f"grad_obj={1e3*d['t_grad_obj']/n_miss:.2f} "
            f"jac_loop={1e3*d['t_jac_loop']/n_miss:.2f} "
            f"to_numpy={1e3*d['t_to_numpy']/n_miss:.2f}",
            flush=True,
        )


def _print_run_header(bench: Benchmark, n: int, cfg: IPOPTConfig, mode: str = "train") -> None:
    """One-line banner at the start of a run."""
    jac_mode = os.environ.get("PAL_IPOPT_JAC_MODE", "jacrev").strip().lower()
    bench_name = getattr(bench.spec, "name", type(bench).__name__)
    print(
        f"[ipopt-run] mode={mode} bench={bench_name} dim={bench.spec.dim} "
        f"K={len(bench.spec.constraint_types)} n_queries={n} "
        f"multi_start={cfg.multi_start} max_iter={cfg.max_iter} "
        f"device={cfg.device} jac_mode={jac_mode} "
        f"iter_log_every={_ITER_LOG_EVERY}",
        flush=True,
    )


def _print_query_header(i: int, n: int, t0: float) -> None:
    """Pre-solve heartbeat: progress + ETA from per-query mean so far."""
    elapsed = time.monotonic() - t0
    if i > 0:
        mean_t = elapsed / i
        eta_s = mean_t * (n - i)
        eta = _fmt_duration(eta_s)
        mean = f"{mean_t:.1f}s"
    else:
        mean = "n/a"
        eta = "n/a"
    print(
        f"[ipopt-prog] query {i+1}/{n} starting "
        f"elapsed={_fmt_duration(elapsed)} mean_t_query={mean} ETA={eta}",
        flush=True,
    )


def _print_query_footer(
    i: int,
    n: int,
    t_q: float,
    t0: float,
    diag: dict[str, Any],
    feasible_total: int | None,
) -> None:
    """Post-solve summary: feasibility, obj, max_viol + running rate."""
    elapsed = time.monotonic() - t0
    feas_str = ""
    if feasible_total is not None:
        feas_str = f" feasible_so_far={feasible_total}/{i+1} ({100.0*feasible_total/(i+1):.0f}%)"
    status = int(diag['status'])
    status_txt = _IPOPT_STATUS.get(status, "unknown")
    print(
        f"[ipopt-prog] query {i+1}/{n} done t_query={t_q:.2f}s "
        f"feasible={int(diag['feasible'])} obj={diag['obj']:.4e} "
        f"max_viol={diag['max_violation']:.2e} "
        f"status={status}({status_txt}) "
        f"n_feas_restarts={diag['n_feasible_restarts']}"
        f"{feas_str} elapsed={_fmt_duration(elapsed)}",
        flush=True,
    )


def _fmt_duration(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{int(m)}m{int(s):02d}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m"


_PRECISION_TO_TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _torch_dtype_from_spec(spec) -> torch.dtype:
    # PAL_IPOPT_F64=1 evaluates the torch oracle in float64.
    if os.environ.get("PAL_IPOPT_F64") == "1":
        return torch.float64
    return _PRECISION_TO_TORCH_DTYPE.get(
        getattr(spec, "precision", "fp32"), torch.float32
    )


def _merge_cfg(base: IPOPTConfig, overrides: dict[str, Any]) -> IPOPTConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"IPOPTSolver.train: unknown hparam(s) {sorted(unknown)}")
    return IPOPTConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
