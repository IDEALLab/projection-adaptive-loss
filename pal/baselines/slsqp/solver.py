"""SLSQP classical baseline via `scipy.optimize.minimize(method='SLSQP')`.

Solves each eval query independently in `train()`, populating
`final_x_on_eval` so the runner's final-eval pipeline has solutions ready
without re-running. `predict()` returns the cached solutions when the
queries match the train-time eval set, otherwise re-solves on the fly.

`multi_start` draws N initial points per query (zeta-seeded for
reproducibility) and selects the winner by Deb constraint-dominance
(feasible > infeasible; ties broken by objective; infeasible ordered
by max violation).

Sign convention: pal reports `g(x) <= 0` for inequalities; scipy's `'ineq'`
means `g(x) >= 0`. The shim negates at the boundary.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.optimize
import torch
from torch import Tensor

from pal.baselines._shared import seed_everything
from pal.baselines.nlp_adapter import (
    NLPView,
    max_violation,
    select_best_by_dominance,
    zeta_seeded_rng,
)
from pal.benchmarks.base import Benchmark, Query
from pal.solvers.base import PredictionOutputs, TrainResult
from pal.tracking.base import Logger


@dataclass
class SLSQPConfig:
    seed: int = 0
    maxiter: int = 200
    ftol: float = 1e-9
    multi_start: int = 1
    device: str = "cpu"


class SLSQPSolver:
    name = "slsqp"

    def __init__(self, config: SLSQPConfig | None = None) -> None:
        self.config = config or SLSQPConfig()

    def train(
        self,
        bench: Benchmark,
        seed: int,
        logger: Logger,
        on_epoch_end: Any = None,  # accepted for CLI conformance; unused (no training loop)
        **hp: Any,
    ) -> TrainResult:
        cfg = _merge_cfg(self.config, dict(hp, seed=seed))
        seed_everything(cfg.seed)
        t0 = time.monotonic()

        spec = bench.spec
        tol = float(spec.tolerance)
        torch_dtype = _torch_dtype_from_spec(spec)

        queries = bench.eval_queries(seed)
        n = len(queries)
        solutions = torch.empty((n, spec.dim), dtype=torch_dtype)
        per_query: list[dict[str, Any]] = []
        feasible_total = 0

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
            best = _solve_one_query(view, zeta_i, cond_i, cfg, tol)
            solutions[i] = torch.from_numpy(best["x"]).to(dtype=torch_dtype)
            per_query.append(best["diag"])
            feasible_total += int(best["diag"]["feasible"])

            logger.log_step(
                i,
                query_idx=i,
                obj=float(best["diag"]["obj"]),
                max_violation=float(best["diag"]["max_violation"]),
                feasible=int(best["diag"]["feasible"]),
                n_feasible_restarts=int(best["diag"]["n_feasible_restarts"]),
                slsqp_status=int(best["diag"]["status"]),
            )

        wall = time.monotonic() - t0
        return TrainResult(
            solver_name=self.name,
            train_wall_time_s=wall,
            n_restarts=cfg.multi_start,
            final_x_on_eval=solutions,
            extras={
                "per_query": per_query,
                "feasible_fraction": feasible_total / max(n, 1),
                "solve_mode": "classical",
            },
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

        # Queries differ from the train-time eval set -> re-solve on the fly.
        cfg = self.config
        tol = float(spec.tolerance)
        solutions = torch.empty((n, spec.dim), dtype=torch_dtype)
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
            best = _solve_one_query(view, zeta_i, cond_i, cfg, tol)
            solutions[i] = torch.from_numpy(best["x"]).to(dtype=torch_dtype)
        return PredictionOutputs(raw=solutions, post=solutions, projection=None)


# helpers


def _solve_one_query(
    view: NLPView,
    zeta: Tensor,
    conditions: Tensor | None,
    cfg: SLSQPConfig,
    tol: float,
) -> dict[str, Any]:
    """Run `cfg.multi_start` SLSQP solves and pick the winner by Deb dominance."""
    rng = zeta_seeded_rng(zeta, conditions)
    candidates: list[dict[str, Any]] = []
    n_feasible = 0

    bounds = list(zip(view.lo.tolist(), view.hi.tolist(), strict=False))
    constraints = _build_scipy_constraints(view)

    for _restart in range(max(1, cfg.multi_start)):
        x0 = rng.uniform(view.lo, view.hi)
        status = -1
        status_msg = ""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                res = scipy.optimize.minimize(
                    fun=view.f,
                    x0=x0,
                    jac=view.grad_f,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": cfg.maxiter, "ftol": cfg.ftol, "disp": False},
                )
                x = np.asarray(res.x, dtype=np.float64)
                status = int(res.status)
                status_msg = str(res.message)
            except Exception as exc:  # noqa: BLE001, SLSQP can raise on singular KKT
                x = np.clip(x0, view.lo, view.hi)
                status_msg = f"exception: {type(exc).__name__}: {exc}"

        x = np.clip(x, view.lo, view.hi)
        g_val = view.g(x) if view.n_constraints else np.zeros(0)
        obj = float(view.f(x))
        max_viol = max_violation(view, g_val)
        feasible = bool(max_viol <= tol and np.isfinite(obj))
        if feasible:
            n_feasible += 1
        candidates.append(
            {
                "x": x,
                "obj": obj,
                "max_violation": max_viol,
                "feasible": feasible,
                "status": status,
                "status_msg": status_msg,
            }
        )

    best = select_best_by_dominance(candidates)
    diag = {
        "obj": best["obj"],
        "max_violation": best["max_violation"],
        "feasible": best["feasible"],
        "status": best["status"],
        "status_msg": best["status_msg"],
        "n_feasible_restarts": n_feasible,
    }
    return {"x": best["x"], "diag": diag}


def _build_scipy_constraints(view: NLPView) -> list[dict[str, Any]]:
    """Build scipy eq + ineq dicts from NLPView.

    pal convention: `g(x) <= 0` feasible.
    scipy 'ineq' convention: `g(x) >= 0` feasible. Negate at the boundary.
    """
    out: list[dict[str, Any]] = []
    eq_idx = view.eq_idx
    ineq_idx = view.ineq_idx

    if eq_idx.size:
        def eq_fun(x: np.ndarray, _idx: np.ndarray = eq_idx) -> np.ndarray:
            return view.g(x)[_idx]

        def eq_jac(x: np.ndarray, _idx: np.ndarray = eq_idx) -> np.ndarray:
            return view.jac_g(x)[_idx]

        out.append({"type": "eq", "fun": eq_fun, "jac": eq_jac})

    if ineq_idx.size:
        def ineq_fun(x: np.ndarray, _idx: np.ndarray = ineq_idx) -> np.ndarray:
            return -view.g(x)[_idx]

        def ineq_jac(x: np.ndarray, _idx: np.ndarray = ineq_idx) -> np.ndarray:
            return -view.jac_g(x)[_idx]

        out.append({"type": "ineq", "fun": ineq_fun, "jac": ineq_jac})

    return out


_PRECISION_TO_TORCH_DTYPE = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


def _torch_dtype_from_spec(spec) -> torch.dtype:
    return _PRECISION_TO_TORCH_DTYPE.get(
        getattr(spec, "precision", "fp32"), torch.float32
    )


def _merge_cfg(base: SLSQPConfig, overrides: dict[str, Any]) -> SLSQPConfig:
    known = {f for f in base.__dataclass_fields__}
    unknown = set(overrides) - known
    if unknown:
        raise TypeError(f"SLSQPSolver.train: unknown hparam(s) {sorted(unknown)}")
    return SLSQPConfig(
        **{**base.__dict__, **{k: v for k, v in overrides.items() if k in known}}
    )
