"""Per-(query, restart) IPOPT solve tracer, enabled by ``PAL_IPOPT_TRACE``.

It never triggers a torch evaluation: it only records callback values and
``NLPView`` cache hits. Files, one pair per solve, under ``PAL_IPOPT_TRACE``:

    q{query:04d}_r{restart:03d}.jsonl   # kind=iter | deriv | final rows
    q{query:04d}_r{restart:03d}.npz     # x0, x_final, x_best, cons_*, viol_*
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

_FLUSH_EVERY = int(os.environ.get("PAL_IPOPT_TRACE_FLUSH_EVERY", "200"))


def trace_dir_from_env() -> str | None:
    """Return the trace directory if ``PAL_IPOPT_TRACE`` is set (and non-empty)."""
    d = os.environ.get("PAL_IPOPT_TRACE")
    return d if d else None


def _to_jsonable(v: Any) -> Any:
    if isinstance(v, (np.floating,)):
        f = float(v)
        return f if np.isfinite(f) else None
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, np.ndarray):
        return [_to_jsonable(x) for x in v.tolist()]
    if isinstance(v, float):
        return v if np.isfinite(v) else None
    return v


class SolveTracer:
    """Per-(query, restart) recorder. One instance per IPOPT solve."""

    def __init__(
        self,
        trace_dir: str,
        query_idx: int,
        restart_idx: int,
        constraint_names: list[str],
        constraint_types: list[str],
    ) -> None:
        self.dir = Path(trace_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.query_idx = int(query_idx)
        self.restart_idx = int(restart_idx)
        self.constraint_names = list(constraint_names)
        self.constraint_types = list(constraint_types)
        stem = f"q{self.query_idx:04d}_r{self.restart_idx:03d}"
        self.jsonl_path = self.dir / f"{stem}.jsonl"
        self.npz_path = self.dir / f"{stem}.npz"

        self._buf: list[str] = []
        self._t0 = time.perf_counter()
        self.last_iter: int = 0

        self.best_max_viol: float = float("inf")
        self.best: dict[str, Any] | None = None

        self.jsonl_path.write_text("")

    def record_iter(
        self,
        *,
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
    ) -> None:
        self.last_iter = int(iter_count)
        self._append(
            {
                "kind": "iter",
                "iter": int(iter_count),
                "restoration": int(alg_mod == 1),
                "obj": obj_value,
                "inf_pr": inf_pr,
                "inf_du": inf_du,
                "mu": mu,
                "d_norm": d_norm,
                "regularization_size": regularization_size,
                "alpha_du": alpha_du,
                "alpha_pr": alpha_pr,
                "ls_trials": int(ls_trials),
                "t": time.perf_counter() - self._t0,
            }
        )

    def record_deriv(
        self, grad_f: np.ndarray, jac_g: np.ndarray, g: np.ndarray
    ) -> None:
        row_norms = (
            np.linalg.norm(jac_g, axis=1) if jac_g.size else np.zeros(0)
        )
        self._append(
            {
                "kind": "deriv",
                "iter": self.last_iter,
                "grad_f_norm": float(np.linalg.norm(grad_f)),
                "jac_row_norm_min": float(row_norms.min()) if row_norms.size else None,
                "jac_row_norm_max": float(row_norms.max()) if row_norms.size else None,
                "jac_row_norms": row_norms,
                "g": g,
                "t": time.perf_counter() - self._t0,
            }
        )

    def maybe_update_best(
        self,
        x: np.ndarray,
        obj: float,
        g: np.ndarray,
        per_constraint_viol: np.ndarray,
        max_viol: float,
    ) -> None:
        if not (max_viol < self.best_max_viol):
            return
        self.best_max_viol = float(max_viol)
        self.best = {
            "iter": self.last_iter,
            "x": np.asarray(x, dtype=np.float64).copy(),
            "obj": float(obj),
            "g": np.asarray(g, dtype=np.float64).copy(),
            "viol": np.asarray(per_constraint_viol, dtype=np.float64).copy(),
            "max_viol": float(max_viol),
        }

    def finalize(
        self,
        *,
        status: int,
        status_msg: str,
        n_iter: int,
        wall_s: float,
        x0: np.ndarray,
        x_final: np.ndarray,
        obj_final: float,
        g_final: np.ndarray,
        viol_final: np.ndarray,
        max_viol_final: float,
        paper_tolerance: float,
        timing_buckets: dict[str, Any],
        cb_counts: dict[str, int],
        ipopt_options: dict[str, Any] | None = None,
        init_mode: str = "box",
    ) -> None:
        feas_1e4 = bool(max_viol_final <= 1e-4 and np.isfinite(obj_final))
        feas_paper = bool(
            max_viol_final <= paper_tolerance and np.isfinite(obj_final)
        )
        best = self.best
        best_max_viol = self.best_max_viol if best is not None else None
        final_row = {
            "kind": "final",
            "query_idx": self.query_idx,
            "restart_idx": self.restart_idx,
            "status": int(status),
            "status_msg": status_msg,
            "n_iter": int(n_iter),
            "wall_s": float(wall_s),
            "obj_final": obj_final,
            "max_viol_final": max_viol_final,
            "feasible_at_1e-4": feas_1e4,
            "feasible_at_paper_tol": feas_paper,
            "paper_tolerance": float(paper_tolerance),
            "best_iter": best["iter"] if best else None,
            "best_obj": best["obj"] if best else None,
            "best_max_viol": best_max_viol
            if (best_max_viol is not None and np.isfinite(best_max_viol))
            else None,
            "constraint_names": self.constraint_names,
            "constraint_types": self.constraint_types,
            "viol_final_named": _named(self.constraint_names, viol_final),
            "viol_best_named": _named(self.constraint_names, best["viol"])
            if best
            else None,
            "timing_buckets": timing_buckets,
            "cb_counts": cb_counts,
            "ipopt_options": ipopt_options or {},
            "init_mode": init_mode,
        }
        self._append(final_row)
        self.flush()

        arrays: dict[str, np.ndarray] = {
            "x0": np.asarray(x0, dtype=np.float64),
            "x_final": np.asarray(x_final, dtype=np.float64),
            "g_final": np.asarray(g_final, dtype=np.float64),
            "viol_final": np.asarray(viol_final, dtype=np.float64),
        }
        if best is not None:
            arrays["x_best"] = best["x"]
            arrays["g_best"] = best["g"]
            arrays["viol_best"] = best["viol"]
        np.savez(self.npz_path, **arrays)

    def _append(self, row: dict[str, Any]) -> None:
        self._buf.append(
            json.dumps({k: _to_jsonable(v) for k, v in row.items()})
        )
        if len(self._buf) >= _FLUSH_EVERY:
            self.flush()

    def flush(self) -> None:
        if not self._buf:
            return
        with self.jsonl_path.open("a") as f:
            f.write("\n".join(self._buf) + "\n")
        self._buf.clear()


def _named(names: list[str], vec: np.ndarray) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    v = np.asarray(vec, dtype=np.float64).ravel()
    for i, name in enumerate(names):
        val = float(v[i]) if i < v.size else None
        out[name] = val if (val is not None and np.isfinite(val)) else None
    return out
