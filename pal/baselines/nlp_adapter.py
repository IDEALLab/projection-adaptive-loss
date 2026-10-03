"""Flat-NumPy NLP view over a pal Benchmark for classical solvers.

`NLPView` wraps a single query (one `conditions` row + the benchmark) and
exposes `f(x), grad_f(x), g(x), jac_g(x)` on flat NumPy vectors so IPOPT,
SLSQP, etc. can drive the optimization. Key properties:

- **Native constraint order preserved.** `eq_idx` / `ineq_idx` index into
  the positional layout from `spec.constraint_types`. No eq-first reordering.
- **Signed values.** `g(x)` returns pal's raw signed values (equality:
  `h == 0` feasible; inequality: `g <= 0` feasible). Solver shims flip signs
  at their own boundary if they use a different convention.
- **Single forward per novel x.** One call to `bench.forward(x, conditions)`
  yields the objective and all constraint values on a shared autograd graph;
  `f, grad_f, g, jac_g` are extracted from that graph and memoized.
- **LRU cache** (default size 4) collapses the back-to-back
  `f / grad_f / g / jac_g` queries IPOPT issues at each line-search point
  into one surrogate evaluation.

`zeta_seeded_rng` produces a cross-process-stable RNG from `(zeta, conditions)`
so multi-start restart draws are reproducible.
"""

from __future__ import annotations

import hashlib
import os
import time
from collections import OrderedDict

import numpy as np
import torch
from torch import Tensor

from pal.benchmarks.base import Benchmark

# PAL_IPOPT_JAC_MODE controls how NLPView builds the constraint Jacobian:
#   "jacrev" (default), torch.func.jacrev: one batched backward over all K rows.
#   "loop", sequential autograd.grad per constraint.
_JAC_MODE = os.environ.get("PAL_IPOPT_JAC_MODE", "jacrev").strip().lower()

_PRECISION_TO_DTYPE = {
    "fp32": torch.float32,
    "fp64": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class NLPView:
    def __init__(
        self,
        bench: Benchmark,
        conditions: Tensor | None,
        *,
        cache_size: int = 4,
        torch_dtype: torch.dtype | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Flat-NumPy NLP view over `bench` for one query.

        `torch_dtype` controls the dtype of the torch forward pass, which must
        match the bench's internal weights. If `None`, it's inferred from
        `spec.precision` (defaults to float32, pal's norm). The NumPy-facing
        interface (`f`, `grad_f`, `g`, `jac_g`, `lo`, `hi`) is always float64
        since that's what IPOPT / SLSQP expect.
        """
        self.bench = bench
        self.spec = bench.spec
        self.dim = self.spec.dim
        if torch_dtype is None:
            torch_dtype = _PRECISION_TO_DTYPE.get(
                getattr(self.spec, "precision", "fp32"), torch.float32
            )
        self._torch_dtype = torch_dtype
        self._device = torch.device(device)

        lo_t, hi_t = self.spec.output_bounds
        self.lo = lo_t.detach().cpu().numpy().astype(np.float64)
        self.hi = hi_t.detach().cpu().numpy().astype(np.float64)

        types = self.spec.constraint_types
        self.n_constraints = len(types)
        self.eq_idx = np.array(
            [i for i, t in enumerate(types) if t == "eq"], dtype=np.int64
        )
        self.ineq_idx = np.array(
            [i for i, t in enumerate(types) if t == "ineq"], dtype=np.int64
        )

        if conditions is None or conditions.numel() == 0:
            self._conditions_batched: Tensor | None = None
        else:
            if conditions.ndim == 1:
                conditions = conditions.unsqueeze(0)
            self._conditions_batched = conditions.to(
                dtype=self._torch_dtype, device=self._device
            )

        self._cache_size = int(cache_size)
        self._cache: OrderedDict[bytes, dict] = OrderedDict()

        # [DIAG] per-view counters / timers; reset by reset_diag().
        self.diag = _new_diag()

    def f(self, x: np.ndarray) -> float:
        return self._eval(x)["f"]

    def grad_f(self, x: np.ndarray) -> np.ndarray:
        return self._eval(x)["grad_f"].copy()

    def g(self, x: np.ndarray) -> np.ndarray:
        return self._eval(x)["g"].copy()

    def jac_g(self, x: np.ndarray) -> np.ndarray:
        return self._eval(x)["jac_g"].copy()

    def _eval(self, x: np.ndarray) -> dict:
        x = np.ascontiguousarray(x, dtype=np.float64)
        if x.shape != (self.dim,):
            raise ValueError(
                f"NLPView expects x of shape ({self.dim},), got {x.shape}"
            )

        self.diag["n_eval_calls"] += 1
        key = x.tobytes()
        if key in self._cache:
            self.diag["n_cache_hits"] += 1
            self._cache.move_to_end(key)
            return self._cache[key]
        self.diag["n_cache_miss"] += 1

        t0 = time.perf_counter()
        x_t = torch.tensor(
            x, dtype=self._torch_dtype, device=self._device, requires_grad=True
        ).unsqueeze(0)
        obj_t, cons_list = self.bench.forward(x_t, self._conditions_batched)
        self.diag["t_forward"] += time.perf_counter() - t0

        K = len(cons_list)
        # Both modes reuse the forward graph for the constraint Jacobian
        # (jacrev: batched VJP via is_grads_batched; loop: per-constraint
        # autograd.grad). Retain the graph across the obj-grad call when K>0.
        retain_obj = K > 0
        t0 = time.perf_counter()
        grad_f_t = torch.autograd.grad(
            obj_t.sum(), x_t, retain_graph=retain_obj, create_graph=False
        )[0]
        self.diag["t_grad_obj"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        f_val = float(obj_t.sum().item())
        grad_f = grad_f_t.squeeze(0).detach().cpu().numpy().astype(np.float64)
        self.diag["t_to_numpy"] += time.perf_counter() - t0

        g_vals = np.empty(K, dtype=np.float64)
        jac_g = np.empty((K, self.dim), dtype=np.float64)
        if K > 0:
            t0 = time.perf_counter()
            for k, c in enumerate(cons_list):
                g_vals[k] = float(c.value.sum().item())
            self.diag["t_to_numpy"] += time.perf_counter() - t0

            if _JAC_MODE == "jacrev":
                # Batched VJP over the existing forward graph: K backward
                # passes via vmap, no re-forward of the bench.
                t0 = time.perf_counter()
                flat_cons = torch.cat([c.value.flatten() for c in cons_list])
                K_flat = flat_cons.shape[0]
                eye = torch.eye(
                    K_flat, dtype=flat_cons.dtype, device=flat_cons.device
                )
                jac_full = torch.autograd.grad(
                    flat_cons, x_t,
                    grad_outputs=eye, is_grads_batched=True,
                    retain_graph=False, create_graph=False,
                )[0]
                self.diag["t_jac_loop"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                jac_g = (
                    jac_full.squeeze(1).detach().cpu().numpy().astype(np.float64)
                )
                if jac_g.ndim == 1:
                    jac_g = jac_g.reshape(K, self.dim)
                self.diag["t_to_numpy"] += time.perf_counter() - t0
            else:
                for k, c in enumerate(cons_list):
                    retain = k < K - 1
                    t0 = time.perf_counter()
                    jac_row = torch.autograd.grad(
                        c.value.sum(), x_t,
                        retain_graph=retain, create_graph=False,
                    )[0]
                    self.diag["t_jac_loop"] += time.perf_counter() - t0
                    t0 = time.perf_counter()
                    jac_g[k, :] = (
                        jac_row.squeeze(0).detach().cpu().numpy().astype(np.float64)
                    )
                    self.diag["t_to_numpy"] += time.perf_counter() - t0
        self.diag["jac_rows"] += K

        entry = {"f": f_val, "grad_f": grad_f, "g": g_vals, "jac_g": jac_g}
        self._cache[key] = entry
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return entry

    def reset_diag(self) -> None:
        self.diag = _new_diag()


def _new_diag() -> dict:
    return {
        "n_eval_calls": 0,
        "n_cache_hits": 0,
        "n_cache_miss": 0,
        "jac_rows": 0,
        "t_forward": 0.0,
        "t_grad_obj": 0.0,
        "t_jac_loop": 0.0,
        "t_to_numpy": 0.0,
    }


def max_violation(view: NLPView, g: np.ndarray) -> float:
    """Max constraint violation in pal's signed convention.

    eq: `|h|`; ineq: `max(g, 0)`. Returns 0.0 for an unconstrained view.
    """
    if g.size == 0:
        return 0.0
    viol = np.zeros_like(g)
    if view.eq_idx.size:
        viol[view.eq_idx] = np.abs(g[view.eq_idx])
    if view.ineq_idx.size:
        viol[view.ineq_idx] = np.maximum(g[view.ineq_idx], 0.0)
    return float(viol.max())


def select_best_by_dominance(candidates: list[dict]) -> dict:
    """Deb (2000) constraint-dominance tournament over a list of candidate dicts.

    Each candidate must carry keys `feasible: bool`, `obj: float`, and
    `max_violation: float`. Returns the winner:

    - feasible > infeasible
    - among feasible: lowest objective
    - among infeasible: lowest max-violation, with objective as tie-breaker
    """
    feasible = [c for c in candidates if c["feasible"]]
    if feasible:
        return min(feasible, key=lambda c: c["obj"])
    return min(candidates, key=lambda c: (c["max_violation"], c["obj"]))


def _hashable_bytes(t: Tensor | None) -> bytes:
    if t is None or t.numel() == 0:
        return b""
    return t.detach().cpu().numpy().astype(np.float64).tobytes()


def zeta_seeded_rng(
    zeta: Tensor, conditions: Tensor | None
) -> np.random.Generator:
    """Deterministic RNG for one (zeta, conditions) row.

    Cross-process stable, uses MD5 rather than Python's `hash()` so the same
    (zeta, conditions) produces the same restart draws across runs and
    machines.
    """
    digest = hashlib.md5(
        _hashable_bytes(zeta) + b"|" + _hashable_bytes(conditions)
    ).digest()
    seed_int = int.from_bytes(digest[:4], "little")
    return np.random.default_rng(seed_int)


def zeta_restart_rng(
    zeta: Tensor, conditions: Tensor | None, restart_idx: int
) -> np.random.Generator:
    """Deterministic per-restart RNG for one (zeta, conditions, restart_idx) row.

    Used when multistart restarts may be split across processes (e.g. SLURM
    `--restart-shard r/R`). Hashing `restart_idx` into the seed makes restart `r`
    standalone-reproducible: a single sub-job that runs only restart 5
    produces the same `x0` as slot 5 of an in-process `multi_start=20` run.
    """
    digest = hashlib.md5(
        _hashable_bytes(zeta)
        + b"|"
        + _hashable_bytes(conditions)
        + b"|"
        + int(restart_idx).to_bytes(8, "little", signed=True)
    ).digest()
    seed_int = int.from_bytes(digest[:4], "little")
    return np.random.default_rng(seed_int)
