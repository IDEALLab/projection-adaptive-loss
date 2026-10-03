"""Per-benchmark DC3 partition registry.

Linear-eq benches get a random well-conditioned partition, nonlinear benches a
hand-picked one, e3 the two-step ACOPF partition, and pure-ineq benches ``None``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import torch
from torch import Tensor
from torch.func import jacrev


CompletionStrategy = Literal["linear", "generic_newton", "acopf_two_step"]


@dataclass
class BenchDC3Spec:
    """DC3 completion metadata for a single benchmark.

    ``partial_vars``: y-vector indices the NN predicts (``len = dim - n_eq -
    |known_vars|``).

    ``known_vars`` / ``known_values``: optional pinned y-slots (e3 slack va = 0).
    """

    partial_vars: list[int]
    linear: bool
    newton_reg: float = 1e-8
    warm_start_fn: Callable[..., Tensor] | None = None
    notes: str = ""
    known_vars: list[int] = field(default_factory=list)
    known_values: list[float] = field(default_factory=list)
    completion_strategy: CompletionStrategy = "generic_newton"
    # Holds e.g. the ACOPFPartition under "acopf_partition".
    meta: dict[str, Any] = field(default_factory=dict)


# ``None`` entries are resolved dynamically (linear, pure-ineq, e3).
_REGISTRY: dict[str, BenchDC3Spec | None] = {
    "rosenbrock_eq":       None,   # dim=4, n_eq=1, |partial|=3
    "two_basins":          None,   # dim=4, n_eq=1 (y.sum()=0), |partial|=3
    "equality_dominated":  None,   # dim=10, n_eq=8, |partial|=2

    "e3/acopf_ieee30":  None,
    "e3/acopf_ieee118": None,

    # Pure-ineq benches (n_eq == 0).
    "e2/urban_wind":          None,
    "e4/chip_layout":         None,
}


def _random_partition_with_det_check(
    bench, seed: int = 0, n_tries: int = 100, tol: float = 1e-4,
) -> BenchDC3Spec:
    """Sample random ``partial_vars`` until ``|det(J_other)| > tol``.

    Matches upstream ``utils.py:66-72``. ``J = dh/dy`` is evaluated at ``y = 0``
    (constant for affine eq).
    """
    from pal.baselines.dc3.data_shim import make_eq_resid_per_sample

    spec = bench.spec
    ydim = spec.dim
    n_eq = spec.n_eq
    if n_eq == 0:
        raise ValueError("random-partition: n_eq == 0; caller should skip")

    eq_resid_per_sample_fn = make_eq_resid_per_sample(bench)
    if spec.condition_dim > 0:
        q = bench.sample_queries(n=1, split="train", seed=seed)
        x_proto = q.conditions[0]
    else:
        x_proto = torch.zeros(0)

    y0 = torch.zeros(ydim)
    J = jacrev(lambda y: eq_resid_per_sample_fn(y, x_proto))(y0)  # [n_eq, ydim]

    g = torch.Generator().manual_seed(seed)
    best: BenchDC3Spec | None = None
    for i in range(n_tries):
        perm = torch.randperm(ydim, generator=g).tolist()
        other_vars = sorted(perm[:n_eq])
        partial_vars = sorted(perm[n_eq:])
        J_other = J[:, other_vars]
        det = float(torch.linalg.det(J_other).abs().item())
        if det > tol:
            return BenchDC3Spec(
                partial_vars=partial_vars, linear=True,
                notes=f"random-partition det={det:.3e} after {i + 1} tries",
                meta={"other_vars": other_vars, "seed": seed, "det": det},
            )
        if best is None or det > best.meta.get("det", 0.0):
            best = BenchDC3Spec(
                partial_vars=partial_vars, linear=True,
                notes=f"best-effort random partition; det={det:.3e}",
                meta={"other_vars": other_vars, "seed": seed, "det": det},
            )
    assert best is not None
    return best


def resolve_partial_vars(bench) -> BenchDC3Spec | None:
    """Return the ``BenchDC3Spec`` for ``bench``, or ``None`` for pure-ineq."""
    spec = bench.spec
    if spec.n_eq == 0:
        return None

    # n_eq > dim admits no free/dependent partition.
    if spec.n_eq > spec.dim:
        raise NotImplementedError(
            f"DC3 is structurally inapplicable on {spec.id}: "
            f"n_eq={spec.n_eq} > dim={spec.dim}, no free/dependent partition"
        )

    bench_id = spec.id
    if bench_id.startswith("e3/"):
        from pal.baselines.dc3._completion_acopf import build_partition
        part = build_partition(bench)
        return BenchDC3Spec(
            partial_vars=list(part.partial_vars),
            known_vars=list(part.known_vars),
            known_values=list(part.known_values),
            completion_strategy="acopf_two_step",
            linear=False,
            newton_reg=1e-8,
            notes=(
                "ACOPF two-step completion (paper App. C.3). Step 1 Newton "
                "on vm_D + va_non_slack; Step 2 closed-form for pg_slack + "
                "qg_all. Slack va pinned to 0 via known_vars; NN predicts "
                "pg_pv + vm_spv only."
            ),
            meta={"acopf_partition": part},
        )

    entry = _REGISTRY.get(bench_id)
    if isinstance(entry, BenchDC3Spec):
        if entry.completion_strategy == "generic_newton" and entry.linear:
            entry.completion_strategy = "linear"
        return entry
    if entry is None:
        spec_out = _random_partition_with_det_check(bench)
        spec_out.completion_strategy = "linear"
        return spec_out

    raise TypeError(f"unexpected registry entry for {bench_id!r}: {type(entry)}")
