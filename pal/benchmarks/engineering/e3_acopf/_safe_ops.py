"""vmap-safe re-implementations of the ml4opf AC-OPF ops the benchmark uses.

ml4opf's ``@torch.jit.script`` primitives reject functorch ``BatchedTensor`` proxies, so these
mirror ``ml4opf.functional.ac`` in plain torch using last-dim operations only.
"""

from __future__ import annotations

import torch
from torch import Tensor


def objective(pg: Tensor, c0: Tensor, c1: Tensor, c2: Tensor) -> Tensor:
    """ACOPF quadratic cost per sample. pg: [..., n_gen] -> [...]."""
    return (c0 + c1 * pg + c2 * pg.pow(2)).sum(dim=-1)


def angle_difference(va: Tensor, bus_fr: Tensor, bus_to: Tensor) -> Tensor:
    """Per-branch angle difference. va: [..., n_bus] -> [..., n_branch]."""
    return va[..., bus_fr] - va[..., bus_to]


def bound_residual(
    x: Tensor, xmin: Tensor, xmax: Tensor
) -> tuple[Tensor, Tensor]:
    """Signed bound violations (negative = inside). Broadcasts over batch prefix."""
    return xmin - x, x - xmax


def thermal_residual(
    pf: Tensor, pt: Tensor, qf: Tensor, qt: Tensor, smax: Tensor
) -> tuple[Tensor, Tensor]:
    """Thermal limit residuals (pf^2+qf^2-smax^2, pt^2+qt^2-smax^2)."""
    smaxsq = smax.pow(2)
    thrm_1 = pf.pow(2) + qf.pow(2) - smaxsq
    thrm_2 = pt.pow(2) + qt.pow(2) - smaxsq
    return thrm_1, thrm_2


def flows_from_voltage(
    vm: Tensor,
    dva: Tensor,
    bus_fr: Tensor,
    bus_to: Tensor,
    gff: Tensor, gft: Tensor, gtf: Tensor, gtt: Tensor,
    bff: Tensor, bft: Tensor, btf: Tensor, btt: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Branch real/reactive power flows (from/to ends) from voltage + branch admittance."""
    vm_fr = vm[..., bus_fr]
    vm_to = vm[..., bus_to]

    wf = vm_fr.pow(2)
    wt = vm_to.pow(2)

    vm_frto = vm_fr * vm_to
    cosdva = torch.cos(dva)
    sindva = torch.sin(dva)
    wr = vm_frto * cosdva
    wi = vm_frto * sindva

    pf = gff * wf + gft * wr + bft * wi
    qf = -bff * wf - bft * wr + gft * wi
    pt = gtt * wt + gtf * wr - btf * wi
    qt = -btt * wt - btf * wr - gtf * wi
    return pf, pt, qf, qt


def balance_residual_bus(
    pd_bus: Tensor, qd_bus: Tensor,
    pg_bus: Tensor, qg_bus: Tensor,
    vm: Tensor,
    pf_bus: Tensor, pt_bus: Tensor,
    qf_bus: Tensor, qt_bus: Tensor,
    gs: Tensor, bs: Tensor,
) -> tuple[Tensor, Tensor]:
    """Per-bus active/reactive power balance residuals."""
    vm2 = vm.pow(2)
    p_viol = pg_bus - pd_bus - pt_bus - pf_bus - gs * vm2
    q_viol = qg_bus - qd_bus - qt_bus - qf_bus + bs * vm2
    return p_viol, q_viol


def map_to_bus_pad(x: Tensor, x_per_bus: Tensor) -> Tensor:
    """Aggregate component-wise values to buses via padded ragged index.

    ``x`` has shape ``[..., n_comp]`` (last dim = generators / loads / branches).
    ``x_per_bus`` has shape ``[n_bus, max_at_bus]`` with pad value ``n_comp``
    (i.e. the index one past the last real component; gather into a padded
    zero slot sums to zero).

    Output: ``[..., n_bus]``. Uses only last-dim ops so it composes under
    ``vmap`` regardless of leading batch prefix.
    """
    zero = x.new_zeros(x.shape[:-1] + (1,))
    x_padded = torch.cat([x, zero], dim=-1)  # [..., n_comp + 1]
    gathered = x_padded[..., x_per_bus]       # [..., n_bus, max_at_bus]
    return gathered.sum(dim=-1)               # [..., n_bus]


def calc_violations(
    pd: Tensor, qd: Tensor,
    pg: Tensor, qg: Tensor,
    vm: Tensor,
    va: Tensor | None,
    *,
    data: dict[str, Tensor],
    dva: Tensor | None = None,
) -> dict[str, Tensor]:
    """Signed (unclamped) violations for every AC-OPF constraint group.

    Mirrors ``ml4opf.formulations.ac.violation.ACViolation.calc_violations``
    with ``reduction='none', clamp=False``, which is exactly how
    ``E3ACOPF.forward`` calls it. All leading batch dims are preserved;
    last-dim ops only.
    """
    if dva is None:
        if va is None:
            raise ValueError("calc_violations: either va or dva must be provided")
        dva = angle_difference(va, data["bus_fr"], data["bus_to"])

    vm_lower, vm_upper = bound_residual(vm, data["vmin"], data["vmax"])
    pg_lower, pg_upper = bound_residual(pg, data["pgmin"], data["pgmax"])
    qg_lower, qg_upper = bound_residual(qg, data["qgmin"], data["qgmax"])
    dva_lower, dva_upper = bound_residual(dva, data["dvamin"], data["dvamax"])

    pf, pt, qf, qt = flows_from_voltage(
        vm, dva, data["bus_fr"], data["bus_to"],
        data["gff"], data["gft"], data["gtf"], data["gtt"],
        data["bff"], data["bft"], data["btf"], data["btt"],
    )

    thrm_1, thrm_2 = thermal_residual(pf, pt, qf, qt, data["smax"])

    # Embed component-wise quantities to the bus dim (method='pad' default).
    pd_bus = map_to_bus_pad(pd, data["bus_loads"])
    qd_bus = map_to_bus_pad(qd, data["bus_loads"])
    pg_bus = map_to_bus_pad(pg, data["bus_gens"])
    qg_bus = map_to_bus_pad(qg, data["bus_gens"])
    pf_bus = map_to_bus_pad(pf, data["bus_arcs_fr"])
    pt_bus = map_to_bus_pad(pt, data["bus_arcs_to"])
    qf_bus = map_to_bus_pad(qf, data["bus_arcs_fr"])
    qt_bus = map_to_bus_pad(qt, data["bus_arcs_to"])

    p_balance, q_balance = balance_residual_bus(
        pd_bus, qd_bus, pg_bus, qg_bus, vm,
        pf_bus, pt_bus, qf_bus, qt_bus,
        data["gs"], data["bs"],
    )

    return {
        "vm_lower": vm_lower, "vm_upper": vm_upper,
        "pg_lower": pg_lower, "pg_upper": pg_upper,
        "qg_lower": qg_lower, "qg_upper": qg_upper,
        "thrm_1": thrm_1, "thrm_2": thrm_2,
        "p_balance": p_balance, "q_balance": q_balance,
        "dva_lower": dva_lower, "dva_upper": dva_upper,
    }
