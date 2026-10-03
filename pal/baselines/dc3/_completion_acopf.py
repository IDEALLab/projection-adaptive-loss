"""Two-step ACOPF completion (Donti et al. 2021, Appendix C.3).

Step 1 solves part of the variables by Newton (Jacobian via ``torch.func.jacrev``),
Step 2 reads off the rest in closed form.

Partition (y = [pg, qg, vm, va] of length ``2*n_gen + 2*n_bus``):

  +---------------------------+--------------------------------------+
  | partial_vars (NN-pred.)   | pg at non-slack gens + vm at gen-    |
  |                           | attached buses                       |
  +---------------------------+--------------------------------------+
  | known_vars (pinned = 0)   | va at slack bus(es)                  |
  +---------------------------+--------------------------------------+
  | step1_vars (Newton)       | vm at load buses + va at non-slack   |
  |                           | buses                                |
  +---------------------------+--------------------------------------+
  | step2_vars (closed-form)  | pg at slack gen + qg at every gen    |
  +---------------------------+--------------------------------------+

Step 2 closed-form trick. After Step 1 converges (``h[step1_eqs] ~ 0`` with
``step2_vars`` held at 0), the power-balance formulae

    p_viol_b = pg_bus_b - pd_b - flows_b - gs_b * vm_b^2        (C.1g)
    q_viol_b = qg_bus_b - qd_b - flows_b + bs_b * vm_b^2        (C.1h)

are linear in ``pg_bus`` and ``qg_bus``. With ``pg_bus[slack] = 0``,
``p_viol[slack] = -(target pg_slack)``, so the closed form is

    pg_slack = -h[step2_eq_for_slack]   when evaluated at step2_vars = 0.

Same for ``qg[b]`` at each generator bus ``b``, which requires at most one
generator per bus (``verify_one_gen_per_bus``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor
from torch.func import jacrev, vmap

from pal.baselines.dc3._completion import CompletionDivergedError  # noqa: F401


@dataclass(frozen=True)
class ACOPFPartition:
    """Precomputed y-index and eq-index partitions for the two-step completion."""

    # y-index lists: pairwise disjoint, together cover range(ydim).
    partial_vars: list[int]    # NN-predicted
    known_vars: list[int]      # pinned at known_values
    step1_vars: list[int]      # Newton
    step2_vars: list[int]      # closed-form

    known_values: list[float]  # same length as known_vars

    # eq-index lists: partition of range(n_eq) into step1/step2.
    step1_eqs: list[int]
    step2_eqs: list[int]

    # Box bounds for the Step-2 read-off, aligned to step2_vars (it is unbounded otherwise).
    step2_lo: list[float]
    step2_hi: list[float]

    vm_start: int
    va_start: int

    # MATPOWER base-case warm start aligned to step1_vars; None = flat 1.0/0.0.
    step1_warm: list[float] | None

    ydim: int
    n_eq: int

    def invariants_ok(self) -> bool:
        all_y = (
            list(self.partial_vars)
            + list(self.known_vars)
            + list(self.step1_vars)
            + list(self.step2_vars)
        )
        all_eqs = list(self.step1_eqs) + list(self.step2_eqs)
        return (
            len(set(all_y)) == len(all_y) == self.ydim
            and len(set(all_eqs)) == len(all_eqs) == self.n_eq
            and len(self.step1_vars) == len(self.step1_eqs)
            and len(self.step2_vars) == len(self.step2_eqs)
            and len(self.known_vars) == len(self.known_values)
            and len(self.step2_lo) == len(self.step2_hi) == len(self.step2_vars)
            and (self.step1_warm is None or len(self.step1_warm) == len(self.step1_vars))
        )


def verify_one_gen_per_bus(bench) -> None:
    """Assert at most one real generator per bus (needed by the Step-2 read-off)."""
    data = bench._data
    bus_gens = data["bus_gens"].cpu().numpy()
    n_gen = int(data["G"].item())
    offenders: list[tuple[int, list[int]]] = []
    for bus_idx in range(bus_gens.shape[0]):
        real = [int(g) for g in bus_gens[bus_idx] if int(g) < n_gen]
        if len(real) > 1:
            offenders.append((bus_idx, real))
    if offenders:
        raise ValueError(
            f"acopf_two_step_complete: bench {bench.spec.id!r} has "
            f"{len(offenders)} bus(es) with multiple generators; Step 2 "
            f"closed-form is not applicable. Example: bus {offenders[0][0]} "
            f"has gens {offenders[0][1]}. Fall back to generic Newton."
        )


def build_partition(bench) -> ACOPFPartition:
    """Build the ACOPF two-step partition from a e3 bench instance."""
    verify_one_gen_per_bus(bench)

    n_bus = bench._n_bus
    n_gen = bench._n_gen
    n_ext = bench.n_ext_grid
    pg_start = bench.pg_start_yidx
    qg_start = bench.qg_start_yidx
    vm_start = bench.vm_start_yidx
    va_start = bench.va_start_yidx

    slack_bus = set(bench.slack_bus_idx)
    spv_bus = set(bench.spv_bus_idx)
    load_bus = set(range(n_bus)) - spv_bus          # D = B \ (R union G)
    non_slack_bus = set(range(n_bus)) - slack_bus   # B \ R

    # Ext-grid gens occupy slots [0, n_ext_grid), real gens the rest, both sorted by bus.
    bus_of_gen = list(bench.slack_bus_idx) + list(bench.pv_bus_idx)
    assert len(bus_of_gen) == n_gen, (
        f"gen->bus derivation mismatch: "
        f"|slack|+|pv|={n_ext}+{len(bench.pv_bus_idx)} vs n_gen={n_gen}"
    )

    partial_vars = (
        [pg_start + g for g in bench.pv_gen_idx]
        + [vm_start + b for b in bench.spv_bus_idx]
    )
    known_vars = [va_start + b for b in bench.slack_bus_idx]
    known_values = [0.0] * len(known_vars)

    step1_vars = (
        sorted(vm_start + b for b in load_bus)
        + sorted(va_start + b for b in non_slack_bus)
    )

    # step2_vars[i] pairs with step2_eqs[i]: pg_slack with p-balance at the slack
    # bus, then qg[g] with q-balance at bus_of_gen[g].
    p_offset = 0
    q_offset = n_bus
    step2_vars = (
        [pg_start + g for g in bench.slack_gen_idx]
        + [qg_start + g for g in range(n_gen)]
    )
    step2_eqs = (
        [p_offset + bus_of_gen[g] for g in bench.slack_gen_idx]
        + [q_offset + bus_of_gen[g] for g in range(n_gen)]
    )

    pgmin_t = bench._data["pgmin"]
    pgmax_t = bench._data["pgmax"]
    qgmin_t = bench._data["qgmin"]
    qgmax_t = bench._data["qgmax"]
    step2_lo = (
        [float(pgmin_t[g]) for g in bench.slack_gen_idx]
        + [float(qgmin_t[g]) for g in range(n_gen)]
    )
    step2_hi = (
        [float(pgmax_t[g]) for g in bench.slack_gen_idx]
        + [float(qgmax_t[g]) for g in range(n_gen)]
    )

    step2_eqs_set = set(step2_eqs)
    step1_eqs = [e for e in range(2 * n_bus) if e not in step2_eqs_set]

    vm_init_t = bench._data.get("vm_init")
    va_init_t = bench._data.get("va_init")
    if vm_init_t is not None and va_init_t is not None:
        step1_warm: list[float] | None = []
        for yidx in step1_vars:
            if yidx >= va_start:
                step1_warm.append(float(va_init_t[yidx - va_start]))
            else:
                step1_warm.append(float(vm_init_t[yidx - vm_start]))
    else:
        step1_warm = None

    ydim = bench.spec.dim
    n_eq = bench.spec.n_eq

    part = ACOPFPartition(
        partial_vars=partial_vars,
        known_vars=known_vars,
        step1_vars=step1_vars,
        step2_vars=step2_vars,
        known_values=known_values,
        step1_eqs=step1_eqs,
        step2_eqs=step2_eqs,
        step2_lo=step2_lo,
        step2_hi=step2_hi,
        vm_start=vm_start,
        va_start=va_start,
        step1_warm=step1_warm,
        ydim=ydim,
        n_eq=n_eq,
    )
    if not part.invariants_ok():
        raise AssertionError(
            f"build_partition({bench.spec.id!r}): invariants violated, "
            f"|p|+|k|+|s1|+|s2|={len(partial_vars)+len(known_vars)+len(step1_vars)+len(step2_vars)} "
            f"vs ydim={ydim}; |s1_eqs|+|s2_eqs|={len(step1_eqs)+len(step2_eqs)} vs n_eq={n_eq}."
        )
    return part


def _scatter_per_sample(
    Z_single: Tensor,
    step1_single: Tensor,
    step2_single: Tensor,
    pv_t: Tensor, kv_t: Tensor, s1v_t: Tensor, s2v_t: Tensor,
    known_vals_t: Tensor,
    ydim: int,
) -> Tensor:
    """Assemble a single-sample full y-vector from the four y-slot groups (vmap-safe)."""
    y = torch.zeros(ydim, dtype=Z_single.dtype, device=Z_single.device)
    y = y.index_copy(0, pv_t, Z_single)
    y = y.index_copy(0, kv_t, known_vals_t.to(dtype=Z_single.dtype))
    y = y.index_copy(0, s1v_t, step1_single)
    y = y.index_copy(0, s2v_t, step2_single)
    return y


def acopf_two_step_complete(
    eq_resid_per_sample_fn: Callable[[Tensor, Tensor], Tensor],
    X: Tensor,
    Z: Tensor,
    partition: ACOPFPartition,
    *,
    max_iter: int = 50,
    tol: float = 1e-6,
    reg: float = 1e-8,
    warm_vm: float = 1.0,
    warm_va: float = 0.0,
    diag: dict | None = None,
    accept_floor: float | None = None,
    in_loop_cap: float | None = 1e3,
) -> Tensor:
    """Two-step ACOPF completion (Appendix C.3).

    Args:
        eq_resid_per_sample_fn: ``(y[ydim], x[cond_dim]) -> [n_eq]``, PAL's
            standard per-sample residual. See ``data_shim.make_eq_resid_per_sample``.
        X: Conditions ``[B, cond_dim]``.
        Z: NN-predicted partial outputs ``[B, n_partial]`` (pg_pv + vm_spv).
        partition: Precomputed ``ACOPFPartition`` (from ``build_partition``).
        max_iter: Step 1 Newton budget. Default 50 matches upstream PFFunction.
        tol: Step 1 convergence threshold on ``||h[step1_eqs]||_inf``.
        reg: Tikhonov for Step 1 solve.
        warm_vm, warm_va: Physical warm start values for load buses / non-slack
            angles (MATPOWER base-case convention).

    Returns:
        Full ``y`` ``[B, ydim]``. Step 2 zeros its own eqs by construction, the
        Step 1 residual must reach ``tol`` or the accept floor.

    Raises:
        CompletionDivergedError: Non-finite residual / Newton step / Step 2
            read-off, or Step 1 exhausted ``max_iter`` without reaching the
            feasibility floor.
    """
    B = Z.shape[0]
    device = Z.device
    dtype = Z.dtype
    ydim = partition.ydim
    n_s1 = len(partition.step1_vars)
    n_s2 = len(partition.step2_vars)

    if Z.shape[1] != len(partition.partial_vars):
        raise ValueError(
            f"Z.shape[1]={Z.shape[1]} does not match "
            f"|partial_vars|={len(partition.partial_vars)}"
        )

    pv_t = torch.as_tensor(partition.partial_vars, dtype=torch.long, device=device)
    kv_t = torch.as_tensor(partition.known_vars, dtype=torch.long, device=device)
    s1v_t = torch.as_tensor(partition.step1_vars, dtype=torch.long, device=device)
    s2v_t = torch.as_tensor(partition.step2_vars, dtype=torch.long, device=device)
    known_vals_t = torch.as_tensor(partition.known_values, dtype=dtype, device=device)
    s1e_t = torch.as_tensor(partition.step1_eqs, dtype=torch.long, device=device)
    s2e_t = torch.as_tensor(partition.step2_eqs, dtype=torch.long, device=device)

    if partition.step1_warm is not None:
        s1_warm = torch.tensor(partition.step1_warm, dtype=dtype, device=device)
    else:
        s1_warm = torch.tensor(
            [warm_va if yidx >= partition.va_start else warm_vm
             for yidx in partition.step1_vars],
            dtype=dtype, device=device,
        )

    step1_vals = s1_warm.unsqueeze(0).expand(B, -1).contiguous()
    step2_vals = torch.zeros(B, n_s2, device=device, dtype=dtype)

    eye = torch.eye(n_s1, device=device, dtype=dtype).expand(B, -1, -1)

    # h[step1_eqs] as a function of the step1 values, with step2 held at 0.
    def f_step1_single(step1_single, Z_single, step2_single, x_single):
        y = _scatter_per_sample(
            Z_single, step1_single, step2_single,
            pv_t, kv_t, s1v_t, s2v_t, known_vals_t, ydim,
        )
        h = eq_resid_per_sample_fn(y, x_single)
        return h.index_select(0, s1e_t)

    jac_fn = vmap(jacrev(f_step1_single, argnums=0))

    step1_converged = False
    h_inf = float("inf")
    for it in range(max_iter):
        h_sub = vmap(f_step1_single)(step1_vals, Z, step2_vals, X)  # [B, n_s1]

        if not torch.isfinite(h_sub).all():
            raise CompletionDivergedError(
                f"Newton diverged (acopf step 1) at iter {it}: non-finite residual"
            )
        h_inf = float(h_sub.abs().max().item())
        if in_loop_cap is not None and h_inf > in_loop_cap:
            raise CompletionDivergedError(
                f"Newton diverged (acopf step 1) at iter {it}: "
                f"||h_step1||_inf={h_inf:.3e}"
            )
        if h_inf < tol:
            step1_converged = True
            break

        J = jac_fn(step1_vals, Z, step2_vals, X)  # [B, n_s1, n_s1]
        delta = torch.linalg.solve(
            J + reg * eye, h_sub.unsqueeze(-1)
        ).squeeze(-1)

        if not torch.isfinite(delta).all():
            raise CompletionDivergedError(
                f"Newton diverged (acopf step 1) at iter {it}: non-finite step"
            )

        # Backtracking on ||h||_inf; if all 5 halvings fail the last candidate is kept.
        alpha = 1.0
        candidate = step1_vals - delta
        for _ in range(5):
            h_candidate = vmap(f_step1_single)(candidate, Z, step2_vals, X)
            if (
                torch.isfinite(h_candidate).all()
                and float(h_candidate.abs().max().item()) <= h_inf
            ):
                break
            alpha *= 0.5
            candidate = step1_vals - alpha * delta

        step1_vals = candidate

    # Accept a non-converged Step 1 only below the floor: its residual feeds Step 2.
    h_final_inf: float
    if not step1_converged:
        h_final = vmap(f_step1_single)(step1_vals, Z, step2_vals, X)
        if not torch.isfinite(h_final).all():
            raise CompletionDivergedError(
                f"Newton diverged (acopf step 1): non-finite residual after "
                f"max_iter={max_iter}"
            )
        h_final_inf = float(h_final.abs().max().item())
        af = accept_floor if accept_floor is not None else max(100.0 * tol, 1e-4)
        if h_final_inf > af:
            raise CompletionDivergedError(
                f"Newton diverged (acopf step 1): exhausted max_iter="
                f"{max_iter} with ||h_step1||_inf={h_final_inf:.3e} "
                f"(accept_floor={af:.1e})"
            )
    else:
        h_final_inf = h_inf  # from last loop iteration, < tol

    # Step 2 closed-form: with step2_vars = 0, the step2 eqs read off target values.
    def f_full_single(Z_single, step1_single, step2_single, x_single):
        y = _scatter_per_sample(
            Z_single, step1_single, step2_single,
            pv_t, kv_t, s1v_t, s2v_t, known_vals_t, ydim,
        )
        return eq_resid_per_sample_fn(y, x_single)

    h_full = vmap(f_full_single)(Z, step1_vals, step2_vals, X)  # [B, n_eq]
    step2_target = -h_full.index_select(1, s2e_t)  # [B, n_s2]

    if not torch.isfinite(step2_target).all():
        raise CompletionDivergedError(
            "Newton diverged (acopf step 2): closed-form produced non-finite values"
        )

    # Clip the read-off to the gen P/Q box: this reintroduces eq residual but keeps y physical.
    step2_lo_t = torch.as_tensor(partition.step2_lo, dtype=dtype, device=device)
    step2_hi_t = torch.as_tensor(partition.step2_hi, dtype=dtype, device=device)
    step2_clipped = torch.clamp(step2_target, min=step2_lo_t, max=step2_hi_t)
    step2_clip_frac = float(
        (step2_clipped != step2_target).float().mean().item()
    )
    step2_vals = step2_clipped

    if diag is not None:
        diag["step1_h_inf"] = h_final_inf
        diag["step1_converged"] = step1_converged
        diag["step2_clip_frac"] = step2_clip_frac

    def assemble_single(Z_s, step1_s, step2_s):
        return _scatter_per_sample(
            Z_s, step1_s, step2_s,
            pv_t, kv_t, s1v_t, s2v_t, known_vals_t, ydim,
        )

    return vmap(assemble_single)(Z, step1_vals, step2_vals)
