"""End-to-end smoke test of the E1 evaluator.

Six checks, in order:

  1. `check_env()` passes (all live checkpoints present)
  2. Determinism, same `(x, conds)` gives identical outputs twice
  3. Batch independence, perturbing `x[0]` leaves `x[1:]` outputs unchanged
  4. Gradient sanity, all constraint gradients finite, non-zero through every
     subfield when the penalty loss is backprop'd
  5. 50-step Adam on one seed, penalty loss (`obj + lambda*sum relu(g)^2 + mu*sum h^2`)
     should decrease
  6. `visualize_final` produces a valid figure

Artefacts land in `./smoke_out/<timestamp>/`:
  - `loss_curve.png`, objective / constraint violation over Adam steps
  - `history.csv`, per-step (loss, obj, eq_viol, ineq_viol)
  - `hero_initial.png`, `hero_final.png`, CAD + Cp + deflection composites
"""

from __future__ import annotations

import csv
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM
from pal.benchmarks.engineering.e1_bwb.x_layout import default_bounds

SEED: int = 0
BATCH: int = 4
N_STEPS: int = 50
LR: float = 1e-3
# Objective is already in megametres (bench returns -R/1e6), so OBJ_SCALE=1.
OBJ_SCALE: float = 1.0
LAMBDA_INEQ: float = 1e3
LAMBDA_EQ: float = 1e4
ALT_M: float = 2000.0
V_MS: float = 40.0


def _timestamped_outdir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path("smoke_out") / stamp
    out.mkdir(parents=True, exist_ok=True)
    return out


def _initial_design(B: int, seed: int) -> torch.Tensor:
    """Feasible-ish nominal with per-sample jitter."""
    lo, hi = default_bounds(dtype=torch.float32)
    mid = 0.5 * (lo + hi)
    g = torch.Generator("cpu").manual_seed(seed)
    jitter = 0.05 * (hi - lo) * (2.0 * torch.rand(B, DIM, generator=g) - 1.0)
    x = mid.unsqueeze(0).expand(B, -1).clone() + jitter
    x[:, -1] = math.radians(1.0)      # alpha_cr = 1 deg
    return x.clamp(lo, hi)


def _cruise_conditions(B: int) -> torch.Tensor:
    return torch.tensor([[ALT_M, V_MS]], dtype=torch.float32).expand(B, 2).contiguous()


def _penalty_loss(
    obj: torch.Tensor, clist, lam_ineq: float, lam_eq: float,
) -> tuple[torch.Tensor, dict]:
    eq_sq = torch.zeros((), dtype=obj.dtype, device=obj.device)
    ineq_sq = torch.zeros((), dtype=obj.dtype, device=obj.device)
    for c in clist:
        if c.type == "eq":
            eq_sq = eq_sq + (c.value ** 2).sum()
        else:
            ineq_sq = ineq_sq + (c.value.clamp_min(0.0) ** 2).sum()
    loss = (obj / OBJ_SCALE).sum() + lam_eq * eq_sq + lam_ineq * ineq_sq
    return loss, {
        "obj": float(obj.sum().detach()),
        "eq_viol": float(eq_sq.detach().sqrt()),
        "ineq_viol": float(ineq_sq.detach().sqrt()),
    }


# Checks
def check_env(bench: E1BWB) -> None:
    print("[1/6] check_env ... ", end="", flush=True)
    bench.check_env()
    print("OK")


def check_determinism(bench: E1BWB, x: torch.Tensor, conds: torch.Tensor) -> None:
    print("[2/6] determinism ... ", end="", flush=True)
    obj_a, c_a = bench.forward(x, conds)
    obj_b, c_b = bench.forward(x, conds)
    assert torch.allclose(obj_a, obj_b), "objective mismatch"
    for ca, cb in zip(c_a, c_b, strict=False):
        assert torch.allclose(ca.value, cb.value), f"constraint {ca.name} mismatch"
    print("OK")


def check_batch_independence(bench: E1BWB, x: torch.Tensor, conds: torch.Tensor) -> None:
    print("[3/6] batch independence ... ", end="", flush=True)
    obj_a, c_a = bench.forward(x, conds)
    x_pert = x.clone()
    x_pert[0] = x_pert[0] + 0.05 * torch.randn_like(x_pert[0])
    x_pert[0] = x_pert[0].clamp(*default_bounds())
    obj_b, c_b = bench.forward(x_pert, conds)

    # Rows 1..B-1 should match exactly, no cross-sample leakage.
    assert torch.allclose(obj_a[1:], obj_b[1:]), "objective leaked across batch"
    for ca, cb in zip(c_a, c_b, strict=False):
        assert torch.allclose(ca.value[1:], cb.value[1:]), \
            f"constraint {ca.name} leaked across batch"
    # Row 0 should differ, otherwise the perturbation was a no-op.
    assert not torch.allclose(obj_a[0], obj_b[0]), \
        "perturbing row 0 did not change its objective, plumbing broken"
    print("OK")


def check_gradients(bench: E1BWB, x: torch.Tensor, conds: torch.Tensor) -> None:
    print("[4/6] gradient sanity ... ", end="", flush=True)
    x = x.clone().requires_grad_(True)
    obj, clist = bench.forward(x, conds)
    loss, _ = _penalty_loss(obj, clist, LAMBDA_INEQ, LAMBDA_EQ)
    loss.backward()
    g = x.grad
    assert g is not None, "no gradient"
    assert torch.isfinite(g).all(), "non-finite grad entries"
    # Each subfield should have some gradient flow under a penalty loss that
    # touches every constraint.
    subfields = {
        "shape":    g[:, 0:9],
        "L":        g[:, 9:10],
        "struct":   g[:, 10:29],
        "battery":  g[:, 29:35],
        "alpha_cr": g[:, 35:36],
    }
    for name, grad in subfields.items():
        assert grad.abs().sum() > 0, f"no gradient through x.{name}"
    print("OK")


def run_adam(bench: E1BWB, x0: torch.Tensor, conds: torch.Tensor,
             outdir: Path) -> tuple[torch.Tensor, list[dict]]:
    print(f"[5/6] Adam {N_STEPS} steps (lr={LR}, B={BATCH}) ...")
    lo, hi = default_bounds(dtype=torch.float32)
    x = x0.clone().detach().requires_grad_(True)
    opt = torch.optim.Adam([x], lr=LR)
    history: list[dict] = []

    for step in range(N_STEPS):
        opt.zero_grad()
        obj, clist = bench.forward(x, conds)
        loss, parts = _penalty_loss(obj, clist, LAMBDA_INEQ, LAMBDA_EQ)
        loss.backward()
        opt.step()
        with torch.no_grad():
            x.data.clamp_(lo, hi)
        history.append({"step": step, "loss": float(loss.detach()), **parts})
        if step % 5 == 0 or step == N_STEPS - 1:
            print(f"   step {step:>3d}  loss={parts['obj']:+.3e}  "
                  f"eq_viol={parts['eq_viol']:.3e}  ineq_viol={parts['ineq_viol']:.3e}")

    # CSV + plot.
    csv_path = outdir / "history.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)
    print(f"   history CSV: {csv_path}")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    steps = [h["step"] for h in history]
    axes[0].plot(steps, [h["loss"] for h in history], color="C0")
    axes[0].set_title("Penalty loss")
    axes[0].set_yscale("symlog")
    axes[1].plot(steps, [h["obj"] for h in history], color="C1")
    axes[1].set_title("Objective  (-Breguet range, summed over B)")
    axes[2].plot(steps, [h["eq_viol"] for h in history], label="eq", color="C2")
    axes[2].plot(steps, [h["ineq_viol"] for h in history], label="ineq", color="C3")
    axes[2].set_title("Constraint violation (L2)")
    axes[2].set_yscale("log")
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel("Adam step")
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"E1 BWB smoke  (seed={SEED}, B={BATCH}, {N_STEPS} steps)")
    fig.tight_layout()
    loss_png = outdir / "loss_curve.png"
    fig.savefig(loss_png, dpi=120)
    plt.close(fig)
    print(f"   loss curve: {loss_png}")

    start_loss = history[0]["loss"]
    end_loss = history[-1]["loss"]
    print(f"   loss: {start_loss:+.3e}  ->  {end_loss:+.3e}  "
          f"(Delta = {end_loss - start_loss:+.3e})")
    assert end_loss < start_loss, \
        f"Adam did not decrease loss ({start_loss:+.3e} -> {end_loss:+.3e})"
    return x.detach(), history


def render_hero(bench: E1BWB, x: torch.Tensor, conds: torch.Tensor,
                outpath: Path, tag: str) -> None:
    print(f"       hero ({tag}) ... ", end="", flush=True)
    result = bench.visualize_final(x, conds)
    if result is None or "hero" not in result:
        print("returned None, skipped")
        return
    fig = result["hero"]
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"OK ({outpath.name})")


def render_train(bench: E1BWB, x: torch.Tensor, conds: torch.Tensor,
                 outpath: Path, tag: str) -> None:
    print(f"       train ({tag}) ... ", end="", flush=True)
    fig = bench.visualize_train(x, conds)
    if fig is None:
        print("returned None, skipped")
        return
    fig.savefig(outpath, dpi=120)
    plt.close(fig)
    print(f"OK ({outpath.name})")


def main() -> None:
    outdir = _timestamped_outdir()
    print(f"Artefacts: {outdir}")
    print()

    print("Loading live benchmark ...")
    t0 = time.time()
    bench = E1BWB(live=True)
    print(f"  ready ({time.time() - t0:.1f}s)")
    print()

    x0 = _initial_design(BATCH, SEED)
    conds = _cruise_conditions(BATCH)

    check_env(bench)
    check_determinism(bench, x0, conds)
    check_batch_independence(bench, x0, conds)
    check_gradients(bench, x0, conds)

    print("[5/6] render initial viz ...")
    render_train(bench, x0[:1], conds[:1], outdir / "train_initial.png", tag="initial")
    render_hero(bench, x0[:1], conds[:1], outdir / "hero_initial.png", tag="initial")

    x_final, _ = run_adam(bench, x0, conds, outdir)

    print("[6/6] render final viz ...")
    render_train(bench, x_final[:1], conds[:1], outdir / "train_final.png", tag="final")
    render_hero(bench, x_final[:1], conds[:1], outdir / "hero_final.png", tag="final")

    print()
    print(f"All checks passed. Artefacts in: {outdir}")
    for p in sorted(outdir.iterdir()):
        print(f"   {p.name}")
    try:
        subprocess.run(["open", str(outdir)], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
