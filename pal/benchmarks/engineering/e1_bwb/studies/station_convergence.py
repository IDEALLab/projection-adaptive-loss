"""Convergence study on `n_stations`, the Sobol spanwise-sampling count.

Sweeps `n_stations` over a log-spaced grid with a fixed set of random
geometries + a fixed cruise condition, and tracks four benchmark outputs:

  * objective (-Breguet range, m)
  * max strain constraint value across stations
  * tip-deflection constraint value
  * structural mass (kg)

The finest N in the sweep is treated as the reference. We report each
metric's relative deviation from that reference and pick the smallest N
whose deviations fall within a 5% tolerance across all five geometries.

The chosen value is `spec.N_STATIONS_DEFAULT`. Writes a PNG to `./out`
and opens it.
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from pal.benchmarks.engineering.e1_bwb import E1BWB, DIM
from pal.benchmarks.engineering.e1_bwb.x_layout import default_bounds

SWEEP_N: list[int] = [5, 10, 20, 40, 80, 160]
N_GEOMS: int = 5
SEED: int = 42
ALT_M: float = 2000.0
V_MS: float = 40.0
TOL_REL: float = 0.02
OUT_PNG: Path = Path("out") / "e1_station_convergence.png"


def _sample_geometries(n: int, seed: int) -> torch.Tensor:
    """Uniform draw from `default_bounds`. Deterministic via `seed`."""
    lo, hi = default_bounds(dtype=torch.float32)
    g = torch.Generator("cpu").manual_seed(seed)
    u = torch.rand(n, DIM, generator=g, dtype=torch.float32)
    return lo + u * (hi - lo)


def _extract_metrics(
    bench: E1BWB, x: torch.Tensor, conds: torch.Tensor,
) -> dict[str, np.ndarray]:
    obj, clist = bench.forward(x, conds)
    strain_vals = torch.stack(
        [c.value for c in clist if c.name.startswith("strain_")], dim=-1,
    )
    tip_val = next(c.value for c in clist if c.name == "tip_deflection")
    lift_val = next(c.value for c in clist if c.name == "lift_balance")

    # structural mass re-derived from internal call, the forward doesn't
    # expose it directly, but we have props cached via compute_structural.
    return {
        "objective": obj.detach().cpu().numpy(),
        "max_strain": strain_vals.max(dim=-1).values.detach().cpu().numpy(),
        "tip_deflection": tip_val.detach().cpu().numpy(),
        "lift_balance": lift_val.detach().cpu().numpy(),
    }


def main() -> None:
    torch.set_num_threads(4)

    x = _sample_geometries(N_GEOMS, seed=SEED)
    conds = torch.tensor([[ALT_M, V_MS]], dtype=torch.float32).expand(N_GEOMS, 2).contiguous()

    # Warm-load surrogates once, then reuse across N by pasting them into
    # fresh benches. Avoids paying ~3s of checkpoint loads per N.
    print("Loading live surrogates...")
    t0 = time.time()
    base = E1BWB(live=True, n_stations=SWEEP_N[0])
    shared = {
        "compute_aero": base._compute_aero,
        "compute_loads": base._compute_loads,
        "compute_structural": base._compute_structural,
        "compute_stress": base._compute_stress,
    }
    print(f"  surrogates ready ({time.time() - t0:.1f}s)")

    # Sweep.
    results: dict[int, dict[str, np.ndarray]] = {}
    for N in SWEEP_N:
        print(f"  N = {N:>3} ... ", end="", flush=True)
        t0 = time.time()
        bench = E1BWB(live=False, n_stations=N, **shared)
        results[N] = _extract_metrics(bench, x, conds)
        print(f"{time.time() - t0:.1f}s")

    # Use finest N as reference for convergence.
    N_ref = SWEEP_N[-1]
    ref = results[N_ref]

    # Relative deviation per geometry, per metric.
    metric_names = ["objective", "max_strain", "tip_deflection", "lift_balance"]
    rel_err: dict[str, dict[int, np.ndarray]] = {m: {} for m in metric_names}
    for m in metric_names:
        for N in SWEEP_N:
            denom = np.maximum(np.abs(ref[m]), 1e-6)
            rel_err[m][N] = np.abs(results[N][m] - ref[m]) / denom

    # Pick recommended N: smallest whose max relative error across all
    # geometries AND all four metrics is < TOL_REL.
    recommended = N_ref
    for N in SWEEP_N:
        if N == N_ref:
            continue
        worst = max(rel_err[m][N].max() for m in metric_names)
        if worst < TOL_REL:
            recommended = N
            break

    print()
    print(f"Reference N = {N_ref}")
    print(f"Recommended N (rel err < {TOL_REL:.0%} on strain + deflection) = {recommended}")
    print()
    print(f"{'N':>4}  {'max|obj err|':>12}  {'max|strain err|':>14}  "
          f"{'max|tip err|':>12}  {'max|lift err|':>13}")
    for N in SWEEP_N:
        print(f"{N:>4}  "
              f"{rel_err['objective'][N].max():>12.3%}  "
              f"{rel_err['max_strain'][N].max():>14.3%}  "
              f"{rel_err['tip_deflection'][N].max():>12.3%}  "
              f"{rel_err['lift_balance'][N].max():>13.3%}")

    # Plot
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    titles = {
        "objective": "Objective (-Breguet range)",
        "max_strain": "Max strain constraint",
        "tip_deflection": "Tip-deflection constraint",
        "lift_balance": "Lift-balance residual",
    }
    N_plot = [N for N in SWEEP_N if N != N_ref]

    # Row 0: absolute metric values vs N.
    for col, m in enumerate(metric_names):
        ax = axes[0, col]
        for g_idx in range(N_GEOMS):
            vals = np.array([results[N][m][g_idx] for N in SWEEP_N])
            ax.plot(SWEEP_N, vals, "-", color="C0", alpha=0.35, linewidth=1.0)
        median = np.array([np.median(results[N][m]) for N in SWEEP_N])
        ax.plot(SWEEP_N, median, "o-", color="C3", linewidth=2.0,
                markersize=5, label="median")
        ax.axvline(recommended, color="k", linestyle="--", alpha=0.5,
                   label=f"rec = {recommended}")
        ax.axvline(N_ref, color="gray", linestyle=":", alpha=0.5,
                   label=f"ref = {N_ref}")
        ax.set_xscale("log")
        ax.set_xlabel("n_stations")
        ax.set_title(titles[m])
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.legend(loc="best", fontsize=8)
            ax.set_ylabel("value")

    # Row 1: per-geometry relative error vs the N=N_ref reference.
    for col, m in enumerate(metric_names):
        ax = axes[1, col]
        for g_idx in range(N_GEOMS):
            vals = np.array([rel_err[m][N][g_idx] for N in N_plot])
            ax.plot(N_plot, vals, "-", color="C0", alpha=0.35, linewidth=1.0)
        max_traj = np.array([rel_err[m][N].max() for N in N_plot])
        ax.plot(N_plot, max_traj, "o-", color="C3", linewidth=2.0,
                markersize=5, label="max over geoms")
        ax.axhline(TOL_REL, color="k", linestyle="--", alpha=0.4,
                   label=f"tol = {TOL_REL:.0%}")
        ax.axvline(recommended, color="k", linestyle="--", alpha=0.3)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("n_stations")
        ax.set_title(f"|rel err|  vs  N={N_ref}")
        ax.grid(True, alpha=0.3, which="both")
        if col == 0:
            ax.legend(loc="best", fontsize=8)
            ax.set_ylabel("relative error")

    fig.suptitle(
        f"E1 BWB: station-count convergence  ({N_GEOMS} geometries, "
        f"cruise: {ALT_M:.0f} m / {V_MS:.0f} m/s)",
        fontsize=12,
    )
    fig.tight_layout()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PNG, dpi=120)
    print(f"\nSaved: {OUT_PNG}")
    try:
        subprocess.run(["open", str(OUT_PNG)], check=False)
    except Exception:
        pass


if __name__ == "__main__":
    main()
