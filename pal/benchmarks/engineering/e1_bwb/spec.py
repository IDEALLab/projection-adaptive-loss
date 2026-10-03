"""BenchmarkSpec factory for E1 BWB (1 eq + 2 ineqs, ineq `c <= 0` is feasible)."""

from __future__ import annotations

from typing import Any

import torch

from pal.benchmarks.base import BenchmarkSpec

from .x_layout import DIM, default_bounds

N_STATIONS_DEFAULT: int = 20
ZETA_DIM: int = 16


def constraint_layout(n_stations: int) -> tuple[list[str], list[str], int, int]:
    """Frozen constraint name+type ordering.

    Returns: (names, types, n_eq, n_ineq). Strain is aggregated to one slot, so K = 3.
    """
    del n_stations
    names = ["lift_balance", "strain_agg", "tip_deflection"]
    # lift_balance is an equality (lift == weight, |c| <= tol).
    types = ["eq", "ineq", "ineq"]
    return names, types, 1, 2


def make_spec(
    n_stations: int = N_STATIONS_DEFAULT,
    dtype: torch.dtype = torch.float32,
) -> Any:
    """Build the BenchmarkSpec. `n_stations` is the Sobol station count."""
    names, types, n_eq, n_ineq = constraint_layout(n_stations)
    lo, hi = default_bounds(dtype=dtype)

    return BenchmarkSpec(
        id="e1/bwb",
        family="e1",
        variant="bwb",
        dim=DIM,
        n_eq=n_eq,
        n_ineq=n_ineq,
        constraint_names=names,
        constraint_types=types,
        output_bounds=(lo, hi),
        condition_dim=2,           # (alt, V_cruise)
        zeta_dim=ZETA_DIM,
        tolerance=1e-3,
        tau=1e-3,
        cost="expensive",
        recommended_device="gpu",
        precision="fp32",
        recommended_batch_per_gpu={"A100-80GB": 4, "H100-80GB": 8},
        train_batch_size=32,
        n_eval_default=64,
        # The surrogates are only defined inside `output_bounds`.
        hard_output_box=True,
        # monotone_multiplier: lambda never decays below its peak.
        solver_hparams={
            "pal_loggap": {
                "max_decades": 5.0,
                "rate": 0.1,
                "monotone_multiplier": True,
            }
        },
        # std=1e-4 puts every output slot at the box midpoint at init.
        model_hparams={"output_init_std": 1e-4},
        notes=(
            f"BWB MDO: 36D design, 2D cond (alt, V), Sobol-{n_stations} spanwise "
            "stations. Objective = -Breguet range (electric). Lift balance is a "
            "min-lift inequality (lift >= weight); pitching-moment / 2.5g "
            "maneuver deferred to v2."
        ),
    )
