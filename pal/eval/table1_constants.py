"""Frozen constants for the paper's Table-1 metrics and the BO objective.

Frozen literals: C_b is the worst per-method seed-mean ``obj_mean_post`` on bench b
over the Table-1 runs, i.e. the worst mean optimality gap since f* = 0.
"""

from __future__ import annotations

TABLE1_METHODS: tuple[str, ...] = (
    "pal_loggap",
    "alm",
    "alm_bolton",
    "dc3",
    "fsnet",
    "enforce_orig",
    "snarenet",
)

# s1-s6, all with analytic f* = 0.
TABLE1_BENCHES: tuple[str, ...] = (
    "s1_sphere_track",
    "s2_active_set_switch",
    "s3_illcond_tube",
    "s4_qv_coupling",
    "s5_overdetermined",
    "s6_redundant_ineq",
)

# An "ok" run whose post-repair objective exceeds this is scored worst-case.
OBJ_FAIL_THRESHOLD: float = 100.0

# Per-bench C_b, with the worst method noted.
BENCH_OBJ_CONSTANTS: dict[str, float] = {
    "s1_sphere_track": 8.664392995404706,       # worst = alm_bolton
    "s2_active_set_switch": 0.2784000475774519,  # worst = alm_bolton
    "s3_illcond_tube": 0.054703855286788894,     # worst = snarenet
    "s4_qv_coupling": 9.818147963099182,         # worst = alm_bolton
    "s5_overdetermined": 4.99016385064539,       # worst = fsnet
    "s6_redundant_ineq": 2.6822263447567822,     # worst = snarenet
}

# Structural inapplicability (dc3 needs n_eq <= dim), not an empirical result.
STRUCTURAL_EXCLUSIONS: frozenset[tuple[str, str]] = frozenset(
    {("dc3", "s5_overdetermined")}
)

# Tie-breakers in S = L1 - EPS1*L2 - EPS2*L3; they never overturn an L1 gap.
EPS1: float = 1e-4
EPS2: float = 1e-7

# L3 maps max violation through a clamped log10 window: 0 at tau_c, 1 at tau_c * 10**decades.
VIOL_TAU_C: float = 1e-4
VIOL_WINDOW_DECADES: float = 4.0
