"""Per-variant breaking-point analysis of finished curvature dial runs.

Covers the curvature_hinge, curvature_sine and curvature_warp families.

Reloads `model.pt` and re-runs `solver.predict()` (eval_rows.parquet stores no
`y`), then prints markdown tables. `--campaign` aggregates arms x variants x seeds.

Usage:
    python scripts/analyze_curvature.py --runs-root runs/sweep1 [--method pal_loggap]
        [--seed 0] [--device cpu] [--out curvature_hinge_table.md]
    python scripts/analyze_curvature.py --runs-root runs/campaign --campaign \
        --expect-seeds 0-9 --out-json campaign.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Resolve `pal` from this checkout, not from the installed package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from pal.benchmarks import get as get_benchmark  # noqa: E402
from pal.benchmarks.synthetic.curvature_hinge import (  # noqa: E402
    KAPPA_BY_VARIANT as CURVATURE_HINGE_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_sine import (  # noqa: E402
    KAPPA_BY_VARIANT as CURVATURE_SINE_KAPPA_BY_VARIANT,
)
from pal.benchmarks.synthetic.curvature_warp import (  # noqa: E402
    KAPPA_BY_VARIANT as CURVATURE_WARP_KAPPA_BY_VARIANT,
)
from pal.runner.cli import _build_cfg_from_hparams, _build_solver  # noqa: E402
from pal.solvers.base import TrainResult  # noqa: E402

CAP_HALF_WIDTHS = 3.0  # curvature_hinge cap stratum: |t*|/r <= 3
CAP_K_FRAC = 0.5  # curvature_sine crest-cap analog: local K(t*) >= kappa/2
DELTA_ARM_BINS = [1e-3, 1e-2, 1e-1]  # also used for Delta_fold (curvature_sine)
K_REL_BINS = [0.1, CAP_K_FRAC]  # curvature_sine local-curvature strata, in units of kappa
FOLD_COUNT_BINS = [1, 2, 3]  # curvature_sine: distinct fold candidates (last bin = >= 4)
DASH = "-"
GRAD_SHARE_CON_KEY = "grad_norm_con"
GRAD_SHARE_TOT_KEY = "grad_norm_tot"
#: trailing logged points averaged (10-epoch cadence, so ~the last 50 epochs).
GRAD_SHARE_TAIL = 5
NA = "n/a"

PREFIX_BY_FAMILY = {
    "curvature_hinge": "curvature_hinge_",
    "curvature_sine": "curvature_sine_",
    "curvature_warp": "curvature_warp_",
}
KAPPA_BY_FAMILY = {
    "curvature_hinge": CURVATURE_HINGE_KAPPA_BY_VARIANT,
    "curvature_sine": CURVATURE_SINE_KAPPA_BY_VARIANT,
    "curvature_warp": CURVATURE_WARP_KAPPA_BY_VARIANT,
}
#: families whose ambiguity unit is the fold rather than the arm.
FOLD_FAMILIES = ("curvature_sine", "curvature_warp")


def _family_of(bench_id: str) -> str | None:
    """Dial family of a benchmark id, or None if it is not a dial bench."""
    for family, prefix in PREFIX_BY_FAMILY.items():
        if bench_id.startswith(prefix):
            return family
    return None


@dataclass
class RunOutputs:
    """One run's rehydrated predictions plus the recorded per-query rows."""

    run_dir: Path
    family: str  # "curvature_hinge" (hinge) | "curvature_sine" (sine ripple)
    variant: str
    kappa: float
    seed: int
    method: str
    bench: Any
    x: torch.Tensor  # [N, 8] float64 eval conditions
    y_raw: torch.Tensor  # [N, 8] float64 pre-repair prediction
    y_post: torch.Tensor  # [N, 8] float64 repaired prediction
    rows: pd.DataFrame  # eval_rows.parquet, sorted by query_idx
    tolerance: float
    obj_check: float  # max |obj_post(recomputed) - obj_post(recorded)|
    grad_share: float | None  # mean con/tot over the last logged points; None = absent


def _discover(
    runs_root: Path, method: str | None, seed: int | None
) -> dict[str, list[Path]]:
    """Dial runs under `runs_root`, grouped by family in `PREFIX_BY_FAMILY` order."""
    out: dict[str, list[Path]] = {}
    for cfg_path in sorted(runs_root.glob("*/config.json")):
        cfg = json.loads(cfg_path.read_text())
        family = _family_of(str(cfg.get("benchmark_id", "")))
        if family is None:
            continue
        if method is not None and cfg.get("method") != method:
            continue
        if seed is not None and int(cfg.get("seed", -1)) != seed:
            continue
        if not (cfg_path.parent / "model.pt").exists():
            print(f"[skip] {cfg_path.parent.name}: no model.pt")
            continue
        if not (cfg_path.parent / "eval_rows.parquet").exists():
            print(f"[skip] {cfg_path.parent.name}: no eval_rows.parquet")
            continue
        out.setdefault(family, []).append(cfg_path.parent)
    return {f: out[f] for f in PREFIX_BY_FAMILY if f in out}


def _final_grad_share(run_dir: Path, tail: int = GRAD_SHARE_TAIL) -> float | None:
    """Mean `grad_norm_con / grad_norm_tot` over the last `tail` logged points, None if absent."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    shares: list[float] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if GRAD_SHARE_CON_KEY not in rec or GRAD_SHARE_TOT_KEY not in rec:
                continue
            con = float(rec[GRAD_SHARE_CON_KEY])
            tot = float(rec[GRAD_SHARE_TOT_KEY])
            if not (tot > 0.0) or con != con:
                continue
            shares.append(con / tot)
    if not shares:
        return None
    tail_vals = shares[-tail:]
    return sum(tail_vals) / len(tail_vals)


def _load_run(run_dir: Path, device: str) -> RunOutputs:
    cfg = json.loads((run_dir / "config.json").read_text())
    bench_id = cfg["benchmark_id"]
    family = _family_of(bench_id)
    assert family is not None, bench_id
    variant = bench_id.removeprefix(PREFIX_BY_FAMILY[family])
    seed = int(cfg["seed"])
    method = cfg["method"]
    n_eval = int(cfg["n_eval_effective"])

    # Predict at the training precision; queries are drawn at native dtype and cast after.
    from pal.runner.cli import _cast_queries, _default_dtype

    precision = cfg.get("precision", "fp32")
    solver_cfg = _build_cfg_from_hparams(method, cfg.get("hparams") or {})
    solver_cfg.device = device
    bench = get_benchmark(bench_id, device=device)
    queries = bench.eval_queries(seed, n=n_eval)

    with _default_dtype(precision):
        solver = _build_solver(method, solver_cfg, bench_id=bench_id)
        queries = _cast_queries(queries, precision)
        train_result = TrainResult(
            solver_name=method,
            train_wall_time_s=0.0,
            n_restarts=1,
            model_state=torch.load(
                run_dir / "model.pt", map_location=device, weights_only=True
            ),
        )
        pred = solver.predict(bench, queries, train_result, logger=None)

    rows = pd.read_parquet(run_dir / "eval_rows.parquet").sort_values("query_idx")
    if len(rows) != len(queries):
        raise SystemExit(
            f"{run_dir.name}: eval_rows has {len(rows)} rows but the eval set has "
            f"{len(queries)} queries, per-query alignment is impossible"
        )
    obj_post = bench.objective(pred.post.detach(), queries.conditions).detach()
    obj_check = float((obj_post.double() - torch.tensor(rows["obj_post"].to_numpy())).abs().max())

    return RunOutputs(
        run_dir=run_dir,
        family=family,
        variant=variant,
        kappa=KAPPA_BY_FAMILY[family][variant],
        seed=seed,
        method=method,
        bench=bench,
        x=queries.conditions.detach().double(),
        y_raw=pred.raw.detach().double(),
        y_post=pred.post.detach().double(),
        rows=rows,
        tolerance=float(cfg["benchmark_spec"]["tolerance"]),
        obj_check=obj_check,
        grad_share=_final_grad_share(run_dir),
    )


def _q(t: torch.Tensor, p: float) -> float:
    if t.numel() == 0:
        return float("nan")
    return float(torch.quantile(t.double(), p))


def _metrics(run: RunOutputs, tie_abs: float, tie_rel: float) -> dict[str, Any]:
    bench = run.bench
    kappa = bench.kappa
    x, y_raw, y_post = run.x, run.y_raw, run.y_post
    n = x.shape[0]

    ref = bench.reference_solution(x)  # exact f*, t*, Delta_arm / Delta_fold per query
    f_star = ref.f_star
    feasible = torch.tensor(run.rows["feasible_post"].to_numpy(), dtype=torch.bool)
    obj_post = bench.objective(y_post, x)

    # curvature_warp warps the objective, so distances use the Euclidean `geometric_reference`.
    geom_ref = getattr(bench, "geometric_reference", bench.reference_solution)
    ref_raw = geom_ref(y_raw)
    d = ref_raw.f_star.clamp(min=0.0).sqrt()
    sign = torch.sign(bench.g(y_raw))
    d_signed = d * sign
    k_local = bench.curvature(ref_raw.t_star)

    # Optimality gap among feasible only, snap gap over all queries.
    gap = obj_post[feasible] - f_star[feasible]
    y_snap = bench.snap(y_post)
    gap_snap = bench.objective(y_snap, x) - f_star

    out: dict[str, Any] = {
        "variant": run.variant,
        "kappa": kappa,
        "n": n,
        "feas": float(feasible.double().mean()),
        "n_feas": int(feasible.sum()),
        "gap_p50": _q(gap, 0.5),
        "gap_p90": _q(gap, 0.9),
        "gap_mean": float(gap.double().mean()) if gap.numel() else float("nan"),
        "grad_share": NA if run.grad_share is None else run.grad_share,
        "snap_p50": _q(gap_snap, 0.5),
        "snap_p90": _q(gap_snap, 0.9),
        "snap_min": float(gap_snap.min()) if n else float("nan"),
        "obj_check": run.obj_check,
        "affine_anchor": bench.is_affine_anchor,
    }
    if run.family == "curvature_hinge":
        out["r"] = bench.r
    else:
        out["omega"] = bench.omega
    if bench.is_affine_anchor:
        return out

    out.update(
        {
            "d_p50": _q(d, 0.5),
            "d_p90": _q(d, 0.9),
            "kd_p50": _q(kappa * d, 0.5),
            "kd_p90": _q(kappa * d, 0.9),
            "kd_p50_signed": _q(k_local * d_signed, 0.5),
            "kd_p90_signed": _q(k_local * d_signed, 0.9),
        }
    )

    # Eligible: two branch candidates, optimum unique by the margin, outside the cap.
    margin = tie_abs + tie_rel * f_star
    y_used = torch.where(feasible.unsqueeze(-1), y_post, y_snap)
    t_used = geom_ref(y_used).t_star
    if run.family == "curvature_hinge":
        r = bench.r
        two = ref.has_two_arms
        delta = ref.delta_arm
        outside_cap = ref.t_star.abs() > CAP_HALF_WIDTHS * r
        branch_model, branch_ref = torch.sign(t_used), torch.sign(ref.t_star)
        rate_key, bin_key = "wrong_arm", "n_arm_bin"
    else:
        two = ref.has_two_folds
        delta = ref.delta_fold
        outside_cap = bench.curvature(ref.t_star) < CAP_K_FRAC * kappa
        branch_model, branch_ref = bench.fold(t_used), ref.fold_star
        rate_key, bin_key = "wrong_fold", "n_fold_bin"
    ambiguous = two & (delta <= margin)
    eligible = two & (delta > margin) & outside_cap
    wrong = eligible & (branch_model != branch_ref)
    n_elig = int(eligible.sum())
    out.update(
        {
            rate_key: (float(wrong.sum()) / n_elig) if n_elig else float("nan"),
            "n_elig": n_elig,
            "n_ambig": int(ambiguous.sum()),
            "n_via_snap": int((eligible & ~feasible).sum()),
        }
    )

    if run.family == "curvature_hinge":
        g_x = bench.g(x)
        out.update(
            {
                "n_cap": int((ref.t_star.abs() <= CAP_HALF_WIDTHS * r).sum()),
                "n_exterior": int((g_x < 0).sum()),
                "n_wedge": int((g_x > 0).sum()),
                "n_arm_none": int((~two).sum()),
            }
        )
    else:
        k_star = bench.curvature(ref.t_star)
        out["n_cap"] = int((k_star >= CAP_K_FRAC * kappa).sum())
        for i, cnt in enumerate(FOLD_COUNT_BINS):
            out[f"n_nfold_bin{i}"] = int((ref.n_folds == cnt).sum())
        out[f"n_nfold_bin{len(FOLD_COUNT_BINS)}"] = int(
            (ref.n_folds > FOLD_COUNT_BINS[-1]).sum()
        )
        k_edges = [0.0, *K_REL_BINS, float("inf")]
        for i in range(len(k_edges) - 1):
            lo_k, hi_k = k_edges[i] * kappa, k_edges[i + 1] * kappa
            out[f"n_k_bin{i}"] = int(((k_star >= lo_k) & (k_star < hi_k)).sum())
    edges = [0.0, *DELTA_ARM_BINS, float("inf")]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        sel = two & (delta >= lo) & (delta < hi)
        out[f"{bin_key}{i}"] = int(sel.sum())
    return out


def _fmt(v: Any, spec: str = "{:.3e}") -> str:
    if v is None:
        return DASH
    if isinstance(v, float):
        if v != v:  # NaN
            return DASH
        return spec.format(v)
    return str(v)


def _table(rows: list[dict[str, Any]], title: str, cols: list[tuple[str, str, str]]) -> str:
    head = "| " + " | ".join(c[0] for c in cols) + " |"
    rule = "|" + "|".join("---" for _ in cols) + "|"
    body = []
    for row in rows:
        cells = []
        for _label, key, spec in cols:
            cells.append(_fmt(row.get(key), spec) if key in row else DASH)
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([f"### {title}", "", head, rule, *body, ""])


def _render(
    rows: list[dict[str, Any]], family: str, tie_abs: float, tie_rel: float
) -> str:
    if family in FOLD_FAMILIES:
        return _render_folds(rows, tie_abs, tie_rel, family)
    rows = sorted(rows, key=lambda rr: (rr["kappa"], rr["variant"]))
    core = _table(
        rows,
        "curvature_hinge core: feasibility, pre-repair distance, optimality gap",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("n", "n", "{}"),
            ("feas", "feas", "{:.3f}"),
            ("n_feas", "n_feas", "{}"),
            ("d p50", "d_p50", "{:.3e}"),
            ("d p90", "d_p90", "{:.3e}"),
            ("k*d p50", "kd_p50", "{:.3e}"),
            ("k*d p90", "kd_p90", "{:.3e}"),
            ("gap p50", "gap_p50", "{:+.3e}"),
            ("gap p90", "gap_p90", "{:+.3e}"),
            ("gap mean", "gap_mean", "{:+.3e}"),
            ("grad share", "grad_share", "{:.3f}"),
        ],
    )
    mech = _table(
        rows,
        "curvature_hinge mechanism: local curvature x signed distance, snap gap, wrong arm",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("K(t*)*d_sgn p50", "kd_p50_signed", "{:+.3e}"),
            ("K(t*)*d_sgn p90", "kd_p90_signed", "{:+.3e}"),
            ("snap gap p50", "snap_p50", "{:+.3e}"),
            ("snap gap p90", "snap_p90", "{:+.3e}"),
            ("snap gap min", "snap_min", "{:+.3e}"),
            ("wrong-arm", "wrong_arm", "{:.3f}"),
            ("n_elig", "n_elig", "{}"),
            ("n_ambig", "n_ambig", "{}"),
            ("n_snap_used", "n_via_snap", "{}"),
        ],
    )
    strata = _table(
        rows,
        "curvature_hinge strata bin counts",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("cap t*/r<=3", "n_cap", "{}"),
            ("exterior", "n_exterior", "{}"),
            ("wedge", "n_wedge", "{}"),
            ("1 arm", "n_arm_none", "{}"),
            ("dArm<1e-3", "n_arm_bin0", "{}"),
            ("1e-3..1e-2", "n_arm_bin1", "{}"),
            ("1e-2..1e-1", "n_arm_bin2", "{}"),
            (">=1e-1", "n_arm_bin3", "{}"),
        ],
    )
    notes = [
        "### notes",
        "",
        f"- ambiguity margin: `Delta_arm > {tie_abs:g} + {tie_rel:g} * f*` "
        "(else counted as ambiguous, not wrong-arm).",
        f"- cap stratum half-width: `|t*|/r <= {CAP_HALF_WIDTHS:g}`; "
        "exterior = `g(x) < 0` (unique projection), wedge = `g(x) > 0`.",
        "- `d` is the EXACT Euclidean distance of the pre-repair prediction to "
        "`{g=0}` from the certified reference solver.",
        "- `gap` is over feasible outputs only; tiny negatives are expected "
        "(approximate feasibility can undershoot `f*`). `gap mean` is the "
        "feasible-only mean of the same quantity.",
        f"- `grad share` = mean `grad_norm_con / grad_norm_tot` over the last "
        f"{GRAD_SHARE_TAIL} logged points of `metrics.jsonl` (10-epoch "
        f"cadence); `{NA}` = the run predates the gradient-share "
        "instrumentation.",
        "- snap gap uses the diagnostic `y_snap = y - g(y)*a`; it must be >= 0 "
        "up to float precision.",
        "- k0 is the affine anchor: cap / arm / curvature strata do not apply.",
        "- predictions were re-run from `model.pt` (eval_rows.parquet stores no "
        "`y`); max |obj_post recomputed - recorded| = "
        + ", ".join(f"{rr['variant']}:{rr['obj_check']:.2e}" for rr in rows)
        + ".",
        "",
    ]
    return "\n".join([core, mech, strata, "\n".join(notes)])


def _render_folds(
    rows: list[dict[str, Any]], tie_abs: float, tie_rel: float, family: str = "curvature_sine"
) -> str:
    """Same three tables for the fold families, folds replacing arms and the K-cap the r-cap."""
    rows = sorted(rows, key=lambda rr: (rr["kappa"], rr["variant"]))
    core = _table(
        rows,
        f"{family} core: feasibility, pre-repair distance, optimality gap",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("n", "n", "{}"),
            ("feas", "feas", "{:.3f}"),
            ("n_feas", "n_feas", "{}"),
            ("d p50", "d_p50", "{:.3e}"),
            ("d p90", "d_p90", "{:.3e}"),
            ("k*d p50", "kd_p50", "{:.3e}"),
            ("k*d p90", "kd_p90", "{:.3e}"),
            ("gap p50", "gap_p50", "{:+.3e}"),
            ("gap p90", "gap_p90", "{:+.3e}"),
            ("gap mean", "gap_mean", "{:+.3e}"),
            ("grad share", "grad_share", "{:.3f}"),
        ],
    )
    mech = _table(
        rows,
        f"{family} mechanism: local curvature x signed distance, snap gap, wrong fold",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("K(t*)*d_sgn p50", "kd_p50_signed", "{:+.3e}"),
            ("K(t*)*d_sgn p90", "kd_p90_signed", "{:+.3e}"),
            ("snap gap p50", "snap_p50", "{:+.3e}"),
            ("snap gap p90", "snap_p90", "{:+.3e}"),
            ("snap gap min", "snap_min", "{:+.3e}"),
            ("wrong-fold", "wrong_fold", "{:.3f}"),
            ("n_elig", "n_elig", "{}"),
            ("n_ambig", "n_ambig", "{}"),
            ("n_snap_used", "n_via_snap", "{}"),
        ],
    )
    strata = _table(
        rows,
        f"{family} strata bin counts",
        [
            ("variant", "variant", "{}"),
            ("kappa", "kappa", "{:g}"),
            ("crest cap K>=k/2", "n_cap", "{}"),
            ("1 fold", "n_nfold_bin0", "{}"),
            ("2 folds", "n_nfold_bin1", "{}"),
            ("3 folds", "n_nfold_bin2", "{}"),
            (">=4 folds", "n_nfold_bin3", "{}"),
            ("K/k<0.1", "n_k_bin0", "{}"),
            ("0.1..0.5", "n_k_bin1", "{}"),
            (">=0.5", "n_k_bin2", "{}"),
            ("dFold<1e-3", "n_fold_bin0", "{}"),
            ("1e-3..1e-2", "n_fold_bin1", "{}"),
            ("1e-2..1e-1", "n_fold_bin2", "{}"),
            (">=1e-1", "n_fold_bin3", "{}"),
        ],
    )
    notes = [
        "### notes",
        "",
        f"- ambiguity margin: `Delta_fold > {tie_abs:g} + {tie_rel:g} * f*` "
        "(else counted as ambiguous, not wrong-fold).",
        f"- crest-cap stratum: local `K(t*) >= {CAP_K_FRAC:g} * kappa` (the curvature_sine "
        "analog of curvature_hinge's `|t*|/r <= 3`); fold index = `floor(omega*t/pi)`.",
        "- `kappa = omega^2` IS the max curvature (attained at every crest).",
        "- `d` is the EXACT Euclidean distance of the pre-repair prediction to "
        "`{g=0}` from the certified reference solver.",
        "- `gap` is over feasible outputs only; tiny negatives are expected "
        "(approximate feasibility can undershoot `f*`). `gap mean` is the "
        "feasible-only mean of the same quantity.",
        f"- `grad share` = mean `grad_norm_con / grad_norm_tot` over the last "
        f"{GRAD_SHARE_TAIL} logged points of `metrics.jsonl` (10-epoch "
        f"cadence); `{NA}` = the run predates the gradient-share "
        "instrumentation.",
        "- snap gap uses the diagnostic `y_snap = y - g(y)*a`; it must be >= 0 "
        "up to float precision.",
        "- k0 is the linear anchor (omega = 0): cap / fold / curvature strata do "
        "not apply.",
        "- predictions were re-run from `model.pt` (eval_rows.parquet stores no "
        "`y`); max |obj_post recomputed - recorded| = "
        + ", ".join(f"{rr['variant']}:{rr['obj_check']:.2e}" for rr in rows)
        + ".",
        "",
    ]
    if family == "curvature_warp":
        notes[2:2] = [
            "- curvature_warp shares curvature_sine's manifold, kappa dial, (a, b) and "
            "query sets; only "
            "the objective is warped, `f(y;x) = |T^-1(y) - T^-1(x)|^2` with "
            "`T(v) = v + a*sin(omega*b.v)`. Restricted to `{g=0}` that IS the "
            "affine k0 problem at every kappa, so `y* = x - g(x)*a` is unique and "
            "smooth in `x`.",
            "- consequence: `n_folds == 1` and `Delta_fold` is undefined at every "
            "row, so `n_elig = 0` and the wrong-fold column is empty BY "
            "CONSTRUCTION. Empty here is the control passing, not a missing "
            "measurement.",
            "- `d`, `K(t*)*d_sgn` and the fold of the model output use "
            "`geometric_reference` (curvature_sine's certified EUCLIDEAN projection), so "
            "they are directly comparable with the curvature_sine arm; `gap` and `snap gap` "
            "use the warped objective.",
        ]
    return "\n".join([core, mech, strata, "\n".join(notes)])


#: marks a grad-share value borrowed from the source run of a `--method-override` sibling.
SHARED_MARKER = "*"
CONTRACTION_EPS = 1e-12
CONTRACTION_PRE_KEY = "repair_c_pre"
CONTRACTION_POST_KEY = "repair_c_post"
#: sibling run dirs are named `<source run id>__as_<override method>`.
OVERRIDE_SUFFIX = "__as_"
OVERRIDE_SOURCE_KEY = "override_source_run_id"
#: (label, cell key, mean spec, std spec).
CAMPAIGN_METRIC_ROWS = [
    ("feas", "feas", "{:.3f}", "{:.3f}"),
    ("gap (feasible only)", "gap", "{:+.3e}", "{:.1e}"),
    ("grad share", "grad_share", "{:.3f}", "{:.3f}"),
    ("repair contraction", "contraction", "{:.3f}", "{:.3f}"),
]
_VARIANT_K_RE = re.compile(r"k(\d+)")


def _fmt_tau(tau: float | None) -> str:
    """`tau` in short scientific form: 1e-4 -> `1e-4`, None -> `default`."""
    if tau is None:
        return "default"
    mantissa, exponent = f"{float(tau):.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".") or "0"
    exp = int(exponent)
    return mantissa if exp == 0 else f"{mantissa}e{exp}"


def _arm_label(cfg: dict[str, Any]) -> str:
    """Arm identity of a run: the method string, with `pal_loggap` split by `tau`."""
    method = str(cfg.get("method", ""))
    if method == "dc3":
        hp = cfg.get("hparams") or {}
        tags = []
        if hp.get("completion_strategy_override"):
            tags.append(str(hp["completion_strategy_override"]).replace("generic_", ""))
        yf = hp.get("newton_yf_damping")
        if yf and float(yf) > 0:
            tags.append(f"yf={float(yf):g}")
        lm = hp.get("newton_lm_damping")
        if lm and float(lm) > 0:
            tags.append(f"lm={float(lm):g}")
        return f"dc3[{','.join(tags)}]" if tags else "dc3"
    if method != "pal_loggap":
        return method
    tau = (cfg.get("hparams") or {}).get("tau")
    return f"pal_loggap[tau={_fmt_tau(None if tau is None else float(tau))}]"


def _variant_sort_key(variant: str) -> tuple[int, int, str]:
    """Sort dial variants by the NUMBER after `k` (so k11 follows k10, not k1)."""
    m = _VARIANT_K_RE.search(variant)
    if m is None:
        return (1, 0, variant)
    return (0, int(m.group(1)), variant)


def _parse_seed_spec(spec: str) -> list[int]:
    """`0-9`, `0,1,2`, `0-4,7` -> the explicit sorted seed list."""
    seeds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                lo_s, hi_s = part.split("-", 1)
                lo, hi = int(lo_s), int(hi_s)
                if hi < lo:
                    raise ValueError(f"empty range {part!r}")
                seeds.update(range(lo, hi + 1))
            else:
                seeds.add(int(part))
        except ValueError as exc:
            raise SystemExit(f"--expect-seeds: cannot parse {part!r} ({exc})") from exc
    if not seeds:
        raise SystemExit(f"--expect-seeds: {spec!r} names no seeds")
    return sorted(seeds)


@dataclass
class RunRef:
    """A candidate campaign run: located and labelled, not yet evaluated."""

    run_dir: Path
    cfg: dict[str, Any]
    arm: str
    benchmark_id: str
    variant: str
    seed: int
    skip_reason: str | None = None


@dataclass
class CampaignRecord:
    """One run's contribution to its (arm, variant) cell. `None` == NA."""

    arm: str
    variant: str
    benchmark_id: str
    seed: int
    run_dir_name: str
    feas: float | None = None
    n_feas: float | None = None
    n: int | None = None
    gap_mean: float | None = None
    grad_share: float | None = None
    grad_share_shared: bool = False
    contraction: float | None = None
    skip_reason: str | None = None


def _scan_campaign(runs_root: Path) -> tuple[list[RunRef], list[tuple[Path, str]]]:
    """Every dial run under `runs_root`, unusable ones carrying a `skip_reason`."""
    refs: list[RunRef] = []
    unplaceable: list[tuple[Path, str]] = []
    for cfg_path in sorted(runs_root.glob("*/config.json")):
        run_dir = cfg_path.parent
        try:
            cfg = json.loads(cfg_path.read_text())
        except (OSError, ValueError) as exc:
            unplaceable.append((run_dir, f"config.json unreadable: {type(exc).__name__}: {exc}"))
            continue
        bench_id = str(cfg.get("benchmark_id", ""))
        family = _family_of(bench_id)
        if family is None:
            continue
        try:
            seed = int(cfg["seed"])
        except (KeyError, TypeError, ValueError):
            unplaceable.append((run_dir, "config.json carries no usable `seed`"))
            continue
        reason: str | None = None
        if not (run_dir / "model.pt").exists():
            reason = "missing model.pt"
        elif not (run_dir / "eval_rows.parquet").exists():
            reason = "missing eval_rows.parquet"
        refs.append(
            RunRef(
                run_dir=run_dir,
                cfg=cfg,
                arm=_arm_label(cfg),
                benchmark_id=bench_id,
                variant=bench_id.removeprefix(PREFIX_BY_FAMILY[family]),
                seed=seed,
                skip_reason=reason,
            )
        )
    return refs, unplaceable


def _check_duplicates(refs: list[RunRef]) -> None:
    """Hard-fail on a repeated (arm, benchmark, seed): picking one would be silent."""
    by_key: dict[tuple[str, str, int], list[RunRef]] = {}
    for ref in refs:
        by_key.setdefault((ref.arm, ref.benchmark_id, ref.seed), []).append(ref)
    dups = {k: v for k, v in by_key.items() if len(v) > 1}
    if not dups:
        return
    lines = ["duplicate (arm, benchmark, seed) run dirs, refusing to pick one:"]
    for (arm, bench_id, seed), group in sorted(dups.items()):
        lines.append(f"  arm={arm} benchmark={bench_id} seed={seed}:")
        lines.extend(f"    {ref.run_dir}" for ref in group)
    raise SystemExit("\n".join(lines))


def _current_git_sha(repo_root: Path) -> str | None:
    """HEAD of the checkout that is about to re-run `solver.predict`."""
    try:
        proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return proc.stdout.strip() or None


def _check_checkout_stability(
    refs: list[RunRef], head_sha: str | None, allow_mismatch: bool
) -> list[str]:
    """Fail unless all runs came from the current checkout (predict is re-run with this code)."""
    warnings: list[str] = []
    if head_sha is None:
        msg = "could not determine the current checkout's HEAD sha (`git rev-parse HEAD` failed)"
        if not allow_mismatch:
            raise SystemExit(msg + "; pass --allow-sha-mismatch to analyze anyway")
        warnings.append(f"!! {msg}, run/checkout agreement was not verified")
    else:
        bad = [(ref, str(ref.cfg.get("pal_git_sha") or "<missing>")) for ref in refs]
        bad = [(ref, sha) for ref, sha in bad if sha != head_sha]
        if bad:
            lines = [
                f"{len(bad)} run(s) were produced by a different checkout than HEAD "
                f"({head_sha[:12]}); the analyzer re-runs predict() with the CURRENT code:"
            ]
            lines.extend(f"  {sha[:12]}  {ref.run_dir.name}" for ref, sha in bad)
            if not allow_mismatch:
                raise SystemExit(
                    "\n".join([*lines, "pass --allow-sha-mismatch to analyze anyway"])
                )
            warnings.append("!! SHA MISMATCH (allowed via --allow-sha-mismatch)\n" + "\n".join(lines[1:]))
    dirty = [ref for ref in refs if str(ref.cfg.get("pal_git_diff") or "").strip()]
    if dirty:
        warnings.append(
            "!! "
            + f"{len(dirty)} run(s) were produced with uncommitted changes "
            + "(non-empty `pal_git_diff`):\n"
            + "\n".join(f"  {ref.run_dir.name}" for ref in dirty)
        )
    return warnings


def _grad_share_source_dir(run_dir: Path, cfg: dict[str, Any]) -> tuple[Path | None, bool]:
    """Where this run's training diagnostics live, and whether they are borrowed."""
    if (run_dir / "metrics.jsonl").exists():
        return run_dir, False
    src_id = cfg.get(OVERRIDE_SOURCE_KEY)
    if not src_id:
        return None, False
    candidates = []
    if OVERRIDE_SUFFIX in run_dir.name:
        candidates.append(run_dir.parent / run_dir.name.rsplit(OVERRIDE_SUFFIX, 1)[0])
    candidates.append(run_dir.parent / str(src_id))
    for cand in candidates:
        if cand != run_dir and (cand / "metrics.jsonl").exists():
            return cand, True
    for cfg_path in sorted(run_dir.parent.glob("*/config.json")):
        cand = cfg_path.parent
        if cand.name == str(src_id) and (cand / "metrics.jsonl").exists():
            return cand, True
    return None, False


def _campaign_grad_share(run_dir: Path, cfg: dict[str, Any]) -> tuple[float | None, bool]:
    """`_final_grad_share`, with the override-sibling fallback."""
    src, shared = _grad_share_source_dir(run_dir, cfg)
    if src is None:
        return None, False
    value = _final_grad_share(src)
    return value, shared and value is not None


def _final_repair_contraction(run_dir: Path) -> float | None:
    """`repair_c_post / repair_c_pre` at the last logged point carrying both keys."""
    path = run_dir / "metrics.jsonl"
    if not path.exists():
        return None
    last: tuple[float, float] | None = None
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if CONTRACTION_PRE_KEY not in rec or CONTRACTION_POST_KEY not in rec:
                continue
            try:
                pre = float(rec[CONTRACTION_PRE_KEY])
                post = float(rec[CONTRACTION_POST_KEY])
            except (TypeError, ValueError):
                continue
            last = (pre, post)
    if last is None:
        return None
    pre, post = last
    if not math.isfinite(pre) or not math.isfinite(post) or pre < CONTRACTION_EPS:
        return None
    return post / pre


def _mean_std(values: list[float]) -> tuple[float | None, float | None]:
    """Mean and SAMPLE std (ddof=1). std is None for a single value."""
    if not values:
        return None, None
    mean = sum(values) / len(values)
    if len(values) < 2:
        return mean, None
    var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
    return mean, math.sqrt(var)


def _aggregate_metric(records: list[CampaignRecord], attr: str) -> dict[str, Any]:
    """Aggregate one per-run scalar over seeds; `None` values are NA, not zero."""
    per_seed = {rec.seed: getattr(rec, attr) for rec in records}
    present = [v for v in per_seed.values() if v is not None]
    mean, std = _mean_std(present)
    return {
        "mean": mean,
        "std": std,
        "n": len(present),
        "n_runs": len(per_seed),
        "values": {str(s): per_seed[s] for s in sorted(per_seed)},
    }


def _build_cells(
    records: list[CampaignRecord], expected_seeds: list[int] | None
) -> list[dict[str, Any]]:
    """One aggregate per (arm, benchmark_id), sorted by arm then kappa."""
    by_cell: dict[tuple[str, str], list[CampaignRecord]] = {}
    for rec in records:
        by_cell.setdefault((rec.arm, rec.benchmark_id), []).append(rec)
    cells: list[dict[str, Any]] = []
    for (arm, _bench_id), group in by_cell.items():
        group = sorted(group, key=lambda r: r.seed)
        usable = [r for r in group if r.skip_reason is None]
        seeds_ok = {r.seed for r in usable}
        expected = list(expected_seeds) if expected_seeds is not None else sorted(seeds_ok)
        missing = [s for s in expected if s not in seeds_ok]
        skipped = {
            str(r.seed): f"{r.skip_reason} ({r.run_dir_name})"
            for r in group
            if r.skip_reason is not None
        }
        cells.append(
            {
                "arm": arm,
                "variant": group[0].variant,
                "benchmark_id": group[0].benchmark_id,
                "n_completed": len(usable),
                "n_expected": len(expected),
                "missing_seeds": missing,
                "skipped": skipped,
                "run_dirs": {str(r.seed): r.run_dir_name for r in group},
                "grad_share_shared": any(r.grad_share_shared for r in usable),
                "metrics": {
                    "feas": _aggregate_metric(usable, "feas"),
                    "gap": _aggregate_metric(usable, "gap_mean"),
                    "n_feas": _aggregate_metric(usable, "n_feas"),
                    "grad_share": _aggregate_metric(usable, "grad_share"),
                    "contraction": _aggregate_metric(usable, "contraction"),
                },
            }
        )
    return sorted(cells, key=lambda c: (c["arm"], _variant_sort_key(c["variant"])))


def _fmt_mean_std(agg: dict[str, Any] | None, spec: str, std_spec: str) -> str:
    """`mean +/- std`, `mean +/- -` for one seed, or a dash if all seeds are NA."""
    if agg is None or agg.get("mean") is None:
        return DASH
    std = agg.get("std")
    tail = std_spec.format(std) if std is not None else DASH
    return f"{spec.format(agg['mean'])} +/- {tail}"


def _campaign_cell_text(cell: dict[str, Any], key: str, spec: str, std_spec: str) -> str:
    """One rendered table cell, with the gap's NA count and the shared marker."""
    agg = cell["metrics"].get(key)
    text = _fmt_mean_std(agg, spec, std_spec)
    if key == "gap" and agg is not None and agg["n"] < agg["n_runs"]:
        text += f" ({agg['n']}/{agg['n_runs']})"
    if key == "grad_share" and text != DASH and cell["grad_share_shared"]:
        text += SHARED_MARKER
    return text


def _campaign_columns(cells: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Ordered `(benchmark_id, column label)`, sorted by the kappa index."""
    by_bench = {c["benchmark_id"]: c["variant"] for c in cells}
    counts: dict[str, int] = {}
    for variant in by_bench.values():
        counts[variant] = counts.get(variant, 0) + 1
    ordered = sorted(by_bench, key=lambda b: (_variant_sort_key(by_bench[b]), b))
    return [(b, by_bench[b] if counts[by_bench[b]] == 1 else b) for b in ordered]


def _render_campaign_table(cells: list[dict[str, Any]], columns: list[tuple[str, str]]) -> str:
    """Rows = arms (one metric block each), columns = variants in kappa order."""
    by_arm: dict[str, dict[str, dict[str, Any]]] = {}
    for cell in cells:
        by_arm.setdefault(cell["arm"], {})[cell["benchmark_id"]] = cell
    head = "| " + " | ".join(["arm", "metric", *(lab for _, lab in columns)]) + " |"
    rule = "|" + "|".join("---" for _ in range(len(columns) + 2)) + "|"
    body: list[str] = []
    for arm in sorted(by_arm):
        per_bench = by_arm[arm]
        rows = [*CAMPAIGN_METRIC_ROWS, ("seeds", None, "", "")]
        for i, (label, key, spec, std_spec) in enumerate(rows):
            cells_text = []
            for bench_id, _col_label in columns:
                cell = per_bench.get(bench_id)
                if cell is None:
                    cells_text.append(DASH)
                elif key is None:
                    cells_text.append(f"{cell['n_completed']}/{cell['n_expected']}")
                else:
                    cells_text.append(_campaign_cell_text(cell, key, spec, std_spec))
            first = arm if i == 0 else ""
            body.append("| " + " | ".join([first, label, *cells_text]) + " |")
    return "\n".join(["### campaign: arm x variant, mean+/-std over seeds", "", head, rule, *body, ""])


def _render_campaign_warnings(
    cells: list[dict[str, Any]],
    unplaceable: list[tuple[Path, str]],
    extra: list[str],
) -> str:
    lines = ["### WARNINGS", ""]
    n = 0
    for msg in extra:
        lines.append(f"- {msg}")
        n += 1
    for run_dir, reason in unplaceable:
        lines.append(f"- [unplaceable] {run_dir.name}: {reason}")
        n += 1
    for cell in cells:
        if not cell["missing_seeds"]:
            continue
        missing = ", ".join(str(s) for s in cell["missing_seeds"])
        lines.append(
            f"- [missing seeds] arm={cell['arm']} variant={cell['variant']}: "
            f"{cell['n_completed']}/{cell['n_expected']} present, missing seed(s) {missing}"
        )
        n += 1
        for seed in cell["missing_seeds"]:
            reason = cell["skipped"].get(str(seed))
            if reason is not None:
                lines.append(f"    - seed {seed}: run dir exists but was skipped: {reason}")
    if n == 0:
        lines.append("- none.")
    lines.append("")
    return "\n".join(lines)


def _render_campaign_notes() -> str:
    return "\n".join(
        [
            "### notes",
            "",
            "- cells are `mean +/- std` over seeds (sample std, ddof=1), `+/- -` = a single "
            "seed, `-` = every seed NA.",
            "- `gap` is the per-run mean of `f - f*` over that run's feasible queries "
            "only; a run with zero feasible queries is NA for the gap (and excluded "
            "from its mean/std) while still counting for `feas`. `(k/m)` after a gap "
            "cell = k of m runs contributed.",
            f"- `grad share` = mean `grad_norm_con / grad_norm_tot` over the last "
            f"{GRAD_SHARE_TAIL} logged points; `{SHARED_MARKER}` = read from the "
            "source run of a `--method-override` sibling, i.e. it is the source run's "
            "training diagnostic, not the sibling's.",
            "- `repair contraction` = `repair_c_post / repair_c_pre` at the last "
            "logged point carrying both keys; NA (never 0) for methods that log "
            f"neither, and NA when `repair_c_pre < {CONTRACTION_EPS:g}`. Read from the "
            "run's OWN `metrics.jsonl` only (never borrowed).",
            "- `seeds` = completed / expected runs in the cell; every missing seed is "
            "named in WARNINGS.",
            "- arms: the method string, with `pal_loggap` split by `tau`; a "
            "`--method-override` sibling is its own arm.",
            "",
        ]
    )


def _campaign_json(
    runs_root: Path,
    head_sha: str | None,
    expected_seeds: list[int] | None,
    cells: list[dict[str, Any]],
    warnings: list[str],
    unplaceable: list[tuple[Path, str]],
) -> dict[str, Any]:
    columns = _campaign_columns(cells)
    return {
        "runs_root": str(runs_root),
        "head_sha": head_sha,
        "expected_seeds": expected_seeds,
        "benchmark_ids": [b for b, _ in columns],
        "variants": [lab for _, lab in columns],
        "arms": sorted({c["arm"] for c in cells}),
        "cells": cells,
        "warnings": warnings,
        "unplaceable_runs": [{"run_dir": str(p), "reason": r} for p, r in unplaceable],
    }


@dataclass
class CampaignReport:
    """Everything `--campaign` prints, plus the machine-readable payload."""

    markdown: str
    payload: dict[str, Any]
    cells: list[dict[str, Any]] = field(default_factory=list)


def _render_campaign(
    runs_root: Path,
    head_sha: str | None,
    expected_seeds: list[int] | None,
    cells: list[dict[str, Any]],
    warnings: list[str],
    unplaceable: list[tuple[Path, str]],
) -> CampaignReport:
    markdown = "\n".join(
        [
            _render_campaign_table(cells, _campaign_columns(cells)),
            _render_campaign_notes(),
            _render_campaign_warnings(cells, unplaceable, warnings),
        ]
    )
    payload = _campaign_json(runs_root, head_sha, expected_seeds, cells, warnings, unplaceable)
    return CampaignReport(markdown=markdown, payload=payload, cells=cells)


def _campaign_record(
    ref: RunRef, metrics: dict[str, Any], grad_share: float | None, shared: bool
) -> CampaignRecord:
    """Per-run campaign record from an already computed `_metrics` dict."""
    gap = metrics.get("gap_mean")
    gap = None if gap is None or gap != gap else float(gap)  # NaN == zero feasible queries
    return CampaignRecord(
        arm=ref.arm,
        variant=ref.variant,
        benchmark_id=ref.benchmark_id,
        seed=ref.seed,
        run_dir_name=ref.run_dir.name,
        feas=float(metrics["feas"]),
        n_feas=float(metrics["n_feas"]),
        n=int(metrics["n"]),
        gap_mean=gap,
        grad_share=grad_share,
        grad_share_shared=shared,
        contraction=_final_repair_contraction(ref.run_dir),
    )


def _run_campaign(args: argparse.Namespace) -> None:
    """`--campaign`: aggregate a whole arms x variants x seeds campaign."""
    refs, unplaceable = _scan_campaign(args.runs_root)
    if not refs:
        raise SystemExit(f"no curvature dial runs found under {args.runs_root}")
    _check_duplicates(refs)
    head_sha = _current_git_sha(_REPO_ROOT)
    warnings = _check_checkout_stability(refs, head_sha, args.allow_sha_mismatch)
    expected_seeds = _parse_seed_spec(args.expect_seeds) if args.expect_seeds else None

    records: list[CampaignRecord] = []
    for ref in refs:
        if ref.skip_reason is not None:
            records.append(
                CampaignRecord(
                    arm=ref.arm,
                    variant=ref.variant,
                    benchmark_id=ref.benchmark_id,
                    seed=ref.seed,
                    run_dir_name=ref.run_dir.name,
                    skip_reason=ref.skip_reason,
                )
            )
            continue
        try:
            run = _load_run(ref.run_dir, args.device)
            metrics = _metrics(run, args.tie_abs, args.tie_rel)
        except Exception as exc:  # a broken run is REPORTED, never skipped silently
            records.append(
                CampaignRecord(
                    arm=ref.arm,
                    variant=ref.variant,
                    benchmark_id=ref.benchmark_id,
                    seed=ref.seed,
                    run_dir_name=ref.run_dir.name,
                    skip_reason=f"load error: {type(exc).__name__}: {exc}",
                )
            )
            continue
        grad_share, shared = _campaign_grad_share(ref.run_dir, ref.cfg)
        records.append(_campaign_record(ref, metrics, grad_share, shared))
        print(f"[{ref.arm}] {ref.variant:>4}  seed={ref.seed}  {ref.run_dir.name}")

    cells = _build_cells(records, expected_seeds)
    report = _render_campaign(
        args.runs_root, head_sha, expected_seeds, cells, warnings, unplaceable
    )
    print()
    print(report.markdown)
    if args.out is not None:
        args.out.write_text(report.markdown)
        print(f"[campaign] wrote {args.out}")
    if args.out_json is not None:
        args.out_json.write_text(json.dumps(report.payload, indent=2))
        print(f"[campaign] wrote {args.out_json}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs-root", required=True, type=Path)
    ap.add_argument("--method", default="pal_loggap", help="'' for all methods")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tie-abs", type=float, default=1e-4)
    ap.add_argument("--tie-rel", type=float, default=1e-2)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--campaign",
        action="store_true",
        help="aggregate an arms x variants x seeds campaign into one table "
        "(mean+/-std over seeds). Covers all methods and seeds under the root: "
        "--method / --seed are ignored, arms are the grouping unit.",
    )
    ap.add_argument(
        "--expect-seeds",
        default=None,
        help="campaign mode: seeds every (arm, variant) cell must have, e.g. "
        "'0-9' or '0,1,5-7'. Missing ones are named in the WARNINGS section.",
    )
    ap.add_argument(
        "--allow-sha-mismatch",
        action="store_true",
        help="campaign mode: downgrade the run-vs-HEAD checkout mismatch from a "
        "hard error to a prominent warning.",
    )
    ap.add_argument(
        "--out-json",
        type=Path,
        default=None,
        help="campaign mode: write the per-cell aggregates + per-seed raw values here.",
    )
    args = ap.parse_args()

    if args.campaign:
        _run_campaign(args)
        return

    by_family = _discover(args.runs_root, args.method or None, args.seed)
    if not by_family:
        raise SystemExit(f"no curvature dial runs found under {args.runs_root}")

    # One table set per method when `--method ''` admits several.
    by_family_method: dict[str, dict[str, list[Path]]] = {}
    for family, run_dirs in by_family.items():
        per_method: dict[str, list[Path]] = {}
        for run_dir in run_dirs:
            m = str(json.loads((run_dir / "config.json").read_text())["method"])
            per_method.setdefault(m, []).append(run_dir)
        by_family_method[family] = per_method
    methods_seen = {m for pm in by_family_method.values() for m in pm}
    multi_method = len(methods_seen) > 1

    sections = []
    for family, per_method in by_family_method.items():
        for method_id, run_dirs in sorted(per_method.items()):
            rows = []
            for run_dir in run_dirs:
                run = _load_run(run_dir, args.device)
                print(
                    f"[{family}] {run.variant:>3}  seed={run.seed}  "
                    f"n={run.x.shape[0]}  {run_dir.name}"
                )
                rows.append(_metrics(run, args.tie_abs, args.tie_rel))
            section = _render(rows, family, args.tie_abs, args.tie_rel)
            if multi_method:
                section = f"## method: {method_id}\n\n{section}"
            sections.append(section)

    md = "\n".join(sections)
    print()
    print(md)
    if args.out is not None:
        args.out.write_text(md)
        print(f"[{'+'.join(by_family)}] wrote {args.out}")


if __name__ == "__main__":
    main()
