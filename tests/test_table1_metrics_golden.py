"""Golden-fixture test: pal.eval.table1_metrics reproduces Table 1.

Source: results/2026-05-04_paper_cost_table/{runs.parquet,paper_cost_table.tex}.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from pal.eval.table1_constants import (
    BENCH_OBJ_CONSTANTS,
    OBJ_FAIL_THRESHOLD,
    TABLE1_BENCHES,
    TABLE1_METHODS,
)
from pal.eval.table1_metrics import (
    aggregate_cells,
    compute_bo_objective,
    load_run_rows,
    method_feas_mean,
    method_obj_mean,
    select_best_trial,
)
from scripts.render_paper_heatmap import (
    _fmt_feas,
    _fmt_mean_feas,
    _fmt_mean_obj,
    _fmt_obj,
)

_REPO = Path(__file__).resolve().parents[1]
PARQUET = _REPO / "results/2026-05-04_paper_cost_table/runs.parquet"


# Golden per-cell (feas, obj) strings from paper_cost_table.tex; None == an em dash.
GOLDEN_FEAS: dict[str, dict[str, str | None]] = {
    "alm_bolton": {"s1": r"$100\pm0$", "s2": r"$100\pm0$", "s3": r"$100\pm0$",
                   "s4": r"$98\pm4$", "s5": r"$100\pm0$", "s6": r"$100\pm0$"},
    "enforce_orig": {"s1": None, "s2": r"$1\pm1$", "s3": r"$5\pm3$",
                     "s4": None, "s5": r"$0\pm0$", "s6": r"$0\pm0$"},
    "dc3": {"s1": None, "s2": r"$80\pm33$", "s3": r"$100\pm0$",
            "s4": None, "s5": None, "s6": None},
    "fsnet": {"s1": r"$100\pm0$", "s2": r"$100\pm0$", "s3": r"$100\pm0$",
              "s4": r"$100\pm0$", "s5": r"$66\pm24$", "s6": r"$100\pm0$"},
    "snarenet": {"s1": r"$100\pm0$", "s2": r"$100\pm0$", "s3": r"$92\pm3$",
                 "s4": r"$0\pm0$", "s5": r"$100\pm0$", "s6": r"$100\pm0$"},
    "pal_loggap": {"s1": r"$100\pm0$", "s2": r"$100\pm0$", "s3": r"$100\pm0$",
                   "s4": r"$100\pm0$", "s5": r"$100\pm0$", "s6": r"$100\pm0$"},
}
GOLDEN_OBJ: dict[str, dict[str, str | None]] = {
    "alm_bolton": {"s1": r"$8.66\pm7.98$", "s2": r"$0.28\pm0.18$", "s3": r"$0.00\pm0.00$",
                   "s4": r"$9.82\pm4.98$", "s5": r"$0.00\pm0.00$", "s6": r"$0.14\pm0.02$"},
    "enforce_orig": {"s1": None, "s2": r"$0.00\pm0.00$", "s3": r"$0.00\pm0.00$",
                     "s4": None, "s5": r"$0.06\pm0.11$", "s6": r"$0.01\pm0.00$"},
    "dc3": {"s1": None, "s2": r"$0.00\pm0.00$", "s3": r"$0.00\pm0.00$",
            "s4": None, "s5": None, "s6": None},
    "fsnet": {"s1": r"$0.00\pm0.00$", "s2": r"$0.00\pm0.00$", "s3": r"$0.00\pm0.00$",
              "s4": r"$0.00\pm0.00$", "s5": r"$4.99\pm2.04$", "s6": r"$0.00\pm0.00$"},
    "snarenet": {"s1": r"$0.83\pm0.10$", "s2": r"$0.16\pm0.00$", "s3": r"$0.05\pm0.02$",
                 "s4": r"$0.91\pm0.35$", "s5": r"$0.17\pm0.06$", "s6": r"$2.68\pm0.16$"},
    "pal_loggap": {"s1": r"$0.01\pm0.02$", "s2": r"$0.00\pm0.00$", "s3": r"$0.00\pm0.00$",
                   "s4": r"$0.01\pm0.00$", "s5": r"$0.00\pm0.00$", "s6": r"$0.08\pm0.01$"},
}
# Mean-row goldens for the four full-coverage methods (enforce/dc3 render an em dash).
GOLDEN_MEAN_FEAS = {"alm_bolton": r"$100$", "fsnet": r"$94$",
                    "snarenet": r"$82$", "pal_loggap": r"$100$"}
GOLDEN_MEAN_OBJ = {"alm_bolton": r"$3.15$", "fsnet": r"$0.83$",
                   "snarenet": r"$0.80$", "pal_loggap": r"$0.02$"}

_SHORT = {b: b.split("_", 1)[0] for b in TABLE1_BENCHES}  # "s1_sphere_track" -> "s1"

# The one partial cell: an em dash in the .tex, a seed-level worst-cased value in the module.
PARTIAL_CELLS = {("enforce_orig", "s4_qv_coupling")}


@pytest.fixture(scope="module")
def cells():
    assert PARQUET.exists(), f"golden parquet missing: {PARQUET}"
    return aggregate_cells(load_run_rows(PARQUET))


def test_per_cell_feasibility_matches_committed_tex(cells):
    for method, per_bench in GOLDEN_FEAS.items():
        for bench in TABLE1_BENCHES:
            c = cells[(method, bench)]
            golden = per_bench[_SHORT[bench]]
            if golden is None:
                if (method, bench) in PARTIAL_CELLS:
                    assert c.n_ok > 0 and c.n_diverged > 0
                    continue
                assert c.is_structural or c.is_fully_diverged, (
                    f"{method} x {bench}: expected '---' cell, got data"
                )
                continue
            got = _fmt_feas(c.feas_disp_mean, c.feas_disp_std or 0.0)
            assert got == golden, f"{method} x {bench} feas: {got} != {golden}"


def test_per_cell_objective_matches_committed_tex(cells):
    for method, per_bench in GOLDEN_OBJ.items():
        for bench in TABLE1_BENCHES:
            c = cells[(method, bench)]
            golden = per_bench[_SHORT[bench]]
            if golden is None:
                if (method, bench) in PARTIAL_CELLS:
                    assert c.n_ok > 0 and c.n_diverged > 0
                    continue
                assert c.is_structural or c.is_fully_diverged
                continue
            got = _fmt_obj(c.obj_disp_mean, c.obj_disp_std or 0.0)
            assert got == golden, f"{method} x {bench} obj: {got} != {golden}"


def test_mean_row_matches_committed_tex_for_full_coverage_methods(cells):
    for method, golden in GOLDEN_MEAN_FEAS.items():
        got = _fmt_mean_feas(method_feas_mean(cells, method))
        assert got == golden, f"{method} feas Mean: {got} != {golden}"
    for method, golden in GOLDEN_MEAN_OBJ.items():
        got = _fmt_mean_obj(method_obj_mean(cells, method))
        assert got == golden, f"{method} obj Mean: {got} != {golden}"


def test_hand_checked_raw_cell_values(cells):
    # alm_bolton x s1 is the worst obj cell and equals C_s1 exactly.
    c = cells[("alm_bolton", "s1_sphere_track")]
    assert c.feas_disp_mean == pytest.approx(1.0, abs=1e-12)
    assert c.obj_disp_mean == pytest.approx(8.664392995404706, abs=1e-12)
    assert c.n_diverged == 0

    c = cells[("pal_loggap", "s6_redundant_ineq")]
    assert c.feas_disp_mean == pytest.approx(1.0, abs=1e-12)
    assert c.obj_disp_mean == pytest.approx(0.07754682011436671, abs=1e-12)
    assert c.l3_b == pytest.approx(0.0, abs=1e-12)  # feasible -> zero violation score


def test_partial_divergence_is_seed_level_worst_cased(cells):
    # enforce_orig x s4: 8 clean seeds + 2 diverged; diverged imputed at C_s4.
    c = cells[("enforce_orig", "s4_qv_coupling")]
    assert (c.n_attempted, c.n_ok, c.n_diverged) == (10, 8, 2)
    assert not c.is_fully_diverged
    c_b = BENCH_OBJ_CONSTANTS["s4_qv_coupling"]
    # obj_bo = (sum of 8 clean objs + 2 * C_b) / 10; clean-obj mean was 0.7539.
    clean_sum = c.obj_disp_mean * 10 - 2 * c_b
    assert clean_sum / 8 == pytest.approx(0.7539, abs=1e-3)
    assert c.obj_bo == c.obj_disp_mean  # display uses the imputed seed-mean
    assert 0.0 < c.l2_b < 1.0


def test_structural_and_full_divergence_flags(cells):
    struct = cells[("dc3", "s5_overdetermined")]
    assert struct.is_structural and not struct.is_fully_diverged

    for bench in ("s1_sphere_track", "s4_qv_coupling", "s6_redundant_ineq"):
        c = cells[("dc3", bench)]
        assert c.is_fully_diverged, f"dc3 x {bench} should be fully diverged"
        assert c.feas_bo == 0.0
        # imputed at ceiling (mean of N copies of C_b, up to fp rounding).
        assert c.obj_bo == pytest.approx(BENCH_OBJ_CONSTANTS[bench], abs=1e-12)
        assert c.l2_b == pytest.approx(1.0, abs=1e-12)
        assert c.l3_b == pytest.approx(1.0, abs=1e-12)

    enforce_s1 = cells[("enforce_orig", "s1_sphere_track")]
    assert enforce_s1.is_fully_diverged


def test_bo_objective_and_lexicographic_selection(cells):
    objs = {m: compute_bo_objective(cells, m) for m in TABLE1_METHODS}

    # One row per (method, bench, seed) and every column the module reads.
    import polars as pl

    df = pl.read_parquet(PARQUET)
    assert {"method", "benchmark_id", "seed", "status", "obj_mean_post",
            "feasibility_post", "viol_max_post", "n_queries", "wall_start"} <= set(df.columns)
    rows = load_run_rows(PARQUET)
    assert len(rows) == len(TABLE1_METHODS) * len(TABLE1_BENCHES) * 10
    for o in objs.values():
        assert 0.0 <= o.l1 <= 1.0
        assert o.l2 >= 0.0 and o.l3 >= 0.0

    # dc3 drops its single structural cell (s5) -> 5 applicable benches.
    assert objs["dc3"].n_applicable_benches == 5
    assert objs["dc3"].n_diverged_cells == 3      # s1, s4, s6
    assert objs["dc3"].diverged
    for m in TABLE1_METHODS:
        if m != "dc3":
            assert objs[m].n_applicable_benches == 6

    assert objs["pal_loggap"].n_seeds == 10
    assert objs["pal_loggap"].n_queries == 64

    # select_best_trial returns the lexicographic minimum of lex_key, and the
    # scalar S is monotone in that order (non-increasing along the sorted keys).
    order = list(TABLE1_METHODS)
    trials = [objs[m] for m in order]
    best = select_best_trial(trials)
    assert trials[best].lex_key == min(t.lex_key for t in trials)
    ranked = sorted(trials, key=lambda o: o.lex_key)
    scalars = [o.scalar for o in ranked]
    assert all(a >= b for a, b in zip(scalars, scalars[1:]))
    assert max(trials, key=lambda o: o.scalar) is trials[best]


def test_scalar_tiebreaker_bound_holds(cells):
    # compute_bo_objective asserts eps1+eps2 < 1/(q*s*b) = 1/(64*10*6).
    o = compute_bo_objective(cells, "pal_loggap")
    assert 1e-4 + 1e-7 < 1.0 / (o.n_queries * o.n_seeds * o.n_applicable_benches)


def test_renderer_consumes_module_preserves_numbers_and_adds_alm(tmp_path):
    """Rendered numbers are unchanged; only the alm column, mean fill-ins and notes are new."""
    from scripts.render_paper_heatmap import render

    out = tmp_path / "table.tex"
    render(PARQUET, out, standalone_path=None, preamble_path=None)
    text = out.read_text()

    assert r"\textbf{ALM}" in text

    preserved = [
        r"$8.66\pm7.98$", r"$9.82\pm4.98$", r"$0.28\pm0.18$",  # alm_bolton obj
        r"$98\pm4$",                                            # alm_bolton feas
        r"$1\pm1$", r"$5\pm3$", r"$0.06\pm0.11$",               # enforce
        r"$80\pm33$",                                           # dc3 feas
        r"$66\pm24$", r"$4.99\pm2.04$", r"$593\pm41$",          # fsnet
        r"$92\pm3$", r"$2.68\pm0.16$", r"$0.91\pm0.35$",        # snarenet
        r"$0.08\pm0.01$", r"$0.01\pm0.02$",                     # pal
    ]
    for s in preserved:
        assert s in text, f"previously-rendered value vanished: {s}"

    assert r"$36$$^{\ddag\S}$" in text   # dc3 feas Mean (drops structural s5)
    assert r"$1$$^{\ddag}$" in text      # enforce feas Mean (worst-cased)

    assert r"---$^{\S}$" in text           # dc3 x s5 structural
    assert r"---$^{\ddag}$" in text        # fully-diverged cells
    assert r"$^{2/10}$" in text          # enforce x s4 partial (2 of 10 seeds)
    assert r"structurally inapplicable" in text  # caption footnote


def test_bench_obj_constants_reproduce_worst_method_means():
    """C_b == max over methods of the per-method seed-mean obj_mean_post."""
    rows = load_run_rows(PARQUET)
    for bench in TABLE1_BENCHES:
        worst = -math.inf
        for method in TABLE1_METHODS:
            objs = []
            clean: dict = {}
            for r in rows:
                if r.method != method or r.bench != bench:
                    continue
                ok = r.status == "ok" and (
                    r.obj_post is None
                    or (math.isfinite(r.obj_post) and r.obj_post <= OBJ_FAIL_THRESHOLD)
                )
                if not ok:
                    continue
                prev = clean.get(r.seed)
                if prev is None or (str(r.wall_start) > str(prev.wall_start)):
                    clean[r.seed] = r
            objs = [c.obj_post for c in clean.values() if c.obj_post is not None]
            if objs:
                worst = max(worst, sum(objs) / len(objs))
        assert BENCH_OBJ_CONSTANTS[bench] == pytest.approx(worst, abs=1e-12), bench
