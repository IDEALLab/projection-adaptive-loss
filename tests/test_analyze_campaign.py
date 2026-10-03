"""Tests for the campaign aggregation layer of `scripts/analyze_curvature.py` (`--campaign`).

Synthetic run dirs on `tmp_path`; no model or benchmark is loaded.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analyze_curvature import (
    DASH,
    OVERRIDE_SOURCE_KEY,
    SHARED_MARKER,
    CampaignRecord,
    RunRef,
    _aggregate_metric,
    _arm_label,
    _build_cells,
    _campaign_cell_text,
    _campaign_columns,
    _campaign_grad_share,
    _check_checkout_stability,
    _check_duplicates,
    _final_repair_contraction,
    _fmt_tau,
    _grad_share_source_dir,
    _mean_std,
    _parse_seed_spec,
    _render_campaign,
    _render_campaign_table,
    _scan_campaign,
    _variant_sort_key,
)

SHA = "a" * 40


def _cfg(
    method: str = "pal_loggap",
    bench_id: str = "curvature_hinge_k4",
    seed: int = 0,
    tau: float | None = None,
    sha: str = SHA,
    diff: str = "",
    **extra: object,
) -> dict:
    cfg = {
        "method": method,
        "benchmark_id": bench_id,
        "seed": seed,
        "hparams": {"tau": tau},
        "pal_git_sha": sha,
        "pal_git_branch": "main",
        "pal_git_diff": diff,
    }
    cfg.update(extra)
    return cfg


def _make_run(
    root: Path,
    name: str,
    *,
    cfg: dict | None = None,
    model: bool = True,
    rows: bool = True,
    metrics: list[dict] | None = None,
) -> Path:
    run_dir = root / name
    run_dir.mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps(cfg if cfg is not None else _cfg()))
    if model:
        (run_dir / "model.pt").write_bytes(b"")
    if rows:
        (run_dir / "eval_rows.parquet").write_bytes(b"")
    if metrics is not None:
        (run_dir / "metrics.jsonl").write_text(
            "".join(json.dumps(rec) + "\n" for rec in metrics)
        )
    return run_dir


def _rec(
    seed: int,
    *,
    arm: str = "alm",
    variant: str = "k4",
    feas: float | None = 1.0,
    gap: float | None = 1e-3,
    n_feas: float | None = 512.0,
    grad_share: float | None = 0.5,
    shared: bool = False,
    contraction: float | None = 0.1,
    skip_reason: str | None = None,
) -> CampaignRecord:
    return CampaignRecord(
        arm=arm,
        variant=variant,
        benchmark_id=f"curvature_hinge_{variant}",
        seed=seed,
        run_dir_name=f"run_{arm}_{variant}_seed{seed}",
        feas=feas,
        n_feas=n_feas,
        n=512,
        gap_mean=gap,
        grad_share=grad_share,
        grad_share_shared=shared,
        contraction=contraction,
        skip_reason=skip_reason,
    )


def _ref(run_dir: Path, cfg: dict, *, seed: int = 0, variant: str = "k4") -> RunRef:
    return RunRef(
        run_dir=run_dir,
        cfg=cfg,
        arm=_arm_label(cfg),
        benchmark_id=cfg["benchmark_id"],
        variant=variant,
        seed=seed,
    )


@pytest.mark.parametrize(
    ("tau", "expected"),
    [
        (None, "default"),
        (1e-4, "1e-4"),
        (1e-2, "1e-2"),
        (2.5e-4, "2.5e-4"),
        (1.0, "1"),
        (10.0, "1e1"),
    ],
)
def test_fmt_tau_short_scientific(tau, expected):
    assert _fmt_tau(tau) == expected


def test_arm_label_splits_pal_loggap_by_tau():
    assert _arm_label(_cfg("pal_loggap", tau=1e-4)) == "pal_loggap[tau=1e-4]"
    assert _arm_label(_cfg("pal_loggap", tau=1e-2)) == "pal_loggap[tau=1e-2]"
    # distinct taus are distinct arms
    assert _arm_label(_cfg("pal_loggap", tau=1e-4)) != _arm_label(_cfg("pal_loggap", tau=1e-2))


def test_arm_label_tau_absent_is_default():
    assert _arm_label(_cfg("pal_loggap", tau=None)) == "pal_loggap[tau=default]"
    assert _arm_label({"method": "pal_loggap"}) == "pal_loggap[tau=default]"


def test_arm_label_other_methods_are_the_method_string():
    assert _arm_label(_cfg("alm")) == "alm"
    assert _arm_label(_cfg("dc3")) == "dc3"


def test_arm_label_override_sibling_is_its_own_arm():
    """`alm_bolton` evaluated off an `alm` run is a SEPARATE arm."""
    src = _cfg("alm", seed=3)
    sibling = _cfg(
        "alm_bolton",
        seed=3,
        **{OVERRIDE_SOURCE_KEY: "20260725T00Z_alm_curvature_hinge_k4_seed3_abc",
           "override_source_method": "alm"},
    )
    assert _arm_label(src) == "alm"
    assert _arm_label(sibling) == "alm_bolton"


def test_variant_sort_is_numeric_in_kappa_index():
    variants = ["k12", "k0", "k11", "k4", "k10", "k13", "k8", "k6"]
    assert sorted(variants, key=_variant_sort_key) == [
        "k0", "k4", "k6", "k8", "k10", "k11", "k12", "k13",
    ]


def test_variant_sort_k11_after_k10_not_lexicographic():
    assert _variant_sort_key("k10") < _variant_sort_key("k11")
    assert _variant_sort_key("k2") < _variant_sort_key("k10")
    assert sorted(["k10", "k1", "k2"], key=_variant_sort_key) == ["k1", "k2", "k10"]


def test_variant_sort_unparsable_goes_last():
    assert sorted(["weird", "k9"], key=_variant_sort_key) == ["k9", "weird"]


def test_mean_std_sample_std():
    mean, std = _mean_std([1.0, 2.0, 3.0])
    assert mean == pytest.approx(2.0)
    assert std == pytest.approx(1.0)  # ddof=1, not the population 0.8165


def test_mean_std_single_value_has_no_std():
    assert _mean_std([4.0]) == (4.0, None)


def test_mean_std_empty_is_all_na():
    assert _mean_std([]) == (None, None)


def test_aggregate_metric_skips_na_values():
    recs = [_rec(0, gap=1.0), _rec(1, gap=None), _rec(2, gap=3.0)]
    agg = _aggregate_metric(recs, "gap_mean")
    assert agg["n"] == 2
    assert agg["n_runs"] == 3
    assert agg["mean"] == pytest.approx(2.0)
    assert agg["values"] == {"0": 1.0, "1": None, "2": 3.0}


def test_aggregate_metric_all_na():
    agg = _aggregate_metric([_rec(0, contraction=None), _rec(1, contraction=None)], "contraction")
    assert (agg["mean"], agg["std"], agg["n"]) == (None, None, 0)


def test_zero_feasible_run_is_na_for_gap_but_counts_for_feasibility():
    recs = [
        _rec(0, feas=1.0, gap=2.0, n_feas=512.0),
        _rec(1, feas=0.0, gap=None, n_feas=0.0),  # nothing feasible -> gap NA
    ]
    (cell,) = _build_cells(recs, expected_seeds=[0, 1])
    assert cell["metrics"]["feas"]["n"] == 2
    assert cell["metrics"]["feas"]["mean"] == pytest.approx(0.5)
    assert cell["metrics"]["gap"]["n"] == 1
    assert cell["metrics"]["gap"]["n_runs"] == 2
    assert cell["metrics"]["gap"]["mean"] == pytest.approx(2.0)
    assert cell["metrics"]["n_feas"]["mean"] == pytest.approx(256.0)


def test_gap_cell_annotates_how_many_seeds_contributed():
    recs = [_rec(0, gap=2.0), _rec(1, gap=None)]
    (cell,) = _build_cells(recs, expected_seeds=[0, 1])
    text = _campaign_cell_text(cell, "gap", "{:+.3e}", "{:.1e}")
    assert text.endswith("(1/2)")
    # a complete cell carries no annotation
    (full,) = _build_cells([_rec(0, gap=2.0), _rec(1, gap=4.0)], expected_seeds=[0, 1])
    assert "(" not in _campaign_cell_text(full, "gap", "{:+.3e}", "{:.1e}")


def test_all_na_cell_renders_dash_and_single_seed_renders_dash_std():
    (na_cell,) = _build_cells([_rec(0, contraction=None)], expected_seeds=[0])
    assert _campaign_cell_text(na_cell, "contraction", "{:.3f}", "{:.3f}") == DASH
    (one,) = _build_cells([_rec(0, feas=0.5)], expected_seeds=[0])
    assert _campaign_cell_text(one, "feas", "{:.3f}", "{:.3f}") == f"0.500 +/- {DASH}"


def test_mean_std_cell_text():
    recs = [_rec(0, feas=0.8), _rec(1, feas=1.0)]
    (cell,) = _build_cells(recs, expected_seeds=[0, 1])
    assert _campaign_cell_text(cell, "feas", "{:.3f}", "{:.3f}") == "0.900 +/- 0.141"


def test_missing_seeds_are_detected_against_expect_seeds():
    recs = [_rec(s) for s in (0, 1, 2, 4, 9)]
    (cell,) = _build_cells(recs, expected_seeds=_parse_seed_spec("0-9"))
    assert cell["n_completed"] == 5
    assert cell["n_expected"] == 10
    assert cell["missing_seeds"] == [3, 5, 6, 7, 8]


def test_skipped_run_is_missing_and_carries_its_reason():
    recs = [_rec(0), _rec(1, skip_reason="missing model.pt")]
    (cell,) = _build_cells(recs, expected_seeds=[0, 1])
    assert cell["n_completed"] == 1
    assert cell["missing_seeds"] == [1]
    assert "missing model.pt" in cell["skipped"]["1"]
    report = _render_campaign(Path("runs"), SHA, [0, 1], [cell], [], [])
    assert "missing seed(s) 1" in report.markdown
    assert "run dir exists but was skipped: missing model.pt" in report.markdown


def test_without_expect_seeds_expected_equals_completed():
    recs = [_rec(0), _rec(1)]
    (cell,) = _build_cells(recs, expected_seeds=None)
    assert (cell["n_completed"], cell["n_expected"]) == (2, 2)
    assert cell["missing_seeds"] == []


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("0-9", list(range(10))),
        ("0,1,2", [0, 1, 2]),
        ("0-4,7", [0, 1, 2, 3, 4, 7]),
        ("3", [3]),
        (" 0 - 2 , 5 ", [0, 1, 2, 5]),
    ],
)
def test_parse_seed_spec(spec, expected):
    assert _parse_seed_spec(spec) == expected


@pytest.mark.parametrize("bad", ["", "abc", "5-1", "1-", "1-2-3"])
def test_parse_seed_spec_rejects_garbage(bad):
    with pytest.raises(SystemExit):
        _parse_seed_spec(bad)


def test_duplicate_arm_variant_seed_is_a_hard_error(tmp_path):
    cfg = _cfg("alm", seed=0)
    a = _make_run(tmp_path, "runA", cfg=cfg)
    b = _make_run(tmp_path, "runB", cfg=cfg)
    refs = [_ref(a, cfg), _ref(b, cfg)]
    with pytest.raises(SystemExit) as exc:
        _check_duplicates(refs)
    msg = str(exc.value)
    assert "runA" in msg and "runB" in msg
    assert "duplicate" in msg


def test_same_seed_in_different_arms_is_not_a_duplicate(tmp_path):
    src_cfg = _cfg("alm", seed=0)
    sib_cfg = _cfg("alm_bolton", seed=0, **{OVERRIDE_SOURCE_KEY: "runA"})
    refs = [
        _ref(_make_run(tmp_path, "runA", cfg=src_cfg), src_cfg),
        _ref(_make_run(tmp_path, "runA__as_alm_bolton", cfg=sib_cfg), sib_cfg),
    ]
    _check_duplicates(refs)  # must not raise


def test_scan_reports_skip_reasons_instead_of_dropping_runs(tmp_path):
    _make_run(tmp_path, "ok", cfg=_cfg("alm", seed=0))
    _make_run(tmp_path, "no_model", cfg=_cfg("alm", seed=1), model=False)
    _make_run(tmp_path, "no_rows", cfg=_cfg("alm", seed=2), rows=False)
    _make_run(tmp_path, "not_a_dial", cfg=_cfg("alm", bench_id="s2_active_set_switch", seed=3))
    bad = tmp_path / "bad_cfg"
    bad.mkdir()
    (bad / "config.json").write_text("{not json")

    refs, unplaceable = _scan_campaign(tmp_path)
    by_name = {r.run_dir.name: r for r in refs}
    assert set(by_name) == {"ok", "no_model", "no_rows"}  # non-dial run filtered out
    assert by_name["ok"].skip_reason is None
    assert by_name["no_model"].skip_reason == "missing model.pt"
    assert by_name["no_rows"].skip_reason == "missing eval_rows.parquet"
    assert by_name["ok"].variant == "k4"
    assert [p.name for p, _ in unplaceable] == ["bad_cfg"]


def test_sha_mismatch_is_a_hard_error(tmp_path):
    cfg = _cfg("alm", sha="b" * 40)
    refs = [_ref(_make_run(tmp_path, "stale", cfg=cfg), cfg)]
    with pytest.raises(SystemExit) as exc:
        _check_checkout_stability(refs, head_sha=SHA, allow_mismatch=False)
    assert "stale" in str(exc.value)
    assert "--allow-sha-mismatch" in str(exc.value)


def test_sha_mismatch_downgrades_to_warning_when_allowed(tmp_path):
    cfg = _cfg("alm", sha="b" * 40)
    refs = [_ref(_make_run(tmp_path, "stale", cfg=cfg), cfg)]
    warnings = _check_checkout_stability(refs, head_sha=SHA, allow_mismatch=True)
    assert any("SHA MISMATCH" in w and "stale" in w for w in warnings)


def test_matching_sha_and_clean_diff_warns_about_nothing(tmp_path):
    cfg = _cfg("alm")
    refs = [_ref(_make_run(tmp_path, "clean", cfg=cfg), cfg)]
    assert _check_checkout_stability(refs, head_sha=SHA, allow_mismatch=False) == []


def test_missing_sha_in_config_counts_as_mismatch(tmp_path):
    cfg = _cfg("alm")
    del cfg["pal_git_sha"]
    refs = [_ref(_make_run(tmp_path, "nosha", cfg=cfg), cfg)]
    with pytest.raises(SystemExit):
        _check_checkout_stability(refs, head_sha=SHA, allow_mismatch=False)


def test_uncommitted_changes_warn_even_when_the_sha_matches(tmp_path):
    cfg = _cfg("alm", diff="diff --git a/pal/x.py b/pal/x.py\n+1")
    refs = [_ref(_make_run(tmp_path, "dirty", cfg=cfg), cfg)]
    warnings = _check_checkout_stability(refs, head_sha=SHA, allow_mismatch=False)
    assert any("uncommitted changes" in w and "dirty" in w for w in warnings)


def test_unavailable_head_sha_is_fatal_unless_allowed(tmp_path):
    cfg = _cfg("alm")
    refs = [_ref(_make_run(tmp_path, "run", cfg=cfg), cfg)]
    with pytest.raises(SystemExit):
        _check_checkout_stability(refs, head_sha=None, allow_mismatch=False)
    assert _check_checkout_stability(refs, head_sha=None, allow_mismatch=True)


def test_contraction_uses_the_last_point_carrying_both_keys(tmp_path):
    run = _make_run(
        tmp_path,
        "r",
        metrics=[
            {"step": 0, "repair_c_pre": 1.0, "repair_c_post": 0.9},
            {"step": 10, "loss": 3.0},  # no keys: ignored
            {"step": 20, "repair_c_pre": 0.5, "repair_c_post": 0.05},
            {"step": 30, "loss": 1.0},
        ],
    )
    assert _final_repair_contraction(run) == pytest.approx(0.1)


def test_contraction_is_na_when_c_pre_has_collapsed(tmp_path):
    run = _make_run(
        tmp_path, "r", metrics=[{"repair_c_pre": 1e-15, "repair_c_post": 1e-16}]
    )
    assert _final_repair_contraction(run) is None


def test_contraction_is_na_not_zero_for_methods_that_never_log_it(tmp_path):
    run = _make_run(tmp_path, "r", metrics=[{"step": 0, "loss": 1.0, "grad_norm": 2.0}])
    assert _final_repair_contraction(run) is None


def test_contraction_is_na_without_metrics_file(tmp_path):
    assert _final_repair_contraction(_make_run(tmp_path, "r")) is None


def test_contraction_ignores_non_finite_and_unparsable_rows(tmp_path):
    run = _make_run(tmp_path, "r", metrics=[{"repair_c_pre": 1.0, "repair_c_post": 0.25}])
    (run / "metrics.jsonl").write_text(
        json.dumps({"repair_c_pre": 1.0, "repair_c_post": 0.25}) + "\n"
        + "{ broken json\n"
        + '{"repair_c_pre": "NaN-ish", "repair_c_post": null}\n'
    )
    assert _final_repair_contraction(run) == pytest.approx(0.25)


def _share_rows(value: float) -> list[dict]:
    return [{"grad_norm_con": value, "grad_norm_tot": 1.0}]


def test_grad_share_reads_the_runs_own_metrics_when_present(tmp_path):
    cfg = _cfg("alm")
    run = _make_run(tmp_path, "solo", cfg=cfg, metrics=_share_rows(0.25))
    assert _grad_share_source_dir(run, cfg) == (run, False)
    assert _campaign_grad_share(run, cfg) == (pytest.approx(0.25), False)


def test_grad_share_falls_back_to_the_source_run_via_the_as_suffix(tmp_path):
    src_name = "20260725T00Z_alm_curvature_hinge_k4_seed3_abc"
    src = _make_run(tmp_path, src_name, cfg=_cfg("alm", seed=3), metrics=_share_rows(0.4))
    sib_cfg = _cfg("alm_bolton", seed=3, **{OVERRIDE_SOURCE_KEY: src_name})
    sib = _make_run(tmp_path, f"{src_name}__as_alm_bolton", cfg=sib_cfg)
    assert _grad_share_source_dir(sib, sib_cfg) == (src, True)
    value, shared = _campaign_grad_share(sib, sib_cfg)
    assert value == pytest.approx(0.4)
    assert shared is True


def test_grad_share_falls_back_via_override_source_run_id_when_the_name_differs(tmp_path):
    src_name = "20260725T00Z_alm_curvature_hinge_k4_seed3_abc"
    src = _make_run(tmp_path, src_name, cfg=_cfg("alm", seed=3), metrics=_share_rows(0.6))
    sib_cfg = _cfg("alm_bolton", seed=3, **{OVERRIDE_SOURCE_KEY: src_name})
    sib = _make_run(tmp_path, "renamed_sibling", cfg=sib_cfg)
    assert _grad_share_source_dir(sib, sib_cfg) == (src, True)


def test_grad_share_is_na_when_no_source_can_be_resolved(tmp_path):
    sib_cfg = _cfg("alm_bolton", seed=3, **{OVERRIDE_SOURCE_KEY: "gone"})
    sib = _make_run(tmp_path, "orphan__as_alm_bolton", cfg=sib_cfg)
    assert _grad_share_source_dir(sib, sib_cfg) == (None, False)
    assert _campaign_grad_share(sib, sib_cfg) == (None, False)
    plain_cfg = _cfg("alm")
    plain = _make_run(tmp_path, "plain", cfg=plain_cfg)
    assert _campaign_grad_share(plain, plain_cfg) == (None, False)


def test_shared_grad_share_is_marked_in_the_table():
    (cell,) = _build_cells([_rec(0, grad_share=0.4, shared=True)], expected_seeds=[0])
    assert cell["grad_share_shared"] is True
    assert _campaign_cell_text(cell, "grad_share", "{:.3f}", "{:.3f}").endswith(SHARED_MARKER)
    (own,) = _build_cells([_rec(0, grad_share=0.4, shared=False)], expected_seeds=[0])
    assert SHARED_MARKER not in _campaign_cell_text(own, "grad_share", "{:.3f}", "{:.3f}")


def _campaign_cells():
    recs = []
    for arm in ("alm", "pal_loggap[tau=1e-4]"):
        for variant in ("k0", "k10", "k11", "k4"):
            for seed in (0, 1):
                recs.append(_rec(seed, arm=arm, variant=variant))
    return _build_cells(recs, expected_seeds=[0, 1])


def test_table_columns_follow_the_kappa_order():
    cells = _campaign_cells()
    header = _render_campaign_table(cells, _campaign_columns(cells)).splitlines()[2]
    assert header.startswith("| arm | metric | k0 | k4 | k10 | k11 |")


def test_columns_fall_back_to_the_full_benchmark_id_on_a_variant_clash():
    """Two dial families under one root must not share a `k4` column."""
    curvature_hinge = _rec(0, arm="alm", variant="k4")
    curvature_warp = _rec(0, arm="alm", variant="k4")
    curvature_warp.benchmark_id = "curvature_warp_k4"
    cells = _build_cells([curvature_hinge, curvature_warp], expected_seeds=[0])
    assert len(cells) == 2  # cells are keyed by (arm, benchmark_id), not by variant
    assert [lab for _, lab in _campaign_columns(cells)] == [
        "curvature_hinge_k4",
        "curvature_warp_k4",
    ]


def test_table_has_one_row_block_per_arm_with_a_seed_row():
    cells = _campaign_cells()
    table = _render_campaign_table(cells, _campaign_columns(cells))
    assert table.count("| alm |") == 1  # arm name printed once, on the block's first row
    assert table.count("| pal_loggap[tau=1e-4] |") == 1
    assert table.count("| seeds |") == 2
    assert "2/2" in table


def test_json_payload_carries_per_cell_aggregates_and_per_seed_values():
    cells = _campaign_cells()
    report = _render_campaign(Path("runs/c"), SHA, [0, 1], cells, ["a warning"], [])
    payload = report.payload
    assert payload["head_sha"] == SHA
    assert payload["expected_seeds"] == [0, 1]
    assert payload["variants"] == ["k0", "k4", "k10", "k11"]
    assert payload["arms"] == ["alm", "pal_loggap[tau=1e-4]"]
    assert payload["warnings"] == ["a warning"]
    cell = payload["cells"][0]
    assert cell["metrics"]["feas"]["values"] == {"0": 1.0, "1": 1.0}
    assert cell["metrics"]["gap"]["mean"] == pytest.approx(1e-3)
    assert set(cell["run_dirs"]) == {"0", "1"}
    json.dumps(payload)  # must be serialisable as-is


def test_warnings_section_says_none_when_everything_is_complete():
    report = _render_campaign(Path("runs/c"), SHA, [0, 1], _campaign_cells(), [], [])
    assert "### WARNINGS" in report.markdown
    assert "- none." in report.markdown


def test_unplaceable_runs_surface_in_warnings():
    report = _render_campaign(
        Path("runs/c"), SHA, None, _campaign_cells(), [], [(Path("runs/c/bad"), "config.json unreadable")]
    )
    assert "[unplaceable] bad" in report.markdown
