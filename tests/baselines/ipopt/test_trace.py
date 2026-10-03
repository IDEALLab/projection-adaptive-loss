"""Tests for the IPOPT solve tracer (solver wiring tests need cyipopt)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pal.baselines.ipopt.trace import SolveTracer, trace_dir_from_env


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def test_trace_dir_from_env(monkeypatch):
    monkeypatch.delenv("PAL_IPOPT_TRACE", raising=False)
    assert trace_dir_from_env() is None
    monkeypatch.setenv("PAL_IPOPT_TRACE", "")
    assert trace_dir_from_env() is None
    monkeypatch.setenv("PAL_IPOPT_TRACE", "/some/dir")
    assert trace_dir_from_env() == "/some/dir"


def test_tracer_writes_expected_records(tmp_path):
    names = ["c0", "c1", "c2"]
    types = ["ineq", "ineq", "eq"]
    tr = SolveTracer(str(tmp_path), query_idx=3, restart_idx=1,
                     constraint_names=names, constraint_types=types)

    tr.record_iter(alg_mod=0, iter_count=0, obj_value=10.0, inf_pr=1e-1,
                   inf_du=5.0, mu=1e-1, d_norm=2.0, regularization_size=0.0,
                   alpha_du=1.0, alpha_pr=1.0, ls_trials=1)
    tr.record_iter(alg_mod=1, iter_count=1, obj_value=9.0, inf_pr=1e-3,
                   inf_du=4e9, mu=1.3e-1, d_norm=1.0, regularization_size=1e-8,
                   alpha_du=0.5, alpha_pr=0.5, ls_trials=3)

    grad = np.array([3.0, 4.0])  # norm 5
    jac = np.array([[1.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    g_bad = np.array([0.5, 0.5, 0.5])
    tr.record_deriv(grad, jac, g_bad)
    tr.maybe_update_best(np.array([1.0, 1.0]), obj=9.0, g=g_bad,
                         per_constraint_viol=np.array([0.5, 0.5, 0.5]),
                         max_viol=0.5)
    g_good = np.array([-1.0, -1.0, 0.01])
    tr.maybe_update_best(np.array([2.0, 2.0]), obj=8.0, g=g_good,
                         per_constraint_viol=np.array([0.0, 0.0, 0.01]),
                         max_viol=0.01)
    # a worse point must not overwrite best
    tr.maybe_update_best(np.array([9.0, 9.0]), obj=1.0, g=g_bad,
                         per_constraint_viol=np.array([0.9, 0.9, 0.9]),
                         max_viol=0.9)

    tr.finalize(
        status=-1, status_msg="Maximum_Iterations_Exceeded", n_iter=1,
        wall_s=42.0, x0=np.array([0.0, 0.0]), x_final=np.array([2.0, 2.0]),
        obj_final=8.0, g_final=g_good, viol_final=np.array([0.0, 0.0, 0.01]),
        max_viol_final=0.01, paper_tolerance=1e-4,
        timing_buckets={"t_forward": 1.0, "n_cache_miss": 5},
        cb_counts={"obj": 10, "grad": 5, "cons": 10, "jac": 5},
    )

    rows = _read_jsonl(tmp_path / "q0003_r001.jsonl")
    kinds = [r["kind"] for r in rows]
    assert kinds.count("iter") == 2
    assert kinds.count("deriv") == 1
    assert kinds.count("final") == 1

    iter1 = [r for r in rows if r["kind"] == "iter"][1]
    assert iter1["restoration"] == 1 and iter1["inf_du"] == 4e9

    deriv = [r for r in rows if r["kind"] == "deriv"][0]
    assert deriv["grad_f_norm"] == pytest.approx(5.0)
    assert deriv["jac_row_norm_max"] == pytest.approx(2.0)

    final = [r for r in rows if r["kind"] == "final"][0]
    assert final["status"] == -1
    assert final["status_msg"] == "Maximum_Iterations_Exceeded"
    assert final["max_viol_final"] == pytest.approx(0.01)
    # 0.01 > 1e-4 -> infeasible at both tolerances
    assert final["feasible_at_1e-4"] is False
    assert final["feasible_at_paper_tol"] is False
    # best-iterate must be the 0.01-violation point, not the later 0.9 point
    assert final["best_max_viol"] == pytest.approx(0.01)
    assert final["best_obj"] == pytest.approx(8.0)
    assert final["viol_final_named"] == {"c0": 0.0, "c1": 0.0, "c2": pytest.approx(0.01)}

    arr = np.load(tmp_path / "q0003_r001.npz")
    assert np.allclose(arr["x0"], [0.0, 0.0])
    assert np.allclose(arr["x_final"], [2.0, 2.0])
    assert np.allclose(arr["x_best"], [2.0, 2.0])


def test_tracer_feasible_flags(tmp_path):
    tr = SolveTracer(str(tmp_path), 0, 0, ["c0"], ["ineq"])
    tr.finalize(status=0, status_msg="Solve_Succeeded", n_iter=12, wall_s=1.0,
                x0=np.array([0.0]), x_final=np.array([1.0]), obj_final=2.0,
                g_final=np.array([-1.0]), viol_final=np.array([0.0]),
                max_viol_final=0.0, paper_tolerance=1e-4,
                timing_buckets={}, cb_counts={})
    final = _read_jsonl(tmp_path / "q0000_r000.jsonl")[-1]
    assert final["feasible_at_1e-4"] is True
    assert final["feasible_at_paper_tol"] is True
    assert final["best_max_viol"] is None  # no best tracked


def test_tracer_nan_obj_is_infeasible(tmp_path):
    tr = SolveTracer(str(tmp_path), 0, 0, ["c0"], ["ineq"])
    tr.finalize(status=-13, status_msg="Invalid_Number_Detected", n_iter=3,
                wall_s=1.0, x0=np.array([0.0]), x_final=np.array([1.0]),
                obj_final=float("nan"), g_final=np.array([-1.0]),
                viol_final=np.array([0.0]), max_viol_final=0.0,
                paper_tolerance=1e-4, timing_buckets={}, cb_counts={})
    final = _read_jsonl(tmp_path / "q0000_r000.jsonl")[-1]
    assert final["feasible_at_1e-4"] is False
    assert final["obj_final"] is None  # non-finite coerced to null


cyipopt = pytest.importorskip("cyipopt")

import types  # noqa: E402

import torch  # noqa: E402

from pal.baselines.ipopt.solver import (  # noqa: E402
    IPOPTConfig,
    _effective_ipopt_options,
    _resolve_eval_queries,
    _sample_x0,
    _torch_dtype_from_spec,
)
from pal.benchmarks.base import BenchmarkSpec, Query  # noqa: E402


class _MiniBench:
    def __init__(self) -> None:
        self.spec = BenchmarkSpec(
            id="mini", family="mini", variant=None, dim=2, n_eq=0, n_ineq=1,
            constraint_names=["c0"], constraint_types=["ineq"],
            output_bounds=(torch.full((2,), -1.0), torch.full((2,), 1.0)),
            condition_dim=0, zeta_dim=2, tolerance=1e-4, cost="cheap",
            recommended_device="cpu", n_eval_default=4,
        )

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = n or 4
        g = torch.Generator("cpu").manual_seed(seed)
        return Query(zeta=torch.randn(n, 2, generator=g), conditions=torch.empty(n, 0))


def test_resolve_eval_queries_default(monkeypatch):
    monkeypatch.delenv("PAL_IPOPT_EVAL_POINTS", raising=False)
    q = _resolve_eval_queries(_MiniBench(), seed=0)
    assert q.zeta.shape == (4, 2)


def test_resolve_eval_queries_frozen_points(monkeypatch, tmp_path):
    pts = tmp_path / "pts.json"
    pts.write_text(json.dumps({
        "benchmark": "mini", "frozen_at": "x", "rationale": "y",
        "points": [{"zeta": [0.1, 0.2], "condition": []},
                   {"zeta": [0.3, 0.4], "condition": []}],
        "ipopt": {},
    }))
    monkeypatch.setenv("PAL_IPOPT_EVAL_POINTS", str(pts))
    q = _resolve_eval_queries(_MiniBench(), seed=0)
    assert q.zeta.shape == (2, 2)
    assert q.zeta[0].tolist() == pytest.approx([0.1, 0.2])


def _e1_stub():
    """Cheap e1/bwb stub bench (live=False, no surrogate artifacts)."""
    from pal.benchmarks.engineering.e1_bwb import E1BWB
    return E1BWB(live=False)


def test_tracer_bench_agnostic_on_e1(monkeypatch, tmp_path):
    import numpy as np  # noqa: F401

    from pal.baselines.ipopt.solver import IPOPTConfig, IPOPTSolver

    bench = _e1_stub()
    assert bench.spec.dim == 36 and len(bench.spec.constraint_types) == 3
    pts = tmp_path / "e1.json"
    pts.write_text(json.dumps({
        "benchmark": "e1", "frozen_at": "t", "rationale": "t",
        "points": [
            {"zeta": [0.0] * 16, "condition": [2000.0, 60.0]},
            {"zeta": [0.1] * 16, "condition": [2000.0, 60.0]},
        ],
        "ipopt": {},
    }))
    monkeypatch.setenv("PAL_IPOPT_EVAL_POINTS", str(pts))
    tdir = tmp_path / "trace"
    monkeypatch.setenv("PAL_IPOPT_TRACE", str(tdir))

    class _NL:
        def log_config(self, **k): ...
        def log_step(self, *a, **k): ...
        def log_final(self, **k): ...
        def finish(self, *a, **k): ...

    IPOPTSolver(IPOPTConfig(seed=0, multi_start=2, max_iter=30, device="cpu")).train(
        bench, seed=0, logger=_NL(),
    )
    jsonls = sorted(tdir.glob("*.jsonl"))
    assert len(jsonls) == 4  # 2 queries x 2 restarts
    final = [json.loads(x) for x in jsonls[0].read_text().splitlines() if x][-1]
    assert list(final["viol_final_named"].keys()) == [
        "lift_balance", "strain_agg", "tip_deflection",
    ]


def test_e1_stub_accepts_float64():
    """PAL_IPOPT_F64=1 is viable for e1 (plain-MLP surrogates take float64)."""
    bench = _e1_stub()
    x = torch.zeros(1, bench.spec.dim, dtype=torch.float64, requires_grad=True)
    cond = torch.tensor([[2000.0, 60.0]], dtype=torch.float64)
    obj, cons = bench.forward(x, cond)
    g = torch.autograd.grad(obj.sum(), x, retain_graph=True)[0]
    assert obj.dtype == torch.float64
    assert bool(torch.isfinite(g).all())
    assert len(cons) == 3


def test_effective_options_default(monkeypatch):
    for k in ("PAL_IPOPT_SCALING", "PAL_IPOPT_ACCEPTABLE"):
        monkeypatch.delenv(k, raising=False)
    opts = _effective_ipopt_options(IPOPTConfig(max_iter=500, tol=1e-8))
    assert opts["max_iter"] == 500 and opts["tol"] == 1e-8
    assert "nlp_scaling_method" not in opts and "acceptable_tol" not in opts


def test_effective_options_besteffort(monkeypatch):
    monkeypatch.setenv("PAL_IPOPT_SCALING", "gradient-based")
    monkeypatch.setenv("PAL_IPOPT_ACCEPTABLE", "1")
    opts = _effective_ipopt_options(IPOPTConfig())
    assert opts["nlp_scaling_method"] == "gradient-based"
    assert opts["acceptable_tol"] == 1e-4 and opts["acceptable_iter"] == 15


def test_torch_dtype_f64_knob(monkeypatch):
    spec = types.SimpleNamespace(precision="fp32")
    monkeypatch.delenv("PAL_IPOPT_F64", raising=False)
    assert _torch_dtype_from_spec(spec) is torch.float32
    monkeypatch.setenv("PAL_IPOPT_F64", "1")
    assert _torch_dtype_from_spec(spec) is torch.float64


def _fake_view(dim=10, lo=-100.0, hi=1200.0):
    import numpy as np
    v = types.SimpleNamespace()
    v.dim = dim
    v.lo = np.full(dim, lo, dtype=np.float64)
    v.hi = np.full(dim, hi, dtype=np.float64)
    return v


def test_init_box_is_byte_identical(monkeypatch):
    import numpy as np

    from pal.baselines.nlp_adapter import zeta_restart_rng
    monkeypatch.delenv("PAL_IPOPT_INIT_MODE", raising=False)
    v = _fake_view()
    zeta = torch.tensor([0.5, -0.3])
    got = _sample_x0(v, zeta, None, restart_idx=2)
    ref = zeta_restart_rng(zeta, None, 2).uniform(v.lo, v.hi)
    assert np.array_equal(got, ref)


def test_init_site_restricts_positions(monkeypatch):
    import numpy as np
    monkeypatch.setenv("PAL_IPOPT_INIT_MODE", "site")
    v = _fake_view(dim=10)  # 2*10//5 = 4 position slots
    x0 = _sample_x0(v, torch.tensor([0.1, 0.2]), None, restart_idx=0)
    assert np.all(x0[:4] >= 175.0) and np.all(x0[:4] <= 925.0)
    # non-position slots keep the full box (draws can exceed the site range)
    assert x0.shape == (10,)


def test_init_warmstart_uses_bench_layout(monkeypatch):
    monkeypatch.setenv("PAL_IPOPT_INIT_MODE", "warmstart")
    monkeypatch.setenv("PAL_IPOPT_WARMSTART_SIGMA", "0")  # deterministic
    n = 2
    inner = types.SimpleNamespace(
        make_initial_raw_params=lambda batch_size=1, device=None: {
            "cx": torch.full((1, n), 300.0), "cy": torch.full((1, n), 400.0),
            "w": torch.zeros(1, n), "d": torch.zeros(1, n), "h": torch.zeros(1, n),
        }
    )
    v = _fake_view(dim=5 * n)
    v.bench = types.SimpleNamespace(_bench=inner)
    x0 = _sample_x0(v, torch.tensor([0.0, 0.0]), None, restart_idx=0)
    assert x0[:n].tolist() == [300.0, 300.0]  # cx
    assert x0[n:2 * n].tolist() == [400.0, 400.0]  # cy
