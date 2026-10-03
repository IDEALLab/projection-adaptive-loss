"""CLI integration tests for slsqp/ipopt dispatch."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import pytest

from pal.baselines.slsqp import SLSQPConfig
from pal.runner.cli import (
    _CLI_DEFAULT_BATCH_SIZE,
    _SUPPORTED_METHODS,
    _build_cfg,
    _build_cfg_from_hparams,
    _build_solver,
)


def _ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        device="cpu",
        multi_start=None,
        max_iter=None,
        tol=None,
        epochs=None,
        batch_size=None,
        lr=None,
        eval_every=None,
        eval_samples=None,
        projection_log_every=None,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_supported_methods_include_slsqp_and_ipopt() -> None:
    assert "slsqp" in _SUPPORTED_METHODS
    assert "ipopt" in _SUPPORTED_METHODS


def test_build_cfg_slsqp_uses_dataclass_defaults_when_flags_unset() -> None:
    cfg = _build_cfg("slsqp", _ns(), seed=42)
    assert isinstance(cfg, SLSQPConfig)
    assert cfg.seed == 42
    assert cfg.device == "cpu"
    assert cfg.multi_start == SLSQPConfig().multi_start
    assert cfg.maxiter == SLSQPConfig().maxiter
    assert cfg.ftol == SLSQPConfig().ftol


def test_build_cfg_slsqp_applies_classical_flags() -> None:
    cfg = _build_cfg("slsqp", _ns(multi_start=5, max_iter=100, tol=1e-7), seed=0)
    assert cfg.multi_start == 5
    assert cfg.maxiter == 100
    assert cfg.ftol == 1e-7


def test_build_cfg_ipopt_applies_classical_flags() -> None:
    pytest.importorskip("cyipopt")
    cfg = _build_cfg("ipopt", _ns(multi_start=3, max_iter=250, tol=1e-9), seed=7)
    assert cfg.seed == 7
    assert cfg.multi_start == 3
    assert cfg.max_iter == 250
    assert cfg.tol == 1e-9


def test_build_solver_slsqp_dispatch() -> None:
    cfg = _build_cfg("slsqp", _ns(), seed=0)
    solver = _build_solver("slsqp", cfg)
    assert solver.name == "slsqp"


def test_build_solver_ipopt_dispatch_when_cyipopt_available() -> None:
    pytest.importorskip("cyipopt")
    cfg = _build_cfg("ipopt", _ns(), seed=0)
    solver = _build_solver("ipopt", cfg)
    assert solver.name == "ipopt"


def test_build_solver_ipopt_surfaces_friendly_error_without_cyipopt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An ImportError from cyipopt becomes a RuntimeError with install guidance."""
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("pal.baselines.ipopt"):
            raise ImportError("simulated missing cyipopt")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(RuntimeError, match="cyipopt"):
        _build_solver("ipopt", object())


def test_build_cfg_from_hparams_roundtrip_slsqp() -> None:
    cfg = _build_cfg("slsqp", _ns(multi_start=5, max_iter=150, tol=1e-8), seed=11)
    hparams = dataclasses.asdict(cfg)
    roundtripped = _build_cfg_from_hparams("slsqp", hparams)
    assert roundtripped == cfg


def test_build_cfg_from_hparams_roundtrip_ipopt() -> None:
    pytest.importorskip("cyipopt")
    cfg = _build_cfg("ipopt", _ns(multi_start=4, max_iter=500, tol=1e-8), seed=11)
    hparams = dataclasses.asdict(cfg)
    roundtripped = _build_cfg_from_hparams("ipopt", hparams)
    assert roundtripped == cfg


def test_build_cfg_from_hparams_drops_unknown_keys() -> None:
    hparams = dict(seed=0, multi_start=2, garbage_key="ignored", another_bad=99)
    cfg = _build_cfg_from_hparams("slsqp", hparams)
    assert cfg.multi_start == 2


def test_pal_run_then_eval_roundtrips_for_slsqp(tmp_path: Path) -> None:
    from pal.runner.cli import main

    runs_root = tmp_path / "runs"
    rc = main([
        "run",
        "--method", "slsqp",
        "--benchmarks", "rosenbrock_eq",
        "--seeds", "0",
        "--n-eval", "2",
        "--multi-start", "2",
        "--runs-root", str(runs_root),
    ])
    assert rc == 0

    run_dirs = list(runs_root.iterdir())
    assert len(run_dirs) == 1
    run_dir = run_dirs[0]

    config = json.loads((run_dir / "config.json").read_text())
    assert config["method"] == "slsqp"
    assert config["n_eval_effective"] == 2
    assert (run_dir / "final.json").exists()
    assert not (run_dir / "model.pt").exists()

    rc = main([
        "eval",
        "--run-id", run_dir.name,
        "--runs-root", str(runs_root),
        "--n-eval", "2",
    ])
    assert rc == 0


def test_pal_run_then_eval_roundtrips_for_ipopt(tmp_path: Path) -> None:
    pytest.importorskip("cyipopt")
    from pal.runner.cli import main

    runs_root = tmp_path / "runs"
    rc = main([
        "run",
        "--method", "ipopt",
        "--benchmarks", "rosenbrock_eq",
        "--seeds", "0",
        "--n-eval", "2",
        "--multi-start", "1",
        "--max-iter", "100",
        "--runs-root", str(runs_root),
    ])
    assert rc == 0

    run_dir = next(iter((tmp_path / "runs").iterdir()))
    config = json.loads((run_dir / "config.json").read_text())
    assert config["method"] == "ipopt"
    assert not (run_dir / "model.pt").exists()

    rc = main([
        "eval",
        "--run-id", run_dir.name,
        "--runs-root", str(runs_root),
        "--n-eval", "2",
    ])
    assert rc == 0


_E3 = "e3/acopf_ieee57"


def _pf_ns(**overrides) -> argparse.Namespace:
    """Namespace for `_build_cfg` on a learned method under --protocol."""
    defaults = dict(
        protocol="paper-faithful",
        batch_size=_CLI_DEFAULT_BATCH_SIZE,
        lr=1e-4,
        measure_repair_mem=False,
        measure_repair_mem_every=100,
        set_overrides=[],
    )
    defaults.update(overrides)
    return _ns(**defaults)


def test_build_cfg_fsnet_paper_faithful_on_e3() -> None:
    cfg = _build_cfg("fsnet", _pf_ns(), seed=0, bench_id=_E3)
    assert (cfg.hidden_dim, cfg.num_layers) == (200, 2)
    assert cfg.zeta_zero is True
    assert (cfg.epochs, cfg.batch_size, cfg.steps_per_epoch) == (2000, 200, 1)
    assert cfg.lr == 5e-4  # paper Table C.1, not the suite-parity YAML value
    # Solver-internal knobs stay at authors' defaults.
    assert (cfg.memory_size, cfg.max_iter, cfg.eq_pen_weight) == (30, 50, 10.0)


def test_build_cfg_snarenet_paper_faithful_on_e3() -> None:
    cfg = _build_cfg("snarenet", _pf_ns(), seed=0, bench_id=_E3)
    assert (cfg.hidden_size, cfg.num_hidden_layers) == (200, 2)
    assert cfg.zeta_zero is True
    assert (cfg.epochs, cfg.batch_size) == (2000, 200)
    assert cfg.rtol == 1e-6
    assert (cfg.learning_rate, cfg.lambd, cfg.soft_epochs) == (1e-4, 1e-2, 0)


def test_build_cfg_enforce_v4_paper_faithful_on_e3() -> None:
    cfg = _build_cfg("enforce_v4", _pf_ns(), seed=0, bench_id=_E3)
    assert (cfg.hidden, cfg.n_layers) == (200, 2)
    assert cfg.zeta_zero is True
    assert (cfg.epochs, cfg.batch_size, cfg.lr) == (2000, 200, 1e-4)
    assert cfg.epoch_start_hard_constrained == 0
    assert (cfg.eps_chol, cfg.max_it, cfg.weight_loss_displacement) == (1e-8, 100, 0.5)


def test_paper_faithful_baseline_overrides_are_e3_gated() -> None:
    for method in ("fsnet", "snarenet", "enforce_v4"):
        cfg = _build_cfg(method, _pf_ns(), seed=0, bench_id="s1_sphere_track")
        assert cfg.zeta_zero is False, method
        cfg = _build_cfg(method, _pf_ns(protocol="synthetic"), seed=0, bench_id=_E3)
        assert cfg.zeta_zero is False, method
