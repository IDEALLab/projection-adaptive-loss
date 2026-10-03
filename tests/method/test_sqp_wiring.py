"""Wiring tests for ``pal_sqp``: hparams inherited from pal_loggap, CLI dispatch, rho plumbing."""

from __future__ import annotations

import dataclasses

import pytest

from pal.method import PALLogGapConfig, PALSqpConfig, PALSqpSolver

pytest.importorskip("qpsolvers", reason="pal_sqp repair step needs qpsolvers + osqp")


def test_config_only_differs_in_repair_step() -> None:
    lg = {f.name: f.default for f in dataclasses.fields(PALLogGapConfig)}
    sq = {f.name: f.default for f in dataclasses.fields(PALSqpConfig)}
    assert set(sq) - set(lg) == {"proj_sqp_rho"}
    differing = {k for k in lg if lg[k] != sq[k]}
    assert differing == {"proj_method"}
    assert PALSqpConfig().proj_method == "sqp"
    assert PALLogGapConfig().proj_method == "lm_k"


def test_solver_name_and_inherited_train_loop() -> None:
    from pal.method.loggap.solver import PALLogGapSolver

    assert PALSqpSolver.name == "pal_sqp"
    assert issubclass(PALSqpSolver, PALLogGapSolver)
    # `train` is inherited verbatim, the ablation must not fork the loop.
    assert PALSqpSolver.train is PALLogGapSolver.train


def test_cli_dispatch() -> None:
    from pal.runner.cli import _SUPPORTED_METHODS, _build_solver

    assert "pal_sqp" in _SUPPORTED_METHODS
    solver = _build_solver("pal_sqp", PALSqpConfig())
    assert isinstance(solver, PALSqpSolver)


def test_rho_reaches_the_inference_projector() -> None:
    from pal.benchmarks import get as get_benchmark

    bench = get_benchmark("s1_sphere_track")
    solver = PALSqpSolver(PALSqpConfig(device="cpu", proj_sqp_rho=42.0))
    projector, _fn, _iters, _tol = solver.build_inference_projector(bench)
    assert projector.method == "sqp"
    assert projector.sqp_rho == 42.0
