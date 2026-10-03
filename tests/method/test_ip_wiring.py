"""Wiring tests for ``pal_ip``: hparams inherited from pal_loggap, CLI dispatch, mu0 plumbing."""

from __future__ import annotations

import dataclasses

from pal.method import PALIpConfig, PALIpSolver, PALLogGapConfig


def test_config_only_differs_in_repair_step() -> None:
    lg = {f.name: f.default for f in dataclasses.fields(PALLogGapConfig)}
    ip = {f.name: f.default for f in dataclasses.fields(PALIpConfig)}
    assert set(ip) - set(lg) == {"proj_ip_mu0", "proj_ip_fixed_mu"}
    differing = {k for k in lg if lg[k] != ip[k]}
    assert differing == {"proj_method"}
    assert PALIpConfig().proj_method == "ip"
    assert PALLogGapConfig().proj_method == "lm_k"
    # The knob is inert by default: None selects the R1 rule.
    assert PALIpConfig().proj_ip_fixed_mu is None
    assert PALIpConfig().proj_ip_mu0 == 0.1


def test_solver_name_and_inherited_train_loop() -> None:
    from pal.method.loggap.solver import PALLogGapSolver

    assert PALIpSolver.name == "pal_ip"
    assert issubclass(PALIpSolver, PALLogGapSolver)
    # `train` is inherited verbatim, the ablation must not fork the loop.
    assert PALIpSolver.train is PALLogGapSolver.train


def test_cli_dispatch() -> None:
    from pal.runner.cli import _SUPPORTED_METHODS, _build_solver

    assert "pal_ip" in _SUPPORTED_METHODS
    solver = _build_solver("pal_ip", PALIpConfig())
    assert isinstance(solver, PALIpSolver)


def test_mu0_reaches_the_inference_projector() -> None:
    from pal.benchmarks import get as get_benchmark

    bench = get_benchmark("s1_sphere_track")
    solver = PALIpSolver(
        PALIpConfig(device="cpu", proj_ip_mu0=42.0, proj_ip_fixed_mu=7.0)
    )
    projector, _fn, _iters, _tol = solver.build_inference_projector(bench)
    assert projector.method == "ip"
    assert projector.ip_mu0 == 42.0
    assert projector.ip_fixed_mu == 7.0
