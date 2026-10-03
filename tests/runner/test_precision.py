"""`pal run --precision`: flag plumbing, fp64 default-dtype wrap, query cast."""

from __future__ import annotations

import argparse

import pytest
import torch

from pal.benchmarks.base import Query
from pal.runner import cli as runner_cli
from pal.runner.cli import (
    _cast_queries,
    _default_dtype,
    _parse_args,
    _resolve_precision,
    _use_dtype,
)

BENCH_ID = "rosenbrock_eq"


def _spec(precision: str = "fp32") -> argparse.Namespace:
    return argparse.Namespace(precision=precision)


def test_cli_accepts_precision_fp64() -> None:
    args = _parse_args(
        ["run", "--method", "alm", "--benchmarks", BENCH_ID, "--precision", "fp64"]
    )
    assert args.precision == "fp64"


def test_cli_precision_defaults_to_none() -> None:
    args = _parse_args(["run", "--method", "alm", "--benchmarks", BENCH_ID])
    assert args.precision is None


def test_cli_rejects_unknown_precision() -> None:
    with pytest.raises(SystemExit):
        _parse_args(
            ["run", "--method", "alm", "--benchmarks", BENCH_ID, "--precision", "fp8"]
        )


def test_resolve_precision_cli_overrides_spec() -> None:
    assert _resolve_precision(argparse.Namespace(precision="fp64"), _spec("fp32")) == "fp64"


def test_resolve_precision_falls_back_to_spec() -> None:
    assert _resolve_precision(argparse.Namespace(precision=None), _spec("fp32")) == "fp32"
    assert _resolve_precision(argparse.Namespace(), _spec("bf16")) == "bf16"


def test_default_dtype_fp64_sets_and_restores() -> None:
    before = torch.get_default_dtype()
    with _default_dtype("fp64"):
        assert torch.get_default_dtype() is torch.float64
    assert torch.get_default_dtype() is before


def test_default_dtype_restores_on_exception() -> None:
    before = torch.get_default_dtype()
    with pytest.raises(RuntimeError), _default_dtype("fp64"):
        raise RuntimeError("boom")
    assert torch.get_default_dtype() is before


def test_default_dtype_fp32_is_a_noop() -> None:
    before = torch.get_default_dtype()
    with _default_dtype("fp32"):
        assert torch.get_default_dtype() is before
    assert torch.get_default_dtype() is before


def test_default_dtype_nesting_restores_outer_fp64() -> None:
    # DC3 sets float64 in its own train(); the nested restore must land on fp64.
    with _default_dtype("fp64"):
        with _default_dtype("fp64"):
            assert torch.get_default_dtype() is torch.float64
        assert torch.get_default_dtype() is torch.float64


def test_cast_queries_fp64_and_noop() -> None:
    q = Query(zeta=torch.zeros(2, 3), conditions=torch.zeros(2, 0))
    cast = _cast_queries(q, "fp64")
    assert cast.zeta.dtype is torch.float64
    assert cast.conditions.dtype is torch.float64
    assert _cast_queries(q, "fp32") is q


def test_fingerprint_is_precision_invariant() -> None:
    """fp32 -> fp64 is value-exact, so the fingerprint must not depend on the precision."""
    from pal.eval import query_sha256

    q = Query(zeta=torch.randn(4, 3), conditions=torch.randn(4, 2))
    assert query_sha256(q) == query_sha256(_cast_queries(q, "fp64"))


def test_use_dtype_sets_and_restores() -> None:
    before = torch.get_default_dtype()
    with _use_dtype(torch.float64):
        assert torch.get_default_dtype() is torch.float64
        # inner block can drop back to the native dtype without leaking
        with _use_dtype(before):
            assert torch.get_default_dtype() is before
        assert torch.get_default_dtype() is torch.float64
    assert torch.get_default_dtype() is before


def test_run_one_fp64_wraps_the_whole_cell(monkeypatch) -> None:
    """`_run_one` runs `_run_one_impl` under float64 and restores the previous dtype."""
    seen: dict[str, object] = {}

    def _fake_impl(
        method, bench_id, seed, args, runs_root, raw_bench, precision, native_dtype
    ):
        seen["dtype"] = torch.get_default_dtype()
        seen["precision"] = precision
        seen["spec_precision"] = raw_bench.spec.precision
        seen["native_dtype"] = native_dtype

    monkeypatch.setattr(runner_cli, "_run_one_impl", _fake_impl)
    args = _parse_args(
        ["run", "--method", "alm", "--benchmarks", BENCH_ID, "--precision", "fp64"]
    )
    before = torch.get_default_dtype()
    runner_cli._run_one("alm", BENCH_ID, 0, args, None)

    assert seen["dtype"] is torch.float64
    assert seen["precision"] == "fp64"
    # CLI override becomes the run's spec value (baselines read spec.precision).
    assert seen["spec_precision"] == "fp64"
    # The eval set is built at the native dtype so its fingerprint matches fp32.
    assert seen["native_dtype"] is before
    assert torch.get_default_dtype() is before


def test_run_one_without_flag_uses_spec_precision(monkeypatch) -> None:
    seen: dict[str, object] = {}

    def _fake_impl(
        method, bench_id, seed, args, runs_root, raw_bench, precision, native_dtype
    ):
        seen["dtype"] = torch.get_default_dtype()
        seen["precision"] = precision

    monkeypatch.setattr(runner_cli, "_run_one_impl", _fake_impl)
    args = _parse_args(["run", "--method", "alm", "--benchmarks", BENCH_ID])
    runner_cli._run_one("alm", BENCH_ID, 0, args, None)

    assert seen["precision"] == "fp32"
    assert seen["dtype"] is torch.float32
