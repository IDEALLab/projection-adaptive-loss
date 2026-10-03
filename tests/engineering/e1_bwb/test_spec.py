"""Smoke: scalar modules build cleanly and the spec has the expected shape."""

from __future__ import annotations

import torch

from pal.benchmarks.base import BenchmarkSpec
from pal.benchmarks.engineering.e1_bwb import DIM, decode_x, encode_x, make_spec
from pal.benchmarks.engineering.e1_bwb.spec import (
    N_STATIONS_DEFAULT,
    ZETA_DIM,
    constraint_layout,
)
from pal.benchmarks.engineering.e1_bwb.x_layout import default_bounds


def test_make_spec_default_shape():
    spec = make_spec()
    assert isinstance(spec, BenchmarkSpec)
    assert spec.id == "e1/bwb"
    assert spec.family == "e1"
    assert spec.variant == "bwb"
    assert spec.dim == DIM == 36
    assert spec.n_eq == 1
    # K = lift_balance + strain_agg + tip_deflection = 3, for any N_STATIONS_DEFAULT.
    assert spec.n_ineq == 2
    assert spec.constraint_types == ["eq", "ineq", "ineq"]
    assert spec.condition_dim == 2
    assert spec.zeta_dim == ZETA_DIM == 16
    assert spec.n_eval_default == 64
    assert spec.cost == "expensive"
    assert spec.recommended_device == "gpu"


def test_constraint_layout_ordering():
    names, types, n_eq, n_ineq = constraint_layout(N_STATIONS_DEFAULT)
    assert n_eq == 1
    assert n_ineq == 2
    assert names == ["lift_balance", "strain_agg", "tip_deflection"]
    assert types == ["eq", "ineq", "ineq"]
    assert types[0] == "eq"
    assert types[names.index("lift_balance")] == "eq"


def test_output_bounds_shape_and_order():
    lo, hi = default_bounds()
    assert lo.shape == (DIM,)
    assert hi.shape == (DIM,)
    assert (hi > lo).all()


def test_decode_encode_roundtrip():
    x = torch.randn(4, DIM)
    dec = decode_x(x)
    x2 = encode_x(dec.shape, dec.L, dec.struct, dec.battery, dec.alpha_cr)
    assert torch.equal(x, x2)
