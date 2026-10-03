"""Atmosphere helper tests: ISA values, broadcasting, autograd."""

from __future__ import annotations

import math

import pytest
import torch

from pal.benchmarks.engineering.e1_bwb import atmosphere as atm


def test_sea_level_references():
    """Pinned to standard ISA sea-level values."""
    T = atm.temperature(torch.tensor(0.0))
    p = atm.pressure(torch.tensor(0.0))
    rho = atm.rho_air(torch.tensor(0.0))
    a = atm.speed_of_sound(torch.tensor(0.0))

    assert math.isclose(T.item(), 288.15, rel_tol=1e-6)
    assert math.isclose(p.item(), 101325.0, rel_tol=1e-6)
    assert math.isclose(rho.item(), 1.225, abs_tol=5e-3)
    assert math.isclose(a.item(), 340.29, abs_tol=0.5)


def test_altitude_monotonic_trends():
    alts = torch.linspace(0.0, 10000.0, 21)
    T = atm.temperature(alts)
    p = atm.pressure(alts)
    rho = atm.rho_air(alts)
    assert (T[1:] - T[:-1] < 0).all()
    assert (p[1:] - p[:-1] < 0).all()
    assert (rho[1:] - rho[:-1] < 0).all()


def test_batch_broadcast():
    alt = torch.tensor([0.0, 1000.0, 5000.0])
    V = torch.tensor([30.0, 60.0, 120.0])
    Ma = atm.mach(V, alt)
    q = atm.q_dyn(alt, V)
    Re = atm.reynolds(alt, V, L=1.5)

    for out in (Ma, q, Re):
        assert out.shape == (3,)
        assert torch.isfinite(out).all()

    assert Ma[2] > Ma[0]
    assert q[2] > q[0]


def test_grad_through_altitude():
    alt = torch.tensor([0.0, 2000.0, 5000.0], requires_grad=True)
    V = torch.tensor([40.0, 40.0, 40.0])
    q = atm.q_dyn(alt, V).sum()
    q.backward()
    # Lower air density at higher alt -> dq/d(alt) < 0.
    assert alt.grad is not None
    assert (alt.grad < 0).all()


def test_mach_matches_speed_of_sound():
    alt = torch.tensor(3000.0)
    a = atm.speed_of_sound(alt).item()
    V = 0.5 * a
    assert math.isclose(atm.mach(torch.tensor(V), alt).item(), 0.5, rel_tol=1e-6)


@pytest.mark.parametrize("alt,V,L", [(0.0, 30.0, 1.0), (5000.0, 60.0, 2.0)])
def test_reynolds_positive(alt, V, L):
    Re = atm.reynolds(torch.tensor(alt), torch.tensor(V), torch.tensor(L))
    assert Re.item() > 0
