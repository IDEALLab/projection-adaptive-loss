"""`jacobian_mode='loop'` and `'vmap_jacrev'` agree up to reduction-order noise."""

from __future__ import annotations

import pytest
import torch

from pal.benchmarks.synthetic.equality_dominated import EqualityDominated
from pal.benchmarks.synthetic.s5_overdetermined import S5Overdetermined
from pal.method.solver import _make_constraint_values_fn
from pal.projection import Projector


def _project_step(bench, jacobian_mode: str, B: int = 8, seed: int = 0):
    """Run one Projector.step() with the given jacobian_mode; return y_tilde, info and J."""
    torch.manual_seed(seed)
    spec = bench.spec
    K = spec.n_eq + spec.n_ineq
    D = spec.dim

    zeta = torch.randn(B, max(spec.zeta_dim, 1), requires_grad=False)
    linear = torch.nn.Linear(zeta.shape[1], D)
    y_hat = linear(zeta)

    if spec.condition_dim > 0:
        q = bench.sample_queries(B, split="train", seed=seed)
        conditions = q.conditions
    else:
        conditions = None

    values_fn = _make_constraint_values_fn(bench)

    obj_live, clist_live = bench.forward(y_hat, conditions)
    c_pre = torch.stack([c.value for c in clist_live], dim=-1)

    projector = Projector(
        n_constraints=K,
        constraint_types=list(spec.constraint_types),
        method="lm_k",
        prescale=False,
        jacobian_mode=jacobian_mode,
    )
    y_tilde, info = projector.step(y_hat, c_pre, values_fn, conditions)
    return y_tilde.detach(), info, c_pre.detach()


@pytest.mark.parametrize(
    "bench_cls,bench_name",
    [(S5Overdetermined, "s5_conditional_K5"), (EqualityDominated, "equality_dominated_uncond_K10")],
)
def test_jacobian_modes_agree(bench_cls, bench_name):
    """J_loop and J_vmap agree within fp reduction noise."""
    bench = bench_cls()

    _, info_loop, _ = _project_step(bench, "loop")
    _, info_vmap, _ = _project_step(bench, "vmap_jacrev")

    J_loop = info_loop["J"]
    J_vmap = info_vmap["J"]

    assert J_loop.shape == J_vmap.shape, (
        f"{bench_name}: shape mismatch {J_loop.shape} vs {J_vmap.shape}"
    )

    # Scale-aware tolerance: same math, different reduction orders.
    abs_diff = (J_loop - J_vmap).abs().max().item()
    rel_scale = J_loop.abs().max().item() + 1e-12
    assert abs_diff / rel_scale < 1e-4, (
        f"{bench_name}: J max rel diff {abs_diff / rel_scale:.2e} "
        f"(abs {abs_diff:.2e}, scale {rel_scale:.2e})"
    )


@pytest.mark.parametrize(
    "bench_cls",
    [S5Overdetermined, EqualityDominated],
)
def test_y_tilde_matches_across_modes(bench_cls):
    """Full step() output (y_tilde, c_post) agrees across modes."""
    bench = bench_cls()

    y_loop, info_loop, _ = _project_step(bench, "loop")
    y_vmap, info_vmap, _ = _project_step(bench, "vmap_jacrev")

    torch.testing.assert_close(y_loop, y_vmap, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(
        info_loop["c_post"], info_vmap["c_post"], rtol=1e-4, atol=1e-5,
    )


def test_invalid_mode_rejected():
    """Constructor validates the mode string."""
    with pytest.raises(ValueError, match="unknown jacobian_mode"):
        Projector(n_constraints=1, constraint_types=["eq"], jacobian_mode="bogus")


def test_default_mode_is_loop():
    """Default stays 'loop', counter-faithful on synthetic benches."""
    p = Projector(n_constraints=1, constraint_types=["eq"])
    assert p.jacobian_mode == "loop"
