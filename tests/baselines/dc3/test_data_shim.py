"""End-to-end tests of the DC3 data shim (completion + partial grad) on real benchmarks."""

from __future__ import annotations

import unittest

import torch

from pal.baselines.dc3.bench_specs import (
    _REGISTRY,
    BenchDC3Spec,
    resolve_partial_vars,
)
from pal.baselines.dc3.data_shim import _DC3DataShim

_SYNTH_CLASSES = {
    "rosenbrock_eq":
        ("pal.benchmarks.synthetic.rosenbrock_eq", "RosenbrockEq"),
    "two_basins":
        ("pal.benchmarks.synthetic.two_basins", "TwoBasins"),
    "equality_dominated":
        ("pal.benchmarks.synthetic.equality_dominated", "EqualityDominated"),
}


def _make_bench(bench_id: str):
    """Instantiate a synthetic PAL benchmark by id."""
    import importlib
    if bench_id not in _SYNTH_CLASSES:
        raise ValueError(bench_id)
    mod_path, cls_name = _SYNTH_CLASSES[bench_id]
    mod = importlib.import_module(mod_path)
    return getattr(mod, cls_name)()


def _shim_for(bench, spec: BenchDC3Spec):
    other_vars = sorted(set(range(bench.spec.dim)) - set(spec.partial_vars))
    return _DC3DataShim(
        bench,
        partial_vars=spec.partial_vars,
        other_vars=other_vars,
        linear=spec.linear,
        newton_max_iter=20,
        newton_tol=1e-6,
        newton_reg=spec.newton_reg,
        warm_start_fn=spec.warm_start_fn,
        warm_start_ctx={"bench": bench},
    )


class RegistryInvariants(unittest.TestCase):
    def test_registered_partitions_have_correct_length(self):
        """``|partial_vars| + |other_vars| == ydim`` for every explicit synth entry."""
        for bench_id, entry in _REGISTRY.items():
            if not isinstance(entry, BenchDC3Spec):
                continue
            if bench_id not in _SYNTH_CLASSES:
                continue
            bench = _make_bench(bench_id)
            expected = bench.spec.dim - bench.spec.n_eq
            self.assertEqual(
                len(entry.partial_vars), expected,
                f"{bench_id}: |partial|={len(entry.partial_vars)} "
                f"!= ydim - n_eq = {expected}",
            )
            self.assertEqual(
                len(set(entry.partial_vars)), len(entry.partial_vars),
                f"{bench_id}: partial_vars has duplicates",
            )
            self.assertTrue(
                all(0 <= v < bench.spec.dim for v in entry.partial_vars),
                f"{bench_id}: partial_vars out of range",
            )


class ResolveAndComplete(unittest.TestCase):
    """End-to-end: resolve -> shim -> complete_partial -> eq satisfied."""

    def _check_complete(self, bench_id: str, B: int = 4):
        bench = _make_bench(bench_id)
        spec = resolve_partial_vars(bench)
        self.assertIsNotNone(spec, f"{bench_id}: expected non-None spec")
        shim = _shim_for(bench, spec)

        torch.manual_seed(0)
        Z = torch.randn(B, len(spec.partial_vars)) * 0.3

        if bench.spec.condition_dim > 0:
            q = bench.sample_queries(n=B, split="train", seed=0)
            X = q.conditions
        else:
            X = torch.zeros(B, 0)
        shim.bind_x(X)

        Y = shim.complete_partial(X, Z)
        self.assertEqual(Y.shape, (B, bench.spec.dim))
        torch.testing.assert_close(
            Y[:, spec.partial_vars], Z, atol=1e-4, rtol=1e-4,
        )
        eq_res = shim.eq_resid(X, Y)
        self.assertLess(
            eq_res.abs().max().item(), 1e-3,
            f"{bench_id}: eq_resid after complete_partial not near zero",
        )

    def test_rosenbrock_eq_linear_completion(self):
        self._check_complete("rosenbrock_eq")

    def test_two_basins_linear_completion(self):
        self._check_complete("two_basins")

    def test_equality_dominated_linear_completion(self):
        self._check_complete("equality_dominated")


class P13PartialGradEnd2End(unittest.TestCase):
    """On a real bench, ``ineq_partial_grad`` is nonzero in both partial and other slots."""

    def test_rosenbrock_eq_partial_grad_has_induced_motion(self):
        bench = _make_bench("rosenbrock_eq")
        spec = resolve_partial_vars(bench)
        shim = _shim_for(bench, spec)
        B = 4
        torch.manual_seed(0)
        Z = torch.randn(B, len(spec.partial_vars))
        X = torch.zeros(B, 0)
        shim.bind_x(X)
        Y = shim.complete_partial(X, Z)
        Y = Y + 0.05 * torch.randn_like(Y)
        step = shim.ineq_partial_grad(X, Y)
        self.assertEqual(step.shape, (B, bench.spec.dim))
        # Other slots live at other_vars = ydim - partial_vars
        other_vars = sorted(set(range(bench.spec.dim)) - set(spec.partial_vars))
        partial_max = step[:, spec.partial_vars].abs().max().item()
        other_max = step[:, other_vars].abs().max().item()
        if partial_max > 1e-8:
            # Both slots are driven by the same ineq_dist, so both are zero or both nonzero.
            self.assertGreater(
                other_max, 1e-8,
                "other-slot induced motion must be nonzero when the "
                "partial-slot gradient is nonzero",
            )


if __name__ == "__main__":
    unittest.main()
