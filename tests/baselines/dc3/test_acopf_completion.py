"""Tests for the e3 two-step ACOPF completion used by DC3."""

from __future__ import annotations

import unittest

import torch

from pal.baselines.dc3._completion import (
    CompletionDivergedError as _GenericCompletionDivergedError,
)
from pal.baselines.dc3._completion_acopf import (
    ACOPFPartition,
    CompletionDivergedError,
    acopf_two_step_complete,
    build_partition,
    verify_one_gen_per_bus,
)
from pal.baselines.dc3.bench_specs import resolve_partial_vars
from pal.baselines.dc3.data_shim import _DC3DataShim, make_eq_resid_per_sample
from pal.benchmarks.engineering.e3_acopf.benchmark import E3ACOPF


def _build_realistic_Z(bench, partition: ACOPFPartition, B: int, seed: int = 0) -> torch.Tensor:
    """Build a physically plausible NN output: pg at midpoint, vm = 1.0."""
    lo, hi = bench.spec.output_bounds
    Z = 0.5 * (lo + hi)[partition.partial_vars].unsqueeze(0).expand(B, -1).clone()
    vm_start = bench.vm_start_yidx
    va_start = bench.va_start_yidx
    for i, yidx in enumerate(partition.partial_vars):
        if vm_start <= yidx < va_start:
            Z[:, i] = 1.0
    return Z


class PartitionInvariants(unittest.TestCase):
    """Structural checks, do not require a torch run; just index arithmetic."""

    def test_ieee30_sizes(self):
        b = E3ACOPF(case="ieee30")
        part = build_partition(b)
        self.assertEqual(len(part.partial_vars), 11)       # pg_pv(5) + vm_spv(6)
        self.assertEqual(len(part.known_vars), 1)          # slack va
        self.assertEqual(len(part.step1_vars), 53)         # vm_D(24) + va_non_slack(29)
        self.assertEqual(len(part.step2_vars), 7)          # pg_slack(1) + qg_all(6)
        self.assertEqual(len(part.step1_eqs), 53)
        self.assertEqual(len(part.step2_eqs), 7)
        self.assertEqual(part.ydim, 72)
        self.assertEqual(part.n_eq, 60)
        self.assertTrue(part.invariants_ok())

    def test_ieee57_sizes(self):
        b = E3ACOPF(case="ieee57")
        part = build_partition(b)
        self.assertEqual(len(part.partial_vars), 13)       # pg_pv(6) + vm_spv(7)
        self.assertEqual(len(part.known_vars), 1)          # slack va
        self.assertEqual(len(part.step1_vars), 106)        # vm_D(50) + va_non_slack(56)
        self.assertEqual(len(part.step2_vars), 8)          # pg_slack(1) + qg_all(7)
        self.assertEqual(len(part.step1_eqs), 106)
        self.assertEqual(len(part.step2_eqs), 8)
        self.assertEqual(part.ydim, 128)
        self.assertEqual(part.n_eq, 114)
        self.assertTrue(part.invariants_ok())

    def test_ieee118_sizes(self):
        b = E3ACOPF(case="ieee118")
        part = build_partition(b)
        self.assertEqual(len(part.partial_vars), 107)      # pg_pv(53) + vm_spv(54)
        self.assertEqual(len(part.known_vars), 1)
        self.assertEqual(len(part.step1_vars), 181)        # vm_D(64) + va_non_slack(117)
        self.assertEqual(len(part.step2_vars), 55)         # pg_slack(1) + qg_all(54)
        self.assertEqual(len(part.step1_eqs), 181)
        self.assertEqual(len(part.step2_eqs), 55)
        self.assertEqual(part.ydim, 344)
        self.assertEqual(part.n_eq, 236)
        self.assertTrue(part.invariants_ok())

    def test_y_index_groups_are_disjoint_and_cover_ydim(self):
        for case in ("ieee30", "ieee57", "ieee118"):
            b = E3ACOPF(case=case)
            part = build_partition(b)
            all_y = (
                set(part.partial_vars)
                | set(part.known_vars)
                | set(part.step1_vars)
                | set(part.step2_vars)
            )
            self.assertEqual(len(all_y), part.ydim, msg=f"case={case}")
            self.assertEqual(all_y, set(range(part.ydim)), msg=f"case={case}")

    def test_eq_index_groups_are_disjoint_and_cover_n_eq(self):
        for case in ("ieee30", "ieee57", "ieee118"):
            b = E3ACOPF(case=case)
            part = build_partition(b)
            all_eqs = set(part.step1_eqs) | set(part.step2_eqs)
            self.assertEqual(len(all_eqs), part.n_eq, msg=f"case={case}")
            self.assertEqual(all_eqs, set(range(part.n_eq)), msg=f"case={case}")

    def test_slack_va_is_known_not_partial(self):
        """Slack va must be in known_vars, not partial_vars."""
        for case in ("ieee30", "ieee57", "ieee118"):
            b = E3ACOPF(case=case)
            part = build_partition(b)
            slack_va_yidx = b.va_start_yidx + b.slack_bus_idx[0]
            self.assertIn(slack_va_yidx, part.known_vars, msg=f"case={case}")
            self.assertNotIn(slack_va_yidx, part.partial_vars, msg=f"case={case}")
            self.assertEqual(part.known_values, [0.0], msg=f"case={case}")


class R1Invariant(unittest.TestCase):
    """``verify_one_gen_per_bus``, load-bearing for Step 2's closed-form."""

    def test_ieee30_passes(self):
        verify_one_gen_per_bus(E3ACOPF(case="ieee30"))

    def test_ieee57_passes(self):
        verify_one_gen_per_bus(E3ACOPF(case="ieee57"))

    def test_ieee118_passes(self):
        verify_one_gen_per_bus(E3ACOPF(case="ieee118"))


class EqOrdering(unittest.TestCase):
    """Step 2 read-off requires p_balance x n_bus then q_balance x n_bus ordering."""

    def test_ordering_ieee30(self):
        self._check(E3ACOPF(case="ieee30"))

    def test_ordering_ieee57(self):
        self._check(E3ACOPF(case="ieee57"))

    def test_ordering_ieee118(self):
        self._check(E3ACOPF(case="ieee118"))

    def _check(self, b):
        n_bus = b._n_bus
        names = b.spec.constraint_names
        self.assertEqual(
            names[:n_bus],
            [f"p_balance_{j}" for j in range(n_bus)],
        )
        self.assertEqual(
            names[n_bus:2 * n_bus],
            [f"q_balance_{j}" for j in range(n_bus)],
        )


class CompletionResidual(unittest.TestCase):
    """After completion, Step-1 eqs converge and y lies in the gen box."""

    def test_ieee30(self):
        self._run_case("ieee30", tol=5e-5)

    def test_ieee57(self):
        self._run_case("ieee57", tol=5e-5)

    def test_ieee118(self):
        # IEEE-118 Step 1 Newton stalls near 1e-5 (fp32 floor).
        self._run_case("ieee118", tol=5e-5)

    def _run_case(self, case: str, tol: float):
        torch.manual_seed(0)
        b = E3ACOPF(case=case)
        part = build_partition(b)
        eq_fn = make_eq_resid_per_sample(b)
        B = 4
        q = b.sample_queries(n=B, split="eval", seed=0)
        X = q.conditions
        Z = _build_realistic_Z(b, part, B)

        diag: dict = {}
        Y = acopf_two_step_complete(eq_fn, X, Z, part, max_iter=50, tol=1e-5, diag=diag)
        self.assertEqual(Y.shape, (B, part.ydim))

        h = torch.stack([eq_fn(Y[i], X[i]) for i in range(B)])
        self.assertTrue(torch.isfinite(h).all())

        s1_idx = torch.as_tensor(part.step1_eqs, dtype=torch.long)
        h_s1 = h.index_select(1, s1_idx)
        self.assertLess(
            float(h_s1.abs().max()), tol,
            msg=f"{case}: step1 eq residual too large, Newton didn't converge",
        )

        s2_v = torch.as_tensor(part.step2_vars, dtype=torch.long)
        s2_lo = torch.as_tensor(part.step2_lo, dtype=Y.dtype)
        s2_hi = torch.as_tensor(part.step2_hi, dtype=Y.dtype)
        y_s2 = Y.index_select(1, s2_v)
        self.assertTrue(
            torch.all(y_s2 >= s2_lo - 1e-6) and torch.all(y_s2 <= s2_hi + 1e-6),
            msg=f"{case}: step2 y out of box",
        )

    def test_slack_va_slot_is_zero_in_Y(self):
        for case in ("ieee30", "ieee57", "ieee118"):
            b = E3ACOPF(case=case)
            part = build_partition(b)
            eq_fn = make_eq_resid_per_sample(b)
            B = 3
            q = b.sample_queries(n=B, split="eval", seed=1)
            X = q.conditions
            Z = _build_realistic_Z(b, part, B)

            Y = acopf_two_step_complete(eq_fn, X, Z, part, max_iter=50, tol=1e-5)
            slack_va_yidx = part.known_vars[0]
            self.assertTrue(
                torch.allclose(Y[:, slack_va_yidx], torch.zeros(B)),
                msg=f"case={case}: slack va slot drifted from 0.0",
            )


class SlackVAInertUnderCorrection(unittest.TestCase):
    """The correction step gives slack va zero gradient, so ``grad_steps`` leaves it fixed."""

    def test_ieee30(self):
        self._run_case("ieee30")

    def test_ieee57(self):
        self._run_case("ieee57")

    def test_ieee118(self):
        self._run_case("ieee118")

    def _run_case(self, case: str):
        torch.manual_seed(0)
        b = E3ACOPF(case=case)
        spec = resolve_partial_vars(b)
        assert spec is not None
        other_vars = sorted(
            set(range(b.spec.dim)) - set(spec.partial_vars) - set(spec.known_vars)
        )
        shim = _DC3DataShim(
            b,
            partial_vars=spec.partial_vars,
            other_vars=other_vars,
            linear=False,
            newton_max_iter=50,
            newton_tol=1e-5,
            newton_reg=1e-8,
            known_vars=spec.known_vars,
            known_values=spec.known_values,
            completion_strategy=spec.completion_strategy,
            acopf_partition=spec.meta["acopf_partition"],
        )
        B = 4
        q = b.sample_queries(n=B, split="train", seed=0)
        X = q.conditions
        shim.bind_x(X)

        part = spec.meta["acopf_partition"]
        Z = _build_realistic_Z(b, part, B)

        Y = shim.complete_partial(X, Z)
        slack_va_yidx = part.known_vars[0]
        self.assertTrue(
            torch.allclose(Y[:, slack_va_yidx], torch.zeros(B)),
            msg="slack va not 0 in Y after completion",
        )

        # The slack-va slot of the correction step must be 0.
        Y_step = shim.ineq_partial_grad(X, Y)
        self.assertTrue(torch.isfinite(Y_step).all())
        self.assertTrue(
            torch.allclose(Y_step[:, slack_va_yidx], torch.zeros(B)),
            msg=(
                f"case={case}: ineq_partial_grad returned non-zero motion on "
                f"slack va (yidx={slack_va_yidx}); grad_steps would move it "
                f"every correction iteration."
            ),
        )


class ExceptionSurface(unittest.TestCase):
    """The acopf module shares ``CompletionDivergedError`` with the generic completion path."""

    def test_exception_class_is_shared(self):
        self.assertIs(CompletionDivergedError, _GenericCompletionDivergedError)


class NonConvergenceGate(unittest.TestCase):
    """Step 1 hitting ``max_iter`` off the manifold must raise, not return ``Y``.

    The DC3 partial gradient ``-(J_other)^{-1} J_partial`` assumes ``h(x, y) = 0``.
    """

    def test_raises_on_max_iter_exhaustion(self):
        torch.manual_seed(0)
        b = E3ACOPF(case="ieee30")
        part = build_partition(b)
        eq_fn = make_eq_resid_per_sample(b)
        B = 2
        q = b.sample_queries(n=B, split="train", seed=0)
        X = q.conditions

        # vm=2.0 puts the residual past accept_floor and max_iter=0 skips the loop.
        Z = _build_realistic_Z(b, part, B)
        for i, yidx in enumerate(part.partial_vars):
            if b.vm_start_yidx <= yidx < b.va_start_yidx:
                Z[:, i] = 2.0

        with self.assertRaises(CompletionDivergedError) as cm:
            acopf_two_step_complete(eq_fn, X, Z, part, max_iter=0, tol=1e-8)
        self.assertIn("Newton diverged", str(cm.exception))
        self.assertIn("max_iter", str(cm.exception))


class BackpropThroughCompletion(unittest.TestCase):
    """NN -> completion -> objective chain gives finite gradients on Z."""

    def test_ieee30_gradients_finite(self):
        torch.manual_seed(0)
        b = E3ACOPF(case="ieee30")
        part = build_partition(b)
        eq_fn = make_eq_resid_per_sample(b)
        B = 2
        q = b.sample_queries(n=B, split="train", seed=0)
        X = q.conditions
        Z = _build_realistic_Z(b, part, B).clone().requires_grad_(True)

        Y = acopf_two_step_complete(eq_fn, X, Z, part, max_iter=50, tol=1e-5)
        loss = Y.pow(2).sum()
        loss.backward()
        self.assertIsNotNone(Z.grad)
        self.assertTrue(torch.isfinite(Z.grad).all())
        self.assertGreater(float(Z.grad.abs().max()), 0.0)


if __name__ == "__main__":
    unittest.main()
