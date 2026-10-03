"""Unit tests for Projector's hard-output-box clamping."""

from __future__ import annotations

import pytest
import torch

from pal.projection.projector import Projector


def _trivial_constraint_fn(y_target: torch.Tensor):
    """constraint_fn whose sole constraint is `y - y_target == 0`, dim-wise."""
    from types import SimpleNamespace

    dim = int(y_target.shape[-1])

    def fn(y, conditions):  # noqa: ARG001, conditions unused
        c_value = y - y_target  # [B, dim]
        clist = [
            SimpleNamespace(
                value=c_value[..., k],
                type="eq",
                name=f"eq_{k}",
            )
            for k in range(dim)
        ]
        obj = (y - y_target).pow(2).sum(dim=-1)
        return obj, clist

    return fn


class TestBoxValidation:
    def test_both_none_accepted(self) -> None:
        p = Projector(
            n_constraints=1,
            constraint_types=["eq"],
            box_lower=None,
            box_upper=None,
        )
        assert p.box_lower is None
        assert p.box_upper is None

    def test_only_lower_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be None"):
            Projector(
                n_constraints=1,
                constraint_types=["eq"],
                box_lower=torch.zeros(3),
                box_upper=None,
            )

    def test_only_upper_set_rejected(self) -> None:
        with pytest.raises(ValueError, match="both be set or both be None"):
            Projector(
                n_constraints=1,
                constraint_types=["eq"],
                box_lower=None,
                box_upper=torch.ones(3),
            )

    def test_shape_mismatch_rejected(self) -> None:
        with pytest.raises(ValueError, match="same shape"):
            Projector(
                n_constraints=1,
                constraint_types=["eq"],
                box_lower=torch.zeros(3),
                box_upper=torch.ones(4),
            )


class TestClampBehavior:
    def test_clamp_is_noop_when_unset(self) -> None:
        p = Projector(n_constraints=1, constraint_types=["eq"])
        y = torch.tensor([[-5.0, 5.0, 0.1]])
        assert torch.equal(p._clamp_to_box(y), y)

    def test_clamp_pushes_above_lower(self) -> None:
        p = Projector(
            n_constraints=1,
            constraint_types=["eq"],
            box_lower=torch.tensor([0.0, 0.0, 0.0]),
            box_upper=torch.tensor([1.0, 1.0, 1.0]),
        )
        y = torch.tensor([[-5.0, 0.5, 2.0]])
        out = p._clamp_to_box(y)
        assert torch.allclose(out, torch.tensor([[0.0, 0.5, 1.0]]))

    def test_clamp_handles_batch(self) -> None:
        p = Projector(
            n_constraints=1,
            constraint_types=["eq"],
            box_lower=torch.tensor([-1.0, -1.0]),
            box_upper=torch.tensor([1.0, 1.0]),
        )
        y = torch.tensor([
            [2.0, -3.0],
            [0.5, 0.5],
            [-10.0, 10.0],
        ])
        expected = torch.tensor([
            [1.0, -1.0],
            [0.5, 0.5],
            [-1.0, 1.0],
        ])
        assert torch.allclose(p._clamp_to_box(y), expected)

    def test_clamp_promotes_device_dtype(self) -> None:
        """Stored bounds should be promoted to match the queried y's dtype."""
        p = Projector(
            n_constraints=1,
            constraint_types=["eq"],
            box_lower=torch.tensor([0.0, 0.0], dtype=torch.float64),
            box_upper=torch.tensor([1.0, 1.0], dtype=torch.float64),
        )
        y = torch.tensor([[-0.5, 2.0]], dtype=torch.float32)
        out = p._clamp_to_box(y)
        assert out.dtype == torch.float32
        assert torch.allclose(out, torch.tensor([[0.0, 1.0]]))


class TestProjectStaysInBox:
    """project() must return an in-box point even when the constraint target is outside."""

    def test_project_clamps_output_when_target_outside_box(self) -> None:
        torch.manual_seed(0)
        dim = 2
        y_start = torch.tensor([[0.5, 0.5]], requires_grad=False)
        # y_target is outside the box, so project() should land on the box boundary.
        y_target = torch.tensor([[10.0, -10.0]])
        p = Projector(
            n_constraints=dim,
            constraint_types=["eq"] * dim,
            delta=0.0,
            box_lower=torch.tensor([-1.0, -1.0]),
            box_upper=torch.tensor([1.0, 1.0]),
            detach_j=True,
        )
        y_final, _info = p.project(
            y_start.clone(),
            _trivial_constraint_fn(y_target),
            conditions=None,
            max_iters=30,
            tol=1e-6,
        )
        assert torch.all(y_final >= torch.tensor([-1.0, -1.0]) - 1e-6)
        assert torch.all(y_final <= torch.tensor([1.0, 1.0]) + 1e-6)

    def test_project_without_box_is_unaffected(self) -> None:
        """Without a box, project() converges straight to an in-box target."""
        torch.manual_seed(0)
        dim = 2
        y_start = torch.tensor([[0.5, 0.5]])
        y_target = torch.tensor([[0.3, -0.2]])
        p_no_box = Projector(
            n_constraints=dim,
            constraint_types=["eq"] * dim,
            delta=0.0,
            detach_j=True,
        )
        y_final, _info = p_no_box.project(
            y_start.clone(),
            _trivial_constraint_fn(y_target),
            conditions=None,
            max_iters=30,
            tol=1e-6,
        )
        assert torch.allclose(y_final, y_target, atol=1e-4)
