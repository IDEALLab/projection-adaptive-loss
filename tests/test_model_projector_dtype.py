"""fp64 dtype invariants for `pal.model` and `pal.projection.projector`."""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch

from pal.model import CoordinationMLP, DC3MLP
from pal.projection.projector import Projector

_DTYPES = [torch.float32, torch.float64]


@contextmanager
def _default_dtype(dtype: torch.dtype):
    prev = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(prev)


def _offending(module: torch.nn.Module, dtype: torch.dtype) -> list[str]:
    """Names of floating params/buffers whose dtype is not `dtype`."""
    named = list(module.named_parameters()) + list(module.named_buffers())
    return [n for n, t in named if t.is_floating_point() and t.dtype != dtype]


class TestModelBufferDtype:
    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_coordination_mlp_follows_default_dtype(self, dtype) -> None:
        with _default_dtype(dtype):
            model = CoordinationMLP(
                dim_zeta=2,
                dim_conditions=3,
                dim_output=4,
                output_bounds=[(-1.0, 1.0)] * 4,
                condition_bounds=[(0.0, 2.0)] * 3,
                hidden=8,
                n_layers=2,
            )
            assert _offending(model, dtype) == []
            out = model(torch.randn(5, 2), torch.randn(5, 3))
        assert out.dtype == dtype

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_dc3_mlp_follows_default_dtype(self, dtype) -> None:
        with _default_dtype(dtype):
            model = DC3MLP(
                dim_zeta=2,
                dim_conditions=3,
                dim_output=4,
                output_bounds=[(-1.0, 1.0)] * 4,
                condition_bounds=[(0.0, 2.0)] * 3,
                hidden=8,
                n_layers=2,
            )
            assert _offending(model, dtype) == []
            model.eval()
            out = model(torch.randn(5, 2), torch.randn(5, 3))
        assert out.dtype == dtype

    def test_buffers_are_not_pinned_to_float32(self) -> None:
        """`lower`/`upper` buffers follow the default dtype, not float32."""
        with _default_dtype(torch.float64):
            model = CoordinationMLP(
                dim_zeta=1,
                dim_conditions=0,
                dim_output=2,
                output_bounds=[(-1.0, 1.0)] * 2,
                hidden=4,
                n_layers=1,
            )
            state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        assert [
            k for k, v in state.items()
            if v.is_floating_point() and v.dtype != torch.float64
        ] == []


def _eq_constraint_fn(y_target: torch.Tensor):
    """`y - y_target == 0` dim-wise, plus one slack inequality."""
    dim = int(y_target.shape[-1])

    def fn(y, conditions):  # noqa: ARG001, conditions unused
        c_eq = y - y_target
        clist = [
            SimpleNamespace(value=c_eq[..., k], type="eq", name=f"eq_{k}")
            for k in range(dim)
        ]
        clist.append(
            SimpleNamespace(value=y.sum(dim=-1) - 10.0, type="ineq", name="ineq_0")
        )
        obj = c_eq.pow(2).sum(dim=-1)
        return obj, clist

    return fn


def _bad_dtypes(obj, dtype, path="") -> list[str]:
    """Paths of floating tensors inside a nested dict/list that are not `dtype`."""
    out: list[str] = []
    if torch.is_tensor(obj):
        if obj.is_floating_point() and obj.dtype != dtype:
            out.append(f"{path}:{obj.dtype}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            out += _bad_dtypes(v, dtype, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _bad_dtypes(v, dtype, f"{path}[{i}]")
    return out


class TestProjectorDtype:
    @pytest.mark.parametrize("dtype", _DTYPES)
    @pytest.mark.parametrize("method", ["eigh", "lm_k"])
    def test_step_and_project_stay_in_default_dtype(self, dtype, method) -> None:
        with _default_dtype(dtype):
            dim = 3
            y_target = torch.zeros(dim)
            fn = _eq_constraint_fn(y_target)
            proj = Projector(
                n_constraints=dim + 1,
                constraint_types=["eq"] * dim + ["ineq"],
                method=method,
                box_lower=torch.full((dim,), -5.0),
                box_upper=torch.full((dim,), 5.0),
            )

            def values_fn(y, conditions):
                return torch.stack([c.value for c in fn(y, conditions)[1]], dim=-1)

            y = torch.randn(4, dim, requires_grad=True)
            c_pre = values_fn(y, None)
            y_tilde, info = proj.step(y, c_pre, values_fn, conditions=None)
            assert y_tilde.dtype == dtype
            assert _bad_dtypes(info, dtype, "step_info") == []

            y2, pinfo = proj.project(
                torch.randn(4, dim), fn, conditions=None, max_iters=3
            )
            assert y2.dtype == dtype
            assert _bad_dtypes(pinfo, dtype, "project_info") == []

    @pytest.mark.parametrize("dtype", _DTYPES)
    def test_compute_residuals_follows_input_dtype(self, dtype) -> None:
        """The mixed eq/ineq residual mask follows the input dtype."""
        with _default_dtype(dtype):
            proj = Projector(
                n_constraints=2, constraint_types=["eq", "ineq"], method="lm_k"
            )
            c = torch.tensor([[-0.5, 0.25], [0.75, -1.0]], dtype=dtype)
            active = proj._build_active_mask(c)
            res = proj._compute_residuals(c, active)
        assert res.dtype == dtype
        torch.testing.assert_close(
            res, torch.tensor([[0.5, 0.25], [0.75, 0.0]], dtype=dtype)
        )
