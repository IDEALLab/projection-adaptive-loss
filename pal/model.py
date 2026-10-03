"""Coordination MLP model."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor


def box_squash(raw: Tensor, lower: Tensor, upper: Tensor) -> Tensor:
    """Squash raw pre-activations into the design box ``[lower, upper]``.

    Shared with baselines that reuse the same tanh output head.
    """
    return lower + (upper - lower) * (torch.tanh(raw) + 1.0) / 2.0


class CoordinationMLP(nn.Module):
    """Problem-agnostic coordination MLP with bounded outputs.

    Args:
        dim_zeta: Dimension of latent noise input.
        dim_conditions: Dimension of condition input (0 for unconditional).
        dim_output: Number of output dimensions.
        output_bounds: (lo, hi) per output dimension.
        condition_bounds: (lo, hi) per condition dimension for normalization.
        hidden: Hidden layer width.
        n_layers: Number of hidden layers.
        skip_layer: If set, add skip connection from layer N to layer 2N.
        output_init_std: Stdev for the output-layer weight init. `None`
            (default) uses `sqrt(4/hidden)`; a tiny value puts every output
            at its box midpoint at step 1.
    """

    def __init__(
        self,
        dim_zeta: int,
        dim_conditions: int,
        dim_output: int,
        output_bounds: list[tuple[float, float]],
        condition_bounds: list[tuple[float, float]] | None = None,
        hidden: int = 256,
        n_layers: int = 4,
        skip_layer: int | None = None,
        output_init_std: float | None = None,
    ):
        super().__init__()
        assert len(output_bounds) == dim_output
        if condition_bounds is not None:
            assert len(condition_bounds) == dim_conditions

        self.dim_zeta = dim_zeta
        self.dim_conditions = dim_conditions
        self.dim_output = dim_output
        self.hidden = hidden
        self.n_layers = n_layers
        self.skip_layer = skip_layer

        lower = torch.tensor(
            [lo for lo, hi in output_bounds], dtype=torch.get_default_dtype()
        )
        upper = torch.tensor(
            [hi for lo, hi in output_bounds], dtype=torch.get_default_dtype()
        )
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)

        if condition_bounds is not None:
            cond_lo = torch.tensor(
                [lo for lo, hi in condition_bounds], dtype=torch.get_default_dtype()
            )
            cond_hi = torch.tensor(
                [hi for lo, hi in condition_bounds], dtype=torch.get_default_dtype()
            )
            self.register_buffer("cond_lo", cond_lo)
            self.register_buffer("cond_hi", cond_hi)
        else:
            self.cond_lo = None
            self.cond_hi = None

        dim_in = dim_zeta + dim_conditions
        layers = []
        for i in range(n_layers):
            in_dim = dim_in if i == 0 else hidden
            if skip_layer is not None and i == min(2 * skip_layer, n_layers):
                in_dim = hidden + hidden
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.ReLU())
        self.layers = nn.ModuleList(
            [layers[j] for j in range(0, len(layers), 2)]
        )
        self.activations = nn.ModuleList(
            [layers[j] for j in range(1, len(layers), 2)]
        )

        self.output_layer = nn.Linear(hidden, dim_output)
        std = (
            output_init_std if output_init_std is not None
            else math.sqrt(4.0 / hidden)
        )
        nn.init.normal_(self.output_layer.weight, std=std)
        nn.init.zeros_(self.output_layer.bias)

    def _normalize_conditions(self, conditions: Tensor) -> Tensor:
        """Normalize conditions to [-1, 1] using registered bounds."""
        if self.cond_lo is None:
            return conditions
        mid = (self.cond_lo + self.cond_hi) / 2.0
        half_range = (self.cond_hi - self.cond_lo) / 2.0
        half_range = half_range.clamp(min=1e-8)
        return (conditions - mid) / half_range

    def forward(
        self, zeta: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        """Forward pass.

        Args:
            zeta: Latent noise [B, dim_zeta].
            conditions: Optional conditions [B, dim_conditions].

        Returns:
            Bounded output [B, dim_output].
        """
        if conditions is not None and self.dim_conditions > 0:
            conditions = self._normalize_conditions(conditions)
            x = torch.cat([zeta, conditions], dim=-1)
        else:
            x = zeta

        skip_target = (
            min(2 * self.skip_layer, self.n_layers)
            if self.skip_layer is not None
            else None
        )
        saved = None
        for i, (linear, act) in enumerate(zip(self.layers, self.activations, strict=False)):
            if self.skip_layer is not None and i == self.skip_layer:
                saved = x
            if skip_target is not None and i == skip_target and saved is not None:
                x = torch.cat([x, saved], dim=-1)
            x = act(linear(x))

        raw = self.output_layer(x)

        scaled = box_squash(raw, self.lower, self.upper)
        return scaled


class DC3MLP(nn.Module):
    """Paper-faithful DC3 backbone (Donti et al. ICLR 2021).

    ``[Linear -> BatchNorm1d -> ReLU -> Dropout(p)] x n_layers -> Linear`` with
    Kaiming-normal init and a sigmoid bound-interp head,
    ``out = lo + (hi - lo) * sigmoid(raw)``.
    """

    def __init__(
        self,
        dim_zeta: int,
        dim_conditions: int,
        dim_output: int,
        output_bounds: list[tuple[float, float]],
        condition_bounds: list[tuple[float, float]] | None = None,
        hidden: int = 200,
        n_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert len(output_bounds) == dim_output
        if condition_bounds is not None:
            assert len(condition_bounds) == dim_conditions

        self.dim_zeta = dim_zeta
        self.dim_conditions = dim_conditions
        self.dim_output = dim_output
        self.hidden = hidden
        self.n_layers = n_layers

        lower = torch.tensor(
            [lo for lo, hi in output_bounds], dtype=torch.get_default_dtype()
        )
        upper = torch.tensor(
            [hi for lo, hi in output_bounds], dtype=torch.get_default_dtype()
        )
        self.register_buffer("lower", lower)
        self.register_buffer("upper", upper)

        if condition_bounds is not None:
            cond_lo = torch.tensor(
                [lo for lo, hi in condition_bounds], dtype=torch.get_default_dtype()
            )
            cond_hi = torch.tensor(
                [hi for lo, hi in condition_bounds], dtype=torch.get_default_dtype()
            )
            self.register_buffer("cond_lo", cond_lo)
            self.register_buffer("cond_hi", cond_hi)
        else:
            self.cond_lo = None
            self.cond_hi = None

        dim_in = dim_zeta + dim_conditions
        blocks = []
        in_dim = dim_in
        for _ in range(n_layers):
            lin = nn.Linear(in_dim, hidden)
            nn.init.kaiming_normal_(lin.weight)
            nn.init.zeros_(lin.bias)
            blocks += [
                lin,
                nn.BatchNorm1d(hidden),
                nn.ReLU(),
                nn.Dropout(p=dropout),
            ]
            in_dim = hidden
        self.hidden_stack = nn.Sequential(*blocks)

        self.output_layer = nn.Linear(hidden, dim_output)
        nn.init.kaiming_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def _normalize_conditions(self, conditions: Tensor) -> Tensor:
        if self.cond_lo is None:
            return conditions
        mid = (self.cond_lo + self.cond_hi) / 2.0
        half_range = (self.cond_hi - self.cond_lo) / 2.0
        half_range = half_range.clamp(min=1e-8)
        return (conditions - mid) / half_range

    def forward(
        self, zeta: Tensor, conditions: Tensor | None = None
    ) -> Tensor:
        if conditions is not None and self.dim_conditions > 0:
            conditions = self._normalize_conditions(conditions)
            x = torch.cat([zeta, conditions], dim=-1)
        else:
            x = zeta
        raw = self.output_layer(self.hidden_stack(x))
        return self.lower + (self.upper - self.lower) * torch.sigmoid(raw)
