"""E4: Chip layout macro placement benchmark.

Minimize HPWL + lambda_area * bbox_area subject to non-overlap constraints.
Based on FloorSet (Intel Labs, ICCAD 2024). Unconditional. A deterministic
random instance is generated on construction (seeded).
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import Tensor

from geometry.measure.overlap import aabb_overlap_volume
from pal.benchmarks.base import BenchmarkSpec, Query
from pal.constraints import Constraint

W_MIN = 0.5
LSE_TEMP = 10.0


def _make_spec(n_blocks: int, canvas_size: float) -> BenchmarkSpec:
    dim = 3 * n_blocks
    pos_lo = [0.0] * (2 * n_blocks)
    pos_hi = [canvas_size] * (2 * n_blocks)
    w_lo = [-5.0] * n_blocks
    w_hi = [5.0] * n_blocks
    lo = torch.tensor(pos_lo + w_lo, dtype=torch.float32)
    hi = torch.tensor(pos_hi + w_hi, dtype=torch.float32)
    return BenchmarkSpec(
        id="e4/chip_layout",
        family="e4",
        variant="chip_layout",
        dim=dim,
        n_eq=0,
        n_ineq=1,
        constraint_names=["overlap"],
        constraint_types=["ineq"],
        output_bounds=(lo, hi),
        condition_dim=0,
        zeta_dim=8,
        tolerance=1e-3,
        tau=1e-3,
        cost="cheap",
        recommended_device="cpu",
        precision="fp32",
        train_batch_size=256,
        n_eval_default=64,
        notes=f"FloorSet macro placement; {n_blocks} blocks, HPWL+bbox objective, aggregated AABB overlap",
    )


class E4ChipLayout:
    """pal Benchmark wrapper for chip-layout macro placement."""

    def __init__(
        self,
        n_blocks: int = 30,
        lambda_area: float = 0.01,
        canvas_size: float = 200.0,
        seed: int = 42,
    ) -> None:
        self.lambda_area = lambda_area
        self.canvas_size = canvas_size
        self.n_blocks = n_blocks
        self.spec = _make_spec(n_blocks, canvas_size)

        self._generate_instance(seed)

    def _generate_instance(self, seed: int) -> None:
        """Generate random block areas + connectivity + boundary pins."""
        rng = torch.Generator().manual_seed(seed)
        N = self.n_blocks

        log_areas = math.log(20) + (math.log(400) - math.log(20)) * torch.rand(
            N, generator=rng
        )
        self.area_target = log_areas.exp()

        n_edges = max(1, N)
        src = torch.randint(0, N, (n_edges,), generator=rng)
        dst = torch.randint(0, N, (n_edges,), generator=rng)
        valid = src != dst
        src, dst = src[valid], dst[valid]
        weights = 0.5 + 1.5 * torch.rand(src.shape[0], generator=rng)
        self.b2b_i = src
        self.b2b_j = dst
        self.b2b_w = weights

        n_pins = max(5, N // 3)
        pins_pos = torch.zeros(n_pins, 2)
        for p in range(n_pins):
            side = torch.randint(0, 4, (1,), generator=rng).item()
            t = self.canvas_size * torch.rand(1, generator=rng).item()
            if side == 0:
                pins_pos[p] = torch.tensor([t, 0.0])
            elif side == 1:
                pins_pos[p] = torch.tensor([self.canvas_size, t])
            elif side == 2:
                pins_pos[p] = torch.tensor([t, self.canvas_size])
            else:
                pins_pos[p] = torch.tensor([0.0, t])
        self.pins_pos = pins_pos

        pin_idx = torch.arange(n_pins)
        blk_idx = torch.randint(0, N, (n_pins,), generator=rng)
        p2b_w = 0.5 + torch.rand(n_pins, generator=rng)
        self.p2b_pin = pin_idx
        self.p2b_blk = blk_idx
        self.p2b_w = p2b_w

        self.pair_a, self.pair_b = torch.triu_indices(N, N, offset=1)

    def _to(self, device_str: str) -> None:
        if self.area_target.device.type != device_str:
            for attr in [
                "area_target", "b2b_i", "b2b_j", "b2b_w",
                "pins_pos", "p2b_pin", "p2b_blk", "p2b_w",
                "pair_a", "pair_b",
            ]:
                setattr(self, attr, getattr(self, attr).to(device_str))

    def forward(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> tuple[Tensor, list[Constraint]]:
        B = x.shape[0]
        N = self.n_blocks
        device = x.device
        self._to(device.type if hasattr(device, "type") else str(device))

        positions = x[:, :2 * N]
        widths_raw = x[:, 2 * N:]

        cx = positions[:, 0::2]  # [B, N]
        cy = positions[:, 1::2]  # [B, N]

        w = F.softplus(widths_raw, beta=1.0) + W_MIN
        h = self.area_target.unsqueeze(0) / w

        # HPWL
        O_hpwl = torch.zeros(B, device=device)
        if self.b2b_i.numel() > 0:
            dx = (cx[:, self.b2b_i] - cx[:, self.b2b_j]).abs()
            dy = (cy[:, self.b2b_i] - cy[:, self.b2b_j]).abs()
            O_hpwl = (self.b2b_w.unsqueeze(0) * (dx + dy)).sum(1)
        if self.p2b_pin.numel() > 0:
            pin_x = self.pins_pos[self.p2b_pin, 0]
            pin_y = self.pins_pos[self.p2b_pin, 1]
            dx_p = (cx[:, self.p2b_blk] - pin_x.unsqueeze(0)).abs()
            dy_p = (cy[:, self.p2b_blk] - pin_y.unsqueeze(0)).abs()
            O_hpwl = O_hpwl + (self.p2b_w.unsqueeze(0) * (dx_p + dy_p)).sum(1)

        # Soft bbox area (LSE-smoothed min/max)
        left = cx - w / 2
        right = cx + w / 2
        bottom = cy - h / 2
        top = cy + h / 2
        t = LSE_TEMP
        x_max = (1 / t) * torch.logsumexp(t * right, dim=1)
        x_min = -(1 / t) * torch.logsumexp(-t * left, dim=1)
        y_max = (1 / t) * torch.logsumexp(t * top, dim=1)
        y_min = -(1 / t) * torch.logsumexp(-t * bottom, dim=1)
        O_bbox = (x_max - x_min) * (y_max - y_min)
        obj = O_hpwl + self.lambda_area * O_bbox

        # Pair-wise AABB overlap (aggregated)
        sizes_3d = torch.stack([w, h, torch.full_like(w, 0.1)], dim=-1)
        centers_3d = torch.stack([cx, cy, torch.zeros_like(cx)], dim=-1)
        sa = sizes_3d[:, self.pair_a].reshape(-1, 3)
        ca = centers_3d[:, self.pair_a].reshape(-1, 3)
        sb = sizes_3d[:, self.pair_b].reshape(-1, 3)
        cb = centers_3d[:, self.pair_b].reshape(-1, 3)
        overlap_vol = aabb_overlap_volume(sa, ca, sb, cb)
        P = self.pair_a.shape[0]
        C_overlap = overlap_vol.reshape(B, P).sum(dim=1)  # [B]

        cons = [
            Constraint(
                value=C_overlap, type="ineq",
                tol=torch.zeros(B, device=device),
                margin=torch.full((B,), 1e-4, device=device),
                name="overlap",
            ),
        ]
        return obj, cons

    def objective(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        return self.forward(x, conditions)[0]

    def constraints(self, x: Tensor, conditions: Tensor | None = None) -> Tensor:
        _, cons = self.forward(x, conditions)
        return cons[0].value.unsqueeze(-1)

    def constraint_list(
        self, x: Tensor, conditions: Tensor | None = None
    ) -> list[Constraint]:
        return self.forward(x, conditions)[1]

    def sample_queries(
        self, n: int, split: Literal["train", "eval"], seed: int,
    ) -> Query:
        g = torch.Generator("cpu").manual_seed(int(seed))
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def eval_queries(self, seed: int, n: int | None = None) -> Query:
        n = int(self.spec.n_eval_default if n is None else n)
        g = torch.Generator("cpu").manual_seed(int(seed) * 1000 + 7919)
        zeta = torch.randn(n, self.spec.zeta_dim, generator=g)
        conditions = torch.empty(n, 0)
        return Query(zeta=zeta, conditions=conditions)

    def check_env(self) -> None:
        """Tiny forward pass proves geometry overlap + tensor wiring are fine."""
        lo, hi = self.spec.output_bounds
        x = 0.5 * (lo + hi).unsqueeze(0)
        obj, _ = self.forward(x)
        if not torch.isfinite(obj).all():
            raise RuntimeError("e4 check_env: non-finite objective from midpoint")

    def visualize_train(self, x: Tensor, conditions: Tensor | None = None):
        """Matplotlib diagnostic: placement + overlap flags + objective summary."""
        from pal.benchmarks.engineering.e4_chip_layout.viz import render_diagnostic_2d

        x_row = x[0] if x.ndim == 2 else x
        return render_diagnostic_2d(self, x_row.detach().cpu())

    def visualize_final(self, x: Tensor, conditions: Tensor | None = None):
        """PyVista 3D render. Falls back to matplotlib if no GL."""
        from pal.benchmarks.engineering.e4_chip_layout.viz import render_hero_3d

        x_row = x[0] if x.ndim == 2 else x
        return {"chip_3d": render_hero_3d(self, x_row.detach().cpu())}
