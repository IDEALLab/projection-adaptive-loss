"""Urban wind benchmark core: box-city geometry and WinDiNet surrogate."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from inverse.surrogate import load_ltx_surrogate
from parallel_surrogate import ParallelSurrogate, detect_cuda_devices
from windinet.checkpoints import ensure_checkpoint

from geometry import CADProgram

_N_BUILDINGS_ENV = int(os.environ.get("E2_N_BUILDINGS", "10"))
CITY_YAML_PATH = Path(__file__).with_name(f"city_{_N_BUILDINGS_ENV}.yaml")


def inverse_softplus(x: torch.Tensor) -> torch.Tensor:
    """Approximate inverse of softplus on positive inputs."""
    x = x.clamp_min(1e-6)
    return x + torch.log(-torch.expm1(-x))


@dataclass
class PhysicalBoxParams:
    """Decoded box parameters with batch dimension [B, N]."""

    cx: torch.Tensor
    cy: torch.Tensor
    w: torch.Tensor
    d: torch.Tensor
    h: torch.Tensor

    @property
    def cz(self) -> torch.Tensor:
        return 0.5 * self.h

    @property
    def volume(self) -> torch.Tensor:
        return self.w * self.d * self.h


@dataclass
class BenchmarkResult:
    """Full benchmark outputs for one forward evaluation."""

    objective: torch.Tensor
    volume: torch.Tensor
    total_volume: torch.Tensor
    constraints: dict[str, torch.Tensor]
    decoded: PhysicalBoxParams
    building_mask: torch.Tensor
    u_pred: torch.Tensor
    v_pred: torch.Tensor
    tail_mean_u: torch.Tensor
    tail_mean_v: torch.Tensor
    tail_mean_speed: torch.Tensor
    hard_zone_labels: torch.Tensor
    hard_danger_fraction: torch.Tensor
    hard_danger_pixels: torch.Tensor
    hard_fluid_pixels: torch.Tensor
    coverage_ratio: torch.Tensor
    program: Any

    def constraint_tensor(self) -> torch.Tensor:
        """Return [B, 5] constraint vector in a stable order."""
        names = ("site", "clearance", "height", "danger", "coverage")
        return torch.stack([self.constraints[name] for name in names], dim=1)


class UrbanWindBenchmark:
    """N-building WinDiNet benchmark with 5N trainable box variables (N=10 default)."""

    def __init__(
        self,
        yaml_path: str | Path = CITY_YAML_PATH,
        *,
        device: str = "cpu",
        surrogate: Any | None = None,
        domain_lo: float = 175.0,
        domain_hi: float = 925.0,
        height_cap: float = 250.0,
        danger_threshold: float = 15.0,
        tail_frames: int = 10,
        raster_resolution: tuple[int, int] = (256, 256),
        raster_extents: tuple[tuple[float, float], tuple[float, float]] = (
            (0.0, 1100.0),
            (0.0, 1100.0),
        ),
        raster_epsilon: float = 2.0,
        inlet_u: float = 10.0,
        inlet_v: float = 0.0,
        min_width: float = 5.0,
        min_depth: float = 5.0,
        min_height: float = 1.0,
        d_min: float = 10.0,
        clearance_softplus_beta: float = 0.5,
        site_softplus_beta: float = 0.5,
        max_coverage: float = 0.5,
        volume_ref: float = 1.0e6,
        clearance_normalise_by_site: bool = True,
        danger_weight: float = 100.0,
    ) -> None:
        self.yaml_path = Path(yaml_path)
        self.device = device
        self.surrogate = surrogate
        self.domain_lo = float(domain_lo)
        self.domain_hi = float(domain_hi)
        self.height_cap = float(height_cap)
        self.danger_threshold = float(danger_threshold)
        self.tail_frames = int(tail_frames)
        self.raster_resolution = raster_resolution
        self.raster_extents = raster_extents
        self.raster_epsilon = float(raster_epsilon)
        self.inlet_u = float(inlet_u)
        self.inlet_v = float(inlet_v)
        self.min_width = float(min_width)
        self.min_depth = float(min_depth)
        self.min_height = float(min_height)
        self.d_min = float(d_min)
        self.clearance_softplus_beta = float(clearance_softplus_beta)
        self.site_softplus_beta = float(site_softplus_beta)
        self.max_coverage = float(max_coverage)
        self.site_area = (self.domain_hi - self.domain_lo) ** 2
        self.volume_ref = float(volume_ref)
        self.clearance_normalise_by_site = bool(clearance_normalise_by_site)
        self.danger_weight = float(danger_weight)

        self.program = CADProgram.load_from_yaml(str(self.yaml_path), device=device)
        self._init_params = self._load_initial_params()
        self.n_buildings = self._init_params["cx"].numel()
        self._pair_i, self._pair_j = torch.triu_indices(
            self.n_buildings, self.n_buildings, offset=1
        )
        self._site_mask = self._build_site_mask()  # [H, W] float, 1 inside inner domain

    def _build_site_mask(self) -> torch.Tensor:
        """Float mask [H, W] = 1.0 inside the 750x750 inner domain, 0 outside."""
        H, W = self.raster_resolution
        (x_lo, x_hi), (y_lo, y_hi) = self.raster_extents
        dx = (x_hi - x_lo) / W
        dy = (y_hi - y_lo) / H
        xs = torch.arange(W, dtype=torch.float32) * dx + x_lo + 0.5 * dx
        ys = torch.arange(H, dtype=torch.float32) * dy + y_lo + 0.5 * dy
        x_in = (xs >= self.domain_lo) & (xs <= self.domain_hi)  # [W]
        y_in = (ys >= self.domain_lo) & (ys <= self.domain_hi)  # [H]
        return (y_in.unsqueeze(1) & x_in.unsqueeze(0)).float()  # [H, W]

    def _load_initial_params(self) -> dict[str, torch.Tensor]:
        with open(self.yaml_path) as f:
            cfg = yaml.safe_load(f)
        params = cfg["params"]
        n = sum(1 for k in params if k.startswith("cx") and k[2:].isdigit())
        data = {}
        for key in ("cx", "cy", "w", "d", "h"):
            vals = [float(params[f"{key}{i+1}"]) for i in range(n)]
            data[key] = torch.tensor(vals, dtype=torch.float32)
        return data

    def make_initial_raw_params(
        self,
        *,
        batch_size: int = 1,
        device: str | None = None,
        jitter_xy: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Create initial raw parameters with shape [B, N]."""
        dev = device or self.device
        out: dict[str, torch.Tensor] = {}
        for key in ("cx", "cy"):
            x = self._init_params[key].to(dev).unsqueeze(0).repeat(batch_size, 1)
            if jitter_xy > 0:
                x = x + jitter_xy * torch.randn_like(x)
            out[key] = x

        out["w"] = inverse_softplus(
            (self._init_params["w"].to(dev) - self.min_width).clamp_min(1e-4)
        ).unsqueeze(0).repeat(batch_size, 1)
        out["d"] = inverse_softplus(
            (self._init_params["d"].to(dev) - self.min_depth).clamp_min(1e-4)
        ).unsqueeze(0).repeat(batch_size, 1)
        out["h"] = inverse_softplus(
            (self._init_params["h"].to(dev) - self.min_height).clamp_min(1e-4)
        ).unsqueeze(0).repeat(batch_size, 1)
        return out

    def decode_raw_params(self, raw_params: dict[str, torch.Tensor]) -> PhysicalBoxParams:
        """Decode unconstrained raw params into physical box dimensions."""
        return PhysicalBoxParams(
            cx=raw_params["cx"],
            cy=raw_params["cy"],
            w=F.softplus(raw_params["w"]) + self.min_width,
            d=F.softplus(raw_params["d"]) + self.min_depth,
            h=F.softplus(raw_params["h"]) + self.min_height,
        )

    def to_program_params(self, decoded: PhysicalBoxParams) -> dict[str, torch.Tensor]:
        """Map decoded [B, N] tensors to geometry param kwargs."""
        kwargs: dict[str, torch.Tensor] = {}
        for i in range(self.n_buildings):
            j = i + 1
            kwargs[f"cx{j}"] = decoded.cx[:, i : i + 1]
            kwargs[f"cy{j}"] = decoded.cy[:, i : i + 1]
            kwargs[f"w{j}"] = decoded.w[:, i : i + 1]
            kwargs[f"d{j}"] = decoded.d[:, i : i + 1]
            kwargs[f"h{j}"] = decoded.h[:, i : i + 1]
        return kwargs

    def rasterize(self, decoded: PhysicalBoxParams) -> tuple[Any, torch.Tensor]:
        """Return (program_variant, building_mask [B, H, W])."""
        prog = self.program.with_params(**self.to_program_params(decoded))
        result = prog.rasterize_2d(
            name="city",
            plane="xy",
            offset=0.0,
            resolution=self.raster_resolution,
            extents=self.raster_extents,
            epsilon=self.raster_epsilon,
        )
        return prog, result.occupancy.to(self.device)

    def site_constraint(self, decoded: PhysicalBoxParams) -> tuple[torch.Tensor, torch.Tensor]:
        # Squared mean of softplus(beta) boundary excesses (smooth, nonzero gradient near the edge).
        left = decoded.cx - 0.5 * decoded.w
        right = decoded.cx + 0.5 * decoded.w
        bottom = decoded.cy - 0.5 * decoded.d
        top = decoded.cy + 0.5 * decoded.d
        primitives = torch.stack(
            [
                self.domain_lo - left,
                right - self.domain_hi,
                self.domain_lo - bottom,
                top - self.domain_hi,
            ],
            dim=-1,
        )  # [B, N, 4]
        violations = F.softplus(primitives, beta=self.site_softplus_beta).reshape(
            primitives.shape[0], -1
        )
        return violations.square().mean(dim=1), violations

    def clearance_constraint(self, decoded: PhysicalBoxParams) -> tuple[torch.Tensor, torch.Tensor]:
        """Pairwise clearance with d_min buffer (smooth softplus formulation).

        For each unordered pair (i, j) the per-axis "overlap of inflated AABBs"
        is `softplus(half_w_sum_ij + d_min/2 - |Delta cx_ij|, beta)` (and likewise for
        y). The pairwise violation is the product of the two axes, units of
        m^2. With ``clearance_normalise_by_site=True`` (default) it is divided by
        ``self.site_area`` (dimensionless, same convention as ``coverage``).

        Aggregation: squared mean over the N(N-1)/2 pairs (``E2_CLEARANCE_AGG=lse``
        selects a log-sum-exp instead).
        """
        pair_i = self._pair_i.to(decoded.cx.device)
        pair_j = self._pair_j.to(decoded.cx.device)
        half_w_sum = 0.5 * (decoded.w[:, pair_i] + decoded.w[:, pair_j]) + 0.5 * self.d_min
        half_d_sum = 0.5 * (decoded.d[:, pair_i] + decoded.d[:, pair_j]) + 0.5 * self.d_min
        dx = half_w_sum - (decoded.cx[:, pair_i] - decoded.cx[:, pair_j]).abs()
        dy = half_d_sum - (decoded.cy[:, pair_i] - decoded.cy[:, pair_j]).abs()
        beta = self.clearance_softplus_beta
        violation = F.softplus(dx, beta=beta) * F.softplus(dy, beta=beta)
        if self.clearance_normalise_by_site:
            violation = violation / self.site_area
        if os.environ.get("E2_DEBUG_CLEARANCE", "0") == "1":
            self._dbg_clr_count = getattr(self, "_dbg_clr_count", 0) + 1
            every = int(os.environ.get("E2_DEBUG_CLEARANCE_EVERY", "10"))
            if self._dbg_clr_count % every == 0:
                v0 = violation[0].detach().float()
                topk = min(5, v0.numel())
                top_vals, top_idx = torch.topk(v0, topk)
                n_active = int((v0 > v0.max() * 1e-3).sum().item())
                print(
                    f"[clr#{self._dbg_clr_count}] max={v0.max().item():.4e} "
                    f"mean={v0.mean().item():.4e} "
                    f"std={v0.std().item():.4e} "
                    f"n_active(>0.1%max)={n_active}/{v0.numel()} "
                    f"top5_idx={top_idx.tolist()} "
                    f"top5_val={[f'{v:.3e}' for v in top_vals.tolist()]}",
                    flush=True,
                )
        agg = os.environ.get("E2_CLEARANCE_AGG", "sqmean").lower()
        if agg == "lse":
            beta_lse = float(os.environ.get("E2_CLEARANCE_LSE_BETA") or "1000")
            return torch.logsumexp(beta_lse * violation, dim=1) / beta_lse, violation
        return violation.square().mean(dim=1), violation

    def height_constraint(self, decoded: PhysicalBoxParams) -> tuple[torch.Tensor, torch.Tensor]:
        primitives = decoded.h - self.height_cap
        violations = F.relu(primitives)
        return violations.mean(dim=1), violations

    def coverage_constraint(
        self, decoded: PhysicalBoxParams
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Coverage equality: total site-clipped footprint / site area == max_coverage.

        Returns (violation [B], ratio [B]). Ratio is the fraction of the 750x750
        inner site occupied by building footprints (site-clipped, so boxes
        extending past the site edge contribute only the inside portion).
        Rectangle areas are summed without overlap correction.

        `violation` is the signed residual `ratio - max_coverage`, satisfied
        when |violation| <= tol.
        """
        left = torch.clamp(decoded.cx - 0.5 * decoded.w, self.domain_lo, self.domain_hi)
        right = torch.clamp(decoded.cx + 0.5 * decoded.w, self.domain_lo, self.domain_hi)
        bottom = torch.clamp(decoded.cy - 0.5 * decoded.d, self.domain_lo, self.domain_hi)
        top = torch.clamp(decoded.cy + 0.5 * decoded.d, self.domain_lo, self.domain_hi)
        area = F.relu(right - left) * F.relu(top - bottom)  # [B, N]
        ratio = area.sum(dim=1) / self.site_area  # [B]
        violation = ratio - self.max_coverage
        return violation, ratio

    def compute_tail_means(
        self, u_pred: torch.Tensor, v_pred: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tail = min(self.tail_frames, u_pred.shape[1])
        u_tail = u_pred[:, -tail:]
        v_tail = v_pred[:, -tail:]
        tail_mean_u = u_tail.mean(dim=1)
        tail_mean_v = v_tail.mean(dim=1)
        tail_mean_speed = torch.sqrt(tail_mean_u.square() + tail_mean_v.square() + 1e-8)
        return tail_mean_u, tail_mean_v, tail_mean_speed

    def danger_constraint(
        self,
        tail_mean_speed: torch.Tensor,
        building_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fluid-probability-weighted mean of speed above ``danger_threshold``.

        The aggregate is multiplied by ``self.danger_weight`` (default 100) so a
        typical excess (O(1e-4) m/s) lands above the tolerance. The per-pixel
        ``violations`` are returned in raw m/s.
        """
        site_mask = self._site_mask.to(building_mask.device)
        fluid_prob = (1.0 - building_mask).clamp(0.0, 1.0) * site_mask
        violations = F.relu(tail_mean_speed - self.danger_threshold)
        weighted_mean = (violations * fluid_prob).sum(dim=(-2, -1)) / fluid_prob.sum(
            dim=(-2, -1)
        ).clamp_min(1e-6)
        return self.danger_weight * weighted_mean, violations

    def hard_zone_labels(
        self, tail_mean_speed: torch.Tensor, building_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        site_mask = self._site_mask.to(building_mask.device).bool()
        building_hard = building_mask > 0.5
        zones = torch.zeros_like(tail_mean_speed, dtype=torch.long)
        fluid = ~building_hard & site_mask
        zones[fluid & (tail_mean_speed < 1.0)] = 1
        zones[fluid & (tail_mean_speed >= 1.0) & (tail_mean_speed < 5.0)] = 2
        zones[fluid & (tail_mean_speed >= 5.0) & (tail_mean_speed < 15.0)] = 3
        zones[fluid & (tail_mean_speed >= 15.0)] = 4

        danger_pixels = (zones == 4).sum(dim=(-2, -1))
        fluid_pixels = fluid.sum(dim=(-2, -1))
        danger_fraction = danger_pixels.float() / fluid_pixels.clamp_min(1)
        return zones, danger_pixels, fluid_pixels, danger_fraction

    def evaluate_raw_params(
        self,
        raw_params: dict[str, torch.Tensor],
        *,
        u_pred_override: torch.Tensor | None = None,
        v_pred_override: torch.Tensor | None = None,
    ) -> BenchmarkResult:
        """Evaluate objective + 5 aggregated constraints from raw params."""
        decoded = self.decode_raw_params(raw_params)
        prog, building_mask = self.rasterize(decoded)

        if u_pred_override is not None or v_pred_override is not None:
            if u_pred_override is None or v_pred_override is None:
                raise ValueError("u_pred_override and v_pred_override must be provided together")
            u_pred = u_pred_override
            v_pred = v_pred_override
        else:
            if self.surrogate is None:
                raise RuntimeError("Benchmark evaluate_raw_params() needs a surrogate or overrides")
            inlet_u = torch.full(
                (building_mask.shape[0],), self.inlet_u, device=self.device, dtype=torch.float32
            )
            inlet_v = torch.full(
                (building_mask.shape[0],), self.inlet_v, device=self.device, dtype=torch.float32
            )
            u_pred, v_pred = self.surrogate(building_mask, inlet_u, inlet_v)

        tail_mean_u, tail_mean_v, tail_mean_speed = self.compute_tail_means(u_pred, v_pred)
        c_site, _ = self.site_constraint(decoded)
        c_clearance, _ = self.clearance_constraint(decoded)
        c_height, _ = self.height_constraint(decoded)
        c_danger, _ = self.danger_constraint(tail_mean_speed, building_mask)
        c_coverage, coverage_ratio = self.coverage_constraint(decoded)

        zones, danger_pixels, fluid_pixels, danger_fraction = self.hard_zone_labels(
            tail_mean_speed, building_mask
        )
        volume = decoded.volume
        total_volume = volume.sum(dim=1)

        # Normalized objective: pure volume maximization in O(1) scale.
        objective = -(total_volume / self.volume_ref)

        return BenchmarkResult(
            objective=objective,
            volume=volume,
            total_volume=total_volume,
            constraints={
                "site": c_site,
                "clearance": c_clearance,
                "height": c_height,
                "danger": c_danger,
                "coverage": c_coverage,
            },
            decoded=decoded,
            building_mask=building_mask,
            u_pred=u_pred,
            v_pred=v_pred,
            tail_mean_u=tail_mean_u,
            tail_mean_v=tail_mean_v,
            tail_mean_speed=tail_mean_speed,
            hard_zone_labels=zones,
            hard_danger_fraction=danger_fraction,
            hard_danger_pixels=danger_pixels,
            hard_fluid_pixels=fluid_pixels,
            coverage_ratio=coverage_ratio,
            program=prog,
        )


def choose_device(requested: str | None = None) -> str:
    """Choose an available accelerator unless a specific device is requested."""
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def choose_surrogate_dtype(device: str) -> torch.dtype:
    """Use fp32 for gradient fidelity by default, bf16 if E2_DTYPE=bf16.

    fp32 needs a large-memory GPU (80+ GB) for the 3.17B-parameter surrogate;
    bf16 fits on 24 GB cards for smoke tests.
    """
    del device
    import os
    if os.environ.get("E2_DTYPE", "fp32").lower() == "bf16":
        return torch.bfloat16
    return torch.float32


def load_default_surrogate(
    *,
    device: str,
    dtype: torch.dtype | None = None,
    num_frames: int = 57,
    num_inference_steps: int = 2,
    field_size_m: float = 1100.0,
) -> tuple[Any, str]:
    """Load WinDiNet surrogate and return (surrogate, temp diffusion dir)."""
    use_dtype = dtype or choose_surrogate_dtype(device)
    dit_ckpt = ensure_checkpoint("dit")
    scalar_ckpt = ensure_checkpoint("scalar_embedding")
    vae_ckpt = ensure_checkpoint("vae_decoder")

    diffusion_dir = tempfile.mkdtemp(prefix="windinet_city_")
    ckpt_subdir = os.path.join(diffusion_dir, "checkpoints")
    os.makedirs(ckpt_subdir, exist_ok=True)
    os.symlink(dit_ckpt, os.path.join(ckpt_subdir, "model_weights_step_00000.safetensors"))
    os.symlink(
        scalar_ckpt,
        os.path.join(ckpt_subdir, "scalar_embedding_step_00000.safetensors"),
    )

    try:
        surrogate = load_ltx_surrogate(
            diffusion_dir=diffusion_dir,
            vae_adapter_ckpt=vae_ckpt,
            device=device,
            dtype=use_dtype,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            field_size_m=field_size_m,
        )
    except Exception as exc:
        cleanup_surrogate_cache(diffusion_dir)
        raise RuntimeError(
            "Failed to load the WinDiNet surrogate. The fine-tuned WindiNet checkpoints "
            "were found, but the base LTX-Video weights are not available locally and "
            "could not be fetched. Pre-cache the LTXV_2B_0.9.6_DEV weights or run this "
            "example in an environment with network access."
        ) from exc

    cuda_devices = detect_cuda_devices()
    print(f"[e2] CUDA devices visible: {len(cuda_devices)}")
    for i, d in enumerate(cuda_devices):
        name = torch.cuda.get_device_name(i)
        free_gb = torch.cuda.mem_get_info(i)[0] / 1024**3
        print(f"[e2]   {d}: {name} ({free_gb:.1f} GB free)")

    if len(cuda_devices) >= 2 and device.startswith("cuda"):
        print(f"[e2] wrapping surrogate in ParallelSurrogate across {len(cuda_devices)} GPUs")
        surrogate = ParallelSurrogate(surrogate, devices=cuda_devices)
    else:
        print(f"[e2] single-replica surrogate (device={device})")

    return surrogate, diffusion_dir


def cleanup_surrogate_cache(diffusion_dir: str | None) -> None:
    """Delete the temporary checkpoint staging directory."""
    if diffusion_dir:
        shutil.rmtree(diffusion_dir, ignore_errors=True)
