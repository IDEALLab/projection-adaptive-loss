"""
LTX diffusion model wrapped as a differentiable surrogate for inverse
optimization of urban building layouts.

The forward pass:
  1. Creates a conditioning frame from the soft building footprint + inlet velocity
  2. Encodes the conditioning via the adapted VAE
  3. Runs a differentiable denoising loop (with per-step gradient checkpointing)
  4. Decodes the denoised latents via the adapted VAE
  5. Returns u, v velocity fields in m/s

Uses scalar conditioning (inlet_speed_mps + field_size_m) via ScalarEmbedding.

Usage:
    surrogate = load_ltx_surrogate(
        diffusion_dir="/path/to/diffusion_checkpoint_dir",
        vae_adapter_ckpt="/path/to/vae_physics.pt",
        field_size_m=1100.0,
    )
    u_pred, v_pred = surrogate(building_mask, inlet_u, inlet_v)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from windinet.inference.model_loader import (
    load_transformer,
    load_vae,
)
from windinet.inference.pipeline import (
    LTXConditionPipeline,
    linear_quadratic_schedule,
)
from windinet.vae_adapter import load_adapted_vae

# ---------------------------------------------------------------------------
# Constants  (must match LTX training)
# ---------------------------------------------------------------------------
MAG_CAP_MPS = 30.0  # velocity cap used during training (wind_norm)
FRAME_RATE = 25      # default LTX frame rate for coordinate scaling

# VAE compression ratios (LTX-Video)
VAE_SPATIAL = 32
VAE_TEMPORAL = 8


# ===================================================================
# Surrogate
# ===================================================================

class LTXSurrogate(nn.Module):
    """Differentiable surrogate: (building_mask, inlet_u, inlet_v) -> (u, v).

    All heavy model weights are frozen; gradients flow through the computation
    graph so that upstream parameters (e.g. DifferentiableFootprint centres)
    receive useful gradients.
    """

    def __init__(
        self,
        transformer: nn.Module,
        adapted_vae: nn.Module,
        scalar_embedding: nn.Module,
        *,
        field_size_m: float = 1100.0,
        num_frames: int = 113,
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 10,
        image_cond_noise_scale: float = 0.0,
        use_checkpoint: bool = True,
        text_prompt_embeds: torch.Tensor | None = None,
        text_prompt_mask: torch.Tensor | None = None,
    ):
        super().__init__()
        self.transformer = transformer
        self.adapted_vae = adapted_vae
        self.scalar_embedding = scalar_embedding
        self.field_size_m = field_size_m

        self.num_frames = num_frames
        self.height = height
        self.width = width
        self.num_inference_steps = num_inference_steps
        self.image_cond_noise_scale = image_cond_noise_scale
        self.use_checkpoint = use_checkpoint

        # Derived latent dimensions
        self.lat_F = (num_frames - 1) // VAE_TEMPORAL + 1
        self.lat_H = height // VAE_SPATIAL
        self.lat_W = width // VAE_SPATIAL
        self.lat_C = transformer.config.in_channels  # 128

        # Pre-compute sigma schedule  (high->low noise)
        sigmas = linear_quadratic_schedule(num_inference_steps)
        self.register_buffer("_sigmas", sigmas)

        # Pre-computed empty text encoder outputs (for concat with scalar embeds)
        if text_prompt_embeds is not None:
            self.register_buffer("_text_prompt_embeds", text_prompt_embeds)
            self.register_buffer("_text_prompt_mask", text_prompt_mask)
        else:
            self._text_prompt_embeds = None
            self._text_prompt_mask = None

        # Freeze everything
        self.transformer.requires_grad_(False)
        self.adapted_vae.requires_grad_(False)
        self.scalar_embedding.requires_grad_(False)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _make_cond_frame(
        self,
        building_mask: torch.Tensor,
        inlet_u: torch.Tensor,
        inlet_v: torch.Tensor,
    ) -> torch.Tensor:
        """Create a conditioning tensor  [K, 3, 1, H, W] in [-1, 1].

        Channels: u, v, b.  Convention matches ``build_n_channel_video`` in
        ``load_adapted_vae.py``.
        """
        K, H, W = building_mask.shape
        fluid = 1.0 - building_mask  # 1 = fluid, 0 = building

        u = (inlet_u.view(K, 1, 1) * fluid / MAG_CAP_MPS).clamp(-1, 1)
        v = (inlet_v.view(K, 1, 1) * fluid / MAG_CAP_MPS).clamp(-1, 1)
        b = (2.0 * fluid - 1.0).clamp(-1, 1)

        return torch.stack([u, v, b], dim=1).unsqueeze(2)  # [K,3,1,H,W]

    def _normalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        vae = self.adapted_vae.vae
        sf = float(getattr(vae.config, "scaling_factor", 1.0))
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        return (z - mean) * sf / std

    def _denormalize_latents(self, z: torch.Tensor) -> torch.Tensor:
        vae = self.adapted_vae.vae
        sf = float(getattr(vae.config, "scaling_factor", 1.0))
        mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        std = vae.latents_std.view(1, -1, 1, 1, 1).to(z.device, z.dtype)
        return z * std / sf + mean

    def _encode_cond(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode conditioning frame -> normalised latents [K, C, 1, h, w]."""
        vae_dtype = next(self.adapted_vae.vae.parameters()).dtype
        enc_out = self.adapted_vae.encode(cond.to(vae_dtype))
        z = enc_out.latent_dist.mean  # deterministic
        return self._normalize_latents(z)

    def _compute_prompt_embeds(
        self,
        inlet_u: torch.Tensor,
        inlet_v: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute encoder_hidden_states from scalar + text embeddings.

        Matches ``build_scalar_embeds()`` from the paper pipeline:
        concatenates scalar embeddings (8 tokens) with pre-computed empty
        text encoder embeddings (256 tokens) -> 264 tokens total.
        """
        K = inlet_u.shape[0]
        device = inlet_u.device

        inlet_speed = torch.sqrt(inlet_u ** 2 + inlet_v ** 2)  # (K,)
        field_size = torch.full_like(inlet_speed, self.field_size_m)
        scalars = torch.stack([inlet_speed, field_size], dim=1)  # (K, 2)

        scalar_embeds = self.scalar_embedding(scalars)  # (K, 8, 4096)

        # Cast to transformer dtype (bf16) for compatibility
        t_dtype = next(self.transformer.parameters()).dtype
        scalar_embeds = scalar_embeds.to(t_dtype)

        if self._text_prompt_embeds is not None:
            # Concat scalar + text embeddings (paper config: 8 + 256 = 264 tokens)
            text_pe = self._text_prompt_embeds.to(t_dtype).expand(K, -1, -1)
            text_pa = self._text_prompt_mask.expand(K, -1)
            pe = torch.cat([scalar_embeds, text_pe], dim=1)
            scalar_mask = torch.ones(K, scalar_embeds.shape[1],
                                     dtype=text_pa.dtype, device=device)
            pa = torch.cat([scalar_mask, text_pa], dim=1)
        else:
            # Fallback: scalar-only (8 tokens)
            pe = scalar_embeds
            pa = torch.ones(K, pe.shape[1], device=device, dtype=torch.long)

        return pe, pa

    # --- packing / unpacking (delegate to pipeline statics) ---

    @staticmethod
    def _pack(x: torch.Tensor) -> torch.Tensor:
        return LTXConditionPipeline._pack_latents(x, patch_size=1, patch_size_t=1)

    @staticmethod
    def _unpack(x: torch.Tensor, F: int, H: int, W: int) -> torch.Tensor:
        return LTXConditionPipeline._unpack_latents(x, F, H, W, patch_size=1, patch_size_t=1)

    # --- video coordinates (RoPE positional IDs) ---

    def _video_coords(self, K: int, device: torch.device) -> torch.Tensor:
        """[K, 3, S] float positional IDs for the transformer."""
        t_ids = torch.arange(self.lat_F, device=device)
        h_ids = torch.arange(self.lat_H, device=device)
        w_ids = torch.arange(self.lat_W, device=device)
        grid = torch.meshgrid(t_ids, h_ids, w_ids, indexing="ij")
        coords = torch.stack(grid, dim=0).reshape(3, -1)  # [3, S]

        # scale to pixel space (matches pipeline convention)
        scale = torch.tensor(
            [VAE_TEMPORAL, VAE_SPATIAL, VAE_SPATIAL],
            device=device, dtype=torch.float32,
        ).unsqueeze(1)
        coords = coords.float() * scale
        coords[0] = (coords[0] + 1 - VAE_TEMPORAL).clamp(min=0)  # temporal shift
        coords[0] = coords[0] / FRAME_RATE  # convert to seconds

        return coords.unsqueeze(0).expand(K, -1, -1)  # [K, 3, S]

    # --- latent preparation ---

    def _prepare_latents(
        self,
        cond_latents: torch.Tensor,
        generator: torch.Generator | None = None,
    ):
        """Prepare initial noisy latents with conditioning blended at frame 0.

        Returns
        -------
        latents_packed : (K, S, D)
        cond_mask      : (K, S)   1.0 for conditioned tokens, 0.0 otherwise
        video_coords   : (K, 3, S)
        init_packed    : (K, S, D)  copy used for optional noise injection
        """
        K = cond_latents.shape[0]
        device = cond_latents.device
        dtype = cond_latents.dtype

        shape = (K, self.lat_C, self.lat_F, self.lat_H, self.lat_W)
        noise = torch.randn(shape, device=device, dtype=dtype, generator=generator)

        # blend conditioning into frame 0
        n_cond = cond_latents.shape[2]  # typically 1
        noise[:, :, :n_cond] = cond_latents

        # per-frame conditioning mask  -> per-token mask
        frame_mask = torch.zeros(K, self.lat_F, device=device)
        frame_mask[:, :n_cond] = 1.0

        video_coords = self._video_coords(K, device)

        # frame index for each token: video_coords[b, 0, :] before scaling
        # We need the raw frame index to gather the mask.
        raw_frame_ids = torch.arange(self.lat_F, device=device)
        raw_h = torch.arange(self.lat_H, device=device)
        raw_w = torch.arange(self.lat_W, device=device)
        grid_f = torch.meshgrid(raw_frame_ids, raw_h, raw_w, indexing="ij")[0]
        token_frame_idx = grid_f.reshape(-1).long()  # [S]
        cond_mask = frame_mask[:, token_frame_idx]  # [K, S]

        packed = self._pack(noise)
        init_packed = packed.clone()

        return packed, cond_mask, video_coords, init_packed

    # --- single denoising step (checkpoint-friendly) ---

    def _step(
        self,
        latents: torch.Tensor,
        t_cur: torch.Tensor,
        t_next: torch.Tensor,
        cond_mask: torch.Tensor,
        video_coords: torch.Tensor,
        prompt_embeds: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        K, S, _ = latents.shape
        device = latents.device
        t_val = t_cur.item()

        # per-token timestep: conditioning tokens see t = 0
        timestep = torch.full((K, S), t_val, device=device, dtype=torch.float32)
        timestep = torch.min(timestep, (1.0 - cond_mask) * 1000.0)

        noise_pred = self.transformer(
            hidden_states=latents.to(prompt_embeds.dtype),
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            encoder_attention_mask=prompt_attention_mask,
            video_coords=video_coords,
            return_dict=False,
        )[0]

        # Euler step -- matches scheduler per-token path:
        #   dt = sigma_cur - sigma_next  (positive, going from high to low noise)
        #   prev_sample = sample + dt * (-noise_pred)
        sigma_cur = t_val / 1000.0
        sigma_next = t_next.item() / 1000.0
        denoised = latents + (sigma_cur - sigma_next) * (-noise_pred.float())

        # keep conditioning tokens frozen
        should_denoise = (sigma_cur - 1e-6 < (1.0 - cond_mask)).unsqueeze(-1)
        return torch.where(should_denoise, denoised, latents)

    # --- VAE decoder gradient checkpointing ---

    def _enable_vae_decoder_checkpointing(self):
        """Wrap each resnet in the VAE decoder with gradient checkpointing.

        This reduces peak VRAM from ~26 GB to ~17.6 GB for 225-frame decodes
        at the cost of ~2x slower backward pass through the decoder.
        """
        decoder = self.adapted_vae.vae.decoder

        def _wrap_resnet(resnet):
            original_forward = resnet.forward

            def ckpt_forward(*args, **kwargs):
                return checkpoint(original_forward, *args, use_reentrant=False, **kwargs)

            resnet.forward = ckpt_forward

        # mid_block resnets
        if hasattr(decoder, "mid_block") and hasattr(decoder.mid_block, "resnets"):
            for r in decoder.mid_block.resnets:
                _wrap_resnet(r)

        # up_blocks resnets
        if hasattr(decoder, "up_blocks"):
            for block in decoder.up_blocks:
                if hasattr(block, "resnets"):
                    for r in block.resnets:
                        _wrap_resnet(r)

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(
        self,
        building_mask: torch.Tensor,
        inlet_u: torch.Tensor,
        inlet_v: torch.Tensor,
        seed: int = 42,
    ):
        """Run the surrogate.

        Parameters
        ----------
        building_mask : (K, 1, H, W) or (K, H, W)
            Soft building occupancy in [0, 1].
        inlet_u, inlet_v : (K,)
            Inlet velocities in m/s.
        seed : int
            Fixed seed for the initial noise.  Using the same seed across
            optimisation steps ensures the gradient signal is clean
            (only the conditioning changes, not the noise realisation).

        Returns
        -------
        u_pred, v_pred : (K, T, H, W) in m/s
            T = num_frames - 1 (conditioning frame excluded).
        """
        if building_mask.dim() == 4:
            building_mask = building_mask[:, 0]  # drop channel dim

        device = building_mask.device

        # 1  conditioning frame
        cond = self._make_cond_frame(building_mask, inlet_u, inlet_v)

        # 2  encode
        cond_lat = self._encode_cond(cond)

        # 3  scalar embeddings
        pe, pa = self._compute_prompt_embeds(inlet_u, inlet_v)

        # 4  prepare noisy latents (deterministic noise)
        gen = torch.Generator(device=device).manual_seed(seed)
        latents, cond_mask, coords, init_lat = self._prepare_latents(
            cond_lat, generator=gen,
        )

        # 5  denoising loop
        ts = self._sigmas.to(device) * 1000.0
        ts_ext = torch.cat([ts, torch.zeros(1, device=device)])

        for i in range(len(ts)):
            t_cur = ts_ext[i]
            t_next = ts_ext[i + 1]

            # optional noise injection on conditioning tokens
            if self.image_cond_noise_scale > 0:
                noise = torch.randn_like(latents)
                need = (cond_mask > 0.5).unsqueeze(-1)
                sig = t_cur / 1000.0
                noised = init_lat + self.image_cond_noise_scale * noise * (sig ** 2)
                latents = torch.where(need, noised, latents)

            if self.use_checkpoint:
                latents = checkpoint(
                    self._step, latents, t_cur, t_next, cond_mask, coords,
                    pe, pa,
                    use_reentrant=False,
                )
            else:
                latents = self._step(
                    latents, t_cur, t_next, cond_mask, coords, pe, pa,
                )

        # 6  unpack -> denormalise -> decode
        z = self._unpack(latents, self.lat_F, self.lat_H, self.lat_W)
        z = self._denormalize_latents(z)

        vae_dtype = next(self.adapted_vae.vae.parameters()).dtype
        temb = torch.zeros(z.shape[0], device=device, dtype=vae_dtype)
        decoded_rgb = self.adapted_vae.vae.decode(
            z.to(vae_dtype), temb, return_dict=False,
        )[0]
        decoded = self.adapted_vae.out_adapter(decoded_rgb.float())
        decoded = decoded.clamp(-1.0, 1.0)

        # 7  extract u, v (skip conditioning frame at t = 0)
        u_norm = decoded[:, 0, 1:]
        v_norm = decoded[:, 1, 1:]

        return u_norm * MAG_CAP_MPS, v_norm * MAG_CAP_MPS


# ===================================================================
# Factory
# ===================================================================

def load_ltx_surrogate(
    diffusion_dir: str | Path = "/path/to/diffusion_checkpoint_dir",
    vae_adapter_ckpt: str | Path = "/path/to/vae_physics.pt",
    *,
    model_source: str = "LTXV_2B_0.9.6_DEV",
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    num_inference_steps: int = 10,
    num_frames: int = 113,
    use_checkpoint: bool = True,
    field_size_m: float = 1100.0,
) -> LTXSurrogate:
    """Load all components and return a ready-to-use surrogate.

    Uses ScalarEmbedding (inlet_speed_mps + field_size_m) concatenated with
    empty text encoder embeddings for conditioning (264 tokens total, matching
    the paper inference pipeline).
    """

    diffusion_dir = Path(diffusion_dir)
    vae_adapter_ckpt = Path(vae_adapter_ckpt)

    print(f"[ltx-surr] model_source     : {model_source}")
    print(f"[ltx-surr] diffusion_dir    : {diffusion_dir}")
    print(f"[ltx-surr] vae_adapter_ckpt : {vae_adapter_ckpt}")

    # --- base VAE + adapter ---
    base_vae = load_vae(model_source, dtype=dtype)
    adapted_vae, _meta = load_adapted_vae(
        base_vae,
        ckpt_path=vae_adapter_ckpt,
        device=device,
        dtype=torch.float32,
        default_temb=0.0,
        verbose=True,
    )
    adapted_vae.eval()

    # --- transformer ---
    transformer = load_transformer(model_source, dtype=dtype)

    ckpt_dir = diffusion_dir / "checkpoints"
    safetensors = sorted(ckpt_dir.glob("model_weights_*.safetensors"))
    if not safetensors:
        raise FileNotFoundError(f"No model_weights_*.safetensors in {ckpt_dir}")
    ckpt_file = safetensors[-1]
    print(f"[ltx-surr] transformer ckpt : {ckpt_file}")

    from safetensors.torch import load_file

    sd = load_file(str(ckpt_file))
    if any(k.startswith("module.") for k in sd):
        sd = {k.replace("module.", "", 1): v for k, v in sd.items()}
    if any(k.startswith("transformer.") for k in sd):
        sd = {
            k.replace("transformer.", "", 1): v
            for k, v in sd.items()
            if k.startswith("transformer.")
        }
    transformer.load_state_dict(sd, strict=False)
    transformer.to(device).eval()

    # enable per-layer gradient checkpointing inside the transformer itself
    if hasattr(transformer, "gradient_checkpointing_enable"):
        transformer.gradient_checkpointing_enable()

    # --- scalar embedding ---
    from windinet.config import ScalarConditioningConfig
    from windinet.scalar_embeddings import ScalarEmbedding

    scalar_cfg = ScalarConditioningConfig(
        enabled=True,
        scalar_names=["inlet_speed_mps", "field_size_m"],
        scalar_ranges={
            "inlet_speed_mps": (0.1, 20.0),
            "field_size_m": (900.0, 1400.0),
        },
        embedding_dim=4096,
        num_tokens_per_scalar=4,
        fourier_features=64,
        mlp_hidden_dim=256,
        dropout=0.1,
    )
    scalar_embedding = ScalarEmbedding(scalar_cfg)

    # find matching scalar checkpoint
    stem = ckpt_file.stem  # e.g. "model_weights_step_10048"
    step_str = stem.split("_step_")[-1]
    scalar_ckpt = ckpt_dir / f"scalar_embedding_step_{step_str}.safetensors"
    if not scalar_ckpt.exists():
        # fallback: find latest scalar checkpoint
        scalar_ckpts = sorted(ckpt_dir.glob("scalar_embedding_step_*.safetensors"))
        if not scalar_ckpts:
            raise FileNotFoundError(f"No scalar_embedding_step_*.safetensors in {ckpt_dir}")
        scalar_ckpt = scalar_ckpts[-1]
    print(f"[ltx-surr] scalar emb ckpt  : {scalar_ckpt}")

    scalar_sd = load_file(str(scalar_ckpt))
    scalar_embedding.load_state_dict(scalar_sd)
    scalar_embedding.to(device).eval()
    print(f"[ltx-surr] field_size_m     : {field_size_m}")

    # --- assemble (scalar-only embeddings, no text encoder needed) ---
    surrogate = LTXSurrogate(
        transformer=transformer,
        adapted_vae=adapted_vae,
        scalar_embedding=scalar_embedding,
        field_size_m=field_size_m,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        use_checkpoint=use_checkpoint,
        text_prompt_embeds=None,
    ).to(device)

    # Enable per-resnet gradient checkpointing in VAE decoder (for 225+ frames)
    surrogate._enable_vae_decoder_checkpointing()

    n_params = sum(p.numel() for p in surrogate.parameters())
    print(f"[ltx-surr] ready  ({n_params/1e9:.2f}B params, all frozen, "
          f"VAE decoder checkpointing enabled)")
    return surrogate
