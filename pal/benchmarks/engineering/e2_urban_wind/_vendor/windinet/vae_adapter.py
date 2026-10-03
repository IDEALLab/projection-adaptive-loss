"""
VAE physics adapter for wind field encoding/decoding.

Wraps the LTX-Video VAE with lightweight adapters that transform between
n-channel wind fields (u, v, building mask) and 3-channel RGB space.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name in ("silu", "swish"):
        return nn.SiLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    raise ValueError(f"unknown activation: {name}")


def _strict_load(module: nn.Module, sd: dict[str, torch.Tensor], what: str) -> None:
    try:
        module.load_state_dict(sd, strict=True)
    except Exception as e:
        raise RuntimeError(f"strict load failed for {what}: {e}") from e


class InAdapter(nn.Module):
    """Maps n input channels to 3 RGB channels for the VAE encoder."""

    def __init__(self, n: int, k: int, activation: str):
        super().__init__()
        self.n = n
        self.k = k
        if k > 0:
            self.proj1 = nn.Conv3d(n, k, kernel_size=1)
            self.act = _get_activation(activation)
            self.proj2 = nn.Conv3d(k, 3, kernel_size=1)

    def forward(self, x_n: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x_n
        y = self.proj2(self.act(self.proj1(x_n)))
        return torch.tanh(y * torch.pi)


class OutAdapter(nn.Module):
    """Maps 3 RGB channels from VAE decoder to n output channels."""

    def __init__(self, n: int, k: int, activation: str):
        super().__init__()
        self.n = n
        self.k = k
        if k > 0:
            self.proj1 = nn.Conv3d(3, k, kernel_size=1)
            self.act = _get_activation(activation)
            self.proj2 = nn.Conv3d(k, n, kernel_size=1)

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        if self.k == 0:
            return x_rgb
        y = self.proj2(self.act(self.proj1(x_rgb)))
        return torch.tanh(y * torch.pi)


class AdaptedVAE(nn.Module):
    """
    Wraps an LTX-Video VAE with input/output adapters for wind field data.

    All channels (u, v, b) are in [-1, 1]:
      - u, v: wind velocity normalized by wind_norm
      - b: footprint mask where -1 = building, +1 = fluid
    """

    DEFAULT_INFERENCE_TEMB: float = 0.025

    def __init__(self, vae, in_adapter, out_adapter, *, channels, k, default_temb=0.025):
        super().__init__()
        self.vae = vae
        self.in_adapter = in_adapter
        self.out_adapter = out_adapter
        self.channels = list(channels)
        self.k = int(k)
        self.default_temb = float(default_temb)

    @property
    def config(self):
        return self.vae.config

    def __getattr__(self, name: str) -> Any:
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.vae, name)

    def _as_bcfhw(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5:
            raise ValueError(f"expected 5d tensor, got {tuple(x.shape)}")
        n = int(getattr(self.in_adapter, "n", -1))
        if x.shape[1] in (3, n):
            return x.contiguous()
        if x.shape[2] in (3, n):
            return x.permute(0, 2, 1, 3, 4).contiguous()
        raise ValueError(f"cannot infer channel dimension for shape {tuple(x.shape)}")

    def encode(self, x: torch.Tensor, *args, **kwargs):
        x = self._as_bcfhw(x)
        x_rgb = self.in_adapter(x)
        return self.vae.encode(x_rgb, *args, **kwargs)

    def decode(self, *args, **kwargs):
        out = self.vae.decode(*args, **kwargs)

        def _postprocess(sample):
            return self.out_adapter(sample).clamp(-1.0, 1.0)

        if isinstance(out, (tuple, list)):
            if len(out) == 0 or not torch.is_tensor(out[0]):
                return out
            return (_postprocess(out[0]), *out[1:])

        if hasattr(out, "sample") and torch.is_tensor(out.sample):
            out.sample = _postprocess(out.sample)

        return out


def _load_safetensors_vae(ckpt_path: str | Path):
    """Load a .safetensors VAE checkpoint. Returns (tensors_dict, metadata_dict)."""
    from safetensors import safe_open

    f = safe_open(str(ckpt_path), framework="pt", device="cpu")
    metadata = f.metadata() or {}

    # Separate decoder weights from adapter weights
    decoder_sd, in_sd, out_sd = {}, {}, {}
    for key in f.keys():
        tensor = f.get_tensor(key)
        if key.startswith("decoder."):
            decoder_sd[key[len("decoder."):]] = tensor
        elif key.startswith("in_adapter."):
            in_sd[key[len("in_adapter."):]] = tensor
        elif key.startswith("out_adapter."):
            out_sd[key[len("out_adapter."):]] = tensor

    # Parse metadata strings back to proper types
    import ast
    channels = ast.literal_eval(metadata.get("channels", "['u', 'v', 'b']"))
    n = int(metadata.get("n", len(channels)))
    k = int(metadata.get("k", "0"))
    activation = metadata.get("activation", "gelu")
    default_temb = float(metadata.get("default_temb", "0.0"))

    ckpt = {
        "channels": channels, "n": n, "k": k, "activation": activation,
        "default_temb": default_temb,
        "in_adapter": in_sd,
        "out_adapter": out_sd,
    }
    if decoder_sd:
        ckpt["decoder"] = decoder_sd
    return ckpt


def _load_pt_vae(ckpt_path: str | Path):
    """Load a legacy .pt VAE checkpoint."""
    return torch.load(str(ckpt_path), map_location="cpu", weights_only=False)


def load_adapted_vae(
    vae,
    ckpt_path: str | Path | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
    *,
    default_temb: float = AdaptedVAE.DEFAULT_INFERENCE_TEMB,
    verbose: bool = True,
) -> tuple[AdaptedVAE, dict[str, Any]]:
    """
    Load or create an AdaptedVAE.

    Supports both .safetensors and legacy .pt checkpoint formats.
    If ckpt_path is None, creates passthrough adapters (k=0).
    """
    ckpt = None
    if ckpt_path is not None:
        path = str(ckpt_path)
        if path.endswith(".safetensors"):
            ckpt = _load_safetensors_vae(path)
        else:
            ckpt = _load_pt_vae(path)

    channels = list((ckpt or {}).get("channels", ["u", "v", "b"]))
    n = int((ckpt or {}).get("n", len(channels)))
    k = int((ckpt or {}).get("k", 0))
    activation = str((ckpt or {}).get("activation", "gelu"))
    has_decoder = isinstance((ckpt or {}).get("decoder", None), dict)

    if default_temb == AdaptedVAE.DEFAULT_INFERENCE_TEMB and ckpt is not None:
        ckpt_temb = ckpt.get("default_temb", None)
        if ckpt_temb is not None:
            default_temb = float(ckpt_temb)

    if verbose:
        print(f"[windinet] channels={channels}, n={n}, k={k}, activation={activation}")
        if ckpt_path:
            print(f"[windinet] loading adapter checkpoint: {ckpt_path}")

    in_adapt = InAdapter(n=n, k=k, activation=activation).to(device=device, dtype=dtype)
    out_adapt = OutAdapter(n=n, k=k, activation=activation).to(device=device, dtype=dtype)

    if ckpt is not None:
        if ckpt.get("in_adapter"):
            _strict_load(in_adapt, ckpt["in_adapter"], "in_adapter")
        if ckpt.get("out_adapter"):
            _strict_load(out_adapt, ckpt["out_adapter"], "out_adapter")

    model = AdaptedVAE(
        vae=vae, in_adapter=in_adapt, out_adapter=out_adapt,
        channels=channels, k=k, default_temb=default_temb,
    ).to(device)

    if ckpt is not None and has_decoder:
        decoder_mod = vae.decoder if hasattr(vae, "decoder") else vae
        _strict_load(decoder_mod, ckpt["decoder"], "decoder")
        if verbose:
            print("[windinet] loaded finetuned decoder weights")

    meta = {
        "channels": channels, "n": n, "k": k, "activation": activation,
        "decoder_loaded": bool(ckpt is not None and has_decoder),
        "default_temb": float(default_temb),
    }
    return model, meta
