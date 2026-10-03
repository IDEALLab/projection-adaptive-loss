"""geometry neural3d adapter for the BWB SDF network.

Interface: METADATA, load_model(device), forward(model, points, conditioning).
Params are normalized inside forward(), so YAML params stay in real units.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

# geometry loads this file standalone, so make `import sdf_net` resolve.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from sdf_net import BWBSDFNet  # noqa: E402  (sys.path injection above)


METADATA = {
    "name": "bwb_sdf",
    "version": "1.0.0",
    "dim": 3,
    "bounds": {
        "x": [-1.2, 0.2],
        "y": [-1.05, 1.05],
        "z": [-0.2, 0.2],
    },
    "input": {
        "points": {
            "shape": "[B, N, 3]",
            "range": [-1.5, 1.5],
        }
    },
    "output": {
        "sdf": {"shape": "[B, N]"}
    },
    "conditioning": {
        "B1": {"dim": 1, "required": True, "description": "Body param 1 [0.10, 0.20]"},
        "B2": {"dim": 1, "required": True, "description": "Body param 2 [0.05, 0.20]"},
        "B3": {"dim": 1, "required": True, "description": "Body param 3 [0.20, 0.70]"},
        "C2": {"dim": 1, "required": True, "description": "Chord param 2 [0.55, 0.85]"},
        "C3": {"dim": 1, "required": True, "description": "Chord param 3 [0.18, 0.28]"},
        "C4": {"dim": 1, "required": True, "description": "Chord param 4 [0.06, 0.09]"},
        "S1": {"dim": 1, "required": True, "description": "Sweep param 1 [40, 60] deg"},
        "S2": {"dim": 1, "required": True, "description": "Sweep param 2 [40, 60] deg"},
        "S3": {"dim": 1, "required": True, "description": "Sweep param 3 [24, 40] deg"},
    },
    # Per-chunk N for SDF eval, caps peak GPU memory in the wide block2 Linear.
    "chunk_size": 4096,
}

PARAM_ORDER = ["B1", "B2", "B3", "C2", "C3", "C4", "S1", "S2", "S3"]


def _resolve_checkpoint() -> Path:
    """Return the BWB SDF weights path via PAL's artifact system."""
    from pal.benchmarks.engineering.e1_bwb._artifacts import ensure_artifact_path
    return Path(ensure_artifact_path("bwb_sdf_weights"))


_model = None
_param_min = None
_param_max = None


def load_model(
    device: torch.device,
    half: bool = False,
) -> tuple[nn.Module, dict]:
    """Load trained BWB SDF net. Returns (model, METADATA).

    Args:
        device: target device
        half: if True, convert model to fp16 for faster inference

    Module state is mutated only on the first call (functorch-safe), a later
    call with a different device raises.
    """
    global _model, _param_min, _param_max

    if _model is None:
        ckpt = torch.load(_resolve_checkpoint(), map_location=device, weights_only=False)
        cfg = ckpt["config"]

        _model = BWBSDFNet(
            fourier_bands=cfg.get("fourier_bands", 10),
            hidden_dim=cfg.get("hidden_dim", 512),
            cond_dim=cfg.get("cond_dim", 9),
            n_blocks=cfg.get("n_blocks", 3),
            activation=cfg.get("activation", "gelu"),
            omega_0=cfg.get("omega_0", 30.0),
        )
        _model.load_state_dict(ckpt["model_state_dict"])
        _model = _model.to(device).eval()
        if half and device.type == "cuda":
            _model = _model.half()
        _param_min = ckpt["param_min"].to(device)
        _param_max = ckpt["param_max"].to(device)
    else:
        cached_device = next(_model.parameters()).device
        if cached_device != torch.device(device):
            raise RuntimeError(
                f"bwb_sdf.load_model: cached model is on {cached_device}, "
                f"caller requested {device}. The first call wins; re-import "
                f"the module to switch devices."
            )

    return _model, METADATA


def forward(
    model: nn.Module,
    points: Tensor,
    conditioning: dict[str, Tensor],
) -> Tensor:
    """Evaluate BWB SDF. No torch.no_grad(), gradients must flow.

    Args:
        model: BWBSDFNet from load_model()
        points: [B, N, 3] xyz coordinates
        conditioning: {"B1": [B,1], ..., "S3": [B,1]} in real units

    Returns:
        [B, N] SDF values (negative inside, positive outside)
    """
    dev = points.device
    B, N, _ = points.shape

    # X-flip: model trained with LE at -x, aircraft convention is LE at +x.
    flip = points.new_tensor([-1.0, 1.0, 1.0])
    points = points * flip

    cond_raw = torch.cat(
        [conditioning[k].to(dev) for k in PARAM_ORDER], dim=-1
    )  # [B, 9]

    cond_norm = 2.0 * (cond_raw - _param_min.to(dev)) / (
        _param_max.to(dev) - _param_min.to(dev) + 1e-8
    ) - 1.0  # [B, 9]

    model_dtype = next(model.parameters()).dtype
    points = points.to(dtype=model_dtype)
    cond_norm = cond_norm.to(dtype=model_dtype)

    # BWBSDFNet expects [B, Q, 3] and [B, 9] -> [B, Q], chunked along N.
    chunk_n = METADATA["chunk_size"]
    if N <= chunk_n:
        return model(points, cond_norm)
    sdf_chunks: list[Tensor] = []
    for start in range(0, N, chunk_n):
        end = min(start + chunk_n, N)
        sdf_chunks.append(model(points[:, start:end, :], cond_norm))
    return torch.cat(sdf_chunks, dim=1)
