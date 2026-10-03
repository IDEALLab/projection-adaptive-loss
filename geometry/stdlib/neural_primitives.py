"""
Neural SDF Primitives - Load trained neural networks as SDF shapes.

Neural primitives allow integrating externally trained neural SDF models
into geometry programs. The models must conform to a specific interface
contract (see METADATA schema below).

Primitives:
    neural3d: 3D neural SDF shape
    neural2d: 2D neural SDF shape (evaluated in a specified plane)

Interface Contract:
    The user's inference.py must provide:
    - load_model(device) -> (model, METADATA)
    - forward(model, points, conditioning) -> sdf

    forward() must not use torch.no_grad(), since gradients must flow.

METADATA Schema:
    METADATA = {
        "name": str,                    # Model name
        "version": str,                 # Model version
        "dim": 3 or 2,                  # Spatial dimension
        "bounds": {                     # Valid input ranges
            "x": [min, max],
            "y": [min, max],
            "z": [min, max],            # Only for dim=3
        },
        "input": {
            "points": {
                "shape": "[B, N, 3]" or "[B, N, 2]",
                "range": [min, max],
            }
        },
        "output": {
            "sdf": {"shape": "[B, N]"}
        },
        "conditioning": {               # Optional conditioning parameters
            "latent": {
                "dim": int,             # e.g., 256
                "required": bool,
                "description": str,
            },
            # ... other conditioning params
        },
        "chunk_size": int,              # Optional, default 65536
    }
"""

import importlib.util
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn

from geometry.loader import loadable_shape

from ..core import Shape
from .primitives import PLANE_AXES, _get_2d_coords

# Constants

DEFAULT_CHUNK_SIZE = 65536  # 64K points per forward pass

# Override METADATA fields for benchmarking (merged after load_model).
# Set from external code: neural_primitives._metadata_overrides = {"precision": "fp32"}
_metadata_overrides: dict = {}

# Reuse inference.py modules by resolved path so module-level model caches
# survive repeated neural shape reconstruction within the same process.
_inference_module_cache: dict[Path, Any] = {}


# Device Selection


def _get_device() -> torch.device:
    """
    Auto-select best available device: cuda > mps > cpu.

    Returns:
        torch.device for model loading
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# Model Loading


def _load_inference_module(path: Path) -> Any:
    """
    Dynamically load a user's inference.py module.

    Args:
        path: Path to the inference.py file

    Returns:
        The loaded module

    Raises:
        FileNotFoundError: If inference.py doesn't exist
        ImportError: If module can't be loaded
    """
    resolved_path = path.resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(
            f"Neural model inference module not found: {resolved_path}\n"
            f"Expected file: inference.py in the model directory"
        )

    cached_module = _inference_module_cache.get(resolved_path)
    if cached_module is not None:
        return cached_module

    module_name = f"neural_inference_{len(_inference_module_cache)}"
    spec = importlib.util.spec_from_file_location(module_name, resolved_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {resolved_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    _inference_module_cache[resolved_path] = module

    return module


def _validate_metadata(metadata: dict[str, Any], expected_dim: int, path: Path) -> None:
    """
    Validate METADATA conforms to expected schema.

    Args:
        metadata: METADATA dict from user's inference module
        expected_dim: Expected dimension (2 or 3)
        path: Path for error messages

    Raises:
        ValueError: If metadata is invalid
    """
    required_keys = ["name", "version", "dim", "bounds", "input", "output"]
    missing = [k for k in required_keys if k not in metadata]
    if missing:
        raise ValueError(
            f"Neural shape at {path}: METADATA missing required keys: {missing}\n"
            f"See neural_primitives.py docstring for required schema."
        )

    # Validate dimension
    if metadata["dim"] != expected_dim:
        raise ValueError(
            f"Neural shape at {path}: METADATA.dim={metadata['dim']} "
            f"but expected {expected_dim} for neural{'3d' if expected_dim == 3 else '2d'}()"
        )

    # Validate input shape format
    input_info = metadata.get("input", {})
    points_info = input_info.get("points", {})
    input_shape = points_info.get("shape", "")

    if expected_dim == 3 and "3" not in input_shape:
        raise ValueError(
            f"Neural shape at {path}: METADATA.input.points.shape must contain '3' for 3D, "
            f"got: {input_shape}"
        )
    if expected_dim == 2 and "2" not in input_shape:
        raise ValueError(
            f"Neural shape at {path}: METADATA.input.points.shape must contain '2' for 2D, "
            f"got: {input_shape}"
        )

    # Validate output shape format
    output_info = metadata.get("output", {})
    sdf_info = output_info.get("sdf", {})
    output_shape = sdf_info.get("shape", "")

    if "[B, N]" not in output_shape and "B" not in output_shape:
        raise ValueError(
            f"Neural shape at {path}: METADATA.output.sdf.shape should be '[B, N]', "
            f"got: {output_shape}"
        )


def _validate_conditioning(
    conditioning: dict[str, Tensor],
    metadata: dict[str, Any],
    batch_size: int,
    path: Path,
) -> dict[str, Tensor]:
    """
    Validate conditioning parameters match METADATA spec.

    Args:
        conditioning: User-provided conditioning dict
        metadata: METADATA from model
        batch_size: Expected batch size from YAML
        path: Path for error messages

    Returns:
        Validated conditioning dict (may expand tensors to match batch_size)

    Raises:
        ValueError: If conditioning doesn't match spec
        TypeError: If conditioning values aren't tensors
    """
    spec = metadata.get("conditioning", {})
    validated = {}

    # Check required conditioning params are provided
    for param_name, param_spec in spec.items():
        if param_spec.get("required", False) and param_name not in conditioning:
            desc = param_spec.get("description", "N/A")
            dim = param_spec.get("dim", "N/A")
            raise ValueError(
                f"Neural shape at {path}: Missing required conditioning '{param_name}'\n"
                f"Description: {desc}\n"
                f"Expected dimension: {dim}"
            )

    # Validate provided conditioning
    for param_name, value in conditioning.items():
        if param_name not in spec:
            available = list(spec.keys()) if spec else "(none)"
            raise ValueError(
                f"Neural shape at {path}: Unknown conditioning '{param_name}'\n"
                f"Available: {available}"
            )

        param_spec = spec[param_name]
        expected_dim = param_spec.get("dim")

        # Validate tensor format
        if not isinstance(value, Tensor):
            raise TypeError(
                f"Neural shape at {path}: Conditioning '{param_name}' must be a Tensor, "
                f"got {type(value).__name__}"
            )

        # Validate tensor is 2D [B, D]
        if value.dim() != 2:
            raise ValueError(
                f"Neural shape at {path}: Conditioning '{param_name}' must be [B, {expected_dim}], "
                f"got shape {list(value.shape)}"
            )

        # Validate dimension D
        if expected_dim is not None and value.shape[1] != expected_dim:
            raise ValueError(
                f"Neural shape at {path}: Conditioning '{param_name}' dimension mismatch: "
                f"expected D={expected_dim}, got D={value.shape[1]}"
            )

        # Validate/expand batch size
        if value.shape[0] == 1 and batch_size > 1:
            # Expand single conditioning to match batch
            value = value.expand(batch_size, -1).clone()
        elif value.shape[0] != batch_size:
            raise ValueError(
                f"Neural shape at {path}: Conditioning '{param_name}' batch size mismatch: "
                f"expected B={batch_size}, got B={value.shape[0]}"
            )

        validated[param_name] = value

    return validated


def _probe_model(
    forward_fn: Callable,
    model: nn.Module,
    metadata: dict[str, Any],
    conditioning: dict[str, Tensor],
    device: torch.device,
    path: Path,
) -> None:
    """
    Probe model with test input to verify forward pass AND gradient flow.

    Args:
        forward_fn: The forward function from user's inference module
        model: Loaded neural network model
        metadata: METADATA from model
        conditioning: Validated conditioning dict
        device: Target device
        path: Path for error messages

    Raises:
        RuntimeError: If forward pass fails or gradients don't flow
    """
    dim = metadata["dim"]

    # Create small test input with requires_grad
    test_points = torch.randn(1, 4, dim, device=device, requires_grad=True)

    # Get conditioning for single batch (detach to avoid graph conflicts with shared tensors)
    test_cond = {}
    for k, v in conditioning.items():
        test_cond[k] = v[:1].detach().clone().to(device)

    try:
        # Probe must run with grad enabled, callers (e.g. inference trajectory
        # logging) may invoke us inside a `with torch.no_grad():` block, but the
        # probe itself needs gradient flow to validate the model.
        with torch.enable_grad():
            sdf = forward_fn(model, test_points, test_cond)

            # Validate output shape
            if sdf.shape != (1, 4):
                raise RuntimeError(
                    f"Neural shape at {path}: forward() returned shape {list(sdf.shape)}, "
                    f"expected [1, 4]. Output must be [B, N] format."
                )

            # Test gradient flow
            loss = sdf.sum()
            loss.backward()

        if test_points.grad is None:
            raise RuntimeError(
                f"Neural shape at {path}: Gradients did not flow through the model.\n"
                f"Your inference.py forward() must not use torch.no_grad().\n"
                f"Remove any `with torch.no_grad():` blocks from forward()."
            )

    except RuntimeError as e:
        if "no_grad" in str(e).lower() or "does not require grad" in str(e).lower():
            raise RuntimeError(
                f"Neural shape at {path}: Model blocks gradient computation.\n"
                f"Your inference.py forward() appears to use torch.no_grad().\n"
                f"Remove `with torch.no_grad():` to enable gradient-based optimization.\n"
                f"Original error: {e}"
            )
        raise
    except Exception as e:
        raise RuntimeError(f"Neural shape at {path}: Model probe failed.\nError: {e}")


# Neural Primitives


def check_shape_def_neural3d(shape_def: dict[str, Any]) -> None:
    if "path" not in shape_def:
        raise ValueError(
            "neural3d requires 'path' field pointing to model directory. "
            "Example: path: ./my_neural_model"
        )


@loadable_shape(
    check_shape_def=check_shape_def_neural3d,
    # Skip validation (dynamic params or not graph-supported):
    skip_validation=True,
    inject=("batch_size", "yaml_dir", "device"),
)
def neural3d(
    path: str,
    batch_size: int,
    yaml_dir: Path,
    device: torch.device = None,
    **conditioning,
) -> Shape:
    """
    Create a 3D neural SDF shape from a trained model.

    The model must provide an inference.py file with:
    - load_model(device) -> (model, METADATA)
    - forward(model, points, conditioning) -> sdf

    Points are [B, N, 3], output SDF is [B, N].
    IMPORTANT: forward() must NOT use torch.no_grad()!

    Args:
        path: Path to model directory (relative to YAML file)
        batch_size: Batch size from YAML bounds
        yaml_dir: Directory containing the YAML file (for path resolution)
        device: PyTorch device for model inference (auto-select if None)
        **conditioning: Conditioning parameters as [B, K] tensors
                       (e.g., latent=torch.tensor([[0.5, ...]]))

    Returns:
        Shape object with is_neural=True

    Raises:
        FileNotFoundError: If model path doesn't exist
        ValueError: If METADATA is invalid or conditioning doesn't match
        RuntimeError: If model forward pass fails or blocks gradients

    Example YAML:
        shapes:
          learned_shape:
            type: neural3d
            path: ./my_model  # relative to YAML
            latent: [0.5, 0.0, ..., 0.0]  # conditioning
    """
    # Auto-select device if not provided
    if device is None:
        device = _get_device()

    # Resolve path relative to YAML
    model_dir = yaml_dir / path
    inference_path = model_dir / "inference.py"

    # Load inference module
    inference_module = _load_inference_module(inference_path)

    # Validate module has required functions
    if not hasattr(inference_module, "load_model"):
        raise AttributeError(
            f"Neural shape at {model_dir}: inference.py must define load_model(device) function"
        )
    if not hasattr(inference_module, "forward"):
        raise AttributeError(
            f"Neural shape at {model_dir}: inference.py must define "
            f"forward(model, points, conditioning) function"
        )

    # Load model
    model, metadata = inference_module.load_model(device)
    metadata.update(_metadata_overrides)

    # Validate metadata
    _validate_metadata(metadata, expected_dim=3, path=model_dir)

    # Validate conditioning
    cond_validated = _validate_conditioning(
        conditioning, metadata, batch_size, model_dir
    )

    # Freeze model weights (prevent accidental training, but allow input gradients)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Probe model to verify forward pass and gradient flow
    _probe_model(
        inference_module.forward, model, metadata, cond_validated, device, model_dir
    )

    # Get chunk size
    chunk_size = metadata.get("chunk_size", DEFAULT_CHUNK_SIZE)

    # Move conditioning to model device at construction time
    cond_on_device = {k: v.to(device) for k, v in cond_validated.items()}

    # METADATA-driven: compile model if requested
    _forward = inference_module.forward
    if device.type == "cuda" and metadata.get("compile", False):
        _forward = torch.compile(inference_module.forward)

    # METADATA-driven: precision for autocast
    precision = metadata.get("precision", "fp32")
    use_autocast = device.type == "cuda" and precision in ("fp16", "bf16")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)

    # Create SDF closure
    def sdf_fn(p: Tensor) -> Tensor:
        """
        Evaluate neural SDF at points.

        Args:
            p: [B, N, 3] points (already on model device via Shape.__call__)

        Returns:
            [B, N] SDF values
        """
        N = p.shape[-2]
        B = batch_size

        # Expand if input batch dim is 1 but we need B
        p_batched = p.expand(B, -1, -1)  # [B, N, 3]

        def _eval_chunks(p_in):
            if N <= chunk_size:
                return _forward(model, p_in, cond_on_device)
            sdf_chunks = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                p_chunk = p_in[:, start:end, :]
                sdf_chunks.append(_forward(model, p_chunk, cond_on_device))
            return torch.cat(sdf_chunks, dim=1)

        if use_autocast:
            with torch.autocast("cuda", dtype=autocast_dtype):
                sdf = _eval_chunks(p_batched)
            return sdf.float()  # ensure fp32 output for downstream DMC

        return _eval_chunks(p_batched)

    return Shape(
        sdf_fn, batch_size=batch_size, plane=None, is_neural=True, device=device
    )


def check_shape_def_neural2d(shape_def: dict[str, Any]) -> None:
    if "path" not in shape_def:
        raise ValueError(
            "neural2d requires 'path' field pointing to model directory. "
            "Example: path: ./my_neural_model"
        )
    if "plane" not in shape_def:
        raise ValueError("neural2d requires 'plane' field. Example: plane: xy")


@loadable_shape(
    check_shape_def=check_shape_def_neural2d,
    # Skip validation (dynamic params or not graph-supported):
    skip_validation=True,
    inject=("batch_size", "yaml_dir", "device"),
)
def neural2d(
    path: str,
    plane: str,
    batch_size: int,
    yaml_dir: Path,
    device: torch.device = None,
    **conditioning,
) -> Shape:
    """
    Create a 2D neural SDF shape from a trained model.

    Similar to neural3d but:
    - Requires 'plane' parameter ('xy', 'xz', 'yz')
    - Extracts 2D coordinates from 3D points based on plane
    - Model receives [B, N, 2] points

    Args:
        path: Path to model directory (relative to YAML file)
        plane: Which plane the 2D shape lies in ('xy', 'xz', 'yz')
        batch_size: Batch size from YAML bounds
        yaml_dir: Directory containing the YAML file
        device: PyTorch device for model inference (auto-select if None)
        **conditioning: Conditioning parameters as [B, K] tensors

    Returns:
        Shape object with plane set and is_neural=True

    Example YAML:
        shapes:
          learned_profile:
            type: neural2d
            path: ./profile_model
            plane: xy
            latent: [0.5, ...]
    """
    # Validate plane
    if plane not in PLANE_AXES:
        raise ValueError(
            f"neural2d() got invalid plane '{plane}'. "
            f"Must be one of: {tuple(PLANE_AXES.keys())}"
        )

    # Auto-select device if not provided
    if device is None:
        device = _get_device()

    # Resolve path relative to YAML
    model_dir = yaml_dir / path
    inference_path = model_dir / "inference.py"

    # Load inference module
    inference_module = _load_inference_module(inference_path)

    # Validate module has required functions
    if not hasattr(inference_module, "load_model"):
        raise AttributeError(
            f"Neural shape at {model_dir}: inference.py must define load_model(device) function"
        )
    if not hasattr(inference_module, "forward"):
        raise AttributeError(
            f"Neural shape at {model_dir}: inference.py must define "
            f"forward(model, points, conditioning) function"
        )

    # Load model
    model, metadata = inference_module.load_model(device)
    metadata.update(_metadata_overrides)

    # Validate metadata
    _validate_metadata(metadata, expected_dim=2, path=model_dir)

    # Validate conditioning
    cond_validated = _validate_conditioning(
        conditioning, metadata, batch_size, model_dir
    )

    # Freeze model weights
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Probe model
    _probe_model(
        inference_module.forward, model, metadata, cond_validated, device, model_dir
    )

    # Get chunk size
    chunk_size = metadata.get("chunk_size", DEFAULT_CHUNK_SIZE)

    # Move conditioning to model device at construction time
    cond_on_device = {k: v.to(device) for k, v in cond_validated.items()}

    # METADATA-driven: compile model if requested
    _forward = inference_module.forward
    if device.type == "cuda" and metadata.get("compile", False):
        _forward = torch.compile(inference_module.forward)

    # METADATA-driven: precision for autocast
    precision = metadata.get("precision", "fp32")
    use_autocast = device.type == "cuda" and precision in ("fp16", "bf16")
    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(precision)

    # Create SDF closure
    def sdf_fn(p: Tensor) -> Tensor:
        """
        Evaluate neural 2D SDF at points.

        Args:
            p: [B, N, 3] points (3D, already on model device via Shape.__call__)

        Returns:
            [B, N] SDF values
        """
        N = p.shape[-2]
        B = batch_size

        # Extract 2D coordinates based on plane
        p_2d = _get_2d_coords(p, plane)  # [B, N, 2]

        # Expand if input batch dim is 1 but we need B
        p_batched = p_2d.expand(B, -1, -1)  # [B, N, 2]

        def _eval_chunks(p_in):
            if N <= chunk_size:
                return _forward(model, p_in, cond_on_device)
            sdf_chunks = []
            for start in range(0, N, chunk_size):
                end = min(start + chunk_size, N)
                p_chunk = p_in[:, start:end, :]
                sdf_chunks.append(_forward(model, p_chunk, cond_on_device))
            return torch.cat(sdf_chunks, dim=1)

        if use_autocast:
            with torch.autocast("cuda", dtype=autocast_dtype):
                sdf = _eval_chunks(p_batched)
            return sdf.float()

        return _eval_chunks(p_batched)

    return Shape(
        sdf_fn, batch_size=batch_size, plane=plane, is_neural=True, device=device
    )
