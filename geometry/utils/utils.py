"""
Shared utilities for geometry.

Strict tensor validation and batch handling for primitives and operations.

Core Functions:
- validate_tensor(): Validate input is [B, K] tensor (no conversion)
- get_batch_size(): Get batch size from tensors/shapes (strict matching)
- tensorify_shape_def(): Convert YAML values to [B, K] tensors (YAML path only)
- validate_batch_sizes(): Validate shapes have identical batch sizes
"""

from typing import Any

import torch
from torch import Tensor


def validate_tensor(x, vec_size: int, name: str = "parameter") -> Tensor:
    """
    Validate that input is a [B, K] tensor. NO auto-conversion.

    Args:
        x: Input (must be Tensor)
        vec_size: Expected K dimension
        name: Parameter name for error messages

    Returns:
        The input tensor unchanged (if valid)

    Raises:
        TypeError: If x is not a Tensor
        ValueError: If tensor shape is not [B, K]

    Examples:
        >>> validate_tensor(torch.tensor([[1.0]]), vec_size=1, name="radius")
        tensor([[1.0]])

        >>> validate_tensor(1.0, vec_size=1, name="radius")
        TypeError: radius must be a Tensor with shape [B, 1]...

        >>> validate_tensor(torch.tensor([1.0]), vec_size=1, name="radius")
        ValueError: radius must be [B, 1] format, got shape [1]...
    """
    if not isinstance(x, Tensor):
        raise TypeError(
            f"{name} must be a Tensor with shape [B, {vec_size}], "
            f"got {type(x).__name__}.\n"
            f"For single value, use: torch.tensor([[value]])\n"
            f"For batched values, use: torch.tensor([[v1], [v2], ...])"
        )

    if x.dim() != 2:
        raise ValueError(
            f"{name} must be [B, {vec_size}] format, got shape {list(x.shape)}.\n"
            f"For single value, use: torch.tensor([[value]])\n"
            f"For batched values, use: values.unsqueeze(1)"
        )

    if x.shape[1] != vec_size:
        raise ValueError(
            f"{name} must have K={vec_size}, got shape {list(x.shape)}.\n"
            f"Expected [B, {vec_size}] format."
        )

    return x


def get_batch_size(*items, names: list[str] | None = None) -> int:
    """
    Get batch size from tensors and/or shapes. All must match EXACTLY.

    NO broadcasting - if one item has B=5, all must have B=5.

    Args:
        *items: Tensors [B, K] or Shape objects
        names: Optional param names for error messages

    Returns:
        Common batch size (int >= 1)

    Raises:
        ValueError: If batch sizes don't all match

    Examples:
        >>> t1 = torch.zeros(5, 3)
        >>> t2 = torch.zeros(5, 1)
        >>> get_batch_size(t1, t2)  # Returns 5

        >>> t1 = torch.zeros(5, 3)
        >>> t2 = torch.zeros(1, 1)
        >>> get_batch_size(t1, t2)  # ValueError: batch sizes must match
    """
    from geometry.core import Shape  # Import here to avoid circular dependency

    if not items:
        raise ValueError("No items provided")

    batch_sizes = []
    for item in items:
        if isinstance(item, Shape):
            if item.batch_size is None:
                raise ValueError("Shape has batch_size=None (internal error)")
            batch_sizes.append(item.batch_size)
        elif isinstance(item, Tensor):
            if item.dim() < 1:
                raise ValueError(
                    f"Tensor must be at least 1D, got shape {list(item.shape)}"
                )
            batch_sizes.append(item.shape[0])
        else:
            raise TypeError(f"Expected Tensor or Shape, got {type(item).__name__}")

    unique = set(batch_sizes)
    if len(unique) > 1:
        if names:
            details = ", ".join(f"{n}={b}" for n, b in zip(names, batch_sizes))
        else:
            details = str(batch_sizes)
        raise ValueError(
            f"Batch size mismatch: {details}. "
            f"All parameters must have the same batch size."
        )

    return batch_sizes[0]


def tensorify_shape_def(
    shape_def: dict[str, Any], batch_size: int = 1, device=None
) -> dict[str, Any]:
    """
    Convert Python types -> [B, K] tensors in shape definition.

    This is the ONLY place where auto-conversion happens.
    Used by YAML loading path, not direct Python API.

    Args:
        shape_def: Shape definition dict (after substitute_params)
        batch_size: Target batch size (from YAML bounds section)

    Returns:
        Shape definition with all numerics converted to [batch_size, K] tensors

    Rules:
        - Scalars (int, float) -> [batch_size, 1] tensor (value repeated)
        - Lists of numbers -> [batch_size, K] tensor (values repeated)
        - Tensors -> pass through unchanged (must already be [B, K])
        - Strings, bools, None -> pass through unchanged

    Raises:
        ValueError: If mixed list contains batched tensors (B>1) with literals

    Examples:
        >>> tensorify_shape_def({"radius": 1.0}, batch_size=1)
        {"radius": tensor([[1.0]])}

        >>> tensorify_shape_def({"radius": 1.0}, batch_size=3)
        {"radius": tensor([[1.0], [1.0], [1.0]])}

        >>> tensorify_shape_def({"center": [0, 0, 0]}, batch_size=2)
        {"center": tensor([[0., 0., 0.], [0., 0., 0.]])}
    """

    def tensorify_value(x):
        """Recursively convert value to tensor format."""
        if isinstance(x, bool):
            # Booleans are config flags, not numeric params, skip
            return x

        if isinstance(x, Tensor):
            # Already a tensor, move to target device if needed
            if device is not None and x.device != torch.device(device):
                return x.to(device)
            return x

        if isinstance(x, (int, float)):
            # Scalar -> [batch_size, 1] with repeated value
            return torch.tensor(
                [[float(x)]] * batch_size, dtype=torch.float32, device=device
            )

        if isinstance(x, (list, tuple)):
            # Check if list contains any tensors
            has_tensors = any(isinstance(elem, Tensor) for elem in x)

            if has_tensors:
                # Mixed list detected - validate and convert
                tensor_list = []
                elem_batch_sizes = []

                for i, elem in enumerate(x):
                    if isinstance(elem, Tensor):
                        # Validate tensor is [B, 1]
                        if elem.dim() != 2 or elem.shape[1] != 1:
                            raise ValueError(
                                f"Tensor in list must be [B, 1], got shape {list(elem.shape)}"
                            )
                        tensor_list.append(elem)
                        elem_batch_sizes.append(elem.shape[0])
                    elif isinstance(elem, (int, float)):
                        # Convert to [batch_size, 1] tensor
                        t = torch.tensor(
                            [[float(elem)]] * batch_size,
                            dtype=torch.float32,
                            device=device,
                        )
                        tensor_list.append(t)
                        elem_batch_sizes.append(batch_size)
                    else:
                        raise TypeError(
                            f"List contains unsupported type: {type(elem).__name__}"
                        )

                # Check for invalid batched mixed lists
                # Only allow: all same batch size
                unique_sizes = set(elem_batch_sizes)
                if len(unique_sizes) > 1:
                    # Check if mismatch involves different non-batch_size values
                    raise ValueError(
                        f"Batch size mismatch in list: {elem_batch_sizes}\n"
                        f"All parameters must have the same batch size."
                    )

                # All same batch size - concatenate
                return torch.cat(tensor_list, dim=1)  # [B, K]

            # Check if list contains only numeric types
            if all(isinstance(elem, (int, float)) for elem in x):
                # Pure numeric list -> [batch_size, K] tensor with repeated row
                row = [float(elem) for elem in x]
                return torch.tensor(
                    [row] * batch_size, dtype=torch.float32, device=device
                )
            # Non-numeric list (strings, etc.) -> pass through
            # This handles: ['sphere1', 'sphere2'] -> unchanged
            return x

        if isinstance(x, dict):
            # Recursively process dict
            return {k: tensorify_value(v) for k, v in x.items()}

        # String, bool, None, etc. -> pass through
        return x

    return tensorify_value(shape_def)


def validate_batch_sizes(shapes: list, operation_name: str) -> int:
    """
    Validate all shapes have identical batch_size.

    Args:
        shapes: List of Shape objects to validate
        operation_name: Name of the operation (for error messages)

    Returns:
        Common batch size (always int >= 1)

    Raises:
        ValueError: If batch sizes don't match or any is None
    """
    batch_sizes = [s.batch_size for s in shapes]

    # Check for None (shouldn't happen in new architecture)
    if any(bs is None for bs in batch_sizes):
        raise ValueError(
            f"{operation_name}() encountered shape with batch_size=None. "
            f"This is an internal error."
        )

    unique = set(batch_sizes)
    if len(unique) > 1:
        # Build helpful error message for YAML users
        details = ", ".join(
            f"shape {i + 1}: batch_size={bs}" for i, bs in enumerate(batch_sizes)
        )
        raise ValueError(
            f"{operation_name} requires all shapes to have the same batch size. "
            f"Got mismatched sizes: [{details}]. "
            f"Check that all shapes in your YAML use the same parameter batching "
            f"(e.g., all use scalar params, or all use the same number of batched params)."
        )

    return batch_sizes[0]
