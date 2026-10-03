"""Union operation."""

from typing import Any

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import assert_at_least_2_shapes, loadable_shape
from geometry.stdlib.operations.plane_ops import _get_common_plane, _validate_same_plane
from geometry.utils import validate_batch_sizes


def deserialize_float(val: Any) -> float:
    # k should be scalar float, convert if tensorified
    if isinstance(val, Tensor):
        return float(val.squeeze())
    return float(val)


def check_shape_def(shape_def: dict[str, Any]) -> None:
    assert_at_least_2_shapes("smooth_union")(shape_def)
    k = shape_def.get("k")
    if k is None:
        raise ValueError(
            "smooth_union requires 'k' parameter for blend sharpness. Example: k: 5.0"
        )


@loadable_shape(
    check_shape_def=check_shape_def,
    deserialize_args={"k": deserialize_float},
)
def smooth_union(*shapes: Shape, k: float) -> Shape:
    """
    Compute the smooth union of multiple shapes.

    Unlike standard union (hard min), smooth_union creates a rounded
    blend between shapes. This has continuous gradients everywhere,
    making it suitable for gradient-based optimization.

    Formula: -log(sum(exp(-k*sdf_i))) / k
    Uses logsumexp for numerical stability.

    Args:
        *shapes: Two or more Shape objects to combine
                 For 2D shapes, all must be in the same plane.
        k: Blend sharpness parameter (REQUIRED, scalar float, must be > 0)
           - Larger k -> sharper transition (approaches hard min)
           - Smaller k -> smoother/rounder blend
           - Typical range: 2.0 to 50.0

    Returns:
        Shape representing the smooth union of all inputs

    Raises:
        ValueError: If fewer than 2 shapes provided
        ValueError: If k is not positive
        ValueError: If batch sizes are incompatible
        ValueError: If 2D shapes have different planes
        TypeError: If k is not a number

    Note:
        k=0 is equivalent to hard union (min) but is not differentiable
        at the boundaries where shapes meet. Use k > 0 for differentiability.

    Example:
        >>> s1 = sphere(radius=torch.tensor([[1.0]]), center=torch.tensor([[0., 0., 0.]]))
        >>> s2 = sphere(radius=torch.tensor([[1.0]]), center=torch.tensor([[1., 0., 0.]]))
        >>> blended = smooth_union(s1, s2, k=5.0)  # Rounded blend
    """
    if len(shapes) < 2:
        raise ValueError(f"smooth_union requires at least 2 shapes, got {len(shapes)}")

    # Validate 2D shapes have same plane
    _validate_same_plane(list(shapes), "smooth_union")

    # Validate k is a scalar number
    if not isinstance(k, (int, float)):
        raise TypeError(f"k must be a number, got {type(k).__name__}")

    if k <= 0:
        raise ValueError(
            f"smooth_union requires k > 0, got {k}. "
            "Note: k=0 is equivalent to hard union (min) but is not differentiable."
        )

    batch_size = validate_batch_sizes(list(shapes), "smooth_union")
    _dev = shapes[0].device
    k_tensor = torch.tensor(k, dtype=torch.float32, device=_dev)

    def sdf_fn(p: Tensor) -> Tensor:
        # Evaluate all SDFs: each is [B, N]
        sdfs = [shape(p) for shape in shapes]

        # Stack: [num_shapes, B, N]
        stacked = torch.stack(sdfs, dim=0)

        # Smooth min via logsumexp:
        # smooth_min(a, b) = -log(exp(-k*a) + exp(-k*b)) / k
        #                  = -logsumexp([-k*a, -k*b]) / k
        scaled = -k_tensor * stacked  # [num_shapes, B, N]

        # logsumexp along the shapes dimension
        lse = torch.logsumexp(scaled, dim=0)  # [B, N]

        return -lse / k_tensor

    # Preserve plane if all inputs have the same plane
    common_plane = _get_common_plane(list(shapes))
    return Shape(sdf_fn, batch_size=batch_size, plane=common_plane, device=_dev)
