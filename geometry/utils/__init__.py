"""
Geometry utilities.

- utils: Tensor validation and batch handling
"""

from .utils import (
    get_batch_size,
    tensorify_shape_def,
    validate_batch_sizes,
    validate_tensor,
)

__all__ = [
    "validate_tensor",
    "get_batch_size",
    "tensorify_shape_def",
    "validate_batch_sizes",
]
