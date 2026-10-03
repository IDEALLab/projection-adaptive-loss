"""Shell operation."""

from enum import Enum, auto
from typing import Self

import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.warning import _warn_neural_csg


class ShellMode(Enum):
    """Modi used by `shell`."""

    Inward = auto()
    """Preserve outer dimensions."""
    Outward = auto()
    """Preserve inner dimensions."""
    Symmetric = auto()
    """Centered on surface."""

    @classmethod
    def from_str(cls, val: str) -> Self:
        """Convert from a string."""
        return cls[val.capitalize()]


@loadable_shape(deserialize_args={"mode": ShellMode.from_str})
def shell(shape: Shape, thickness: float, mode: ShellMode = ShellMode.Inward) -> Shape:
    """
    Hollow out a solid bounded by a shape.

    The solid bounded by the original shape is hollowed out, such that for
    - `mode = ShellMode.Inward`, the resulting outer boundary is the original shape,
    - `mode = ShellMode.Outward`, the resulting inner boundary is the original shape,
    - `mode = ShellMode.Symmetric`, the original shape is centered between the resulting inner and outer boundaries.
    SDF formulae:
    - `mode = ShellMode.Inward`: max(sdf(p), -sdf(p) - thickness)
    - `mode = ShellMode.Outward`: max(sdf(p) - thickness, -sdf(p))
    - `mode = ShellMode.Symmetric`: |sdf(p)| - thickness / 2

    Args:
        shape: A Shape objects.
        thickness: Total wall thickness of the shell.
        mode: See `ShellMode`

    Returns:
        Shape representing the shell of the input shape

    Example:
        >>> b = box(size=torch.tensor([[2., 2., 2.]]))
        >>> s = shell(b, thickness=0.5, mode=ShellMode.Inward)
    """

    # Warn if any neural shapes are involved
    _warn_neural_csg("shell", [shape])

    match mode:
        case ShellMode.Inward:

            def sdf_fn(p: Tensor) -> Tensor:
                """Evaluate SDF and take elementwise inward shell."""
                sdf = shape(p)
                return torch.maximum(sdf, -sdf - thickness)

        case ShellMode.Outward:

            def sdf_fn(p: Tensor) -> Tensor:
                # Evaluate SDF and take elementwise outward shell
                sdf = shape(p)
                return torch.maximum(sdf - thickness, -sdf)

        case ShellMode.Symmetric:

            def sdf_fn(p: Tensor) -> Tensor:
                # Evaluate SDF and take elementwise symmetric shell
                return torch.abs(shape(p)) - 0.5 * thickness

    return Shape(
        sdf_fn, batch_size=shape.batch_size, plane=shape.plane, device=shape.device
    )
