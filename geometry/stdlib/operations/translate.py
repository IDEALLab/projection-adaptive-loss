from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.coordinates import PLANE_PERPENDICULAR_AXIS
from geometry.utils import get_batch_size, validate_tensor


@loadable_shape()
def translate(shape: Shape, offset: Tensor) -> Shape:
    """
    Translate (move) a shape by an offset vector.

    SDF formula: sdf(p - offset)

    Args:
        shape: Shape to translate
        offset: [B, 3] tensor - translation vector - REQUIRED
                For 2D shapes, must be in-plane (perpendicular component = 0):
                - XY plane -> offset[:, 2] (Z) must be 0
                - XZ plane -> offset[:, 1] (Y) must be 0
                - YZ plane -> offset[:, 0] (X) must be 0

    Returns:
        Translated Shape

    Example:
        >>> s = sphere(
        ...     radius=torch.tensor([[1.0]]),
        ...     center=torch.tensor([[0., 0., 0.]])
        ... )
        >>> moved = translate(s, torch.tensor([[2., 0., 0.]]))  # Move 2 units along X
    """
    # Strict tensor validation
    offset_t = validate_tensor(offset, vec_size=3, name="offset")  # [B, 3]

    # Validate 2D shape translation (must be in-plane)
    _validate_2d_translation(shape, offset_t)

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(shape, offset_t, names=["shape", "offset"])

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate translated shape: [B, N, 3] -> [B, N]"""
        off_exp = offset_t.unsqueeze(1)  # [B, 1, 3]
        translated = p - off_exp  # [B, N, 3]

        return shape(translated)  # [B, N]

    # Preserve plane attribute for 2D shapes
    return Shape(
        sdf_fn, batch_size=batch_size, plane=shape.plane, device=offset_t.device
    )


def _validate_2d_translation(shape: "Shape", offset: Tensor) -> None:
    """
    Validate that a 2D shape is translated only within its plane.

    Args:
        shape: Shape to validate
        offset: [B, 3] tensor offset (must already be validated)

    Raises:
        ValueError: If the offset has a non-zero component perpendicular to the plane.
    """
    if shape.plane is None:
        return  # 3D shape, no validation needed

    perp_axis = PLANE_PERPENDICULAR_AXIS[shape.plane]

    # Check all batch items have zero perpendicular component
    perp_values = offset[:, perp_axis.index].abs()
    if perp_values.max() > 1e-6:
        raise ValueError(
            f"Cannot translate {shape.plane.upper()}-plane shape in {perp_axis.name.upper()} direction. "
            f"2D shapes can only be translated within their plane."
        )
