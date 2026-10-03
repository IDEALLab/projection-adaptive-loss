import torch
from torch import Tensor

from geometry.core import Shape
from geometry.loader import loadable_shape
from geometry.stdlib.operations.coordinates import PLANE_PERPENDICULAR_AXIS
from geometry.utils import get_batch_size, validate_tensor


@loadable_shape()
def mirror(shape: Shape, plane: str, offset: Tensor = None) -> Shape:
    """
    Reflect shape across a plane. Returns reflected copy only (not merged).

    To get symmetric merged shape, use: union(original, mirror(original, plane))

    Args:
        shape: Shape to reflect
        plane: Mirror plane - 'xy', 'xz', or 'yz'
        offset: [1, 1] tensor - offset of mirror plane from origin (default: 0)
                (Only scalar offsets supported)

    Returns:
        Reflected Shape (at mirrored position)

    Example:
        >>> s = sphere(
        ...     radius=torch.tensor([[1.]]),
        ...     center=torch.tensor([[2., 0., 0.]])
        ... )
        >>> reflected = mirror(s, plane='yz', offset=torch.tensor([[0.]]))  # Now at [-2, 0, 0]
        >>> symmetric = union(s, reflected)  # Two spheres
    """
    if plane not in VALID_MIRROR_PLANES:
        raise ValueError(
            f"mirror() got invalid plane '{plane}'. Must be one of: {VALID_MIRROR_PLANES}"
        )

    # Default offset to 0 (mirror through origin), on shape's device
    if offset is None:
        offset = torch.zeros(shape.batch_size, 1, device=shape.device)

    # Validate offset is a tensor
    offset_t = validate_tensor(offset, vec_size=1, name="offset")  # [B, 1]

    axis_idx = PLANE_PERPENDICULAR_AXIS[plane].index

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(shape, offset_t, names=["shape", "offset"])

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate mirrored shape: [B, N, 3] -> [B, N]"""
        p_reflected = p.expand(batch_size, -1, -1).clone()  # [B, N, 3]
        p_reflected[..., axis_idx] = 2 * offset_t - p_reflected[..., axis_idx]

        return shape(p_reflected)  # [B, N]

    # Mirror preserves plane: reflection doesn't change which plane a 2D shape is in
    return Shape(
        sdf_fn, batch_size=batch_size, plane=shape.plane, device=offset_t.device
    )


VALID_MIRROR_PLANES = ("xy", "xz", "yz")
