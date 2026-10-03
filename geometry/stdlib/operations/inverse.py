from torch import Tensor

from geometry.core import Shape

# inverse has no YAML dispatch.


def inverse(shape: Shape) -> Shape:
    """
    Invert a shape by negating its SDF (swap inside/outside).

    This flips the sign of the signed distance function, turning
    the interior into exterior and vice versa.

    Args:
        shape: Shape to invert

    Returns:
        Inverted Shape where inside becomes outside and vice versa

    Example:
        >>> s = sphere(radius=torch.tensor([[1.0]]), center=torch.tensor([[0., 0., 0.]]))
        >>> s_inv = inverse(s)
        >>> # Points inside sphere now have positive distance
    """

    def sdf_fn(p: Tensor) -> Tensor:
        return -shape(p)

    # Inverse preserves plane attribute
    return Shape(
        sdf_fn, batch_size=shape.batch_size, plane=shape.plane, device=shape.device
    )
