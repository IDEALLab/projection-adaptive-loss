"""
SDF Primitives - 2D and 3D Shapes

Strict tensor validation: ALL parameters must be [B, K] tensors.
No auto-conversion from Python types - use tensorify_shape_def() in YAML path.

2D Primitives:
- circle: 2D circle in specified plane
- rectangle: 2D rectangle with Euclidean distance (smooth gradients)
- rectangle_sharp: 2D rectangle with Chebyshev distance (sharp corners)

3D Primitives:
- sphere: 3D sphere
- box: 3D box with Euclidean distance (smooth gradients)
- box_sharp: 3D box with Chebyshev distance (sharp corners)

Bezier Primitives:
- bezier_halfplane: 3D half-space bounded by a 2D Bezier curve on a workplane
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from geometry.loader import loadable_shape

from ..core import Shape
from ..utils import get_batch_size, validate_tensor

if TYPE_CHECKING:
    from geometry.stdlib.curves import Curve2D

# Constants

# Plane coordinate mapping: (first_axis, second_axis, perpendicular_axis)
PLANE_AXES = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}

VALID_PLANES = tuple(PLANE_AXES.keys())
VALID_AXES = ("x", "y", "z")


# Validation Helpers


def _validate_plane(plane, func_name: str) -> None:
    """
    Validate that plane parameter is provided and valid.

    Args:
        plane: The plane parameter value
        func_name: Function name for error message

    Raises:
        ValueError: If plane is None or invalid
    """
    if plane is None:
        raise ValueError(
            f"{func_name}() missing required argument: 'plane'. "
            f"Must be one of: {VALID_PLANES}"
        )
    if plane not in VALID_PLANES:
        raise ValueError(
            f"{func_name}() got invalid plane '{plane}'. Must be one of: {VALID_PLANES}"
        )


def _validate_mode_2d(min, max, size, center, func_name: str) -> str:
    """
    Validate and determine the mode for 2D primitives (rectangle).

    Args:
        min: min corner parameter
        max: max corner parameter
        size: size parameter
        center: center parameter
        func_name: Function name for error messages

    Returns:
        'corner' or 'centered' indicating the mode

    Raises:
        ValueError: If invalid combination of parameters
    """
    has_min = min is not None
    has_max = max is not None
    has_size = size is not None
    has_center = center is not None

    # Check for conflicting modes
    corner_mode = has_min or has_max
    centered_mode = has_size or has_center

    if corner_mode and centered_mode:
        raise ValueError(
            f"{func_name}() got conflicting arguments: cannot specify both corner mode "
            f"(min, max) and centered mode (size, center). Use one or the other."
        )

    # Corner mode validation
    if has_min and not has_max:
        raise ValueError(
            f"{func_name}() corner mode requires both 'min' and 'max'. Got min but not max."
        )
    if has_max and not has_min:
        raise ValueError(
            f"{func_name}() corner mode requires both 'min' and 'max'. Got max but not min."
        )
    if has_min and has_max:
        return "corner"

    # Centered mode validation
    if has_center and not has_size:
        raise ValueError(
            f"{func_name}() centered mode requires 'size'. Got center but not size."
        )
    if has_size:
        return "centered"

    # Nothing provided
    raise ValueError(
        f"{func_name}() requires either (min, max) for corner mode or (size,) or "
        f"(size, center) for centered mode. Got: min={min}, max={max}, size={size}, center={center}"
    )


def _validate_mode_3d(min, max, size, center, func_name: str) -> str:
    """
    Validate and determine the mode for 3D primitives (box).
    Same logic as 2D but for 3D shapes.
    """
    return _validate_mode_2d(min, max, size, center, func_name)


def _get_2d_coords(points_3d: Tensor, plane: str) -> Tensor:
    """
    Extract 2D coordinates from 3D points based on plane.

    Args:
        points_3d: [N, 3] or [B, N, 3] tensor of 3D points
        plane: 'xy', 'xz', or 'yz'

    Returns:
        [N, 2] or [B, N, 2] tensor of 2D coordinates
    """
    ax1, ax2, _ = PLANE_AXES[plane]
    # Use stack of simple index selects instead of fancy indexing
    # to avoid implicit data copies on MPS/CUDA.
    if points_3d.dim() == 2:
        return torch.stack([points_3d[:, ax1], points_3d[:, ax2]], dim=-1)
    # [B, N, 3]
    return torch.stack([points_3d[:, :, ax1], points_3d[:, :, ax2]], dim=-1)


# 2D Primitives


@loadable_shape()
def circle(
    radius: Tensor,
    center: Tensor,
    plane: str,
) -> Shape:
    """
    Create a 2D circle SDF in the specified plane.

    The signed distance function for a circle is:
        d(p) = ||p_2d - center|| - radius

    where p_2d are the coordinates in the specified plane.

    Args:
        radius: [B, 1] tensor - circle radius - REQUIRED
        center: [B, 2] tensor - circle center in-plane coordinates - REQUIRED
                For plane='xz', center columns are [x_val, z_val]
        plane: Which plane the circle lies in - REQUIRED
               One of: 'xy', 'xz', 'yz'

    Returns:
        Shape object that evaluates the circle SDF

    Examples:
        >>> # Circle in XY plane
        >>> c = circle(
        ...     radius=torch.tensor([[1.0]]),
        ...     center=torch.tensor([[0., 0.]]),
        ...     plane='xy'
        ... )
        >>> distances = c(torch.rand(100, 3))  # [1, 100]

        >>> # 10 circles with different radii
        >>> radii = torch.linspace(0.5, 2.0, 10).unsqueeze(1)  # [10, 1]
        >>> centers = torch.zeros(10, 2)  # [10, 2]
        >>> c_batch = circle(radius=radii, center=centers, plane='xy')
        >>> distances = c_batch(torch.rand(100, 3))  # [10, 100]
    """
    _validate_plane(plane, "circle")

    # Strict tensor validation - no auto-conversion
    radius_t = validate_tensor(radius, vec_size=1, name="radius")  # [B, 1]
    center_t = validate_tensor(center, vec_size=2, name="center")  # [B, 2]

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(radius_t, center_t, names=["radius", "center"])

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate circle SDF: [B, N, 3] -> [B, N]"""
        # Extract 2D coordinates from 3D points
        p_2d = _get_2d_coords(p, plane)  # [B, N, 2]

        c_exp = center_t.unsqueeze(1)  # [B, 1, 2]
        r_exp = radius_t.unsqueeze(1)  # [B, 1, 1]

        # Output is ALWAYS [B, N], no squeezing
        dist = torch.norm(p_2d - c_exp, dim=-1) - r_exp.squeeze(-1)
        return dist  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, plane=plane, device=radius_t.device)


def _rectangle_impl(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
    plane: str,
    exact: bool,
) -> Shape:
    """
    Shared implementation for rectangle and rectangle_sharp.

    Args:
        min: Corner mode - [B, 2] tensor minimum corner
        max: Corner mode - [B, 2] tensor maximum corner
        size: Centered mode - [B, 2] tensor size
        center: Centered mode - [B, 2] tensor center
        plane: Which plane ('xy', 'xz', 'yz') - REQUIRED
        exact: If True, use Euclidean distance; if False, use Chebyshev

    Returns:
        Shape object
    """
    func_name = "rectangle" if exact else "rectangle_sharp"
    _validate_plane(plane, func_name)
    mode = _validate_mode_2d(min, max, size, center, func_name)

    # Convert to centered representation internally with strict validation
    if mode == "corner":
        min_t = validate_tensor(min, vec_size=2, name="min")  # [B, 2]
        max_t = validate_tensor(max, vec_size=2, name="max")  # [B, 2]
        # Validate batch sizes match
        get_batch_size(min_t, max_t, names=["min", "max"])
        size_t = max_t - min_t
        center_t = (min_t + max_t) / 2
    else:
        size_t = validate_tensor(size, vec_size=2, name="size")  # [B, 2]
        if center is not None:
            center_t = validate_tensor(center, vec_size=2, name="center")  # [B, 2]
        else:
            # Default center to origin - match size batch size
            center_t = torch.zeros(size_t.shape[0], 2, device=size_t.device)

    batch_size = get_batch_size(size_t, center_t, names=["size", "center"])
    half_size_t = size_t / 2

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate rectangle SDF: [B, N, 3] -> [B, N]"""
        # Extract 2D coordinates
        p_2d = _get_2d_coords(p, plane)  # [B, N, 2]

        c_exp = center_t.unsqueeze(1)  # [B, 1, 2]
        half_exp = half_size_t.unsqueeze(1)  # [B, 1, 2]

        q = torch.abs(p_2d - c_exp) - half_exp  # [B, N, 2]

        if exact:
            # Euclidean: smooth distance
            interior = torch.clamp(q, max=0).max(dim=-1).values  # [B, N]
            exterior = torch.norm(torch.clamp(q, min=0), dim=-1)  # [B, N]
            return interior + exterior
        # Chebyshev: max of axis distances
        return q.max(dim=-1).values  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, plane=plane, device=half_size_t.device)


@loadable_shape()
def rectangle(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
    plane: str,
) -> Shape:
    """
    Create a 2D rectangle SDF with Euclidean distance (smooth gradients).

    Supports two parameter modes:
    - Corner mode: provide min and max corners
    - Centered mode: provide size and optionally center (defaults to origin)

    The Euclidean distance gives smooth gradients at corners, making it
    suitable for optimization and smooth blending operations.

    Args:
        min: Corner mode - [B, 2] tensor minimum corner
        max: Corner mode - [B, 2] tensor maximum corner
        size: Centered mode - [B, 2] tensor rectangle size
        center: Centered mode - [B, 2] tensor center point (defaults to origin)
        plane: Which plane the rectangle lies in - REQUIRED
               One of: 'xy', 'xz', 'yz'
               Center coordinates follow plane order: plane='xz' -> center=[x, z]

    Returns:
        Shape object that evaluates the rectangle SDF

    Examples:
        >>> # Corner mode
        >>> r = rectangle(
        ...     min=torch.tensor([[0., 0.]]),
        ...     max=torch.tensor([[2., 3.]]),
        ...     plane='xy'
        ... )

        >>> # Centered mode
        >>> r = rectangle(
        ...     size=torch.tensor([[2., 3.]]),
        ...     center=torch.tensor([[1., 1.5]]),
        ...     plane='xy'
        ... )

        >>> # Centered at origin
        >>> r = rectangle(size=torch.tensor([[2., 3.]]), plane='xz')
    """
    return _rectangle_impl(
        min=min, max=max, size=size, center=center, plane=plane, exact=True
    )


@loadable_shape()
def rectangle_sharp(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
    plane: str,
) -> Shape:
    """
    Create a 2D rectangle SDF with Chebyshev distance (sharp corners).

    Same parameters as rectangle(), but uses Chebyshev (L-infinity) distance
    instead of Euclidean. This preserves sharp corners when the shape is
    offset or used in certain operations.

    See rectangle() for full parameter documentation.
    """
    return _rectangle_impl(
        min=min, max=max, size=size, center=center, plane=plane, exact=False
    )


# 3D Primitives


@loadable_shape()
def sphere(
    radius: Tensor,
    center: Tensor,
) -> Shape:
    """
    Create a sphere SDF.

    The signed distance function for a sphere is:
        d(p) = ||p - center|| - radius

    Args:
        radius: [B, 1] tensor - sphere radius - REQUIRED
        center: [B, 3] tensor - sphere center - REQUIRED

    Returns:
        Shape object that evaluates the sphere SDF

    Examples:
        >>> # Single sphere
        >>> s = sphere(
        ...     radius=torch.tensor([[1.0]]),
        ...     center=torch.tensor([[0., 0., 0.]])
        ... )
        >>> distances = s(torch.rand(100, 3))  # [1, 100]

        >>> # 10 spheres with different radii
        >>> radii = torch.linspace(0.5, 2.0, 10).unsqueeze(1)  # [10, 1]
        >>> centers = torch.zeros(10, 3)  # [10, 3]
        >>> s_batch = sphere(radius=radii, center=centers)
        >>> distances = s_batch(torch.rand(100, 3))  # [10, 100]
    """
    # Strict tensor validation - no auto-conversion
    radius_t = validate_tensor(radius, vec_size=1, name="radius")  # [B, 1]
    center_t = validate_tensor(center, vec_size=3, name="center")  # [B, 3]

    # Get batch size (strict - all must match)
    batch_size = get_batch_size(radius_t, center_t, names=["radius", "center"])

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate sphere SDF: [B, N, 3] -> [B, N]"""
        c_exp = center_t.unsqueeze(1)  # [B, 1, 3]
        r_exp = radius_t.unsqueeze(1)  # [B, 1, 1]

        # Output is ALWAYS [B, N], no squeezing
        dist = torch.norm(p - c_exp, dim=-1) - r_exp.squeeze(-1)
        return dist  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, device=radius_t.device)


def _box_impl(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
    exact: bool,
) -> Shape:
    """
    Shared implementation for box and box_sharp.

    Args:
        min: Corner mode - [B, 3] tensor minimum corner
        max: Corner mode - [B, 3] tensor maximum corner
        size: Centered mode - [B, 3] tensor size
        center: Centered mode - [B, 3] tensor center
        exact: If True, use Euclidean distance; if False, use Chebyshev

    Returns:
        Shape object
    """
    func_name = "box" if exact else "box_sharp"
    mode = _validate_mode_3d(min, max, size, center, func_name)

    # Convert to centered representation internally with strict validation
    if mode == "corner":
        min_t = validate_tensor(min, vec_size=3, name="min")  # [B, 3]
        max_t = validate_tensor(max, vec_size=3, name="max")  # [B, 3]
        # Validate batch sizes match
        get_batch_size(min_t, max_t, names=["min", "max"])
        size_t = max_t - min_t
        center_t = (min_t + max_t) / 2
    else:
        size_t = validate_tensor(size, vec_size=3, name="size")  # [B, 3]
        if center is not None:
            center_t = validate_tensor(center, vec_size=3, name="center")  # [B, 3]
        else:
            # Default center to origin - match size batch size
            center_t = torch.zeros(size_t.shape[0], 3, device=size_t.device)

    batch_size = get_batch_size(size_t, center_t, names=["size", "center"])
    half_size_t = size_t / 2

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate box SDF: [B, N, 3] -> [B, N]"""
        c_exp = center_t.unsqueeze(1)  # [B, 1, 3]
        half_exp = half_size_t.unsqueeze(1)  # [B, 1, 3]

        q = torch.abs(p - c_exp) - half_exp  # [B, N, 3]

        if exact:
            # Euclidean: smooth distance
            interior = torch.clamp(q, max=0).max(dim=-1).values  # [B, N]
            exterior = torch.norm(torch.clamp(q, min=0), dim=-1)  # [B, N]
            return interior + exterior
        # Chebyshev: max of axis distances
        return q.max(dim=-1).values  # [B, N]

    return Shape(sdf_fn, batch_size=batch_size, device=half_size_t.device)


@loadable_shape()
def box(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
) -> Shape:
    """
    Create a 3D axis-aligned box SDF with Euclidean distance (smooth gradients).

    Supports two parameter modes:
    - Corner mode: provide min and max corners
    - Centered mode: provide size and optionally center (defaults to origin)

    The Euclidean distance gives smooth gradients at corners and edges,
    making it suitable for optimization and smooth blending operations.

    The formula is:
        q = abs(p - center) - half_size
        d(p) = norm(max(q, 0)) + min(max(q.x, q.y, q.z), 0)

    Args:
        min: Corner mode - [B, 3] tensor minimum corner
        max: Corner mode - [B, 3] tensor maximum corner
        size: Centered mode - [B, 3] tensor box size (width, height, depth)
        center: Centered mode - [B, 3] tensor center point (defaults to origin)

    Returns:
        Shape object that evaluates the box SDF

    Examples:
        >>> # Corner mode
        >>> b = box(
        ...     min=torch.tensor([[0., 0., 0.]]),
        ...     max=torch.tensor([[2., 3., 4.]])
        ... )

        >>> # Centered mode
        >>> b = box(
        ...     size=torch.tensor([[2., 3., 4.]]),
        ...     center=torch.tensor([[1., 1.5, 2.]])
        ... )

        >>> # Centered at origin
        >>> b = box(size=torch.tensor([[2., 2., 2.]]))

        >>> # Batched boxes
        >>> sizes = torch.tensor([[1., 1., 1.], [2., 2., 2.], [3., 3., 3.]])
        >>> centers = torch.zeros(3, 3)
        >>> b_batch = box(size=sizes, center=centers)
    """
    return _box_impl(min=min, max=max, size=size, center=center, exact=True)


@loadable_shape()
def box_sharp(
    *,
    min: Tensor | None = None,
    max: Tensor | None = None,
    size: Tensor | None = None,
    center: Tensor | None = None,
) -> Shape:
    """
    Create a 3D axis-aligned box SDF with Chebyshev distance (sharp corners).

    Same parameters as box(), but uses Chebyshev (L-infinity) distance
    instead of Euclidean. This preserves sharp corners and edges when
    the shape is offset or used in certain operations.

    See box() for full parameter documentation.
    """
    return _box_impl(min=min, max=max, size=size, center=center, exact=False)


# Formula Primitives


@loadable_shape(skip_validation=True)
def formula3d(expression: str, **params) -> Shape:
    """
    Create 3D shape from mathematical expression.

    The expression defines a signed distance function where:
    - Negative values = inside the shape
    - Zero = on the surface
    - Positive values = outside the shape

    Variables available: x, y, z (3D coordinates)
    Parameters referenced with $param_name syntax

    Args:
        expression: Mathematical expression string.
                   Example: "sqrt(x*x + y*y + z*z) - $radius"
        **params: Parameters as [B, 1] tensors

    Returns:
        Shape object with batch_size and plane=None

    Supported operations:
        - Arithmetic: +, -, *, / (unary -)
        - Functions: sqrt()
        - Variables: x, y, z
        - Parameters: $param_name

    Examples:
        >>> # Custom sphere
        >>> s = formula3d(
        ...     expression="sqrt(x*x + y*y + z*z) - $r",
        ...     r=torch.tensor([[1.0]])
        ... )
        >>>
        >>> # Batched parameters
        >>> radii = torch.linspace(0.5, 2.0, 10).unsqueeze(1)  # [10, 1]
        >>> s_batch = formula3d(expression="sqrt(x*x + y*y + z*z) - $r", r=radii)
        >>> s_batch.batch_size  # 10

    Note:
        Parameters are fixed at Shape creation time. To evaluate with
        different parameters, create a new Shape or use CADProgram.with_params().

    Raises:
        ValueError: If expression is invalid or uses unsupported operations
        TypeError: If parameters are not tensors
    """
    from ..expression import parse_expression

    # Validate params are tensors with [B, 1] format
    params_t = {}
    for name, value in params.items():
        params_t[name] = validate_tensor(value, vec_size=1, name=f"param '{name}'")

    # Determine batch size (all params must have same batch size)
    if params_t:
        batch_size = get_batch_size(*params_t.values(), names=list(params_t.keys()))
    else:
        # No parameters - unbatched
        batch_size = 1

    # Parse expression -> returns sdf_fn(points, params_dict)
    sdf_fn_parametric = parse_expression(
        expression=expression, params=params_t, allowed_vars=["x", "y", "z"]
    )

    # Infer device from parameter tensors (if any)
    _device = next(iter(params_t.values())).device if params_t else None

    # Close over params to match Shape's interface: sdf_fn(points) -> [B, N]
    def sdf_fn(p: Tensor) -> Tensor:
        """[N, 3] -> [B, N]"""
        return sdf_fn_parametric(p, params_t)

    return Shape(sdf_fn, batch_size=batch_size, plane=None, device=_device)


@loadable_shape(skip_validation=True)
def formula2d(expression: str, plane: str, **params) -> Shape:
    """
    Create 2D shape from mathematical expression.

    Similar to formula3d but:
    - Requires 'plane' parameter (xy, xz, or yz)
    - Expression must only use in-plane variables
    - Out-of-plane variable usage raises error

    Args:
        expression: Mathematical expression string.
                   Example: "sqrt(x*x + y*y) - $radius"
        plane: Which plane the 2D shape lies in ('xy', 'xz', or 'yz')
        **params: Parameters as [B, 1] tensors

    Returns:
        Shape object with batch_size and plane=plane

    Examples:
        >>> # Custom circle in XY plane
        >>> c = formula2d(
        ...     expression="sqrt(x*x + y*y) - $r",
        ...     plane='xy',
        ...     r=torch.tensor([[1.0]])
        ... )
        >>>
        >>> # This raises error (z not allowed in XY plane)
        >>> bad = formula2d(
        ...     expression="sqrt(x*x + y*y + z*z) - $r",
        ...     plane='xy',
        ...     r=torch.tensor([[1.0]])
        ... )
        ... # ValueError: Variable 'z' not allowed. Allowed variables: ['x', 'y']

    Note:
        Parameters are fixed at Shape creation time. To evaluate with
        different parameters, create a new Shape or use CADProgram.with_params().

    Raises:
        ValueError: If plane invalid or expression uses out-of-plane variables
        TypeError: If parameters are not tensors
    """
    from ..expression import parse_expression

    # Validate plane
    _validate_plane(plane, "formula2d")

    # Determine which variables are allowed based on plane
    plane_vars = {
        "xy": ["x", "y"],
        "xz": ["x", "z"],
        "yz": ["y", "z"],
    }
    allowed_vars = plane_vars[plane]

    # Validate params are tensors with [B, 1] format
    params_t = {}
    for name, value in params.items():
        params_t[name] = validate_tensor(value, vec_size=1, name=f"param '{name}'")

    # Determine batch size
    if params_t:
        batch_size = get_batch_size(*params_t.values(), names=list(params_t.keys()))
    else:
        batch_size = 1

    # Parse expression (validates plane constraints automatically)
    sdf_fn_parametric = parse_expression(
        expression=expression,
        params=params_t,
        allowed_vars=allowed_vars,  # Only in-plane variables allowed
    )

    # Infer device from parameter tensors (if any)
    _device = next(iter(params_t.values())).device if params_t else None

    # Close over params
    def sdf_fn(p: Tensor) -> Tensor:
        """[N, 3] -> [B, N]"""
        return sdf_fn_parametric(p, params_t)

    return Shape(sdf_fn, batch_size=batch_size, plane=plane, device=_device)


# Bezier Halfplane Helpers


def _eval_bezier_at(cp: Tensor, t: Tensor) -> Tensor:
    """
    Evaluate cubic Bezier at per-query-point t values using Bernstein basis.

    Unlike CubicBezier2D.sample() which takes shared (S,) t values,
    this accepts arbitrary-shape t for per-point closest-t computations.

    Args:
        cp: [B, 4, 2] control points
        t: [B, N, K] or [B, N] parameter values

    Returns:
        [B, N, K, 2] or [B, N, 2] points on curve (same leading dims as t, plus 2)
    """
    s = 1.0 - t

    # Expand t and s for broadcasting with 2D coords: (..., 1)
    t_ = t.unsqueeze(-1)  # (..., 1)
    s_ = s.unsqueeze(-1)  # (..., 1)

    # Reshape control points: (B, 2) -> broadcast with t shape
    # For t of shape (B, N, K), we need cp as (B, 1, 1, 2) per point
    n_extra = t.dim() - 1  # number of dims after B
    P0 = cp[:, 0, :]  # (B, 2)
    P1 = cp[:, 1, :]  # (B, 2)
    P2 = cp[:, 2, :]  # (B, 2)
    P3 = cp[:, 3, :]  # (B, 2)
    for _ in range(n_extra):
        P0 = P0.unsqueeze(1)
        P1 = P1.unsqueeze(1)
        P2 = P2.unsqueeze(1)
        P3 = P3.unsqueeze(1)

    result = (s_**3) * P0 + 3 * (s_**2) * t_ * P1 + 3 * s_ * (t_**2) * P2 + (t_**3) * P3
    return result


def _eval_bezier_deriv_at(cp: Tensor, t: Tensor) -> Tensor:
    """
    Evaluate cubic Bezier derivative (tangent) at per-query-point t values.

    C'(t) = 3[(1-t)^2 (P1-P0) + 2(1-t)t (P2-P1) + t^2 (P3-P2)]

    Args:
        cp: [B, 4, 2] control points
        t: [B, N] parameter values

    Returns:
        [B, N, 2] tangent vectors
    """
    s = 1.0 - t

    t_ = t.unsqueeze(-1)  # (B, N, 1)
    s_ = s.unsqueeze(-1)  # (B, N, 1)

    D01 = (cp[:, 1, :] - cp[:, 0, :]).unsqueeze(1)  # (B, 1, 2)
    D12 = (cp[:, 2, :] - cp[:, 1, :]).unsqueeze(1)  # (B, 1, 2)
    D23 = (cp[:, 3, :] - cp[:, 2, :]).unsqueeze(1)  # (B, 1, 2)

    return 3 * ((s_**2) * D01 + 2 * s_ * t_ * D12 + (t_**2) * D23)  # (B, N, 2)


def _closest_t(
    query_2d: Tensor, cp: Tensor, n_coarse: int = 16, n_fine: int = 8
) -> Tensor:
    """
    Find closest parameter t on cubic Bezier for each query point.

    Two-pass algorithm:
    1. Coarse pass (no_grad): n_coarse uniform samples, hard argmin to find basin
    2. Fine pass (differentiable): n_fine local samples in basin,
       constant-temperature softmin (tau=0.01) -> t*

    Args:
        query_2d: [B, N, 2] query points in 2D workplane coordinates
        cp: [B, 4, 2] control points
        n_coarse: Number of coarse samples (default: 16)
        n_fine: Number of fine samples (default: 8)

    Returns:
        [B, N] closest parameter t for each query point
    """
    B, N, _ = query_2d.shape
    device = query_2d.device

    # Pass 1: Coarse (no_grad)
    with torch.no_grad():
        t_coarse = torch.linspace(0.0, 1.0, n_coarse, device=device)  # (n_coarse,)
        # Expand to (B, N, n_coarse) for _eval_bezier_at
        t_expanded = t_coarse.unsqueeze(0).unsqueeze(0).expand(B, N, n_coarse)
        pts_coarse = _eval_bezier_at(cp, t_expanded)  # (B, N, n_coarse, 2)

        # Distance from each query to each coarse sample
        q_exp = query_2d.unsqueeze(2)  # (B, N, 1, 2)
        dist_sq = ((pts_coarse - q_exp) ** 2).sum(dim=-1)  # (B, N, n_coarse)

        # Hard argmin to find basin
        best_idx = dist_sq.argmin(dim=-1)  # (B, N)
        t_best = t_coarse[best_idx]  # (B, N)

    # Pass 2: Fine (differentiable)
    dt = 1.0 / (n_coarse - 1)
    t_lo = t_best - dt  # (B, N), no clamp, allow extrapolation
    t_hi = t_best + dt  # (B, N)

    # n_fine samples in the window for each query point
    alpha = torch.linspace(0.0, 1.0, n_fine, device=device)  # (n_fine,)
    alpha = alpha.reshape(1, 1, n_fine)
    t_lo_exp = t_lo.unsqueeze(-1)  # (B, N, 1)
    t_hi_exp = t_hi.unsqueeze(-1)  # (B, N, 1)
    t_fine = t_lo_exp + alpha * (t_hi_exp - t_lo_exp)  # (B, N, n_fine)

    # Evaluate curve at fine samples
    pts_fine = _eval_bezier_at(cp, t_fine)  # (B, N, n_fine, 2)

    # Distances
    q_exp = query_2d.unsqueeze(2)  # (B, N, 1, 2)
    dist_sq_fine = ((pts_fine - q_exp) ** 2).sum(dim=-1)  # (B, N, n_fine)

    # Constant temperature softmin (testing: was adaptive)
    tau = torch.tensor(0.01, device=device).unsqueeze(0).unsqueeze(0)  # (1, 1)

    weights = torch.softmax(-dist_sq_fine / tau, dim=-1)  # (B, N, n_fine)
    t_star = (weights * t_fine).sum(dim=-1)  # (B, N)
    t_star = t_star.clamp(0.0, 1.0)  # clamp after softmin to stay on curve

    return t_star


# Bezier Halfplane Primitive


@loadable_shape(deserialize_args={"end_caps": str, "flip": bool}, skip_validation=True)
def bezier_halfplane(curve: "Curve2D", end_caps: str = "square", flip: bool = False):
    """
    Create a 3D SDF half-space bounded by a 2D Bezier curve on a workplane.

    The curve divides 3D space into two halves:
    - LEFT of curve direction (negative cross product) = inside (negative SDF)
    - RIGHT of curve direction = outside (positive SDF)

    The half-plane extends infinitely along the workplane normal.
    Beyond the curve endpoints (P0, P3), the boundary continues as
    straight G1 tangent rays (same direction as the curve tangent at
    each endpoint).

    Signed distance is computed as the cross product of the direction
    vector with the normalized tangent, both on the curve and on the
    tangent ray extensions.

    The workplane is read from curve.workplane (set at curve creation time).

    Args:
        curve: CubicBezier2D with [B, 4, 2] control points and .workplane set
        end_caps: Cap style at curve endpoints. Currently only "square" supported.
            "square" uses G1 tangent ray extensions at endpoints.
        flip: If True, invert the sign convention (RIGHT = inside)

    Returns:
        Shape object with sdf_fn: [N, 3] -> [B, N]

    Raises:
        ValueError: If end_caps is not "square" or curve has no workplane
    """
    if end_caps != "square":
        raise ValueError(
            f"bezier_halfplane end_caps='{end_caps}' not supported. "
            f"Only 'square' is currently implemented."
        )

    workplane = curve.workplane
    if workplane is None:
        raise ValueError(
            "bezier_halfplane requires a curve with a workplane set. "
            "Create curve with: CubicBezier2D(cp, workplane='xz') or "
            "CubicBezier2D(cp, workplane=Workplane.from_base('xz'))"
        )

    bs = curve.batch_size
    cp = curve.cp  # (B, 4, 2)

    def sdf_fn(p: Tensor) -> Tensor:
        """Evaluate bezier_halfplane SDF: [N, 3] -> [B, N]"""
        # 1. Project 3D points onto workplane -> (B, N, 2)
        query_2d = workplane.project(p)  # (B, N, 2)

        B = cp.shape[0]

        # 2. Find closest t* on curve
        t_star = _closest_t(query_2d, cp)  # (B, N)

        # 3. Evaluate curve position + tangent at t*
        closest_pt = _eval_bezier_at(cp, t_star)  # (B, N, 2)
        tangent = _eval_bezier_deriv_at(cp, t_star)  # (B, N, 2)

        # 4. Signed distance via cross product with normalized tangent
        tangent_norm = tangent / (tangent.norm(dim=-1, keepdim=True) + 1e-12)
        direction = query_2d - closest_pt  # (B, N, 2)
        # cross(direction, tangent_hat) = signed distance to local tangent line
        f_curve = (
            direction[..., 0] * tangent_norm[..., 1]
            - direction[..., 1] * tangent_norm[..., 0]
        )  # (B, N)

        if flip:
            f_curve = -f_curve

        # 6. G1 tangent ray extensions at endpoints
        # Beyond each endpoint, replace the curve distance with the
        # signed distance to the tangent ray (straight line continuation).
        # This gives G1 continuity at the junction.

        P0 = cp[:, 0, :].unsqueeze(1)  # (B, 1, 2)
        tang_0 = _eval_bezier_deriv_at(
            cp, torch.zeros(B, 1, device=cp.device)
        )  # (B, 1, 2)
        tang_0_norm = tang_0 / (tang_0.norm(dim=-1, keepdim=True) + 1e-12)

        P3 = cp[:, 3, :].unsqueeze(1)  # (B, 1, 2)
        tang_1 = _eval_bezier_deriv_at(
            cp, torch.ones(B, 1, device=cp.device)
        )  # (B, 1, 2)
        tang_1_norm = tang_1 / (tang_1.norm(dim=-1, keepdim=True) + 1e-12)

        # Signed distance to tangent ray at each endpoint:
        # cross(tangent, point - endpoint) gives signed distance to the line
        # (positive = right of tangent direction, negative = left)
        d0 = query_2d - P0  # (B, N, 2)
        ray_cross_0 = (
            d0[..., 0] * tang_0_norm[..., 1] - d0[..., 1] * tang_0_norm[..., 0]
        )  # (B, N)

        d3 = query_2d - P3  # (B, N, 2)
        ray_cross_3 = (
            d3[..., 0] * tang_1_norm[..., 1] - d3[..., 1] * tang_1_norm[..., 0]
        )  # (B, N)

        # Cross product with unit tangent IS the signed distance to the line
        f_ray_0 = ray_cross_0
        f_ray_3 = ray_cross_3

        if flip:
            f_ray_0 = -f_ray_0
            f_ray_3 = -f_ray_3

        # Behind P0 along tangent = use ray_0 instead of curve
        behind_start = ((query_2d - P0) * tang_0_norm).sum(dim=-1) < 0  # (B, N)
        # Past P3 along tangent = use ray_3 instead of curve
        past_end = ((query_2d - P3) * tang_1_norm).sum(dim=-1) > 0  # (B, N)

        f = torch.where(behind_start, f_ray_0, f_curve)
        f = torch.where(past_end, f_ray_3, f)

        return f

    return Shape(
        sdf_fn, batch_size=bs, plane=None, workplane=workplane, device=cp.device
    )
