"""
Core Shape Class

The Shape class wraps an SDF (Signed Distance Function) callable and provides
methods for mesh extraction and visualization. Supports batched parameters
for parallel evaluation of multiple shape variants.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from .mesh import MeshTimer


class Shape:
    """
    Wraps an SDF function that supports batched parameters.

    The SDF function signature is: sdf_fn(points: [B, N, 3]) -> [B, N]
    - __call__ auto-unsqueezes [N, 3] -> [1, N, 3] for convenience
    - Output is always [B, N] (B parameter batches × N points)

    Attributes:
        sdf_fn: The SDF function that computes signed distances
        batch_size: None for unbatched, int for batched parameters
        plane: For 2D shapes, the plane they lie in ('xy', 'xz', 'yz'). None for 3D shapes.
        is_neural: True if this shape uses a neural network SDF. Used to emit CSG warnings.
    """

    def __init__(
        self,
        sdf_fn: Callable[[Tensor], Tensor],
        batch_size: int | None = None,
        plane: str | None = None,
        is_neural: bool = False,
        workplane=None,
        device: str | torch.device | None = None,
    ):
        """
        Initialize a Shape with an SDF function.

        Args:
            sdf_fn: Function that takes points [B, N, 3] and returns distances [B, N]
            batch_size: None for unbatched, int for batched parameters
            plane: For 2D shapes, the plane they lie in ('xy', 'xz', 'yz'). None for 3D.
            is_neural: True if this shape uses a neural network SDF (default: False)
            workplane: Optional Workplane used to create this shape (e.g. from bezier_halfplane).
            device: Device where this shape's parameter tensors live. If set,
                    __call__ will move query points to this device before evaluation.
        """
        self.sdf_fn = sdf_fn
        self.batch_size = batch_size
        self.plane = plane
        self.is_neural = is_neural
        self.workplane = workplane
        self.device = torch.device(device) if device is not None else None

    def __call__(self, points: Tensor) -> Tensor:
        """
        Evaluate SDF at given points.

        Args:
            points: [N, 3] or [B, N, 3] tensor of 3D points to evaluate

        Returns:
            [B, N] tensor of signed distances
        """
        if points.dim() == 2:
            points = points.unsqueeze(0)  # [N, 3] -> [1, N, 3]
        if self.device is not None and points.device != self.device:
            points = points.to(self.device)
        return self.sdf_fn(points)

    def get_mesh(
        self,
        xyz_min: tuple[float, float, float],
        xyz_max: tuple[float, float, float],
        resolution: int = 64,
        backend: str | None = None,
        device: str | torch.device | None = None,
        timer: "MeshTimer | None" = None,
    ):
        """
        Extract mesh using specified backend.

        Args:
            xyz_min: Bounding box minimum corner
            xyz_max: Bounding box maximum corner
            resolution: Grid resolution (default 64)
            backend: 'skimage', 'diso-mc', or 'diso-dmc' (REQUIRED)
            device: Torch device for grid/SDF evaluation (default: shape's device or CPU)
            timer: Optional MeshTimer for phase profiling (sparse-dmc only)

        Returns:
            MeshResult with lists of per-batch tensors

        Raises:
            ValueError: If backend not specified
        """
        if backend is None:
            raise ValueError(
                "backend parameter is required. "
                "Use backend='skimage', 'diso-mc', or 'diso-dmc'."
            )

        # Resolve device: explicit arg > shape's device > None (let backend decide)
        mesh_device = device if device is not None else self.device

        from .mesh import sdf_to_mesh

        return sdf_to_mesh(
            self,
            xyz_min,
            xyz_max,
            resolution,
            backend=backend,
            batch_size=self.batch_size,
            device=mesh_device,
            timer=timer,
        )

    def show(
        self,
        xyz_min: tuple[float, float, float],
        xyz_max: tuple[float, float, float],
        resolution: int = 64,
        backend: str | None = None,
        origin: bool = True,
    ):
        """
        Extract mesh and display it using PyVista.

        This is a convenience method that combines mesh extraction and visualization.

        Args:
            xyz_min: Bounding box minimum corner (x, y, z)
            xyz_max: Bounding box maximum corner (x, y, z)
            resolution: Grid resolution (default 64)
            backend: 'skimage', 'diso-mc', or 'diso-dmc' (REQUIRED)
            origin: If True, show origin reference planes (default: True)

        Returns:
            PyVista plotter object
        """
        if backend is None:
            raise ValueError(
                "backend parameter is required. "
                "Use backend='skimage', 'diso-mc', or 'diso-dmc'."
            )

        from .render.pyvista import render_shape

        return render_shape(
            self, xyz_min, xyz_max, resolution, backend=backend, origin=origin
        )

    def __repr__(self) -> str:
        parts = []
        if self.batch_size is not None:
            parts.append(f"batch_size={self.batch_size}")
        if self.plane is not None:
            parts.append(f"plane='{self.plane}'")
        if self.is_neural:
            parts.append("neural=True")
        if self.device is not None:
            parts.append(f"device={self.device}")
        if parts:
            return f"<Shape ({', '.join(parts)})>"
        return "<Shape (unbatched)>"
