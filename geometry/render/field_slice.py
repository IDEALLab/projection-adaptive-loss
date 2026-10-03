"""Pre-computed 2D scalar field on an axis-aligned plane.

A ``FieldSlice`` holds the result of an external simulation (CFD wind
field, stress field, temperature, etc.) evaluated on a regular 2D grid.
It can be overlaid on a PyVista 3D scene via ``CADProgram.visualize()``.

Unlike the interactive SDF/raster slice widgets (which re-evaluate a
function when the user drags the plane), a FieldSlice is **static**,
the data is pre-computed at a fixed plane and offset.

For volumetric (3D) fields that support interactive slicing, see
``VolumeField`` (planned).
"""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor


@dataclass
class FieldSlice:
    """A pre-computed 2D scalar field on an axis-aligned plane.

    Attributes:
        values: [H, W] or [B, H, W] scalar field data.
        extents: ((u_min, u_max), (v_min, v_max)) spatial bounds of the grid.
        plane: Axis-aligned plane ('xy', 'xz', or 'yz').
        offset: Position along the normal axis (e.g. z-value for 'xy').
        name: Label for the colorbar (e.g. 'wind speed [m/s]').
        cmap: Matplotlib/PyVista colormap name (default: 'viridis').
        clim: Optional (min, max) color range. If None, auto-scaled from data.
        opacity: Field overlay opacity in [0, 1] (default: 0.9).
    """

    values: Tensor
    extents: tuple[tuple[float, float], tuple[float, float]]
    plane: str
    offset: float = 0.0
    name: str = "field"
    cmap: str = "viridis"
    clim: tuple[float, float] | None = None
    opacity: float = 0.9
