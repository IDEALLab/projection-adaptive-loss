"""
Mesh rendering and visualization.

Submodules:
- pyvista: Interactive PyVista-based 3D viewer and image saving
- box_detect: Fast-path detection for box-only configs (skip SDF meshing)
"""

# Re-export PyVista functions for backwards compatibility
from .box_detect import BoxSpec, extract_boxes
from .field_slice import FieldSlice
from .pyvista import (
    DEFAULT_COLORS,
    VIEW_CAMERAS,
    _add_bounding_box,
    _add_curves_to_plotter,
    _add_field_slice,
    _add_origin_reference,
    _add_raster_slice_widget,
    _add_sdf_slice_widget,
    _add_workplanes_to_plotter,
    _mesh_result_to_pv,
    render_boxes,
    render_mesh,
    render_mesh_result,
    render_meshes,
    render_shape,
    save_mesh_image_pyvista,
)

__all__ = [
    "BoxSpec",
    "DEFAULT_COLORS",
    "FieldSlice",
    "VIEW_CAMERAS",
    "extract_boxes",
    "render_boxes",
    "_add_bounding_box",
    "_add_curves_to_plotter",
    "_add_field_slice",
    "_add_origin_reference",
    "_add_raster_slice_widget",
    "_add_sdf_slice_widget",
    "_add_workplanes_to_plotter",
    "_mesh_result_to_pv",
    "render_mesh",
    "render_mesh_result",
    "render_meshes",
    "render_shape",
    "save_mesh_image",
]


def save_mesh_image(
    mesh_result,
    path: str,
    view: str = "iso",
    format: str = "png",
    color: str = "lightgrey",
    colors: dict[str, str] | None = None,
    opacities: dict[str, float] | None = None,
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    window_size: tuple[int, int] = (800, 600),
) -> None:
    """
    Save mesh visualization to image file (PyVista backend).

    Args:
        mesh_result: MeshResult object, or dict[str, MeshResult] for multi-body
        path: Output file path (extension added if not present)
        view: Camera view ('iso', 'front', 'back', 'left', 'right', 'top', 'bottom')
        format: Output format ('png', 'jpg', 'pdf')
        color: Mesh color for single-body (default: 'lightgrey')
        colors: Color dict for multi-body (default: auto-assign from palette)
        opacities: Opacity dict for multi-body (default: 1.0 for all). Values in [0, 1].
        show_edges: Show wireframe edges (default: False)
        origin: Show origin reference planes (default: True)
        show_bounds: Show bounding box wireframe (default: True)
        window_size: Image dimensions (width, height) in pixels

    Example:
        >>> mesh = prog.mesh(backend='skimage')
        >>> save_mesh_image(mesh, 'output.png', view='iso')
        >>> # Multi-body:
        >>> meshes = prog.mesh(backend='skimage')  # dict
        >>> save_mesh_image(meshes, 'assembly.png', colors={'wing': 'steelblue'})
    """
    save_mesh_image_pyvista(
        mesh_result,
        path,
        view=view,
        format=format,
        color=color,
        colors=colors,
        opacities=opacities,
        show_edges=show_edges,
        origin=origin,
        show_bounds=show_bounds,
        window_size=window_size,
    )
