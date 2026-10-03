"""
Mesh Visualization using PyVista

Provides functions to render triangle meshes in an interactive 3D viewer.
Supports both single-mesh and multi-mesh rendering.
"""

from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..core import Shape
    from .box_detect import BoxSpec


# Default color palette for multi-body rendering
DEFAULT_COLORS = [
    "lightgrey",
    "lightcoral",
    "lightgreen",
    "lightyellow",
    "lightpink",
    "lightsalmon",
    "lightcyan",
    "plum",
]


def _add_origin_reference(
    plotter,
    bounds: tuple[float, float, float, float, float, float],
    plane_opacity: float = 0.15,
    plane_scale: float = 0.25,
    edge_width: float = 4,
) -> None:
    """
    Add origin axes and reference planes to the plotter.

    Args:
        plotter: PyVista plotter object
        bounds: Tuple of (xmin, xmax, ymin, ymax, zmin, zmax)
        plane_opacity: Opacity for reference planes (default: 0.15)
        plane_scale: Scale factor for plane size relative to bounds (default: 0.25)
        edge_width: Line width for plane edges (default: 4)
    """
    import pyvista as pv

    # Calculate plane size from bounds
    x_range = bounds[1] - bounds[0]
    y_range = bounds[3] - bounds[2]
    z_range = bounds[5] - bounds[4]
    plane_size = max(x_range, y_range, z_range) * plane_scale

    # Create XY plane (Z=0, blue)
    xy_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 0, 1),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        xy_plane,
        color="blue",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="blue",
        line_width=edge_width,
    )

    # Create XZ plane (Y=0, green)
    xz_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(0, 1, 0),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        xz_plane,
        color="green",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="green",
        line_width=edge_width,
    )

    # Create YZ plane (X=0, red)
    yz_plane = pv.Plane(
        center=(0, 0, 0),
        direction=(1, 0, 0),
        i_size=plane_size,
        j_size=plane_size,
        i_resolution=1,
        j_resolution=1,
    )
    plotter.add_mesh(
        yz_plane,
        color="red",
        opacity=plane_opacity,
        show_edges=True,
        edge_color="red",
        line_width=edge_width,
    )


def _add_bounding_box(
    plotter,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    color: str = "black",
    line_width: float = 0.5,
    opacity: float = 1.0,
) -> None:
    """
    Add wireframe bounding box to plotter.

    Draws 12 edges of the box defined by xyz_min and xyz_max.

    Args:
        plotter: PyVista plotter object
        xyz_min: Minimum corner of bounding box (x, y, z)
        xyz_max: Maximum corner of bounding box (x, y, z)
        color: Line color (default: 'black')
        line_width: Line width in pixels (default: 0.5)
        opacity: Line opacity (default: 1.0)
    """
    import pyvista as pv

    # 8 corners of the box
    corners = [
        [xyz_min[0], xyz_min[1], xyz_min[2]],  # 0
        [xyz_max[0], xyz_min[1], xyz_min[2]],  # 1
        [xyz_max[0], xyz_max[1], xyz_min[2]],  # 2
        [xyz_min[0], xyz_max[1], xyz_min[2]],  # 3
        [xyz_min[0], xyz_min[1], xyz_max[2]],  # 4
        [xyz_max[0], xyz_min[1], xyz_max[2]],  # 5
        [xyz_max[0], xyz_max[1], xyz_max[2]],  # 6
        [xyz_min[0], xyz_max[1], xyz_max[2]],  # 7
    ]

    # 12 edges (pairs of corner indices)
    edges = [
        # Bottom face (z=min)
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        # Top face (z=max)
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        # Vertical edges
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]

    # Create line segments for each edge
    for i, j in edges:
        line = pv.Line(corners[i], corners[j])
        plotter.add_mesh(
            line,
            color=color,
            line_width=line_width,
            opacity=opacity,
        )


def render_mesh(
    verts: Tensor,
    faces: Tensor,
    show: bool = True,
    color: str = "lightgrey",
    show_edges: bool = False,
    origin: bool = True,
):
    """
    Render a triangle mesh using PyVista.

    Args:
        verts: [V, 3] vertex positions
        faces: [F, 3] face indices (0-indexed)
        show: If True, display interactive window. If False, return plotter.
        color: Mesh color (default: 'lightgrey')
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)

    Returns:
        PyVista plotter object (can be used for further customization or saving)

    Example:
        >>> verts = torch.rand(100, 3)
        >>> faces = torch.randint(0, 100, (50, 3))
        >>> render_mesh(verts, faces)  # Opens interactive window
    """
    import numpy as np
    import pyvista as pv

    # Convert to numpy
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Compute bounds for origin reference
    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),  # x_min, x_max
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),  # y_min, y_max
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),  # z_min, z_max
    )

    # PyVista expects faces in VTK format: [n_verts, v0, v1, v2, ...]
    # For triangles: [3, v0, v1, v2, 3, v3, v4, v5, ...]
    n_faces = faces_np.shape[0]
    pv_faces = np.hstack(
        [
            np.full((n_faces, 1), 3, dtype=np.int64),  # 3 verts per face
            faces_np,
        ]
    ).flatten()

    # Create PyVista mesh and compute normals for smooth shading
    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.compute_normals(inplace=True)

    # Set up plotter with nice lighting
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        color=color,
        show_edges=show_edges,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_axes()
    if origin:
        _add_origin_reference(plotter, bounds)

    if show:
        plotter.show()

    return plotter


def render_mesh_result(
    mesh_result,
    show: bool = True,
    color: str = "lightgrey",
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    curves_data: list[tuple] | None = None,
    workplanes_data: list[tuple] | None = None,
    opacity: float = 1.0,
):
    """
    Render a MeshResult object using PyVista.

    Args:
        mesh_result: MeshResult object
        show: If True, display interactive window (default: True)
        color: Mesh color (default: 'lightgrey')
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)
        show_bounds: If True, show bounding box wireframe (default: True)

    Returns:
        PyVista plotter object

    Example:
        >>> from geometry import CADProgram
        >>> prog = CADProgram.load_from_yaml("sphere.yaml")
        >>> mesh = prog.mesh(backend='skimage')
        >>> render_mesh_result(mesh)
    """
    import numpy as np
    import pyvista as pv

    # MeshResult always contains lists (batch_size >= 1)
    # For visualization, we only render the first batch element
    if mesh_result.batch_size > 1:
        print(
            f"Warning: MeshResult has {mesh_result.batch_size} batches. Rendering only the first batch."
        )

    # Get vertices and faces from first batch
    verts = mesh_result.vertices[0]  # [V, 3]

    if mesh_result.faces is None:
        raise ValueError("MeshResult has no faces. Cannot render.")
    faces = mesh_result.faces[0]  # [F, 3]

    # Convert to numpy
    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    # Compute bounds for origin reference
    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),  # x_min, x_max
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),  # y_min, y_max
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),  # z_min, z_max
    )

    # PyVista expects faces in VTK format
    n_faces = faces_np.shape[0]
    pv_faces = np.hstack([np.full((n_faces, 1), 3, dtype=np.int64), faces_np]).flatten()

    # Create PyVista mesh and compute normals
    mesh = pv.PolyData(verts_np, pv_faces)
    mesh.compute_normals(inplace=True)

    # Set up plotter with nice lighting
    plotter = pv.Plotter()
    plotter.add_mesh(
        mesh,
        color=color,
        opacity=opacity,
        show_edges=show_edges,
        smooth_shading=True,
        specular=0.5,
        specular_power=15,
    )
    plotter.add_axes()

    if origin:
        _add_origin_reference(plotter, bounds)

    if show_bounds:
        _add_bounding_box(
            plotter,
            mesh_result.xyz_min,
            mesh_result.xyz_max,
            color="black",
            line_width=0.5,
        )

    if curves_data:
        _add_curves_to_plotter(plotter, curves_data)

    if workplanes_data:
        # Use YAML bounding box (not mesh vertex bounds) for workplane sizing
        wp_bounds = (
            mesh_result.xyz_min[0],
            mesh_result.xyz_max[0],
            mesh_result.xyz_min[1],
            mesh_result.xyz_max[1],
            mesh_result.xyz_min[2],
            mesh_result.xyz_max[2],
        )
        _add_workplanes_to_plotter(plotter, workplanes_data, wp_bounds)

    if show:
        plotter.show()

    return plotter


def render_shape(
    shape: "Shape",
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int = 64,
    backend: str = "skimage",
    show: bool = True,
    color: str = "lightgrey",
    origin: bool = True,
):
    """
    Extract mesh from a Shape and render it.

    This is a convenience function that combines mesh extraction and visualization.

    Args:
        shape: Shape object to render
        xyz_min: Bounding box minimum corner (x, y, z)
        xyz_max: Bounding box maximum corner (x, y, z)
        resolution: Grid resolution for marching cubes (default: 64)
        backend: 'skimage', 'diso-mc', or 'diso-dmc' (default: 'skimage')
        show: If True, display interactive window
        color: Mesh color (default: 'lightgrey')
        origin: If True, show origin reference planes (default: True)

    Returns:
        PyVista plotter object

    Example:
        >>> from geometry.stdlib import sphere
        >>> s = sphere(radius=1.0)
        >>> render_shape(s, (-2,-2,-2), (2,2,2), backend='skimage')
    """
    result = shape.get_mesh(xyz_min, xyz_max, resolution, backend=backend)
    verts, faces = result  # Unpack MeshResult
    return render_mesh(verts, faces, show=show, color=color, origin=origin)


def render_meshes(
    meshes: dict[str, tuple[Tensor, Tensor]],
    show: bool = True,
    colors: dict[str, str] | None = None,
    show_edges: bool = False,
    origin: bool = True,
    show_bounds: bool = True,
    curves_data: list[tuple] | None = None,
    workplanes_data: list[tuple] | None = None,
    opacity: float = 1.0,
):
    """
    Render multiple meshes in the same viewer with different colors.

    Args:
        meshes: Dict mapping body name to MeshResult
        show: If True, display interactive window. If False, return plotter.
        colors: Optional dict mapping body name to color. If not provided,
                colors are assigned automatically from DEFAULT_COLORS.
        show_edges: If True, show wireframe edges (default: False)
        origin: If True, show origin reference planes (default: True)
        show_bounds: If True, show bounding box wireframe (default: True)

    Returns:
        PyVista plotter object

    Example:
        >>> meshes = {"bracket": surf_mesh, "bolt": surf_mesh2}
        >>> render_meshes(meshes)
    """
    import numpy as np
    import pyvista as pv

    plotter = pv.Plotter()

    # Initialize bound tracking for origin reference
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")
    z_min, z_max = float("inf"), float("-inf")

    # Track bounds from first mesh (all should have same bounds from YAML)
    bbox_min = None
    bbox_max = None

    for idx, (name, mesh_result) in enumerate(meshes.items()):
        # NEW: MeshResult always contains lists (batch_size >= 1)
        # For visualization, we only render the first batch element
        if mesh_result.batch_size > 1:
            print(
                f"Warning: Body '{name}' has {mesh_result.batch_size} batches. Rendering only the first batch."
            )

        # Get vertices and faces from first batch
        verts = mesh_result.vertices[0]  # [V, 3]

        if mesh_result.faces is None:
            raise ValueError(f"Body '{name}' has no faces. Cannot render.")
        faces = mesh_result.faces[0]  # [F, 3]

        # Get color for this body
        if colors and name in colors:
            color = colors[name]
        else:
            color = DEFAULT_COLORS[idx % len(DEFAULT_COLORS)]

        # Convert to numpy
        verts_np = verts.detach().cpu().numpy()
        faces_np = faces.detach().cpu().numpy()

        # Update bounds for origin reference
        x_min = min(x_min, verts_np[:, 0].min())
        x_max = max(x_max, verts_np[:, 0].max())
        y_min = min(y_min, verts_np[:, 1].min())
        y_max = max(y_max, verts_np[:, 1].max())
        z_min = min(z_min, verts_np[:, 2].min())
        z_max = max(z_max, verts_np[:, 2].max())

        # Store bounding box from first mesh
        if bbox_min is None:
            bbox_min = mesh_result.xyz_min
            bbox_max = mesh_result.xyz_max

        # PyVista format
        n_faces = faces_np.shape[0]
        pv_faces = np.hstack(
            [np.full((n_faces, 1), 3, dtype=np.int64), faces_np]
        ).flatten()

        mesh = pv.PolyData(verts_np, pv_faces)
        mesh.compute_normals(inplace=True)

        plotter.add_mesh(
            mesh,
            color=color,
            opacity=opacity,
            show_edges=show_edges,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            label=name,
        )

    plotter.add_axes()
    if origin:
        bounds = (x_min, x_max, y_min, y_max, z_min, z_max)
        _add_origin_reference(plotter, bounds)

    if show_bounds and bbox_min is not None:
        _add_bounding_box(
            plotter,
            bbox_min,
            bbox_max,
            color="black",
            line_width=0.5,
        )

    if curves_data:
        _add_curves_to_plotter(plotter, curves_data)

    if workplanes_data and bbox_min is not None:
        # Use YAML bounding box (not mesh vertex bounds) for workplane sizing
        wp_bounds = (
            bbox_min[0],
            bbox_max[0],
            bbox_min[1],
            bbox_max[1],
            bbox_min[2],
            bbox_max[2],
        )
        _add_workplanes_to_plotter(plotter, workplanes_data, wp_bounds)

    plotter.add_legend()

    if show:
        plotter.show()

    return plotter


def _add_curves_to_plotter(
    plotter,
    curves_data: list[tuple],
) -> None:
    """
    Add smooth curve splines to an existing plotter.

    Args:
        plotter: PyVista plotter object to add curves to
        curves_data: List of (polyline_np, cp_np, color, name) tuples.
                     polyline_np is (S, 3) numpy array of curve sample points.
                     cp_np is (N_cp, 3) numpy array of control points.
    """
    import pyvista as pv

    for polyline_np, cp_np, color, name in curves_data:
        spline = pv.Spline(polyline_np, n_points=200)
        plotter.add_mesh(
            spline,
            color=color,
            line_width=3,
            label=name,
        )

        # Add control point markers
        cp_poly = pv.PolyData(cp_np)
        plotter.add_mesh(
            cp_poly,
            color=color,
            point_size=12,
            render_points_as_spheres=False,
            style="points",
        )


def _add_workplanes_to_plotter(
    plotter,
    workplanes_data: list[tuple],
    bounds: tuple,
) -> None:
    """
    Add semi-transparent workplane quads with normal/axis arrows.

    Args:
        plotter: PyVista plotter object
        workplanes_data: List of (origin_np, normal_np, u_np, v_np, color, name)
        bounds: (xmin, xmax, ymin, ymax, zmin, zmax)
    """
    import numpy as np
    import pyvista as pv

    for origin_np, normal_np, u_np, v_np, color, name in workplanes_data:
        # Auto-size: project bounding box ranges onto u and v directions
        x_range = bounds[1] - bounds[0]
        y_range = bounds[3] - bounds[2]
        z_range = bounds[5] - bounds[4]

        ranges = np.array([x_range, y_range, z_range])
        i_size = abs(np.dot(ranges, np.abs(u_np))) * 0.5
        j_size = abs(np.dot(ranges, np.abs(v_np))) * 0.5

        # Fallback: use max range * 0.5 if projection gives 0
        if i_size < 1e-6:
            i_size = max(ranges) * 0.5
        if j_size < 1e-6:
            j_size = max(ranges) * 0.5

        # Create plane quad centered at origin
        plane = pv.Plane(
            center=origin_np,
            direction=normal_np,
            i_size=i_size,
            j_size=j_size,
            i_resolution=1,
            j_resolution=1,
        )
        plotter.add_mesh(
            plane,
            color=color,
            opacity=0.2,
            show_edges=True,
            edge_color=color,
            line_width=3,
            label=f"wp:{name}",
        )

        # Normal arrow (from center, along normal)
        arrow_scale = min(i_size, j_size) * 0.2
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([normal_np]) * arrow_scale,
            color="white",
        )

        # u-axis arrow (from center, along u)
        u_arrow_scale = i_size * 0.15
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([u_np]) * u_arrow_scale,
            color="red",
        )

        # v-axis arrow (from center, along v)
        v_arrow_scale = j_size * 0.15
        plotter.add_arrows(
            np.array([origin_np]),
            np.array([v_np]) * v_arrow_scale,
            color="green",
        )


def _add_sdf_slice_widget(
    plotter,
    sdf_fn,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    plane: str = "xy",
    slice_resolution: int = 256,
    batch_idx: int = 0,
) -> None:
    """
    Add an interactive SDF slice plane widget to an existing plotter.

    The user drags an axis-aligned plane through the 3D bounding box.
    The callback evaluates the SDF on a 2D grid at the plane offset and
    displays filled colors (RdBu_r), regular isolines, and a bold zero contour.

    Args:
        plotter: PyVista plotter object
        sdf_fn: Callable that takes [N, 3] Tensor and returns [N] or [B, N] Tensor
        xyz_min: Bounding box minimum corner (x, y, z)
        xyz_max: Bounding box maximum corner (x, y, z)
        plane: Slice plane - 'xy', 'xz', or 'yz'
        slice_resolution: Number of grid points per axis (default: 256)
        batch_idx: Which batch element to use for SDF evaluation (default: 0)
    """
    import numpy as np
    import pyvista as pv

    PLANE_CFG = {
        "xy": {"grid_axes": (0, 1), "normal_axis": 2, "normal_vec": (0, 0, 1)},
        "xz": {"grid_axes": (0, 2), "normal_axis": 1, "normal_vec": (0, 1, 0)},
        "yz": {"grid_axes": (1, 2), "normal_axis": 0, "normal_vec": (1, 0, 0)},
    }

    if plane not in PLANE_CFG:
        raise ValueError(
            f"Invalid slice plane '{plane}'. Must be one of: 'xy', 'xz', 'yz'"
        )

    cfg = PLANE_CFG[plane]
    nax = cfg["normal_axis"]
    ax0, ax1 = cfg["grid_axes"]

    # Initial plane position: center of bounding box along normal axis
    center = [(xyz_min[i] + xyz_max[i]) / 2.0 for i in range(3)]

    # Pre-compute the fixed 2D grid (u, v never change, only offset moves)
    u = np.linspace(xyz_min[ax0], xyz_max[ax0], slice_resolution)
    v = np.linspace(xyz_min[ax1], xyz_max[ax1], slice_resolution)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    n_pts = slice_resolution * slice_resolution

    def on_plane_move(normal, origin):
        # Clamp offset to meshing bounds
        offset = float(np.clip(origin[nax], xyz_min[nax], xyz_max[nax]))

        # Build [N, 3] query points (F-order to match StructuredGrid point ordering)
        points = torch.zeros(n_pts, 3)
        points[:, ax0] = torch.from_numpy(uu.ravel(order="F")).float()
        points[:, ax1] = torch.from_numpy(vv.ravel(order="F")).float()
        points[:, nax] = offset

        # Evaluate SDF
        with torch.no_grad():
            sdf_vals = sdf_fn(points)
            if sdf_vals.dim() == 2:
                sdf_vals = sdf_vals[batch_idx]

        # Build StructuredGrid for display
        coords = [None, None, None]
        coords[ax0] = uu
        coords[ax1] = vv
        coords[nax] = np.full_like(uu, float(offset))
        grid = pv.StructuredGrid(coords[0], coords[1], coords[2])
        sdf_np = sdf_vals.cpu().numpy()
        grid.point_data["sdf"] = sdf_np

        lo, hi = float(sdf_np.min()), float(sdf_np.max())

        # Symmetric color range so SDF=0 is always at the center (white)
        abs_max = max(abs(lo), abs(hi))
        plotter.add_mesh(
            grid,
            name="sdf_slice",
            scalars="sdf",
            cmap="RdBu_r",
            clim=[-abs_max, abs_max],
            opacity=0.8,
            show_scalar_bar=False,
        )

        # Isolines colored by local SDF value
        iso_spacing = 0.125
        levels = np.arange(np.ceil(lo / iso_spacing) * iso_spacing, hi, iso_spacing)
        levels = levels[levels != 0.0]  # exclude zero (drawn separately)

        if len(levels) > 0:
            isocontours = grid.contour(isosurfaces=levels.tolist(), scalars="sdf")
            if isocontours.n_points > 0:
                plotter.add_mesh(
                    isocontours,
                    name="sdf_isolines",
                    scalars="sdf",
                    cmap="RdBu_r",
                    clim=[-abs_max, abs_max],
                    line_width=1.5,
                    opacity=1.0,
                    show_scalar_bar=False,
                )
            else:
                plotter.remove_actor("sdf_isolines", render=False)
        else:
            plotter.remove_actor("sdf_isolines", render=False)

        # Zero-level isoline: bold black, semi-transparent
        if lo < 0.0 < hi:
            zero_contour = grid.contour(isosurfaces=[0.0], scalars="sdf")
            if zero_contour.n_points > 0:
                plotter.add_mesh(
                    zero_contour,
                    name="sdf_zero",
                    color="black",
                    line_width=4.0,
                    opacity=1.0,
                )
            else:
                plotter.remove_actor("sdf_zero", render=False)
        else:
            plotter.remove_actor("sdf_zero", render=False)

    plotter.add_plane_widget(
        on_plane_move,
        normal=cfg["normal_vec"],
        origin=center,
        bounds=(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2]),
        factor=1.0,
        normal_rotation=False,
        outline_translation=False,
        origin_translation=True,
        implicit=True,
        tubing=False,
        outline_opacity=0.0,
    )

    # Trigger initial render at center position
    on_plane_move(cfg["normal_vec"], center)


def _add_raster_slice_widget(
    plotter,
    sdf_fn,
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    plane: str = "xy",
    slice_resolution: int = 256,
    batch_idx: int = 0,
    epsilon: float = 0.01,
) -> None:
    """
    Add an interactive raster occupancy slice widget to an existing plotter.

    Same drag-plane interaction as _add_sdf_slice_widget, but renders
    black/white occupancy (sigmoid of negative SDF) instead of colored
    SDF values.

    Args:
        plotter: PyVista plotter object
        sdf_fn: Callable that takes [N, 3] Tensor and returns [N] or [B, N] Tensor
        xyz_min: Bounding box minimum corner (x, y, z)
        xyz_max: Bounding box maximum corner (x, y, z)
        plane: Slice plane - 'xy', 'xz', or 'yz'
        slice_resolution: Number of grid points per axis (default: 256)
        batch_idx: Which batch element to use for SDF evaluation (default: 0)
        epsilon: Sigmoid sharpness for occupancy (default: 0.01)
    """
    import numpy as np
    import pyvista as pv

    PLANE_CFG = {
        "xy": {"grid_axes": (0, 1), "normal_axis": 2, "normal_vec": (0, 0, 1)},
        "xz": {"grid_axes": (0, 2), "normal_axis": 1, "normal_vec": (0, 1, 0)},
        "yz": {"grid_axes": (1, 2), "normal_axis": 0, "normal_vec": (1, 0, 0)},
    }

    if plane not in PLANE_CFG:
        raise ValueError(
            f"Invalid slice plane '{plane}'. Must be one of: 'xy', 'xz', 'yz'"
        )

    cfg = PLANE_CFG[plane]
    nax = cfg["normal_axis"]
    ax0, ax1 = cfg["grid_axes"]

    center = [(xyz_min[i] + xyz_max[i]) / 2.0 for i in range(3)]

    u = np.linspace(xyz_min[ax0], xyz_max[ax0], slice_resolution)
    v = np.linspace(xyz_min[ax1], xyz_max[ax1], slice_resolution)
    uu, vv = np.meshgrid(u, v, indexing="ij")
    n_pts = slice_resolution * slice_resolution

    def on_plane_move(normal, origin):
        offset = float(np.clip(origin[nax], xyz_min[nax], xyz_max[nax]))

        points = torch.zeros(n_pts, 3)
        points[:, ax0] = torch.from_numpy(uu.ravel(order="F")).float()
        points[:, ax1] = torch.from_numpy(vv.ravel(order="F")).float()
        points[:, nax] = offset

        with torch.no_grad():
            sdf_vals = sdf_fn(points)
            if sdf_vals.dim() == 2:
                sdf_vals = sdf_vals[batch_idx]

        # Occupancy: 1 inside, 0 outside
        occ = torch.sigmoid(-sdf_vals / epsilon).cpu().numpy()

        coords = [None, None, None]
        coords[ax0] = uu
        coords[ax1] = vv
        coords[nax] = np.full_like(uu, float(offset))
        grid = pv.StructuredGrid(coords[0], coords[1], coords[2])
        grid.point_data["occupancy"] = occ

        # Black (inside=1) on white (outside=0): use gray_r colormap
        plotter.add_mesh(
            grid,
            name="raster_slice",
            scalars="occupancy",
            cmap="gray_r",
            clim=[0, 1],
            opacity=0.9,
            show_scalar_bar=False,
        )

    plotter.add_plane_widget(
        on_plane_move,
        normal=cfg["normal_vec"],
        origin=center,
        bounds=(xyz_min[0], xyz_max[0], xyz_min[1], xyz_max[1], xyz_min[2], xyz_max[2]),
        factor=1.0,
        normal_rotation=False,
        outline_translation=False,
        origin_translation=True,
        implicit=True,
        tubing=False,
        outline_opacity=0.0,
    )

    on_plane_move(cfg["normal_vec"], center)


# Map view names to PyVista camera position strings
VIEW_CAMERAS = {
    "iso": "iso",
    "front": "yz",  # looking along -X
    "back": "-yz",  # looking along +X
    "left": "xz",  # looking along -Y
    "right": "-xz",  # looking along +Y
    "top": "xy",  # looking along -Z
    "bottom": "-xy",  # looking along +Z
}


_MULTI_BODY_PALETTE = [
    "lightgrey",
    "steelblue",
    "coral",
    "mediumseagreen",
    "orchid",
    "goldenrod",
    "slategrey",
    "tomato",
]


def _mesh_result_to_pv(mesh_result, np_module):
    """Convert a single MeshResult (batch 0) to PyVista PolyData + bounds."""
    if mesh_result.batch_size > 1:
        print(
            f"Warning: MeshResult has {mesh_result.batch_size} batches. "
            "Rendering only the first batch."
        )

    verts = mesh_result.vertices[0]
    if mesh_result.faces is None:
        raise ValueError("MeshResult has no faces. Cannot render.")
    faces = mesh_result.faces[0]

    verts_np = verts.detach().cpu().numpy()
    faces_np = faces.detach().cpu().numpy()

    n_faces = faces_np.shape[0]
    if n_faces == 0:
        return None, None

    import pyvista as pv

    pv_faces = np_module.hstack(
        [np_module.full((n_faces, 1), 3, dtype=np_module.int64), faces_np]
    ).flatten()
    pv_mesh = pv.PolyData(verts_np, pv_faces)
    pv_mesh.compute_normals(inplace=True)

    bounds = (
        verts_np[:, 0].min(),
        verts_np[:, 0].max(),
        verts_np[:, 1].min(),
        verts_np[:, 1].max(),
        verts_np[:, 2].min(),
        verts_np[:, 2].max(),
    )
    return pv_mesh, bounds


def save_mesh_image_pyvista(
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
    """Save mesh visualization to image file using PyVista."""
    from pathlib import Path

    import numpy as np
    import pyvista as pv

    if view not in VIEW_CAMERAS:
        raise ValueError(
            f"Unknown view '{view}'. Available views: {list(VIEW_CAMERAS.keys())}"
        )

    # Ensure path has correct extension
    path = Path(path)
    if not path.suffix:
        path = path.with_suffix(f".{format}")

    # Normalize to list of (name, MeshResult) pairs
    if isinstance(mesh_result, dict):
        bodies = list(mesh_result.items())
    else:
        bodies = [(None, mesh_result)]

    # Convert all bodies to PyVista meshes
    pv_meshes: list[tuple[str | None, pv.PolyData]] = []
    all_bounds = []
    for name, mr in bodies:
        pv_mesh, body_bounds = _mesh_result_to_pv(mr, np)
        if pv_mesh is not None:
            pv_meshes.append((name, pv_mesh))
            all_bounds.append(body_bounds)

    if not pv_meshes:
        raise ValueError("No non-empty meshes to render.")

    # Unified bounds across all bodies
    bounds = (
        min(b[0] for b in all_bounds),
        max(b[1] for b in all_bounds),
        min(b[2] for b in all_bounds),
        max(b[3] for b in all_bounds),
        min(b[4] for b in all_bounds),
        max(b[5] for b in all_bounds),
    )

    # Set up off-screen plotter
    plotter = pv.Plotter(off_screen=True, window_size=window_size)

    for i, (name, pv_mesh) in enumerate(pv_meshes):
        if name is not None and colors and name in colors:
            body_color = colors[name]
        elif name is None:
            body_color = color
        else:
            body_color = _MULTI_BODY_PALETTE[i % len(_MULTI_BODY_PALETTE)]
        body_opacity = 1.0
        if name is not None and opacities and name in opacities:
            body_opacity = opacities[name]
        plotter.add_mesh(
            pv_mesh,
            color=body_color,
            opacity=body_opacity,
            show_edges=show_edges,
            smooth_shading=True,
            specular=0.5,
            specular_power=15,
            label=name,
        )

    plotter.add_axes()

    if origin:
        _add_origin_reference(plotter, bounds)

    if show_bounds:
        # Use unified bounds for bounding box
        if isinstance(mesh_result, dict):
            # Compute unified xyz_min/max from all bodies
            first_mr = next(iter(mesh_result.values()))
            xyz_min = list(first_mr.xyz_min)
            xyz_max = list(first_mr.xyz_max)
            for mr in mesh_result.values():
                for j in range(3):
                    xyz_min[j] = min(xyz_min[j], mr.xyz_min[j])
                    xyz_max[j] = max(xyz_max[j], mr.xyz_max[j])
            _add_bounding_box(
                plotter,
                tuple(xyz_min),
                tuple(xyz_max),
                color="black",
                line_width=0.5,
            )
        else:
            _add_bounding_box(
                plotter,
                mesh_result.xyz_min,
                mesh_result.xyz_max,
                color="black",
                line_width=0.5,
            )

    # Set camera view (PyVista handles zoom automatically)
    plotter.camera_position = VIEW_CAMERAS[view]

    # Save screenshot
    plotter.screenshot(str(path))


def _add_field_slice(
    plotter,
    field_slice,
    batch_idx: int = 0,
    actor_name: str | None = None,
) -> None:
    """Render a pre-computed FieldSlice as a colored 2D grid on the plotter.

    Unlike the interactive SDF/raster slice widgets, this is static,
    the data is pre-computed at a fixed plane and offset.

    Args:
        plotter: PyVista plotter object.
        field_slice: FieldSlice dataclass with values, extents, plane, etc.
        batch_idx: Which batch element to display if values is [B, H, W].
        actor_name: Optional unique actor name (for multi-field overlays).
    """
    import numpy as np
    import pyvista as pv

    PLANE_CFG = {
        "xy": {"grid_axes": (0, 1), "normal_axis": 2},
        "xz": {"grid_axes": (0, 2), "normal_axis": 1},
        "yz": {"grid_axes": (1, 2), "normal_axis": 0},
    }

    plane = field_slice.plane
    if plane not in PLANE_CFG:
        raise ValueError(
            f"Invalid field slice plane '{plane}'. Must be 'xy', 'xz', or 'yz'"
        )

    cfg = PLANE_CFG[plane]
    ax0, ax1 = cfg["grid_axes"]
    nax = cfg["normal_axis"]

    # Extract [H, W] values
    vals = field_slice.values
    if vals.dim() == 2:
        vals = vals.unsqueeze(0)
    vals_2d = vals[batch_idx].detach().cpu().numpy()
    H, W = vals_2d.shape

    # Build 2D meshgrid matching the extents
    (u_min, u_max), (v_min, v_max) = field_slice.extents
    u = np.linspace(u_min, u_max, H)
    v = np.linspace(v_min, v_max, W)
    uu, vv = np.meshgrid(u, v, indexing="ij")

    # Assemble StructuredGrid at the correct plane + offset
    coords = [None, None, None]
    coords[ax0] = uu
    coords[ax1] = vv
    coords[nax] = np.full_like(uu, float(field_slice.offset))
    grid = pv.StructuredGrid(coords[0], coords[1], coords[2])

    # Assign field data (Fortran order to match StructuredGrid point layout)
    grid.point_data[field_slice.name] = vals_2d.ravel(order="F")

    # Color range
    if field_slice.clim is not None:
        clim = list(field_slice.clim)
    else:
        lo, hi = float(vals_2d.min()), float(vals_2d.max())
        clim = [lo, hi]

    name = actor_name or f"field_slice_{field_slice.name}"
    plotter.add_mesh(
        grid,
        name=name,
        scalars=field_slice.name,
        cmap=field_slice.cmap,
        clim=clim,
        opacity=field_slice.opacity,
        show_scalar_bar=True,
        scalar_bar_args={"title": field_slice.name},
    )


def render_boxes(
    boxes: "list[BoxSpec]",
    show: bool = True,
    origin: bool = True,
    show_bounds: bool = True,
    xyz_min: tuple[float, float, float] | None = None,
    xyz_max: tuple[float, float, float] | None = None,
    color: str | None = None,
    colors: dict[str, str] | None = None,
    show_edges: bool = False,
):
    """Render axis-aligned boxes directly using pv.Box(), no SDF meshing needed.

    This is the fast path for configs composed entirely of box primitives.
    Each :class:`BoxSpec` becomes a native PyVista box with proper normals
    and shading, skipping the expensive SDF evaluation and marching cubes.

    Args:
        boxes: List of BoxSpec objects to render.
        show: If True, display interactive window (default: True).
        origin: Show origin reference planes (default: True).
        show_bounds: Show bounding box wireframe (default: True).
        xyz_min: Bounding box min corner (auto-computed from boxes if None).
        xyz_max: Bounding box max corner (auto-computed from boxes if None).
        color: Single color for all boxes (default: use palette per shape name).
        colors: Color dict keyed by shape name (overrides color and palette).
        show_edges: Show wireframe edges on boxes (default: False).

    Returns:
        PyVista plotter object.
    """
    import pyvista as pv

    # Auto-compute bounding box from all boxes if not provided
    if xyz_min is None or xyz_max is None:
        all_bounds = [b.bounds for b in boxes]
        xyz_min = (
            min(bn[0] for bn in all_bounds),
            min(bn[2] for bn in all_bounds),
            min(bn[4] for bn in all_bounds),
        )
        xyz_max = (
            max(bn[1] for bn in all_bounds),
            max(bn[3] for bn in all_bounds),
            max(bn[5] for bn in all_bounds),
        )

    plotter = _create_plotter()
    plotter.add_axes()

    scene_bounds = (
        xyz_min[0],
        xyz_max[0],
        xyz_min[1],
        xyz_max[1],
        xyz_min[2],
        xyz_max[2],
    )

    if origin:
        _add_origin_reference(plotter, scene_bounds)

    if show_bounds:
        _add_bounding_box(plotter, xyz_min, xyz_max, color="black", line_width=0.5)

    # Assign colors: explicit dict > single color > auto palette
    color_map: dict[str, str] = {}
    seen_names: list[str] = []
    for b in boxes:
        if b.name not in seen_names:
            seen_names.append(b.name)

    if colors is not None:
        color_map = colors
    elif color is not None:
        color_map = {n: color for n in seen_names}
    else:
        # Single color for all boxes (they're one body); use palette
        # only when caller explicitly provides a colors dict.
        color_map = {n: "lightgrey" for n in seen_names}

    for b in boxes:
        box_mesh = pv.Box(bounds=b.bounds)
        plotter.add_mesh(
            box_mesh,
            color=color_map.get(b.name, "lightgrey"),
            show_edges=show_edges,
            smooth_shading=False,
            specular=0.3,
            specular_power=10,
        )

    if show:
        plotter.show()

    return plotter


def _create_plotter():
    """Create a new PyVista Plotter instance. Isolated here so tests can patch it without importing pyvista."""
    import pyvista as pv

    return pv.Plotter()
