"""
Mesh Extraction Module

Converts SDF functions to triangle meshes.
Supports four backends:
- 'skimage': scikit-image marching cubes (CPU only, not differentiable)
- 'diso-mc': DiffMC - Differentiable Marching Cubes (CUDA only, differentiable)
- 'diso-dmc': DiffDMC - Differentiable Dual Marching Cubes (CUDA only, differentiable)
- 'sparse-dmc': Sparse octree + dual marching cubes (any device, differentiable)

Boundary clipping detection:
    After SDF grid evaluation, all dense backends (skimage, diso-mc, diso-dmc)
    check whether any SDF values on the 6 boundary faces of the bounding box are
    negative. If so, a ``warnings.warn()`` is emitted listing which faces are
    clipped. The sparse-dmc backend performs the same check inside
    ``sparsedmc/core.py:octree_sdf_eval()``.
"""

import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch
from torch import Tensor


@runtime_checkable
class MeshTimer(Protocol):
    """Protocol for mesh extraction phase timing.

    Any object with a ``start(name)`` method satisfies this protocol.
    Each ``start()`` call implicitly ends the previous phase.
    """

    def start(self, name: str) -> None: ...


@dataclass
class MeshResult:
    """
    Batched mesh result.

    Uses lists of tensors because mesh sizes vary per batch.

    FUTURE: If batched meshing is implemented, redesign to use:
    - vertices: Tensor [B, V_max, 3] (padded)
    - vertex_mask: Tensor [B, V_max] (validity mask)

    Attributes:
        vertices: List of vertex tensors, one per batch element. Each: [V_i, 3]
        faces: List of face tensors, one per batch element. Each: [F_i, 3]
        batch_size: Number of batch elements (always >= 1, never None)
        backend: Which backend was used ('skimage', 'diso-mc', 'diso-dmc', or 'sparse-dmc')
        xyz_min: Minimum bounds of mesh extraction volume [x, y, z]
        xyz_max: Maximum bounds of mesh extraction volume [x, y, z]
        resolution: Grid resolution used for mesh extraction
    """

    # Lists of per-batch tensors (length = batch_size)
    vertices: list[Tensor]
    faces: list[Tensor]

    # Metadata
    batch_size: int  # Always >= 1, never None
    backend: str
    xyz_min: tuple[float, float, float]
    xyz_max: tuple[float, float, float]
    resolution: int

    def __getitem__(self, idx: int) -> "MeshResult":
        """Extract single batch element."""
        if idx >= self.batch_size:
            raise IndexError(
                f"Index {idx} out of range for batch_size={self.batch_size}"
            )

        return MeshResult(
            vertices=[self.vertices[idx]],
            faces=[self.faces[idx]],
            batch_size=1,
            backend=self.backend,
            xyz_min=self.xyz_min,
            xyz_max=self.xyz_max,
            resolution=self.resolution,
        )

    def __iter__(self):
        """
        Tuple unpacking returns lists of tensors.

        Returns (List[vertices], List[faces]).
        """
        return iter((self.vertices, self.faces))

    def __repr__(self):
        if self.batch_size == 1:
            # Single batch - show mesh stats
            v = self.vertices[0].shape[0]
            f = self.faces[0].shape[0]
            return (
                f"<MeshResult batch=1 vertices={v} faces={f} backend='{self.backend}'>"
            )
        # Multiple batches - just show batch count
        return f"<MeshResult batch={self.batch_size} backend='{self.backend}'>"


def _check_boundary_clipping(
    sdf_grid: "Tensor | np.ndarray",
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    batch_index: int | None = None,
) -> None:
    """
    Check if SDF values on boundary faces of the bounding box are negative.

    Negative SDF on a boundary face means the shape extends beyond the mesh
    extraction volume and is being clipped. Emits a warning listing which
    faces are affected.

    Args:
        sdf_grid: 3D SDF grid [res_x, res_y, res_z] (numpy or torch tensor).
        xyz_min: Bounding box min corner (x, y, z).
        xyz_max: Bounding box max corner (x, y, z).
        batch_index: If not None, include batch index in warning message.
    """
    # Convert to numpy for uniform handling
    if isinstance(sdf_grid, Tensor):
        grid = sdf_grid.detach().cpu().numpy()
    else:
        grid = sdf_grid

    face_labels = {
        "x_min": grid[0, :, :],
        "x_max": grid[-1, :, :],
        "y_min": grid[:, 0, :],
        "y_max": grid[:, -1, :],
        "z_min": grid[:, :, 0],
        "z_max": grid[:, :, -1],
    }

    clipped_faces = []
    for label, face_sdf in face_labels.items():
        if np.any(face_sdf < 0):
            clipped_faces.append(label)

    if clipped_faces:
        batch_str = f" (batch {batch_index})" if batch_index is not None else ""
        faces_str = ", ".join(clipped_faces)
        warnings.warn(
            f"Shape extends beyond mesh bounds{batch_str}. "
            f"Clipped faces: {faces_str}. "
            f"Current bounds: x=[{xyz_min[0]}, {xyz_max[0]}], "
            f"y=[{xyz_min[1]}, {xyz_max[1]}], "
            f"z=[{xyz_min[2]}, {xyz_max[2]}]. "
            f"Increase bounds to capture the full shape.",
            stacklevel=4,
        )


def project_mesh(
    mesh: "MeshResult",
    sdf_fn: Callable[[Tensor], Tensor],
    max_displacement: float | None = None,
) -> "MeshResult":
    """
    Project existing mesh vertices onto a new isosurface.

    Lightweight alternative to full remeshing: v_new = v - sdf(v) * normal.
    Requires only 1 SDF eval per vertex with zero octree/table overhead.
    Differentiable, gradients flow through sdf_fn back to network weights.

    All heavy compute (normals, SDF eval, projection) is fully vectorized
    over the batch dimension, no Python for-loops over B.

    Args:
        mesh: Existing MeshResult to project
        sdf_fn: SDF function. Accepts [B, N, 3] -> [B, N] (Shape objects work).
        max_displacement: If set, clamp per-vertex displacement magnitude.

    Returns:
        New MeshResult with updated vertices, same faces and metadata.
    """
    B = mesh.batch_size
    device = mesh.vertices[0].device
    dtype = mesh.vertices[0].dtype

    V_counts = [v.shape[0] for v in mesh.vertices]
    F_counts = [f.shape[0] for f in mesh.faces]
    V_max = max(V_counts)
    F_max = max(F_counts)

    # 1. Pad vertices and faces to [B, V_max/F_max, 3]
    #    Padded faces point to vertex 0, whose padded position is [0,0,0],
    #    so cross products for padded faces are zero, no corruption.
    verts_padded = torch.zeros(B, V_max, 3, device=device, dtype=dtype)
    faces_padded = torch.zeros(B, F_max, 3, device=device, dtype=torch.long)
    for b in range(B):
        verts_padded[b, : V_counts[b]] = mesh.vertices[b]
        faces_padded[b, : F_counts[b]] = mesh.faces[b]

    # 2. Batched vertex normals, fully vectorized
    #    Gather face corner vertices via the padded face indices
    def _gather_corner(corner: int) -> Tensor:
        idx = (
            faces_padded[:, :, corner].unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, F_max, 3]
        return torch.gather(verts_padded, 1, idx)  # [B, F_max, 3]

    v0, v1, v2 = _gather_corner(0), _gather_corner(1), _gather_corner(2)
    face_normals = torch.cross(v1 - v0, v2 - v0, dim=-1)  # [B, F_max, 3]

    #    Scatter-add face normals onto vertices (dim=1 operates per batch)
    vertex_normals = torch.zeros_like(verts_padded)  # [B, V_max, 3]
    for corner in range(3):
        idx = (
            faces_padded[:, :, corner].unsqueeze(-1).expand(-1, -1, 3)
        )  # [B, F_max, 3]
        vertex_normals.scatter_add_(1, idx, face_normals)

    norms = vertex_normals.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    vertex_normals = vertex_normals / norms

    # 3. Batched SDF eval: [B, V_max, 3] -> [B, V_max]
    sdf_vals = sdf_fn(verts_padded)

    # 4. Project: v_new = v - sdf * normal  (fully batched)
    displacement = sdf_vals.unsqueeze(-1) * vertex_normals  # [B, V_max, 3]

    if max_displacement is not None:
        disp_mag = displacement.norm(dim=-1, keepdim=True).clamp(min=1e-12)
        scale = (max_displacement / disp_mag).clamp(max=1.0)
        displacement = displacement * scale

    new_verts_padded = verts_padded - displacement

    # 5. Unpad back to list
    new_vertices = [new_verts_padded[b, : V_counts[b]] for b in range(B)]

    return MeshResult(
        vertices=new_vertices,
        faces=list(mesh.faces),
        batch_size=B,
        backend=mesh.backend,
        xyz_min=mesh.xyz_min,
        xyz_max=mesh.xyz_max,
        resolution=mesh.resolution,
    )


def sdf_to_mesh(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    backend: Literal["skimage", "diso-mc", "diso-dmc", "sparse-dmc"],
    batch_size: int | None = 1,
    device: torch.device | None = None,
    timer: MeshTimer | None = None,
) -> MeshResult:
    """
    Convert SDF to mesh.

    Always returns single MeshResult with lists (batch_size always int >= 1).

    Args:
        sdf_fn: SDF function [N, 3] -> [B, N]
        xyz_min: Bounding box min corner
        xyz_max: Bounding box max corner
        resolution: Grid resolution per axis
        backend: 'skimage', 'diso-mc', or 'diso-dmc' (REQUIRED)
        batch_size: Batch size (always int >= 1, default 1)

    Returns:
        MeshResult with lists of per-batch tensors

    Raises:
        ValueError: If backend not specified or invalid
        RuntimeError: If diso backend used without CUDA
    """
    # Validate backend
    # Treat None batch_size as 1 (unbatched)
    if batch_size is None:
        batch_size = 1

    # Validate backend
    valid_backends = ("skimage", "diso-mc", "diso-dmc", "sparse-dmc")
    if backend not in valid_backends:
        raise ValueError(
            f"backend must be one of {valid_backends}, got '{backend}'. "
            f"This parameter is required - there is no default."
        )

    # Dispatch to backend
    if backend == "skimage":
        return _mesh_skimage(sdf_fn, xyz_min, xyz_max, resolution, batch_size)
    if backend in ("diso-mc", "diso-dmc"):
        algorithm = "mc" if backend == "diso-mc" else "dmc"
        return _mesh_diso(sdf_fn, xyz_min, xyz_max, resolution, batch_size, algorithm)
    if backend == "sparse-dmc":
        return _mesh_sparse_dmc(
            sdf_fn, xyz_min, xyz_max, resolution, batch_size, device=device, timer=timer
        )


def _mesh_diso(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    batch_size: int,
    algorithm: Literal["mc", "dmc"],
) -> MeshResult:
    """
    Mesh extraction using diso (DiffMC or DiffDMC).

    CUDA-only. Raises RuntimeError on CPU/MPS devices.
    """
    # CUDA check - ERROR not warning
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"diso backend 'diso-{algorithm}' requires CUDA. "
            f"No CUDA device available. "
            f"Use 'skimage' instead."
        )

    from diso import DiffDMC, DiffMC

    device = "cuda"

    # Select algorithm
    if algorithm == "mc":
        extractor = DiffMC(dtype=torch.float32).cuda()
    else:
        extractor = DiffDMC(dtype=torch.float32).cuda()

    # Create 3D grid for SDF evaluation
    x = torch.linspace(float(xyz_min[0]), float(xyz_max[0]), resolution, device=device)
    y = torch.linspace(float(xyz_min[1]), float(xyz_max[1]), resolution, device=device)
    z = torch.linspace(float(xyz_min[2]), float(xyz_max[2]), resolution, device=device)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    grid_points = grid.reshape(-1, 3)

    # Evaluate SDF
    sdf_values = sdf_fn(grid_points)  # [B, N] or [N]
    if sdf_values.dim() == 1:
        sdf_values = sdf_values.unsqueeze(0)

    # Reshape to 3D grid: [B, res, res, res]
    sdf_grids = sdf_values.reshape(batch_size, resolution, resolution, resolution)

    # Check for boundary clipping
    for b in range(batch_size):
        _check_boundary_clipping(
            sdf_grids[b],
            xyz_min,
            xyz_max,
            batch_index=b if batch_size > 1 else None,
        )

    # Extract meshes per batch
    vertices_list = []
    faces_list = []

    for b in range(batch_size):
        # diso outputs vertices in [0,1], normalize=True
        verts, faces = extractor(sdf_grids[b], isovalue=0.0, normalize=True)

        # Scale from [0,1] to world coordinates
        scale = torch.tensor(
            [xyz_max[0] - xyz_min[0], xyz_max[1] - xyz_min[1], xyz_max[2] - xyz_min[2]],
            device=device,
        )
        offset = torch.tensor([xyz_min[0], xyz_min[1], xyz_min[2]], device=device)
        verts = verts * scale + offset

        vertices_list.append(verts)
        faces_list.append(faces)

    return MeshResult(
        vertices=vertices_list,
        faces=faces_list,
        batch_size=batch_size,
        backend=f"diso-{algorithm}",
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        resolution=resolution,
    )


def _mesh_skimage(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    batch_size: int,
) -> MeshResult:
    """
    Mesh extraction using scikit-image marching cubes.

    Always returns single MeshResult with lists (batch_size always int >= 1).
    """
    from skimage.measure import marching_cubes

    # Create 3D grid
    x = torch.linspace(float(xyz_min[0]), float(xyz_max[0]), resolution)
    y = torch.linspace(float(xyz_min[1]), float(xyz_max[1]), resolution)
    z = torch.linspace(float(xyz_min[2]), float(xyz_max[2]), resolution)
    grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)
    points = grid.reshape(-1, 3)

    # Evaluate SDF
    with torch.no_grad():
        sdf_values = sdf_fn(points)

    # Handle unbatched SDF output: [N] -> [1, N]
    if sdf_values.dim() == 1:
        sdf_values = sdf_values.unsqueeze(0)

    def extract_single(sdf_3d: np.ndarray) -> tuple[Tensor, Tensor]:
        """Extract mesh from single 3D SDF grid. Returns (vertices, faces)."""
        verts, faces, normals, values = marching_cubes(sdf_3d, level=0.0)

        # Scale to world coordinates
        scale = np.array(
            [
                (xyz_max[0] - xyz_min[0]) / (resolution - 1),
                (xyz_max[1] - xyz_min[1]) / (resolution - 1),
                (xyz_max[2] - xyz_min[2]) / (resolution - 1),
            ]
        )
        offset = np.array([xyz_min[0], xyz_min[1], xyz_min[2]])
        verts = verts * scale + offset

        return (
            torch.from_numpy(verts.copy()).float(),
            torch.from_numpy(faces.copy()).long(),
        )

    # Always return single MeshResult with lists
    # batch_size is always int >= 1
    sdf_grids = sdf_values.reshape(batch_size, resolution, resolution, resolution)

    # Check for boundary clipping
    for b in range(batch_size):
        _check_boundary_clipping(
            sdf_grids[b],
            xyz_min,
            xyz_max,
            batch_index=b if batch_size > 1 else None,
        )

    # Extract per-batch meshes
    vertices_list = []
    faces_list = []

    for b in range(batch_size):
        verts_b, faces_b = extract_single(sdf_grids[b].cpu().numpy())
        vertices_list.append(verts_b)
        faces_list.append(faces_b)

    return MeshResult(
        vertices=vertices_list,
        faces=faces_list,
        batch_size=batch_size,
        backend="skimage",
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        resolution=resolution,
    )


def _mesh_sparse_dmc(
    sdf_fn: Callable[[Tensor], Tensor],
    xyz_min: tuple[float, float, float],
    xyz_max: tuple[float, float, float],
    resolution: int,
    batch_size: int,
    device: torch.device | None = None,
    timer: MeshTimer | None = None,
) -> MeshResult:
    """
    Mesh extraction using sparse octree + dual marching cubes.

    Works on any device (CPU, MPS, CUDA). Differentiable.
    Resolution must be a power of 2. If not, rounds up to the next power of 2.
    """
    from .sparsedmc import sparse_sdf_to_mesh

    # Round resolution up to next power of 2 if needed
    if resolution < 2 or (resolution & (resolution - 1)) != 0:
        res_po2 = 1
        while res_po2 < resolution:
            res_po2 *= 2
        resolution = res_po2

    vertices_list, faces_list, _ = sparse_sdf_to_mesh(
        sdf_fn,
        xyz_min,
        xyz_max,
        resolution,
        batch_size,
        triangulate=True,
        device=device,
        timer=timer,
    )

    return MeshResult(
        vertices=vertices_list,
        faces=faces_list,
        batch_size=batch_size,
        backend="sparse-dmc",
        xyz_min=xyz_min,
        xyz_max=xyz_max,
        resolution=resolution,
    )
