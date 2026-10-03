"""
Mesh metric computation utilities for volume and surface area.

This module provides helper functions for computing geometric properties
from surface meshes. All functions are fully differentiable
and implemented in pure PyTorch.
"""

import torch
from torch import Tensor


def _compute_volume_from_mesh(vertices: Tensor, faces: Tensor) -> Tensor:
    """
    Compute volume from surface mesh using divergence theorem.

    For a closed triangle mesh, volume can be computed using the divergence theorem:
    V = (1/6) * Σ v0 · (v1 × v2)

    where each triangle contributes a signed volume of the tetrahedron formed
    by the triangle and the origin.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] triangle indices

    Returns:
        Scalar tensor with volume
    """
    # Divergence theorem formula
    v0 = vertices[faces[:, 0]]  # [F, 3]
    v1 = vertices[faces[:, 1]]  # [F, 3]
    v2 = vertices[faces[:, 2]]  # [F, 3]

    # Compute signed volume contribution from each triangle
    cross = torch.cross(v1, v2, dim=1)  # [F, 3]
    signed_volumes = torch.sum(v0 * cross, dim=1) / 6.0  # [F]

    # Sum all signed volumes and take absolute value
    return signed_volumes.sum().abs()


def _compute_surface_area_from_mesh(vertices: Tensor, faces: Tensor) -> Tensor:
    """
    Compute surface area from triangle mesh.

    Surface area is the sum of areas of all triangles.
    Area of triangle with vertices v0, v1, v2:
    A = 0.5 * ||(v1 - v0) × (v2 - v0)||

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] triangle indices

    Returns:
        Scalar tensor with total surface area
    """
    v0 = vertices[faces[:, 0]]  # [F, 3]
    v1 = vertices[faces[:, 1]]  # [F, 3]
    v2 = vertices[faces[:, 2]]  # [F, 3]

    # Compute cross product of edge vectors
    edge1 = v1 - v0  # [F, 3]
    edge2 = v2 - v0  # [F, 3]
    cross = torch.cross(edge1, edge2, dim=1)  # [F, 3]

    # Area of each triangle = 0.5 * ||cross product||
    # Use epsilon-safe norm to avoid NaN gradient for degenerate triangles
    areas = torch.sqrt((cross**2).sum(dim=1) + 1e-12) / 2.0  # [F]

    # Sum all triangle areas
    return areas.sum()


def _is_watertight(faces: Tensor) -> bool:
    """
    Check if mesh is watertight (every edge shared by exactly 2 faces).

    A watertight mesh has no boundary edges (count=1) and no non-manifold
    edges (count>2).

    Args:
        faces: [F, 3] triangle indices

    Returns:
        True if watertight, False otherwise
    """
    edges = torch.cat([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], dim=0)
    edges = torch.sort(edges, dim=1).values
    _, counts = torch.unique(edges, dim=0, return_counts=True)
    return bool((counts == 2).all())


def _compute_surface_area_and_flux_from_mesh(
    vertices: Tensor, faces: Tensor
) -> tuple[Tensor, Tensor]:
    """
    Compute surface area and normal flux from triangle mesh in one pass.

    This function computes both metrics efficiently by sharing the cross
    product computation, which is the most expensive part.

    Normal flux measures mesh closure: for a watertight mesh, the sum of
    all face normals (weighted by area) should be zero. Non-zero flux
    indicates holes or non-manifold geometry.

    Args:
        vertices: [V, 3] vertex positions
        faces: [F, 3] triangle indices

    Returns:
        Tuple of (area, flux_loss):
        - area: Scalar tensor with total surface area
        - flux_loss: Scalar tensor with squared magnitude of normal flux
                     (0 for watertight mesh, >0 for open mesh)
    """
    v0 = vertices[faces[:, 0]]  # [F, 3]
    v1 = vertices[faces[:, 1]]  # [F, 3]
    v2 = vertices[faces[:, 2]]  # [F, 3]

    # Compute cross product of edge vectors (shared computation)
    edge1 = v1 - v0  # [F, 3]
    edge2 = v2 - v0  # [F, 3]
    cross = torch.cross(edge1, edge2, dim=1)  # [F, 3] - unnormalized normal * 2*area

    # Surface area: sum of triangle areas
    # Area of each triangle = 0.5 * ||cross product||
    # Use epsilon-safe norm to avoid NaN gradient for degenerate triangles
    areas = torch.sqrt((cross**2).sum(dim=1) + 1e-12) / 2.0  # [F]
    total_area = areas.sum()

    # Normal flux: sum of (normal * area) vectors
    # For watertight mesh, this should sum to zero
    # cross = normal * 2*area, so we divide by 2
    flux_vec = cross.sum(dim=0) / 2.0  # [3]
    flux_loss = (flux_vec**2).sum()  # scalar

    return total_area, flux_loss
