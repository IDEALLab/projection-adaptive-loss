"""
Sparse Differentiable Dual Marching Cubes.

Octree-based adaptive SDF evaluation + sparse DMC mesh extraction.
Pure PyTorch, works on CPU, MPS, and CUDA.
"""

from .core import octree_sdf_eval, sparse_dual_marching_cubes, sparse_sdf_to_mesh

__all__ = [
    "octree_sdf_eval",
    "sparse_dual_marching_cubes",
    "sparse_sdf_to_mesh",
]
