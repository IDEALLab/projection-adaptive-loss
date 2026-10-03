"""
geometry - PyTorch-based Signed Distance Function (SDF) library

A minimal reimplementation of libfive's SDF primitives using PyTorch,
designed for differentiable CAD and batched parameter evaluation.

Usage:
    from geometry import CADProgram

    prog = CADProgram.load_from_yaml("sphere.yaml")
    mesh = prog.mesh(backend='sparse-dmc')
    prog.visualize(mesh)
"""

# Enable MPS fallback to CPU for unsupported ops (must be set before importing torch)
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

from .graph import (
    AXIS,
    EDGE_TYPES,
    NODE_TYPES,
    PARAM_NAMES,
    PLANE,
    VAR_DIM,
    graph_to_yaml,
    load_graphs,
    save_graphs,
    yaml_to_graph,
)
from .mesh import MeshResult, MeshTimer
from .measure.isocontour import IsocontourResult
from .measure.overlap import OverlapResult
from .measure.rasterize import RasterResult
from .render.field_slice import FieldSlice
from .program import CADProgram

__version__ = "0.1.0"
__all__ = [
    "AXIS",
    "EDGE_TYPES",
    "NODE_TYPES",
    "PARAM_NAMES",
    "PLANE",
    "VAR_DIM",
    "CADProgram",
    "IsocontourResult",
    "MeshResult",
    "MeshTimer",
    "OverlapResult",
    "FieldSlice",
    "RasterResult",
    "graph_to_yaml",
    "load_graphs",
    "save_graphs",
    "yaml_to_graph",
]
