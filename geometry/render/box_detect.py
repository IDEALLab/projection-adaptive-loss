"""
Box Fast-Path Detection

Walks a YAML shape config to detect when the output is composed entirely of
axis-aligned box primitives (possibly under union, translate, scale, mirror).
When detected, returns BoxSpec objects that can be rendered directly with
PyVista's pv.Box(), bypassing the SDF -> mesh pipeline entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from torch import Tensor

from ..loader import deep_copy_config, substitute_params
from ..utils import tensorify_shape_def


@dataclass
class BoxSpec:
    """One axis-aligned box, ready for pv.Box(bounds=...) rendering.

    Attributes:
        center: (cx, cy, cz) world-space center
        size: (sx, sy, sz) full width, height, depth (all positive)
        name: Shape name from the YAML config (for labeling / color assignment)
    """

    center: tuple[float, float, float]
    size: tuple[float, float, float]
    name: str

    @property
    def bounds(self) -> tuple[float, float, float, float, float, float]:
        """Return (xmin, xmax, ymin, ymax, zmin, zmax) for pv.Box()."""
        cx, cy, cz = self.center
        sx, sy, sz = self.size
        return (
            cx - sx / 2,
            cx + sx / 2,
            cy - sy / 2,
            cy + sy / 2,
            cz - sz / 2,
            cz + sz / 2,
        )


# Shape types that are box-compatible leaves
_BOX_TYPES = {"box", "box_sharp"}

# Operations that preserve axis-aligned boxes
_UNION_TYPES = {"union"}
_TRANSFORM_TYPES = {"translate", "scale", "mirror"}


def extract_boxes(
    config: dict[str, Any],
    params: dict[str, Any],
    batch_size: int,
    batch_idx: int = 0,
) -> list[BoxSpec] | None:
    """Attempt to extract box specs from a YAML config without building shapes.

    Walks the shape graph from ``output`` backward. If every leaf is a ``box``
    or ``box_sharp`` and every operation is union / translate / scale / mirror,
    returns a list of :class:`BoxSpec` ready for direct PyVista rendering.
    Otherwise returns ``None`` (caller should fall back to normal SDF meshing).

    Args:
        config: Raw YAML config dict (with ``shapes`` and ``output`` keys).
        params: Parameter dict (``$param`` values, may include tensors).
        batch_size: Batch size from ``config["bounds"]["batch_size"]``.
        batch_idx: Which batch element to extract (default 0).

    Returns:
        List of BoxSpec if the graph is box-only, else None.
    """
    shapes_config = config.get("shapes")
    output = config.get("output")
    if shapes_config is None or output is None:
        return None

    # Resolve output names (single or multi-body)
    if isinstance(output, list):
        output_names = output
    else:
        output_names = [output]

    # Resolve all shape definitions (substitute params + tensorify) once
    resolved: dict[str, dict[str, Any]] = {}
    for name, raw_def in shapes_config.items():
        sdef = deep_copy_config(raw_def)
        # Protect expression key (same as program.py does)
        expr = sdef.pop("expression", None)
        sdef = substitute_params(sdef, params)
        if expr is not None:
            sdef["expression"] = expr
        sdef = tensorify_shape_def(sdef, batch_size=batch_size)
        resolved[name] = sdef

    # Walk from each output shape
    all_boxes: list[BoxSpec] = []
    for oname in output_names:
        result = _walk(oname, resolved, batch_idx)
        if result is None:
            return None
        all_boxes.extend(result)

    return all_boxes if all_boxes else None


def _walk(
    name: str,
    resolved: dict[str, dict[str, Any]],
    batch_idx: int,
) -> list[BoxSpec] | None:
    """Recursively walk one shape node. Returns BoxSpecs or None."""
    if name not in resolved:
        return None

    sdef = resolved[name]
    stype = sdef.get("type")
    if stype is None:
        return None

    # Leaf: box / box_sharp
    if stype in _BOX_TYPES:
        return _extract_box(name, sdef, batch_idx)

    # Union: recurse into children
    if stype in _UNION_TYPES:
        child_names = sdef.get("shapes")
        if not isinstance(child_names, list):
            return None
        boxes: list[BoxSpec] = []
        for cname in child_names:
            result = _walk(cname, resolved, batch_idx)
            if result is None:
                return None
            boxes.extend(result)
        return boxes

    # Translate: shift children
    if stype == "translate":
        child_name = sdef.get("shape")
        offset = sdef.get("offset")
        if child_name is None or offset is None:
            return None
        children = _walk(child_name, resolved, batch_idx)
        if children is None:
            return None
        ox, oy, oz = _tensor_to_xyz(offset, batch_idx)
        return [
            BoxSpec(
                center=(b.center[0] + ox, b.center[1] + oy, b.center[2] + oz),
                size=b.size,
                name=b.name,
            )
            for b in children
        ]

    # Scale: multiply size and center
    if stype == "scale":
        child_name = sdef.get("shape")
        factor = sdef.get("factor")
        if child_name is None or factor is None:
            return None
        children = _walk(child_name, resolved, batch_idx)
        if children is None:
            return None
        fx, fy, fz = _factor_to_xyz(factor, batch_idx)
        return [
            BoxSpec(
                center=(b.center[0] * fx, b.center[1] * fy, b.center[2] * fz),
                size=(b.size[0] * fx, b.size[1] * fy, b.size[2] * fz),
                name=b.name,
            )
            for b in children
        ]

    # Mirror: flip center across axis
    if stype == "mirror":
        child_name = sdef.get("shape")
        plane = sdef.get("plane")
        if child_name is None or plane is None:
            return None
        children = _walk(child_name, resolved, batch_idx)
        if children is None:
            return None
        mirror_offset = sdef.get("offset")
        off_val = (
            _scalar_val(mirror_offset, batch_idx) if mirror_offset is not None else 0.0
        )
        axis_idx = _mirror_axis(plane)
        if axis_idx is None:
            return None
        return [_mirror_box(b, axis_idx, off_val) for b in children]

    # Anything else: not box-compatible
    return None


# Helpers


def _extract_box(
    name: str, sdef: dict[str, Any], batch_idx: int
) -> list[BoxSpec] | None:
    """Extract a BoxSpec from a resolved box shape definition."""
    min_t = sdef.get("min")
    max_t = sdef.get("max")
    size_t = sdef.get("size")
    center_t = sdef.get("center")

    if min_t is not None and max_t is not None:
        # Corner mode
        lo = _tensor_to_xyz(min_t, batch_idx)
        hi = _tensor_to_xyz(max_t, batch_idx)
        cx = (lo[0] + hi[0]) / 2
        cy = (lo[1] + hi[1]) / 2
        cz = (lo[2] + hi[2]) / 2
        sx = hi[0] - lo[0]
        sy = hi[1] - lo[1]
        sz = hi[2] - lo[2]
        return [BoxSpec(center=(cx, cy, cz), size=(sx, sy, sz), name=name)]

    if size_t is not None:
        sx, sy, sz = _tensor_to_xyz(size_t, batch_idx)
        if center_t is not None:
            cx, cy, cz = _tensor_to_xyz(center_t, batch_idx)
        else:
            cx, cy, cz = 0.0, 0.0, 0.0
        return [BoxSpec(center=(cx, cy, cz), size=(sx, sy, sz), name=name)]

    return None


def _tensor_to_xyz(
    t: Tensor | list | tuple, batch_idx: int
) -> tuple[float, float, float]:
    """Extract (x, y, z) from a [B, 3] tensor at the given batch index."""
    if isinstance(t, Tensor):
        return (
            float(t[batch_idx, 0].detach()),
            float(t[batch_idx, 1].detach()),
            float(t[batch_idx, 2].detach()),
        )
    # Fallback for raw lists (shouldn't happen after tensorify, but be safe)
    return (float(t[0]), float(t[1]), float(t[2]))


def _factor_to_xyz(
    t: Tensor | list | tuple, batch_idx: int
) -> tuple[float, float, float]:
    """Extract scale factors as (fx, fy, fz) from a [B, 1] or [B, 3] tensor."""
    if isinstance(t, Tensor):
        t = t.detach()
        if t.shape[1] == 1:
            v = float(t[batch_idx, 0])
            return (v, v, v)
        return (
            float(t[batch_idx, 0]),
            float(t[batch_idx, 1]),
            float(t[batch_idx, 2]),
        )
    v = float(t[0]) if len(t) == 1 else None
    if v is not None:
        return (v, v, v)
    return (float(t[0]), float(t[1]), float(t[2]))


def _scalar_val(t: Tensor | float | int, batch_idx: int) -> float:
    """Extract a scalar from a [B, 1] tensor or plain number."""
    if isinstance(t, Tensor):
        t = t.detach()
        if t.dim() == 2:
            return float(t[batch_idx, 0])
        if t.dim() == 1:
            return float(t[batch_idx])
        return float(t)
    return float(t)


_PLANE_TO_AXIS = {"yz": 0, "xz": 1, "xy": 2}


def _mirror_axis(plane: str) -> int | None:
    """Map mirror plane name to the axis index that gets flipped."""
    return _PLANE_TO_AXIS.get(plane)


def _mirror_box(box: BoxSpec, axis_idx: int, offset: float) -> BoxSpec:
    """Mirror a BoxSpec across an axis at the given offset."""
    c = list(box.center)
    c[axis_idx] = 2 * offset - c[axis_idx]
    return BoxSpec(center=tuple(c), size=box.size, name=box.name)
