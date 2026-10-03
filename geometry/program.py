"""
CADProgram - Main Entry Point for File-Based CAD Programs

The CADProgram class loads CAD definitions from YAML files and provides
a unified interface for mesh extraction and visualization.

Supports both single-body and multi-body output:
- Single body: `output: my_shape` -> returns tensor/tuple directly
- Multi body: `output: [shape1, shape2]` -> returns Dict[str, ...] keyed by name
"""

from __future__ import annotations

import copy
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .measure.rasterize import RasterResult
    from .mesh import MeshTimer
    from .render.field_slice import FieldSlice

import torch
from torch import Tensor

from .core import Shape
from .graph import graph_to_yaml, yaml_to_graph
from .loader import (
    SHAPE_LOADERS,
    _eval_binding_derived,
    _eval_param_expr,
    _extract_binding_params,
    deep_copy_config,
    load_yaml,
    substitute_params,
    validate_config,
    validate_shapes,
)
from .mesh import MeshResult
from .measure.metrics import (
    _compute_surface_area_and_flux_from_mesh,
    _compute_surface_area_from_mesh,
    _compute_volume_from_mesh,
    _is_watertight,
)
from .utils import tensorify_shape_def

if TYPE_CHECKING:
    from torch_geometric.data import Data

    from .measure.isocontour import IsocontourResult
    from .measure.overlap import OverlapResult
    from .measure.structural import SectionProperties
    from .stdlib.curves import Curve2D


def _tensorify_cp(cp_list: list, batch_size: int, device=None) -> Tensor:
    """Convert control point list -> [B, N, 2] tensor, preserving gradients.

    After substitute_params, cp_list is [[val, val], ...] where each val is
    either a [B, 1] tensor (with grad) or a plain float/int.  torch.tensor()
    would detach all gradient tensors; this function stacks them instead.
    """
    rows = []
    for pt in cp_list:
        cols = []
        for val in pt:
            if isinstance(val, Tensor):
                cols.append(val)  # [B, 1] with grad
            else:
                cols.append(torch.full((batch_size, 1), float(val), device=device))
        rows.append(torch.cat(cols, dim=1))  # [B, 2]
    return torch.stack(rows, dim=1)  # [B, N, 2]


def _tensorify_control_net(raw: list, batch_size: int, device=None) -> Tensor:
    """Convert nested list -> [B, 4, 4, 3] tensor, preserving gradients.

    After substitute_params, raw is [4][4][3] containing [B, 1] gradient
    tensors and plain floats.  torch.tensor() would detach all of them.
    """
    rows_u = []
    for u_row in raw:
        rows_v = []
        for pt in u_row:
            cols = []
            for val in pt:
                if isinstance(val, Tensor):
                    cols.append(val)  # [B, 1]
                else:
                    cols.append(torch.full((batch_size, 1), float(val), device=device))
            rows_v.append(torch.cat(cols, dim=1))  # [B, 3]
        rows_u.append(torch.stack(rows_v, dim=1))  # [B, 4, 3]
    return torch.stack(rows_u, dim=1)  # [B, 4, 4, 3]


class CADProgram:
    """
    Loads a CAD program from YAML and provides mesh/visualize methods.

    A CADProgram represents a parametric shape definition. Parameters can be
    overridden using with_params() to create variations, including batched
    tensors for parallel evaluation of multiple variants.

    Supports multi-body output via `output: [shape1, shape2]` syntax.
    Multi-body programs return Dict[str, ...] keyed by body name.

    Attributes:
        config: The raw YAML configuration dictionary
        params: Current parameter values (can be modified via with_params)

    Example:
        >>> # Single body
        >>> prog = CADProgram.load_from_yaml("examples/core/yaml/sphere.yaml")
        >>> verts, faces = prog.mesh(xyz_min=(-2,-2,-2), xyz_max=(2,2,2))
        >>>
        >>> # Multi-body
        >>> prog = CADProgram.load_from_yaml("examples/core/yaml/assembly.yaml")
        >>> meshes = prog.mesh(xyz_min=(-2,-2,-2), xyz_max=(2,2,2))
        >>> # meshes = {"bracket": (verts, faces), "bolt": (verts, faces)}
        >>>
        >>> # Create 100 variants with different radii
        >>> prog_batch = prog.with_params(radius=torch.linspace(0.5, 2.0, 100))
        >>> meshes = prog_batch.mesh(xyz_min=(-3,-3,-3), xyz_max=(3,3,3))
    """

    def __init__(
        self,
        config: dict[str, Any],
        params: dict[str, Any],
        yaml_path: Path | None = None,
        device=None,
    ):
        """
        Initialize a CADProgram with configuration and parameters.

        You typically don't call this directly - use CADProgram.load_from_yaml() instead.

        Args:
            config: Parsed YAML configuration dictionary
            params: Parameter values to use (can be overridden later)
            yaml_path: Path to the YAML file (for relative path resolution in neural primitives)
            device: PyTorch device for tensor creation (None = CPU, default)
        """
        # In order to register all available shapes to `SHAPE_LOADERS`,
        # the corresponding modules have to be executed:
        exec_module("geometry.stdlib.neural_primitives")
        exec_module("geometry.stdlib.primitives")
        exec_module("geometry.stdlib.operations")
        # Validate config has required bounds fields
        validate_config(config)

        self._device = torch.device(device) if device is not None else None

        self.config = config

        # Keep raw params - tensorify happens in _build_single_shape
        self.params = params

        # Store YAML path for neural primitive path resolution
        self._yaml_path = yaml_path
        self._yaml_dir = yaml_path.parent if yaml_path else Path.cwd()

        # Extract and store bounds from config
        bounds = config["bounds"]
        self.xyz_min: tuple[float, float, float] = (
            bounds["x"][0],
            bounds["y"][0],
            bounds["z"][0],
        )
        self.xyz_max: tuple[float, float, float] = (
            bounds["x"][1],
            bounds["y"][1],
            bounds["z"][1],
        )
        self.resolution: int = bounds["resolution"]
        self.batch_size: int = bounds["batch_size"]

        # Single-body output (backwards compatible)
        self._shape: Shape | None = None
        # Multi-body output: Dict[name, Shape]
        self._shapes: dict[str, Shape] | None = None
        # Imported shapes from assembly imports: Dict[ns.name, Shape]
        self._imported_shapes: dict[str, Shape] = {}
        # Retained sub-programs for with_bindings propagation
        self._sub_programs: dict[str, CADProgram] = {}
        # Curves (lazy-built from YAML)
        self._curves: dict | None = None
        # Workplanes (lazy-built from YAML)
        self._workplanes: dict | None = None

    @classmethod
    def load_from_yaml(cls, path: str | Path, device=None) -> CADProgram:
        """
        Load a CAD program from a YAML file.

        The YAML file should have the following structure:
        ```yaml
        name: "my_shape"

        params:
          radius: 1.0
          center_x: 0.0

        shapes:
          my_sphere:
            type: sphere
            radius: $radius
            center: [$center_x, 0, 0]

        output: my_sphere
        ```

        Args:
            path: Path to the YAML file
            device: PyTorch device for tensor creation (None = CPU, default)

        Returns:
            CADProgram instance ready for mesh extraction

        Raises:
            FileNotFoundError: If the YAML file doesn't exist
            yaml.YAMLError: If the YAML is malformed
        """
        path = Path(path)
        config = load_yaml(path)
        params = config.get("params", {})
        # Extract binding defaults and evaluate derived expressions
        binding_params = _extract_binding_params(config)
        params = {**params, **binding_params}
        params = _eval_binding_derived(config, params)
        return cls(config, params, yaml_path=path, device=device)

    @classmethod
    def load_from_graph(
        cls, path_or_data: str | Path | Data, device=None
    ) -> CADProgram:
        """
        Load a CAD program from a PyG graph (.pt file or Data object).

        The graph is converted to YAML config internally, then the normal
        shape building process is used. Note that numeric parameters (radius,
        size, etc.) are NOT preserved in graph format - placeholder values
        are used.

        Args:
            path_or_data: Either:
                - Path to .pt file containing a single Data object
                - PyG Data object directly
            device: PyTorch device for tensor creation (None = CPU, default)

        Returns:
            CADProgram instance ready for mesh extraction

        Raises:
            FileNotFoundError: If .pt file doesn't exist
            ValueError: If graph is invalid

        Example:
            >>> # From file
            >>> prog = CADProgram.load_from_graph("examples/core/graph/sphere.pt")
            >>>
            >>> # From Data object
            >>> from geometry.graph import yaml_to_graph, load_graphs
            >>> graphs = load_graphs("dataset.pt")
            >>> prog = CADProgram.load_from_graph(graphs[0])
        """
        # Handle Data object vs path
        if isinstance(path_or_data, (str, Path)):
            import torch

            path = Path(path_or_data)
            data = torch.load(path, weights_only=False)
            # Handle both single Data and list of Data
            if isinstance(data, list):
                if len(data) == 0:
                    raise ValueError(f"Empty graph list in {path}")
                data = data[0]
        else:
            data = path_or_data

        # Convert graph to YAML config
        config = graph_to_yaml(data)

        # Validate shapes before adding bounds (catches invalid GNN output)
        validate_shapes(config)

        # Add default bounds (required by CADProgram)
        if "bounds" not in config:
            config["bounds"] = {
                "x": [-2.0, 2.0],
                "y": [-2.0, 2.0],
                "z": [-2.0, 2.0],
                "resolution": 64,
                "batch_size": 1,
            }

        return cls(config, config.get("params", {}), device=device)

    def save_mesh_image(
        self,
        mesh_result: MeshResult | dict[str, MeshResult],
        path: str,
        view: str = "iso",
        format: str = "png",
        colors: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        """
        Save mesh visualization to image file.

        Args:
            mesh_result: MeshResult (single-body) or dict[str, MeshResult] (multi-body)
            path: Output file path (extension added if not present)
            view: Camera view - one of:
                - 'iso': isometric (default)
                - 'front': looking along -X, views YZ plane
                - 'back': looking along +X, views YZ plane
                - 'left': looking along -Y, views XZ plane
                - 'right': looking along +Y, views XZ plane
                - 'top': looking along -Z, views XY plane
                - 'bottom': looking along +Z, views XY plane
            format: Output format ('png', 'jpg', 'pdf')
            colors: Color dict for multi-body (default: auto-assign from palette)
            **kwargs: Additional arguments passed to the render backend.
                backend: 'pyvista' (default) or 'blender'.
                samples: Blender Cycles samples (default: 128).
                blender_path: Explicit Blender binary path (default: auto-detect).

        Example:
            >>> prog = CADProgram.load_from_yaml("examples/core/yaml/sphere.yaml")
            >>> mesh = prog.mesh(backend='skimage')
            >>> prog.save_mesh_image(mesh, "sphere_iso.png", view='iso')
            >>> # Blender backend:
            >>> prog.save_mesh_image(mesh, "fig.png", backend='blender')
        """
        from .render import save_mesh_image as _save_mesh_image

        _save_mesh_image(
            mesh_result, path, view=view, format=format, colors=colors, **kwargs
        )

    def with_params(self, **overrides) -> CADProgram:
        """
        Create a new CADProgram with overridden parameters.

        This returns a new instance - the original is not modified.
        Supports batched tensors for parallel evaluation.

        Namespaced overrides (e.g. ``ns.param_name``) are delegated to
        sub-programs when this is an assembly with retained sub-programs.

        Args:
            **overrides: Parameter values to override.
                        Python scalars/lists are auto-converted to [batch_size, K] format.
                        Tensors must already be in [B, K] format.
                        If any tensor has B > 1, all other params will be expanded to match.

        Returns:
            New CADProgram with updated parameters

        Example:
            >>> prog = CADProgram.load_from_yaml("sphere.yaml")
            >>> # Override single parameter (auto-converted to [1, 1])
            >>> prog2 = prog.with_params(radius=2.0)
            >>> # Batched parameters for parallel evaluation (must use .unsqueeze(1))
            >>> prog_batch = prog.with_params(radius=torch.linspace(0.5, 2.0, 100).unsqueeze(1))
        """
        # Split namespaced vs local overrides
        ns_overrides: dict[str, dict[str, Any]] = {}
        local_overrides: dict[str, Any] = {}
        for name, value in overrides.items():
            if "." in name and name.split(".", 1)[0] in self._sub_programs:
                ns, param_name = name.split(".", 1)
                ns_overrides.setdefault(ns, {})[param_name] = value
            else:
                local_overrides[name] = value

        new_params = {**self.params, **local_overrides}

        # Detect effective batch size from tensor overrides
        effective_batch_size = self.batch_size  # Start with YAML batch_size
        for name, value in local_overrides.items():
            if isinstance(value, Tensor):
                if value.dim() >= 1:
                    param_batch = value.shape[0]
                    if param_batch > 1:
                        if (
                            effective_batch_size > 1
                            and param_batch != effective_batch_size
                        ):
                            raise ValueError(
                                f"Batch size mismatch in with_params: "
                                f"'{name}' has B={param_batch} but expected B={effective_batch_size}. "
                                f"All batched parameters must have the same batch size."
                            )
                        effective_batch_size = param_batch

        # Create modified config with effective batch size
        if effective_batch_size != self.batch_size:
            new_config = copy.deepcopy(self.config)
            new_config["bounds"]["batch_size"] = effective_batch_size
            new_prog = CADProgram(
                new_config, new_params, yaml_path=self._yaml_path, device=self._device
            )
        else:
            new_prog = CADProgram(
                self.config, new_params, yaml_path=self._yaml_path, device=self._device
            )

        # Propagate namespaced overrides to sub-programs
        updated_subs = dict(self._sub_programs)
        for ns, group in ns_overrides.items():
            updated_subs[ns] = self._sub_programs[ns].with_params(**group)
        new_prog._sub_programs = updated_subs

        return new_prog

    def with_bindings(self, **tensors) -> CADProgram:
        """Unpack named tensors into params using YAML-declared bindings.

        Each keyword argument maps a binding name (declared in the YAML
        ``bindings:`` section) to a ``[B, K]`` tensor.  Columns are unpacked
        according to the binding's ``map:``, and ``derived:`` expressions are
        evaluated afterwards.

        For assemblies, namespaced bindings (e.g. ``ns.binding_name``) are
        delegated to the corresponding sub-program's ``with_bindings()``.

        Args:
            **tensors: ``name=tensor`` pairs where *name* matches a key in
                the YAML ``bindings:`` section (or ``ns.binding`` for
                assembly sub-programs) and *tensor* is ``[B, K]``.

        Returns:
            New CADProgram with the unpacked parameters applied (via
            ``with_params``).

        Raises:
            KeyError: If a binding name is not declared in the YAML or
                a namespace is not found.
            ValueError: If a tensor's shape doesn't match the declared shape.
        """
        # Split namespaced vs local bindings
        ns_groups: dict[str, dict[str, Tensor]] = {}
        local_tensors: dict[str, Tensor] = {}
        for name, tensor in tensors.items():
            if "." in name and name.split(".", 1)[0] in self._sub_programs:
                ns, bind_name = name.split(".", 1)
                ns_groups.setdefault(ns, {})[bind_name] = tensor
            else:
                local_tensors[name] = tensor

        # Process local bindings (existing logic)
        bindings = self.config.get("bindings", {})
        overrides: dict[str, Any] = {}

        for name, tensor in local_tensors.items():
            if name not in bindings:
                raise KeyError(
                    f"Binding '{name}' not declared in YAML. "
                    f"Available bindings: {sorted(bindings.keys())}"
                )
            spec = bindings[name]
            expected_cols = spec["shape"][1]
            if tensor.dim() != 2 or tensor.shape[1] != expected_cols:
                raise ValueError(
                    f"Binding '{name}' expects shape [B, {expected_cols}], "
                    f"got {list(tensor.shape)}"
                )

            # Unpack mapped columns -> [B, 1] or [B, K] slices
            for col_key, entry in spec.get("map", {}).items():
                param_name = next(iter(entry.keys()))
                col_str = str(col_key)
                if ":" in col_str:
                    start, end = map(int, col_str.split(":"))
                    overrides[param_name] = tensor[:, start:end]
                else:
                    idx = int(col_str)
                    overrides[param_name] = tensor[:, idx : idx + 1]

        # Merge mapped overrides with current params for derived eval
        merged = {**self.params, **overrides}

        # Evaluate derived expressions (in YAML order, can chain)
        for name, tensor in local_tensors.items():
            spec = bindings[name]
            for param_name, expr in spec.get("derived", {}).items():
                overrides[param_name] = _eval_param_expr(expr, merged)
                merged[param_name] = overrides[param_name]

        # Delegate namespaced bindings to sub-programs
        updated_subs = dict(self._sub_programs)
        for ns, group in ns_groups.items():
            if ns not in self._sub_programs:
                raise KeyError(
                    f"Namespace '{ns}' not found. "
                    f"Available: {sorted(self._sub_programs)}"
                )
            updated_subs[ns] = self._sub_programs[ns].with_bindings(**group)

        # Build new program with local overrides
        new_prog = (
            self.with_params(**overrides)
            if overrides
            else CADProgram(
                self.config, self.params, yaml_path=self._yaml_path, device=self._device
            )
        )
        new_prog._sub_programs = updated_subs
        return new_prog

    @property
    def is_multi_body(self) -> bool:
        """Check if this program outputs multiple bodies."""
        output = self.config.get("output")
        return isinstance(output, (list, dict))

    @property
    def body_names(self) -> list[str]:
        """Get list of output body names."""
        output = self.config.get("output")
        if isinstance(output, dict):
            return list(output.keys())
        if isinstance(output, list):
            return output
        return [output] if output else []

    def _build_single_shape(
        self, shape_name: str, shapes_config: dict[str, Any]
    ) -> Shape:
        """
        Build a single Shape from its definition.

        Args:
            shape_name: Name of the shape in shapes_config
            shapes_config: The shapes section of the YAML config

        Returns:
            Shape object ready for evaluation
        """
        # Check imported shapes first (dot-notation refs like 'fus.shell')
        if hasattr(self, "_imported_shapes") and shape_name in self._imported_shapes:
            return self._imported_shapes[shape_name]

        if shape_name not in shapes_config:
            available = list(shapes_config.keys())
            if hasattr(self, "_imported_shapes"):
                available.extend(sorted(self._imported_shapes.keys()))
            raise ValueError(
                f"Shape '{shape_name}' not found. Available shapes: {available}"
            )

        # Get shape definition and substitute parameters
        shape_def = deep_copy_config(shapes_config[shape_name])
        # Protect 'expression' key from param substitution, it contains
        # SDF runtime variables (x, y, z) that _eval_param_expr can't handle.
        _expression = shape_def.pop("expression", None)
        shape_def = substitute_params(shape_def, self.params)
        if _expression is not None:
            shape_def["expression"] = _expression

        # Convert all numeric literals to tensors (pure-tensor architecture)
        # Pass batch_size so scalars/lists get converted to [batch_size, K] tensors
        shape_def = tensorify_shape_def(
            shape_def, batch_size=self.batch_size, device=self._device
        )

        # Extract type and dispatch to primitive factory
        shape_type = shape_def.pop("type")

        loader = SHAPE_LOADERS.get(shape_type)
        if loader is not None:

            def lookup_shape(name: str) -> Shape:
                return self._build_single_shape(name, shapes_config)

            def lookup_curve(name: str) -> "Curve2D":
                curves = self._build_curves()
                try:
                    curve_obj, _plane_str, _color = curves[name]
                except KeyError:
                    available = list(curves.keys())
                    raise ValueError(f"Available curves: {available}") from None
                return curve_obj

            injectable = {
                "batch_size": self.batch_size,
                "yaml_dir": self._yaml_dir,
                "device": self._device,
            }
            inject = {key: injectable[key] for key in loader.inject}
            return loader.load({**shape_def, **inject}, lookup_shape, lookup_curve)

        if shape_type == "bezier_surface":
            from .stdlib import surface

            flip = shape_def.get("flip", False)
            if isinstance(flip, Tensor):
                flip = bool(flip)

            # Control-net mode: explicit 4x4x3 control points
            control_net_raw = shape_def.get("control_net")
            if control_net_raw is not None:
                net = _tensorify_control_net(
                    control_net_raw, self.batch_size, device=self._device
                )
                if net.dim() != 4 or net.shape[1:] != (4, 4, 3):
                    raise ValueError(
                        f"bezier_surface control_net must be 4x4x3 "
                        f"(got shape {list(net.shape)}). Example:\n"
                        "  control_net:\n"
                        "    - [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]\n"
                        "    - [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]\n"
                        "    - [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]\n"
                        "    - [[x,y,z], [x,y,z], [x,y,z], [x,y,z]]"
                    )
                return surface.bezier_surface(
                    control_net=net,
                    flip=flip,
                )

            # Curve-based mode: two curves + r1/r2
            curve1_name = shape_def.get("curve1")
            curve2_name = shape_def.get("curve2")
            if curve1_name is None or curve2_name is None:
                raise ValueError(
                    "bezier_surface requires either 'control_net' or "
                    "'curve1'+'curve2' fields. Examples:\n"
                    "  # Curve mode:\n"
                    "  type: bezier_surface\n"
                    "  curve1: bottom_curve\n  curve2: top_curve\n"
                    "  r1: 1.0\n  r2: 1.0\n\n"
                    "  # Control-net mode:\n"
                    "  type: bezier_surface\n"
                    "  control_net:\n"
                    "    - [[x,y,z], ...]\n    - ..."
                )

            curves = self._build_curves()
            for cname in (curve1_name, curve2_name):
                if cname not in curves:
                    available = list(curves.keys())
                    raise ValueError(
                        f"bezier_surface references curve '{cname}' "
                        f"which was not found. Available curves: {available}"
                    )

            curve1_obj, _plane1, _color1 = curves[curve1_name]
            curve2_obj, _plane2, _color2 = curves[curve2_name]

            surface_params = {}
            for key in ("r1", "r2"):
                val = shape_def.get(key)
                if val is None:
                    raise ValueError(
                        f"bezier_surface requires '{key}' parameter. "
                        f"Example: {key}: 1.0"
                    )
                surface_params[key] = val

            return surface.bezier_surface(
                curve1_obj,
                curve2_obj,
                r1=surface_params["r1"],
                r2=surface_params["r2"],
                flip=flip,
            )

        supported = [
            # Surface operations
            "bezier_surface",
            *SHAPE_LOADERS.keys(),
        ]
        raise ValueError(
            f"Unknown shape type: '{shape_type}'. Supported types: {supported}"
        )

    def _resolve_imports(self) -> dict[str, Shape]:
        """
        Load imported sub-programs and return dict of ns.name -> Shape.

        Each import entry is loaded as a full CADProgram. The assembly's
        batch_size is forced onto sub-programs so tensor dimensions match
        when shapes are composed.

        If sub-programs were already injected (by with_bindings/with_params),
        they are reused instead of re-loading from disk.

        Only shapes listed in the sub-program's output are exposed.

        Returns:
            Dict mapping 'namespace.shape_name' -> pre-built Shape objects
        """
        # Reuse sub-programs injected by with_bindings/with_params
        if self._sub_programs:
            imported: dict[str, Shape] = {}
            for ns, sub_prog in self._sub_programs.items():
                if sub_prog._shape is None and sub_prog._shapes is None:
                    sub_prog._build_shapes()
                sub_output = sub_prog.config.get("output")
                if sub_prog._shapes is not None:
                    for name, shape in sub_prog._shapes.items():
                        imported[f"{ns}.{name}"] = shape
                elif sub_prog._shape is not None:
                    if isinstance(sub_output, str):
                        imported[f"{ns}.{sub_output}"] = sub_prog._shape
                    else:
                        imported[f"{ns}.output"] = sub_prog._shape
            return imported

        imports = self.config.get("import")
        if not imports:
            return {}

        imported = {}

        for ns, spec in imports.items():
            if isinstance(spec, str):
                file_path = spec
                param_overrides = {}
            else:
                file_path = spec["file"]
                param_overrides = spec.get("params", {})

            # Resolve relative to this YAML's directory
            full_path = self._yaml_dir / file_path

            if not full_path.exists():
                raise ValueError(
                    f"Import '{ns}' references '{file_path}' but file not found "
                    f"at {full_path}"
                )

            # Load sub-program config and override batch_size
            sub_config = load_yaml(full_path)
            if "bounds" in sub_config:
                sub_config["bounds"]["batch_size"] = self.batch_size

            # Extract params with bindings
            sub_params = sub_config.get("params", {})
            sub_binding_params = _extract_binding_params(sub_config)
            sub_params = {**sub_params, **sub_binding_params}
            sub_params = _eval_binding_derived(sub_config, sub_params)

            # Apply param overrides from assembly
            if param_overrides:
                sub_params = {**sub_params, **param_overrides}

            sub_prog = CADProgram(
                sub_config,
                sub_params,
                yaml_path=full_path,
                device=self._device,
            )

            # Build shapes in sub-program
            sub_prog._build_shapes()

            # Retain sub-program for with_bindings propagation
            self._sub_programs[ns] = sub_prog

            # Extract exported shapes under namespace
            sub_output = sub_config.get("output")
            if sub_prog._shapes is not None:
                # Multi-body: all output shapes
                for name, shape in sub_prog._shapes.items():
                    imported[f"{ns}.{name}"] = shape
            elif sub_prog._shape is not None:
                # Single-body: use the output name
                if isinstance(sub_output, str):
                    imported[f"{ns}.{sub_output}"] = sub_prog._shape
                else:
                    imported[f"{ns}.output"] = sub_prog._shape

        return imported

    def _build_shapes(self) -> None:
        """
        Build Shape object(s) from config with current parameters.

        This method is called lazily when mesh() or visualize() is called.
        Shapes are cached until parameters change (via with_params).

        For single-body output, sets self._shape.
        For multi-body output, sets self._shapes dict.
        For dict output, keys are export names, values are shape refs.
        """
        # Already built
        if self._shape is not None or self._shapes is not None:
            return

        # Resolve imports first (populates self._imported_shapes)
        self._imported_shapes = self._resolve_imports()

        shapes_config = self.config.get("shapes", {})
        output = self.config.get("output")

        if not output:
            raise ValueError("YAML config must specify 'output' field")

        if isinstance(output, dict):
            # Dict output: {export_name: shape_ref}
            self._shapes = {}
            for export_name, shape_ref in output.items():
                self._shapes[export_name] = self._build_single_shape(
                    shape_ref, shapes_config
                )
        elif isinstance(output, list):
            # Multi-body output (list of shape refs)
            self._shapes = {}
            for name in output:
                self._shapes[name] = self._build_single_shape(name, shapes_config)
        else:
            # Single-body output (backwards compatible)
            self._shape = self._build_single_shape(output, shapes_config)

    def _build_curves(self) -> dict[str, tuple]:
        """
        Parse the 'curves:' section from YAML config.

        Each curve definition has:
            - control_points: list of 4 [x, y] pairs
            - workplane: name referencing workplanes: section (new format)
              OR plane: 'xy' | 'xz' | 'yz' (backward compat format)
            - color: optional, default 'blue'

        The curve object carries its workplane internally. The tuple still
        stores plane_str for visualization backward compat (to_polyline_3d needs it).

        Returns:
            Dict[str, Tuple[CubicBezier2D, str, str]] mapping
            name -> (curve, plane_str, color)
        """
        if self._curves is not None:
            return self._curves

        from .stdlib.curves import CubicBezier2D

        # Ensure workplanes are built first (curves may reference them)
        workplanes = self._build_workplanes()

        curves_config = self.config.get("curves", {})
        if not curves_config:
            self._curves = {}
            return self._curves

        self._curves = {}
        for name, curve_def in curves_config.items():
            curve_def = substitute_params(curve_def, self.params)
            if "control_points" not in curve_def:
                raise ValueError(
                    f"Curve '{name}' missing required 'control_points' field. "
                    f"Expected list of 4 [x, y] pairs."
                )

            cp_list = curve_def["control_points"]
            color = curve_def.get("color", "blue")

            # Convert control points to [1, 4, 2] tensor
            cp_tensor = _tensorify_cp(
                cp_list, self.batch_size, device=self._device
            )  # [B, 4, 2]

            has_workplane = "workplane" in curve_def
            has_plane = "plane" in curve_def

            if has_workplane and has_plane:
                raise ValueError(
                    f"Curve '{name}' has both 'workplane' and 'plane'. "
                    f"Use one or the other."
                )

            if has_workplane:
                # New format: reference named workplane
                wp_name = curve_def["workplane"]
                if wp_name not in workplanes:
                    available = list(workplanes.keys())
                    raise ValueError(
                        f"Curve '{name}' references workplane '{wp_name}' "
                        f"which was not found. Available workplanes: {available}"
                    )
                wp, _wp_color = workplanes[wp_name]
                curve = CubicBezier2D(cp_tensor, workplane=wp)

                # Derive plane_str from workplane definition for visualization
                wp_def = self.config["workplanes"][wp_name]
                plane_str = wp_def["base"]

            elif has_plane:
                # Backward compat format: simple plane string
                plane_str = curve_def["plane"]
                curve = CubicBezier2D(cp_tensor, workplane=plane_str)

            else:
                raise ValueError(
                    f"Curve '{name}' missing required 'workplane' or 'plane' field. "
                    f"Use 'workplane: wp_name' to reference a named workplane, "
                    f"or 'plane: xz' for a simple axis-aligned plane."
                )

            self._curves[name] = (curve, plane_str, color)

        return self._curves

    def _prepare_curves_data(self) -> list[tuple]:
        """
        Prepare curve data for rendering: polylines + control points in 3D.

        Returns:
            List of (polyline_np, cp_np, color, name) tuples where:
                polyline_np: (S, 3) numpy array of sampled curve points
                cp_np: (N_cp, 3) numpy array of control points
        """
        curves = self._build_curves()
        curves_data = []
        for name, (curve, _plane, color) in curves.items():
            # Use stored workplane (handles offset/rotation) when available
            pts_3d = curve.to_polyline_3d()  # (1, S, 3)
            polyline_np = pts_3d[0].detach().cpu().numpy()  # (S, 3)

            cp_3d = curve.control_points_3d()  # (1, N_cp, 3)
            cp_np = cp_3d[0].detach().cpu().numpy()  # (N_cp, 3)

            curves_data.append((polyline_np, cp_np, color, name))
        return curves_data

    def _build_workplanes(self) -> dict[str, tuple]:
        """
        Parse the 'workplanes:' section from YAML config.

        Each workplane definition has:
            - base: 'xy' | 'xz' | 'yz' (required)
            - offset: float (default 0)
            - rotate: float in degrees (default 0)
            - rotate_axis: 'x' | 'y' | 'z' (required if rotate != 0)
            - color: str (default 'yellow')

        Returns:
            Dict[str, Tuple[Workplane, str]] mapping name -> (workplane, color)
        """
        if self._workplanes is not None:
            return self._workplanes

        from .stdlib.workplane import Workplane

        wp_config = self.config.get("workplanes", {})
        if not wp_config:
            self._workplanes = {}
            return self._workplanes

        self._workplanes = {}
        for name, wp_def in wp_config.items():
            wp_def = substitute_params(wp_def, self.params)
            wp_def = tensorify_shape_def(
                wp_def, batch_size=self.batch_size, device=self._device
            )
            if "base" not in wp_def:
                raise ValueError(
                    f"Workplane '{name}' missing required 'base' field. "
                    f"Must be one of: 'xy', 'xz', 'yz'."
                )

            base = wp_def["base"]
            offset = wp_def.get(
                "offset", torch.zeros(self.batch_size, 1, device=self._device)
            )
            rotate = wp_def.get(
                "rotate", torch.zeros(self.batch_size, 1, device=self._device)
            )
            rotate_axis = wp_def.get("rotate_axis", None)
            color = wp_def.get("color", "yellow")

            wp = Workplane.from_base(
                base=base,
                offset=offset,
                rotate=rotate,
                rotate_axis=rotate_axis,
            )
            self._workplanes[name] = (wp, color)

        return self._workplanes

    def _prepare_workplanes_data(self) -> list[tuple]:
        """
        Prepare workplane data for rendering.

        Returns:
            List of (origin_np, normal_np, u_np, v_np, color, name) tuples
        """
        workplanes = self._build_workplanes()
        data = []
        for name, (wp, color) in workplanes.items():
            origin_np = wp.origin[0].detach().cpu().numpy()
            normal_np = wp.normal[0].detach().cpu().numpy()
            u_np = wp.u[0].detach().cpu().numpy()
            v_np = wp.v[0].detach().cpu().numpy()
            data.append((origin_np, normal_np, u_np, v_np, color, name))
        return data

    def mesh(
        self,
        backend: str | None = None,
        timer: "MeshTimer | None" = None,
    ):
        """
        Extract mesh from the CAD program.

        Bounds and resolution are specified in the YAML file's 'bounds' section.

        Args:
            backend: 'skimage', 'diso-mc', or 'diso-dmc' (REQUIRED)

        Returns:
            Single body: MeshResult
            Multi-body: Dict[str, MeshResult] keyed by body name

        Raises:
            ValueError: If backend not specified
        """
        if backend is None:
            raise ValueError(
                "backend parameter is required. "
                "Use backend='skimage', 'diso-mc', or 'diso-dmc'."
            )

        self._build_shapes()

        if self._shapes is not None:
            # Multi-body output
            return {
                name: shape.get_mesh(
                    self.xyz_min,
                    self.xyz_max,
                    self.resolution,
                    backend,
                    device=self._device,
                    timer=timer,
                )
                for name, shape in self._shapes.items()
            }
        # Single-body output
        return self._shape.get_mesh(
            self.xyz_min,
            self.xyz_max,
            self.resolution,
            backend,
            device=self._device,
            timer=timer,
        )

    def visualize(
        self,
        mesh_or_meshes: MeshResult | dict[str, MeshResult] | None = None,
        origin: bool = True,
        show_edges: bool = False,
        color: str | None = None,
        colors: dict[str, str] | None = None,
        show_bounds: bool = True,
        show_curves: bool = False,
        show_workplanes: bool = False,
        sdf_slice: str | None = None,
        raster_slice: str | None = None,
        raster_epsilon: float = 0.01,
        raster_resolution: int | None = None,
        field_slices: "list[FieldSlice] | FieldSlice | None" = None,
        isocontour: list | None = None,
        isocontour_normals: bool = False,
        batch_idx: int = 0,
    ):
        """
        Visualize pre-computed mesh(es) and/or interactive SDF slice using PyVista.

        **Box fast-path**: If ``mesh_or_meshes`` is None and no overlays are
        requested, the config is checked for box-only composition (box/box_sharp
        primitives under union, translate, scale, mirror). When detected, boxes
        are rendered directly with ``pv.Box()``, no ``mesh()`` call needed::

            prog = CADProgram.load_from_yaml("buildings.yaml")
            prog.visualize()  # instant if all shapes are boxes

        Args:
            mesh_or_meshes: Either MeshResult (single-body), Dict[str, MeshResult] (multi-body),
                           or None (box fast-path / slice-only mode). Default: None.
            origin: Show origin reference planes (default: True)
            show_edges: Show wireframe edges (default: False)
            color: Mesh color for single-body (default: 'lightgrey')
            colors: Color dict for multi-body (default: auto-assign from palette)
            show_bounds: Show bounding box wireframe (default: True)
            show_curves: Show YAML-defined curves and control points (default: False)
            show_workplanes: Show YAML-defined workplanes with normal/axis arrows (default: False)
            sdf_slice: Interactive SDF slice plane - 'xy', 'xz', 'yz', or None (default: None).
                       Adds a draggable plane widget that shows SDF values as a heatmap with
                       isolines and a bold zero-level contour.
            raster_slice: Interactive B/W raster occupancy slice - 'xy', 'xz', 'yz', or None
                       (default: None). Adds a draggable plane widget showing sigmoid occupancy
                       as black (inside) on white (outside).
            raster_epsilon: Sigmoid sharpness for raster_slice (default: 0.01). Use a very
                       small value (e.g. 1e-8) for hard black/white pixels.
            raster_resolution: Grid resolution for raster_slice (default: None -> 5 * bounds resolution).
            field_slices: Pre-computed 2D scalar field(s) to overlay on the scene. Pass a single
                       FieldSlice or a list of them. Each is rendered as a static colored grid
                       at its specified plane and offset. Useful for visualizing external simulation
                       results (wind, stress, temperature) alongside the CAD geometry.
            isocontour: List of IsocontourResult from prog.isocontour() to overlay as 3D
                        polylines (default: None). Each contour is drawn as a closed line in
                        3D at its station position.
            isocontour_normals: If True, show outward normal arrows on isocontour lines
                        (default: False). Requires isocontour to be provided.
            batch_idx: Which batch element to use for SDF/raster slice and isocontour display (default: 0)

        Returns:
            PyVista plotter object

        Example:
            # Single-body with mesh
            mesh = prog.mesh(backend='skimage')
            prog.visualize(mesh, origin=True, color='red')

            # Mesh + interactive SDF slice
            mesh = prog.mesh(backend='skimage')
            prog.visualize(mesh, sdf_slice='xy')

            # Slice only (no mesh needed)
            prog.visualize(sdf_slice='xy')

            # Mesh + isocontour overlay
            mesh = prog.mesh(backend='skimage')
            contours = prog.isocontour(plane='xz', stations=[0.0, 0.5, 1.0], normal_mode='in_plane')
            prog.visualize(mesh, isocontour=contours, isocontour_normals=True)

            # Multi-body
            meshes = prog.mesh(backend='skimage')
            prog.visualize(meshes, colors={'bracket': 'blue', 'bolt': 'green'})

            # With batch selection
            prog.visualize(sdf_slice='xz', batch_idx=2)
        """
        from .render.pyvista import (
            _add_sdf_slice_widget,
            _create_plotter,
            render_mesh_result,
            render_meshes,
        )

        if isocontour_normals and isocontour is None:
            raise ValueError(
                "isocontour_normals=True requires isocontour results.\n\n"
                "Usage:\n"
                "  contours = prog.isocontour(plane='xz', stations=[0.0], normal_mode='in_plane')\n"
                "  prog.visualize(mesh, isocontour=contours, isocontour_normals=True)"
            )

        has_overlay = (
            sdf_slice is not None
            or raster_slice is not None
            or isocontour is not None
            or field_slices is not None
        )
        if mesh_or_meshes is None and not has_overlay:
            # Box fast-path: if the config is box-only, render directly
            # with pv.Box(), no SDF evaluation or meshing needed.
            from .render.box_detect import extract_boxes

            boxes = extract_boxes(
                self.config, self.params, self.batch_size, batch_idx=batch_idx
            )
            if boxes is not None:
                from .render.pyvista import render_boxes

                return render_boxes(
                    boxes,
                    show=True,
                    origin=origin,
                    show_bounds=show_bounds,
                    xyz_min=self.xyz_min,
                    xyz_max=self.xyz_max,
                    color=color,
                    colors=colors,
                    show_edges=show_edges,
                )

            raise ValueError(
                "visualize() requires a mesh for non-box shapes.\n\n"
                "Usage:\n"
                "  mesh = prog.mesh(backend='skimage')\n"
                "  prog.visualize(mesh)                     # mesh only\n"
                "  prog.visualize(mesh, sdf_slice='xy')     # mesh + SDF slice\n"
                "  prog.visualize(raster_slice='xy')        # B/W raster only\n"
                "  prog.visualize(field_slices=my_field)     # field overlay only\n"
                "  contours = prog.isocontour(plane='xz', stations=[0.0], normal_mode='in_plane')\n"
                "  prog.visualize(mesh, isocontour=contours)  # mesh + contours\n\n"
                "Tip: If your config uses only box/box_sharp primitives (with union,\n"
                "translate, scale, mirror), prog.visualize() works without mesh()."
            )

        # Prepare curve data if requested
        curves_data = self._prepare_curves_data() if show_curves else None
        # Prepare workplane data if requested
        workplanes_data = self._prepare_workplanes_data() if show_workplanes else None

        plotter = None
        has_slice = (
            sdf_slice is not None
            or raster_slice is not None
            or field_slices is not None
        )
        mesh_opacity = 0.3 if has_slice else 1.0

        # Type detection and routing for mesh
        if mesh_or_meshes is not None:
            if isinstance(mesh_or_meshes, dict):
                # Multi-body: Dict[str, MeshResult]
                for name, mesh in mesh_or_meshes.items():
                    if not isinstance(mesh, MeshResult):
                        raise TypeError(
                            f"Multi-body dict expects Dict[str, MeshResult], "
                            f"but '{name}' has type {type(mesh).__name__}"
                        )
                show = not has_slice  # don't show yet if adding slice
                plotter = render_meshes(
                    mesh_or_meshes,
                    show=show,
                    origin=origin,
                    show_edges=show_edges,
                    colors=colors,
                    show_bounds=show_bounds,
                    curves_data=curves_data,
                    workplanes_data=workplanes_data,
                    opacity=mesh_opacity,
                )

            elif isinstance(mesh_or_meshes, MeshResult):
                # Single-body: MeshResult
                mesh_color = color or "lightgrey"
                show = not has_slice  # don't show yet if adding slice
                plotter = render_mesh_result(
                    mesh_or_meshes,
                    show=show,
                    origin=origin,
                    show_edges=show_edges,
                    color=mesh_color,
                    show_bounds=show_bounds,
                    curves_data=curves_data,
                    workplanes_data=workplanes_data,
                    opacity=mesh_opacity,
                )

            elif isinstance(mesh_or_meshes, tuple):
                raise TypeError(
                    "visualize() expects MeshResult, not raw (verts, faces) tuple.\n\n"
                    "Correct usage:\n"
                    "  mesh = prog.mesh(backend='skimage')\n"
                    "  prog.visualize(mesh)"
                )

            elif isinstance(mesh_or_meshes, list):
                raise TypeError(
                    "visualize() does not support batched meshes (List[MeshResult]).\n\n"
                    "To visualize from a batch:\n"
                    "  meshes = prog.mesh(...)  # Returns List[MeshResult]\n"
                    "  prog.visualize(meshes[0])  # Visualize specific item"
                )

            else:
                raise TypeError(
                    f"visualize() expects MeshResult or Dict[str, MeshResult], "
                    f"got {type(mesh_or_meshes).__name__}.\n\n"
                    f"Correct usage:\n"
                    f"  mesh = prog.mesh(backend='skimage')\n"
                    f"  prog.visualize(mesh)"
                )

        # Add SDF slice widget if requested
        if sdf_slice is not None:
            self._build_shapes()

            # Get SDF function (via Shape.__call__ which handles [N,3] -> [1,N,3])
            if self._shapes is not None:
                first_name = list(self._shapes.keys())[0]
                if len(self._shapes) > 1:
                    print(
                        f"Warning: Multi-body output. Using '{first_name}' for SDF slice."
                    )
                sdf_fn = self._shapes[first_name]
            else:
                sdf_fn = self._shape

            slice_resolution = 5 * self.resolution

            # Create fresh plotter if no mesh was provided
            if plotter is None:
                plotter = _create_plotter()
                plotter.add_axes()
                if origin:
                    from .render.pyvista import _add_bounding_box, _add_origin_reference

                    bounds = (
                        self.xyz_min[0],
                        self.xyz_max[0],
                        self.xyz_min[1],
                        self.xyz_max[1],
                        self.xyz_min[2],
                        self.xyz_max[2],
                    )
                    _add_origin_reference(plotter, bounds)
                    if show_bounds:
                        _add_bounding_box(
                            plotter,
                            self.xyz_min,
                            self.xyz_max,
                            color="black",
                            line_width=0.5,
                        )

            _add_sdf_slice_widget(
                plotter,
                sdf_fn,
                self.xyz_min,
                self.xyz_max,
                plane=sdf_slice,
                slice_resolution=slice_resolution,
                batch_idx=batch_idx,
            )

        # Add raster occupancy slice widget if requested
        if raster_slice is not None:
            from .render.pyvista import _add_raster_slice_widget

            self._build_shapes()

            if self._shapes is not None:
                first_name = list(self._shapes.keys())[0]
                if len(self._shapes) > 1:
                    print(
                        f"Warning: Multi-body output. Using '{first_name}' for raster slice."
                    )
                sdf_fn = self._shapes[first_name]
            else:
                sdf_fn = self._shape

            slice_resolution = (
                raster_resolution
                if raster_resolution is not None
                else 5 * self.resolution
            )

            if plotter is None:
                plotter = _create_plotter()
                plotter.add_axes()
                if origin:
                    from .render.pyvista import _add_bounding_box, _add_origin_reference

                    bounds = (
                        self.xyz_min[0],
                        self.xyz_max[0],
                        self.xyz_min[1],
                        self.xyz_max[1],
                        self.xyz_min[2],
                        self.xyz_max[2],
                    )
                    _add_origin_reference(plotter, bounds)
                    if show_bounds:
                        _add_bounding_box(
                            plotter,
                            self.xyz_min,
                            self.xyz_max,
                            color="black",
                            line_width=0.5,
                        )

            _add_raster_slice_widget(
                plotter,
                sdf_fn,
                self.xyz_min,
                self.xyz_max,
                plane=raster_slice,
                slice_resolution=slice_resolution,
                batch_idx=batch_idx,
                epsilon=raster_epsilon,
            )

        # Add pre-computed field slices if provided
        if field_slices is not None:
            from .render.field_slice import FieldSlice
            from .render.pyvista import _add_field_slice

            # Normalize to list
            if isinstance(field_slices, FieldSlice):
                field_slices = [field_slices]

            if plotter is None:
                plotter = _create_plotter()
                plotter.add_axes()
                if origin:
                    from .render.pyvista import _add_bounding_box, _add_origin_reference

                    bounds = (
                        self.xyz_min[0],
                        self.xyz_max[0],
                        self.xyz_min[1],
                        self.xyz_max[1],
                        self.xyz_min[2],
                        self.xyz_max[2],
                    )
                    _add_origin_reference(plotter, bounds)
                    if show_bounds:
                        _add_bounding_box(
                            plotter,
                            self.xyz_min,
                            self.xyz_max,
                            color="black",
                            line_width=0.5,
                        )

            # Validate extents against program bounds
            _PLANE_AXES = {
                "xy": (0, 1, 2),
                "xz": (0, 2, 1),
                "yz": (1, 2, 0),
            }
            axis_names = ["x", "y", "z"]

            for i, fs in enumerate(field_slices):
                if fs.plane not in _PLANE_AXES:
                    raise ValueError(
                        f"FieldSlice plane must be 'xy', 'xz', or 'yz', got '{fs.plane}'"
                    )
                ax_u, ax_v, ax_n = _PLANE_AXES[fs.plane]
                (u_min, u_max), (v_min, v_max) = fs.extents
                violations = []
                if u_min < self.xyz_min[ax_u] or u_max > self.xyz_max[ax_u]:
                    violations.append(
                        f"{axis_names[ax_u]}: field [{u_min}, {u_max}] vs bounds "
                        f"[{self.xyz_min[ax_u]}, {self.xyz_max[ax_u]}]"
                    )
                if v_min < self.xyz_min[ax_v] or v_max > self.xyz_max[ax_v]:
                    violations.append(
                        f"{axis_names[ax_v]}: field [{v_min}, {v_max}] vs bounds "
                        f"[{self.xyz_min[ax_v]}, {self.xyz_max[ax_v]}]"
                    )
                if fs.offset < self.xyz_min[ax_n] or fs.offset > self.xyz_max[ax_n]:
                    violations.append(
                        f"{axis_names[ax_n]} (offset): {fs.offset} vs bounds "
                        f"[{self.xyz_min[ax_n]}, {self.xyz_max[ax_n]}]"
                    )
                if violations:
                    raise ValueError(
                        f"FieldSlice '{fs.name}' extents exceed program bounds.\n"
                        + "\n".join(f"  {v}" for v in violations)
                        + "\nAdjust the field extents or the program bounds to match."
                    )

                _add_field_slice(
                    plotter,
                    fs,
                    batch_idx=batch_idx,
                    actor_name=f"field_slice_{i}_{fs.name}",
                )

        # Add isocontour polylines if provided
        if isocontour is not None:
            from .measure.isocontour import IsocontourResult
            from .measure.quadtree import _axis_mapping

            if plotter is None:
                plotter = _create_plotter()
                plotter.add_axes()
                if origin:
                    from .render.pyvista import _add_bounding_box, _add_origin_reference

                    bounds = (
                        self.xyz_min[0],
                        self.xyz_max[0],
                        self.xyz_min[1],
                        self.xyz_max[1],
                        self.xyz_min[2],
                        self.xyz_max[2],
                    )
                    _add_origin_reference(plotter, bounds)
                    if show_bounds:
                        _add_bounding_box(
                            plotter,
                            self.xyz_min,
                            self.xyz_max,
                            color="black",
                            line_width=0.5,
                        )

            import numpy as np
            import pyvista as pv

            # Uniform arrow scale from program bounds (not per-polyline)
            domain_diag = float(
                np.linalg.norm(np.array(self.xyz_max) - np.array(self.xyz_min))
            )
            normal_scale = domain_diag * 0.02

            for result in isocontour:
                if not isinstance(result, IsocontourResult):
                    raise TypeError(
                        f"isocontour list must contain IsocontourResult, "
                        f"got {type(result).__name__}"
                    )

                axis_idx, u_idx, v_idx, _, _ = _axis_mapping(result.axis)
                b = min(batch_idx, len(result.polylines) - 1)

                for poly in result.polylines[b]:
                    pts_2d = poly.detach().cpu().numpy()
                    K = len(pts_2d)
                    if K < 2:
                        continue

                    # Lift 2D contour to 3D
                    pts_3d = np.zeros((K + 1, 3))
                    pts_3d[:K, u_idx] = pts_2d[:, 0]
                    pts_3d[:K, v_idx] = pts_2d[:, 1]
                    pts_3d[:K, axis_idx] = result.station
                    pts_3d[K] = pts_3d[0]  # close the loop

                    line = pv.lines_from_points(pts_3d)
                    plotter.add_mesh(line, color="black", line_width=3)

                if isocontour_normals:
                    for poly_i, poly in enumerate(result.polylines[b]):
                        if poly_i >= len(result.normals[b]):
                            continue
                        norms = result.normals[b][poly_i]
                        pts_2d = poly.detach().cpu().numpy()
                        n_2d = norms.detach().cpu().numpy()
                        K = len(pts_2d)
                        step = max(1, K // 30)

                        # Build 3D arrows
                        idxs = list(range(0, K, step))
                        origins_3d = np.zeros((len(idxs), 3))
                        dirs_3d = np.zeros_like(origins_3d)
                        for j, idx in enumerate(idxs):
                            origins_3d[j, u_idx] = pts_2d[idx, 0]
                            origins_3d[j, v_idx] = pts_2d[idx, 1]
                            origins_3d[j, axis_idx] = result.station
                            if n_2d.shape[1] == 3:
                                # 3D normals: already in xyz space
                                dirs_3d[j] = n_2d[idx]
                            else:
                                # In-plane normals: lift to 3D
                                dirs_3d[j, u_idx] = n_2d[idx, 0]
                                dirs_3d[j, v_idx] = n_2d[idx, 1]

                        arrows = pv.PolyData(origins_3d)
                        arrows["vectors"] = dirs_3d
                        arrows.set_active_vectors("vectors")
                        glyphs = arrows.glyph(
                            orient="vectors",
                            scale=False,
                            factor=normal_scale,
                        )
                        plotter.add_mesh(glyphs, color="grey", opacity=0.6)

        # Show if not already shown by slice widget
        if has_slice:
            plotter.show()
        elif isocontour is not None and mesh_or_meshes is None and not has_slice:
            plotter.show()

        return plotter

    def compute_volume(
        self, mesh_result: MeshResult | dict[str, MeshResult] | list[MeshResult]
    ) -> Tensor | dict[str, Tensor]:
        """
        Compute volume from surface mesh using divergence theorem.

        This method only works with surface meshes (requires faces). For watertight
        surface meshes, volume is computed by summing signed tetrahedra formed by
        each triangle and the origin. Higher mesh resolution produces more accurate
        results.

        Args:
            mesh_result: MeshResult from mesh() call. Can be:
                - Single MeshResult (single body, no batch)
                - Dict[str, MeshResult] (multi-body, no batch)
                - List[MeshResult] (single body, batched)
                - List[Dict[str, MeshResult]] (multi-body, batched)

        Returns:
            - Single body, no batch: Tensor (scalar)
            - Single body, batched: Tensor shape [B]
            - Multi-body, no batch: Dict[str, Tensor] (scalars)
            - Multi-body, batched: Dict[str, Tensor] shape [B]

        Raises:
            ValueError: If mesh has no faces (tet-only mesh)
            ValueError: If shape is 2D (infinite volume)

        Examples:
            >>> # Single body
            >>> prog = CADProgram.load_from_yaml("examples/core/yaml/sphere.yaml")
            >>> mesh = prog.mesh(backend='skimage')
            >>> volume = prog.compute_volume(mesh)
            >>> print(volume)  # tensor(33.5103) for r=2 sphere

            >>> # Multi-body
            >>> prog = CADProgram.load_from_yaml("examples/core/yaml/assembly.yaml")
            >>> meshes = prog.mesh(backend='skimage')
            >>> volumes = prog.compute_volume(meshes)
            >>> print(volumes)  # {'bracket': tensor(4.0), 'bolt': tensor(0.113)}

            >>> # Batched
            >>> prog_batch = prog.with_params(radius=torch.linspace(1.0, 2.0, 10))
            >>> meshes = prog_batch.mesh(backend='skimage')
            >>> volumes = prog_batch.compute_volume(meshes)
            >>> print(volumes.shape)  # torch.Size([10])
        """
        # Validate shape is 3D (check self._shape or self._shapes)
        if not self.is_multi_body:
            if self._shape.plane is not None:
                raise ValueError(
                    f"Cannot compute volume for 2D shape in plane '{self._shape.plane}'. "
                    f"Volume is infinite for 2D shapes (circle, rectangle). "
                    f"Only 3D shapes (sphere, box, etc.) have finite volume."
                )
        else:
            for name, shape in self._shapes.items():
                if shape.plane is not None:
                    raise ValueError(
                        f"Cannot compute volume for 2D shape '{name}' in plane '{shape.plane}'. "
                        f"Volume is infinite for 2D shapes. Only 3D shapes have finite volume."
                    )

        # Handle different input types
        if isinstance(mesh_result, dict):
            # Multi-body: Dict[str, MeshResult]
            volumes = {}
            for name, mesh in mesh_result.items():
                if mesh.faces is None:
                    raise ValueError(
                        f"Body '{name}': MeshResult contains no surface mesh. "
                        f"Need faces for volume computation."
                    )
                if mesh.batch_size == 1:
                    volumes[name] = _compute_volume_from_mesh(
                        mesh.vertices[0], mesh.faces[0]
                    )
                else:
                    vols = [
                        _compute_volume_from_mesh(mesh.vertices[b], mesh.faces[b])
                        for b in range(mesh.batch_size)
                    ]
                    volumes[name] = torch.stack(vols)
            return volumes

        # Single body: MeshResult (batch_size >= 1)
        if mesh_result.faces is None:
            raise ValueError(
                "MeshResult contains no surface mesh. "
                "Need faces for volume computation."
            )
        if mesh_result.batch_size == 1:
            return _compute_volume_from_mesh(
                mesh_result.vertices[0],
                mesh_result.faces[0],
            )
        volumes = [
            _compute_volume_from_mesh(mesh_result.vertices[b], mesh_result.faces[b])
            for b in range(mesh_result.batch_size)
        ]
        return torch.stack(volumes)

    def compute_surface_area(
        self,
        mesh_result: MeshResult | dict[str, MeshResult] | list[MeshResult],
        return_flux: bool = False,
    ) -> (
        Tensor
        | dict[str, Tensor]
        | tuple[Tensor, Tensor]
        | tuple[dict[str, Tensor], dict[str, Tensor]]
    ):
        """
        Compute surface area from triangle mesh. Higher mesh resolution produces
        more accurate results.

        Optionally also computes normal flux, a differentiable measure of mesh
        closure. For a watertight mesh, flux should be ~0. Non-zero flux indicates
        holes or boundary edges.

        Args:
            mesh_result: MeshResult from mesh() call. Can be:
                - Single MeshResult (single body, no batch)
                - Dict[str, MeshResult] (multi-body, no batch)
                - List[MeshResult] (single body, batched)
                - List[Dict[str, MeshResult]] (multi-body, batched)
            return_flux: If True, also return normal flux loss (default: False)

        Returns:
            If return_flux=False:
                - Single body, no batch: Tensor (scalar)
                - Single body, batched: Tensor shape [B]
                - Multi-body, no batch: Dict[str, Tensor] (scalars)
                - Multi-body, batched: Dict[str, Tensor] shape [B]
            If return_flux=True:
                - Single body, no batch: (area: Tensor, flux: Tensor)
                - Single body, batched: (areas: Tensor[B], fluxes: Tensor[B])
                - Multi-body: (areas_dict, fluxes_dict)

        Raises:
            ValueError: If mesh has no faces (tet-only mesh)
            ValueError: If shape is 2D (infinite surface area)

        Examples:
            >>> # Single body
            >>> prog = CADProgram.load_from_yaml("examples/core/yaml/sphere.yaml")
            >>> mesh = prog.mesh(backend='skimage')
            >>> area = prog.compute_surface_area(mesh)
            >>> print(area)  # tensor(50.265) for r=2 sphere

            >>> # With flux (for watertightness checking)
            >>> area, flux = prog.compute_surface_area(mesh, return_flux=True)
            >>> print(f"Area: {area}, Flux: {flux}")  # flux ~0 for watertight

            >>> # Batched
            >>> prog_batch = prog.with_params(radius=torch.linspace(1.0, 2.0, 10))
            >>> meshes = prog_batch.mesh(backend='skimage')
            >>> areas = prog_batch.compute_surface_area(meshes)
            >>> print(areas.shape)  # torch.Size([10])
        """
        # Validate shape is 3D
        if not self.is_multi_body:
            if self._shape.plane is not None:
                raise ValueError(
                    f"Cannot compute surface area for 2D shape in plane '{self._shape.plane}'. "
                    f"Surface area is infinite for 2D shapes (circle, rectangle). "
                    f"Only 3D shapes (sphere, box, etc.) have finite surface area."
                )
        else:
            for name, shape in self._shapes.items():
                if shape.plane is not None:
                    raise ValueError(
                        f"Cannot compute surface area for 2D shape '{name}' in plane '{shape.plane}'. "
                        f"Surface area is infinite for 2D shapes. Only 3D shapes have finite surface area."
                    )

        # Handle different input types
        # NEW: MeshResult always contains lists (batch_size >= 1)
        if isinstance(mesh_result, dict):
            # Multi-body
            areas_by_body = {}
            fluxes_by_body = {} if return_flux else None
            for name, mesh in mesh_result.items():
                if mesh.faces is None:
                    raise ValueError(
                        f"Body '{name}': MeshResult contains no surface mesh. "
                        f"Surface area computation requires surface mesh faces."
                    )

                # Compute area (and flux) for each batch element
                areas = []
                fluxes = [] if return_flux else None
                for b in range(mesh.batch_size):
                    if return_flux:
                        area, flux = _compute_surface_area_and_flux_from_mesh(
                            mesh.vertices[b], mesh.faces[b]
                        )
                        fluxes.append(flux)
                    else:
                        area = _compute_surface_area_from_mesh(
                            mesh.vertices[b], mesh.faces[b]
                        )
                    areas.append(area)

                if mesh.batch_size == 1:
                    # Return scalar for single batch
                    areas_by_body[name] = areas[0]
                    if return_flux:
                        fluxes_by_body[name] = fluxes[0]
                else:
                    # Return stacked tensor for multiple batches
                    areas_by_body[name] = torch.stack(areas)
                    if return_flux:
                        fluxes_by_body[name] = torch.stack(fluxes)

            if return_flux:
                return areas_by_body, fluxes_by_body
            return areas_by_body

        # Single body
        if mesh_result.faces is None:
            raise ValueError(
                "MeshResult contains no surface mesh. "
                "Surface area computation requires surface mesh faces."
            )

        # Compute area (and flux) for each batch element
        areas = []
        fluxes = [] if return_flux else None
        for b in range(mesh_result.batch_size):
            if return_flux:
                area, flux = _compute_surface_area_and_flux_from_mesh(
                    mesh_result.vertices[b], mesh_result.faces[b]
                )
                fluxes.append(flux)
            else:
                area = _compute_surface_area_from_mesh(
                    mesh_result.vertices[b], mesh_result.faces[b]
                )
            areas.append(area)

        if mesh_result.batch_size == 1:
            # Return scalar for single batch
            if return_flux:
                return areas[0], fluxes[0]
            return areas[0]
        # Return stacked tensor [B] for multiple batches
        if return_flux:
            return torch.stack(areas), torch.stack(fluxes)
        return torch.stack(areas)

    def is_watertight(
        self, mesh_result: MeshResult | dict[str, MeshResult]
    ) -> bool | list[bool] | dict[str, bool] | dict[str, list[bool]]:
        """
        Check if a mesh is watertight (every edge shared by exactly 2 faces).

        Args:
            mesh_result: MeshResult from mesh() call

        Returns:
            - Single body, batch_size=1: bool
            - Single body, batch_size>1: List[bool]
            - Multi-body, batch_size=1: Dict[str, bool]
            - Multi-body, batch_size>1: Dict[str, List[bool]]
        """
        if isinstance(mesh_result, dict):
            results = {}
            for name, mesh in mesh_result.items():
                if mesh.faces is None:
                    raise ValueError(f"Body '{name}': no surface mesh")
                batch_results = [
                    _is_watertight(mesh.faces[b]) for b in range(mesh.batch_size)
                ]
                results[name] = (
                    batch_results[0] if mesh.batch_size == 1 else batch_results
                )
            return results
        if mesh_result.faces is None:
            raise ValueError("No surface mesh")
        results = [
            _is_watertight(mesh_result.faces[b]) for b in range(mesh_result.batch_size)
        ]
        return results[0] if mesh_result.batch_size == 1 else results

    def overlap_volume(
        self,
        name_a: str,
        name_b: str,
        bounds: dict | None = None,
        base_res: int = 8,
        levels: int = 4,
        step: int = 1,
        lipschitz: float = 1.0,
        epsilon: float = 0.01,
    ) -> "OverlapResult":
        """Compute overlap volume between two named shapes.

        Auto-detects unrotated axis-aligned boxes for fast AABB path.
        Falls back to hierarchical SDF evaluation for general shapes.

        Args:
            name_a: First shape name
            name_b: Second shape name
            bounds: Evaluation domain {x: [min, max], y: ..., z: ...}.
                Defaults to program bounds.
            base_res: Grid cells per axis at coarsest level (SDF path only)
            levels: Refinement levels (SDF path only)
            step: Octree levels per refinement (SDF path only). step=2
                subdivides 4x per axis, skipping intermediate evaluations.
            lipschitz: Lipschitz constant of the SDFs (SDF path only)
            epsilon: Sigmoid sharpness for differentiable volume (SDF path only)

        Returns:
            OverlapResult with volume [B] and uncertainty [B]

        Raises:
            ValueError: If shape name not found
        """
        from .measure.overlap import (
            OverlapResult,
            _resolve_box_params,
            aabb_overlap_volume,
            sdf_overlap_volume,
        )

        # Try AABB fast path (only when local shapes exist)
        if self.config.get("shapes"):
            result_a = _resolve_box_params(
                self.config, self.params, name_a, self.batch_size, device=self._device
            )
            result_b = _resolve_box_params(
                self.config, self.params, name_b, self.batch_size, device=self._device
            )

            if result_a is not None and result_b is not None:
                vol = aabb_overlap_volume(*result_a, *result_b)
                return OverlapResult(volume=vol, uncertainty=torch.zeros_like(vol))

        # SDF path, build individual shapes from config
        shapes_config = self.config.get("shapes", {})
        shape1 = self._build_single_shape(name_a, shapes_config)
        shape2 = self._build_single_shape(name_b, shapes_config)

        # Extract bounds
        b = bounds if bounds is not None else self.config.get("bounds", {})
        bounds_min = torch.tensor(
            [b["x"][0], b["y"][0], b["z"][0]],
            dtype=torch.float32,
            device=self._device,
        )
        bounds_max = torch.tensor(
            [b["x"][1], b["y"][1], b["z"][1]],
            dtype=torch.float32,
            device=self._device,
        )

        return sdf_overlap_volume(
            shape1,
            shape2,
            bounds_min,
            bounds_max,
            base_res=base_res,
            levels=levels,
            step=step,
            lipschitz=lipschitz,
            epsilon=epsilon,
        )

    def structural_properties(
        self,
        axis: str,
        stations: list[float],
        name: str | None = None,
        base_res: int = 8,
        levels: int = 4,
        lipschitz: float = 1.0,
        epsilon: float = 1e-3,
        compute_torsion: bool = False,
        torsion_levels: int | None = None,
    ) -> "SectionProperties":
        """Compute structural cross-section properties at slice stations.

        Evaluates area, centroid, second moments of area, principal moments,
        radii of gyration, and maximum first moments for shear. Optionally
        computes torsion constant J via the Prandtl stress function PDE.

        Args:
            axis: Slicing axis ('x', 'y', or 'z')
            stations: Positions along the slicing axis
            name: Shape name for multi-body output. If None, uses the
                single-body output shape.
            base_res: Coarse quadtree grid resolution
            levels: Number of quadtree refinement levels
            lipschitz: Lipschitz constant of the SDF
            epsilon: Sigmoid sharpness for boundary cell occupancy
            compute_torsion: If True, solve PDE for torsion constant J

        Returns:
            SectionProperties with all fields as [B, S] tensors.
            J is None if compute_torsion=False.
        """
        from .measure.structural import structural_properties as _sp

        # Build shape
        if name is not None:
            shapes_config = self.config.get("shapes", {})
            shape = self._build_single_shape(name, shapes_config)
        else:
            self._build_shapes()
            if self._shapes is not None:
                raise ValueError(
                    "Multi-body output requires 'name' parameter. "
                    f"Available shapes: {sorted(self._shapes.keys())}"
                )
            shape = self._shape

        # Extract bounds from config
        b = self.config.get("bounds", {})
        bounds_min = (b["x"][0], b["y"][0], b["z"][0])
        bounds_max = (b["x"][1], b["y"][1], b["z"][1])

        return _sp(
            shape,
            axis=axis,
            stations=stations,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            base_res=base_res,
            levels=levels,
            lipschitz=lipschitz,
            epsilon=epsilon,
            compute_torsion=compute_torsion,
            torsion_levels=torsion_levels,
        )

    def isocontour(
        self,
        plane: str,
        stations: list[float],
        *,
        normal_mode: str,
        name: str | None = None,
        base_res: int = 8,
        levels: int = 5,
        lipschitz: float = 1.0,
    ) -> "list[IsocontourResult]":
        """Extract 2D isocontour polylines at slice planes through a shape.

        Each contour vertex has an outward normal and arc-length quadrature
        weight, ready for downstream integration (e.g. Cp surface loads).

        Args:
            plane: In-plane axes ('xy', 'xz', or 'yz'). The slicing axis
                is the remaining axis (e.g. 'xz' slices along y).
            stations: Positions along the slicing axis.
            normal_mode: How to compute surface normals. Required.
                'in_plane': [K, 2] normals from tangent rotation.
                '3d': [K, 3] true surface normals via autograd on the SDF.
            name: Shape name for multi-body output. If None, uses the
                single-body output shape.
            base_res: Coarse quadtree grid resolution.
            levels: Number of quadtree refinement levels.
            lipschitz: Lipschitz constant of the SDF.

        Returns:
            List of IsocontourResult, one per station.
        """
        from .measure.isocontour import (
            _PLANE_TO_AXIS,
            isocontour_extract,
        )

        if plane not in _PLANE_TO_AXIS:
            raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")
        axis = _PLANE_TO_AXIS[plane]

        # Build shape
        if name is not None:
            shapes_config = self.config.get("shapes", {})
            shape = self._build_single_shape(name, shapes_config)
        else:
            self._build_shapes()
            if self._shapes is not None:
                raise ValueError(
                    "Multi-body output requires 'name' parameter. "
                    f"Available shapes: {sorted(self._shapes.keys())}"
                )
            shape = self._shape

        # Extract bounds from config
        b = self.config.get("bounds", {})
        bounds_min = (b["x"][0], b["y"][0], b["z"][0])
        bounds_max = (b["x"][1], b["y"][1], b["z"][1])

        return isocontour_extract(
            shape,
            axis=axis,
            stations=stations,
            bounds_min=bounds_min,
            bounds_max=bounds_max,
            normal_mode=normal_mode,
            base_res=base_res,
            levels=levels,
            lipschitz=lipschitz,
        )

    def rasterize_2d(
        self,
        plane: str = "xy",
        offset: float = 0.0,
        resolution: tuple[int, int] = (128, 128),
        extents: tuple[tuple[float, float], tuple[float, float]] | None = None,
        epsilon: float = 0.01,
        name: str | None = None,
    ) -> "RasterResult":
        """Rasterize a shape into differentiable 2D occupancy at a slice plane.

        Args:
            plane: Axis-aligned slice plane ('xy', 'xz', or 'yz').
            offset: Position along the normal axis (e.g. z-value for 'xy').
            resolution: (H, W) grid resolution in pixels.
            extents: ((u_min, u_max), (v_min, v_max)) bounds for the grid
                axes. Defaults to the program's bounds for those axes.
            epsilon: Sigmoid sharpness. Smaller = sharper boundary.
            name: Shape name for multi-body output. If None, uses the
                single-body output shape.

        Returns:
            RasterResult with occupancy [B, H, W] and raw SDF [B, H, W].
        """
        from .measure.rasterize import _PLANE_AXIS_MAP, rasterize_2d

        if plane not in _PLANE_AXIS_MAP:
            raise ValueError(f"plane must be 'xy', 'xz', or 'yz', got '{plane}'")

        # Resolve shape (same pattern as isocontour)
        if name is not None:
            shapes_config = self.config.get("shapes", {})
            shape = self._build_single_shape(name, shapes_config)
        else:
            self._build_shapes()
            if self._shapes is not None:
                raise ValueError(
                    "Multi-body output requires 'name' parameter. "
                    f"Available shapes: {sorted(self._shapes.keys())}"
                )
            shape = self._shape

        # Auto-derive extents from program bounds if not provided
        if extents is None:
            ax_u, ax_v, _ax_n = _PLANE_AXIS_MAP[plane]
            axis_names = ["x", "y", "z"]
            b = self.config["bounds"]
            extents = (
                tuple(b[axis_names[ax_u]]),
                tuple(b[axis_names[ax_v]]),
            )

        return rasterize_2d(
            shape,
            plane=plane,
            offset=offset,
            resolution=resolution,
            extents=extents,
            epsilon=epsilon,
        )

    def __call__(self, points: Tensor) -> Tensor | dict[str, Tensor]:
        """
        Evaluate the SDF at given points.

        Args:
            points: [N, 3] tensor of 3D points

        Returns:
            Single body: [N] tensor of signed distances (or [B, N] if batched)
            Multi-body: Dict[str, Tensor] keyed by body name
        """
        self._build_shapes()

        if self._shapes is not None:
            # Multi-body output
            return {name: shape(points) for name, shape in self._shapes.items()}
        # Single-body output (backwards compatible)
        return self._shape(points)

    def __repr__(self) -> str:
        name = self.config.get("name", "unnamed")
        output = self.config.get("output", "?")
        if self.is_multi_body:
            return f"<CADProgram '{name}' output={output}>"
        return f"<CADProgram '{name}' output='{output}'>"


def exec_module(name: str) -> None:
    """Execute a module, but dont bring them into scope.

    This is useful to populate `SHAPE_LOADERS`.
    """
    spec = importlib.util.find_spec(name)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
