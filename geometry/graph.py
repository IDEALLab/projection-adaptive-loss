"""
Graph Format Conversion for geometry

Converts between YAML CAD programs and PyTorch Geometric graph format.
Graphs use PyG Data objects suitable for GNN encoders (GAT, etc.)

Graph structure:
- x: [num_nodes] int - node type vocabulary index
- node_attr: [num_nodes, 2] - [axis_idx/dim_idx, plane_idx] per node
  - For shape nodes: [axis_idx, plane_idx]
  - For variable nodes: [dim_idx, 0] where dim_idx is scalar=0, vec2=1, etc.
- edge_index: [2, num_edges] - (src, dst) children->parent direction
- edge_attr: [num_edges, 2] - [edge_type_idx, param_name_idx]
  - param_name_idx is 0 for non-param edges
- input_mask: [num_nodes] bool - True for input variable nodes (from params section)
- output_mask: [num_nodes] bool - True for output shape nodes
- Output node is always the last node (index num_nodes-1)
"""

from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from geometry.loader import validate_no_orphan_params

if TYPE_CHECKING:
    from torch_geometric.data import Data


# Lazy import to avoid hard dependency
def _get_pyg_data():
    try:
        from torch_geometric.data import Data  # noqa: E402

        return Data
    except ImportError:
        raise ImportError(
            "torch_geometric is required for graph operations. "
            "Install with: pip install torch-geometric"
        )


# Vocabularies

NODE_TYPES: dict[str, int] = {
    # Primitives (3D)
    "sphere": 0,
    "box": 1,
    "box_sharp": 2,
    # Primitives (2D)
    "circle": 3,
    "rectangle": 4,
    "rectangle_sharp": 5,
    # Boolean operations
    "union": 6,
    "intersection": 7,
    "difference": 8,
    "smooth_union": 9,
    "smooth_intersection": 10,
    "smooth_difference": 11,
    "inverse": 12,
    # Transform operations
    "translate": 13,
    "rotate": 14,
    "mirror": 15,
    "extrude": 16,
    "revolve": 17,
    # Variable (parametric input)
    "variable": 18,
    # Special
    "assembly": 19,  # virtual node for multi-body output
}

# Reverse mapping for decoding
NODE_TYPES_INV: dict[int, str] = {v: k for k, v in NODE_TYPES.items()}

AXIS: dict[str, int] = {
    "x": 0,
    "y": 1,
    "z": 2,
    "none": 3,
}

AXIS_INV: dict[int, str] = {v: k for k, v in AXIS.items()}

PLANE: dict[str, int] = {
    "xy": 0,
    "xz": 1,
    "yz": 2,
    "none": 3,
}

PLANE_INV: dict[int, str] = {v: k for k, v in PLANE.items()}

EDGE_TYPES: dict[str, int] = {
    "child": 0,  # generic child (union, intersection operands)
    "base": 1,  # base shape for difference
    "subtract": 2,  # subtracted shape for difference
    "input": 3,  # single input (translate, rotate, mirror, extrude, revolve, inverse)
    "param": 4,  # variable feeds into shape parameter
}

EDGE_TYPES_INV: dict[int, str] = {v: k for k, v in EDGE_TYPES.items()}

# Parameter names for param edges
# edge_attr = [edge_type_idx, param_name_idx]
# param_name_idx = 0 for non-param edges (unused)
PARAM_NAMES: dict[str, int] = {
    "none": 0,  # placeholder for non-param edges
    "radius": 1,
    "size_x": 2,
    "size_y": 3,
    "size_z": 4,
    "center_x": 5,
    "center_y": 6,
    "center_z": 7,
    "offset_x": 8,
    "offset_y": 9,
    "offset_z": 10,
    "angle": 11,
    "k": 12,
    "min": 13,  # extrude min bound
    "max": 14,  # extrude max bound
    "start_angle": 15,
    "end_angle": 16,
    "offset": 17,  # mirror offset (scalar)
}

PARAM_NAMES_INV: dict[int, str] = {v: k for k, v in PARAM_NAMES.items()}

# Variable dimensionality (stored in node_attr[0] for variable nodes)
VAR_DIM: dict[str, int] = {
    "scalar": 0,
    "vec2": 1,
    "vec3": 2,
    "vec4": 3,
}

VAR_DIM_INV: dict[int, str] = {v: k for k, v in VAR_DIM.items()}


# YAML to Graph Conversion


def _is_var_ref(value: Any) -> bool:
    """Check if a value is a variable reference (e.g., '$width')."""
    return isinstance(value, str) and value.startswith("$")


def _get_var_name(ref: str) -> str:
    """Extract variable name from reference (e.g., '$width' -> 'width')."""
    return ref[1:]  # Remove leading '$'


def yaml_to_graph(config: dict[str, Any]) -> "Data":
    """
    Convert parsed YAML config to PyG Data graph.

    Args:
        config: Parsed YAML configuration dict (from load_yaml or CADProgram.config)
                May contain 'params' and 'constants' sections for variable nodes.

    Returns:
        PyG Data object with:
        - x: [N] node type indices
        - node_attr: [N, 2] - [axis_idx/dim_idx, plane_idx]
        - edge_index: [2, E] - edges
        - edge_attr: [E, 2] - [edge_type_idx, param_name_idx]
        - input_mask: [N] bool - True for input variable nodes
        - output_mask: [N] bool - True for output shape nodes

    Raises:
        ValueError: If config contains unsupported types (formula, neural)
        ValueError: If config is empty or malformed
        ValueError: If $ref references undefined variable
    """
    Data = _get_pyg_data()

    shapes_config = config.get("shapes", {})
    params_config = config.get("params", {})
    constants_config = config.get("constants", {})
    output = config.get("output")

    if not shapes_config:
        raise ValueError("Empty shapes config - cannot create graph")

    if not output:
        raise ValueError("Missing 'output' field in config")

    # Validate no orphan params (using shared function from loader)
    validate_no_orphan_params(config)

    # Collect all variable definitions
    all_variables: dict[str, bool] = {}  # var_name -> is_input
    for var_name in params_config:
        all_variables[var_name] = True  # params are inputs
    for var_name in constants_config:
        all_variables[var_name] = False  # constants are not inputs

    # Track nodes as we build them
    # node_info: list of (type_idx, attr1, attr2, is_input_var)
    # For shape nodes: (type_idx, axis_idx, plane_idx, False)
    # For variable nodes: (type_idx, dim_idx, 0, is_input)
    node_info: list[tuple[int, int, int, bool]] = []
    # edges: list of (src_idx, dst_idx, edge_type_idx, param_name_idx)
    edges: list[tuple[int, int, int, int]] = []
    # Map shape_name -> node_index
    shape_to_node: dict[str, int] = {}
    # Map var_name -> node_index
    var_to_node: dict[str, int] = {}
    # Track which nodes are outputs
    output_node_indices: list[int] = []

    # First, create variable nodes (they come before shape nodes)
    for var_name, is_input in all_variables.items():
        var_idx = len(node_info)
        # All variables are scalar (dim_idx=0) per design decision
        node_info.append((NODE_TYPES["variable"], VAR_DIM["scalar"], 0, is_input))
        var_to_node[var_name] = var_idx

    def get_axis_plane(shape_def: dict[str, Any], shape_type: str) -> tuple[int, int]:
        """Extract axis and plane indices from shape definition."""
        axis_str = shape_def.get("axis", "none")
        plane_str = shape_def.get("plane", "none")

        # Normalize to lowercase
        if isinstance(axis_str, str):
            axis_str = axis_str.lower()
        else:
            axis_str = "none"
        if isinstance(plane_str, str):
            plane_str = plane_str.lower()
        else:
            plane_str = "none"

        axis_idx = AXIS.get(axis_str, AXIS["none"])
        plane_idx = PLANE.get(plane_str, PLANE["none"])

        return axis_idx, plane_idx

    def add_param_edges(
        shape_def: dict[str, Any], shape_type: str, node_idx: int
    ) -> None:
        """
        Scan shape definition for $var references and create param edges.
        """
        # Map of parameter names to their PARAM_NAMES keys
        # For vector params (size, center, offset), we use component names
        param_mappings = {
            # Scalar params
            "radius": ["radius"],
            "angle": ["angle"],
            "k": ["k"],
            "min": ["min"],
            "max": ["max"],
            "start_angle": ["start_angle"],
            "end_angle": ["end_angle"],
            # Vector params (3D)
            "size": ["size_x", "size_y", "size_z"],
            "center": ["center_x", "center_y", "center_z"],
        }

        # Handle 'offset' specially - it's a vector for translate, scalar for mirror
        if shape_type == "translate":
            param_mappings["offset"] = ["offset_x", "offset_y", "offset_z"]
        elif shape_type == "mirror":
            param_mappings["offset"] = ["offset"]

        for param_key, param_names in param_mappings.items():
            if param_key not in shape_def:
                continue
            value = shape_def[param_key]

            if _is_var_ref(value):
                # Scalar variable reference
                var_name = _get_var_name(value)
                if var_name not in var_to_node:
                    raise ValueError(
                        f"Variable '${var_name}' referenced in shape but not defined in "
                        f"'params' or 'constants' section"
                    )
                var_idx = var_to_node[var_name]
                param_name_idx = PARAM_NAMES.get(param_names[0], PARAM_NAMES["none"])
                edges.append((var_idx, node_idx, EDGE_TYPES["param"], param_name_idx))

            elif isinstance(value, list):
                # Vector param - check each component
                for i, component in enumerate(value):
                    if _is_var_ref(component):
                        var_name = _get_var_name(component)
                        if var_name not in var_to_node:
                            raise ValueError(
                                f"Variable '${var_name}' referenced in shape but not defined in "
                                f"'params' or 'constants' section"
                            )
                        var_idx = var_to_node[var_name]
                        # Use appropriate component name (e.g., center_x, center_y, center_z)
                        if i < len(param_names):
                            param_name_idx = PARAM_NAMES.get(
                                param_names[i], PARAM_NAMES["none"]
                            )
                        else:
                            param_name_idx = PARAM_NAMES["none"]
                        edges.append(
                            (var_idx, node_idx, EDGE_TYPES["param"], param_name_idx)
                        )

    def build_node(shape_name: str) -> int:
        """
        Recursively build graph nodes for a shape.
        Returns the node index of the built shape.
        Builds children FIRST (depth-first) so leaf nodes have lower indices.
        """
        # Check if already built (for shared references)
        if shape_name in shape_to_node:
            return shape_to_node[shape_name]

        if shape_name not in shapes_config:
            raise ValueError(f"Shape '{shape_name}' not found in shapes config")

        shape_def = shapes_config[shape_name]
        shape_type = shape_def.get("type")

        if shape_type is None:
            raise ValueError(f"Shape '{shape_name}' has no 'type' field")

        # Reject unsupported types
        if shape_type in ("formula3d", "formula2d"):
            raise ValueError(
                f"Formula primitives ('{shape_type}') are not supported in graph format. "
                f"Shape '{shape_name}' uses formula - defer to v2."
            )
        if shape_type in ("neural3d", "neural2d"):
            raise ValueError(
                f"Neural primitives ('{shape_type}') are not supported in graph format. "
                f"Shape '{shape_name}' uses neural - defer to v2."
            )

        if shape_type not in NODE_TYPES:
            raise ValueError(
                f"Unknown shape type '{shape_type}' for shape '{shape_name}'. "
                f"Supported: {list(NODE_TYPES.keys())}"
            )

        type_idx = NODE_TYPES[shape_type]
        axis_idx, plane_idx = get_axis_plane(shape_def, shape_type)

        # Build children FIRST (depth-first traversal)
        # This ensures leaf nodes have lower indices than their parents
        child_indices: list[tuple[int, int]] = []  # (child_idx, edge_type)

        if shape_type in (
            "union",
            "intersection",
            "smooth_union",
            "smooth_intersection",
        ):
            child_names = shape_def.get("shapes", [])
            for child_name in child_names:
                child_idx = build_node(child_name)
                child_indices.append((child_idx, EDGE_TYPES["child"]))

        elif shape_type in ("difference", "smooth_difference"):
            child_names = shape_def.get("shapes", [])
            for i, child_name in enumerate(child_names):
                child_idx = build_node(child_name)
                edge_type = EDGE_TYPES["base"] if i == 0 else EDGE_TYPES["subtract"]
                child_indices.append((child_idx, edge_type))

        elif shape_type in (
            "translate",
            "rotate",
            "mirror",
            "extrude",
            "revolve",
            "inverse",
        ):
            child_name = shape_def.get("shape")
            if child_name:
                child_idx = build_node(child_name)
                child_indices.append((child_idx, EDGE_TYPES["input"]))

        # NOW create this node (after all children are built)
        node_idx = len(node_info)
        node_info.append((type_idx, axis_idx, plane_idx, False))
        shape_to_node[shape_name] = node_idx

        # Add edges from children to this node
        for child_idx, edge_type in child_indices:
            edges.append((child_idx, node_idx, edge_type, PARAM_NAMES["none"]))

        # Add param edges from variables to this node
        add_param_edges(shape_def, shape_type, node_idx)

        return node_idx

    # Handle single vs multi-body output
    if isinstance(output, list):
        # Multi-body: create assembly node
        for out_name in output:
            out_idx = build_node(out_name)
            output_node_indices.append(out_idx)

        # Create assembly node
        assembly_idx = len(node_info)
        node_info.append((NODE_TYPES["assembly"], AXIS["none"], PLANE["none"], False))
        output_node_indices = [assembly_idx]  # assembly is the output

        # Connect all outputs to assembly
        for out_idx in [shape_to_node[name] for name in output]:
            edges.append(
                (out_idx, assembly_idx, EDGE_TYPES["child"], PARAM_NAMES["none"])
            )
    else:
        # Single body
        out_idx = build_node(output)
        output_node_indices = [out_idx]

    # Convert to tensors
    num_nodes = len(node_info)

    x = torch.tensor([n[0] for n in node_info], dtype=torch.long)
    node_attr = torch.tensor([[n[1], n[2]] for n in node_info], dtype=torch.long)

    # Create input_mask (True for input variable nodes)
    input_mask = torch.tensor([n[3] for n in node_info], dtype=torch.bool)

    # Create output_mask (True for output nodes)
    output_mask = torch.zeros(num_nodes, dtype=torch.bool)
    for idx in output_node_indices:
        output_mask[idx] = True

    if edges:
        edge_index = torch.tensor(
            [[e[0] for e in edges], [e[1] for e in edges]], dtype=torch.long
        )
        edge_attr = torch.tensor([[e[2], e[3]] for e in edges], dtype=torch.long)
    else:
        # No edges (single primitive without variables)
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.long)

    # Validate no self-loops
    if edge_index.shape[1] > 0:
        self_loops = (edge_index[0] == edge_index[1]).any()
        if self_loops:
            raise ValueError("Graph contains self-loops - invalid CAD structure")

    return Data(
        x=x,
        node_attr=node_attr,
        edge_index=edge_index,
        edge_attr=edge_attr,
        input_mask=input_mask,
        output_mask=output_mask,
    )


# Graph to YAML Conversion


def graph_to_yaml(data: "Data") -> dict[str, Any]:
    """
    Convert PyG Data graph back to YAML config dict.

    Args:
        data: PyG Data object with x, node_attr, edge_index, edge_attr,
              input_mask, output_mask

    Returns:
        YAML-compatible config dict with shapes, params, constants, and output fields

    Note:
        - Generated shape names follow pattern: "{type}_{index}" (e.g., "sphere_0")
        - Variable names follow pattern: "var_{index}" (e.g., "var_0")
        - Numeric parameters are NOT preserved in graph format
        - Returns config with placeholder params or $var refs where applicable
    """
    num_nodes = data.x.shape[0]

    if num_nodes == 0:
        raise ValueError("Empty graph - cannot convert to YAML")

    # Check if edge_attr is 1D (old format) or 2D (new format with param names)
    edge_attr = data.edge_attr
    has_param_names = len(edge_attr.shape) == 2 and edge_attr.shape[1] == 2

    # Build adjacency: for each node, list of (child_idx, edge_type, param_name_idx)
    children: dict[int, list[tuple[int, int, int]]] = {i: [] for i in range(num_nodes)}

    edge_index = data.edge_index

    for i in range(edge_index.shape[1]):
        src = edge_index[0, i].item()
        dst = edge_index[1, i].item()
        if has_param_names:
            etype = edge_attr[i, 0].item()
            param_name_idx = edge_attr[i, 1].item()
        else:
            etype = edge_attr[i].item()
            param_name_idx = 0
        children[dst].append((src, etype, param_name_idx))

    # Check for input_mask (may not exist in old graphs)
    has_input_mask = hasattr(data, "input_mask") and data.input_mask is not None

    # Generate names for each node
    type_counts: dict[str, int] = {}
    node_names: dict[int, str] = {}
    var_counter = 0

    for i in range(num_nodes):
        type_idx = data.x[i].item()
        type_name = NODE_TYPES_INV[type_idx]
        if type_name == "variable":
            # Use var_N naming for variables
            node_names[i] = f"var_{var_counter}"
            var_counter += 1
        else:
            count = type_counts.get(type_name, 0)
            node_names[i] = f"{type_name}_{count}"
            type_counts[type_name] = count + 1

    # Build params and constants dicts from variable nodes
    params: dict[str, float] = {}
    constants: dict[str, float] = {}

    for i in range(num_nodes):
        type_idx = data.x[i].item()
        type_name = NODE_TYPES_INV[type_idx]
        if type_name == "variable":
            var_name = node_names[i]
            is_input = has_input_mask and data.input_mask[i].item()
            if is_input:
                params[var_name] = 1.0  # placeholder value
            else:
                constants[var_name] = 1.0  # placeholder value

    # Build mapping from (shape_node_idx, param_name) -> var_name for $refs
    param_refs: dict[tuple[int, str], str] = {}
    for dst_idx, incoming_edges in children.items():
        for src_idx, etype, param_name_idx in incoming_edges:
            if etype == EDGE_TYPES["param"]:
                param_name = PARAM_NAMES_INV.get(param_name_idx, "none")
                var_name = node_names[src_idx]
                param_refs[(dst_idx, param_name)] = var_name

    def get_param_value(node_idx: int, param_name: str, default: Any) -> Any:
        """Get parameter value, using $ref if a variable feeds this param."""
        key = (node_idx, param_name)
        if key in param_refs:
            return f"${param_refs[key]}"
        return default

    def get_vector_param(
        node_idx: int, base_name: str, components: list[str], default: list
    ) -> list:
        """Get vector parameter, with $refs for individual components."""
        result = []
        for j, comp_name in enumerate(components):
            key = (node_idx, comp_name)
            if key in param_refs:
                result.append(f"${param_refs[key]}")
            else:
                result.append(default[j] if j < len(default) else 0)
        return result

    # Build shapes dict
    shapes: dict[str, Any] = {}

    for i in range(num_nodes):
        type_idx = data.x[i].item()
        type_name = NODE_TYPES_INV[type_idx]

        # Skip variable nodes - they go in params/constants
        if type_name == "variable":
            continue

        axis_idx = data.node_attr[i, 0].item()
        plane_idx = data.node_attr[i, 1].item()

        shape_def: dict[str, Any] = {"type": type_name}

        # Add axis/plane if not 'none'
        if axis_idx != AXIS["none"]:
            shape_def["axis"] = AXIS_INV[axis_idx]
        if plane_idx != PLANE["none"]:
            shape_def["plane"] = PLANE_INV[plane_idx]

        # Get non-param children (structural edges)
        node_children = [
            (src, etype)
            for src, etype, _ in children[i]
            if etype != EDGE_TYPES["param"]
        ]

        if type_name in ("sphere", "box", "box_sharp"):
            # 3D primitives
            if type_name == "sphere":
                shape_def["radius"] = get_param_value(i, "radius", 1.0)
                shape_def["center"] = get_vector_param(
                    i, "center", ["center_x", "center_y", "center_z"], [0, 0, 0]
                )
            else:
                shape_def["size"] = get_vector_param(
                    i, "size", ["size_x", "size_y", "size_z"], [1, 1, 1]
                )
                shape_def["center"] = get_vector_param(
                    i, "center", ["center_x", "center_y", "center_z"], [0, 0, 0]
                )

        elif type_name in ("circle", "rectangle", "rectangle_sharp"):
            # 2D primitives
            if type_name == "circle":
                shape_def["radius"] = get_param_value(i, "radius", 1.0)
                shape_def["center"] = get_vector_param(
                    i, "center", ["center_x", "center_y"], [0, 0]
                )
            else:
                shape_def["size"] = get_vector_param(
                    i, "size", ["size_x", "size_y"], [1, 1]
                )
                shape_def["center"] = get_vector_param(
                    i, "center", ["center_x", "center_y"], [0, 0]
                )

        elif type_name in (
            "union",
            "intersection",
            "smooth_union",
            "smooth_intersection",
        ):
            # Multi-child operations (filter out param edges)
            child_names = [
                node_names[c[0]] for c in node_children if c[1] == EDGE_TYPES["child"]
            ]
            shape_def["shapes"] = child_names
            if type_name.startswith("smooth_"):
                shape_def["k"] = get_param_value(i, "k", 5.0)

        elif type_name in ("difference", "smooth_difference"):
            # Ordered children: base first, then subtract
            base_children = [c for c in node_children if c[1] == EDGE_TYPES["base"]]
            subtract_children = [
                c for c in node_children if c[1] == EDGE_TYPES["subtract"]
            ]
            child_names = [node_names[c[0]] for c in base_children] + [
                node_names[c[0]] for c in subtract_children
            ]
            shape_def["shapes"] = child_names
            if type_name == "smooth_difference":
                shape_def["k"] = get_param_value(i, "k", 5.0)

        elif type_name in (
            "translate",
            "rotate",
            "mirror",
            "extrude",
            "revolve",
            "inverse",
        ):
            # Single input (filter out param edges)
            input_children = [c for c in node_children if c[1] == EDGE_TYPES["input"]]
            if input_children:
                shape_def["shape"] = node_names[input_children[0][0]]

            # Add operation-specific params with potential $refs
            if type_name == "translate":
                shape_def["offset"] = get_vector_param(
                    i, "offset", ["offset_x", "offset_y", "offset_z"], [0, 0, 0]
                )
            elif type_name == "rotate":
                shape_def["angle"] = get_param_value(i, "angle", 0)
            elif type_name == "mirror":
                shape_def["offset"] = get_param_value(i, "offset", 0)
            elif type_name == "extrude":
                shape_def["min"] = get_param_value(i, "min", 0)
                shape_def["max"] = get_param_value(i, "max", 1.0)
            elif type_name == "revolve":
                shape_def["start_angle"] = get_param_value(i, "start_angle", 0)
                shape_def["end_angle"] = get_param_value(i, "end_angle", 360)

        elif type_name == "assembly":
            # Skip assembly node - handled in output
            continue

        shapes[node_names[i]] = shape_def

    # Determine output
    last_idx = num_nodes - 1
    last_type = NODE_TYPES_INV[data.x[last_idx].item()]

    if last_type == "assembly":
        # Multi-body: output is list of assembly's children (non-param edges)
        assembly_children = [
            (src, etype)
            for src, etype, _ in children[last_idx]
            if etype != EDGE_TYPES["param"]
        ]
        output = [node_names[c[0]] for c in assembly_children]
    else:
        # Single body
        output = node_names[last_idx]

    # Build config
    config: dict[str, Any] = {}

    if params:
        config["params"] = params
    if constants:
        config["constants"] = constants

    config["shapes"] = shapes
    config["output"] = output

    return config


# Serialization


def save_graphs(graphs: list["Data"], path: str | Path) -> None:
    """
    Save list of PyG Data graphs to a .pt file.

    Args:
        graphs: List of PyG Data objects
        path: Output path (should end with .pt)
    """
    torch.save(graphs, path)


def load_graphs(path: str | Path) -> list["Data"]:
    """
    Load list of PyG Data graphs from a .pt file.

    Args:
        path: Path to .pt file

    Returns:
        List of PyG Data objects
    """
    return torch.load(path, weights_only=False)


