"""
YAML Loading and Parameter Substitution

Handles loading CAD program YAML files and substituting $param references
with actual values from the params section.
"""

import ast
import copy
import inspect
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

import yaml

from geometry.core import Shape

if TYPE_CHECKING:
    from geometry.stdlib.curves import Curve2D


def load_yaml(path: str | Path) -> dict[str, Any]:
    """
    Load YAML file and return config dict.

    Args:
        path: Path to YAML file

    Returns:
        Parsed configuration dictionary
    """
    with open(path) as f:
        return yaml.safe_load(f)


# Whitelisted single-arg functions for _eval_param_expr.
# Each maps to (math_func, torch_func_name), torch looked up lazily.
_EXPR_FUNCTIONS = {"sin", "cos", "tan", "sqrt", "abs"}

# Whitelisted constants (name -> value)
_EXPR_CONSTANTS = {"pi": math.pi}


_DEG_TO_RAD = math.pi / 180.0


@dataclass(frozen=True)
class ShapeFactoryArgs:
    """Arguments of a shape factory.

    This is used to translate shape defs to shape factory arguments."""

    positionals: tuple[tuple[str, type[Any]], ...]
    var_positionals: tuple[tuple[str, type[Any]], ...]
    kwargs: tuple[tuple[str, type[Any]], ...]

    def all(self) -> tuple[tuple[str, type[Any]], ...]:
        return self.positionals + self.var_positionals + self.kwargs

    @classmethod
    def from_signature(
        cls,
        f: Callable[..., Any],
        feed_positionals_from: str | None = None,
    ) -> Self:
        parameters = list(inspect.signature(f).parameters.values())
        if feed_positionals_from:
            positional_indices = [
                i
                for i, p in enumerate(parameters)
                if p.kind in {p.POSITIONAL_ONLY, p.VAR_POSITIONAL}
            ]
            if not positional_indices:
                raise RuntimeError(
                    "At least one positional only parameter must be present if `combine_positionals` is used."
                )
            positionals = parameters[: max(positional_indices) + 1]
            kw_parameters = parameters[max(positional_indices) + 1 :]
            not_positional_only = [
                p
                for p in positionals
                if p.kind not in {p.POSITIONAL_ONLY, p.VAR_POSITIONAL}
            ]
            if not_positional_only:
                raise RuntimeError(
                    "The positional only arguments must be contiguous if `combine_positionals` is used. "
                    f"Found parameters {','.join(p.name for p in not_positional_only)}"
                    f" with kind {','.join(str(p.kind) for p in not_positional_only)}"
                )
            try:
                (positional_type,) = set(p.annotation for p in positionals)
            except ValueError:
                raise RuntimeError(
                    "All type hints for positional only parameters must be the same if `combine_positionals` is used."
                )
            parameters = [
                inspect.Parameter(
                    feed_positionals_from,
                    inspect.Parameter.VAR_POSITIONAL,
                    annotation=positional_type,
                ),
                *kw_parameters,
            ]
        return cls(
            positionals=tuple(
                (p.name, p.annotation)
                for p in parameters
                if p.kind is inspect.Parameter.POSITIONAL_ONLY
            ),
            var_positionals=tuple(
                (p.name, p.annotation)
                for p in parameters
                if p.kind is inspect.Parameter.VAR_POSITIONAL
            ),
            kwargs=tuple(
                (p.name, p.annotation)
                for p in parameters
                if p.kind
                not in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.POSITIONAL_ONLY,
                }
            ),
        )


@dataclass(frozen=True)
class NodeSchema:
    """Schema of a schape def."""

    shape_refs: tuple[str, ...] = ()
    """References to shapes."""
    curve_refs: tuple[str, ...] = ()
    """References to curces."""
    params: frozenset[str] = frozenset()
    """Other parameters."""

    @classmethod
    def from_shape_factory_args(cls, args: ShapeFactoryArgs) -> Self:
        """Convert from `ShapeFactoryArgs`."""

        out = cls(
            shape_refs=tuple(name for name, t in args.all() if is_shape(t)),
            curve_refs=tuple(name for name, t in args.all() if is_curve(t)),
            params=frozenset(
                name for name, t in args.all() if t is not Shape and not is_curve(t)
            ),
        )
        return out


def is_curve(annotation: Any) -> bool:
    """Check for subclasses of Curve2D without importing Curve2D.

    Does not work for string "CubicBezier2D".
    """
    return (
        annotation == "Curve2D"
        or isinstance(annotation, type)
        and (
            annotation.__name__ == "Curve2D"
            or any(t.__name__ == "Curve2D" for t in annotation.__mro__)
        )
    )


def is_shape(annotation: Any) -> bool:
    """Check if an annotation represents a shape."""
    return annotation is Shape


class ShapeLoader:
    """Translate structured data (`dict`) from a yaml file to positional and keyword args of a shape factory.

    This is created by the @loadable_shape decorator.
    """

    def __init__(
        self,
        shape_factory: Callable[..., Shape],
        args: ShapeFactoryArgs,
        *,
        inject: tuple[str, ...] = (),
        input_shape_dim: int | None = None,
        check_shape_def: Callable[[dict[str, Any]], None] | None = None,
        deserialize_args: dict[str, Callable[[Any], Any]] | None = None,
    ) -> None:
        self.shape_factory = shape_factory
        self.inject = inject
        self.args = args
        self.input_shape_dim = input_shape_dim
        self.check_shape_def = (
            check_shape_def if check_shape_def is not None else lambda _: None
        )
        self.deserialize_args = deserialize_args or {}

    def load(
        self,
        shape_def: dict[str, Any],
        lookup_shape: Callable[[str], Shape],
        lookup_curve: "Callable[[str], Curve2D]",
    ) -> Shape:
        """Instantiate a shape object from a shape def."""

        # Report missing shapes to the user:
        missing_shape_refs = sorted(
            set(name for name, t in self.args.all() if is_shape(t)) - set(shape_def)
        )
        if missing_shape_refs:
            fields = ", ".join(f"'{s}" for s in missing_shape_refs)
            pluralization = "" if len(missing_shape_refs) == 1 else "s"
            shape_dim = (
                "2D or 3D"
                if self.input_shape_dim is None
                else f"{self.input_shape_dim}D"
            )
            raise ValueError(
                f"{self.shape_factory.__name__} requires field{pluralization} {fields} referencing a {shape_dim} shape. "
                f"Example:\n"
                + craft_example(
                    self.shape_factory.__name__,
                    is_shape,
                    self.args,
                    self.input_shape_dim,
                )
            )
        # Report missing curves to the user:
        missing_curve_refs = sorted(
            set(name for name, t in self.args.all() if is_curve(t)) - set(shape_def)
        )
        if missing_curve_refs:
            fields = ", ".join(f"'{s}" for s in missing_shape_refs)
            pluralization = "" if len(missing_curve_refs) == 1 else "s"
            raise ValueError(
                f"{self.shape_factory.__name__} requires field{pluralization} {fields} referencing a named curve. "
                f"Example:\n"
                + craft_example(
                    self.shape_factory.__name__,
                    is_curve,
                    self.args,
                    self.input_shape_dim,
                )
            )

        # Custom assertions:
        self.check_shape_def(shape_def)

        loaded_shapes: dict[str, Any] = {}

        # Special deserialization. Apply first matching method in order:
        # - Custom deserialization
        # - Lookup shape
        # - Lookup curve
        # - Pass unodified argument
        def deserialize_arg(key: str, value: Any, annotation: Any) -> Any:
            deserializer = self.deserialize_args.get(key)
            if deserializer is not None:
                return deserializer(value)
            if is_shape(annotation):
                if value in loaded_shapes:
                    return loaded_shapes[value]
                shape = loaded_shapes[value] = lookup_shape(value)
                return shape
            if is_curve(annotation):
                try:
                    return lookup_curve(value)
                except ValueError as e:
                    raise ValueError(
                        f"{self.shape_factory.__name__} references curve '{value}' "
                        f"which was not found. {e}"
                    ) from None
            return value

        # Collect `*args` arguments (positionals):
        var_args = [
            deserialize_arg(n, shape_def.pop(n), t) for n, t in self.args.positionals
        ] + [
            a
            for n, t in self.args.var_positionals
            for a in [deserialize_arg(n, val, t) for val in shape_def.pop(n)]
        ]
        # Collect keyword arguments:
        kwarg_defs = {key: val for key, val in self.args.kwargs}
        kwargs = {
            key: deserialize_arg(key, val, kwarg_defs[key])
            if key in kwarg_defs
            else val
            for key, val in shape_def.items()
        }

        return self.shape_factory(*var_args, **kwargs)


def craft_example(
    shape: str,
    predicate: Callable[[Any], bool],
    args: ShapeFactoryArgs,
    dim: int | None,
) -> str:
    """Create user friendly examples if a parameter is missing in the shape def."""
    examples = {2: "my_circle", 3: "my_sphere", None: "my_shape"}
    var_arg_shapes = (n for n, t in args.var_positionals if predicate(t))
    kwarg_shapes = (n for n, t in args.positionals + args.kwargs if predicate(t))
    return f"""my_{shape}:
  type: {shape}
  """ + "\n  ".join(
        [f"{a}: [{examples[dim]}_1, {examples[dim]}_2]" for a in var_arg_shapes]
        + [f"{a}: {examples[dim]}" for a in kwarg_shapes]
    )


SHAPE_LOADERS: dict[str, ShapeLoader] = {}
"""All shape loaders registred by the @loadable_shape decorator."""


def loadable_shape(
    *,
    feed_positionals_from: str | None = None,
    inject: tuple[str, ...] = (),
    skip_validation: bool = False,
    input_shape_dim: int | None = None,
    check_shape_def: Callable[[dict[str, Any]], None] | None = None,
    deserialize_args: dict[str, Callable[[Any], Any]] | None = None,
) -> Callable[[Callable[..., Shape]], Callable[..., Shape]]:
    """Register a shape factory so it can be loaded from a yaml file.

    Args:
        feed_positionals_from: If specified, the value will be extracted from the
            shape def and passed as `*args` to the shape factory.
        inject: Inject additional arguments from the scope of `CADProgram`.
        skip_validation: Skip additional validation of shape defs happening before instantiation of the shape.
        input_shape_dim: Expected dimension of input shapes. Will only be used in error messages.
        check_shape_def: Custom assertions performed during instantiation of the shape.
        deserialize_args: Custom deserialization of specific arguments.
            By default all shapes and curves will be looked up and passed to the shape factory.
    """

    def decorator(shape_factory: Callable[..., Shape]) -> Callable[..., Shape]:
        name = shape_factory.__name__
        args = ShapeFactoryArgs.from_signature(
            shape_factory, feed_positionals_from=feed_positionals_from
        )
        node_schema = NodeSchema.from_shape_factory_args(args)
        SHAPE_LOADERS[name] = ShapeLoader(
            shape_factory,
            args=args,
            inject=inject,
            input_shape_dim=input_shape_dim,
            check_shape_def=check_shape_def,
            deserialize_args=deserialize_args,
        )
        NODE_SCHEMA[name] = node_schema if not skip_validation else None
        return shape_factory

    return decorator


def _apply_func(name: str, arg):
    """Apply a whitelisted math function, dispatching to torch for tensors."""
    import torch

    # sin/cos/tan take degrees in YAML expressions (matches rotate convention)
    if name in ("sin", "cos", "tan"):
        arg = arg * _DEG_TO_RAD
    if isinstance(arg, torch.Tensor):
        return getattr(torch, name)(arg)
    if name == "abs":
        return abs(arg)
    return getattr(math, name)(arg)


def _eval_param_expr(expr: str, params: dict):
    """Evaluate arithmetic expression with $param references.

    Returns float (YAML params) or Tensor (batched with_params).
    Uses ast.parse, no eval()/exec().

    Supported operators: +, -, *, /, ^/**,  unary -, parentheses.
    Supported functions: sin, cos, tan (degrees), sqrt, abs.
    Supported constants: pi.

    Args:
        expr: Expression string, e.g. "($a + $b) / 2 - $c"
        params: Dict mapping parameter names to float or Tensor values

    Returns:
        Evaluated result (float or Tensor)

    Raises:
        ValueError: If expression contains unknown params or unsafe AST nodes
    """
    # Replace ^ with ** (power operator convention from expression.py)
    expr = expr.replace("^", "**")

    # Extract all $param_name tokens and validate they exist
    param_refs = re.findall(r"\$([a-zA-Z_]\w*)", expr)
    for ref in param_refs:
        if ref not in params:
            raise ValueError(
                f"Unknown parameter '${ref}' in expression '{expr}'. "
                f"Available params: {list(params.keys())}"
            )

    # Replace $param_name with safe placeholder identifiers
    safe_expr = re.sub(r"\$([a-zA-Z_]\w*)", r"_p_\1", expr)

    # Parse with ast
    try:
        tree = ast.parse(safe_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid expression syntax: '{expr}' — {e}")

    # Validate AST whitelist and evaluate
    def _eval_node(node):
        if isinstance(node, ast.Expression):
            return _eval_node(node.body)
        if isinstance(node, ast.BinOp):
            left = _eval_node(node.left)
            right = _eval_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(
                f"Unsupported operator {type(node.op).__name__} in expression '{expr}'"
            )
        if isinstance(node, ast.UnaryOp):
            operand = _eval_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return +operand
            raise ValueError(
                f"Unsupported unary operator {type(node.op).__name__} in expression '{expr}'"
            )
        if isinstance(node, ast.Call):
            # Whitelisted function calls: sin, cos, tan, sqrt, abs
            if not isinstance(node.func, ast.Name):
                raise ValueError(
                    f"Only simple function calls allowed in expression '{expr}'"
                )
            func_name = node.func.id
            if func_name not in _EXPR_FUNCTIONS:
                raise ValueError(
                    f"Function '{func_name}' not allowed in expression '{expr}'. "
                    f"Allowed: {sorted(_EXPR_FUNCTIONS)}"
                )
            if len(node.args) != 1 or node.keywords:
                raise ValueError(
                    f"Function '{func_name}' takes exactly 1 argument in expression '{expr}'"
                )
            arg = _eval_node(node.args[0])
            return _apply_func(func_name, arg)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return float(node.value)
        if isinstance(node, ast.Name):
            if node.id.startswith("_p_"):
                param_name = node.id[3:]  # strip _p_ prefix
                return params[param_name]
            if node.id in _EXPR_CONSTANTS:
                return _EXPR_CONSTANTS[node.id]
            raise ValueError(
                f"Unknown name '{node.id}' in expression '{expr}'. "
                f"Use $name for parameters. Available constants: {sorted(_EXPR_CONSTANTS)}"
            )
        raise ValueError(
            f"Unsafe or unsupported expression node {type(node).__name__} "
            f"in expression '{expr}'"
        )

    return _eval_node(tree)


def substitute_params(obj: Any, params: dict[str, Any]) -> Any:
    """
    Recursively substitute $param_name with values from params dict.

    Handles three string cases:
    - Simple "$param" -> direct value lookup
    - Expression with params "$a + $b * 2" -> AST-based evaluation
    - Pure numeric expression "3/2" -> AST-based evaluation

    Args:
        obj: The object to process (can be dict, list, str, or primitive)
        params: Dictionary mapping parameter names to values

    Returns:
        Object with all $param references replaced by their values

    Examples:
        >>> substitute_params("$radius", {"radius": 1.0})
        1.0
        >>> substitute_params(["$x", "$y"], {"x": 1, "y": 2})
        [1, 2]
        >>> substitute_params({"r": "$radius"}, {"radius": 1.0})
        {"r": 1.0}
        >>> substitute_params({"center": ["$cx", "$cy", "$cz"]}, {"cx": 0, "cy": 0, "cz": 0})
        {"center": [0, 0, 0]}

    Raises:
        ValueError: If a parameter reference is not found in params
    """
    if isinstance(obj, str):
        # Simple case: entire string is a single param reference
        if obj.startswith("$") and obj[1:].isidentifier() and obj[1:] in params:
            return params[obj[1:]]
        # Expression case: string contains $param references (possibly with arithmetic)
        if "$" in obj:
            return _eval_param_expr(obj, params)
        # Pure expression without $params (e.g. "3/2", "sin(pi/4)", "pi")
        if obj in _EXPR_CONSTANTS:
            return _EXPR_CONSTANTS[obj]
        if re.fullmatch(r"[\w\s\.\+\-\*/\^()]+", obj) and re.search(r"[+\-*/^()]", obj):
            return _eval_param_expr(obj, params)
        return obj

    if isinstance(obj, list):
        return [substitute_params(item, params) for item in obj]

    if isinstance(obj, dict):
        # Make a copy to avoid modifying the original
        return {k: substitute_params(v, params) for k, v in obj.items()}

    # Numbers, bools, None - return as-is
    return obj


NODE_SCHEMA: dict[str, NodeSchema | None] = {
    # New types (not yet in graph format, skip for now)
    "bezier_surface": None,
}


# Params that are always scalar when used as a direct value (not inside a list)
_SCALAR_PARAMS = {
    "radius",
    "angle",
    "k",
    "min",
    "max",
    "start_angle",
    "end_angle",
    "offset",
}


def _extract_binding_params(config: dict[str, Any]) -> dict[str, Any]:
    """Extract default param values from bindings map entries.

    Each binding map entry is ``{col_index: {param_name: default_value}}``.
    Returns a flat dict of ``{param_name: default_value}`` aggregated across
    all bindings.

    Args:
        config: Configuration dictionary (may or may not have 'bindings')

    Returns:
        Dict mapping param names to their default values
    """
    params: dict[str, Any] = {}
    for binding_spec in config.get("bindings", {}).values():
        for col_idx, entry in binding_spec.get("map", {}).items():
            name, default = next(iter(entry.items()))
            params[name] = default
    return params


def _eval_binding_derived(
    config: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Evaluate all binding derived expressions against current params.

    Derived expressions are evaluated in YAML order and can reference
    earlier derived values. Works with both scalar defaults and tensor
    values (from ``with_bindings()``).

    Args:
        config: Configuration dictionary (may or may not have 'bindings')
        params: Current param values (scalars or tensors)

    Returns:
        New params dict with derived values added/updated
    """
    params = dict(params)  # copy
    for binding_spec in config.get("bindings", {}).values():
        for param_name, expr in binding_spec.get("derived", {}).items():
            params[param_name] = _eval_param_expr(expr, params)
    return params


def _validate_bindings(config: dict[str, Any]) -> None:
    """Validate bindings section of a YAML config.

    Checks:
    - ``shape:`` is a 2-element list ``["B", int]``
    - ``map:`` keys are integers in range ``[0, shape[1])``
    - Each map value is a single-entry dict ``{param_name: default}``
    - ``derived:`` values are strings (expression format)
    - No duplicate param names across map + derived within one binding
    - No overlap between binding param names and ``params:`` section keys

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If any binding definition is invalid
    """
    bindings = config.get("bindings", {})
    if not bindings:
        return

    top_params = set(config.get("params", {}).keys())
    all_binding_params: set[str] = set()

    for bind_name, spec in bindings.items():
        # shape validation
        shape = spec.get("shape")
        if shape is None:
            raise ValueError(
                f"Binding '{bind_name}' missing required 'shape' field. "
                f"Example: shape: [B, 15]"
            )
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or shape[0] != "B"
            or not isinstance(shape[1], int)
            or shape[1] < 1
        ):
            raise ValueError(
                f"Binding '{bind_name}' shape must be [B, <positive int>], got: {shape}"
            )
        num_cols = shape[1]

        # map validation
        mapping = spec.get("map", {})
        local_params: set[str] = set()

        claimed_cols: set = set()  # track which columns are claimed

        for col_key, entry in mapping.items():
            col_str = str(col_key)
            if ":" in col_str:
                # Range notation: "start:end" (exclusive end)
                parts = col_str.split(":")
                if len(parts) != 2:
                    raise ValueError(
                        f"Binding '{bind_name}' map key '{col_str}' is not a "
                        f"valid range. Use 'start:end' format."
                    )
                start, end = int(parts[0]), int(parts[1])
                if start < 0 or end > num_cols or start >= end:
                    raise ValueError(
                        f"Binding '{bind_name}' map range {col_str} out of "
                        f"range [0, {num_cols}). shape declares {num_cols} "
                        f"columns."
                    )
                col_set = set(range(start, end))
            else:
                col_idx = int(col_str)
                if col_idx < 0 or col_idx >= num_cols:
                    raise ValueError(
                        f"Binding '{bind_name}' map column {col_idx} out of "
                        f"range [0, {num_cols}). shape declares {num_cols} "
                        f"columns."
                    )
                col_set = {col_idx}

            # Check for overlapping columns
            overlap = col_set & claimed_cols
            if overlap:
                raise ValueError(
                    f"Binding '{bind_name}' map key '{col_str}' overlaps "
                    f"with previously claimed column(s) {sorted(overlap)}."
                )
            claimed_cols |= col_set

            if not isinstance(entry, dict) or len(entry) != 1:
                raise ValueError(
                    f"Binding '{bind_name}' map[{col_str}] must be a "
                    f"single-entry dict {{param_name: default}}, got: {entry}"
                )
            param_name = next(iter(entry.keys()))

            # Validate default list length for range bindings
            if ":" in col_str:
                default_val = next(iter(entry.values()))
                expected_len = end - start
                if not isinstance(default_val, list):
                    raise ValueError(
                        f"Binding '{bind_name}' map[{col_str}] default must "
                        f"be a list of length {expected_len} for range "
                        f"binding, got: {type(default_val).__name__}"
                    )
                if len(default_val) != expected_len:
                    raise ValueError(
                        f"Binding '{bind_name}' map[{col_str}] default list "
                        f"length {len(default_val)} doesn't match range width "
                        f"{expected_len}."
                    )

            if param_name in local_params:
                raise ValueError(
                    f"Binding '{bind_name}' has duplicate param '{param_name}' in map."
                )
            local_params.add(param_name)

        # derived validation
        derived = spec.get("derived", {})
        for param_name, expr in derived.items():
            if not isinstance(expr, str):
                raise ValueError(
                    f"Binding '{bind_name}' derived '{param_name}' must be a "
                    f"string expression, got: {type(expr).__name__}"
                )
            if param_name in local_params:
                raise ValueError(
                    f"Binding '{bind_name}' has duplicate param '{param_name}' "
                    f"(appears in both map and derived)."
                )
            local_params.add(param_name)

        # cross-binding and top-level overlap
        overlap_top = local_params & top_params
        if overlap_top:
            raise ValueError(
                f"Binding '{bind_name}' param(s) {sorted(overlap_top)} also "
                f"appear in top-level 'params:' section. Bound params must "
                f"only be defined in bindings."
            )
        overlap_other = local_params & all_binding_params
        if overlap_other:
            raise ValueError(
                f"Binding '{bind_name}' param(s) {sorted(overlap_other)} "
                f"already declared in another binding."
            )
        all_binding_params |= local_params


def validate_bounds(config: dict[str, Any]) -> None:
    """
    Validate that configuration has required bounds fields.

    Extracted from validate_config() for use independently.

    Args:
        config: Configuration dictionary from YAML

    Raises:
        ValueError: If required bounds fields are missing or invalid
    """
    if "bounds" not in config:
        raise ValueError(
            "Missing required 'bounds' section in YAML config.\n"
            "Please add bounds specification:\n\n"
            "bounds:\n"
            "  x: [-2.0, 2.0]\n"
            "  y: [-2.0, 2.0]\n"
            "  z: [-2.0, 2.0]\n"
            "  resolution: 512\n"
        )

    bounds = config["bounds"]

    for axis in ["x", "y", "z"]:
        if axis not in bounds:
            raise ValueError(
                f"Missing '{axis}' in bounds section. Example: {axis}: [-2.0, 2.0]"
            )

        axis_bounds = bounds[axis]

        if not isinstance(axis_bounds, list) or len(axis_bounds) != 2:
            raise ValueError(
                f"bounds.{axis} must be a list of 2 numbers [min, max], got: {axis_bounds}"
            )

        for i, val in enumerate(axis_bounds):
            if not isinstance(val, (int, float)):
                raise ValueError(
                    f"bounds.{axis}[{i}] must be a number, got: {val} (type: {type(val).__name__})"
                )

        if axis_bounds[1] <= axis_bounds[0]:
            raise ValueError(
                f"bounds.{axis}[1] (max={axis_bounds[1]}) must be greater than "
                f"bounds.{axis}[0] (min={axis_bounds[0]})"
            )

    if "resolution" not in bounds:
        raise ValueError(
            "Missing 'resolution' in bounds section. Example: resolution: 512"
        )

    resolution = bounds["resolution"]
    if not isinstance(resolution, int) or resolution <= 0:
        raise ValueError(f"resolution must be a positive integer, got: {resolution}")

    if "batch_size" not in bounds:
        raise ValueError(
            "Missing required 'batch_size' in bounds section.\n"
            "Please add batch_size specification:\n\n"
            "bounds:\n"
            "  x: [-2.0, 2.0]\n"
            "  y: [-2.0, 2.0]\n"
            "  z: [-2.0, 2.0]\n"
            "  resolution: 512\n"
            "  batch_size: 1  # REQUIRED\n"
        )

    batch_size = bounds["batch_size"]
    if not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError(
            f"batch_size must be a positive integer (>= 1), got: {batch_size}"
        )


def validate_shapes(config: dict[str, Any]) -> None:
    """
    Validate shape definitions in config against NODE_SCHEMA.

    Performs 4 checks:
        a) Per-node schema, every key must be allowed for that node type
        b) Structural child validation, shape refs must exist and be valid
        c) Orphan detection, all shapes must be reachable from output
        d) Variable consistency, $var must not mix scalar and vector roles

    Args:
        config: Configuration dictionary (must have 'shapes' and 'output')

    Raises:
        ValueError: If any shape definition is invalid
    """
    shapes = config.get("shapes", {})
    if not shapes:
        return

    shape_names = set(shapes.keys())

    # Collect import namespaces for dot-notation ref validation
    import_namespaces = set()
    imports = config.get("import", {})
    if isinstance(imports, dict):
        import_namespaces = set(imports.keys())

    def _is_imported_ref(ref: str) -> bool:
        """Check if a shape reference is a dot-notation import ref."""
        if "." not in ref:
            return False
        ns = ref.split(".")[0]
        return ns in import_namespaces

    # (a) Per-node schema check
    for name, shape_def in shapes.items():
        if "type" not in shape_def:
            raise ValueError(f"Shape '{name}' missing required 'type' key.")

        stype = shape_def["type"]
        if stype not in NODE_SCHEMA:
            raise ValueError(
                f"Shape '{name}' has unknown type '{stype}'. "
                f"Known types: {sorted(NODE_SCHEMA.keys())}"
            )

        schema = NODE_SCHEMA[stype]
        if schema is None:
            continue  # skip validation for formula/neural/etc.

        allowed = schema.params | set(schema.shape_refs) | {"type"}
        actual = set(shape_def.keys())
        extra = actual - allowed
        if extra:
            allowed_display = sorted(schema.params | set(schema.shape_refs))
            raise ValueError(
                f"'{stype}' got unexpected keys: {extra}. "
                f"Allowed: {', '.join(allowed_display)}"
            )

        # (b) Structural child validation
        for ref_key in schema.shape_refs:
            if ref_key == "shape":
                if "shape" not in shape_def:
                    raise ValueError(
                        f"'{stype}' (shape '{name}') requires 'shape' "
                        f"referencing another shape."
                    )
                ref_val = shape_def["shape"]
                if not isinstance(ref_val, str):
                    raise ValueError(
                        f"'{stype}' (shape '{name}') references shape "
                        f"'{ref_val}' which was not found in shapes. "
                        f"Available: {sorted(shape_names)}"
                    )
                # Skip validation for imported refs (resolved at build time)
                if not _is_imported_ref(ref_val) and ref_val not in shape_names:
                    raise ValueError(
                        f"'{stype}' (shape '{name}') references shape "
                        f"'{ref_val}' which was not found in shapes. "
                        f"Available: {sorted(shape_names)}"
                    )
            elif ref_key == "shapes":
                if "shapes" not in shape_def:
                    raise ValueError(
                        f"'{stype}' (shape '{name}') requires 'shapes' list."
                    )
                ref_list = shape_def["shapes"]
                if not isinstance(ref_list, list) or len(ref_list) < 2:
                    raise ValueError(
                        f"'{stype}' (shape '{name}') requires at least 2 "
                        f"entries in 'shapes' list, got {len(ref_list) if isinstance(ref_list, list) else 0}."
                    )
                for child_ref in ref_list:
                    if not isinstance(child_ref, str):
                        raise ValueError(
                            f"'{stype}' (shape '{name}') references shape "
                            f"'{child_ref}' which was not found in shapes. "
                            f"Available: {sorted(shape_names)}"
                        )
                    # Skip validation for imported refs
                    if not _is_imported_ref(child_ref) and child_ref not in shape_names:
                        raise ValueError(
                            f"'{stype}' (shape '{name}') references shape "
                            f"'{child_ref}' which was not found in shapes. "
                            f"Available: {sorted(shape_names)}"
                        )

    # (c) Orphan detection
    output = config.get("output")
    if output is not None:
        # Extract root shape refs from all output forms
        if isinstance(output, dict):
            roots = list(output.values())
        elif isinstance(output, list):
            roots = list(output)
        else:
            roots = [output]

        visited: set[str] = set()

        def _walk(name: str) -> None:
            # Skip imported refs (ns.shape), they're external
            if "." in name and name.split(".")[0] in import_namespaces:
                return
            if name in visited or name not in shapes:
                return
            visited.add(name)
            sdef = shapes[name]
            stype = sdef.get("type")
            schema = NODE_SCHEMA.get(stype) if stype else None
            if schema is None:
                # For None-schema types, walk any key that looks like a ref
                for v in sdef.values():
                    if isinstance(v, str) and v in shape_names:
                        _walk(v)
                    elif isinstance(v, list):
                        for item in v:
                            if isinstance(item, str) and item in shape_names:
                                _walk(item)
                return
            for ref_key in schema.shape_refs:
                val = sdef.get(ref_key)
                if isinstance(val, str):
                    _walk(val)
                elif isinstance(val, list):
                    for item in val:
                        if isinstance(item, str):
                            _walk(item)

        for root in roots:
            _walk(root)

        orphans = shape_names - visited
        if orphans:
            raise ValueError(
                f"Orphan shape(s) not reachable from output: {sorted(orphans)}"
            )

    # (d) Variable consistency
    var_roles: dict[
        str, set[str]
    ] = {}  # var_name -> {'scalar_param', 'vector_component'}

    for name, shape_def in shapes.items():
        stype = shape_def.get("type")
        schema = NODE_SCHEMA.get(stype) if stype else None
        if schema is None:
            continue

        for key, val in shape_def.items():
            if key == "type" or key in schema.shape_refs:
                continue
            if isinstance(val, str) and "$" in val:
                for match in re.finditer(r"\$([a-zA-Z_]\w*)", val):
                    var_roles.setdefault(match.group(1), set()).add("scalar_param")
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and "$" in item:
                        for match in re.finditer(r"\$([a-zA-Z_]\w*)", item):
                            var_roles.setdefault(match.group(1), set()).add(
                                "vector_component"
                            )

    for var_name, roles in var_roles.items():
        if len(roles) > 1:
            raise ValueError(
                f"Variable '${var_name}' used as both scalar param and "
                f"vector component — mixed roles are not allowed."
            )

    # Orphan params check (moved from validate_config)
    validate_no_orphan_params(config)


def validate_imports(config: dict[str, Any]) -> None:
    """
    Validate the 'import' section of an assembly YAML.

    Checks that:
        - import: is a dict
        - Each value is a string (file path) or dict with 'file' key
        - Namespace names are valid identifiers (no dots)

    Args:
        config: Configuration dictionary from YAML

    Raises:
        ValueError: If import section is malformed
    """
    imports = config.get("import")
    if imports is None:
        return

    if not isinstance(imports, dict):
        raise ValueError(
            f"'import' must be a dict mapping namespace -> file path, "
            f"got {type(imports).__name__}"
        )

    for ns, spec in imports.items():
        if not isinstance(ns, str) or "." in ns or not ns.isidentifier():
            raise ValueError(
                f"Import namespace '{ns}' must be a valid identifier without dots "
                f"(letters, digits, underscores, not starting with a digit). "
                f"Dots are used as separators for accessing imported shapes."
            )
        if isinstance(spec, str):
            continue  # short form: ns: filepath
        if isinstance(spec, dict):
            if "file" not in spec:
                raise ValueError(
                    f"Import '{ns}' uses dict form but missing required 'file' key. "
                    f"Expected: {ns}: {{file: path.yaml, params: {{...}}}}"
                )
        else:
            raise ValueError(
                f"Import '{ns}' must be a file path string or a dict with 'file' key, "
                f"got {type(spec).__name__}"
            )


def validate_config(config: dict[str, Any]) -> None:
    """
    Validate full configuration: bounds + shapes + imports.

    Backwards-compatible wrapper that calls validate_bounds(),
    validate_shapes(), and validate_imports().

    Args:
        config: Configuration dictionary from YAML

    Raises:
        ValueError: If required fields are missing or invalid
    """
    validate_bounds(config)
    validate_imports(config)
    _validate_bindings(config)
    validate_shapes(config)


def _find_var_references(obj: Any, refs: set[str]) -> None:
    """Recursively find all $var references in an object."""
    if isinstance(obj, str) and "$" in obj:
        for match in re.finditer(r"\$([a-zA-Z_]\w*)", obj):
            refs.add(match.group(1))
    elif isinstance(obj, list):
        for item in obj:
            _find_var_references(item, refs)
    elif isinstance(obj, dict):
        for v in obj.values():
            _find_var_references(v, refs)


def validate_no_orphan_params(config: dict[str, Any]) -> None:
    """
    Validate that all defined params/constants are referenced in shapes.

    Binding-derived params are included in the "defined" set, and binding
    derived expressions are searched for $param references.

    Raises:
        ValueError: If any params are defined but never referenced
    """
    params = config.get("params", {})
    constants = config.get("constants", {})
    shapes = config.get("shapes", {})

    # Include binding-derived params in the defined set
    binding_param_names: set[str] = set()
    for binding_spec in config.get("bindings", {}).values():
        for entry in binding_spec.get("map", {}).values():
            binding_param_names.add(next(iter(entry.keys())))
        for pname in binding_spec.get("derived", {}).keys():
            binding_param_names.add(pname)

    defined = set(params.keys()) | set(constants.keys()) | binding_param_names
    if not defined:
        return  # no params to validate

    referenced: set[str] = set()
    _find_var_references(shapes, referenced)
    _find_var_references(config.get("curves", {}), referenced)
    _find_var_references(config.get("workplanes", {}), referenced)

    # Search binding derived expressions for $param references
    for binding_spec in config.get("bindings", {}).values():
        for expr in binding_spec.get("derived", {}).values():
            _find_var_references(expr, referenced)

    # Binding map params that are referenced in shapes or derived count as used
    # Binding derived params that are referenced in shapes count as used
    # Params only used internally by bindings (map -> derived) are still "used"
    # because the binding system consumes them at runtime.
    # Mark all binding map params as referenced (they're consumed by with_bindings)
    referenced |= binding_param_names

    orphans = defined - referenced
    if orphans:
        names = ", ".join(f"'{v}'" for v in sorted(orphans))
        raise ValueError(
            f"Parameter(s) defined but never referenced: {names}. "
            f"Remove unused params or reference them with '$name' in shapes."
        )


def deep_copy_config(config: dict[str, Any]) -> dict[str, Any]:
    """
    Create a deep copy of a configuration dictionary.

    This is useful when you want to modify a config without affecting
    the original (e.g., when building shapes with modified params).

    Args:
        config: Configuration dictionary to copy

    Returns:
        Deep copy of the configuration
    """
    return copy.deepcopy(config)


def assert_at_least_2_shapes(name: str) -> Callable[[dict[str, Any]], None]:
    """To be used as the argument `check_shape_args` of `loadable_shape`."""

    def check(shape_def: dict[str, Any]) -> None:
        shapes = shape_def["shapes"]
        if len(shapes) < 2:
            raise ValueError(
                f"{name} requires at least 2 shapes in 'shapes' list, "
                f"got {len(shapes)}. Example: shapes: [shape1, shape2]"
            )

    return check
