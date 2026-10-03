"""
Expression Parser - Convert string formulas to PyTorch SDF functions.

Four-phase pipeline:
1. Preprocessing: Replace $param with __param, validate params exist
2. AST Validation: Parse with ast.parse(), validate whitelist operations
3. PyTorch Codegen: Convert AST to executable PyTorch function
4. Primitive Integration: Wrap in Shape with batch_size

Security: Uses ast.parse() with whitelist validation - NEVER eval().
"""

import ast
import re
from collections.abc import Callable

import torch
from torch import Tensor

# Preprocessing


def replace_power_operator(expression: str) -> str:
    """
    Replace caret (^) with Python power operator (**).

    The caret symbol is commonly used for exponentiation in mathematical
    notation, but Python uses ** for power and ^ for bitwise XOR.
    This function converts ^ to ** before AST parsing.

    Args:
        expression: Mathematical expression potentially containing ^

    Returns:
        Expression with ^ replaced by **

    Examples:
        >>> replace_power_operator("x^2 + y^3")
        "x**2 + y**3"
        >>> replace_power_operator("(x + y)^0.5")
        "(x + y)**0.5"

    Note:
        This is a simple character replacement. If you need XOR operation,
        use the 'xor()' function (not yet implemented).
    """
    # Simple replacement: all occurrences of ^ become **
    # This is safe because we don't support bitwise operations in whitelist
    return expression.replace("^", "**")


def preprocess_params(expression: str, params: dict[str, Tensor]) -> tuple:
    """
    Replace $param with __param, validate params exist, replace ^ with **.

    Args:
        expression: String expression like "sqrt(x*x) - $radius"
        params: Dict of parameter tensors {name: [B, K] tensor}

    Returns:
        (preprocessed_expr, param_map)
        Example: ("sqrt(x*x) - __radius", {"__radius": "radius"})

    Raises:
        ValueError: If referenced parameter doesn't exist
    """
    # NEW: Replace ^ with ** for power operator (BEFORE parameter substitution)
    # This must happen first to avoid interfering with parameter names
    expression = replace_power_operator(expression)

    param_map = {}
    pattern = r"\$([a-zA-Z_][a-zA-Z0-9_]*)"

    # Find all $param_name tokens
    for match in re.finditer(pattern, expression):
        param_name = match.group(1)

        # Validate parameter exists (eager validation)
        if param_name not in params:
            available = list(params.keys())
            raise ValueError(
                f"Unknown parameter '${param_name}'. Available parameters: {available}"
            )

        param_map[f"__{param_name}"] = param_name

    # Replace $param with __param
    expr_preprocessed = re.sub(pattern, r"__\1", expression)

    return expr_preprocessed, param_map


# AST Validation


def validate_ast(
    node: ast.AST, param_map: dict[str, str], allowed_vars: list[str]
) -> None:
    """
    Recursively validate AST contains only safe operations.

    Whitelist:
        - BinOp: Add, Sub, Mult, Div, Pow
        - UnaryOp: USub (negation)
        - Call: sqrt, abs, sin, cos, tan, atan, atan2, min, max, softmin, softmax, softabs
        - Name: x, y, z, __param_name
        - Constant: numbers
        - Expression: top-level wrapper

    Args:
        node: AST node to validate
        param_map: Map of __param_name -> param_name
        allowed_vars: Which variables are allowed (e.g., ['x', 'y'] for 2D)

    Raises:
        ValueError: If unsupported operation found
    """
    if isinstance(node, ast.BinOp):
        allowed_ops = (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)
        if not isinstance(node.op, allowed_ops):
            raise ValueError(
                f"Unsupported binary operator: {node.op.__class__.__name__}"
            )
        validate_ast(node.left, param_map, allowed_vars)
        validate_ast(node.right, param_map, allowed_vars)

    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub):
            raise ValueError(
                f"Unsupported unary operator: {node.op.__class__.__name__}"
            )
        validate_ast(node.operand, param_map, allowed_vars)

    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Unsupported function call")

        func_name = node.func.id
        num_args = len(node.args)

        # Define allowed functions with their argument counts
        ALLOWED_FUNCTIONS = {
            # Basic math
            "sqrt": 1,
            "abs": 1,
            # Trigonometry
            "sin": 1,
            "cos": 1,
            "tan": 1,
            "atan": 1,
            "atan2": 2,
            # Min/Max (hard)
            "min": (2, None),  # 2 or more arguments
            "max": (2, None),  # 2 or more arguments
            # Smooth operations
            "softmin": 3,  # (a, b, k)
            "softmax": 3,  # (a, b, k)
            "softabs": 2,  # (x, k)
        }

        if func_name not in ALLOWED_FUNCTIONS:
            supported = ", ".join(sorted(ALLOWED_FUNCTIONS.keys()))
            raise ValueError(
                f"Unsupported function: {func_name}. Supported functions: {supported}"
            )

        # Validate argument count
        expected = ALLOWED_FUNCTIONS[func_name]
        if isinstance(expected, tuple):  # Variable argument count
            min_args, max_args = expected
            if num_args < min_args:
                raise ValueError(
                    f"{func_name}() requires at least {min_args} arguments, got {num_args}"
                )
            if max_args is not None and num_args > max_args:
                raise ValueError(
                    f"{func_name}() requires at most {max_args} arguments, got {num_args}"
                )
        elif num_args != expected:
            raise ValueError(
                f"{func_name}() takes {expected} argument(s), got {num_args}"
            )

        # Recursively validate all arguments
        for arg in node.args:
            validate_ast(arg, param_map, allowed_vars)

    elif isinstance(node, ast.Name):
        # Check if it's a valid variable or parameter
        all_allowed_vars = allowed_vars + ["x", "y", "z"]
        if node.id in all_allowed_vars:
            pass  # Valid variable
        elif node.id in param_map:
            pass  # Valid parameter
        else:
            raise ValueError(
                f"Unknown variable: {node.id}. Allowed variables: {allowed_vars}"
            )

    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise ValueError(
                f"Unsupported constant type: {type(node.value).__name__}. "
                f"Only numeric constants are allowed."
            )

    elif isinstance(node, ast.Expression):
        validate_ast(node.body, param_map, allowed_vars)

    else:
        raise ValueError(f"Unsupported operation: {node.__class__.__name__}")


def detect_variables(tree: ast.AST) -> set:
    """
    Walk AST and collect all variable names (x, y, z).

    Args:
        tree: AST to walk

    Returns:
        Set of variable names used in expression
    """
    variables = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id in ["x", "y", "z"]:
            variables.add(node.id)
    return variables


# PyTorch Code Generation


def _softmin(a: Tensor, b: Tensor, k: Tensor) -> Tensor:
    """
    Smooth minimum using log-sum-exp approximation (numerically stable).

    Formula: softmin(a, b, k) = -log(exp(-k*a) + exp(-k*b)) / k

    Properties:
    - As k -> ∞, softmin -> min (hard minimum)
    - As k -> 0, softmin -> (a + b) / 2 (average)
    - Fully differentiable everywhere

    Args:
        a: First value [B, N]
        b: Second value [B, N]
        k: Smoothing parameter [B, N], [B, 1], or scalar
            Higher k = sharper transition

    Returns:
        Smooth minimum [B, N]

    Note:
        Uses numerically stable log-sum-exp implementation with max-factoring
        to prevent overflow in exponentials.
    """
    # Numerically stable: factor out maximum to prevent overflow
    neg_ka = -k * a
    neg_kb = -k * b
    max_val = torch.max(neg_ka, neg_kb)

    result = (
        -torch.log(torch.exp(neg_ka - max_val) + torch.exp(neg_kb - max_val)) / k
        - max_val / k
    )

    return result


def _softmax(a: Tensor, b: Tensor, k: Tensor) -> Tensor:
    """
    Smooth maximum using log-sum-exp approximation (numerically stable).

    Formula: softmax(a, b, k) = log(exp(k*a) + exp(k*b)) / k

    Properties:
    - As k -> ∞, softmax -> max (hard maximum)
    - As k -> 0, softmax -> (a + b) / 2 (average)
    - Fully differentiable everywhere

    Args:
        a: First value [B, N]
        b: Second value [B, N]
        k: Smoothing parameter [B, N], [B, 1], or scalar
            Higher k = sharper transition

    Returns:
        Smooth maximum [B, N]

    Note:
        Uses numerically stable log-sum-exp implementation with max-factoring
        to prevent overflow in exponentials.
    """
    # Numerically stable: factor out maximum to prevent overflow
    ka = k * a
    kb = k * b
    max_val = torch.max(ka, kb)

    result = (
        torch.log(torch.exp(ka - max_val) + torch.exp(kb - max_val)) / k + max_val / k
    )

    return result


def _softabs(x: Tensor, k: Tensor) -> Tensor:
    """
    Smooth absolute value approximation.

    Formula: softabs(x, k) = sqrt(x^2 + 1/k^2) - 1/k

    Properties:
    - As k -> ∞, softabs -> abs (hard absolute value)
    - As k -> 0, softabs -> |x| with very rounded corner
    - Fully differentiable everywhere (including x=0, where gradient=0)

    Args:
        x: Input value [B, N]
        k: Smoothing parameter [B, N], [B, 1], or scalar
            Higher k = sharper transition at x=0

    Returns:
        Smooth absolute value [B, N]

    Note:
        This is the smooth approximation commonly used in SDF implementations.
        The term 1/k^2 is added under the square root to ensure smoothness.
    """
    epsilon = 1.0 / (k * k + 1e-10)  # Add small constant to prevent division by zero
    # Ensure epsilon is a tensor (handles case where k is a Python scalar)
    if not isinstance(epsilon, Tensor):
        epsilon = torch.tensor(epsilon, dtype=x.dtype, device=x.device)
    return torch.sqrt(x * x + epsilon) - torch.sqrt(epsilon)


def _deg_to_rad(degrees: Tensor) -> Tensor:
    """Convert degrees to radians, preserving device and dtype."""
    pi = torch.tensor(torch.pi, device=degrees.device, dtype=degrees.dtype)
    return degrees * (pi / 180.0)


def _rad_to_deg(radians: Tensor) -> Tensor:
    """Convert radians to degrees, preserving device and dtype."""
    pi = torch.tensor(torch.pi, device=radians.device, dtype=radians.dtype)
    return radians * (180.0 / pi)


def ast_to_pytorch(tree: ast.AST, param_map: dict[str, str]) -> Callable:
    """
    Convert validated AST to executable PyTorch function.

    Returns function: sdf_fn(points: [N, 3], params: dict) -> [B, N]

    Args:
        tree: Validated AST
        param_map: Map of __param_name -> param_name

    Returns:
        Callable that evaluates the expression on query points
    """
    # Detect if division used (for warning)
    has_division = any(
        isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div)
        for node in ast.walk(tree)
    )

    # Build nested evaluation function
    def _build_eval_fn(node: ast.AST):
        """Recursively build evaluation function from AST node."""

        if isinstance(node, ast.BinOp):
            left_fn = _build_eval_fn(node.left)
            right_fn = _build_eval_fn(node.right)

            if isinstance(node.op, ast.Add):
                return lambda coords, params: (
                    left_fn(coords, params) + right_fn(coords, params)
                )
            if isinstance(node.op, ast.Sub):
                return lambda coords, params: (
                    left_fn(coords, params) - right_fn(coords, params)
                )
            if isinstance(node.op, ast.Mult):
                return lambda coords, params: (
                    left_fn(coords, params) * right_fn(coords, params)
                )
            if isinstance(node.op, ast.Div):
                return lambda coords, params: (
                    left_fn(coords, params) / right_fn(coords, params)
                )
            if isinstance(node.op, ast.Pow):
                return lambda coords, params: torch.pow(
                    left_fn(coords, params), right_fn(coords, params)
                )

        elif isinstance(node, ast.UnaryOp):
            operand_fn = _build_eval_fn(node.operand)
            if isinstance(node.op, ast.USub):
                return lambda coords, params: -operand_fn(coords, params)

        elif isinstance(node, ast.Call):
            func_name = node.func.id
            arg_fns = [_build_eval_fn(arg) for arg in node.args]

            # Single-argument functions
            if func_name == "sqrt":
                return lambda coords, params: torch.sqrt(arg_fns[0](coords, params))
            if func_name == "abs":
                return lambda coords, params: torch.abs(arg_fns[0](coords, params))
            if func_name == "sin":
                # Convert degrees to radians (deg * pi/180)
                return lambda coords, params: torch.sin(
                    _deg_to_rad(arg_fns[0](coords, params))
                )
            if func_name == "cos":
                # Convert degrees to radians (deg * pi/180)
                return lambda coords, params: torch.cos(
                    _deg_to_rad(arg_fns[0](coords, params))
                )
            if func_name == "tan":
                # Convert degrees to radians (deg * pi/180)
                return lambda coords, params: torch.tan(
                    _deg_to_rad(arg_fns[0](coords, params))
                )
            if func_name == "atan":
                # Convert radians to degrees (rad * 180/pi)
                return lambda coords, params: _rad_to_deg(
                    torch.atan(arg_fns[0](coords, params))
                )

            # Two-argument functions
            if func_name == "atan2":
                # Convert radians to degrees (rad * 180/pi)
                return lambda coords, params: _rad_to_deg(
                    torch.atan2(
                        arg_fns[0](coords, params),  # y
                        arg_fns[1](coords, params),  # x
                    )
                )

            # Variable-argument functions
            if func_name == "min":
                return lambda coords, params: (
                    torch.min(
                        torch.stack([fn(coords, params) for fn in arg_fns], dim=0),
                        dim=0,
                    ).values
                )
            if func_name == "max":
                return lambda coords, params: (
                    torch.max(
                        torch.stack([fn(coords, params) for fn in arg_fns], dim=0),
                        dim=0,
                    ).values
                )

            # Smooth operations
            if func_name == "softmin":
                return lambda coords, params: _softmin(
                    arg_fns[0](coords, params),  # a
                    arg_fns[1](coords, params),  # b
                    arg_fns[2](coords, params),  # k
                )
            if func_name == "softmax":
                return lambda coords, params: _softmax(
                    arg_fns[0](coords, params),  # a
                    arg_fns[1](coords, params),  # b
                    arg_fns[2](coords, params),  # k
                )
            if func_name == "softabs":
                return lambda coords, params: _softabs(
                    arg_fns[0](coords, params),  # x
                    arg_fns[1](coords, params),  # k
                )

            raise RuntimeError(f"Unexpected function during codegen: {func_name}")

        elif isinstance(node, ast.Name):
            if node.id in ["x", "y", "z"]:
                var_name = node.id
                return lambda coords, params: coords[var_name]
            if node.id in param_map:
                param_name = param_map[node.id]
                return lambda coords, params: params[param_name]

        elif isinstance(node, ast.Constant):
            value = float(node.value)

            # Return a function that broadcasts the constant to match coordinate shape
            # coords['x'] is [B, N], so we create a [B, 1] tensor that broadcasts
            def const_fn(coords, params):
                x_shape = coords["x"].shape  # [B, N]
                # Create constant with same batch dimension as coords
                return torch.full(
                    (x_shape[0], 1),
                    value,
                    dtype=coords["x"].dtype,
                    device=coords["x"].device,
                ).expand_as(coords["x"])

            return const_fn

        raise RuntimeError(
            f"Unexpected node type during codegen: {node.__class__.__name__}"
        )

    eval_fn = _build_eval_fn(tree.body)

    # Wrap in final SDF function
    def sdf_fn(p: Tensor, params: dict[str, Tensor]) -> Tensor:
        """
        Evaluate formula SDF.

        Args:
            p: [B, N, 3] query points
            params: dict of parameters {name: [B, K] tensor}

        Returns:
            [B, N] signed distances
        """
        # Extract coordinates [B, N]
        x = p[..., 0]
        y = p[..., 1]
        z = p[..., 2]

        coords = {"x": x, "y": y, "z": z}

        # Evaluate expression (broadcasting handles [1, N] op [B, 1] -> [B, N])
        result = eval_fn(coords, params)

        return result

    # Add division warning if needed
    if has_division:
        sdf_fn.__doc__ += (
            "\n\nWarning: Expression contains division which may "
            "cause gradient instability near zero."
        )

    return sdf_fn


# Main Parser Entry Point


def parse_expression(
    expression: str, params: dict[str, Tensor], allowed_vars: list[str] = None
) -> Callable[[Tensor, dict[str, Tensor]], Tensor]:
    """
    Parse mathematical expression into PyTorch SDF function.

    Four-phase pipeline:
    1. Preprocessing: Replace $param with __param, validate existence
    2. AST Validation: Parse and validate whitelist operations
    3. PyTorch Codegen: Convert to executable function
    4. Return callable

    Args:
        expression: String expression like "sqrt(x*x + y*y) - $radius"
        params: Dict of parameter tensors already in [B, K] format
                (primitives handle ensure_batched_tensor before calling)
        allowed_vars: Which variables are allowed (for 2D validation)
                     Default: ['x', 'y', 'z'] for 3D

    Returns:
        sdf_fn(points: [N, 3], params: dict) -> [B, N]

    Raises:
        ValueError: If expression invalid or uses disallowed operations
        SyntaxError: If expression has syntax errors

    Examples:
        >>> # 3D sphere formula
        >>> params = {"radius": torch.tensor([[1.0]])}
        >>> sdf_fn = parse_expression("sqrt(x*x + y*y + z*z) - $radius", params)
        >>> points = torch.rand(100, 3)
        >>> distances = sdf_fn(points, params)  # [1, 100]

        >>> # 2D circle formula (only x, y allowed)
        >>> params = {"radius": torch.tensor([[1.0]])}
        >>> sdf_fn = parse_expression(
        ...     "sqrt(x*x + y*y) - $radius",
        ...     params,
        ...     allowed_vars=['x', 'y']
        ... )
    """
    if allowed_vars is None:
        allowed_vars = ["x", "y", "z"]

    # Check for empty expression
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty")

    # Preprocessing
    expr_preprocessed, param_map = preprocess_params(expression, params)

    # AST Validation
    try:
        tree = ast.parse(expr_preprocessed, mode="eval")
    except SyntaxError as e:
        raise ValueError(
            f"Syntax error in expression: {e.msg}. "
            f"Check for balanced parentheses and valid operators."
        ) from e

    validate_ast(tree, param_map, allowed_vars)
    variables_used = detect_variables(tree)

    # Check 2D plane constraints (if allowed_vars is restricted)
    if len(allowed_vars) < 3:
        for var in variables_used:
            if var not in allowed_vars:
                raise ValueError(
                    f"Variable '{var}' not allowed in this plane. "
                    f"Allowed variables: {allowed_vars}"
                )

    # PyTorch Code Generation
    sdf_fn = ast_to_pytorch(tree, param_map)

    return sdf_fn
