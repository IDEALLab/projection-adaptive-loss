"""Neural Shape CSG Warning."""

import warnings

_neural_csg_warned = False


def _warn_neural_csg(operation_name: str, shapes: list) -> None:
    """
    Emit a one-time warning if any shape in a CSG operation is neural.

    Neural SDFs may not satisfy exact distance field properties (|grad(f)|=1),
    which can cause artifacts in boolean operations.
    """
    global _neural_csg_warned

    if _neural_csg_warned:
        return

    has_neural = any(getattr(s, "is_neural", False) for s in shapes)

    if has_neural:
        _neural_csg_warned = True
        warnings.warn(
            f"{operation_name}() includes neural SDF shape(s). "
            f"Neural SDFs may not satisfy exact distance field properties, "
            f"which can cause artifacts in boolean operations. "
            f"For best results, ensure the model was trained with Eikonal loss (|grad(f)|=1).",
            UserWarning,
            stacklevel=3,
        )
