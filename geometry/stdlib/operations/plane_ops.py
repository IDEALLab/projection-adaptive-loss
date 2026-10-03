def _get_common_plane(shapes: list) -> str | None:
    """
    Get the common plane from a list of shapes.

    Returns the plane if ALL shapes have the same non-None plane,
    otherwise returns None.
    """
    planes = [s.plane for s in shapes]
    if not planes:
        return None
    # If all planes are the same (and not None)
    first_plane = planes[0]
    if first_plane is not None and all(p == first_plane for p in planes):
        return first_plane
    return None


def _validate_same_plane(shapes: list, operation_name: str) -> None:
    """
    Validate that all 2D shapes have the same plane.

    Raises:
        ValueError: If shapes have different planes.
    """
    planes = [s.plane for s in shapes]

    # Filter to only 2D shapes (those with a plane)
    planes_2d = [p for p in planes if p is not None]

    if len(planes_2d) < 2:
        return  # Not enough 2D shapes to compare

    first_plane = planes_2d[0]
    for i, p in enumerate(planes_2d[1:], start=1):
        if p != first_plane:
            raise ValueError(
                f"{operation_name}() cannot combine 2D shapes in different planes: "
                f"'{first_plane}' vs '{p}'. All 2D shapes must be in the same plane."
            )
