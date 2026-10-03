class Axis:
    def __init__(self, index: int, name: str) -> None:
        self.index = index
        self.name = name


X = Axis(0, "x")
Y = Axis(1, "y")
Z = Axis(2, "z")
AXES = [X, Y, Z]

PLANE_PERPENDICULAR_AXIS = {
    "xy": Z,
    "xz": Y,
    "yz": X,
}
"""Perpendicular axis for each plane"""
