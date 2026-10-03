"""
Workplane - Coordinate Frames for Projecting 3D Points to 2D

Workplanes define a local 2D coordinate frame in 3D space, specified by
an origin point and two orthonormal basis vectors (u, v). They project
3D query points into 2D coordinates for use by bezier_halfplane (Chunk 3).

Usage:
    wp = Workplane.from_base("xz", offset=0.5, rotate=15, rotate_axis="x")
    uv_coords = wp.project(points_3d)  # (B, N, 2)
"""

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .primitives import PLANE_AXES

# Axis name -> index mapping
_AXIS_INDEX = {"x": 0, "y": 1, "z": 2}

# Axis index -> name mapping
_INDEX_AXIS = {0: "x", 1: "y", 2: "z"}


@dataclass
class Workplane:
    """
    A batched workplane defined by origin + orthonormal basis (u, v).

    Attributes:
        origin: (B, 3), point on the plane
        u: (B, 3), first basis vector (unit)
        v: (B, 3), second basis vector (unit, orthogonal to u)
    """

    origin: Tensor  # (B, 3)
    u: Tensor  # (B, 3)
    v: Tensor  # (B, 3)

    @property
    def normal(self) -> Tensor:
        """Cross product u x v. Returns (B, 3)."""
        return torch.linalg.cross(self.u, self.v)

    @property
    def batch_size(self) -> int:
        return self.origin.shape[0]

    @property
    def device(self) -> torch.device:
        return self.origin.device

    @staticmethod
    def from_base(
        base: str,
        offset: Tensor = None,
        rotate: Tensor = None,
        rotate_axis: str = None,
        device=None,
    ) -> "Workplane":
        """
        Construct a workplane from an axis-aligned base plane.

        Args:
            base: "xy", "xz", or "yz"
            offset: [B, 1] tensor, shift along the normal direction
            rotate: [B, 1] tensor, rotation angle in degrees
            rotate_axis: Axis to rotate around (must be in-plane, not the normal).
                         Required if any rotate value != 0.
            device: PyTorch device for tensor creation. If None, inferred from
                    offset tensor device (falls back to CPU).

        Returns:
            Workplane with batched origin/u/v tensors of shape (B, 3)

        Raises:
            ValueError: If base is invalid, rotate_axis is the normal axis,
                        or rotate != 0 without rotate_axis specified.
        """
        if offset is None:
            offset = torch.zeros(1, 1, device=device)
        elif not isinstance(offset, Tensor):
            offset = torch.tensor([[float(offset)]], device=device)
        elif offset.dim() == 0:
            offset = offset.reshape(1, 1)

        # Infer device from offset if not explicitly given
        if device is None:
            device = offset.device

        if rotate is None:
            rotate = torch.zeros(1, 1, device=device)
        elif not isinstance(rotate, Tensor):
            rotate = torch.tensor([[float(rotate)]], device=device)
        elif rotate.dim() == 0:
            rotate = rotate.reshape(1, 1)

        B = offset.shape[0]

        if base not in PLANE_AXES:
            raise ValueError(
                f"Invalid base plane '{base}'. Must be one of: 'xy', 'xz', 'yz'."
            )

        ax1_idx, ax2_idx, normal_idx = PLANE_AXES[base]

        # Build initial basis vectors
        u = torch.zeros(3, device=device)
        u[ax1_idx] = 1.0

        v = torch.zeros(3, device=device)
        v[ax2_idx] = 1.0

        normal_vec = torch.zeros(3, device=device)
        normal_vec[normal_idx] = 1.0

        # Apply rotation if requested
        has_rotation = rotate.abs().max().item() > 0.0
        if has_rotation:
            if rotate_axis is None:
                raise ValueError(
                    f"rotate has nonzero values but 'rotate_axis' not specified. "
                    f"For base='{base}', valid rotate_axis values are: "
                    f"'{_INDEX_AXIS[ax1_idx]}' or '{_INDEX_AXIS[ax2_idx]}'."
                )

            if rotate_axis not in _AXIS_INDEX:
                raise ValueError(
                    f"Invalid rotate_axis '{rotate_axis}'. Must be 'x', 'y', or 'z'."
                )

            rot_idx = _AXIS_INDEX[rotate_axis]

            if rot_idx == normal_idx:
                raise ValueError(
                    f"rotate_axis='{rotate_axis}' is the normal to the '{base}' plane. "
                    f"Rotating around the normal doesn't tilt the plane. "
                    f"Use an in-plane axis: '{_INDEX_AXIS[ax1_idx]}' or '{_INDEX_AXIS[ax2_idx]}'."
                )

            # Batched rotation: rotate is [B, 1]
            angle_rad = rotate * (math.pi / 180.0)  # [B, 1]
            c = torch.cos(angle_rad)  # [B, 1]
            s = torch.sin(angle_rad)  # [B, 1]

            # Build [B, 3, 3] rotation matrices
            R = (
                torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone()
            )  # [B, 3, 3]
            if rot_idx == 0:  # rotate around x
                R[:, 1, 1] = c.squeeze(-1)
                R[:, 1, 2] = -s.squeeze(-1)
                R[:, 2, 1] = s.squeeze(-1)
                R[:, 2, 2] = c.squeeze(-1)
            elif rot_idx == 1:  # rotate around y
                R[:, 0, 0] = c.squeeze(-1)
                R[:, 0, 2] = s.squeeze(-1)
                R[:, 2, 0] = -s.squeeze(-1)
                R[:, 2, 2] = c.squeeze(-1)
            elif rot_idx == 2:  # rotate around z
                R[:, 0, 0] = c.squeeze(-1)
                R[:, 0, 1] = -s.squeeze(-1)
                R[:, 1, 0] = s.squeeze(-1)
                R[:, 1, 1] = c.squeeze(-1)

            # [B, 3, 3] @ [3] -> [B, 3]
            u = (R @ u).contiguous()  # [B, 3]
            v = (R @ v).contiguous()  # [B, 3]
        else:
            # No rotation, expand to (B, 3)
            u = u.unsqueeze(0).expand(B, -1).contiguous()
            v = v.unsqueeze(0).expand(B, -1).contiguous()

        # Origin = offset along the normal direction: [B, 1] * [3] -> [B, 3]
        origin = offset * normal_vec

        return Workplane(origin=origin, u=u, v=v)

    def project(self, points_3d: Tensor) -> Tensor:
        """
        Project 3D points onto this workplane.

        Args:
            points_3d: (N, 3) or (B, N, 3) tensor of 3D points

        Returns:
            (B, N, 2) tensor of u/v coordinates on the plane
        """
        if points_3d.dim() == 2:
            # (N, 3) -> (1, N, 3) then broadcast with (B, 1, 3) origin
            points_3d = points_3d.unsqueeze(0)

        # d = points - origin: (B, N, 3)
        d = points_3d - self.origin.unsqueeze(1)  # (B, 1, 3) broadcast

        # Project onto u and v: dot product along last dim
        u_coord = (d * self.u.unsqueeze(1)).sum(dim=-1)  # (B, N)
        v_coord = (d * self.v.unsqueeze(1)).sum(dim=-1)  # (B, N)

        return torch.stack([u_coord, v_coord], dim=-1)  # (B, N, 2)

    def lift_2d(self, points_2d: Tensor) -> Tensor:
        """
        Lift 2D workplane coordinates to 3D world coordinates.

        Inverse of project(): point_3d = origin + u * u_coord + v * v_coord

        Args:
            points_2d: (B, N, 2) tensor of u/v coordinates on the plane

        Returns:
            (B, N, 3) tensor of 3D points
        """
        u_coord = points_2d[..., 0:1]  # (B, N, 1)
        v_coord = points_2d[..., 1:2]  # (B, N, 1)

        # origin (B, 1, 3) + u (B, 1, 3) * u_coord (B, N, 1) + ...
        return (
            self.origin.unsqueeze(1)
            + self.u.unsqueeze(1) * u_coord
            + self.v.unsqueeze(1) * v_coord
        )
