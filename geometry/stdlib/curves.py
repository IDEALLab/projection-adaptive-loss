"""
Curve Primitives - 2D Parametric Curves for Bezier Blend SDF

Provides abstract curve interface and cubic Bezier implementation.
Curves are 2D parametric objects that can be lifted to 3D via workplanes.

Curve Primitives:
- Curve2D: Abstract base class for 2D parametric curves
- CubicBezier2D: Single cubic Bezier segment with batched control points
"""

from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import torch
from torch import Tensor

from .primitives import PLANE_AXES

if TYPE_CHECKING:
    from .workplane import Workplane


# Validation Helpers


def _validate_control_points(
    cp, n_points: int, dim: int, name: str = "control_points"
) -> Tensor:
    """
    Validate that control points tensor has shape [B, n_points, dim].

    Args:
        cp: Input (must be Tensor)
        n_points: Expected number of control points
        dim: Expected coordinate dimension (2 for 2D curves)
        name: Parameter name for error messages

    Returns:
        The input tensor unchanged (if valid)

    Raises:
        TypeError: If cp is not a Tensor
        ValueError: If tensor shape is not [B, n_points, dim]
    """
    if not isinstance(cp, Tensor):
        raise TypeError(
            f"{name} must be a Tensor with shape [B, {n_points}, {dim}], "
            f"got {type(cp).__name__}.\n"
            f"For single curve, use: torch.tensor([[[x0,y0], [x1,y1], [x2,y2], [x3,y3]]])"
        )

    if cp.dim() != 3:
        raise ValueError(
            f"{name} must be [B, {n_points}, {dim}] format, got shape {list(cp.shape)}.\n"
            f"For single curve, use: tensor.unsqueeze(0) to add batch dimension"
        )

    if cp.shape[1] != n_points:
        raise ValueError(
            f"{name} must have {n_points} control points, got {cp.shape[1]}.\n"
            f"Expected shape [B, {n_points}, {dim}]."
        )

    if cp.shape[2] != dim:
        raise ValueError(
            f"{name} must have {dim}-dimensional coordinates, got {cp.shape[2]}.\n"
            f"Expected shape [B, {n_points}, {dim}]."
        )

    return cp


# Abstract Base Class


class Curve2D(ABC):
    """
    Abstract base class for 2D parametric curves.

    A Curve2D maps parameter t in [0, 1] to 2D points. The SDF layer
    only sees the .sample() and .sample_deriv() interface, never the
    parametric form.

    Subclasses must implement:
        - batch_size (property)
        - device (property)
        - sample(t) -> (B, S, 2)
        - sample_deriv(t) -> (B, S, 2)
    """

    @property
    def workplane(self) -> Workplane | None:
        """Workplane this curve is associated with. None if not set."""
        return getattr(self, "_workplane", None)

    @property
    @abstractmethod
    def batch_size(self) -> int:
        """Number of curves in the batch."""

    @property
    @abstractmethod
    def device(self) -> torch.device:
        """Device of the curve parameters."""

    @abstractmethod
    def sample(self, t: Tensor) -> Tensor:
        """
        Evaluate curve at parameter values t.

        Args:
            t: (S,) tensor of parameter values. Values outside [0, 1]
               are allowed (extrapolation).

        Returns:
            (B, S, 2) tensor of 2D points
        """

    @abstractmethod
    def sample_deriv(self, t: Tensor) -> Tensor:
        """
        Evaluate curve derivative (tangent) at parameter values t.

        Args:
            t: (S,) tensor of parameter values. Values outside [0, 1]
               are allowed (extrapolation).

        Returns:
            (B, S, 2) tensor of 2D tangent vectors
        """

    def arc_length(self, n_samples: int = 64) -> Tensor:
        """
        Approximate arc length via polyline summation.

        Args:
            n_samples: Number of sample points (default: 64)

        Returns:
            (B,) tensor of arc lengths
        """
        t = torch.linspace(0.0, 1.0, n_samples, device=self.device)
        pts = self.sample(t)  # (B, S, 2)
        segments = pts[:, 1:] - pts[:, :-1]  # (B, S-1, 2)
        return segments.norm(dim=-1).sum(dim=-1)  # (B,)

    def control_points_3d(self, plane: str = None) -> Tensor:
        """
        Lift 2D control points to 3D.

        If plane is None, uses the stored workplane (with offset/rotation).
        If plane is a string ('xy', 'xz', 'yz'), uses axis-aligned mapping.

        Args:
            plane: Optional plane string. If None, uses self.workplane.

        Returns:
            (B, N_cp, 3) tensor of 3D control points

        Raises:
            ValueError: If plane is invalid or no plane/workplane available
        """
        if plane is None:
            if self.workplane is None:
                raise ValueError(
                    "No plane specified and curve has no workplane. "
                    "Pass plane='xy'/'xz'/'yz' or set a workplane."
                )
            return self.workplane.lift_2d(self.cp)

        if plane not in PLANE_AXES:
            raise ValueError(
                f"Invalid plane '{plane}'. Must be one of: {tuple(PLANE_AXES.keys())}"
            )

        cp_2d = self.cp  # (B, N_cp, 2)
        B, N_cp, _ = cp_2d.shape

        ax1, ax2, _ = PLANE_AXES[plane]
        # Out-of-place construction for torch.compile compatibility
        channels = [
            torch.zeros(B, N_cp, 1, device=self.device, dtype=cp_2d.dtype)
            for _ in range(3)
        ]
        channels[ax1] = cp_2d[:, :, 0:1]
        channels[ax2] = cp_2d[:, :, 1:2]
        cp_3d = torch.cat(channels, dim=-1)

        return cp_3d

    def to_polyline_3d(self, plane: str = None, n_samples: int = 64) -> Tensor:
        """
        Lift 2D curve samples to 3D.

        If plane is None, uses the stored workplane (with offset/rotation).
        If plane is a string ('xy', 'xz', 'yz'), uses axis-aligned mapping.

        Args:
            plane: Optional plane string. If None, uses self.workplane.
            n_samples: Number of sample points (default: 64)

        Returns:
            (B, S, 3) tensor of 3D points
        """
        t = torch.linspace(0.0, 1.0, n_samples, device=self.device)
        pts_2d = self.sample(t)  # (B, S, 2)

        if plane is None:
            if self.workplane is None:
                raise ValueError(
                    "No plane specified and curve has no workplane. "
                    "Pass plane='xy'/'xz'/'yz' or set a workplane."
                )
            return self.workplane.lift_2d(pts_2d)

        if plane not in PLANE_AXES:
            raise ValueError(
                f"Invalid plane '{plane}'. Must be one of: {tuple(PLANE_AXES.keys())}"
            )

        B, S, _ = pts_2d.shape
        pts_3d = torch.zeros(B, S, 3, device=self.device, dtype=pts_2d.dtype)

        ax1, ax2, _ = PLANE_AXES[plane]
        pts_3d[:, :, ax1] = pts_2d[:, :, 0]
        pts_3d[:, :, ax2] = pts_2d[:, :, 1]

        return pts_3d


# Cubic Bezier Implementation


class CubicBezier2D(Curve2D):
    """
    Single cubic Bezier segment with batched [B, 4, 2] control points.

    The cubic Bezier curve is defined by four control points P0, P1, P2, P3:
        C(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Values of t outside [0, 1] produce extrapolation.

    Args:
        control_points: [B, 4, 2] tensor of control points
        workplane: Optional Workplane object or plane string ('xy', 'xz', 'yz').
                   If a string, a Workplane is auto-created via Workplane.from_base().
    """

    def __init__(self, control_points: Tensor, workplane=None):
        _validate_control_points(control_points, 4, 2, "control_points")
        self.cp = control_points
        # Accept Workplane object OR plane string ('xy', 'xz', 'yz')
        if isinstance(workplane, str):
            from .workplane import Workplane as WP

            B = control_points.shape[0]
            workplane = WP.from_base(
                workplane, offset=torch.zeros(B, 1, device=control_points.device)
            )
        self._workplane = workplane  # Optional[Workplane]
        self._validate_non_degenerate()

    def _validate_non_degenerate(self) -> None:
        """Warn if control polygon has near-zero length."""
        diffs = self.cp[:, 1:] - self.cp[:, :-1]  # (B, 3, 2)
        polygon_length = diffs.norm(dim=-1).sum(dim=-1)  # (B,)
        if (polygon_length < 1e-8).any():
            warnings.warn(
                "near-zero control polygon length detected",
                UserWarning,
                stacklevel=3,
            )

    @property
    def batch_size(self) -> int:
        return self.cp.shape[0]

    @property
    def device(self) -> torch.device:
        return self.cp.device

    def sample(self, t: Tensor) -> Tensor:
        """
        Evaluate cubic Bezier at parameter values t.

        Uses Bernstein polynomial basis:
            C(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

        Args:
            t: (S,) tensor of parameter values

        Returns:
            (B, S, 2) tensor of 2D points

        Raises:
            ValueError: If t is not 1D
        """
        if t.dim() != 1:
            raise ValueError(
                f"t must be 1D tensor with shape (S,), got shape {list(t.shape)}."
            )

        t = t.to(self.cp.device)
        s = 1.0 - t  # (S,)

        # Reshape for broadcasting: t -> (1, S, 1), cp -> (B, 1, 2) per point
        t_ = t.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
        s_ = s.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)

        P0 = self.cp[:, 0:1, :]  # (B, 1, 2)
        P1 = self.cp[:, 1:2, :]  # (B, 1, 2)
        P2 = self.cp[:, 2:3, :]  # (B, 1, 2)
        P3 = self.cp[:, 3:4, :]  # (B, 1, 2)

        # Bernstein basis
        result = (
            (s_**3) * P0 + 3 * (s_**2) * t_ * P1 + 3 * s_ * (t_**2) * P2 + (t_**3) * P3
        )

        return result  # (B, S, 2)

    def sample_deriv(self, t: Tensor) -> Tensor:
        """
        Evaluate cubic Bezier derivative at parameter values t.

        Derivative:
            C'(t) = 3[(1-t)^2 (P1-P0) + 2(1-t)t (P2-P1) + t^2 (P3-P2)]

        Args:
            t: (S,) tensor of parameter values

        Returns:
            (B, S, 2) tensor of 2D tangent vectors

        Raises:
            ValueError: If t is not 1D
        """
        if t.dim() != 1:
            raise ValueError(
                f"t must be 1D tensor with shape (S,), got shape {list(t.shape)}."
            )

        t = t.to(self.cp.device)
        s = 1.0 - t  # (S,)

        # Reshape for broadcasting
        t_ = t.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)
        s_ = s.unsqueeze(0).unsqueeze(-1)  # (1, S, 1)

        # Differences between consecutive control points
        D01 = self.cp[:, 1:2, :] - self.cp[:, 0:1, :]  # (B, 1, 2)
        D12 = self.cp[:, 2:3, :] - self.cp[:, 1:2, :]  # (B, 1, 2)
        D23 = self.cp[:, 3:4, :] - self.cp[:, 2:3, :]  # (B, 1, 2)

        result = 3 * ((s_**2) * D01 + 2 * s_ * t_ * D12 + (t_**2) * D23)

        return result  # (B, S, 2)
