"""Field-of-view cone geometry for ArgusNet sensors."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import numpy as np

from argusnet.core.types import Vector3
from argusnet.core.geometry import normalize, angle_between

__all__ = [
    "FoVModel",
    "point_in_cone",
    "bearing_angle_to_point",
]


@dataclass(frozen=True)
class FoVModel:
    """Symmetric or asymmetric sensor field-of-view cone.

    Angles are in radians.  The cone is defined by a boresight direction and
    a half-angle.  For cameras with different horizontal/vertical FoVs use
    *h_half_angle_rad* and *v_half_angle_rad*; for omnidirectional sensors
    use *h_half_angle_rad = pi* (default).
    """

    h_half_angle_rad: float = math.pi
    """Horizontal (azimuth) half-angle. Default: omnidirectional."""

    v_half_angle_rad: float = math.pi / 2
    """Vertical (elevation) half-angle. Default: hemisphere."""

    min_range_m: float = 0.0
    """Minimum detection range (metres)."""

    max_range_m: float = 0.0
    """Maximum detection range (metres).  0 = unlimited."""

    def in_range(self, range_m: float) -> bool:
        """True if *range_m* is within [min_range_m, max_range_m]."""
        if range_m < self.min_range_m:
            return False
        if self.max_range_m > 0 and range_m > self.max_range_m:
            return False
        return True

    def point_visible(
        self,
        origin: Vector3,
        boresight: Vector3,
        target: Vector3,
    ) -> bool:
        """True if *target* falls within this FoV cone from *origin*.

        *boresight* is the sensor pointing direction (need not be normalised).
        Uses separate H/V half-angle checks in the boresight frame.
        """
        diff = target - origin
        range_m = float(np.linalg.norm(diff))
        if not self.in_range(range_m):
            return False
        return point_in_cone(
            origin=origin,
            boresight=boresight,
            point=target,
            h_half_angle_rad=self.h_half_angle_rad,
            v_half_angle_rad=self.v_half_angle_rad,
        )


def point_in_cone(
    origin: Vector3,
    boresight: Vector3,
    point: Vector3,
    h_half_angle_rad: float,
    v_half_angle_rad: float,
) -> bool:
    """Return True if *point* lies inside a boresight cone.

    The cone is described by separate horizontal and vertical half-angles,
    evaluated in a local frame aligned with the boresight direction.
    For a symmetric cone set both angles equal.
    """
    diff = point - origin
    dist = float(np.linalg.norm(diff))
    if dist < 1e-9:
        return True  # Point is at origin — trivially inside

    bore_n = normalize(np.asarray(boresight, dtype=float))
    diff_n = diff / dist

    # Full 3-D angle check (sufficient for symmetric or near-symmetric cones)
    full_angle = angle_between(bore_n, diff_n)

    # For symmetric cones this is exact; for asymmetric use H/V decomposition
    if abs(h_half_angle_rad - v_half_angle_rad) < 1e-6:
        return full_angle <= h_half_angle_rad

    # Decompose into horizontal and vertical components relative to boresight
    # Build local frame: forward = bore_n, up = global Z projected
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(bore_n, world_up)
    right_n = normalize(right)
    if np.linalg.norm(right_n) < 1e-9:
        # Boresight is vertical — degenerate; fall back to full angle
        return full_angle <= max(h_half_angle_rad, v_half_angle_rad)
    up_local = np.cross(right_n, bore_n)

    h_angle = abs(math.atan2(float(np.dot(diff_n, right_n)), float(np.dot(diff_n, bore_n))))
    v_angle = abs(math.atan2(float(np.dot(diff_n, up_local)), float(np.dot(diff_n, bore_n))))

    return h_angle <= h_half_angle_rad and v_angle <= v_half_angle_rad


def bearing_angle_to_point(
    origin: Vector3,
    boresight: Vector3,
    target: Vector3,
) -> float:
    """Return the angle in radians between *boresight* and the direction to *target*."""
    diff = target - origin
    return angle_between(boresight, diff)
