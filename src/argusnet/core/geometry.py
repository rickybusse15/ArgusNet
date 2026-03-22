"""Core geometry utilities for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from argusnet.core.types import Vector3, vec3

__all__ = [
    "Vector3",
    "vec3",
    "BoundingBox3D",
    "normalize",
    "rotate_z",
    "rotation_matrix_z",
    "bearing_to_unit_vector",
    "angle_between",
    "clamp",
]


@dataclass
class BoundingBox3D:
    """Axis-aligned bounding box in 3D meter-based ENU coordinates."""

    min: Vector3
    max: Vector3

    def contains(self, point: Vector3) -> bool:
        return bool(np.all(point >= self.min) and np.all(point <= self.max))

    def center(self) -> Vector3:
        return (self.min + self.max) * 0.5

    def size(self) -> Vector3:
        return self.max - self.min

    def expand(self, margin: float) -> "BoundingBox3D":
        m = np.array([margin, margin, margin])
        return BoundingBox3D(self.min - m, self.max + m)

    @classmethod
    def from_points(cls, points: np.ndarray) -> "BoundingBox3D":
        """Build from an (N, 3) array of points."""
        return cls(points.min(axis=0), points.max(axis=0))


def normalize(v: Vector3) -> Vector3:
    """Return unit vector; returns v unchanged if near-zero."""
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def rotation_matrix_z(angle_rad: float) -> np.ndarray:
    """3×3 rotation matrix about the Z axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotate_z(v: Vector3, angle_rad: float) -> Vector3:
    """Rotate a 3-D vector about the Z axis by *angle_rad* radians."""
    return rotation_matrix_z(angle_rad) @ v


def bearing_to_unit_vector(az_rad: float, el_rad: float = 0.0) -> Vector3:
    """Convert azimuth (from +X toward +Y) and elevation to a unit vector.

    az_rad: 0 = +X axis, pi/2 = +Y axis (counterclockwise in XY plane)
    el_rad: 0 = horizontal, pi/2 = straight up
    """
    cos_el = np.cos(el_rad)
    return vec3(
        cos_el * np.cos(az_rad),
        cos_el * np.sin(az_rad),
        np.sin(el_rad),
    )


def angle_between(a: Vector3, b: Vector3) -> float:
    """Angle in radians between two vectors (0 to pi)."""
    dot = np.dot(normalize(a), normalize(b))
    return float(np.arccos(np.clip(dot, -1.0, 1.0)))


def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))
