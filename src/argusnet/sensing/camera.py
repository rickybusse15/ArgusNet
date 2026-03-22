"""Pinhole camera sensor model for ArgusNet."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from argusnet.core.types import Vector3, vec3
from argusnet.core.geometry import normalize

__all__ = [
    "CameraModel",
    "PinholeCamera",
]


@dataclass(frozen=True)
class CameraModel:
    """Intrinsic parameters for a pinhole camera.

    All angles in radians.  Pixel coordinates are not used internally;
    the model works in normalised image coordinates and bearing vectors.
    """

    h_fov_rad: float = 1.0472  # 60 degrees default
    """Horizontal field-of-view (radians)."""

    v_fov_rad: float = 0.7854  # 45 degrees default
    """Vertical field-of-view (radians)."""

    image_width_px: int = 1920
    image_height_px: int = 1080

    # Derived intrinsics (computed from FoV + image size)
    @property
    def fx(self) -> float:
        """Focal length in x (pixels)."""
        return self.image_width_px / (2.0 * np.tan(self.h_fov_rad / 2.0))

    @property
    def fy(self) -> float:
        """Focal length in y (pixels)."""
        return self.image_height_px / (2.0 * np.tan(self.v_fov_rad / 2.0))

    @property
    def cx(self) -> float:
        return self.image_width_px / 2.0

    @property
    def cy(self) -> float:
        return self.image_height_px / 2.0

    def project(self, point_cam: Vector3) -> Optional[Tuple[float, float]]:
        """Project a 3-D point in *camera* frame to pixel (u, v).

        Returns None if the point is behind the camera (z <= 0).
        """
        x, y, z = point_cam
        if z <= 0:
            return None
        u = self.fx * (x / z) + self.cx
        v = self.fy * (y / z) + self.cy
        return float(u), float(v)

    def backproject(self, u: float, v: float) -> Vector3:
        """Back-project pixel (u, v) to a unit bearing vector in camera frame."""
        x = (u - self.cx) / self.fx
        y = (v - self.cy) / self.fy
        d = vec3(x, y, 1.0)
        return normalize(d)

    def in_image(self, u: float, v: float) -> bool:
        return 0 <= u < self.image_width_px and 0 <= v < self.image_height_px


class PinholeCamera:
    """Camera sensor with a pose in world coordinates.

    Provides project/backproject in world frame via a rotation matrix
    and translation vector (both in metres / radians).
    """

    def __init__(
        self,
        model: CameraModel,
        position: Vector3,
        rotation: np.ndarray,
    ) -> None:
        """
        Args:
            model: Intrinsic camera parameters.
            position: Camera centre in world frame (metres).
            rotation: 3×3 rotation matrix, world-to-camera.
        """
        self.model = model
        self.position = np.asarray(position, dtype=float)
        self.rotation = np.asarray(rotation, dtype=float)

    def world_to_cam(self, world_point: Vector3) -> Vector3:
        return self.rotation @ (np.asarray(world_point, dtype=float) - self.position)

    def project_world(self, world_point: Vector3) -> Optional[Tuple[float, float]]:
        """Project a world-frame 3-D point to pixel coordinates."""
        return self.model.project(self.world_to_cam(world_point))

    def bearing_to_world(self, u: float, v: float) -> Vector3:
        """Return bearing unit vector in world frame for pixel (u, v)."""
        cam_bearing = self.model.backproject(u, v)
        return normalize(self.rotation.T @ cam_bearing)
