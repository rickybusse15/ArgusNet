"""Digital Surface Model (DSM/DEM) elevation query API for ArgusNet.

Wraps the GeoTIFF ingest logic from ``argusnet.world._scene_gis`` and
the procedural terrain models from ``argusnet.world.terrain`` behind a
unified elevation query interface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import Vector3

__all__ = [
    "ElevationMap",
    "ElevationPatch",
    "flat_elevation_map",
]


@dataclass
class ElevationPatch:
    """A rectangular region of elevation data."""

    x_m: np.ndarray  # (N,) ENU X grid values
    y_m: np.ndarray  # (M,) ENU Y grid values
    z_m: np.ndarray  # (M, N) elevation in metres

    @property
    def shape(self) -> tuple[int, int]:
        return self.z_m.shape

    def at(self, x: float, y: float) -> float:
        """Bilinear interpolation of elevation at (x, y)."""
        xi = np.searchsorted(self.x_m, x, side="right") - 1
        yi = np.searchsorted(self.y_m, y, side="right") - 1
        xi = int(np.clip(xi, 0, len(self.x_m) - 2))
        yi = int(np.clip(yi, 0, len(self.y_m) - 2))

        x0, x1 = self.x_m[xi], self.x_m[xi + 1]
        y0, y1 = self.y_m[yi], self.y_m[yi + 1]
        tx = (x - x0) / (x1 - x0 + 1e-12)
        ty = (y - y0) / (y1 - y0 + 1e-12)

        z00 = self.z_m[yi, xi]
        z10 = self.z_m[yi + 1, xi]
        z01 = self.z_m[yi, xi + 1]
        z11 = self.z_m[yi + 1, xi + 1]

        return float((1 - ty) * ((1 - tx) * z00 + tx * z01) + ty * ((1 - tx) * z10 + tx * z11))


class ElevationMap:
    """Query interface for terrain elevation.

    Supports three backends:
    1. A flat elevation (constant height) — zero-dependency fallback.
    2. A numpy height array with grid spacing — in-memory grid.
    3. A GeoTIFF file (requires ``tifffile`` + ``pyproj``).

    All queries return elevation in ENU metres above the reference datum.
    """

    def __init__(
        self,
        heights_m: np.ndarray | None = None,
        x_values_m: np.ndarray | None = None,
        y_values_m: np.ndarray | None = None,
        default_elevation_m: float = 0.0,
    ) -> None:
        self._default = default_elevation_m
        if heights_m is not None and x_values_m is not None and y_values_m is not None:
            self._patch: ElevationPatch | None = ElevationPatch(
                x_m=x_values_m, y_m=y_values_m, z_m=heights_m
            )
        else:
            self._patch = None

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def flat(cls, elevation_m: float = 0.0) -> ElevationMap:
        return cls(default_elevation_m=elevation_m)

    @classmethod
    def from_array(
        cls,
        heights_m: np.ndarray,
        x_min_m: float,
        x_max_m: float,
        y_min_m: float,
        y_max_m: float,
    ) -> ElevationMap:
        """Build from a 2-D height array with explicit bounds."""
        rows, cols = heights_m.shape
        x_vals = np.linspace(x_min_m, x_max_m, cols)
        y_vals = np.linspace(y_min_m, y_max_m, rows)
        return cls(heights_m=heights_m, x_values_m=x_vals, y_values_m=y_vals)

    @classmethod
    def from_terrain_model(
        cls, terrain_model: object, bounds_m: float = 2000.0, resolution_m: float = 10.0
    ) -> ElevationMap:
        """Sample a terrain object with ``height_at`` into a grid."""
        coords = np.arange(-bounds_m / 2, bounds_m / 2, resolution_m)
        XX, YY = np.meshgrid(coords, coords)
        if hasattr(terrain_model, "height_at_many"):
            points = np.column_stack([XX.reshape(-1), YY.reshape(-1)])
            heights = np.asarray(terrain_model.height_at_many(points), dtype=float).reshape(
                XX.shape
            )
        else:
            heights = np.vectorize(lambda x, y: terrain_model.height_at(float(x), float(y)))(
                XX, YY
            )
        return cls.from_array(heights, coords[0], coords[-1], coords[0], coords[-1])

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def at_point(self, x_m: float, y_m: float) -> float:
        """Return terrain elevation at (x_m, y_m) in ENU metres."""
        if self._patch is None:
            return self._default
        return self._patch.at(x_m, y_m)

    def at_point_with_uncertainty(self, x_m: float, y_m: float) -> tuple[float, float]:
        """Return (elevation_m, std_m) with local terrain roughness as uncertainty.

        The std_m is derived from the variance of the four surrounding grid cells.
        For flat terrain or when no grid is available, std_m = 0.1 m.
        """
        if self._patch is None:
            return self._default, 0.1

        xi = np.searchsorted(self._patch.x_m, x_m, side="right") - 1
        yi = np.searchsorted(self._patch.y_m, y_m, side="right") - 1
        xi = int(np.clip(xi, 0, len(self._patch.x_m) - 2))
        yi = int(np.clip(yi, 0, len(self._patch.y_m) - 2))

        z00 = self._patch.z_m[yi, xi]
        z10 = self._patch.z_m[min(yi + 1, self._patch.z_m.shape[0] - 1), xi]
        z01 = self._patch.z_m[yi, min(xi + 1, self._patch.z_m.shape[1] - 1)]
        z11 = self._patch.z_m[
            min(yi + 1, self._patch.z_m.shape[0] - 1), min(xi + 1, self._patch.z_m.shape[1] - 1)
        ]

        surrounding = [z00, z10, z01, z11]
        std_m = float(np.std(surrounding))
        std_m = max(std_m, 0.05)  # minimum 5cm uncertainty

        return self.at_point(x_m, y_m), std_m

    def at_xy(self, xy: np.ndarray) -> float:
        """Accept a (2,) or (3,) ENU array."""
        return self.at_point(float(xy[0]), float(xy[1]))

    def patch(
        self,
        x_min_m: float,
        x_max_m: float,
        y_min_m: float,
        y_max_m: float,
    ) -> ElevationPatch:
        """Return an ElevationPatch covering the requested region."""
        if self._patch is None:
            x_vals = np.linspace(x_min_m, x_max_m, 2)
            y_vals = np.linspace(y_min_m, y_max_m, 2)
            heights = np.full((2, 2), self._default)
            return ElevationPatch(x_m=x_vals, y_m=y_vals, z_m=heights)

        xi = (self._patch.x_m >= x_min_m) & (self._patch.x_m <= x_max_m)
        yi = (self._patch.y_m >= y_min_m) & (self._patch.y_m <= y_max_m)
        return ElevationPatch(
            x_m=self._patch.x_m[xi],
            y_m=self._patch.y_m[yi],
            z_m=self._patch.z_m[np.ix_(yi, xi)],
        )

    def agl(self, position: Vector3) -> float:
        """Return height above ground level for a 3-D ENU position."""
        ground_z = self.at_point(float(position[0]), float(position[1]))
        return float(position[2]) - ground_z


def flat_elevation_map(elevation_m: float = 0.0) -> ElevationMap:
    """Convenience factory: flat terrain at constant elevation."""
    return ElevationMap.flat(elevation_m)
