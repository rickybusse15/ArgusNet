"""Survey coverage tracker for ArgusNet.

Tracks which cells of the area of interest have been observed by one or
more sensor footprints, and computes coverage completeness metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.mapping.occupancy import GridBounds

__all__ = [
    "CoverageMap",
    "CoverageStats",
    "circular_footprint",
    "rectangular_footprint",
]


@dataclass(frozen=True)
class CoverageStats:
    total_cells: int
    covered_cells: int
    coverage_fraction: float
    """Fraction of cells with at least one observation [0, 1]."""
    mean_revisits: float
    """Mean number of observations per covered cell."""


def circular_footprint(
    center_xy: tuple[float, float],
    radius_m: float,
    resolution_m: float,
) -> list[tuple[float, float]]:
    """Return a list of (x, y) cell centres inside a circular footprint."""
    cx, cy = center_xy
    r = radius_m
    xs = np.arange(cx - r, cx + r + resolution_m, resolution_m)
    ys = np.arange(cy - r, cy + r + resolution_m, resolution_m)
    pts = []
    for x in xs:
        for y in ys:
            if (x - cx) ** 2 + (y - cy) ** 2 <= r**2:
                pts.append((float(x), float(y)))
    return pts


def rectangular_footprint(
    center_xy: tuple[float, float],
    width_m: float,
    height_m: float,
    resolution_m: float,
    yaw_rad: float = 0.0,
) -> list[tuple[float, float]]:
    """Return cell centres inside a (possibly rotated) rectangular footprint."""
    cx, cy = center_xy
    hw, hh = width_m / 2, height_m / 2
    cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)

    us = np.arange(-hw, hw + resolution_m, resolution_m)
    vs = np.arange(-hh, hh + resolution_m, resolution_m)
    pts = []
    for u in us:
        for v in vs:
            x = cx + cos_y * u - sin_y * v
            y = cy + sin_y * u + cos_y * v
            pts.append((float(x), float(y)))
    return pts


class CoverageMap:
    """Tracks how many times each cell in a grid has been observed."""

    def __init__(self, bounds: GridBounds) -> None:
        self.bounds = bounds
        self._count = np.zeros((bounds.nx, bounds.ny), dtype=np.int32)

    def mark(self, footprint_cells: list[tuple[float, float]]) -> None:
        """Increment the observation count for each cell in *footprint_cells*."""
        for x, y in footprint_cells:
            if (
                self.bounds.x_min_m <= x <= self.bounds.x_max_m
                and self.bounds.y_min_m <= y <= self.bounds.y_max_m
            ):
                i, j = self.bounds.xy_to_ij(x, y)
                self._count[i, j] += 1

    def mark_circular(
        self,
        center_xy: tuple[float, float],
        radius_m: float,
    ) -> None:
        fp = circular_footprint(center_xy, radius_m, self.bounds.resolution_m)
        self.mark(fp)

    def mark_rectangular(
        self,
        center_xy: tuple[float, float],
        width_m: float,
        height_m: float,
        yaw_rad: float = 0.0,
    ) -> None:
        fp = rectangular_footprint(center_xy, width_m, height_m, self.bounds.resolution_m, yaw_rad)
        self.mark(fp)

    def count_at(self, x: float, y: float) -> int:
        i, j = self.bounds.xy_to_ij(x, y)
        return int(self._count[i, j])

    def is_covered(self, x: float, y: float) -> bool:
        return self.count_at(x, y) > 0

    @property
    def stats(self) -> CoverageStats:
        total = self._count.size
        covered = int(np.sum(self._count > 0))
        total_obs = int(np.sum(self._count))
        mean_rev = total_obs / covered if covered > 0 else 0.0
        return CoverageStats(
            total_cells=total,
            covered_cells=covered,
            coverage_fraction=covered / total if total > 0 else 0.0,
            mean_revisits=mean_rev,
        )

    @property
    def count_grid(self) -> np.ndarray:
        """Return (nx, ny) integer observation count array."""
        return self._count.copy()
