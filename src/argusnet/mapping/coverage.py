"""Survey coverage tracker for ArgusNet.

Tracks which cells of the area of interest have been observed by one or
more sensor footprints, and computes coverage completeness metrics.
"""

from __future__ import annotations

from collections.abc import Callable
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
        visibility_predicate: Callable[[float, float], bool] | None = None,
        los_max_samples: int | None = None,
    ) -> None:
        # Vectorized equivalent of mark(circular_footprint(...)): same arange
        # cell centres, same inclusive bounds test, same index truncation.
        cx, cy = center_xy
        res = self.bounds.resolution_m
        xs = np.arange(cx - radius_m, cx + radius_m + res, res)
        ys = np.arange(cy - radius_m, cy + radius_m + res, res)
        grid_x, grid_y = np.meshgrid(xs, ys, indexing="ij")
        inside = (grid_x - cx) ** 2 + (grid_y - cy) ** 2 <= radius_m**2
        mark_x = grid_x[inside]
        mark_y = grid_y[inside]
        if visibility_predicate is not None and mark_x.size:
            # Cap the number of (expensive) LOS tests per footprint by
            # deterministically striding the candidate cells. Only cells that
            # actually pass the predicate are marked, so the "marked ⟹ visible"
            # invariant holds; the cap only bites at very fine resolutions where
            # a footprint spans many cells. `None` / large cap ⇒ exact behavior.
            if los_max_samples is not None and mark_x.size > los_max_samples:
                stride = int(np.ceil(mark_x.size / los_max_samples))
                mark_x = mark_x[::stride]
                mark_y = mark_y[::stride]
            visible = np.fromiter(
                (
                    bool(visibility_predicate(float(x_m), float(y_m)))
                    for x_m, y_m in zip(mark_x, mark_y, strict=False)
                ),
                dtype=bool,
                count=mark_x.size,
            )
            mark_x = mark_x[visible]
            mark_y = mark_y[visible]
        self._mark_xy_arrays(mark_x, mark_y)

    def mark_rectangular(
        self,
        center_xy: tuple[float, float],
        width_m: float,
        height_m: float,
        yaw_rad: float = 0.0,
    ) -> None:
        cx, cy = center_xy
        res = self.bounds.resolution_m
        hw, hh = width_m / 2, height_m / 2
        cos_y, sin_y = np.cos(yaw_rad), np.sin(yaw_rad)
        us = np.arange(-hw, hw + res, res)
        vs = np.arange(-hh, hh + res, res)
        grid_u, grid_v = np.meshgrid(us, vs, indexing="ij")
        xs = cx + cos_y * grid_u - sin_y * grid_v
        ys = cy + sin_y * grid_u + cos_y * grid_v
        self._mark_xy_arrays(xs.ravel(), ys.ravel())

    def _mark_xy_arrays(self, xs: np.ndarray, ys: np.ndarray) -> None:
        """Increment counts for cell-centre coordinate arrays (bounds-checked)."""
        b = self.bounds
        keep = (xs >= b.x_min_m) & (xs <= b.x_max_m) & (ys >= b.y_min_m) & (ys <= b.y_max_m)
        if not keep.any():
            return
        xs = xs[keep]
        ys = ys[keep]
        nx, ny = self._count.shape
        i = ((xs - b.x_min_m) / b.resolution_m).astype(np.int64)
        j = ((ys - b.y_min_m) / b.resolution_m).astype(np.int64)
        np.clip(i, 0, nx - 1, out=i)
        np.clip(j, 0, ny - 1, out=j)
        np.add.at(self._count, (i, j), 1)

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
