"""Simulation-layer coverage map with vectorized numpy operations.

Provides a CoverageMap that tracks which grid cells have been observed by
sensor footprints during a simulation run, using numpy for performance.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "CoverageMap",
]


class CoverageMap:
    """Tracks how many times each cell in a 2-D grid has been observed.

    All coordinates are in metres (local projected XY).  The grid is
    axis-aligned with cell size *resolution_m*.

    Parameters
    ----------
    x_min_m, x_max_m, y_min_m, y_max_m:
        World-coordinate extents of the grid.
    resolution_m:
        Side length of each square grid cell in metres.
    """

    def __init__(
        self,
        x_min_m: float,
        x_max_m: float,
        y_min_m: float,
        y_max_m: float,
        resolution_m: float,
    ) -> None:
        if resolution_m <= 0:
            raise ValueError("resolution_m must be positive.")
        self.x_min_m = float(x_min_m)
        self.x_max_m = float(x_max_m)
        self.y_min_m = float(y_min_m)
        self.y_max_m = float(y_max_m)
        self.resolution_m = float(resolution_m)

        self.nx = max(1, int(np.ceil((x_max_m - x_min_m) / resolution_m)))
        self.ny = max(1, int(np.ceil((y_max_m - y_min_m) / resolution_m)))

        # counts[row, col] == counts[iy, ix]
        self.counts = np.zeros((self.ny, self.nx), dtype=np.int32)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _xy_to_ix(self, x_m: float) -> int:
        """Convert world X to grid column index (clipped)."""
        ix = int(np.floor((x_m - self.x_min_m) / self.resolution_m))
        return int(np.clip(ix, 0, self.nx - 1))

    def _xy_to_iy(self, y_m: float) -> int:
        """Convert world Y to grid row index (clipped)."""
        iy = int(np.floor((y_m - self.y_min_m) / self.resolution_m))
        return int(np.clip(iy, 0, self.ny - 1))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def mark(self, x_m: float, y_m: float) -> None:
        """Increment the observation count for the cell at *(x_m, y_m)*.

        Uses numpy integer arithmetic rather than ``floor`` + ``int`` so
        that the vectorized path and the scalar path share the same
        rounding semantics.
        """
        ix = int(np.floor((x_m - self.x_min_m) / self.resolution_m))
        iy = int(np.floor((y_m - self.y_min_m) / self.resolution_m))
        ix = int(np.clip(ix, 0, self.nx - 1))
        iy = int(np.clip(iy, 0, self.ny - 1))
        self.counts[iy, ix] += 1

    def circular_footprint(
        self,
        cx_m: float,
        cy_m: float,
        radius_m: float,
    ) -> None:
        """Mark all cells whose centres fall within *radius_m* of *(cx_m, cy_m)*.

        Uses fully vectorised numpy operations (meshgrid + boolean mask) so
        that no Python-level loops over individual cells are needed.
        """
        res = self.resolution_m
        r_cells = radius_m / res

        # Grid-space coordinates of the circle centre
        cx_cell = (cx_m - self.x_min_m) / res
        cy_cell = (cy_m - self.y_min_m) / res

        # Bounding box of cells that *could* be inside the circle
        ix_min = int(np.clip(int(np.floor(cx_cell - r_cells)), 0, self.nx - 1))
        ix_max = int(np.clip(int(np.ceil(cx_cell + r_cells)), 0, self.nx - 1))
        iy_min = int(np.clip(int(np.floor(cy_cell - r_cells)), 0, self.ny - 1))
        iy_max = int(np.clip(int(np.ceil(cy_cell + r_cells)), 0, self.ny - 1))

        if ix_min > ix_max or iy_min > iy_max:
            return

        xs = np.arange(ix_min, ix_max + 1)
        ys = np.arange(iy_min, iy_max + 1)
        gx, gy = np.meshgrid(xs, ys)  # shapes: (len(ys), len(xs))

        dist_sq = (gx - cx_cell) ** 2 + (gy - cy_cell) ** 2
        mask = dist_sq <= r_cells ** 2

        # Safe scatter-add: use np.add.at to handle any duplicate indices
        # (in practice the mask produces unique indices, but this is robust).
        np.add.at(self.counts, (gy[mask], gx[mask]), 1)

    def bulk_mark(self, positions_xy: np.ndarray) -> None:
        """Mark an Nx2 array of (x, y) world-coordinate positions in one pass.

        Converts all positions to grid indices with ``np.floor`` + ``np.clip``
        and uses advanced indexing for a single scatter-add, so no Python
        loop over individual points is needed.

        Parameters
        ----------
        positions_xy:
            Array of shape ``(N, 2)`` with columns [x_m, y_m].
            Positions outside the grid bounds are clipped to the nearest cell.
        """
        positions_xy = np.asarray(positions_xy, dtype=float)
        if positions_xy.ndim != 2 or positions_xy.shape[1] != 2:
            raise ValueError("positions_xy must be an Nx2 array.")
        if len(positions_xy) == 0:
            return

        xs = positions_xy[:, 0]
        ys = positions_xy[:, 1]

        ix = np.floor((xs - self.x_min_m) / self.resolution_m).astype(np.int64)
        iy = np.floor((ys - self.y_min_m) / self.resolution_m).astype(np.int64)

        ix = np.clip(ix, 0, self.nx - 1)
        iy = np.clip(iy, 0, self.ny - 1)

        # np.add.at handles repeated indices correctly (accumulates).
        np.add.at(self.counts, (iy, ix), 1)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def count_at(self, x_m: float, y_m: float) -> int:
        """Return the observation count for the cell containing *(x_m, y_m)*."""
        return int(self.counts[self._xy_to_iy(y_m), self._xy_to_ix(x_m)])

    def is_covered(self, x_m: float, y_m: float) -> bool:
        """Return ``True`` if the cell at *(x_m, y_m)* has been observed."""
        return self.count_at(x_m, y_m) > 0

    @property
    def coverage_fraction(self) -> float:
        """Fraction of cells with at least one observation [0, 1]."""
        total = self.counts.size
        if total == 0:
            return 0.0
        return float(np.sum(self.counts > 0)) / total

    @property
    def covered_cells(self) -> int:
        """Number of cells with at least one observation."""
        return int(np.sum(self.counts > 0))

    @property
    def total_cells(self) -> int:
        """Total number of cells in the grid."""
        return self.counts.size

    @property
    def mean_revisits(self) -> float:
        """Mean number of observations per covered cell (0 if none covered)."""
        covered = int(np.sum(self.counts > 0))
        if covered == 0:
            return 0.0
        return float(np.sum(self.counts)) / covered

    @property
    def count_grid(self) -> np.ndarray:
        """Return a copy of the (ny, nx) integer observation count array."""
        return self.counts.copy()
