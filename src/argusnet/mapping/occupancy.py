"""2.5D voxel occupancy grid for ArgusNet.

Represents the environment as a 2-D grid of cells, each with an
occupancy probability and a maximum height.  Suitable for drone path
planning and obstacle mapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from argusnet.core.types import Vector3

__all__ = [
    "OccupancyGrid",
    "OccupancyCell",
    "GridBounds",
]


@dataclass(frozen=True)
class GridBounds:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float
    resolution_m: float

    @property
    def nx(self) -> int:
        return max(1, int(np.ceil((self.x_max_m - self.x_min_m) / self.resolution_m)))

    @property
    def ny(self) -> int:
        return max(1, int(np.ceil((self.y_max_m - self.y_min_m) / self.resolution_m)))

    def xy_to_ij(self, x: float, y: float) -> Tuple[int, int]:
        i = int((x - self.x_min_m) / self.resolution_m)
        j = int((y - self.y_min_m) / self.resolution_m)
        return (
            int(np.clip(i, 0, self.nx - 1)),
            int(np.clip(j, 0, self.ny - 1)),
        )

    def ij_to_xy(self, i: int, j: int) -> Tuple[float, float]:
        x = self.x_min_m + (i + 0.5) * self.resolution_m
        y = self.y_min_m + (j + 0.5) * self.resolution_m
        return x, y


@dataclass
class OccupancyCell:
    occupancy: float = 0.0
    """Occupancy probability [0, 1].  0 = free, 1 = occupied."""

    max_height_m: float = 0.0
    """Maximum obstacle height within this cell (metres above terrain)."""

    observation_count: int = 0


class OccupancyGrid:
    """2.5D occupancy grid using a log-odds update model.

    Cells start at 0.5 (unknown).  Each observation increments or
    decrements the log-odds, clamped to [-4, 4] to avoid lock-in.
    """

    LOG_ODDS_FREE = -0.4
    LOG_ODDS_OCC = 0.85
    LOG_ODDS_MIN = -4.0
    LOG_ODDS_MAX = 4.0

    def __init__(self, bounds: GridBounds) -> None:
        self.bounds = bounds
        # log-odds grid; 0.0 = p=0.5 (unknown)
        self._log_odds = np.zeros((bounds.nx, bounds.ny), dtype=float)
        self._max_height = np.zeros((bounds.nx, bounds.ny), dtype=float)
        self._obs_count = np.zeros((bounds.nx, bounds.ny), dtype=int)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def mark_free(self, x: float, y: float) -> None:
        i, j = self.bounds.xy_to_ij(x, y)
        self._log_odds[i, j] = np.clip(
            self._log_odds[i, j] + self.LOG_ODDS_FREE,
            self.LOG_ODDS_MIN, self.LOG_ODDS_MAX
        )
        self._obs_count[i, j] += 1

    def mark_occupied(self, x: float, y: float, height_m: float = 0.0) -> None:
        i, j = self.bounds.xy_to_ij(x, y)
        self._log_odds[i, j] = np.clip(
            self._log_odds[i, j] + self.LOG_ODDS_OCC,
            self.LOG_ODDS_MIN, self.LOG_ODDS_MAX
        )
        self._max_height[i, j] = max(self._max_height[i, j], height_m)
        self._obs_count[i, j] += 1

    def mark_ray(
        self,
        origin: Vector3,
        direction: Vector3,
        hit_range_m: Optional[float],
        max_range_m: float = 200.0,
    ) -> None:
        """Mark cells along a ray as free; mark hit cell as occupied."""
        orig = np.asarray(origin[:2], dtype=float)
        d = np.asarray(direction[:2], dtype=float)
        norm = np.linalg.norm(d)
        if norm < 1e-9:
            return
        d = d / norm
        end_range = hit_range_m if hit_range_m is not None else max_range_m
        steps = int(end_range / self.bounds.resolution_m) + 1
        for k in range(steps):
            p = orig + d * k * self.bounds.resolution_m
            self.mark_free(float(p[0]), float(p[1]))
        if hit_range_m is not None:
            hit = orig + d * hit_range_m
            self.mark_occupied(
                float(hit[0]), float(hit[1]),
                float(direction[2]) * hit_range_m if len(direction) > 2 else 0.0,
            )

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def occupancy_at(self, x: float, y: float) -> float:
        i, j = self.bounds.xy_to_ij(x, y)
        lo = self._log_odds[i, j]
        return float(1.0 / (1.0 + np.exp(-lo)))

    def max_height_at(self, x: float, y: float) -> float:
        i, j = self.bounds.xy_to_ij(x, y)
        return float(self._max_height[i, j])

    def is_free(self, x: float, y: float, threshold: float = 0.4) -> bool:
        return self.occupancy_at(x, y) < threshold

    def is_occupied(self, x: float, y: float, threshold: float = 0.6) -> bool:
        return self.occupancy_at(x, y) >= threshold

    def cell(self, x: float, y: float) -> OccupancyCell:
        i, j = self.bounds.xy_to_ij(x, y)
        return OccupancyCell(
            occupancy=self.occupancy_at(x, y),
            max_height_m=float(self._max_height[i, j]),
            observation_count=int(self._obs_count[i, j]),
        )

    @property
    def occupancy_grid(self) -> np.ndarray:
        """Return (nx, ny) probability grid [0, 1]."""
        return 1.0 / (1.0 + np.exp(-self._log_odds))
