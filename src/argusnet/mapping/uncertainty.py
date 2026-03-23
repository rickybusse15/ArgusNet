"""Per-cell observation uncertainty field for ArgusNet.

Tracks the running mean and variance of any scalar quantity (e.g. height,
reflectance) observed at each grid cell.  Useful for identifying regions
where measurements are inconsistent or sparse.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.mapping.occupancy import GridBounds

__all__ = [
    "UncertaintyField",
    "CellStats",
]


@dataclass(frozen=True)
class CellStats:
    count: int
    mean: float
    variance: float
    std: float


class UncertaintyField:
    """Welford online mean and variance estimator per grid cell."""

    def __init__(self, bounds: GridBounds) -> None:
        self.bounds = bounds
        self._count = np.zeros((bounds.nx, bounds.ny), dtype=np.int32)
        self._mean = np.zeros((bounds.nx, bounds.ny), dtype=float)
        self._M2 = np.zeros((bounds.nx, bounds.ny), dtype=float)

    def update(self, x: float, y: float, value: float) -> None:
        """Add an observation at (x, y) with *value*."""
        if not (
            self.bounds.x_min_m <= x <= self.bounds.x_max_m
            and self.bounds.y_min_m <= y <= self.bounds.y_max_m
        ):
            return
        i, j = self.bounds.xy_to_ij(x, y)
        n = self._count[i, j] + 1
        self._count[i, j] = n
        delta = value - self._mean[i, j]
        self._mean[i, j] += delta / n
        delta2 = value - self._mean[i, j]
        self._M2[i, j] += delta * delta2

    def stats_at(self, x: float, y: float) -> CellStats:
        i, j = self.bounds.xy_to_ij(x, y)
        n = int(self._count[i, j])
        mean = float(self._mean[i, j])
        var = float(self._M2[i, j] / (n - 1)) if n >= 2 else 0.0
        return CellStats(count=n, mean=mean, variance=var, std=float(np.sqrt(var)))

    @property
    def variance_grid(self) -> np.ndarray:
        """Return (nx, ny) sample-variance array."""
        with np.errstate(invalid="ignore"):
            var = np.where(
                self._count >= 2,
                self._M2 / np.maximum(self._count - 1, 1),
                0.0,
            )
        return var

    @property
    def mean_grid(self) -> np.ndarray:
        return self._mean.copy()

    @property
    def count_grid(self) -> np.ndarray:
        return self._count.copy()
