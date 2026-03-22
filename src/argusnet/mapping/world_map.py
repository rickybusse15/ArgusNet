"""World map: accumulates sensor coverage and extracts terrain features.

During the scan phase each drone's sensor footprint is registered here.
After sufficient coverage the map can be queried for feature points that
are used by the localization engine.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds


@dataclass
class WorldMap:
    """Fused coverage + feature map for a single operational area.

    Usage::

        wm = WorldMap(bounds)
        # called each sim step per drone:
        wm.add_scan_observation(drone_pos, terrain_height, footprint_radius_m)
        # after scanning completes:
        features = wm.extract_features(max_features=50)
        snap = wm.snapshot()          # dict -> JSON-serialisable
    """

    bounds: GridBounds

    def __post_init__(self) -> None:
        self._coverage: CoverageMap = CoverageMap(self.bounds)
        # Internal arrays use the same (nx, ny) layout as CoverageMap._count,
        # where axis-0 is the x-index and axis-1 is the y-index.
        nx = self.bounds.nx
        ny = self.bounds.ny
        self._nx = nx
        self._ny = ny
        self._height_sum: np.ndarray = np.zeros((nx, ny), dtype=np.float64)
        self._height_count: np.ndarray = np.zeros((nx, ny), dtype=np.int32)
        self._scan_count: int = 0

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def add_scan_observation(
        self,
        drone_position: np.ndarray,
        terrain_height: float,
        footprint_radius_m: float,
    ) -> None:
        """Register one sensor footprint from a drone.

        Args:
            drone_position: Drone XYZ in metres.
            terrain_height: Terrain height directly below the drone (metres).
            footprint_radius_m: Circular footprint radius on the ground.
        """
        cx, cy = float(drone_position[0]), float(drone_position[1])
        self._coverage.mark_circular(center_xy=(cx, cy), radius_m=footprint_radius_m)

        # Record terrain height in cells within footprint.
        # Use (i, j) = (x_index, y_index) to match CoverageMap convention.
        b = self.bounds
        r_cells = max(1, int(footprint_radius_m / b.resolution_m))
        ci = int((cx - b.x_min_m) / b.resolution_m)  # x cell index
        cj = int((cy - b.y_min_m) / b.resolution_m)  # y cell index
        i_range = np.arange(max(0, ci - r_cells), min(self._nx, ci + r_cells + 1))
        j_range = np.arange(max(0, cj - r_cells), min(self._ny, cj + r_cells + 1))
        if i_range.size == 0 or j_range.size == 0:
            return
        gi, gj = np.meshgrid(i_range, j_range, indexing="ij")
        dist_sq = (gi - ci) ** 2 + (gj - cj) ** 2
        mask = dist_sq <= r_cells ** 2
        np.add.at(self._height_sum, (gi[mask], gj[mask]), terrain_height)
        np.add.at(self._height_count, (gi[mask], gj[mask]), 1)
        self._scan_count += 1

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def coverage_map(self) -> CoverageMap:
        return self._coverage

    @property
    def coverage_fraction(self) -> float:
        return self._coverage.stats.coverage_fraction

    @property
    def mean_height_grid(self) -> np.ndarray:
        """Returns mean observed terrain height per cell (NaN where unobserved).

        Shape is (nx, ny) matching CoverageMap layout.
        """
        with np.errstate(invalid="ignore"):
            result = np.where(
                self._height_count > 0,
                self._height_sum / np.maximum(self._height_count, 1),
                np.nan,
            )
        return result

    def coverage_in_region(self, center_xy: Tuple[float, float], radius_m: float) -> float:
        """Fraction of cells within radius that have been visited at least once."""
        b = self.bounds
        ci = int((center_xy[0] - b.x_min_m) / b.resolution_m)
        cj = int((center_xy[1] - b.y_min_m) / b.resolution_m)
        r_cells = max(1, int(radius_m / b.resolution_m))
        i_range = np.arange(max(0, ci - r_cells), min(self._nx, ci + r_cells + 1))
        j_range = np.arange(max(0, cj - r_cells), min(self._ny, cj + r_cells + 1))
        if i_range.size == 0 or j_range.size == 0:
            return 0.0
        gi, gj = np.meshgrid(i_range, j_range, indexing="ij")
        dist_sq = (gi - ci) ** 2 + (gj - cj) ** 2
        mask = dist_sq <= r_cells ** 2
        # count_grid returns a copy with shape (nx, ny).
        counts = self._coverage.count_grid
        patch = counts[gi[mask], gj[mask]]
        if patch.size == 0:
            return 0.0
        return float((patch > 0).mean())

    def extract_features(self, max_features: int = 50) -> List[dict]:
        """Extract local terrain height maxima as map features.

        Returns a list of dicts (JSON-serialisable) with keys:
        feature_id, position (list[float]), feature_type, height_m, confidence.
        """
        hgrid = self.mean_height_grid  # shape (nx, ny)
        b = self.bounds
        features: List[dict] = []

        valid = np.isfinite(hgrid)
        if not valid.any():
            return features

        # Non-maximum suppression on 3x3 neighbourhoods.
        padded = np.pad(hgrid, 1, constant_values=np.nan)
        nx, ny = hgrid.shape

        candidate_is, candidate_js = [], []
        for ii in range(nx):
            for jj in range(ny):
                if not valid[ii, jj]:
                    continue
                neighbourhood = padded[ii:ii + 3, jj:jj + 3]
                local_max = np.nanmax(neighbourhood)
                if hgrid[ii, jj] >= local_max:
                    candidate_is.append(ii)
                    candidate_js.append(jj)

        if not candidate_is:
            return features

        candidate_is = np.array(candidate_is)
        candidate_js = np.array(candidate_js)
        heights = hgrid[candidate_is, candidate_js]
        order = np.argsort(-heights)[:max_features]

        counts = self._coverage.count_grid  # shape (nx, ny)

        for rank, idx in enumerate(order):
            ii, jj = int(candidate_is[idx]), int(candidate_js[idx])
            x_m, y_m = b.ij_to_xy(ii, jj)
            h = float(hgrid[ii, jj])
            visits = int(counts[ii, jj])
            confidence = min(1.0, visits / 5.0)
            features.append({
                "feature_id": f"feat_{rank:04d}",
                "position": [round(x_m, 2), round(y_m, 2), round(h, 2)],
                "feature_type": "terrain_peak",
                "height_m": round(h, 2),
                "confidence": round(confidence, 3),
            })

        return features

    def snapshot(self) -> dict:
        """Return a JSON-serialisable snapshot of key map statistics."""
        stats = self._coverage.stats
        return {
            "coverage_fraction": round(stats.coverage_fraction, 4),
            "covered_cells": stats.covered_cells,
            "total_cells": stats.total_cells,
            "mean_revisits": round(stats.mean_revisits, 2),
            "scan_observations": self._scan_count,
            "feature_count": len(self.extract_features()),
        }
