"""Composable world model container for ArgusNet.

Holds elevation, occupancy, coverage, and uncertainty maps together
and provides a unified query interface for planning and evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from argusnet.mapping.semantics import SemanticMap

import numpy as np

from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.elevation import ElevationMap, flat_elevation_map
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid
from argusnet.mapping.uncertainty import UncertaintyField

__all__ = [
    "WorldModel",
    "WorldModelConfig",
]


@dataclass(frozen=True)
class WorldModelConfig:
    """Configuration for the WorldModel grid."""

    x_min_m: float = -1000.0
    x_max_m: float = 1000.0
    y_min_m: float = -1000.0
    y_max_m: float = 1000.0
    resolution_m: float = 5.0
    """Grid cell size in metres."""

    @property
    def bounds(self) -> GridBounds:
        return GridBounds(
            x_min_m=self.x_min_m,
            x_max_m=self.x_max_m,
            y_min_m=self.y_min_m,
            y_max_m=self.y_max_m,
            resolution_m=self.resolution_m,
        )


class WorldModel:
    """Runtime world model: elevation + occupancy + coverage + uncertainty.

    All layers share the same :class:`GridBounds`.  The elevation map
    may have a different (finer) internal resolution — it is queried by
    point rather than by cell index.
    """

    def __init__(
        self,
        config: WorldModelConfig,
        elevation: ElevationMap | None = None,
    ) -> None:
        self.config = config
        bounds = config.bounds

        self.elevation: ElevationMap = elevation or flat_elevation_map()
        self.occupancy: OccupancyGrid = OccupancyGrid(bounds)
        self.coverage: CoverageMap = CoverageMap(bounds)
        self.uncertainty: UncertaintyField = UncertaintyField(bounds)

    # ------------------------------------------------------------------
    # Convenience queries
    # ------------------------------------------------------------------

    def agl(self, x: float, y: float, z: float) -> float:
        """Height above ground level at (x, y, z)."""
        return z - self.elevation.at_point(x, y)

    def is_safe_altitude(self, x: float, y: float, z: float, clearance_m: float = 5.0) -> bool:
        """True if z is at least *clearance_m* above all known obstacles."""
        ground = self.elevation.at_point(x, y)
        obs_h = self.occupancy.max_height_at(x, y)
        return z >= max(ground, obs_h) + clearance_m

    def is_free(self, x: float, y: float) -> bool:
        return self.occupancy.is_free(x, y)

    def fuse_semantic_into_occupancy(
        self,
        semantic_map: SemanticMap,
        building_confidence_threshold: float = 0.7,
    ) -> int:
        """Fuse semantic labels into occupancy and elevation constraints.

        For BUILDING cells with confidence > threshold: force occupied.
        For WATER cells: clear occupancy (surface is free, but enforces no-fly zone).

        Returns the number of cells updated.
        """
        from argusnet.mapping.semantics import SemanticLabel

        updated = 0
        for r in range(semantic_map.rows):
            for c in range(semantic_map.cols):
                cell = semantic_map._cells[r][c]
                if cell.total_observations == 0:
                    continue
                x, y = semantic_map.cell_center(r, c)
                label = cell.dominant_label
                conf = cell.confidence
                if label == int(SemanticLabel.BUILDING) and conf >= building_confidence_threshold:
                    # Force the corresponding occupancy cell to occupied
                    elev = self.elevation.at_point(x, y)
                    self.occupancy.mark_occupied(
                        x, y, height_m=elev + 10.0
                    )  # 10m building height default
                    updated += 1
                elif label == int(SemanticLabel.WATER):
                    # Water surface is free, but should be noted
                    self.occupancy.mark_free(x, y)
                    updated += 1
        return updated

    # ------------------------------------------------------------------
    # Serialisation (lightweight — arrays as npy blobs)
    # ------------------------------------------------------------------

    def save(self, directory: str | Path) -> None:
        """Persist the occupancy, coverage, and uncertainty grids."""
        p = Path(directory)
        p.mkdir(parents=True, exist_ok=True)
        np.save(p / "occupancy_log_odds.npy", self.occupancy._log_odds)
        np.save(p / "occupancy_max_height.npy", self.occupancy._max_height)
        np.save(p / "coverage_count.npy", self.coverage._count)
        np.save(p / "uncertainty_mean.npy", self.uncertainty._mean)
        np.save(p / "uncertainty_M2.npy", self.uncertainty._M2)
        np.save(p / "uncertainty_count.npy", self.uncertainty._count)
        cfg = {
            "x_min_m": self.config.x_min_m,
            "x_max_m": self.config.x_max_m,
            "y_min_m": self.config.y_min_m,
            "y_max_m": self.config.y_max_m,
            "resolution_m": self.config.resolution_m,
        }
        (p / "config.json").write_text(json.dumps(cfg, indent=2))

    @classmethod
    def load(cls, directory: str | Path) -> WorldModel:
        """Load a previously saved WorldModel."""
        p = Path(directory)
        cfg = json.loads((p / "config.json").read_text())
        config = WorldModelConfig(**cfg)
        model = cls(config)
        model.occupancy._log_odds = np.load(p / "occupancy_log_odds.npy")
        model.occupancy._max_height = np.load(p / "occupancy_max_height.npy")
        model.coverage._count = np.load(p / "coverage_count.npy")
        model.uncertainty._mean = np.load(p / "uncertainty_mean.npy")
        model.uncertainty._M2 = np.load(p / "uncertainty_M2.npy")
        model.uncertainty._count = np.load(p / "uncertainty_count.npy")
        return model
