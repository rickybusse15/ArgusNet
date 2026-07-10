"""Planning-facing belief-world query helpers.

This module adapts the current coverage/world-map runtime into the roadmap
``WorldBeliefQuery`` contract without making planners depend on raw arrays or
simulation truth.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np

from argusnet.core.types import BeliefCell, BeliefCellStatus
from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid
from argusnet.mapping.world_map import WorldMap

GeofencePredicate = Callable[[float, float], bool]


@dataclass(frozen=True)
class FrontierCandidate:
    cell: BeliefCell
    score: float
    expected_coverage_gain: float
    uncertainty_reduction: float
    travel_cost_m: float
    risk_cost: float


class WorldBeliefQuery:
    """Read-only query facade for coverage, height, occupancy, and geofence belief."""

    def __init__(
        self,
        world_map: WorldMap | None = None,
        *,
        coverage_map: CoverageMap | None = None,
        occupancy_grid: OccupancyGrid | None = None,
        geofence: GeofencePredicate | None = None,
        last_observed_s_grid: np.ndarray | None = None,
        semantic_label_at: Callable[[float, float], str | None] | None = None,
    ) -> None:
        if world_map is None and coverage_map is None:
            raise ValueError("WorldBeliefQuery requires a WorldMap or CoverageMap.")
        self.world_map = world_map
        self.coverage_map = coverage_map or world_map.coverage_map  # type: ignore[union-attr]
        self.occupancy_grid = occupancy_grid
        self.bounds: GridBounds = self.coverage_map.bounds
        self.geofence = geofence
        self.last_observed_s_grid = last_observed_s_grid
        self.semantic_label_at = semantic_label_at

    def height_estimate_at(self, x_m: float, y_m: float) -> float | None:
        if self.world_map is None:
            return None
        i, j = self.bounds.xy_to_ij(x_m, y_m)
        value = float(self.world_map.mean_height_grid[i, j])
        return value if np.isfinite(value) else None

    def height_uncertainty_at(self, x_m: float, y_m: float) -> float | None:
        visits = self.coverage_at(x_m, y_m)
        if visits <= 0:
            return None
        # A conservative proxy until dense elevation uncertainty is wired in.
        return float(self.bounds.resolution_m / np.sqrt(visits))

    def obstacle_probability_at(self, x_m: float, y_m: float) -> float:
        if self.occupancy_grid is None:
            return 0.5 if not self.is_observed(x_m, y_m) else 0.0
        return float(self.occupancy_grid.occupancy_at(x_m, y_m))

    def coverage_at(self, x_m: float, y_m: float) -> int:
        if not self._in_bounds(x_m, y_m):
            return 0
        return int(self.coverage_map.count_at(x_m, y_m))

    def confidence_at(self, x_m: float, y_m: float) -> float:
        visits = self.coverage_at(x_m, y_m)
        if visits <= 0 or not self.is_inside_geofence(x_m, y_m):
            return 0.0
        coverage_conf = min(1.0, visits / 5.0)
        obstacle_risk = self.obstacle_probability_at(x_m, y_m)
        return float(max(0.0, coverage_conf * (1.0 - obstacle_risk)))

    def is_observed(self, x_m: float, y_m: float) -> bool:
        return self.coverage_at(x_m, y_m) > 0

    def is_inside_geofence(self, x_m: float, y_m: float) -> bool:
        if not self._in_bounds(x_m, y_m):
            return False
        return True if self.geofence is None else bool(self.geofence(x_m, y_m))

    def is_known_safe(self, x_m: float, y_m: float) -> bool:
        return (
            self.is_inside_geofence(x_m, y_m)
            and self.is_observed(x_m, y_m)
            and self.obstacle_probability_at(x_m, y_m) < 0.4
        )

    def cell_at(self, x_m: float, y_m: float) -> BeliefCell:
        i, j = self.bounds.xy_to_ij(x_m, y_m)
        cx, cy = self.bounds.ij_to_xy(i, j)
        return self.cell_ij(i, j, center_xy=(cx, cy))

    def cell_ij(
        self,
        i: int,
        j: int,
        *,
        center_xy: tuple[float, float] | None = None,
    ) -> BeliefCell:
        x_m, y_m = center_xy or self.bounds.ij_to_xy(i, j)
        inside = self.is_inside_geofence(x_m, y_m)
        observed = self.is_observed(x_m, y_m)
        obstacle_probability = self.obstacle_probability_at(x_m, y_m)
        if not inside:
            status = BeliefCellStatus.OUTSIDE_GEOFENCE.value
        elif not observed:
            status = BeliefCellStatus.UNKNOWN.value
        elif obstacle_probability >= 0.6:
            status = BeliefCellStatus.KNOWN_OBSTACLE.value
        elif obstacle_probability >= 0.4:
            status = BeliefCellStatus.UNCERTAIN.value
        else:
            status = BeliefCellStatus.KNOWN_SAFE.value

        last_observed_s = None
        if self.last_observed_s_grid is not None:
            try:
                raw = float(self.last_observed_s_grid[i, j])
                last_observed_s = raw if np.isfinite(raw) else None
            except IndexError:
                last_observed_s = None

        source_ids: tuple[str, ...] = ()
        if self.semantic_label_at is not None:
            label = self.semantic_label_at(x_m, y_m)
            if label:
                source_ids = (f"semantic:{label}",)

        return BeliefCell(
            cell_id=f"cell-{i}-{j}",
            ij=(int(i), int(j)),
            center_xy_m=(float(x_m), float(y_m)),
            height_estimate_m=self.height_estimate_at(x_m, y_m),
            height_uncertainty_m=self.height_uncertainty_at(x_m, y_m),
            obstacle_probability=obstacle_probability,
            terrain_confidence=self.confidence_at(x_m, y_m),
            coverage_count=self.coverage_at(x_m, y_m),
            last_observed_s=last_observed_s,
            inside_geofence=inside,
            status=status,
            source_ids=source_ids,
        )

    def frontier_cells(self) -> list[BeliefCell]:
        grid = self.coverage_map.count_grid
        covered = grid > 0
        result: list[BeliefCell] = []
        for i, j in zip(*np.where(~covered), strict=False):
            x_m, y_m = self.bounds.ij_to_xy(int(i), int(j))
            if not self.is_inside_geofence(x_m, y_m):
                continue
            patch = covered[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2]
            if patch.any():
                result.append(self.cell_ij(int(i), int(j), center_xy=(x_m, y_m)))
        return result

    def safe_corridor_between(
        self,
        start_xy_m: tuple[float, float],
        end_xy_m: tuple[float, float],
        *,
        sample_count: int = 16,
        allow_unknown: bool = False,
    ) -> bool:
        sx, sy = start_xy_m
        ex, ey = end_xy_m
        for alpha in np.linspace(0.0, 1.0, max(2, sample_count)):
            x_m = float(sx + (ex - sx) * alpha)
            y_m = float(sy + (ey - sy) * alpha)
            if not self.is_inside_geofence(x_m, y_m):
                return False
            if allow_unknown and not self.is_observed(x_m, y_m):
                continue
            if not self.is_known_safe(x_m, y_m):
                return False
        return True

    def score_frontiers(
        self,
        origin_xy_m: tuple[float, float],
        *,
        max_candidates: int = 10,
    ) -> list[FrontierCandidate]:
        candidates: list[FrontierCandidate] = []
        ox, oy = origin_xy_m
        for cell in self.frontier_cells():
            x_m, y_m = cell.center_xy_m
            travel_cost = float(np.hypot(x_m - ox, y_m - oy))
            expected_gain = self._expected_coverage_gain(cell.ij)
            uncertainty_reduction = 1.0 if cell.status == BeliefCellStatus.UNKNOWN.value else 0.25
            risk_cost = cell.obstacle_probability + (0.0 if cell.inside_geofence else 1.0)
            score = expected_gain + uncertainty_reduction - 0.002 * travel_cost - risk_cost
            candidates.append(
                FrontierCandidate(
                    cell=cell,
                    score=float(score),
                    expected_coverage_gain=float(expected_gain),
                    uncertainty_reduction=float(uncertainty_reduction),
                    travel_cost_m=travel_cost,
                    risk_cost=float(risk_cost),
                )
            )
        candidates.sort(key=lambda candidate: (-candidate.score, candidate.cell.ij))
        return candidates[:max_candidates]

    def cells(self) -> Iterable[BeliefCell]:
        for i in range(self.bounds.nx):
            for j in range(self.bounds.ny):
                yield self.cell_ij(i, j)

    def _expected_coverage_gain(self, ij: tuple[int, int]) -> float:
        i, j = ij
        grid = self.coverage_map.count_grid
        patch = grid[max(0, i - 1) : i + 2, max(0, j - 1) : j + 2]
        return float((patch == 0).sum() / max(1, patch.size))

    def _in_bounds(self, x_m: float, y_m: float) -> bool:
        b = self.bounds
        return b.x_min_m <= x_m <= b.x_max_m and b.y_min_m <= y_m <= b.y_max_m
