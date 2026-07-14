"""Runtime belief-world query interface for mapping.

This module is the versioned seam between the coverage/world-map runtime and any
belief consumer — the viewer's terrain reconstruction today, planners and mission
routing next. It exposes coverage, observed terrain height, occupancy, and
geofence *belief* without letting consumers touch raw arrays or simulation truth:
the truth read stays inside observation synthesis (the world-map sensor ingest),
and everything downstream reads back through this facade.

Contract:

* :class:`BeliefQuery` is the runtime protocol consumers depend on.
* :class:`WorldBeliefQuery` is the default backend over a :class:`WorldMap`.
* Every backend exposes ``source_id`` / ``version`` for provenance/lineage, and
  ``BELIEF_QUERY_CONTRACT_VERSION`` versions the interface shape itself.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from argusnet.core.types import BeliefCell, BeliefCellStatus
from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid
from argusnet.mapping.world_map import WorldMap

# Version of the BeliefQuery interface shape (query surface / summary semantics),
# distinct from an individual backend's ``version``. Bump on breaking changes.
BELIEF_QUERY_CONTRACT_VERSION = "1.0"

GeofencePredicate = Callable[[float, float], bool]

__all__ = [
    "BELIEF_QUERY_CONTRACT_VERSION",
    "BeliefQuery",
    "BeliefSummary",
    "FrontierCandidate",
    "WorldBeliefQuery",
]


@dataclass(frozen=True)
class FrontierCandidate:
    cell: BeliefCell
    score: float
    expected_coverage_gain: float
    uncertainty_reduction: float
    travel_cost_m: float
    risk_cost: float


@dataclass(frozen=True)
class BeliefSummary:
    """Aggregate belief statistics over the mapped area (a query result).

    Cheap to compute (vectorized over the coverage/height grids) so the mission
    loop can attach it to every replay frame.
    """

    total_cells: int
    observed_cells: int
    unknown_cells: int
    unsafe_cells: int
    frontier_cells: int
    coverage_fraction: float
    mean_belief_confidence: float
    mean_height_uncertainty_m: float | None


@runtime_checkable
class BeliefQuery(Protocol):
    """Runtime interface a belief backend exposes to mapping/planning consumers."""

    source_id: str
    version: str

    def height_estimate_grid(self) -> np.ndarray:
        """Dense believed terrain height (NaN where unobserved), shape (nx, ny)."""
        ...

    def belief_summary(self) -> BeliefSummary:
        """Aggregate belief statistics over the mapped area."""
        ...


class WorldBeliefQuery:
    """Default :class:`BeliefQuery` backend over coverage/world-map runtime state.

    Beyond the protocol surface it offers richer per-cell queries (height,
    occupancy, geofence, frontier scoring) for planning consumers.
    """

    def __init__(
        self,
        world_map: WorldMap | None = None,
        *,
        coverage_map: CoverageMap | None = None,
        occupancy_grid: OccupancyGrid | None = None,
        geofence: GeofencePredicate | None = None,
        last_observed_s_grid: np.ndarray | None = None,
        semantic_label_at: Callable[[float, float], str | None] | None = None,
        source_id: str = "grid",
        version: str = "1.0",
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
        self.source_id = source_id
        self.version = version

    def height_estimate_at(self, x_m: float, y_m: float) -> float | None:
        if self.world_map is None:
            return None
        i, j = self.bounds.xy_to_ij(x_m, y_m)
        # Prefer the dense observed-height belief layer; fall back to the legacy
        # nadir-height mean grid for callers that did not supply per-cell heights.
        observed = float(self.world_map.observed_height_grid[i, j])
        if np.isfinite(observed):
            return observed
        value = float(self.world_map.mean_height_grid[i, j])
        return value if np.isfinite(value) else None

    def height_estimate_grid(self) -> np.ndarray:
        """Dense believed terrain height (NaN where unobserved), shape (nx, ny).

        Combines the observed-height belief layer with the legacy nadir mean grid
        so the estimate is defined wherever either source has data.
        """
        if self.world_map is None:
            return np.full((self.bounds.nx, self.bounds.ny), np.nan, dtype=np.float64)
        observed = np.asarray(self.world_map.observed_height_grid, dtype=np.float64)
        mean = self.world_map.mean_height_grid
        grid: np.ndarray = np.where(np.isfinite(observed), observed, mean)
        return grid

    def belief_summary(self) -> BeliefSummary:
        """Vectorized aggregate belief statistics over the mapped area."""
        counts = self.coverage_map.count_grid
        observed_mask = counts > 0
        total_cells = int(counts.size)
        observed_cells = int(observed_mask.sum())

        # Per-cell coverage confidence, discounted by obstacle probability when an
        # occupancy grid is available. Vectorized; no per-cell Python callback.
        coverage_conf = np.clip(counts / 5.0, 0.0, 1.0)
        if self.occupancy_grid is not None:
            obstacle_p = np.clip(np.asarray(self.occupancy_grid.occupancy_grid), 0.0, 1.0)
        else:
            obstacle_p = np.zeros_like(coverage_conf)
        confidence = np.maximum(0.0, coverage_conf * (1.0 - obstacle_p))
        mean_conf = float(confidence[observed_mask].mean()) if observed_cells else 0.0
        unsafe_cells = int((observed_mask & (obstacle_p >= 0.4)).sum())

        # Height uncertainty proxy: resolution / sqrt(visits) over observed cells.
        if observed_cells:
            unc = self.bounds.resolution_m / np.sqrt(counts[observed_mask])
            mean_height_unc: float | None = float(unc.mean())
        else:
            mean_height_unc = None

        # Frontier cells: unobserved cells 4/8-adjacent to observed area.
        frontier_cells = 0
        if observed_cells and observed_cells < total_cells:
            padded = np.pad(observed_mask, 1, constant_values=False)
            neighbour = padded[:-2, 1:-1] | padded[2:, 1:-1] | padded[1:-1, :-2] | padded[1:-1, 2:]
            frontier_cells = int((~observed_mask & neighbour).sum())

        return BeliefSummary(
            total_cells=total_cells,
            observed_cells=observed_cells,
            unknown_cells=total_cells - observed_cells,
            unsafe_cells=unsafe_cells,
            frontier_cells=frontier_cells,
            coverage_fraction=(observed_cells / total_cells) if total_cells else 0.0,
            mean_belief_confidence=mean_conf,
            mean_height_uncertainty_m=mean_height_unc,
        )

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
