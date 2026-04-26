"""Frontier-based exploration planner for the scanning phase."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from scipy.ndimage import binary_fill_holes, uniform_filter

__all__ = ["FrontierConfig", "ClaimedCells", "FrontierPlanner"]


@dataclass(frozen=True)
class FrontierConfig:
    """Parameters controlling frontier cell selection."""
    distance_weight: float = 1.0
    coverage_gradient_weight: float = 2.0
    # Scales from 0 → full between 25 % and 75 % coverage to prevent
    # drone clustering at low coverage when the gradient signal is noisy.
    exclusion_radius_cells: int = 3
    gap_fill_min_fraction: float = 0.02  # holes < 2 % of cells → allow transition


class ClaimedCells:
    """Thin wrapper tracking which drone is heading to which grid cell."""

    def __init__(self) -> None:
        self._assignment: Dict[str, Tuple[int, int]] = {}

    def claim(self, drone_id: str, cell: Tuple[int, int]) -> None:
        self._assignment[drone_id] = cell

    def release(self, drone_id: str) -> None:
        self._assignment.pop(drone_id, None)

    def others(self, drone_id: str) -> Set[Tuple[int, int]]:
        return {cell for did, cell in self._assignment.items() if did != drone_id}


class FrontierPlanner:
    """Selects frontier cells with coverage-gradient scoring."""

    def __init__(self, config: Optional[FrontierConfig] = None) -> None:
        self.cfg = config or FrontierConfig()

    def select_frontier_cell(
        self,
        cmap,                           # CoverageMap
        drone_xy: np.ndarray,           # (2,) metres
        claimed: ClaimedCells,
        drone_id: str,
    ) -> Optional[Tuple[int, int]]:
        """Return the best uncovered grid cell for *drone_id* to fly toward.

        Scoring (higher is better):
          ``-distance_weight * dist + gradient_weight * coverage_gradient``

        The gradient weight ramps from 0 → full between 25 % and 75 % overall
        coverage so that at low coverage drones spread out by distance alone,
        while at high coverage they prefer cells adjacent to already-covered
        area (filling holes faster).
        """
        cfg = self.cfg
        grid = cmap.count_grid           # shape (nx, ny), int
        nx, ny = grid.shape
        bounds = cmap.bounds

        ii, jj = np.where(grid == 0)
        if len(ii) == 0:
            return None

        # Exclusion mask: remove cells already claimed by other drones.
        other_claims = claimed.others(drone_id)
        if other_claims:
            excl_i = np.array([c[0] for c in other_claims], dtype=int)
            excl_j = np.array([c[1] for c in other_claims], dtype=int)
            r = cfg.exclusion_radius_cells
            mask = np.ones(len(ii), dtype=bool)
            for k in range(len(ii)):
                if np.any(
                    (np.abs(excl_i - ii[k]) <= r) & (np.abs(excl_j - jj[k]) <= r)
                ):
                    mask[k] = False
            candidates_i = ii[mask]
            candidates_j = jj[mask]
            if len(candidates_i) == 0:
                # Fallback: ignore exclusion
                candidates_i, candidates_j = ii, jj
        else:
            candidates_i, candidates_j = ii, jj

        # Distance score (in grid cells).
        gi, gj = bounds.xy_to_ij(float(drone_xy[0]), float(drone_xy[1]))
        dist_arr = np.hypot(candidates_i - gi, candidates_j - gj)

        # Coverage gradient: how many of the 8 neighbours are already covered.
        # Computed via a box-filter on a binary covered mask, then sampled at
        # the candidate positions.
        coverage_fraction = cmap.stats.coverage_fraction
        gradient_weight = cfg.coverage_gradient_weight * max(
            0.0, min(1.0, (coverage_fraction - 0.25) / 0.50)
        )
        if gradient_weight > 0.0:
            covered_mask = (grid > 0).astype(float)
            # uniform_filter gives the mean in a 3×3 window; multiply by 9 → neighbour count.
            grad_map = uniform_filter(covered_mask, size=3, mode="constant") * 9.0
            grad_arr = grad_map[candidates_i, candidates_j]
        else:
            grad_arr = np.zeros(len(candidates_i), dtype=float)

        # Normalise distance to [0, 1] before combining.
        max_dist = float(np.hypot(nx, ny))
        dist_norm = dist_arr / max(max_dist, 1.0)

        score = -cfg.distance_weight * dist_norm + gradient_weight * grad_arr
        best = int(np.argmax(score))
        return (int(candidates_i[best]), int(candidates_j[best]))

    def find_gap_cells(
        self, cmap, threshold: float = 0.70
    ) -> List[Tuple[int, int]]:
        """Return ONLY enclosed interior holes in the coverage map.

        Uses ``scipy.ndimage.binary_fill_holes`` to distinguish enclosed
        uncovered regions (real holes) from open frontier cells.  An open
        frontier cannot become enclosed until coverage is very high, so this
        returns an empty list at low coverage, letting the gap-fill gate pass.
        """
        grid = cmap.count_grid           # shape (nx, ny)
        covered = grid > 0               # bool mask

        # binary_fill_holes fills any background region that doesn't touch the
        # border — i.e. enclosed holes only.
        filled = binary_fill_holes(covered)
        holes = filled & ~covered        # enclosed cells that are uncovered

        ii, jj = np.where(holes)
        return [(int(ii[k]), int(jj[k])) for k in range(len(ii))]
