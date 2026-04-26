"""Grid-based drone self-localization.

Uses the accumulated CoverageMap as a reference and estimates the drone's
pose (x, y) by comparing the expected sensor footprint against coverage.
As coverage fills in, position uncertainty decreases and confidence rises.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from argusnet.core.types import LocalizationEstimate, Vector3


@dataclass(frozen=True)
class LocalizationConfig:
    """Parameters controlling grid localizer behaviour."""
    search_radius_m: float = 80.0      # uncertainty radius for particle spread
    particle_count: int = 200           # number of particles
    convergence_threshold_m: float = 15.0  # std below which we call it converged
    min_coverage_to_localize: float = 0.15  # need at least 15% map coverage
    confidence_decay: float = 0.50     # carry-over confidence between steps
    annealing_rate: float = 0.02       # per-step fractional radius shrink
    localization_timeout_steps: int = 200  # steps before forced convergence; 0 = disabled


class GridLocalizer:
    """Lightweight particle-filter-inspired pose estimator.

    Given:
      - A CoverageMap (the map built so far)
      - The drone's nominal trajectory position (from LaunchableTrajectoryController)
      - The drone's sensor footprint radius

    Produces a LocalizationEstimate with position_estimate and confidence.

    The key insight: as the map fills in, matching a sensor footprint to the
    map becomes more discriminative, so confidence increases naturally.
    """

    def __init__(self, config: Optional[LocalizationConfig] = None) -> None:
        self.config = config or LocalizationConfig()
        self._estimates: Dict[str, LocalizationEstimate] = {}
        self._step_counts: Dict[str, int] = {}
        self._timed_out: Set[str] = set()
        self._rng = np.random.default_rng(42)

    def update(
        self,
        drone_id: str,
        nominal_position: Vector3,
        heading_rad: float,
        coverage_map,          # argusnet.mapping.coverage.CoverageMap
        footprint_radius_m: float,
        timestamp_s: float,
    ) -> LocalizationEstimate:
        """Update localization estimate for one drone.

        Args:
            drone_id: Unique identifier.
            nominal_position: Position from trajectory (x, y, z metres).
            heading_rad: Current heading in radians.
            coverage_map: The CoverageMap accumulated during scanning.
            footprint_radius_m: Sensor footprint radius on the ground (metres).
            timestamp_s: Current simulation time.

        Returns:
            Updated LocalizationEstimate.
        """
        cfg = self.config
        stats = coverage_map.stats
        coverage_fraction = stats.coverage_fraction

        # Track per-drone step count (skip early-return).
        self._step_counts[drone_id] = self._step_counts.get(drone_id, 0) + 1
        step = self._step_counts[drone_id]

        # Not enough map to localize against yet.
        if coverage_fraction < cfg.min_coverage_to_localize:
            est = LocalizationEstimate(
                drone_id=drone_id,
                timestamp_s=timestamp_s,
                position_estimate=np.array(nominal_position, dtype=float),
                heading_rad=heading_rad,
                position_std_m=cfg.search_radius_m,
                confidence=0.0,
            )
            self._estimates[drone_id] = est
            return est

        # Scatter particles around nominal position with annealing.
        annealing_factor = max(0.05, (1.0 - cfg.annealing_rate) ** step)
        scale = cfg.search_radius_m * max(0.1, 1.0 - coverage_fraction) * annealing_factor
        offsets = self._rng.normal(0.0, scale, size=(cfg.particle_count, 2))
        positions = np.array(nominal_position[:2], dtype=float) + offsets

        # Score each particle: how well does a footprint at that position match
        # the existing coverage map?  Higher coverage around the particle = better.
        scores = np.array([
            self._score_position(pos, footprint_radius_m, coverage_map)
            for pos in positions
        ], dtype=float)

        # Weighted mean position.
        weights = np.exp(scores - scores.max())
        weights /= weights.sum() + 1e-12
        est_xy = (positions * weights[:, None]).sum(axis=0)
        est_position = np.array([est_xy[0], est_xy[1], float(nominal_position[2])],
                                dtype=float)

        # Position std = weighted std of particle cloud.
        diff = positions - est_xy
        var = (weights[:, None] * diff ** 2).sum(axis=0)
        position_std = float(np.sqrt(var.mean()))

        # Confidence: rises as coverage fills in and std shrinks.
        prev_conf = self._estimates.get(drone_id, None)
        base_conf = min(1.0, coverage_fraction / 0.8) * min(
            1.0, cfg.convergence_threshold_m / max(position_std, 1.0)
        )
        if prev_conf is not None:
            confidence = cfg.confidence_decay * prev_conf.confidence + (
                1 - cfg.confidence_decay
            ) * base_conf
        else:
            confidence = base_conf

        # Timeout path: force confidence = 1.0 after enough steps.
        if cfg.localization_timeout_steps > 0 and step >= cfg.localization_timeout_steps:
            self._timed_out.add(drone_id)
            confidence = 1.0

        est = LocalizationEstimate(
            drone_id=drone_id,
            timestamp_s=timestamp_s,
            position_estimate=est_position,
            heading_rad=heading_rad,
            position_std_m=float(position_std),
            confidence=float(np.clip(confidence, 0.0, 1.0)),
        )
        self._estimates[drone_id] = est
        return est

    @property
    def any_timed_out(self) -> bool:
        """True if any drone reached the timeout step limit."""
        return bool(self._timed_out)

    def fuse_estimates(self, drone_ids: List[str]) -> Optional[LocalizationEstimate]:
        """Inverse-variance weighted fusion of converged drone estimates.

        Only considers estimates whose ``position_std_m`` is below half the
        search radius (i.e. reasonably converged).  Returns None if no
        suitable estimates exist.
        """
        half_radius = self.config.search_radius_m * 0.5
        candidates = [
            self._estimates[did]
            for did in drone_ids
            if did in self._estimates and self._estimates[did].position_std_m < half_radius
        ]
        if not candidates:
            return None

        stds = np.array([c.position_std_m for c in candidates], dtype=float)
        vars_ = stds ** 2
        weights = 1.0 / np.maximum(vars_, 1e-6)
        weights /= weights.sum()

        positions = np.array([c.position_estimate for c in candidates], dtype=float)
        fused_pos = (positions * weights[:, None]).sum(axis=0)
        fused_std = float(np.sqrt(1.0 / (1.0 / np.maximum(vars_, 1e-6)).sum()))
        fused_conf = float(np.clip((weights * np.array([c.confidence for c in candidates])).sum(), 0.0, 1.0))
        ref = candidates[0]
        return LocalizationEstimate(
            drone_id="fused",
            timestamp_s=ref.timestamp_s,
            position_estimate=fused_pos,
            heading_rad=ref.heading_rad,
            position_std_m=fused_std,
            confidence=fused_conf,
        )

    def _score_position(
        self,
        xy: np.ndarray,
        radius_m: float,
        coverage_map,
    ) -> float:
        """Score how well a footprint at xy matches the coverage map.

        Uses the public ``count_grid()`` API and the canonical ``xy_to_ij``
        converter so axis ordering matches CoverageMap's (nx, ny) internal layout.
        """
        bounds = coverage_map.bounds
        gi, gj = bounds.xy_to_ij(float(xy[0]), float(xy[1]))
        r_cells = max(1, int(radius_m / bounds.resolution_m))
        nx, ny = bounds.nx, bounds.ny
        i0, i1 = max(0, gi - r_cells), min(nx - 1, gi + r_cells)
        j0, j1 = max(0, gj - r_cells), min(ny - 1, gj + r_cells)
        if i0 >= i1 or j0 >= j1:
            return -1.0
        grid = coverage_map.count_grid      # property: returns a copy with shape (nx, ny)
        patch = grid[i0:i1, j0:j1]
        return float(np.log1p(patch.mean()))

    def reset(self, drone_id: Optional[str] = None) -> None:
        if drone_id is None:
            self._estimates.clear()
            self._step_counts.clear()
            self._timed_out.clear()
        else:
            self._estimates.pop(drone_id, None)
            self._step_counts.pop(drone_id, None)
            self._timed_out.discard(drone_id)

    def all_estimates(self) -> List[LocalizationEstimate]:
        return list(self._estimates.values())
