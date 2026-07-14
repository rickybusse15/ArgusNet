"""Grid-based drone self-localization.

Uses the accumulated CoverageMap as a reference and estimates the drone's
pose (x, y) by comparing the expected sensor footprint against coverage.
As coverage fills in, position uncertainty decreases and confidence rises.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from argusnet.core.types import LocalizationEstimate, LocalizationStatus, PoseEstimate, Vector3

# Grid localization estimates only (x, y); altitude is held by the trajectory
# controller and not estimated here, so the pose covariance carries a fixed
# nominal 1-sigma altitude-hold term rather than a measured one.
Z_HOLD_STD_M = 1.0


@dataclass(frozen=True)
class LocalizationConfig:
    """Parameters controlling grid localizer behaviour."""

    search_radius_m: float = 80.0  # uncertainty radius for particle spread
    particle_count: int = 200  # number of particles
    convergence_threshold_m: float = 15.0  # std below which we call it converged
    min_coverage_to_localize: float = 0.15  # need at least 15% map coverage
    confidence_decay: float = 0.50  # carry-over confidence between steps
    annealing_rate: float = 0.02  # per-step fractional radius shrink
    localization_timeout_steps: int = 200  # steps before forced convergence; 0 = disabled
    localized_confidence: float = 0.6  # confidence at/above which status is LOCALIZED
    lost_confidence: float = 0.1  # confidence below which a prior fix is considered LOST


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

    def __init__(
        self,
        config: LocalizationConfig | None = None,
        *,
        source_id: str = "grid",
        version: str = "1.0",
    ) -> None:
        self.config = config or LocalizationConfig()
        self._estimates: dict[str, LocalizationEstimate] = {}
        self._step_counts: dict[str, int] = {}
        self._timed_out: set[str] = set()
        # Per-drone pose contract state (see PoseEstimate): flattened row-major
        # 3x3 position covariance (m^2) and the current LocalizationStatus value.
        self._covariances: dict[str, tuple[float, ...]] = {}
        self._statuses: dict[str, str] = {}
        self._rng = np.random.default_rng(42)
        # Provenance for the LocalizationQuery contract.
        self.source_id = source_id
        self.version = version

    def update(
        self,
        drone_id: str,
        nominal_position: Vector3,
        heading_rad: float,
        coverage_map,  # argusnet.mapping.coverage.CoverageMap
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
            self._covariances[drone_id] = self._isotropic_cov(cfg.search_radius_m)
            self._statuses[drone_id] = LocalizationStatus.UNLOCALIZED.value
            self._estimates[drone_id] = est
            return est

        # Scatter particles around nominal position with annealing.
        annealing_factor = max(0.05, (1.0 - cfg.annealing_rate) ** step)
        scale = cfg.search_radius_m * max(0.1, 1.0 - coverage_fraction) * annealing_factor
        offsets = self._rng.normal(0.0, scale, size=(cfg.particle_count, 2))
        positions = np.array(nominal_position[:2], dtype=float) + offsets

        # Score each particle: how well does a footprint at that position match
        # the existing coverage map?  Higher coverage around the particle = better.
        scores = np.array(
            [self._score_position(pos, footprint_radius_m, coverage_map) for pos in positions],
            dtype=float,
        )

        # Weighted mean position.
        weights = np.exp(scores - scores.max())
        weights /= weights.sum() + 1e-12
        est_xy = (positions * weights[:, None]).sum(axis=0)
        est_position = np.array([est_xy[0], est_xy[1], float(nominal_position[2])], dtype=float)

        # Position std = weighted std of particle cloud; the full weighted 2x2
        # covariance is the pose contract's uncertainty (isotropic std is its
        # scalar summary, kept for the legacy LocalizationEstimate).
        diff = positions - est_xy
        var = (weights[:, None] * diff**2).sum(axis=0)
        position_std = float(np.sqrt(var.mean()))
        cov_xy = np.einsum("k,ki,kj->ij", weights, diff, diff)

        # Confidence: rises as coverage fills in and std shrinks.
        prev_conf = self._estimates.get(drone_id, None)
        base_conf = min(1.0, coverage_fraction / 0.8) * min(
            1.0, cfg.convergence_threshold_m / max(position_std, 1.0)
        )
        if prev_conf is not None:
            confidence = (
                cfg.confidence_decay * prev_conf.confidence + (1 - cfg.confidence_decay) * base_conf
            )
        else:
            confidence = base_conf

        # Timeout path: force confidence = 1.0 after enough steps.
        if cfg.localization_timeout_steps > 0 and step >= cfg.localization_timeout_steps:
            self._timed_out.add(drone_id)
            confidence = 1.0

        confidence = float(np.clip(confidence, 0.0, 1.0))
        status = self._classify_status(
            self._statuses.get(drone_id, LocalizationStatus.UNLOCALIZED.value),
            confidence=confidence,
            position_std=position_std,
            coverage_fraction=coverage_fraction,
            timed_out=drone_id in self._timed_out,
        )
        self._statuses[drone_id] = status
        self._covariances[drone_id] = self._cov_3x3(cov_xy)

        est = LocalizationEstimate(
            drone_id=drone_id,
            timestamp_s=timestamp_s,
            position_estimate=est_position,
            heading_rad=heading_rad,
            position_std_m=float(position_std),
            confidence=confidence,
        )
        self._estimates[drone_id] = est
        return est

    @property
    def any_timed_out(self) -> bool:
        """True if any drone reached the timeout step limit."""
        return bool(self._timed_out)

    def fuse_estimates(self, drone_ids: list[str]) -> LocalizationEstimate | None:
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
        vars_ = stds**2
        weights = 1.0 / np.maximum(vars_, 1e-6)
        weights /= weights.sum()

        positions = np.array([c.position_estimate for c in candidates], dtype=float)
        fused_pos = (positions * weights[:, None]).sum(axis=0)
        fused_std = float(np.sqrt(1.0 / (1.0 / np.maximum(vars_, 1e-6)).sum()))
        fused_conf = float(
            np.clip((weights * np.array([c.confidence for c in candidates])).sum(), 0.0, 1.0)
        )
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
        grid = coverage_map.count_grid  # property: returns a copy with shape (nx, ny)
        patch = grid[i0:i1, j0:j1]
        return float(np.log1p(patch.mean()))

    def reset(self, drone_id: str | None = None) -> None:
        if drone_id is None:
            self._estimates.clear()
            self._step_counts.clear()
            self._timed_out.clear()
            self._covariances.clear()
            self._statuses.clear()
        else:
            self._estimates.pop(drone_id, None)
            self._step_counts.pop(drone_id, None)
            self._timed_out.discard(drone_id)
            self._covariances.pop(drone_id, None)
            self._statuses.pop(drone_id, None)

    def all_estimates(self) -> list[LocalizationEstimate]:
        return list(self._estimates.values())

    # ------------------------------------------------------------------
    # Pose / covariance / status contract (LocalizationQuery)
    # ------------------------------------------------------------------

    def _classify_status(
        self,
        prev: str,
        *,
        confidence: float,
        position_std: float,
        coverage_fraction: float,
        timed_out: bool,
    ) -> str:
        """Map estimate quality onto the LocalizationStatus model (LOCALIZATION.md §8)."""
        cfg = self.config
        if timed_out:
            return LocalizationStatus.LOCALIZED.value
        if coverage_fraction < cfg.min_coverage_to_localize:
            return LocalizationStatus.UNLOCALIZED.value
        if confidence >= cfg.localized_confidence and position_std <= cfg.convergence_threshold_m:
            return LocalizationStatus.LOCALIZED.value
        was_localized = prev in (
            LocalizationStatus.LOCALIZED.value,
            LocalizationStatus.DEGRADED.value,
        )
        if was_localized:
            # A previously trusted fix that fell below threshold is degrading; if it
            # collapses entirely it is lost.
            if confidence < cfg.lost_confidence or position_std > cfg.search_radius_m:
                return LocalizationStatus.LOST.value
            return LocalizationStatus.DEGRADED.value
        if confidence < cfg.lost_confidence:
            return LocalizationStatus.UNLOCALIZED.value
        return LocalizationStatus.INITIALIZING.value

    def _cov_3x3(self, cov_xy: np.ndarray) -> tuple[float, ...]:
        """Flatten a 2x2 XY covariance into a row-major 3x3 position covariance."""
        m: np.ndarray = np.zeros((3, 3), dtype=float)
        m[:2, :2] = cov_xy
        m[2, 2] = Z_HOLD_STD_M**2
        return tuple(float(v) for v in m.flatten())

    def _isotropic_cov(self, std_m: float) -> tuple[float, ...]:
        return self._cov_3x3(np.diag([std_m**2, std_m**2]))

    def _build_pose(self, drone_id: str) -> PoseEstimate | None:
        est = self._estimates.get(drone_id)
        if est is None:
            return None
        status = self._statuses.get(drone_id, LocalizationStatus.UNLOCALIZED.value)
        failure_reason = None
        if status == LocalizationStatus.LOST.value:
            failure_reason = "position uncertainty diverged beyond the search radius"
        elif status == LocalizationStatus.DEGRADED.value:
            failure_reason = "confidence fell below the localized threshold"
        return PoseEstimate(
            platform_id=drone_id,
            timestamp_s=est.timestamp_s,
            position_m=np.array(est.position_estimate, dtype=float),
            orientation_rpy_rad=(0.0, 0.0, float(est.heading_rad)),
            frame_id="map",
            covariance=self._covariances.get(drone_id, ()),
            confidence=float(est.confidence),
            status=status,
            relocalization_score=float(est.confidence),
            failure_reason=failure_reason,
        )

    def current_pose(self, platform_id: str) -> PoseEstimate | None:
        """Map-relative pose estimate with covariance and status, or None."""
        return self._build_pose(platform_id)

    def current_covariance(self, platform_id: str) -> tuple[float, ...]:
        """Flattened row-major 3x3 position covariance (m^2)."""
        return self._covariances.get(platform_id, ())

    def localization_status(self, platform_id: str) -> str:
        return self._statuses.get(platform_id, LocalizationStatus.UNLOCALIZED.value)

    def confidence(self, platform_id: str) -> float:
        est = self._estimates.get(platform_id)
        return float(est.confidence) if est is not None else 0.0

    def is_localized(self, platform_id: str, threshold: float | None = None) -> bool:
        """True when the platform has a trustworthy fix.

        With no threshold, uses the status model (LOCALIZED); with a threshold,
        compares the scalar confidence.
        """
        if threshold is None:
            return self.localization_status(platform_id) == LocalizationStatus.LOCALIZED.value
        return self.confidence(platform_id) >= threshold

    def pose_estimates(self) -> tuple[PoseEstimate, ...]:
        """All current per-platform poses, ordered by platform id for determinism."""
        poses = [self._build_pose(did) for did in sorted(self._estimates)]
        return tuple(p for p in poses if p is not None)
