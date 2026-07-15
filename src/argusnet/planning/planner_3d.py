"""Terrain- and obstacle-aware altitude profiling for 2D routes.

The visibility-graph planner in :mod:`argusnet.planning.planner_base` routes in
the XY plane only; altitude has historically been supplied separately by the
behaviour profiles in the simulator. This module closes the "3D path planning"
gap by lifting a horizontal route into a feasible 3D polyline:

* it keeps a minimum clearance above terrain along the whole path (not only at
  waypoints), so a straight XY segment over rising ground still stays clear;
* it climbs over the top of any obstacle whose footprint the path crosses
  (vegetation canopy in particular, which the 2D planner does not route around);
* it limits the vertical gradient so the profile is kinematically flyable, which
  means the climb for an upcoming obstacle starts early rather than as a step.

The altitude reference frame matches the rest of ArgusNet: Z is metres above the
world datum, computed as terrain height plus an above-ground-level (AGL) margin.

The profiler is a small, injectable seam: it depends on a terrain height field
through the :class:`TerrainHeightField` protocol and on the existing
:class:`~argusnet.world.environment.ObstacleLayer`, so it can be exercised with
lightweight fakes and later wired behind the mission planner without touching
fusion or mission-loop code.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from argusnet.world.environment import Bounds2D, ObstacleLayer

from .planner_base import PathPlanner2D, PlannerRoute, _path_length

# Bump when the produced Route3D shape/semantics change in a non-additive way.
VERTICAL_ROUTE_CONTRACT_VERSION = "1.0.0"


@runtime_checkable
class TerrainHeightField(Protocol):
    """Minimal terrain sampling surface the profiler depends on.

    Both :class:`argusnet.world.terrain.TerrainLayer` and the environment
    ``TerrainLayer`` already satisfy this via ``height_at_many``.
    """

    def height_at_many(self, xy_m: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
        """Return terrain height (m above datum) for an ``(n, 2)`` array of XY points."""
        ...


@dataclass(frozen=True)
class Route3DConfig:
    """Tunables for turning a 2D route into a feasible altitude profile.

    All distances are metres. ``max_climb_gradient`` is a dimensionless slope
    (vertical over horizontal); ``0.5`` is roughly a 26 degree climb/descent.
    """

    min_terrain_clearance_m: float = 25.0
    cruise_agl_m: float = 60.0
    obstacle_clearance_m: float = 12.0
    max_climb_gradient: float = 0.5
    sample_spacing_m: float = 5.0
    decimate_tolerance_m: float = 0.5
    obstacle_types: frozenset[str] = frozenset({"building", "wall", "vegetation"})
    min_altitude_m: float | None = None
    max_altitude_m: float | None = None
    max_samples: int = 20000

    def __post_init__(self) -> None:
        if self.min_terrain_clearance_m < 0.0:
            raise ValueError("min_terrain_clearance_m must be non-negative.")
        if self.cruise_agl_m < self.min_terrain_clearance_m:
            raise ValueError("cruise_agl_m must be at least min_terrain_clearance_m.")
        if self.obstacle_clearance_m < 0.0:
            raise ValueError("obstacle_clearance_m must be non-negative.")
        if self.max_climb_gradient <= 0.0:
            raise ValueError("max_climb_gradient must be positive.")
        if self.sample_spacing_m <= 0.0:
            raise ValueError("sample_spacing_m must be positive.")
        if self.decimate_tolerance_m < 0.0:
            raise ValueError("decimate_tolerance_m must be non-negative.")
        if self.max_samples < 2:
            raise ValueError("max_samples must be at least 2.")
        if (
            self.min_altitude_m is not None
            and self.max_altitude_m is not None
            and self.max_altitude_m < self.min_altitude_m
        ):
            raise ValueError("max_altitude_m must be >= min_altitude_m.")


@dataclass(frozen=True)
class Route3D:
    """A vertically profiled route.

    ``points_xyz_m`` is an ``(n, 3)`` polyline that maintains the configured
    terrain and obstacle clearance along its full length and whose vertical
    gradient never exceeds ``Route3DConfig.max_climb_gradient``.
    ``ceiling_conflicts`` counts samples where the required safe altitude
    exceeded ``max_altitude_m``; when it is non-zero the ceiling was overridden
    in favour of clearance and the route is not ceiling-compliant.
    """

    points_xyz_m: np.ndarray
    length_m: float
    horizontal_length_m: float
    ascent_m: float
    descent_m: float
    max_altitude_m: float
    min_altitude_m: float
    max_gradient: float
    ceiling_conflicts: int
    contract_version: str = VERTICAL_ROUTE_CONTRACT_VERSION

    def __post_init__(self) -> None:
        points = np.asarray(self.points_xyz_m, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) < 2:
            raise ValueError("Route3D.points_xyz_m must have shape (n, 3) with n >= 2.")
        object.__setattr__(self, "points_xyz_m", points)

    @property
    def points_xy_m(self) -> np.ndarray:
        """The horizontal projection of the route, shape ``(n, 2)``."""
        return self.points_xyz_m[:, :2]

    @property
    def altitudes_m(self) -> np.ndarray:
        """Per-vertex altitude (m above datum), shape ``(n,)``."""
        return self.points_xyz_m[:, 2]


@dataclass(frozen=True)
class _DensePath:
    """Route resampled at a fixed spacing, with original waypoints flagged."""

    samples_xy_m: np.ndarray
    arc_length_m: np.ndarray
    is_waypoint: np.ndarray


class AltitudeProfiler:
    """Lifts horizontal routes into terrain/obstacle-aware 3D routes."""

    def __init__(
        self,
        *,
        terrain: TerrainHeightField,
        obstacle_layer: ObstacleLayer | None = None,
        config: Route3DConfig | None = None,
    ) -> None:
        self.terrain = terrain
        self.obstacle_layer = obstacle_layer
        self.config = config or Route3DConfig()

    def profile_route(
        self,
        route_xy: PlannerRoute | np.ndarray | Sequence[Sequence[float]],
        *,
        cruise_agl_m: float | None = None,
    ) -> Route3D | None:
        """Compute a feasible altitude profile for a horizontal route.

        Accepts a :class:`~argusnet.planning.planner_base.PlannerRoute` or a raw
        ``(n, 2)`` XY polyline. Returns ``None`` for a degenerate (zero-length)
        route.
        """
        points_xy = self._as_xy_array(route_xy)
        dense = self._densify(points_xy)
        if dense is None:
            return None
        samples_xy = dense.samples_xy_m

        terrain_z = np.asarray(self.terrain.height_at_many(samples_xy), dtype=float).reshape(-1)
        if terrain_z.shape[0] != samples_xy.shape[0]:
            raise ValueError("Terrain height field returned an unexpected number of samples.")

        cruise = self.config.cruise_agl_m if cruise_agl_m is None else float(cruise_agl_m)
        floor_z = terrain_z + self.config.min_terrain_clearance_m
        obstacle_z = self._obstacle_floor(samples_xy)
        required_z = np.maximum(floor_z, obstacle_z)

        desired_z = np.maximum(terrain_z + cruise, required_z)
        ceiling_conflicts = 0
        if self.config.max_altitude_m is not None:
            ceiling_conflicts = int(np.count_nonzero(required_z > self.config.max_altitude_m))
            desired_z = np.minimum(desired_z, self.config.max_altitude_m)
            desired_z = np.maximum(desired_z, required_z)
        if self.config.min_altitude_m is not None:
            desired_z = np.maximum(desired_z, self.config.min_altitude_m)

        profile_z = self._slope_limited_envelope(desired_z, dense.arc_length_m)

        samples_xyz = np.column_stack([samples_xy, profile_z])
        kept = self._decimate(samples_xyz, dense, required_z)
        return self._build_route(kept, ceiling_conflicts)

    def plan_route_3d(
        self,
        start_xy: Sequence[float],
        goal_xy: Sequence[float],
        *,
        planner_2d: PathPlanner2D,
        clearance_m: float,
        cruise_agl_m: float | None = None,
    ) -> Route3D | None:
        """Plan an obstacle-avoiding 2D route then profile it into 3D.

        Returns ``None`` if the 2D planner cannot find a horizontal route.
        """
        route = planner_2d.plan_route(start_xy, goal_xy, clearance_m=clearance_m)
        if route is None:
            return None
        return self.profile_route(route, cruise_agl_m=cruise_agl_m)

    # -- internals -----------------------------------------------------------

    @staticmethod
    def _as_xy_array(
        route_xy: PlannerRoute | np.ndarray | Sequence[Sequence[float]],
    ) -> np.ndarray:
        raw = route_xy.points_xy_m if isinstance(route_xy, PlannerRoute) else route_xy
        points: np.ndarray = np.asarray(raw, dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
            raise ValueError("route_xy must have shape (n, 2) with n >= 2.")
        return points

    def _densify(self, points_xy: np.ndarray) -> _DensePath | None:
        """Resample the polyline at ``sample_spacing_m``, flagging waypoints.

        Every original waypoint is retained and flagged so later decimation never
        collapses across a horizontal bend. Returns ``None`` when the route has
        no horizontal extent.
        """
        segment_deltas = points_xy[1:] - points_xy[:-1]
        segment_lengths = np.linalg.norm(segment_deltas, axis=1)
        total_length_m = float(segment_lengths.sum())
        if total_length_m <= 1.0e-6:
            return None

        # Clamp spacing so a very long route cannot explode the sample count.
        spacing_m = max(
            self.config.sample_spacing_m, total_length_m / float(self.config.max_samples - 1)
        )

        samples: list[np.ndarray] = [points_xy[0]]
        arc_lengths: list[float] = [0.0]
        is_waypoint: list[bool] = [True]
        cumulative_m = 0.0
        for start_xy, delta, length_m in zip(
            points_xy[:-1], segment_deltas, segment_lengths, strict=True
        ):
            if length_m <= 1.0e-9:
                continue
            steps = max(int(np.ceil(length_m / spacing_m)), 1)
            for step in range(1, steps + 1):
                fraction = step / steps
                samples.append(start_xy + delta * fraction)
                arc_lengths.append(cumulative_m + length_m * fraction)
                is_waypoint.append(step == steps)  # segment endpoint == original vertex
            cumulative_m += length_m
        return _DensePath(
            samples_xy_m=np.vstack(samples),
            arc_length_m=np.asarray(arc_lengths, dtype=float),
            is_waypoint=np.asarray(is_waypoint, dtype=bool),
        )

    def _obstacle_floor(self, samples_xy: np.ndarray) -> np.ndarray:
        """Return, per sample, the altitude needed to clear obstacles on the path.

        Each *segment* between consecutive samples is tested against obstacle
        footprints, not just the sample points, so a footprint crossed entirely
        between two samples (a thin wall or vegetation strip narrower than the
        sample spacing) is still detected. Both endpoints of a crossing segment
        are raised to the obstacle top plus the configured clearance; since the
        flown path between them is the straight line joining the endpoints, it
        clears the obstacle. Segments over open ground stay at ``-inf`` so they
        are governed purely by the terrain floor.
        """
        required: np.ndarray = np.full(samples_xy.shape[0], -np.inf, dtype=float)
        if self.obstacle_layer is None or samples_xy.shape[0] < 2:
            return required

        clearance_m = self.config.obstacle_clearance_m
        epsilon_m = 1.0e-6
        for index in range(samples_xy.shape[0] - 1):
            start_xy = samples_xy[index]
            end_xy = samples_xy[index + 1]
            segment_bounds = Bounds2D(
                x_min_m=min(float(start_xy[0]), float(end_xy[0])) - epsilon_m,
                x_max_m=max(float(start_xy[0]), float(end_xy[0])) + epsilon_m,
                y_min_m=min(float(start_xy[1]), float(end_xy[1])) - epsilon_m,
                y_max_m=max(float(start_xy[1]), float(end_xy[1])) + epsilon_m,
            )
            for primitive in self.obstacle_layer.query_obstacles(segment_bounds):
                if primitive.blocker_type not in self.config.obstacle_types:
                    continue
                # A horizontal probe at the vertical mid-point (always within
                # [base_z, top_z]) reduces the 3D hit test to a footprint crossing.
                mid_z_m = 0.5 * (primitive.base_z_m + primitive.top_z_m)
                origin = np.array([float(start_xy[0]), float(start_xy[1]), mid_z_m])
                target = np.array([float(end_xy[0]), float(end_xy[1]), mid_z_m])
                if primitive.first_hit_t(origin, target) is not None:
                    top_clearance_m = primitive.top_z_m + clearance_m
                    required[index] = max(required[index], top_clearance_m)
                    required[index + 1] = max(required[index + 1], top_clearance_m)
        return required

    def _slope_limited_envelope(self, desired_z: np.ndarray, arc_len_m: np.ndarray) -> np.ndarray:
        """Lowest gradient-limited altitude that stays at or above ``desired_z``.

        Two monotone (raise-only) passes yield the minimal ``g``-Lipschitz
        function >= ``desired_z``: the forward pass bounds the descent rate, the
        backward pass forces early climbs for upcoming peaks. Because both passes
        only raise altitude, every hard clearance already baked into
        ``desired_z`` is preserved.
        """
        gradient = self.config.max_climb_gradient
        profile: np.ndarray = desired_z.astype(float).copy()
        for index in range(1, len(profile)):
            step_m = arc_len_m[index] - arc_len_m[index - 1]
            profile[index] = max(profile[index], profile[index - 1] - gradient * step_m)
        for index in range(len(profile) - 2, -1, -1):
            step_m = arc_len_m[index + 1] - arc_len_m[index]
            profile[index] = max(profile[index], profile[index + 1] - gradient * step_m)
        return profile

    def _decimate(
        self, samples_xyz: np.ndarray, dense: _DensePath, required_z: np.ndarray
    ) -> np.ndarray:
        """Drop interior samples the straight chord can represent safely.

        Collapsing is confined to a single straight segment (never across an
        original waypoint), so the horizontal path is preserved exactly. Within a
        segment, a run is collapsed while the chord stays above ``required_z`` at
        every skipped sample (hard safety) and within ``decimate_tolerance_m`` of
        the computed profile (fidelity). The chord of a gradient-limited profile
        is itself gradient-limited, so feasibility is preserved.
        """
        count = len(samples_xyz)
        if count <= 2 or self.config.decimate_tolerance_m <= 0.0:
            return samples_xyz
        tolerance_m = self.config.decimate_tolerance_m
        arc_len_m = dense.arc_length_m
        is_waypoint = dense.is_waypoint
        kept_indices = [0]
        anchor = 0
        while anchor < count - 1:
            # Never collapse past the next original waypoint (a horizontal bend).
            segment_end = anchor + 1
            while segment_end < count - 1 and not is_waypoint[segment_end]:
                segment_end += 1
            candidate = segment_end
            while candidate > anchor + 1:
                if self._chord_is_safe(
                    samples_xyz, arc_len_m, required_z, anchor, candidate, tolerance_m
                ):
                    break
                candidate -= 1
            kept_indices.append(candidate)
            anchor = candidate
        return samples_xyz[kept_indices]

    @staticmethod
    def _chord_is_safe(
        samples_xyz: np.ndarray,
        arc_len_m: np.ndarray,
        required_z: np.ndarray,
        anchor: int,
        candidate: int,
        tolerance_m: float,
    ) -> bool:
        anchor_s = arc_len_m[anchor]
        span_m = arc_len_m[candidate] - anchor_s
        if span_m <= 1.0e-9:
            return True
        anchor_z = samples_xyz[anchor, 2]
        candidate_z = samples_xyz[candidate, 2]
        for index in range(anchor + 1, candidate):
            fraction = (arc_len_m[index] - anchor_s) / span_m
            chord_z = anchor_z + (candidate_z - anchor_z) * fraction
            if chord_z + 1.0e-6 < required_z[index]:
                return False
            if abs(chord_z - samples_xyz[index, 2]) > tolerance_m:
                return False
        return True

    def _build_route(self, points_xyz: np.ndarray, ceiling_conflicts: int) -> Route3D:
        altitudes = points_xyz[:, 2]
        delta_z = np.diff(altitudes)
        horizontal_length_m = _path_length(points_xyz[:, :2])
        segment_horizontal = np.linalg.norm(np.diff(points_xyz[:, :2], axis=0), axis=1)
        gradients = np.abs(delta_z) / np.maximum(segment_horizontal, 1.0e-9)
        return Route3D(
            points_xyz_m=points_xyz,
            length_m=float(np.linalg.norm(np.diff(points_xyz, axis=0), axis=1).sum()),
            horizontal_length_m=horizontal_length_m,
            ascent_m=float(delta_z[delta_z > 0.0].sum()),
            descent_m=float(-delta_z[delta_z < 0.0].sum()),
            max_altitude_m=float(altitudes.max()),
            min_altitude_m=float(altitudes.min()),
            max_gradient=float(gradients.max()) if len(gradients) else 0.0,
            ceiling_conflicts=ceiling_conflicts,
        )


__all__ = [
    "VERTICAL_ROUTE_CONTRACT_VERSION",
    "AltitudeProfiler",
    "Route3D",
    "Route3DConfig",
    "TerrainHeightField",
]
