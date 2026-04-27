from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from .environment import Bounds2D, LandCoverClass
from .obstacles import ForestStand, ObstaclePrimitive

if TYPE_CHECKING:
    from .environment import EnvironmentModel
    from .weather import WeatherModel


@dataclass(frozen=True)
class SensorVisibilityModel:
    hard_blockers: frozenset[str] = frozenset({"terrain", "building", "wall", "out_of_coverage"})
    attenuation_by_land_cover: Mapping[int, float] = field(
        default_factory=lambda: {
            int(LandCoverClass.OPEN): 1.0,
            int(LandCoverClass.URBAN): 0.92,
            int(LandCoverClass.FOREST): 0.82,
            int(LandCoverClass.WATER): 0.9,
            int(LandCoverClass.SCRUB): 0.95,
            int(LandCoverClass.WETLAND): 0.88,
            int(LandCoverClass.ROCKY): 0.98,
            int(LandCoverClass.SNOW): 0.97,
            int(LandCoverClass.ROAD): 1.0,
        }
    )
    noise_multiplier_by_land_cover: Mapping[int, float] = field(
        default_factory=lambda: {
            int(LandCoverClass.OPEN): 1.0,
            int(LandCoverClass.URBAN): 1.12,
            int(LandCoverClass.FOREST): 1.45,
            int(LandCoverClass.WATER): 1.18,
            int(LandCoverClass.SCRUB): 1.15,
            int(LandCoverClass.WETLAND): 1.25,
            int(LandCoverClass.ROCKY): 1.05,
            int(LandCoverClass.SNOW): 1.08,
            int(LandCoverClass.ROAD): 1.0,
        }
    )
    vegetation_hard_block_threshold: float = 0.12

    @classmethod
    def optical_default(cls) -> SensorVisibilityModel:
        return cls()


@dataclass(frozen=True)
class VisibilityResult:
    visible: bool
    blocker_type: str = "none"
    first_hit_range_m: float | None = None
    closest_point: np.ndarray | None = None
    transmittance: float = 1.0
    detection_multiplier: float = 1.0
    noise_multiplier: float = 1.0


@dataclass(frozen=True)
class DetectionResult:
    """Probabilistic detection result combining LOS, path loss, weather, and noise.

    Fields
    ------
    p_d : float
        Probability of detection in [0, 1].
    vis_result : VisibilityResult
        The underlying line-of-sight result.
    effective_noise_multiplier : float
        Combined noise multiplier (land cover + range + weather).
    dominant_loss_factor : str
        Human-readable label for the largest contributor to signal loss.
        One of ``"range"``, ``"transmittance"``, ``"weather"``, ``"blocked"``.
    """

    p_d: float
    vis_result: VisibilityResult
    effective_noise_multiplier: float
    dominant_loss_factor: str


# ---------------------------------------------------------------------------
# Helper functions for probabilistic detection
# ---------------------------------------------------------------------------


def free_space_path_loss(range_m: float, max_range_m: float = 2000.0) -> float:
    """Range-dependent free-space path loss factor in [0, 1].

    Uses an exponential decay model: ``exp(-ln(2) * (R / R_half)^2)`` where
    *R_half* = *max_range_m* / 2 is the range at which the factor drops to
    0.5.  This gives a value very close to 1.0 at short ranges and smoothly
    decays toward 0 at long ranges.

    Parameters
    ----------
    range_m : float
        Slant range to the target (m).
    max_range_m : float
        Notional maximum detection range.  The half-power distance is
        ``max_range_m / 2``.
    """
    if range_m <= 0.0:
        return 1.0
    r_half = max(max_range_m / 2.0, 1.0)
    return float(math.exp(-math.log(2.0) * (range_m / r_half) ** 2))


def compute_weather_factor(
    range_m: float,
    weather: WeatherModel | None,
) -> float:
    """Return the weather-induced visibility factor in [0, 1].

    If no weather model is provided the factor is 1.0 (clear conditions).
    """
    if weather is None:
        return 1.0
    return float(max(weather.visibility_at_range(range_m), 0.0))


def compute_effective_noise(
    range_m: float,
    land_cover_noise: float,
    weather: WeatherModel | None,
    max_range_m: float = 2000.0,
) -> float:
    """Compute the effective noise multiplier combining range, land cover, and weather.

    The multiplier is always >= 1.0.
    """
    # Range contribution: noise grows with distance
    range_factor = 1.0 + (range_m / max(max_range_m, 1.0)) * 1.5

    # Weather contribution
    weather_noise = 1.0
    if weather is not None:
        weather_noise = weather.bearing_noise_scale()

    return max(1.0, land_cover_noise * range_factor * weather_noise)


def identify_dominant_loss(
    path_loss: float,
    transmittance: float,
    weather_factor: float,
    is_blocked: bool,
) -> str:
    """Identify which factor contributes most to detection loss.

    Returns one of ``"blocked"``, ``"range"``, ``"transmittance"``, or
    ``"weather"``.
    """
    if is_blocked:
        return "blocked"
    # Lower value = more loss.  Find the factor furthest below 1.0.
    losses = {
        "range": 1.0 - path_loss,
        "transmittance": 1.0 - transmittance,
        "weather": 1.0 - weather_factor,
    }
    return max(losses, key=losses.get)  # type: ignore[arg-type]


def _liang_barsky_interval(
    origin_xy: np.ndarray, target_xy: np.ndarray, bounds: Bounds2D
) -> tuple[float, float] | None:
    dx = float(target_xy[0] - origin_xy[0])
    dy = float(target_xy[1] - origin_xy[1])
    p_values = [-dx, dx, -dy, dy]
    q_values = [
        float(origin_xy[0] - bounds.x_min_m),
        float(bounds.x_max_m - origin_xy[0]),
        float(origin_xy[1] - bounds.y_min_m),
        float(bounds.y_max_m - origin_xy[1]),
    ]
    u1, u2 = 0.0, 1.0
    for p_value, q_value in zip(p_values, q_values, strict=False):
        if abs(p_value) <= 1.0e-12:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            u1 = max(u1, ratio)
        else:
            u2 = min(u2, ratio)
        if u1 > u2:
            return None
    return (max(0.0, u1), min(1.0, u2))


def _grid_dda_intervals(
    *,
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
    grid_bounds: Bounds2D,
    cell_size_m: float,
) -> list[tuple[int, int, float, float]]:
    interval = _liang_barsky_interval(origin_xy, target_xy, grid_bounds)
    if interval is None:
        return []
    t_start, t_end = interval
    segment_xy = target_xy - origin_xy
    start_point = origin_xy + segment_xy * t_start
    end_point = origin_xy + segment_xy * t_end
    dx = float(end_point[0] - start_point[0])
    dy = float(end_point[1] - start_point[1])

    cell_x = int(
        np.clip(
            math.floor((start_point[0] - grid_bounds.x_min_m) / cell_size_m),
            0,
            max(0, int(math.ceil(grid_bounds.width_m / cell_size_m)) - 1),
        )
    )
    cell_y = int(
        np.clip(
            math.floor((start_point[1] - grid_bounds.y_min_m) / cell_size_m),
            0,
            max(0, int(math.ceil(grid_bounds.height_m / cell_size_m)) - 1),
        )
    )
    end_cell_x = int(
        np.clip(
            math.floor((end_point[0] - grid_bounds.x_min_m) / cell_size_m),
            0,
            max(0, int(math.ceil(grid_bounds.width_m / cell_size_m)) - 1),
        )
    )
    end_cell_y = int(
        np.clip(
            math.floor((end_point[1] - grid_bounds.y_min_m) / cell_size_m),
            0,
            max(0, int(math.ceil(grid_bounds.height_m / cell_size_m)) - 1),
        )
    )

    step_x = 0 if abs(dx) <= 1.0e-12 else (1 if dx > 0 else -1)
    step_y = 0 if abs(dy) <= 1.0e-12 else (1 if dy > 0 else -1)

    def next_boundary_x(index: int) -> float:
        return grid_bounds.x_min_m + ((index + (1 if step_x > 0 else 0)) * cell_size_m)

    def next_boundary_y(index: int) -> float:
        return grid_bounds.y_min_m + ((index + (1 if step_y > 0 else 0)) * cell_size_m)

    t_delta_x = float("inf") if step_x == 0 else abs(cell_size_m / (target_xy[0] - origin_xy[0]))
    t_delta_y = float("inf") if step_y == 0 else abs(cell_size_m / (target_xy[1] - origin_xy[1]))
    t_max_x = (
        float("inf")
        if step_x == 0
        else t_start
        + abs(
            (next_boundary_x(cell_x) - start_point[0]) / max(dx, 1.0e-12 if dx >= 0 else -1.0e-12)
        )
    )
    t_max_y = (
        float("inf")
        if step_y == 0
        else t_start
        + abs(
            (next_boundary_y(cell_y) - start_point[1]) / max(dy, 1.0e-12 if dy >= 0 else -1.0e-12)
        )
    )

    current_t = t_start
    intervals: list[tuple[int, int, float, float]] = []
    while current_t < t_end + 1.0e-9:
        next_t = min(t_end, t_max_x, t_max_y)
        intervals.append((cell_x, cell_y, current_t, next_t))
        if next_t >= t_end - 1.0e-9:
            break
        if cell_x == end_cell_x and cell_y == end_cell_y:
            break
        if t_max_x <= t_max_y:
            cell_x += step_x
            t_max_x += t_delta_x
        else:
            cell_y += step_y
            t_max_y += t_delta_y
        current_t = next_t
        if cell_x < 0 or cell_y < 0:
            break
    return intervals


class EnvironmentQuery:
    def __init__(self, environment: EnvironmentModel) -> None:
        self.environment = environment

    def height_at(self, x_m: float, y_m: float) -> float:
        return self.environment.terrain.height_at(x_m, y_m)

    def normal_at(self, x_m: float, y_m: float) -> np.ndarray:
        return self.environment.terrain.normal_at(x_m, y_m)

    def land_cover_at(self, x_m: float, y_m: float) -> LandCoverClass:
        return self.environment.land_cover.land_cover_at(x_m, y_m)

    def query_obstacles(self, segment_aabb: Bounds2D) -> tuple[ObstaclePrimitive, ...]:
        return self.environment.obstacles.query_obstacles(segment_aabb)

    def los(
        self,
        origin_xyz: np.ndarray,
        target_xyz: np.ndarray,
        sensor_profile: SensorVisibilityModel | None = None,
        *,
        terrain_clearance_m: float = 1.0,
    ) -> VisibilityResult:
        profile = sensor_profile or SensorVisibilityModel.optical_default()
        origin = np.asarray(origin_xyz, dtype=float).reshape(3)
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        distance_m = float(np.linalg.norm(target - origin))
        if distance_m <= 1.0e-9:
            return VisibilityResult(visible=True)

        if not self.environment.bounds_xy_m.contains_xy(
            float(origin[0]), float(origin[1])
        ) or not self.environment.bounds_xy_m.contains_xy(float(target[0]), float(target[1])):
            closest_point = self._coverage_closest_point(origin, target)
            return VisibilityResult(
                visible=False,
                blocker_type="out_of_coverage",
                first_hit_range_m=0.0,
                closest_point=closest_point,
                transmittance=0.0,
                detection_multiplier=0.0,
                noise_multiplier=max(profile.noise_multiplier_by_land_cover.values(), default=1.0),
            )

        terrain_hit = self._terrain_intersection(
            origin, target, terrain_clearance_m=terrain_clearance_m
        )
        if terrain_hit is not None:
            blocker_type, hit_t, closest_point = terrain_hit
            return VisibilityResult(
                visible=False,
                blocker_type=blocker_type,
                first_hit_range_m=None if hit_t is None else hit_t * distance_m,
                closest_point=closest_point,
                transmittance=0.0,
                detection_multiplier=0.0,
                noise_multiplier=2.0,
            )

        x_min_m = min(float(origin[0]), float(target[0]))
        x_max_m = max(float(origin[0]), float(target[0]))
        y_min_m = min(float(origin[1]), float(target[1]))
        y_max_m = max(float(origin[1]), float(target[1]))
        if x_max_m <= x_min_m:
            x_max_m = x_min_m + 1.0e-6
        if y_max_m <= y_min_m:
            y_max_m = y_min_m + 1.0e-6
        segment_bounds = Bounds2D(
            x_min_m=x_min_m,
            x_max_m=x_max_m,
            y_min_m=y_min_m,
            y_max_m=y_max_m,
        )
        candidates = self.query_obstacles(segment_bounds)
        nearest_hit_t: float | None = None
        nearest_blocker: str | None = None
        nearest_closest_point: np.ndarray | None = None
        nearest_vegetation_hit_t: float | None = None
        nearest_vegetation_point: np.ndarray | None = None
        transmittance = 1.0
        segment = target - origin
        for primitive in candidates:
            hit_t = primitive.first_hit_t(origin, target)
            if hit_t is None:
                continue
            hit_point = origin + segment * hit_t
            if primitive.blocker_type == "vegetation":
                if nearest_vegetation_hit_t is None or hit_t < nearest_vegetation_hit_t:
                    nearest_vegetation_hit_t = hit_t
                    nearest_vegetation_point = hit_point
                if isinstance(primitive, ForestStand):
                    path_length_m = primitive.path_length_inside(origin, target)
                    if path_length_m > 0.0:
                        transmittance *= math.exp(-primitive.density * (path_length_m / 45.0))
                continue
            if nearest_hit_t is None or hit_t < nearest_hit_t:
                nearest_hit_t = hit_t
                nearest_blocker = primitive.blocker_type
                nearest_closest_point = hit_point

        if nearest_hit_t is not None and nearest_blocker in profile.hard_blockers:
            return VisibilityResult(
                visible=False,
                blocker_type=nearest_blocker or "building",
                first_hit_range_m=nearest_hit_t * distance_m,
                closest_point=nearest_closest_point,
                transmittance=0.0,
                detection_multiplier=0.0,
                noise_multiplier=2.0,
            )

        midpoint = (origin + target) * 0.5
        land_cover = self.land_cover_at(float(midpoint[0]), float(midpoint[1]))
        land_cover_attenuation = float(profile.attenuation_by_land_cover.get(int(land_cover), 1.0))
        land_cover_noise = float(profile.noise_multiplier_by_land_cover.get(int(land_cover), 1.0))
        detection_multiplier = transmittance * land_cover_attenuation
        noise_multiplier = max(1.0, (1.0 / max(transmittance, 0.05)) * land_cover_noise)

        if transmittance < profile.vegetation_hard_block_threshold:
            return VisibilityResult(
                visible=False,
                blocker_type="vegetation",
                first_hit_range_m=None
                if nearest_vegetation_hit_t is None
                else nearest_vegetation_hit_t * distance_m,
                closest_point=nearest_vegetation_point,
                transmittance=transmittance,
                detection_multiplier=0.0,
                noise_multiplier=noise_multiplier,
            )

        return VisibilityResult(
            visible=True,
            blocker_type="none",
            first_hit_range_m=None
            if nearest_vegetation_hit_t is None
            else nearest_vegetation_hit_t * distance_m,
            closest_point=nearest_vegetation_point,
            transmittance=transmittance,
            detection_multiplier=detection_multiplier,
            noise_multiplier=noise_multiplier,
        )

    def compute_detection_probability(
        self,
        origin_xyz: np.ndarray,
        target_xyz: np.ndarray,
        sensor_profile: SensorVisibilityModel | None = None,
        weather: WeatherModel | None = None,
        *,
        max_range_m: float = 2000.0,
        terrain_clearance_m: float = 1.0,
    ) -> DetectionResult:
        """Compute the probabilistic detection result for a sensor-target pair.

        Combines line-of-sight, free-space path loss, foliage/obstacle
        transmittance, weather effects, and land-cover noise into a single
        probability of detection (P_d).

        The returned :class:`DetectionResult` can be used by
        ``build_observations()`` in :mod:`sim` to stochastically accept or
        reject observations.

        Parameters
        ----------
        origin_xyz : np.ndarray
            Sensor position ``[x, y, z]`` (m).
        target_xyz : np.ndarray
            Target position ``[x, y, z]`` (m).
        sensor_profile : SensorVisibilityModel, optional
            Visibility profile.  Defaults to the optical preset.
        weather : WeatherModel, optional
            Weather model for atmospheric / precipitation effects.
        max_range_m : float
            Notional maximum detection range for path-loss normalisation.
        terrain_clearance_m : float
            Vertical clearance for terrain intersection checks.
        """
        origin = np.asarray(origin_xyz, dtype=float).reshape(3)
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        range_m = float(np.linalg.norm(target - origin))

        # 1. LOS check
        vis = self.los(
            origin,
            target,
            sensor_profile=sensor_profile,
            terrain_clearance_m=terrain_clearance_m,
        )

        # 2. If hard-blocked, P_d is essentially zero
        if not vis.visible and vis.transmittance <= 0.0:
            noise = compute_effective_noise(range_m, vis.noise_multiplier, weather, max_range_m)
            return DetectionResult(
                p_d=0.0,
                vis_result=vis,
                effective_noise_multiplier=noise,
                dominant_loss_factor="blocked",
            )

        # 3. Individual loss factors
        path_loss = free_space_path_loss(range_m, max_range_m)
        transmittance = vis.transmittance
        weather_factor = compute_weather_factor(range_m, weather)

        # 4. Combined P_d = product of independent survival probabilities,
        #    scaled by the detection multiplier from the LOS result.
        raw_pd = path_loss * transmittance * weather_factor * vis.detection_multiplier
        p_d = float(max(0.0, min(1.0, raw_pd)))

        # 5. Effective noise
        noise = compute_effective_noise(range_m, vis.noise_multiplier, weather, max_range_m)

        # 6. Dominant loss
        dominant = identify_dominant_loss(path_loss, transmittance, weather_factor, not vis.visible)

        return DetectionResult(
            p_d=p_d,
            vis_result=vis,
            effective_noise_multiplier=noise,
            dominant_loss_factor=dominant,
        )

    def _terrain_intersection(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        *,
        terrain_clearance_m: float,
    ) -> tuple[str, float | None, np.ndarray | None] | None:
        tile_span_m = (
            self.environment.terrain.tile_size_cells * self.environment.terrain.base_resolution_m
        )
        tile_intervals = _grid_dda_intervals(
            origin_xy=origin[:2],
            target_xy=target[:2],
            grid_bounds=self.environment.bounds_xy_m,
            cell_size_m=tile_span_m,
        )
        if not tile_intervals:
            return ("out_of_coverage", None, self._coverage_closest_point(origin, target))

        segment = target - origin
        for tile_x, tile_y, t0, t1 in tile_intervals:
            tile = self.environment.terrain._tiles.get((0, tile_x, tile_y))
            if tile is None:
                return ("out_of_coverage", None, self._coverage_closest_point(origin, target))
            z_min = min(float(origin[2] + (segment[2] * t0)), float(origin[2] + (segment[2] * t1)))
            if tile.max_height_m + terrain_clearance_m < z_min:
                continue

            for _, _, cell_t0, cell_t1 in _grid_dda_intervals(
                origin_xy=origin[:2] + (target[:2] - origin[:2]) * t0,
                target_xy=origin[:2] + (target[:2] - origin[:2]) * t1,
                grid_bounds=tile.bounds_xy_m,
                cell_size_m=tile.cell_size_m,
            ):
                mid_t = t0 + ((cell_t0 + cell_t1) * 0.5 * (t1 - t0))
                point = origin + segment * mid_t
                terrain_height = (
                    self.environment.terrain.height_at(float(point[0]), float(point[1]))
                    + terrain_clearance_m
                )
                if float(point[2]) <= terrain_height:
                    return (
                        "terrain",
                        mid_t,
                        np.array(
                            [float(point[0]), float(point[1]), float(terrain_height)], dtype=float
                        ),
                    )
        return None

    def _coverage_closest_point(self, origin: np.ndarray, target: np.ndarray) -> np.ndarray:
        interval = _liang_barsky_interval(origin[:2], target[:2], self.environment.bounds_xy_m)
        clamped_x = float(
            np.clip(
                float(target[0]),
                self.environment.bounds_xy_m.x_min_m,
                self.environment.bounds_xy_m.x_max_m,
            )
        )
        clamped_y = float(
            np.clip(
                float(target[1]),
                self.environment.bounds_xy_m.y_min_m,
                self.environment.bounds_xy_m.y_max_m,
            )
        )
        if interval is None:
            return np.array([clamped_x, clamped_y, float(target[2])], dtype=float)
        t_start, t_end = interval
        hit_t = (
            t_start
            if not self.environment.bounds_xy_m.contains_xy(float(origin[0]), float(origin[1]))
            else t_end
        )
        point = origin + (target - origin) * hit_t
        point[0] = float(
            np.clip(
                float(point[0]),
                self.environment.bounds_xy_m.x_min_m,
                self.environment.bounds_xy_m.x_max_m,
            )
        )
        point[1] = float(
            np.clip(
                float(point[1]),
                self.environment.bounds_xy_m.y_min_m,
                self.environment.bounds_xy_m.y_max_m,
            )
        )
        return point


__all__ = [
    "DetectionResult",
    "EnvironmentQuery",
    "SensorVisibilityModel",
    "VisibilityResult",
    "compute_effective_noise",
    "compute_weather_factor",
    "free_space_path_loss",
    "identify_dominant_loss",
    "_grid_dda_intervals",
    "_liang_barsky_interval",
]
