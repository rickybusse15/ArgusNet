from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from argusnet._native_world import call

from ._cache import BoundedLRU, quantized_key
from .environment import Bounds2D, LandCoverClass
from .obstacles import ObstaclePrimitive

if TYPE_CHECKING:
    from .environment import EnvironmentModel
    from .weather import WeatherModel


_DEFAULT_PROFILE = call("visibility_profile")


@dataclass(frozen=True)
class SensorVisibilityModel:
    hard_blockers: frozenset[str] = frozenset(_DEFAULT_PROFILE["hard_blockers"])
    attenuation_by_land_cover: Mapping[int, float] = field(
        default_factory=lambda: {
            int(k): v for k, v in _DEFAULT_PROFILE["attenuation_by_land_cover"].items()
        }
    )
    noise_multiplier_by_land_cover: Mapping[int, float] = field(
        default_factory=lambda: {
            int(k): v for k, v in _DEFAULT_PROFILE["noise_multiplier_by_land_cover"].items()
        }
    )
    vegetation_hard_block_threshold: float = _DEFAULT_PROFILE["vegetation_hard_block_threshold"]

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
    return float(call("path_loss", range=range_m, max_range=max_range_m))


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
    return float(
        call(
            "noise",
            range=range_m,
            land_noise=land_cover_noise,
            weather_noise=1.0 if weather is None else weather.bearing_noise_scale(),
            max_range=max_range_m,
        )
    )


def identify_dominant_loss(
    path_loss: float,
    transmittance: float,
    weather_factor: float,
    is_blocked: bool,
) -> str:
    return call(
        "dominant",
        path=path_loss,
        transmittance=transmittance,
        weather=weather_factor,
        blocked=is_blocked,
    )


def _liang_barsky_interval(
    origin_xy: np.ndarray, target_xy: np.ndarray, bounds: Bounds2D
) -> tuple[float, float] | None:
    result = call("clip", origin=origin_xy, target=target_xy, bounds=bounds.to_metadata())
    return None if result is None else tuple(result)


def _grid_dda_intervals(
    *,
    origin_xy: np.ndarray,
    target_xy: np.ndarray,
    grid_bounds: Bounds2D,
    cell_size_m: float,
) -> list[tuple[int, int, float, float]]:
    return [
        tuple(v)
        for v in call(
            "dda",
            origin=origin_xy,
            target=target_xy,
            bounds=grid_bounds.to_metadata(),
            cell=cell_size_m,
        )
    ]


class EnvironmentQuery:
    def __init__(self, environment: EnvironmentModel) -> None:
        self.environment = environment
        self._los_cache: BoundedLRU[tuple[object, ...], VisibilityResult] = BoundedLRU(
            owner=f"EnvironmentQuery:{environment.environment_id}",
            name="los",
            capacity=131072,
        )

    def height_at(self, x_m: float, y_m: float) -> float:
        return self.environment.terrain.height_at(x_m, y_m)

    def normal_at(self, x_m: float, y_m: float) -> np.ndarray:
        return self.environment.terrain.normal_at(x_m, y_m)

    def land_cover_at(self, x_m: float, y_m: float) -> LandCoverClass:
        return self.environment.land_cover.land_cover_at(x_m, y_m)

    def query_obstacles(self, segment_aabb: Bounds2D) -> tuple[ObstaclePrimitive, ...]:
        return self.environment.obstacles.query_obstacles(segment_aabb)

    def clear_caches(self) -> None:
        self._los_cache.clear()

    def cache_metrics(self) -> dict[str, dict]:
        return {"los": self._los_cache.metrics_dict()}

    @staticmethod
    def _profile_cache_key(profile: SensorVisibilityModel) -> tuple[object, ...]:
        return (
            tuple(sorted(profile.hard_blockers)),
            tuple(sorted((int(k), float(v)) for k, v in profile.attenuation_by_land_cover.items())),
            tuple(
                sorted(
                    (int(k), float(v)) for k, v in profile.noise_multiplier_by_land_cover.items()
                )
            ),
            float(profile.vegetation_hard_block_threshold),
        )

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
        q_values = quantized_key(*origin, *target, float(terrain_clearance_m), cell_m=1.0e-5)
        cache_key = (
            int(getattr(self.environment, "scene_version", 0)),
            *q_values,
            self._profile_cache_key(profile),
        )
        cached = self._los_cache.get(cache_key)
        if cached is not None:
            return cached
        midpoint = (origin + target) * 0.5
        result = call(
            "los",
            terrain=self.environment.terrain._native.id,
            obstacles=self.environment.obstacles._native.id,
            origin=origin,
            target=target,
            profile=asdict(profile),
            land_cover=int(self.land_cover_at(float(midpoint[0]), float(midpoint[1]))),
            clearance=terrain_clearance_m,
        )
        if result["closest_point"] is not None:
            result["closest_point"] = np.asarray(result["closest_point"], dtype=float)
        return self._los_cache.put(cache_key, VisibilityResult(**result))

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
        origin = np.asarray(origin_xyz, dtype=float).reshape(3)
        target = np.asarray(target_xyz, dtype=float).reshape(3)
        range_m = float(np.linalg.norm(target - origin))
        vis = self.los(origin, target, sensor_profile, terrain_clearance_m=terrain_clearance_m)
        result = call(
            "detection",
            visibility=asdict(vis),
            range=range_m,
            max_range=max_range_m,
            weather=compute_weather_factor(range_m, weather),
            weather_noise=1.0 if weather is None else weather.bearing_noise_scale(),
        )
        return DetectionResult(vis_result=vis, **result)

    def _terrain_intersection(
        self,
        origin: np.ndarray,
        target: np.ndarray,
        *,
        terrain_clearance_m: float,
    ) -> tuple[str, float | None, np.ndarray | None] | None:
        result = call(
            "terrain_intersection",
            terrain=self.environment.terrain._native.id,
            origin=origin,
            target=target,
            clearance=terrain_clearance_m,
        )
        return (
            None if result is None else (result[0], result[1], np.asarray(result[2], dtype=float))
        )

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
