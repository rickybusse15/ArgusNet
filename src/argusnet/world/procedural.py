from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .environment import (
    Bounds2D,
    LandCoverClass,
    LandCoverLayer,
    SeasonalVariation,
    TerrainLayer,
)
from .obstacles import ForestStand, _point_in_polygon

_KNOWN_TERRAIN_SOURCES = frozenset({"procedural", "dem", "hybrid"})


@dataclass(frozen=True)
class TerrainBuildConfig:
    terrain_source: str = "procedural"
    terrain_preset: str = "alpine"
    terrain_seed: int | None = None
    dem_path: str | None = None
    dem_crs: str | None = None
    detail_strength: float = 1.0
    terrain_resolution_m: float | None = None
    season_month: int = 7
    tile_size_cells: int = 256
    ground_plane_m: float = 0.0

    def __post_init__(self) -> None:
        source = self.terrain_source.lower().strip()
        if source not in _KNOWN_TERRAIN_SOURCES:
            raise ValueError(
                f"Unknown terrain_source {self.terrain_source!r}; "
                f"must be one of {sorted(_KNOWN_TERRAIN_SOURCES)}."
            )
        object.__setattr__(self, "terrain_source", source)
        if self.terrain_resolution_m is not None and self.terrain_resolution_m <= 0.0:
            raise ValueError("terrain_resolution_m must be positive when provided.")
        if self.tile_size_cells < 2:
            raise ValueError("tile_size_cells must be at least 2.")
        if not (1 <= self.season_month <= 12):
            raise ValueError("season_month must be in 1..12.")
        if not np.isfinite(self.detail_strength) or self.detail_strength < 0.0:
            raise ValueError("detail_strength must be finite and non-negative.")

    @property
    def effective_seed(self) -> int:
        if self.terrain_seed is not None:
            return int(self.terrain_seed)
        return _stable_seed("terrain", self.terrain_preset)

    @property
    def season(self) -> SeasonalVariation:
        return SeasonalVariation.from_month(self.season_month)


@dataclass(frozen=True)
class LandCoverPatch:
    polygon_xy_m: np.ndarray
    land_cover_class: LandCoverClass
    density: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygon_xy_m", np.asarray(self.polygon_xy_m, dtype=float))


@dataclass(frozen=True)
class TerrainProfile:
    name: str
    base_elevation_m: float
    relief_m: float
    slope_x: float
    slope_y: float
    ridge_count: int
    valley_count: int
    moisture: float
    roughness: float
    water_bias: float
    snow_bias: float
    vegetation_bias: float = 0.5
    urban_bias: float = 0.0
    hydrology_strength: float = 0.5
    smoothing_passes: int = 0


@dataclass(frozen=True)
class ProceduralTerrainGrid:
    x_values_m: np.ndarray
    y_values_m: np.ndarray
    heights_m: np.ndarray
    masks: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]

    @property
    def water_mask(self) -> np.ndarray:
        return np.asarray(self.masks.get("water", np.zeros_like(self.heights_m, dtype=bool)))


@dataclass(frozen=True)
class LandscapeBuildConfig:
    terrain: TerrainBuildConfig = TerrainBuildConfig()
    land_cover_resolution_m: float | None = None
    patches: tuple[LandCoverPatch, ...] = ()
    obstacles: tuple[object, ...] = ()
    suppress_vegetation: bool = False


@dataclass(frozen=True)
class LandscapeBuildResult:
    terrain: TerrainLayer
    land_cover: LandCoverLayer
    obstacles: tuple[object, ...]
    masks: Mapping[str, np.ndarray]
    metadata: Mapping[str, object]


_PRESET_PROFILES: Mapping[str, TerrainProfile] = {
    "default": TerrainProfile(
        "default", 28.0, 58.0, 0.006, -0.004, 3, 2, 0.45, 0.60, 0.03, 0.65,
        0.50, 0.00, 0.35, 1,
    ),
    "rolling_highlands": TerrainProfile(
        "rolling_highlands", 30.0, 60.0, 0.006, -0.004, 3, 2, 0.48, 0.60,
        0.03, 0.68, 0.55, 0.00, 0.35, 1,
    ),
    "alpine": TerrainProfile(
        "alpine", 58.0, 150.0, 0.020, -0.015, 5, 3, 0.42, 0.90, 0.02,
        0.45, 0.34, 0.00, 0.20, 0,
    ),
    "coastal": TerrainProfile(
        "coastal", 8.0, 28.0, 0.001, -0.018, 1, 2, 0.75, 0.32, 0.28,
        0.90, 0.50, 0.10, 0.85, 2,
    ),
    "urban_flat": TerrainProfile(
        "urban_flat", 6.0, 12.0, 0.001, -0.001, 1, 1, 0.35, 0.18, 0.00,
        1.00, 0.15, 0.70, 0.10, 2,
    ),
    "desert_canyon": TerrainProfile(
        "desert_canyon", 22.0, 92.0, 0.004, -0.002, 2, 4, 0.14, 0.78,
        0.01, 0.95, 0.10, 0.02, 0.18, 0,
    ),
    "lake_district": TerrainProfile(
        "lake_district", 14.0, 62.0, -0.002, 0.004, 2, 3, 0.85, 0.46,
        0.20, 0.78, 0.65, 0.04, 0.90, 2,
    ),
    "jungle_canopy": TerrainProfile(
        "jungle_canopy", 16.0, 44.0, 0.003, -0.002, 2, 3, 0.95, 0.56,
        0.06, 0.95, 0.95, 0.00, 0.55, 1,
    ),
    "arctic_tundra": TerrainProfile(
        "arctic_tundra", 18.0, 46.0, 0.001, 0.002, 2, 2, 0.40, 0.38,
        0.08, 0.20, 0.12, 0.00, 0.45, 2,
    ),
    "military_compound": TerrainProfile(
        "military_compound", 12.0, 24.0, 0.001, -0.001, 1, 1, 0.28, 0.24,
        0.00, 1.00, 0.10, 0.85, 0.08, 2,
    ),
    "river_valley": TerrainProfile(
        "river_valley", 20.0, 70.0, 0.004, -0.003, 3, 4, 0.78, 0.50,
        0.16, 0.82, 0.62, 0.02, 0.95, 1,
    ),
    "mountain_pass": TerrainProfile(
        "mountain_pass", 70.0, 185.0, 0.016, -0.010, 6, 3, 0.45, 0.88,
        0.02, 0.35, 0.25, 0.00, 0.35, 0,
    ),
}


def _stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFF


def _grid_values(bounds: Bounds2D, resolution_m: float) -> tuple[np.ndarray, np.ndarray]:
    x_values = np.arange(bounds.x_min_m, bounds.x_max_m + resolution_m * 0.5, resolution_m)
    y_values = np.arange(bounds.y_min_m, bounds.y_max_m + resolution_m * 0.5, resolution_m)
    if x_values.size < 2:
        x_values = np.array([bounds.x_min_m, bounds.x_max_m], dtype=float)
    if y_values.size < 2:
        y_values = np.array([bounds.y_min_m, bounds.y_max_m], dtype=float)
    return x_values.astype(float), y_values.astype(float)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    return value * value * (3.0 - (2.0 * value))


def _hash_noise(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    h = (ix.astype(np.int64) * 73856093) ^ (iy.astype(np.int64) * 19349663) ^ int(seed)
    h = ((h >> 13) ^ h) * 0x45D9F3B
    h = ((h >> 13) ^ h) * 0x45D9F3B
    h = (h >> 13) ^ h
    return ((h & 0xFFFF).astype(float) / 32767.5) - 1.0


def _value_noise(x: np.ndarray, y: np.ndarray, seed: int) -> np.ndarray:
    ix = np.floor(x).astype(np.int64)
    iy = np.floor(y).astype(np.int64)
    fx = _smoothstep(x - ix)
    fy = _smoothstep(y - iy)
    n00 = _hash_noise(ix, iy, seed)
    n10 = _hash_noise(ix + 1, iy, seed)
    n01 = _hash_noise(ix, iy + 1, seed)
    n11 = _hash_noise(ix + 1, iy + 1, seed)
    nx0 = n00 + fx * (n10 - n00)
    nx1 = n01 + fx * (n11 - n01)
    return nx0 + fy * (nx1 - nx0)


def _fbm(
    x_m: np.ndarray,
    y_m: np.ndarray,
    *,
    seed: int,
    base_wavelength_m: float,
    octaves: int,
    persistence: float = 0.52,
    lacunarity: float = 2.03,
) -> np.ndarray:
    total = np.zeros_like(x_m, dtype=float)
    amp = 1.0
    wavelength = max(float(base_wavelength_m), 1.0)
    amp_sum = 0.0
    for octave in range(max(octaves, 1)):
        total += amp * _value_noise(x_m / wavelength, y_m / wavelength, seed + octave * 9973)
        amp_sum += amp
        wavelength /= lacunarity
        amp *= persistence
    return total / max(amp_sum, 1.0e-9)


def _line_gaussian(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    angle_rad: float,
    offset_m: float,
    width_m: float,
    length_m: float,
    center_xy: tuple[float, float],
) -> np.ndarray:
    dx = xx - center_xy[0]
    dy = yy - center_xy[1]
    normal_x = -math.sin(angle_rad)
    normal_y = math.cos(angle_rad)
    along_x = math.cos(angle_rad)
    along_y = math.sin(angle_rad)
    cross = (dx * normal_x) + (dy * normal_y) - offset_m
    along = (dx * along_x) + (dy * along_y)
    cross_env = np.exp(-0.5 * (cross / max(width_m, 1.0)) ** 2)
    along_env = np.exp(-0.5 * (along / max(length_m, 1.0)) ** 4)
    return cross_env * along_env


def terrain_profile_for_preset(preset_name: str) -> TerrainProfile:
    return _PRESET_PROFILES.get(preset_name, _PRESET_PROFILES["default"])


def _slope_grid(heights_m: np.ndarray, resolution_m: float) -> np.ndarray:
    dz_dy, dz_dx = np.gradient(heights_m, max(float(resolution_m), 1.0))
    return np.arctan(np.sqrt((dz_dx * dz_dx) + (dz_dy * dz_dy)))


def _dilate_mask(mask: np.ndarray, iterations: int = 1) -> np.ndarray:
    dilated = np.asarray(mask, dtype=bool)
    for _ in range(max(iterations, 0)):
        padded = np.pad(dilated, 1, constant_values=False)
        dilated = (
            padded[1:-1, 1:-1]
            | padded[:-2, 1:-1]
            | padded[2:, 1:-1]
            | padded[1:-1, :-2]
            | padded[1:-1, 2:]
            | padded[:-2, :-2]
            | padded[:-2, 2:]
            | padded[2:, :-2]
            | padded[2:, 2:]
        )
    return dilated


def _smooth_heights(
    heights_m: np.ndarray,
    *,
    preserve_mask: np.ndarray,
    passes: int,
    blend: float,
) -> np.ndarray:
    smoothed = np.asarray(heights_m, dtype=float).copy()
    for _ in range(max(passes, 0)):
        padded = np.pad(smoothed, 1, mode="edge")
        neighbor_mean = (
            padded[1:-1, 1:-1] * 4.0
            + padded[:-2, 1:-1]
            + padded[2:, 1:-1]
            + padded[1:-1, :-2]
            + padded[1:-1, 2:]
        ) / 8.0
        candidate = (smoothed * (1.0 - blend)) + (neighbor_mean * blend)
        smoothed = np.where(preserve_mask, smoothed, candidate)
    return smoothed


def _ellipse_mask(
    xx: np.ndarray,
    yy: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radius_x_m: float,
    radius_y_m: float,
) -> np.ndarray:
    return (
        ((xx - center_xy[0]) / max(radius_x_m, 1.0)) ** 2
        + ((yy - center_xy[1]) / max(radius_y_m, 1.0)) ** 2
    ) <= 1.0


def _semantic_masks(
    *,
    preset_name: str,
    profile: TerrainProfile,
    heights_m: np.ndarray,
    xx: np.ndarray,
    yy: np.ndarray,
    bounds_xy_m: Bounds2D,
    resolution_m: float,
    seed: int,
    ground_plane_m: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    scene_scale_m = max(bounds_xy_m.width_m, bounds_xy_m.height_m, 1.0)
    center_x = (bounds_xy_m.x_min_m + bounds_xy_m.x_max_m) * 0.5
    center_y = (bounds_xy_m.y_min_m + bounds_xy_m.y_max_m) * 0.5
    slopes = _slope_grid(heights_m, resolution_m)
    h_min = float(np.min(heights_m))
    h_max = float(np.max(heights_m))
    elev_norm = (heights_m - h_min) / max(h_max - h_min, 1.0)
    water_level = float(
        np.quantile(heights_m, min(max(profile.water_bias, 0.01), 0.35))
        if profile.water_bias > 0.0
        else ground_plane_m
    )

    lowland_mask = (heights_m <= water_level + (profile.relief_m * 0.05)) & (slopes < 0.22)
    water_mask = (heights_m <= ground_plane_m + 1.0) | (
        (profile.water_bias > 0.0) & lowland_mask & (heights_m <= water_level)
    )
    valley_mask = lowland_mask.copy()

    if preset_name in {"river_valley", "mountain_pass"}:
        phase = (_stable_seed("river-phase", preset_name, seed) % 6283) / 1000.0
        amplitude_m = scene_scale_m * (0.015 if preset_name == "mountain_pass" else 0.035)
        river_y = center_y + amplitude_m * np.sin(
            ((xx - center_x) / max(scene_scale_m * 0.23, 1.0)) + phase
        )
        river_width = max(resolution_m * 1.5, scene_scale_m * 0.018)
        river_mask = np.abs(yy - river_y) <= river_width
        river_bank_mask = np.abs(yy - river_y) <= river_width * 3.0
        river_level = (
            float(np.quantile(heights_m[river_mask], 0.18))
            if np.any(river_mask)
            else water_level
        )
        heights_m = np.where(river_bank_mask, heights_m - profile.relief_m * 0.035, heights_m)
        heights_m = np.where(river_mask, river_level, heights_m)
        water_mask |= river_mask
        valley_mask |= river_bank_mask

    if preset_name == "lake_district":
        lakes = (
            _ellipse_mask(
                xx,
                yy,
                center_xy=(center_x - scene_scale_m * 0.22, center_y + scene_scale_m * 0.16),
                radius_x_m=scene_scale_m * 0.09,
                radius_y_m=scene_scale_m * 0.06,
            )
            | _ellipse_mask(
                xx,
                yy,
                center_xy=(center_x + scene_scale_m * 0.24, center_y - scene_scale_m * 0.17),
                radius_x_m=scene_scale_m * 0.08,
                radius_y_m=scene_scale_m * 0.055,
            )
        )
        lake_level = float(np.quantile(heights_m[lakes], 0.28)) if np.any(lakes) else water_level
        heights_m = np.where(
            _dilate_mask(lakes, 1),
            np.minimum(heights_m, lake_level + 1.5),
            heights_m,
        )
        heights_m = np.where(lakes, lake_level, heights_m)
        water_mask |= lakes
        lowland_mask |= _dilate_mask(lakes, 2)

    if preset_name == "coastal":
        coast_limit = bounds_xy_m.y_min_m + bounds_xy_m.height_m * 0.26
        coast_mask = yy <= coast_limit
        shelf = yy <= bounds_xy_m.y_min_m + bounds_xy_m.height_m * 0.34
        coast_level = max(float(ground_plane_m), water_level)
        heights_m = np.where(shelf, np.minimum(heights_m, coast_level + 2.0), heights_m)
        heights_m = np.where(coast_mask, coast_level, heights_m)
        water_mask |= coast_mask
        lowland_mask |= shelf

    if preset_name in {"urban_flat", "military_compound"}:
        pad = _ellipse_mask(
            xx,
            yy,
            center_xy=(center_x, center_y),
            radius_x_m=scene_scale_m * (0.18 if preset_name == "urban_flat" else 0.20),
            radius_y_m=scene_scale_m * (0.15 if preset_name == "urban_flat" else 0.18),
        )
        pad_level = (
            float(np.median(heights_m[pad]))
            if np.any(pad)
            else float(np.median(heights_m))
        )
        heights_m = np.where(
            _dilate_mask(pad, 2),
            (heights_m * 0.35) + (pad_level * 0.65),
            heights_m,
        )

    water_mask = _dilate_mask(water_mask, 0)
    if np.any(water_mask):
        water_level_by_grid = np.minimum(heights_m, water_level)
        heights_m = np.where(water_mask, water_level_by_grid, heights_m)
    heights_m = _smooth_heights(
        heights_m,
        preserve_mask=water_mask,
        passes=profile.smoothing_passes,
        blend=0.28,
    )
    heights_m = np.maximum(heights_m, float(ground_plane_m))

    slopes = _slope_grid(heights_m, resolution_m)
    steep_slope_mask = slopes > (0.52 if profile.roughness < 0.7 else 0.62)
    ridge_mask = (elev_norm > 0.72) | (steep_slope_mask & (elev_norm > 0.55))
    road_mask = np.zeros_like(water_mask, dtype=bool)
    if preset_name in {"urban_flat", "military_compound"}:
        road_width = max(resolution_m * 1.2, scene_scale_m * 0.012)
        road_mask = (np.abs(xx - center_x) <= road_width) | (np.abs(yy - center_y) <= road_width)
    elif preset_name in {"river_valley", "lake_district"}:
        road_mask = valley_mask & (~water_mask) & (slopes < 0.16)

    masks = {
        "water": water_mask.astype(bool),
        "ridge": ridge_mask.astype(bool),
        "valley": valley_mask.astype(bool),
        "steep_slope": steep_slope_mask.astype(bool),
        "lowland": lowland_mask.astype(bool),
        "road": road_mask.astype(bool),
    }
    return heights_m, masks


def procedural_terrain_grid(
    *,
    bounds_xy_m: Bounds2D,
    preset_name: str,
    seed: int,
    resolution_m: float,
    detail_strength: float = 1.0,
    ground_plane_m: float = 0.0,
) -> ProceduralTerrainGrid:
    profile = terrain_profile_for_preset(preset_name)
    rng = np.random.default_rng(_stable_seed("profile", preset_name, seed))
    x_values, y_values = _grid_values(bounds_xy_m, resolution_m)
    xx, yy = np.meshgrid(x_values, y_values)
    width_m = bounds_xy_m.width_m
    height_m = bounds_xy_m.height_m
    scene_scale_m = max(width_m, height_m, 1.0)
    center_xy = (
        (bounds_xy_m.x_min_m + bounds_xy_m.x_max_m) * 0.5,
        (bounds_xy_m.y_min_m + bounds_xy_m.y_max_m) * 0.5,
    )

    warp_wavelength_m = scene_scale_m * 0.42
    warp_strength_m = scene_scale_m * 0.035 * profile.roughness
    warp_x = _fbm(
        xx,
        yy,
        seed=_stable_seed("warp-x", seed, preset_name),
        base_wavelength_m=warp_wavelength_m,
        octaves=3,
    )
    warp_y = _fbm(
        xx,
        yy,
        seed=_stable_seed("warp-y", seed, preset_name),
        base_wavelength_m=warp_wavelength_m,
        octaves=3,
    )
    wx = xx + warp_x * warp_strength_m
    wy = yy + warp_y * warp_strength_m

    slope = (wx - center_xy[0]) * profile.slope_x + (wy - center_xy[1]) * profile.slope_y
    macro = (
        np.sin((wx / max(scene_scale_m * 0.42, 1.0)) + rng.uniform(-math.pi, math.pi))
        * np.cos((wy / max(scene_scale_m * 0.37, 1.0)) + rng.uniform(-math.pi, math.pi))
        * profile.relief_m
        * 0.10
    )

    ridges = np.zeros_like(xx, dtype=float)
    for index in range(profile.ridge_count):
        angle = rng.uniform(-math.pi, math.pi)
        offset = rng.uniform(-0.36, 0.36) * scene_scale_m
        width = scene_scale_m * rng.uniform(0.045, 0.13)
        length = scene_scale_m * rng.uniform(0.36, 0.85)
        amp = profile.relief_m * rng.uniform(0.20, 0.58)
        ridge = _line_gaussian(
            wx,
            wy,
            angle_rad=angle,
            offset_m=offset,
            width_m=width,
            length_m=length,
            center_xy=center_xy,
        )
        crenulation = 1.0 + 0.18 * _fbm(
            wx,
            wy,
            seed=_stable_seed("ridge", seed, preset_name, index),
            base_wavelength_m=max(width * 1.8, 10.0),
            octaves=3,
        )
        ridges += amp * ridge * crenulation

    valleys = np.zeros_like(xx, dtype=float)
    for index in range(profile.valley_count):
        if preset_name in {"river_valley", "mountain_pass"} and index == 0:
            angle = rng.uniform(-0.10, 0.10)
            offset = 0.0
        else:
            angle = rng.uniform(-math.pi, math.pi)
            offset = rng.uniform(-0.30, 0.30) * scene_scale_m
        width = scene_scale_m * rng.uniform(0.035, 0.10)
        length = scene_scale_m * rng.uniform(0.45, 1.05)
        amp = profile.relief_m * rng.uniform(0.12, 0.36)
        valley = _line_gaussian(
            wx,
            wy,
            angle_rad=angle,
            offset_m=offset,
            width_m=width,
            length_m=length,
            center_xy=center_xy,
        )
        valleys += amp * valley

    broad_noise = _fbm(
        wx,
        wy,
        seed=_stable_seed("broad", seed, preset_name),
        base_wavelength_m=scene_scale_m * 0.30,
        octaves=5,
    )
    fine_noise = _fbm(
        wx,
        wy,
        seed=_stable_seed("fine", seed, preset_name),
        base_wavelength_m=scene_scale_m * 0.075,
        octaves=4,
    )
    detail = (
        broad_noise * profile.relief_m * 0.12
        + fine_noise * profile.relief_m * 0.045 * profile.roughness
    ) * float(detail_strength)

    heights = profile.base_elevation_m + slope + macro + ridges - valleys + detail

    if profile.water_bias > 0.0:
        water_level = np.quantile(heights, min(max(profile.water_bias, 0.01), 0.35))
        lowland = np.clip((water_level - heights) / max(profile.relief_m * 0.18, 1.0), 0.0, 1.0)
        heights -= lowland * profile.relief_m * 0.08

    heights = np.maximum(heights, float(ground_plane_m))
    heights, masks = _semantic_masks(
        preset_name=preset_name,
        profile=profile,
        heights_m=heights,
        xx=xx,
        yy=yy,
        bounds_xy_m=bounds_xy_m,
        resolution_m=resolution_m,
        seed=seed,
        ground_plane_m=ground_plane_m,
    )
    metadata = {
        "generator_version": "procedural-landscape-v1",
        "preset": preset_name,
        "seed": int(seed),
        "profile": {
            "name": profile.name,
            "base_elevation_m": profile.base_elevation_m,
            "relief_m": profile.relief_m,
            "roughness": profile.roughness,
            "moisture": profile.moisture,
            "water_bias": profile.water_bias,
            "snow_bias": profile.snow_bias,
            "vegetation_bias": profile.vegetation_bias,
            "urban_bias": profile.urban_bias,
            "hydrology_strength": profile.hydrology_strength,
            "smoothing_passes": profile.smoothing_passes,
        },
        "semantic_masks": sorted(masks),
        "mask_coverage_fraction": {
            name: float(np.mean(mask)) for name, mask in sorted(masks.items())
        },
    }
    return ProceduralTerrainGrid(
        x_values_m=x_values,
        y_values_m=y_values,
        heights_m=heights,
        masks=masks,
        metadata=metadata,
    )


def procedural_height_grid(
    *,
    bounds_xy_m: Bounds2D,
    preset_name: str,
    seed: int,
    resolution_m: float,
    detail_strength: float = 1.0,
    ground_plane_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grid = procedural_terrain_grid(
        bounds_xy_m=bounds_xy_m,
        preset_name=preset_name,
        seed=seed,
        resolution_m=resolution_m,
        detail_strength=detail_strength,
        ground_plane_m=ground_plane_m,
    )
    return grid.x_values_m, grid.y_values_m, grid.heights_m


def _sample_layer_to_grid(
    terrain: TerrainLayer,
    bounds_xy_m: Bounds2D,
    resolution_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_values, y_values = _grid_values(bounds_xy_m, resolution_m)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    heights = terrain.height_at_many(points).reshape(xx.shape)
    return x_values, y_values, heights


def _sample_mask_to_grid(
    mask: np.ndarray,
    *,
    source_bounds: Bounds2D,
    target_xx: np.ndarray,
    target_yy: np.ndarray,
) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 2 or mask_array.size == 0:
        return np.zeros_like(target_xx, dtype=bool)
    rows, cols = mask_array.shape
    col = np.clip(
        np.rint(
            (target_xx - source_bounds.x_min_m)
            / max(source_bounds.width_m, 1.0e-9)
            * max(cols - 1, 1)
        ).astype(int),
        0,
        cols - 1,
    )
    row = np.clip(
        np.rint(
            (target_yy - source_bounds.y_min_m)
            / max(source_bounds.height_m, 1.0e-9)
            * max(rows - 1, 1)
        ).astype(int),
        0,
        rows - 1,
    )
    return mask_array[row, col]


def build_terrain_layer(
    config: TerrainBuildConfig,
    bounds_xy_m: Bounds2D,
    *,
    environment_id: str = "terrain",
) -> TerrainLayer:
    source = config.terrain_source
    if config.dem_path and source == "procedural":
        source = "hybrid"
    if source in {"dem", "hybrid"} and not config.dem_path:
        raise ValueError(f"terrain_source={source!r} requires dem_path.")

    resolution_m = float(config.terrain_resolution_m or 5.0)
    source_metadata: dict[str, object] = {
        "source": source,
        "preset": config.terrain_preset,
        "seed": config.effective_seed,
        "detail_strength": float(config.detail_strength),
        "season_month": int(config.season_month),
    }
    semantic_masks: Mapping[str, np.ndarray] = {}

    if source == "procedural":
        grid = procedural_terrain_grid(
            bounds_xy_m=bounds_xy_m,
            preset_name=config.terrain_preset,
            seed=config.effective_seed,
            resolution_m=resolution_m,
            detail_strength=config.detail_strength,
            ground_plane_m=config.ground_plane_m,
        )
        heights = grid.heights_m
        semantic_masks = grid.masks
        source_metadata.update(grid.metadata)
    else:
        dem_path = Path(str(config.dem_path))
        if not dem_path.exists():
            raise ValueError(f"DEM path does not exist: {dem_path}")
        dem = TerrainLayer.from_geotiff(
            dem_path,
            environment_id=f"{environment_id}-dem",
            source_crs=config.dem_crs,
            ground_plane_m=config.ground_plane_m,
        )
        _, _, heights = _sample_layer_to_grid(dem, bounds_xy_m, resolution_m)
        source_metadata.update(
            {
                "dem_path": str(dem_path),
                "dem_crs": config.dem_crs,
                "dem_bounds_m": dem.bounds_xy_m.to_metadata(),
            }
        )
        if source == "hybrid":
            detail_grid = procedural_terrain_grid(
                bounds_xy_m=bounds_xy_m,
                preset_name=config.terrain_preset,
                seed=config.effective_seed,
                resolution_m=resolution_m,
                detail_strength=1.0,
                ground_plane_m=0.0,
            )
            detail = detail_grid.heights_m
            detail = detail - float(np.mean(detail))
            heights = heights + detail * float(config.detail_strength)
            semantic_masks = detail_grid.masks
            source_metadata.update(
                {
                    "generator_version": "hybrid-landscape-v1",
                    "semantic_masks": sorted(semantic_masks),
                    "mask_coverage_fraction": {
                        name: float(np.mean(mask))
                        for name, mask in sorted(semantic_masks.items())
                    },
                }
            )

    return TerrainLayer.from_height_grid(
        environment_id=environment_id,
        bounds_xy_m=bounds_xy_m,
        heights_m=heights,
        resolution_m=resolution_m,
        tile_size_cells=config.tile_size_cells,
        lod_resolutions_m=(resolution_m,),
        ground_plane_m=config.ground_plane_m,
        source_metadata=source_metadata,
        semantic_masks=semantic_masks,
    )


def build_land_cover_layer(
    *,
    bounds_xy_m: Bounds2D,
    terrain: TerrainLayer,
    obstacles: Sequence[object],
    patches: Sequence[LandCoverPatch] = (),
    resolution_m: float,
    season: SeasonalVariation | None = None,
    terrain_preset: str = "default",
    seed: int = 0,
    suppress_vegetation: bool = False,
    terrain_masks: Mapping[str, np.ndarray] | None = None,
) -> LandCoverLayer:
    season = season or SeasonalVariation.from_month(7)
    profile = terrain_profile_for_preset(terrain_preset)
    cols = max(1, int(math.ceil(bounds_xy_m.width_m / resolution_m)))
    rows = max(1, int(math.ceil(bounds_xy_m.height_m / resolution_m)))
    x_values = bounds_xy_m.x_min_m + ((np.arange(cols, dtype=float) + 0.5) * resolution_m)
    y_values = bounds_xy_m.y_min_m + ((np.arange(rows, dtype=float) + 0.5) * resolution_m)
    xx, yy = np.meshgrid(x_values, y_values)
    points = np.column_stack([xx.reshape(-1), yy.reshape(-1)])
    heights = terrain.height_at_many(points).reshape(rows, cols)
    slopes = terrain.slope_rad_at_many(points).reshape(rows, cols)

    h_min = float(np.min(heights))
    h_max = float(np.max(heights))
    elev_norm = (heights - h_min) / max(h_max - h_min, 1.0)
    moisture_noise = _fbm(
        xx,
        yy,
        seed=_stable_seed("land-cover", terrain_preset, seed),
        base_wavelength_m=max(bounds_xy_m.width_m, bounds_xy_m.height_m) * 0.22,
        octaves=4,
    )
    moisture = np.clip(profile.moisture + (0.28 * moisture_noise), 0.0, 1.0)

    classes = np.full((rows, cols), int(LandCoverClass.OPEN), dtype=np.uint8)
    density = np.zeros((rows, cols), dtype=np.uint8)
    source_masks = (
        terrain_masks if terrain_masks is not None else getattr(terrain, "semantic_masks", {})
    )
    sampled_masks = {
        name: _sample_mask_to_grid(
            mask,
            source_bounds=terrain.bounds_xy_m,
            target_xx=xx,
            target_yy=yy,
        )
        for name, mask in dict(source_masks).items()
    }

    water_mask = (heights <= terrain.ground_plane_m + 1.0) | sampled_masks.get(
        "water", np.zeros((rows, cols), dtype=bool)
    )
    if terrain_preset in {"coastal", "lake_district", "river_valley", "arctic_tundra"}:
        water_level = np.quantile(heights, min(max(profile.water_bias, 0.03), 0.30))
        water_mask |= (heights <= water_level) & (slopes < 0.16)
    classes[water_mask] = int(LandCoverClass.WATER)

    wetland_mask = (
        (~water_mask)
        & (
            sampled_masks.get("lowland", np.zeros((rows, cols), dtype=bool))
            | sampled_masks.get("valley", np.zeros((rows, cols), dtype=bool))
            | ((moisture > 0.68) & (elev_norm < 0.30))
        )
        & (slopes < 0.14)
    )
    if not suppress_vegetation:
        classes[wetland_mask] = int(LandCoverClass.WETLAND)
        density[wetland_mask] = 120

    snow_mask = (~water_mask) & (
        (season.snow_cover & (elev_norm > profile.snow_bias))
        | (terrain_preset == "arctic_tundra")
        | ((terrain_preset in {"alpine", "mountain_pass"}) & (elev_norm > 0.82))
    )
    classes[snow_mask] = int(LandCoverClass.SNOW)
    density[snow_mask] = 40

    rocky_mask = (
        (~water_mask)
        & (~snow_mask)
        & (
            sampled_masks.get("steep_slope", np.zeros((rows, cols), dtype=bool))
            | sampled_masks.get("ridge", np.zeros((rows, cols), dtype=bool))
            | (slopes > 0.52)
            | (elev_norm > 0.78)
        )
    )
    classes[rocky_mask] = int(LandCoverClass.ROCKY)
    density[rocky_mask] = 60

    forest_mask = np.zeros((rows, cols), dtype=bool)
    if not suppress_vegetation:
        forest_mask = (
            (~water_mask)
            & (~snow_mask)
            & (~rocky_mask)
            & (moisture > max(0.35, 0.55 - (profile.vegetation_bias * 0.18)))
            & (elev_norm > 0.18)
            & (slopes < 0.45)
        )
        classes[forest_mask] = int(LandCoverClass.FOREST)
        density[forest_mask] = np.maximum(
            density[forest_mask],
            np.uint8(np.clip(190 * season.foliage_density_factor, 0, 255)),
        )

        scrub_mask = (
            (~water_mask)
            & (~snow_mask)
            & (~rocky_mask)
            & (~forest_mask)
            & ((moisture < 0.42) | (slopes > 0.24))
        )
        classes[scrub_mask] = int(LandCoverClass.SCRUB)
        density[scrub_mask] = np.maximum(
            density[scrub_mask],
            np.uint8(np.clip(105 * season.foliage_density_factor, 0, 255)),
        )

    road_mask = sampled_masks.get("road", np.zeros((rows, cols), dtype=bool)) & (~water_mask)
    classes[road_mask] = int(LandCoverClass.ROAD)
    density[road_mask] = 0

    for obstacle in obstacles:
        if isinstance(obstacle, ForestStand):
            if suppress_vegetation:
                continue
            footprint = obstacle.footprint_xy_m
            cover_class = LandCoverClass.FOREST
            cover_density = int(np.clip(obstacle.density, 0.0, 1.0) * 255)
        elif hasattr(obstacle, "footprint_xy_m"):
            footprint_source = obstacle.footprint_xy_m
            footprint = footprint_source() if callable(footprint_source) else footprint_source
            cover_class = LandCoverClass.URBAN
            cover_density = 180
        else:
            continue
        polygon = np.asarray(footprint, dtype=float)
        for row in range(rows):
            for col in range(cols):
                if _point_in_polygon(np.array([xx[row, col], yy[row, col]], dtype=float), polygon):
                    classes[row, col] = int(cover_class)
                    density[row, col] = max(density[row, col], cover_density)

    for patch in patches:
        for row in range(rows):
            for col in range(cols):
                if _point_in_polygon(
                    np.array([xx[row, col], yy[row, col]], dtype=float),
                    patch.polygon_xy_m,
                ):
                    classes[row, col] = int(patch.land_cover_class)
                    density[row, col] = max(density[row, col], int(np.clip(patch.density, 0, 255)))

    return LandCoverLayer.from_rasters(
        bounds_xy_m=bounds_xy_m,
        classes=classes,
        density=density,
        resolution_m=resolution_m,
        tile_size_cells=256,
    )


def build_landscape(
    config: LandscapeBuildConfig,
    bounds_xy_m: Bounds2D,
    *,
    environment_id: str = "landscape",
) -> LandscapeBuildResult:
    terrain = build_terrain_layer(config.terrain, bounds_xy_m, environment_id=environment_id)
    land_cover_resolution_m = float(
        config.land_cover_resolution_m or max(10.0, terrain.base_resolution_m * 2.0)
    )
    land_cover = build_land_cover_layer(
        bounds_xy_m=bounds_xy_m,
        terrain=terrain,
        obstacles=config.obstacles,
        patches=config.patches,
        resolution_m=land_cover_resolution_m,
        season=config.terrain.season,
        terrain_preset=config.terrain.terrain_preset,
        seed=config.terrain.effective_seed,
        suppress_vegetation=config.suppress_vegetation,
        terrain_masks=terrain.semantic_masks,
    )
    metadata = {
        "generator_version": "landscape-build-v1",
        "terrain": terrain.terrain_summary(),
        "land_cover": land_cover.to_metadata(),
        "obstacle_count": len(config.obstacles),
    }
    return LandscapeBuildResult(
        terrain=terrain,
        land_cover=land_cover,
        obstacles=tuple(config.obstacles),
        masks=terrain.semantic_masks,
        metadata=metadata,
    )


__all__ = [
    "LandscapeBuildConfig",
    "LandscapeBuildResult",
    "LandCoverPatch",
    "ProceduralTerrainGrid",
    "TerrainBuildConfig",
    "TerrainProfile",
    "build_land_cover_layer",
    "build_landscape",
    "build_terrain_layer",
    "procedural_terrain_grid",
    "procedural_height_grid",
    "terrain_profile_for_preset",
]
