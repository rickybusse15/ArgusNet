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
class _PresetProfile:
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


_PRESET_PROFILES: Mapping[str, _PresetProfile] = {
    "default": _PresetProfile(28.0, 58.0, 0.006, -0.004, 3, 2, 0.45, 0.60, 0.03, 0.65),
    "rolling_highlands": _PresetProfile(30.0, 60.0, 0.006, -0.004, 3, 2, 0.48, 0.60, 0.03, 0.68),
    "alpine": _PresetProfile(58.0, 150.0, 0.020, -0.015, 5, 3, 0.42, 0.90, 0.02, 0.45),
    "coastal": _PresetProfile(8.0, 28.0, 0.001, -0.018, 1, 2, 0.75, 0.32, 0.28, 0.90),
    "urban_flat": _PresetProfile(6.0, 12.0, 0.001, -0.001, 1, 1, 0.35, 0.18, 0.00, 1.00),
    "desert_canyon": _PresetProfile(22.0, 92.0, 0.004, -0.002, 2, 4, 0.14, 0.78, 0.01, 0.95),
    "lake_district": _PresetProfile(14.0, 62.0, -0.002, 0.004, 2, 3, 0.85, 0.46, 0.20, 0.78),
    "jungle_canopy": _PresetProfile(16.0, 44.0, 0.003, -0.002, 2, 3, 0.95, 0.56, 0.06, 0.95),
    "arctic_tundra": _PresetProfile(18.0, 46.0, 0.001, 0.002, 2, 2, 0.40, 0.38, 0.08, 0.20),
    "military_compound": _PresetProfile(12.0, 24.0, 0.001, -0.001, 1, 1, 0.28, 0.24, 0.00, 1.00),
    "river_valley": _PresetProfile(20.0, 70.0, 0.004, -0.003, 3, 4, 0.78, 0.50, 0.16, 0.82),
    "mountain_pass": _PresetProfile(70.0, 185.0, 0.016, -0.010, 6, 3, 0.45, 0.88, 0.02, 0.35),
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


def procedural_height_grid(
    *,
    bounds_xy_m: Bounds2D,
    preset_name: str,
    seed: int,
    resolution_m: float,
    detail_strength: float = 1.0,
    ground_plane_m: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    profile = _PRESET_PROFILES.get(preset_name, _PRESET_PROFILES["default"])
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

    return x_values, y_values, np.maximum(heights, float(ground_plane_m))


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

    if source == "procedural":
        _, _, heights = procedural_height_grid(
            bounds_xy_m=bounds_xy_m,
            preset_name=config.terrain_preset,
            seed=config.effective_seed,
            resolution_m=resolution_m,
            detail_strength=config.detail_strength,
            ground_plane_m=config.ground_plane_m,
        )
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
            _, _, detail = procedural_height_grid(
                bounds_xy_m=bounds_xy_m,
                preset_name=config.terrain_preset,
                seed=config.effective_seed,
                resolution_m=resolution_m,
                detail_strength=1.0,
                ground_plane_m=0.0,
            )
            detail = detail - float(np.mean(detail))
            heights = heights + detail * float(config.detail_strength)

    return TerrainLayer.from_height_grid(
        environment_id=environment_id,
        bounds_xy_m=bounds_xy_m,
        heights_m=heights,
        resolution_m=resolution_m,
        tile_size_cells=config.tile_size_cells,
        lod_resolutions_m=(resolution_m,),
        ground_plane_m=config.ground_plane_m,
        source_metadata=source_metadata,
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
) -> LandCoverLayer:
    season = season or SeasonalVariation.from_month(7)
    profile = _PRESET_PROFILES.get(terrain_preset, _PRESET_PROFILES["default"])
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

    water_mask = heights <= terrain.ground_plane_m + 1.0
    if terrain_preset in {"coastal", "lake_district", "river_valley", "arctic_tundra"}:
        water_level = np.quantile(heights, min(max(profile.water_bias, 0.03), 0.30))
        water_mask |= (heights <= water_level) & (slopes < 0.16)
    classes[water_mask] = int(LandCoverClass.WATER)

    wetland_mask = (
        (~water_mask) & (moisture > 0.68) & (elev_norm < 0.30) & (slopes < 0.14)
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

    rocky_mask = (~water_mask) & (~snow_mask) & ((slopes > 0.52) | (elev_norm > 0.78))
    classes[rocky_mask] = int(LandCoverClass.ROCKY)
    density[rocky_mask] = 60

    forest_mask = np.zeros((rows, cols), dtype=bool)
    if not suppress_vegetation:
        forest_mask = (
            (~water_mask)
            & (~snow_mask)
            & (~rocky_mask)
            & (moisture > 0.52)
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


__all__ = [
    "LandCoverPatch",
    "TerrainBuildConfig",
    "build_land_cover_layer",
    "build_terrain_layer",
    "procedural_height_grid",
]
