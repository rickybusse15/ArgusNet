from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple, List

import numpy as np


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


@dataclass(frozen=True)
class Bounds2D:
    x_min_m: float
    x_max_m: float
    y_min_m: float
    y_max_m: float

    def __post_init__(self) -> None:
        if self.x_max_m <= self.x_min_m or self.y_max_m <= self.y_min_m:
            raise ValueError("Bounds2D must have positive width and height.")

    @property
    def width_m(self) -> float:
        return self.x_max_m - self.x_min_m

    @property
    def height_m(self) -> float:
        return self.y_max_m - self.y_min_m

    def padded(self, padding_m: float) -> "Bounds2D":
        return Bounds2D(
            x_min_m=self.x_min_m - padding_m,
            x_max_m=self.x_max_m + padding_m,
            y_min_m=self.y_min_m - padding_m,
            y_max_m=self.y_max_m + padding_m,
        )

    def contains_xy(self, x_m: float, y_m: float) -> bool:
        return self.x_min_m <= x_m <= self.x_max_m and self.y_min_m <= y_m <= self.y_max_m

    def to_metadata(self) -> Dict[str, float]:
        return {
            "x_min_m": float(self.x_min_m),
            "x_max_m": float(self.x_max_m),
            "y_min_m": float(self.y_min_m),
            "y_max_m": float(self.y_max_m),
        }

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, float]) -> "Bounds2D":
        return cls(
            x_min_m=float(mapping["x_min_m"]),
            x_max_m=float(mapping["x_max_m"]),
            y_min_m=float(mapping["y_min_m"]),
            y_max_m=float(mapping["y_max_m"]),
        )


@dataclass(frozen=True)
class EnvironmentCRS:
    source_crs_id: str = "local-synthetic"
    runtime_crs_id: str = "local-enu"
    origin_lat_deg: Optional[float] = None
    origin_lon_deg: Optional[float] = None
    origin_h_m: Optional[float] = None
    xy_units: str = "meters"
    z_datum: str = "local"

    def to_metadata(self) -> Dict[str, object]:
        return {
            "source_crs_id": self.source_crs_id,
            "runtime_crs_id": self.runtime_crs_id,
            "origin_geodetic": {
                "lat_deg": self.origin_lat_deg,
                "lon_deg": self.origin_lon_deg,
                "h_m": self.origin_h_m,
            },
            "xy_units": self.xy_units,
            "z_datum": self.z_datum,
        }


class LandCoverClass(IntEnum):
    OPEN = 0
    URBAN = 1
    FOREST = 2
    WATER = 3
    SCRUB = 4
    WETLAND = 5
    ROCKY = 6
    SNOW = 7
    ROAD = 8

    @classmethod
    def legend(cls) -> Dict[str, int]:
        return {member.name.lower(): int(member) for member in cls}


_MONTH_TO_SEASON = {
    12: "winter", 1: "winter", 2: "winter",
    3: "spring", 4: "spring", 5: "spring",
    6: "summer", 7: "summer", 8: "summer",
    9: "autumn", 10: "autumn", 11: "autumn",
}

_SEASON_DEFAULTS: Dict[str, Tuple[float, bool]] = {
    "spring": (0.6, False),
    "summer": (1.0, False),
    "autumn": (0.4, False),
    "winter": (0.15, True),
}


@dataclass(frozen=True)
class SeasonalVariation:
    """Encapsulates season-dependent environmental parameters."""

    season: str
    foliage_density_factor: float
    snow_cover: bool

    def __post_init__(self) -> None:
        if self.season not in ("spring", "summer", "autumn", "winter"):
            raise ValueError(
                f"season must be one of spring/summer/autumn/winter, got {self.season!r}"
            )
        if not (0.0 <= self.foliage_density_factor <= 1.0):
            raise ValueError("foliage_density_factor must be in [0, 1].")

    @classmethod
    def from_month(cls, month: int) -> "SeasonalVariation":
        """Create a SeasonalVariation from a calendar month (1-12)."""
        if not (1 <= month <= 12):
            raise ValueError(f"month must be 1-12, got {month}")
        season = _MONTH_TO_SEASON[month]
        foliage, snow = _SEASON_DEFAULTS[season]
        return cls(season=season, foliage_density_factor=foliage, snow_cover=snow)


def _build_pyramid(grid: np.ndarray, reducer: str) -> Tuple[np.ndarray, ...]:
    levels = [np.asarray(grid, dtype=float)]
    current = levels[0]
    while current.shape[0] > 2 or current.shape[1] > 2:
        next_rows = max(2, int(math.ceil((current.shape[0] - 1) / 2.0)) + 1)
        next_cols = max(2, int(math.ceil((current.shape[1] - 1) / 2.0)) + 1)
        reduced = np.empty((next_rows, next_cols), dtype=float)
        for row in range(next_rows):
            row_start = min(row * 2, current.shape[0] - 2)
            row_slice = current[row_start:min(row_start + 3, current.shape[0]), :]
            for col in range(next_cols):
                col_start = min(col * 2, current.shape[1] - 2)
                block = row_slice[:, col_start:min(col_start + 3, current.shape[1])]
                reduced[row, col] = float(np.max(block) if reducer == "max" else np.min(block))
        levels.append(reduced)
        current = reduced
        if current.shape == (2, 2):
            break
    return tuple(levels)


@dataclass(frozen=True)
class TerrainTile:
    tx: int
    ty: int
    lod: int
    x_min_m: float
    y_min_m: float
    cell_size_m: float
    heights_m: np.ndarray
    min_pyramid: Tuple[np.ndarray, ...]
    max_pyramid: Tuple[np.ndarray, ...]

    @property
    def tile_size_cells(self) -> int:
        return self.heights_m.shape[0] - 1

    @property
    def span_m(self) -> float:
        return self.tile_size_cells * self.cell_size_m

    @property
    def bounds_xy_m(self) -> Bounds2D:
        return Bounds2D(
            self.x_min_m,
            self.x_min_m + self.span_m,
            self.y_min_m,
            self.y_min_m + self.span_m,
        )

    @property
    def max_height_m(self) -> float:
        return float(np.max(self.max_pyramid[0]))

    @property
    def min_height_m(self) -> float:
        return float(np.min(self.min_pyramid[0]))


class TerrainLayer:
    def __init__(
        self,
        *,
        bounds_xy_m: Bounds2D,
        tile_size_cells: int,
        base_resolution_m: float,
        lod_resolutions_m: Sequence[float],
        interpolation: str,
        ground_plane_m: float,
        tiles: Mapping[Tuple[int, int, int], TerrainTile],
        environment_id: str = "environment",
    ) -> None:
        self.bounds_xy_m = bounds_xy_m
        self.tile_size_cells = int(tile_size_cells)
        self.base_resolution_m = float(base_resolution_m)
        self.lod_resolutions_m = tuple(float(value) for value in lod_resolutions_m)
        self.interpolation = interpolation
        self.ground_plane_m = float(ground_plane_m)
        self.environment_id = environment_id
        self._tiles = dict(tiles)
        self._tile_cache: "OrderedDict[Tuple[int, int, int], TerrainTile]" = OrderedDict()

    @classmethod
    def from_height_grid(
        cls,
        *,
        environment_id: str,
        bounds_xy_m: Bounds2D,
        heights_m: np.ndarray,
        resolution_m: float,
        tile_size_cells: int = 256,
        lod_resolutions_m: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
        interpolation: str = "bilinear",
        ground_plane_m: float = 0.0,
    ) -> "TerrainLayer":
        heights = np.asarray(heights_m, dtype=float)
        if heights.ndim != 2 or heights.shape[0] < 2 or heights.shape[1] < 2:
            raise ValueError("heights_m must be a 2D array with at least shape (2, 2).")
        heights = np.maximum(heights, ground_plane_m)
        tiles: Dict[Tuple[int, int, int], TerrainTile] = {}
        grid_rows, grid_cols = heights.shape
        cell_size_m = float(resolution_m)
        span_cells_y = grid_rows - 1
        span_cells_x = grid_cols - 1
        tile_cols = int(math.ceil(span_cells_x / tile_size_cells))
        tile_rows = int(math.ceil(span_cells_y / tile_size_cells))

        for ty in range(tile_rows):
            for tx in range(tile_cols):
                row_start = ty * tile_size_cells
                col_start = tx * tile_size_cells
                row_end = min(row_start + tile_size_cells, span_cells_y)
                col_end = min(col_start + tile_size_cells, span_cells_x)
                tile_heights = heights[row_start:row_end + 1, col_start:col_end + 1]
                target_shape = (tile_size_cells + 1, tile_size_cells + 1)
                if tile_heights.shape != target_shape:
                    padded = np.empty(target_shape, dtype=float)
                    padded[:tile_heights.shape[0], :tile_heights.shape[1]] = tile_heights
                    padded[tile_heights.shape[0]:, :tile_heights.shape[1]] = tile_heights[-1:, :]
                    padded[:, tile_heights.shape[1]:] = padded[:, tile_heights.shape[1] - 1:tile_heights.shape[1]]
                    tile_heights = padded
                x_min_m = bounds_xy_m.x_min_m + (tx * tile_size_cells * cell_size_m)
                y_min_m = bounds_xy_m.y_min_m + (ty * tile_size_cells * cell_size_m)
                tiles[(0, tx, ty)] = TerrainTile(
                    tx=tx,
                    ty=ty,
                    lod=0,
                    x_min_m=x_min_m,
                    y_min_m=y_min_m,
                    cell_size_m=cell_size_m,
                    heights_m=tile_heights,
                    min_pyramid=_build_pyramid(tile_heights, "min"),
                    max_pyramid=_build_pyramid(tile_heights, "max"),
                )

        return cls(
            bounds_xy_m=bounds_xy_m,
            tile_size_cells=tile_size_cells,
            base_resolution_m=cell_size_m,
            lod_resolutions_m=lod_resolutions_m,
            interpolation=interpolation,
            ground_plane_m=ground_plane_m,
            tiles=tiles,
            environment_id=environment_id,
        )

    @classmethod
    def from_analytic(
        cls,
        *,
        environment_id: str,
        terrain_model: object,
        bounds_xy_m: Bounds2D,
        resolution_m: float = 5.0,
        tile_size_cells: int = 256,
        lod_resolutions_m: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
    ) -> "TerrainLayer":
        x_values = np.arange(bounds_xy_m.x_min_m, bounds_xy_m.x_max_m + resolution_m, resolution_m, dtype=float)
        y_values = np.arange(bounds_xy_m.y_min_m, bounds_xy_m.y_max_m + resolution_m, resolution_m, dtype=float)
        heights = np.empty((len(y_values), len(x_values)), dtype=float)
        for row, y_m in enumerate(y_values):
            for col, x_m in enumerate(x_values):
                heights[row, col] = float(terrain_model.height_at(float(x_m), float(y_m)))
        return cls.from_height_grid(
            environment_id=environment_id,
            bounds_xy_m=bounds_xy_m,
            heights_m=heights,
            resolution_m=resolution_m,
            tile_size_cells=tile_size_cells,
            lod_resolutions_m=lod_resolutions_m,
            ground_plane_m=float(getattr(terrain_model, "ground_plane_m", 0.0)),
        )

    @classmethod
    def from_geotiff(
        cls,
        dem_path: str | Path,
        *,
        environment_id: Optional[str] = None,
        source_crs: Optional[str] = None,
        tile_size_cells: int = 256,
        lod_resolutions_m: Sequence[float] = (5.0, 10.0, 20.0, 40.0),
        interpolation: str = "bilinear",
        ground_plane_m: float = 0.0,
    ) -> "TerrainLayer":
        from ._scene_gis import project_dem_to_runtime

        raster = project_dem_to_runtime(dem_path, source_crs=source_crs)
        heights = np.asarray(raster.heights_m, dtype=float)
        x_values = np.asarray(raster.x_values_m, dtype=float)
        y_values = np.asarray(raster.y_values_m, dtype=float)

        if x_values[0] > x_values[-1]:
            x_values = x_values[::-1]
            heights = heights[:, ::-1]
        if y_values[0] > y_values[-1]:
            y_values = y_values[::-1]
            heights = heights[::-1, :]

        if x_values.size < 2 or y_values.size < 2:
            raise ValueError("Imported DEM must resolve to at least a 2x2 runtime grid.")

        resolution_candidates = [
            abs(float(x_values[1] - x_values[0])),
            abs(float(y_values[1] - y_values[0])),
        ]
        resolution_m = min(value for value in resolution_candidates if value > 0.0)
        bounds_xy_m = Bounds2D(
            x_min_m=float(np.min(x_values)),
            x_max_m=float(np.max(x_values)),
            y_min_m=float(np.min(y_values)),
            y_max_m=float(np.max(y_values)),
        )
        return cls.from_height_grid(
            environment_id=environment_id or Path(dem_path).stem,
            bounds_xy_m=bounds_xy_m,
            heights_m=heights,
            resolution_m=resolution_m,
            tile_size_cells=tile_size_cells,
            lod_resolutions_m=lod_resolutions_m,
            interpolation=interpolation,
            ground_plane_m=ground_plane_m,
        )

    def _tile_key_for_xy(self, x_m: float, y_m: float, lod: int = 0) -> Optional[Tuple[int, int, int]]:
        if not self.bounds_xy_m.contains_xy(x_m, y_m):
            return None
        span_m = self.tile_size_cells * self.base_resolution_m
        tx = min(int(math.floor((x_m - self.bounds_xy_m.x_min_m) / span_m)), int(math.ceil(self.bounds_xy_m.width_m / span_m)) - 1)
        ty = min(int(math.floor((y_m - self.bounds_xy_m.y_min_m) / span_m)), int(math.ceil(self.bounds_xy_m.height_m / span_m)) - 1)
        return (lod, tx, ty)

    def tile_for_xy(self, x_m: float, y_m: float, lod: int = 0) -> Optional[TerrainTile]:
        key = self._tile_key_for_xy(x_m, y_m, lod=lod)
        if key is None:
            return None
        return self._tiles.get(key)

    def covers_xy(self, x_m: float, y_m: float) -> bool:
        return self._tile_key_for_xy(x_m, y_m) in self._tiles

    def height_at(self, x_m: float, y_m: float) -> float:
        tile = self.tile_for_xy(float(x_m), float(y_m))
        if tile is None:
            return self.ground_plane_m
        local_x = (float(x_m) - tile.x_min_m) / tile.cell_size_m
        local_y = (float(y_m) - tile.y_min_m) / tile.cell_size_m
        col = int(np.clip(math.floor(local_x), 0, tile.heights_m.shape[1] - 2))
        row = int(np.clip(math.floor(local_y), 0, tile.heights_m.shape[0] - 2))
        tx = _clamp01(local_x - col)
        ty = _clamp01(local_y - row)
        z00 = tile.heights_m[row, col]
        z10 = tile.heights_m[row, col + 1]
        z01 = tile.heights_m[row + 1, col]
        z11 = tile.heights_m[row + 1, col + 1]
        z0 = z00 + (z10 - z00) * tx
        z1 = z01 + (z11 - z01) * tx
        return float(max(z0 + (z1 - z0) * ty, self.ground_plane_m))

    def gradient_at(self, x_m: float, y_m: float, delta_m: Optional[float] = None) -> np.ndarray:
        delta = max(delta_m or (self.base_resolution_m * 0.5), 0.25)
        dz_dx = (self.height_at(x_m + delta, y_m) - self.height_at(x_m - delta, y_m)) / (2.0 * delta)
        dz_dy = (self.height_at(x_m, y_m + delta) - self.height_at(x_m, y_m - delta)) / (2.0 * delta)
        return np.array([dz_dx, dz_dy], dtype=float)

    def curvature_at(self, x_m: float, y_m: float, delta_m: float = 1.0) -> float:
        """Scalar surface curvature (Laplacian approximation).

        Positive = convex (hilltop), negative = concave (valley).
        """
        d = max(delta_m, 0.1)
        return (
            self.height_at(x_m + d, y_m)
            + self.height_at(x_m - d, y_m)
            + self.height_at(x_m, y_m + d)
            + self.height_at(x_m, y_m - d)
            - 4.0 * self.height_at(x_m, y_m)
        ) / (d * d)

    def slope_rad_at(self, x_m: float, y_m: float, delta_m: float = 1.0) -> float:
        """Slope magnitude in radians at the given point."""
        gx, gy = self.gradient_at(x_m, y_m, delta_m)
        return math.atan(math.sqrt(gx * gx + gy * gy))

    def normal_at(self, x_m: float, y_m: float) -> np.ndarray:
        gradient = self.gradient_at(x_m, y_m)
        normal = np.array([-gradient[0], -gradient[1], 1.0], dtype=float)
        return normal / max(np.linalg.norm(normal), 1.0e-9)

    def clamp_altitude(self, xy_m: Sequence[float], z_m: float, min_agl_m: float) -> float:
        ground_m = self.height_at(float(xy_m[0]), float(xy_m[1]))
        return max(float(z_m), ground_m + max(float(min_agl_m), 0.0))

    def terrain_summary(self) -> Dict[str, object]:
        min_height = min(float(tile.min_height_m) for tile in self._tiles.values())
        max_height = max(float(tile.max_height_m) for tile in self._tiles.values())
        return {
            "kind": "tiled-heightmap-v1",
            "base_resolution_m": self.base_resolution_m,
            "tile_size_cells": self.tile_size_cells,
            "lod_resolutions_m": list(self.lod_resolutions_m),
            "ground_plane_m": self.ground_plane_m,
            "min_height_m": min_height,
            "max_height_m": max_height,
        }

    def viewer_mesh(self, max_dimension: int = 128) -> Dict[str, object]:
        cols = max(2, min(max_dimension, int(math.ceil(self.bounds_xy_m.width_m / self.base_resolution_m))))
        rows = max(2, min(max_dimension, int(math.ceil(self.bounds_xy_m.height_m / self.base_resolution_m))))
        x_values = np.linspace(self.bounds_xy_m.x_min_m, self.bounds_xy_m.x_max_m, num=cols, dtype=float)
        y_values = np.linspace(self.bounds_xy_m.y_min_m, self.bounds_xy_m.y_max_m, num=rows, dtype=float)
        heights = [[self.height_at(float(x_m), float(y_m)) for x_m in x_values] for y_m in y_values]
        return {
            "x_min_m": self.bounds_xy_m.x_min_m,
            "x_max_m": self.bounds_xy_m.x_max_m,
            "y_min_m": self.bounds_xy_m.y_min_m,
            "y_max_m": self.bounds_xy_m.y_max_m,
            "cols": cols,
            "rows": rows,
            "heights_m": heights,
        }

    def to_metadata(self) -> Dict[str, object]:
        summary = self.terrain_summary()
        summary["xy_bounds_m"] = self.bounds_xy_m.to_metadata()
        summary["viewer_mesh"] = self.viewer_mesh()
        return summary


@dataclass(frozen=True)
class LandCoverTile:
    tx: int
    ty: int
    x_min_m: float
    y_min_m: float
    cell_size_m: float
    classes: np.ndarray
    density: Optional[np.ndarray] = None

    @property
    def tile_size_cells(self) -> int:
        return self.classes.shape[0]


class LandCoverLayer:
    def __init__(
        self,
        *,
        bounds_xy_m: Bounds2D,
        tile_size_cells: int,
        base_resolution_m: float,
        tiles: Mapping[Tuple[int, int], LandCoverTile],
    ) -> None:
        self.bounds_xy_m = bounds_xy_m
        self.tile_size_cells = int(tile_size_cells)
        self.base_resolution_m = float(base_resolution_m)
        self._tiles = dict(tiles)

    @classmethod
    def from_rasters(
        cls,
        *,
        bounds_xy_m: Bounds2D,
        classes: np.ndarray,
        density: Optional[np.ndarray],
        resolution_m: float,
        tile_size_cells: int = 256,
    ) -> "LandCoverLayer":
        class_grid = np.asarray(classes, dtype=np.uint8)
        density_grid = None if density is None else np.asarray(density, dtype=np.uint8)
        tiles: Dict[Tuple[int, int], LandCoverTile] = {}
        rows, cols = class_grid.shape
        tile_cols = int(math.ceil(cols / tile_size_cells))
        tile_rows = int(math.ceil(rows / tile_size_cells))
        for ty in range(tile_rows):
            for tx in range(tile_cols):
                row_start = ty * tile_size_cells
                col_start = tx * tile_size_cells
                row_end = min(row_start + tile_size_cells, rows)
                col_end = min(col_start + tile_size_cells, cols)
                classes_tile = np.zeros((tile_size_cells, tile_size_cells), dtype=np.uint8)
                classes_tile[:row_end - row_start, :col_end - col_start] = class_grid[row_start:row_end, col_start:col_end]
                density_tile = None
                if density_grid is not None:
                    density_tile = np.zeros((tile_size_cells, tile_size_cells), dtype=np.uint8)
                    density_tile[:row_end - row_start, :col_end - col_start] = density_grid[row_start:row_end, col_start:col_end]
                tiles[(tx, ty)] = LandCoverTile(
                    tx=tx,
                    ty=ty,
                    x_min_m=bounds_xy_m.x_min_m + (tx * tile_size_cells * resolution_m),
                    y_min_m=bounds_xy_m.y_min_m + (ty * tile_size_cells * resolution_m),
                    cell_size_m=resolution_m,
                    classes=classes_tile,
                    density=density_tile,
                )
        return cls(bounds_xy_m=bounds_xy_m, tile_size_cells=tile_size_cells, base_resolution_m=resolution_m, tiles=tiles)

    @classmethod
    def open_terrain(
        cls,
        *,
        bounds_xy_m: Bounds2D,
        resolution_m: float,
        tile_size_cells: int = 256,
    ) -> "LandCoverLayer":
        cols = max(1, int(math.ceil(bounds_xy_m.width_m / resolution_m)))
        rows = max(1, int(math.ceil(bounds_xy_m.height_m / resolution_m)))
        classes = np.full((rows, cols), int(LandCoverClass.OPEN), dtype=np.uint8)
        density = np.zeros((rows, cols), dtype=np.uint8)
        return cls.from_rasters(bounds_xy_m=bounds_xy_m, classes=classes, density=density, resolution_m=resolution_m, tile_size_cells=tile_size_cells)

    def _tile_for_xy(self, x_m: float, y_m: float) -> Optional[LandCoverTile]:
        if not self.bounds_xy_m.contains_xy(x_m, y_m):
            return None
        span_m = self.tile_size_cells * self.base_resolution_m
        tx = min(int(math.floor((x_m - self.bounds_xy_m.x_min_m) / span_m)), int(math.ceil(self.bounds_xy_m.width_m / span_m)) - 1)
        ty = min(int(math.floor((y_m - self.bounds_xy_m.y_min_m) / span_m)), int(math.ceil(self.bounds_xy_m.height_m / span_m)) - 1)
        return self._tiles.get((tx, ty))

    def land_cover_at(self, x_m: float, y_m: float) -> LandCoverClass:
        tile = self._tile_for_xy(float(x_m), float(y_m))
        if tile is None:
            return LandCoverClass.OPEN
        local_x = int(np.clip(math.floor((float(x_m) - tile.x_min_m) / tile.cell_size_m), 0, tile.classes.shape[1] - 1))
        local_y = int(np.clip(math.floor((float(y_m) - tile.y_min_m) / tile.cell_size_m), 0, tile.classes.shape[0] - 1))
        return LandCoverClass(int(tile.classes[local_y, local_x]))

    def density_at(self, x_m: float, y_m: float) -> float:
        tile = self._tile_for_xy(float(x_m), float(y_m))
        if tile is None or tile.density is None:
            return 0.0
        local_x = int(np.clip(math.floor((float(x_m) - tile.x_min_m) / tile.cell_size_m), 0, tile.density.shape[1] - 1))
        local_y = int(np.clip(math.floor((float(y_m) - tile.y_min_m) / tile.cell_size_m), 0, tile.density.shape[0] - 1))
        return float(tile.density[local_y, local_x]) / 255.0

    def to_metadata(self) -> Dict[str, object]:
        return {
            "kind": "tiled-landcover-v1",
            "base_resolution_m": self.base_resolution_m,
            "tile_size_cells": self.tile_size_cells,
            "xy_bounds_m": self.bounds_xy_m.to_metadata(),
            "legend": LandCoverClass.legend(),
        }


from .obstacles import (
    _as_float_array,
    _point_in_polygon,
    _point_on_segment,
    _segment_intersection_parameters,
    _segment_polygon_intervals,
    _unique_sorted,
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    ObstaclePrimitive,
    OrientedBox,
    PolygonPrism,
    WallSegment,
)


class ObstacleLayer:
    def __init__(
        self,
        *,
        bounds_xy_m: Bounds2D,
        tile_size_m: float,
        primitives: Sequence[ObstaclePrimitive],
    ) -> None:
        self.bounds_xy_m = bounds_xy_m
        self.tile_size_m = float(tile_size_m)
        self.primitives = tuple(primitives)
        self._primitive_by_id = {primitive.primitive_id: primitive for primitive in self.primitives}
        tile_index: Dict[Tuple[int, int], List[str]] = {}
        for primitive in self.primitives:
            bounds = primitive.bounds_xy_m()
            tx_min = int(math.floor((bounds.x_min_m - self.bounds_xy_m.x_min_m) / self.tile_size_m))
            tx_max = int(math.floor((bounds.x_max_m - self.bounds_xy_m.x_min_m) / self.tile_size_m))
            ty_min = int(math.floor((bounds.y_min_m - self.bounds_xy_m.y_min_m) / self.tile_size_m))
            ty_max = int(math.floor((bounds.y_max_m - self.bounds_xy_m.y_min_m) / self.tile_size_m))
            for ty in range(ty_min, ty_max + 1):
                for tx in range(tx_min, tx_max + 1):
                    tile_index.setdefault((tx, ty), []).append(primitive.primitive_id)
        self._tile_index = {
            key: tuple(sorted(set(primitive_ids)))
            for key, primitive_ids in tile_index.items()
        }

    @classmethod
    def empty(cls, *, bounds_xy_m: Bounds2D, tile_size_m: float) -> "ObstacleLayer":
        return cls(bounds_xy_m=bounds_xy_m, tile_size_m=tile_size_m, primitives=())

    def query_obstacles(self, segment_bounds: Bounds2D) -> Tuple[ObstaclePrimitive, ...]:
        tx_min = int(math.floor((segment_bounds.x_min_m - self.bounds_xy_m.x_min_m) / self.tile_size_m))
        tx_max = int(math.floor((segment_bounds.x_max_m - self.bounds_xy_m.x_min_m) / self.tile_size_m))
        ty_min = int(math.floor((segment_bounds.y_min_m - self.bounds_xy_m.y_min_m) / self.tile_size_m))
        ty_max = int(math.floor((segment_bounds.y_max_m - self.bounds_xy_m.y_min_m) / self.tile_size_m))
        primitive_ids: List[str] = []
        for ty in range(ty_min, ty_max + 1):
            for tx in range(tx_min, tx_max + 1):
                primitive_ids.extend(self._tile_index.get((tx, ty), ()))
        return tuple(self._primitive_by_id[primitive_id] for primitive_id in sorted(set(primitive_ids)))

    def point_collides(self, x_m: float, y_m: float, z_m: float) -> Optional[ObstaclePrimitive]:
        epsilon_m = 1.0e-6
        candidate_bounds = Bounds2D(
            x_min_m=float(x_m) - epsilon_m,
            x_max_m=float(x_m) + epsilon_m,
            y_min_m=float(y_m) - epsilon_m,
            y_max_m=float(y_m) + epsilon_m,
        )
        for primitive in self.query_obstacles(candidate_bounds):
            if primitive.blocker_type not in {"building", "wall"}:
                continue
            if primitive.point_inside(float(x_m), float(y_m), float(z_m)):
                return primitive
        return None

    def to_metadata(self) -> List[Dict[str, object]]:
        return [primitive.to_metadata() for primitive in self.primitives]


from .visibility import (
    DetectionResult,
    EnvironmentQuery,
    SensorVisibilityModel,
    VisibilityResult,
    compute_effective_noise,
    compute_weather_factor,
    free_space_path_loss,
    identify_dominant_loss,
    _grid_dda_intervals,
    _liang_barsky_interval,
)


@dataclass
class EnvironmentModel:
    environment_id: str
    crs: EnvironmentCRS
    bounds_xy_m: Bounds2D
    terrain: TerrainLayer
    obstacles: ObstacleLayer
    land_cover: LandCoverLayer
    query: EnvironmentQuery = field(init=False)

    def __post_init__(self) -> None:
        self.query = EnvironmentQuery(self)

    @classmethod
    def from_legacy(
        cls,
        *,
        environment_id: str,
        bounds_xy_m: Bounds2D,
        terrain_model: object,
        occluding_objects: Sequence[object] = (),
        terrain_resolution_m: float = 5.0,
    ) -> "EnvironmentModel":
        terrain = terrain_model if isinstance(terrain_model, TerrainLayer) else TerrainLayer.from_analytic(
            environment_id=environment_id,
            terrain_model=terrain_model,
            bounds_xy_m=bounds_xy_m,
            resolution_m=terrain_resolution_m,
        )
        primitives: List[ObstaclePrimitive] = []
        for object_index, occluder in enumerate(occluding_objects):
            if isinstance(occluder, ObstaclePrimitive):
                primitives.append(occluder)
                continue
            base_z = float(occluder.base_elevation_m(terrain_model))
            top_z = float(occluder.top_elevation_m(terrain_model))
            primitives.append(
                CylinderObstacle(
                    primitive_id=getattr(occluder, "object_id", f"cylinder-{object_index}"),
                    blocker_type="building",
                    center_x_m=float(occluder.center_x_m),
                    center_y_m=float(occluder.center_y_m),
                    radius_m=float(occluder.radius_m),
                    base_z_m=base_z,
                    top_z_m=top_z,
                )
            )
        obstacles = ObstacleLayer(
            bounds_xy_m=bounds_xy_m,
            tile_size_m=terrain.tile_size_cells * terrain.base_resolution_m,
            primitives=primitives,
        )
        land_cover = LandCoverLayer.open_terrain(
            bounds_xy_m=bounds_xy_m,
            resolution_m=max(terrain.base_resolution_m * 2.0, 10.0),
        )
        return cls(
            environment_id=environment_id,
            crs=EnvironmentCRS(),
            bounds_xy_m=bounds_xy_m,
            terrain=terrain,
            obstacles=obstacles,
            land_cover=land_cover,
        )

    def to_replay_metadata(self) -> Dict[str, object]:
        return {
            "environment_id": self.environment_id,
            "crs_id": self.crs.runtime_crs_id,
            "environment_bounds_m": self.bounds_xy_m.to_metadata(),
            "terrain_summary": self.terrain.terrain_summary(),
            "land_cover_legend": LandCoverClass.legend(),
            "terrain": self.terrain.to_metadata(),
            "occluding_objects": self.obstacles.to_metadata(),
        }


from .environment_io import _array_from_blob, _array_to_blob, load_environment_bundle, write_environment_bundle


__all__ = [
    "_array_from_blob",
    "_array_to_blob",
    "_as_float_array",
    "_clamp01",
    "_grid_dda_intervals",
    "_liang_barsky_interval",
    "_point_in_polygon",
    "_point_on_segment",
    "_segment_intersection_parameters",
    "_segment_polygon_intervals",
    "_unique_sorted",
    "Bounds2D",
    "BuildingPrism",
    "CylinderObstacle",
    "DetectionResult",
    "EnvironmentCRS",
    "EnvironmentModel",
    "EnvironmentQuery",
    "ForestStand",
    "LandCoverClass",
    "LandCoverLayer",
    "LandCoverTile",
    "ObstacleLayer",
    "ObstaclePrimitive",
    "OrientedBox",
    "PolygonPrism",
    "SeasonalVariation",
    "SensorVisibilityModel",
    "TerrainLayer",
    "TerrainTile",
    "VisibilityResult",
    "WallSegment",
    "load_environment_bundle",
    "write_environment_bundle",
]
