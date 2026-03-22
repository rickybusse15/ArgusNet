from __future__ import annotations

import json
import math
import zlib
from pathlib import Path
from typing import Dict, Mapping, Tuple, List

import numpy as np

from .environment import (
    Bounds2D,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverClass,
    LandCoverLayer,
    LandCoverTile,
    ObstacleLayer,
    TerrainLayer,
    TerrainTile,
)
from .obstacles import BuildingPrism, CylinderObstacle, ForestStand, OrientedBox, PolygonPrism, WallSegment


def _array_to_blob(array: np.ndarray) -> Dict[str, object]:
    raw = np.ascontiguousarray(array).tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "compression": "zlib",
        "payload_hex": zlib.compress(raw).hex(),
    }


def _array_from_blob(blob: Mapping[str, object]) -> np.ndarray:
    raw = zlib.decompress(bytes.fromhex(str(blob["payload_hex"])))
    return np.frombuffer(raw, dtype=np.dtype(str(blob["dtype"]))).reshape(tuple(int(value) for value in blob["shape"]))


def write_environment_bundle(path: str, environment: EnvironmentModel) -> None:
    bundle_root = Path(path)
    bundle_root.mkdir(parents=True, exist_ok=True)
    (bundle_root / "terrain" / "0").mkdir(parents=True, exist_ok=True)
    (bundle_root / "landcover" / "0").mkdir(parents=True, exist_ok=True)
    (bundle_root / "obstacles").mkdir(parents=True, exist_ok=True)

    terrain_tiles = []
    for (lod, tx, ty), tile in sorted(environment.terrain._tiles.items()):
        payload = {
            "x_min_m": tile.x_min_m,
            "y_min_m": tile.y_min_m,
            "cell_size_m": tile.cell_size_m,
            "heights": _array_to_blob(tile.heights_m),
            "min_pyramid": [_array_to_blob(level) for level in tile.min_pyramid],
            "max_pyramid": [_array_to_blob(level) for level in tile.max_pyramid],
        }
        relative_path = Path("terrain") / str(lod) / f"{tx}_{ty}.json"
        (bundle_root / relative_path).write_text(json.dumps(payload), encoding="utf-8")
        terrain_tiles.append({"lod": lod, "tx": tx, "ty": ty, "path": str(relative_path)})

    landcover_tiles = []
    for (tx, ty), tile in sorted(environment.land_cover._tiles.items()):
        payload = {
            "x_min_m": tile.x_min_m,
            "y_min_m": tile.y_min_m,
            "cell_size_m": tile.cell_size_m,
            "classes": _array_to_blob(tile.classes),
            "density": None if tile.density is None else _array_to_blob(tile.density),
        }
        relative_path = Path("landcover") / "0" / f"{tx}_{ty}.json"
        (bundle_root / relative_path).write_text(json.dumps(payload), encoding="utf-8")
        landcover_tiles.append({"tx": tx, "ty": ty, "path": str(relative_path)})

    obstacle_tiles: Dict[Tuple[int, int], List[Dict[str, object]]] = {}
    for primitive in environment.obstacles.primitives:
        bounds = primitive.bounds_xy_m()
        tx = int(math.floor((bounds.x_min_m - environment.bounds_xy_m.x_min_m) / environment.obstacles.tile_size_m))
        ty = int(math.floor((bounds.y_min_m - environment.bounds_xy_m.y_min_m) / environment.obstacles.tile_size_m))
        obstacle_tiles.setdefault((tx, ty), []).append(primitive.to_metadata())
    obstacle_entries = []
    for (tx, ty), metadata in sorted(obstacle_tiles.items()):
        relative_path = Path("obstacles") / f"{tx}_{ty}.json"
        (bundle_root / relative_path).write_text(json.dumps(metadata), encoding="utf-8")
        obstacle_entries.append({"tx": tx, "ty": ty, "path": str(relative_path)})

    manifest = {
        "format_version": "smartmap-v1",
        "environment_id": environment.environment_id,
        "crs": environment.crs.to_metadata(),
        "bounds_xy_m": environment.bounds_xy_m.to_metadata(),
        "terrain": {
            "base_resolution_m": environment.terrain.base_resolution_m,
            "tile_size_cells": environment.terrain.tile_size_cells,
            "lod_resolutions_m": list(environment.terrain.lod_resolutions_m),
            "interpolation": environment.terrain.interpolation,
            "ground_plane_m": environment.terrain.ground_plane_m,
            "tiles": terrain_tiles,
        },
        "land_cover": {
            "base_resolution_m": environment.land_cover.base_resolution_m,
            "tile_size_cells": environment.land_cover.tile_size_cells,
            "legend": LandCoverClass.legend(),
            "tiles": landcover_tiles,
        },
        "obstacles": {
            "tile_size_m": environment.obstacles.tile_size_m,
            "tiles": obstacle_entries,
        },
    }
    (bundle_root / "environment_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_environment_bundle(path: str) -> EnvironmentModel:
    bundle_root = Path(path)
    manifest = json.loads((bundle_root / "environment_manifest.json").read_text(encoding="utf-8"))
    bounds_xy_m = Bounds2D.from_mapping(manifest["bounds_xy_m"])

    terrain_manifest = manifest["terrain"]
    terrain_tiles: Dict[Tuple[int, int, int], TerrainTile] = {}
    for tile_meta in terrain_manifest["tiles"]:
        payload = json.loads((bundle_root / tile_meta["path"]).read_text(encoding="utf-8"))
        terrain_tiles[(int(tile_meta["lod"]), int(tile_meta["tx"]), int(tile_meta["ty"]))] = TerrainTile(
            tx=int(tile_meta["tx"]),
            ty=int(tile_meta["ty"]),
            lod=int(tile_meta["lod"]),
            x_min_m=float(payload["x_min_m"]),
            y_min_m=float(payload["y_min_m"]),
            cell_size_m=float(payload["cell_size_m"]),
            heights_m=_array_from_blob(payload["heights"]),
            min_pyramid=tuple(_array_from_blob(blob) for blob in payload["min_pyramid"]),
            max_pyramid=tuple(_array_from_blob(blob) for blob in payload["max_pyramid"]),
        )
    terrain = TerrainLayer(
        bounds_xy_m=bounds_xy_m,
        tile_size_cells=int(terrain_manifest["tile_size_cells"]),
        base_resolution_m=float(terrain_manifest["base_resolution_m"]),
        lod_resolutions_m=terrain_manifest["lod_resolutions_m"],
        interpolation=str(terrain_manifest["interpolation"]),
        ground_plane_m=float(terrain_manifest["ground_plane_m"]),
        tiles=terrain_tiles,
        environment_id=str(manifest["environment_id"]),
    )

    land_cover_manifest = manifest["land_cover"]
    land_cover_tiles: Dict[Tuple[int, int], LandCoverTile] = {}
    for tile_meta in land_cover_manifest["tiles"]:
        payload = json.loads((bundle_root / tile_meta["path"]).read_text(encoding="utf-8"))
        land_cover_tiles[(int(tile_meta["tx"]), int(tile_meta["ty"]))] = LandCoverTile(
            tx=int(tile_meta["tx"]),
            ty=int(tile_meta["ty"]),
            x_min_m=float(payload["x_min_m"]),
            y_min_m=float(payload["y_min_m"]),
            cell_size_m=float(payload["cell_size_m"]),
            classes=_array_from_blob(payload["classes"]).astype(np.uint8),
            density=None if payload["density"] is None else _array_from_blob(payload["density"]).astype(np.uint8),
        )
    land_cover = LandCoverLayer(
        bounds_xy_m=bounds_xy_m,
        tile_size_cells=int(land_cover_manifest["tile_size_cells"]),
        base_resolution_m=float(land_cover_manifest["base_resolution_m"]),
        tiles=land_cover_tiles,
    )

    obstacle_manifest = manifest["obstacles"]
    primitives = []
    for tile_meta in obstacle_manifest["tiles"]:
        payload = json.loads((bundle_root / tile_meta["path"]).read_text(encoding="utf-8"))
        for primitive_meta in payload:
            kind = primitive_meta["kind"]
            if kind == "cylinder-v1":
                primitives.append(
                    CylinderObstacle(
                        primitive_id=primitive_meta["object_id"],
                        blocker_type=primitive_meta["blocker_type"],
                        center_x_m=float(primitive_meta["center_x_m"]),
                        center_y_m=float(primitive_meta["center_y_m"]),
                        radius_m=float(primitive_meta["radius_m"]),
                        base_z_m=float(primitive_meta["base_elevation_m"]),
                        top_z_m=float(primitive_meta["top_elevation_m"]),
                    )
                )
            elif kind == "wall-v1":
                primitives.append(
                    WallSegment(
                        primitive_id=primitive_meta["object_id"],
                        blocker_type=primitive_meta["blocker_type"],
                        start_xy_m=np.asarray(primitive_meta["start_xy_m"], dtype=float),
                        end_xy_m=np.asarray(primitive_meta["end_xy_m"], dtype=float),
                        thickness_m=float(primitive_meta["thickness_m"]),
                        base_z_m=float(primitive_meta["base_elevation_m"]),
                        top_z_m=float(primitive_meta["top_elevation_m"]),
                    )
                )
            elif kind == "box-v1":
                primitives.append(
                    OrientedBox(
                        primitive_id=primitive_meta["object_id"],
                        blocker_type=primitive_meta["blocker_type"],
                        center_x_m=float(primitive_meta["center_x_m"]),
                        center_y_m=float(primitive_meta["center_y_m"]),
                        length_m=float(primitive_meta["length_m"]),
                        width_m=float(primitive_meta["width_m"]),
                        yaw_rad=float(primitive_meta["yaw_rad"]),
                        base_z_m=float(primitive_meta["base_elevation_m"]),
                        top_z_m=float(primitive_meta["top_elevation_m"]),
                    )
                )
            elif kind == "forest-stand-v1":
                primitives.append(
                    ForestStand(
                        primitive_id=primitive_meta["object_id"],
                        footprint_xy_m=np.asarray(primitive_meta["footprint_xy_m"], dtype=float),
                        canopy_base_z_m=float(primitive_meta["base_elevation_m"]),
                        canopy_top_z_m=float(primitive_meta["top_elevation_m"]),
                        density=float(primitive_meta.get("density", 0.35)),
                    )
                )
            elif primitive_meta["blocker_type"] == "building":
                primitives.append(
                    BuildingPrism(
                        primitive_id=primitive_meta["object_id"],
                        footprint_xy_m=np.asarray(primitive_meta["footprint_xy_m"], dtype=float),
                        base_z_m=float(primitive_meta["base_elevation_m"]),
                        top_z_m=float(primitive_meta["top_elevation_m"]),
                    )
                )
            else:
                primitives.append(
                    PolygonPrism(
                        primitive_id=primitive_meta["object_id"],
                        blocker_type=primitive_meta["blocker_type"],
                        footprint_xy_m=np.asarray(primitive_meta["footprint_xy_m"], dtype=float),
                        base_z_m=float(primitive_meta["base_elevation_m"]),
                        top_z_m=float(primitive_meta["top_elevation_m"]),
                    )
                )

    obstacles = ObstacleLayer(
        bounds_xy_m=bounds_xy_m,
        tile_size_m=float(obstacle_manifest["tile_size_m"]),
        primitives=primitives,
    )
    crs = EnvironmentCRS(
        source_crs_id=manifest["crs"]["source_crs_id"],
        runtime_crs_id=manifest["crs"]["runtime_crs_id"],
        origin_lat_deg=manifest["crs"]["origin_geodetic"]["lat_deg"],
        origin_lon_deg=manifest["crs"]["origin_geodetic"]["lon_deg"],
        origin_h_m=manifest["crs"]["origin_geodetic"]["h_m"],
        xy_units=manifest["crs"]["xy_units"],
        z_datum=manifest["crs"]["z_datum"],
    )
    return EnvironmentModel(
        environment_id=str(manifest["environment_id"]),
        crs=crs,
        bounds_xy_m=bounds_xy_m,
        terrain=terrain,
        obstacles=obstacles,
        land_cover=land_cover,
    )


__all__ = [
    "_array_from_blob",
    "_array_to_blob",
    "load_environment_bundle",
    "write_environment_bundle",
]
