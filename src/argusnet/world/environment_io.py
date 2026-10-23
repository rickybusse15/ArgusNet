from __future__ import annotations

import json
import math
import zlib
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from argusnet.core.io_limits import read_text_capped, resolve_contained_path

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
from .obstacles import (
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    OrientedBox,
    PolygonPrism,
    WallSegment,
)

# The environment bundle format this module writes and reads.
ENVIRONMENT_BUNDLE_FORMAT = "smartmap-v1"

# A shared bundle is untrusted. Cap how many tile files one manifest can name,
# and how far one zlib blob may expand, before either becomes a file-read or
# memory bomb. A legitimate tile is a few hundred kilobytes.
_MAX_BUNDLE_TILES = 16_384
_MAX_DECOMPRESSED_BLOB_BYTES = 64 * 1024 * 1024
_ALLOWED_BLOB_DTYPES = frozenset(
    {
        "float32",
        "float64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "int8",
        "int16",
        "int32",
        "int64",
        "bool",
    }
)


def _array_to_blob(array: np.ndarray) -> dict[str, object]:
    raw = np.ascontiguousarray(array).tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "compression": "zlib",
        "payload_hex": zlib.compress(raw).hex(),
    }


def _array_from_blob(blob: Mapping[str, object]) -> np.ndarray:
    """Decode one bundle array, refusing bombs and non-numeric dtypes.

    The blob is attacker-controlled when the bundle comes from a shared scene.
    ``zlib.decompress`` without a length cap expands a few kilobytes into
    gigabytes, and ``np.dtype`` accepts object and sub-array dtypes that are
    not height or class grids.
    """
    if blob.get("compression", "zlib") != "zlib":
        raise ValueError("environment bundle array compression must be zlib")
    payload_hex = blob.get("payload_hex")
    if not isinstance(payload_hex, str) or not payload_hex:
        raise ValueError("environment bundle array payload_hex must be a non-empty string")
    if len(payload_hex) > _MAX_DECOMPRESSED_BLOB_BYTES * 2:
        raise ValueError("environment bundle array payload exceeds the size cap")
    dtype_name = blob.get("dtype")
    if not isinstance(dtype_name, str) or dtype_name not in _ALLOWED_BLOB_DTYPES:
        raise ValueError(f"environment bundle array dtype is not allowed: {dtype_name!r}")
    shape_raw = blob.get("shape")
    if not isinstance(shape_raw, (list, tuple)) or not shape_raw:
        raise ValueError("environment bundle array shape must be a non-empty list")
    shape: list[int] = []
    for dim in shape_raw:
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 0:
            raise ValueError("environment bundle array shape must be non-negative integers")
        shape.append(dim)
    dtype = np.dtype(dtype_name)
    expected = int(dtype.itemsize)
    for dim in shape:
        if expected > _MAX_DECOMPRESSED_BLOB_BYTES or dim > _MAX_DECOMPRESSED_BLOB_BYTES:
            raise ValueError("environment bundle array shape exceeds the size cap")
        expected *= dim
        if expected > _MAX_DECOMPRESSED_BLOB_BYTES:
            raise ValueError("environment bundle array shape exceeds the size cap")
    try:
        compressed = bytes.fromhex(payload_hex)
    except ValueError as exc:
        raise ValueError("environment bundle array payload_hex is not hex") from exc
    # zlib.decompress has no output cap. decompressobj(max_length) stops early;
    # a limit of 0 means unlimited, so an empty shape still gets a positive cap.
    decompressor = zlib.decompressobj()
    try:
        raw = decompressor.decompress(compressed, max_length=expected if expected else 1)
    except zlib.error as exc:
        raise ValueError(
            "environment bundle array failed to decompress within the size cap"
        ) from exc
    if decompressor.unconsumed_tail or not decompressor.eof:
        raise ValueError("environment bundle array failed to decompress within the size cap")
    if len(raw) != expected:
        raise ValueError("environment bundle array shape does not match its payload")
    return np.frombuffer(raw, dtype=dtype).reshape(tuple(shape))


def _bundle_json(bundle_root: Path, relative: object, *, field_name: str) -> object:
    if not isinstance(relative, str):
        raise ValueError(f"{field_name} must be a string path")
    path = resolve_contained_path(bundle_root, relative, field_name=field_name)
    return json.loads(read_text_capped(path))


def _tile_list(section: object, name: str) -> list[object]:
    if not isinstance(section, Mapping):
        raise ValueError(f"environment bundle {name} must be a mapping")
    tiles = section.get("tiles")
    if not isinstance(tiles, list):
        raise ValueError(f"environment bundle {name}.tiles must be a list")
    if len(tiles) > _MAX_BUNDLE_TILES:
        raise ValueError(
            f"environment bundle {name} has {len(tiles)} tiles, over the {_MAX_BUNDLE_TILES} cap"
        )
    return tiles


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

    obstacle_tiles: dict[tuple[int, int], list[dict[str, object]]] = {}
    for primitive in environment.obstacles.primitives:
        bounds = primitive.bounds_xy_m()
        tx = int(
            math.floor(
                (bounds.x_min_m - environment.bounds_xy_m.x_min_m)
                / environment.obstacles.tile_size_m
            )
        )
        ty = int(
            math.floor(
                (bounds.y_min_m - environment.bounds_xy_m.y_min_m)
                / environment.obstacles.tile_size_m
            )
        )
        obstacle_tiles.setdefault((tx, ty), []).append(primitive.to_metadata())
    obstacle_entries = []
    for (tx, ty), metadata in sorted(obstacle_tiles.items()):
        relative_path = Path("obstacles") / f"{tx}_{ty}.json"
        (bundle_root / relative_path).write_text(json.dumps(metadata), encoding="utf-8")
        obstacle_entries.append({"tx": tx, "ty": ty, "path": str(relative_path)})

    manifest = {
        "format_version": ENVIRONMENT_BUNDLE_FORMAT,
        "environment_id": environment.environment_id,
        "crs": environment.crs.to_metadata(),
        "bounds_xy_m": environment.bounds_xy_m.to_metadata(),
        "terrain": {
            "base_resolution_m": environment.terrain.base_resolution_m,
            "tile_size_cells": environment.terrain.tile_size_cells,
            "lod_resolutions_m": list(environment.terrain.lod_resolutions_m),
            "interpolation": environment.terrain.interpolation,
            "ground_plane_m": environment.terrain.ground_plane_m,
            "source": dict(environment.terrain.source_metadata),
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
    (bundle_root / "environment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def load_environment_bundle(path: str) -> EnvironmentModel:
    bundle_root = Path(path)
    manifest_path = resolve_contained_path(
        bundle_root,
        "environment_manifest.json",
        field_name="environment_manifest.json",
    )
    manifest = json.loads(read_text_capped(manifest_path))
    declared = manifest.get("format_version")
    if declared != ENVIRONMENT_BUNDLE_FORMAT:
        raise ValueError(
            f"unsupported environment bundle format {declared!r}; "
            f"this reader reads {ENVIRONMENT_BUNDLE_FORMAT!r}"
        )
    bounds_xy_m = Bounds2D.from_mapping(manifest["bounds_xy_m"])

    terrain_manifest = manifest["terrain"]
    terrain_tiles: dict[tuple[int, int, int], TerrainTile] = {}
    for tile_meta in _tile_list(terrain_manifest, "terrain"):
        if not isinstance(tile_meta, Mapping):
            raise ValueError("environment bundle terrain tile must be a mapping")
        payload = _bundle_json(bundle_root, tile_meta.get("path"), field_name="terrain tile path")
        terrain_tiles[(int(tile_meta["lod"]), int(tile_meta["tx"]), int(tile_meta["ty"]))] = (
            TerrainTile(
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
        source_metadata=terrain_manifest.get("source", {"source": "bundle"}),
    )

    land_cover_manifest = manifest["land_cover"]
    land_cover_tiles: dict[tuple[int, int], LandCoverTile] = {}
    for tile_meta in _tile_list(land_cover_manifest, "land_cover"):
        if not isinstance(tile_meta, Mapping):
            raise ValueError("environment bundle land-cover tile must be a mapping")
        payload = _bundle_json(
            bundle_root, tile_meta.get("path"), field_name="land_cover tile path"
        )
        land_cover_tiles[(int(tile_meta["tx"]), int(tile_meta["ty"]))] = LandCoverTile(
            tx=int(tile_meta["tx"]),
            ty=int(tile_meta["ty"]),
            x_min_m=float(payload["x_min_m"]),
            y_min_m=float(payload["y_min_m"]),
            cell_size_m=float(payload["cell_size_m"]),
            classes=_array_from_blob(payload["classes"]).astype(np.uint8),
            density=None
            if payload["density"] is None
            else _array_from_blob(payload["density"]).astype(np.uint8),
        )
    land_cover = LandCoverLayer(
        bounds_xy_m=bounds_xy_m,
        tile_size_cells=int(land_cover_manifest["tile_size_cells"]),
        base_resolution_m=float(land_cover_manifest["base_resolution_m"]),
        tiles=land_cover_tiles,
    )

    obstacle_manifest = manifest["obstacles"]
    primitives = []
    for tile_meta in _tile_list(obstacle_manifest, "obstacles"):
        if not isinstance(tile_meta, Mapping):
            raise ValueError("environment bundle obstacle tile must be a mapping")
        payload = _bundle_json(bundle_root, tile_meta.get("path"), field_name="obstacle tile path")
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
