from __future__ import annotations

import json
import math
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from argusnet.core.types import to_jsonable
from argusnet.evaluation.replay import (
    ReplayDocument,
    load_replay_document,
    validate_replay_document,
)

from ._glb import merge_meshes
from ._scene_geometry import (
    SceneLayerMesh,
    chunked_height_grid_layers,
    height_function_from_mesh,
    mesh_from_extrusion,
    mesh_from_polygon,
    mesh_from_polyline,
    terrain_mesh_from_viewer_mesh,
    write_scene_layers,
)
from ._scene_gis import (
    geojson_geometry_to_runtime,
    load_geojson_layer,
    project_dem_to_runtime,
)
from ._scene_style import STYLE_FORMAT_VERSION, default_style, style_document
from .environment import Bounds2D, EnvironmentCRS, LandCoverClass
from .environment_io import load_environment_bundle

SCENE_FORMAT_VERSION = "smartscene-v1"
GROUND_CONTACT_TOP_PAD_M = 3.0


def _as_bounds(mapping: Mapping[str, object]) -> Bounds2D:
    return Bounds2D.from_mapping(mapping)


def _write_json(path: Path, document: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")


def _normalize_scene_id(candidate: str) -> str:
    cleaned = candidate.strip().lower().replace(" ", "-").replace("_", "-")
    normalized = "".join(ch for ch in cleaned if ch.isalnum() or ch == "-")
    return normalized or "scene"


def _runtime_crs_metadata(crs: EnvironmentCRS) -> dict[str, object]:
    return crs.to_metadata()


def _terrain_height_span_under_footprint(footprint: np.ndarray, height_at) -> tuple[float, float]:
    heights = [float(height_at(float(vertex[0]), float(vertex[1]))) for vertex in footprint]
    return min(heights), max(heights)


def _solid_mesh_elevations(
    object_meta: Mapping[str, object],
    min_terrain_z: float,
    max_terrain_z: float,
) -> tuple[float, float]:
    requested_base_z = float(object_meta.get("base_elevation_m", min_terrain_z))
    requested_top_z = float(
        object_meta.get(
            "top_elevation_m",
            requested_base_z
            + max(float(object_meta.get("height_agl_m", 12.0)), GROUND_CONTACT_TOP_PAD_M),
        )
    )
    base_z = min(requested_base_z, min_terrain_z)
    top_z = max(requested_top_z, max_terrain_z + GROUND_CONTACT_TOP_PAD_M)
    return base_z, top_z


def validate_scene_manifest(document: Mapping[str, object]) -> None:
    if document.get("format_version") != SCENE_FORMAT_VERSION:
        raise ValueError("Scene manifest format_version must be smartscene-v1.")
    if not isinstance(document.get("scene_id"), str) or not str(document.get("scene_id")).strip():
        raise ValueError("Scene manifest scene_id must be a non-empty string.")
    bounds = document.get("bounds_xy_m")
    if not isinstance(bounds, Mapping):
        raise ValueError("Scene manifest must include bounds_xy_m.")
    _as_bounds(bounds)
    layers = document.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError("Scene manifest must include a non-empty layers list.")
    metadata = document.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Scene manifest must include metadata.")
    if not isinstance(metadata.get("environment"), str) or not isinstance(
        metadata.get("style"), str
    ):
        raise ValueError("Scene manifest metadata must include environment and style paths.")
    for layer in layers:
        if not isinstance(layer, Mapping):
            raise ValueError("Scene manifest layers must be mappings.")
        if not isinstance(layer.get("id"), str) or not isinstance(layer.get("asset_path"), str):
            raise ValueError("Each scene manifest layer must include id and asset_path.")
        if not isinstance(layer.get("style_id"), str):
            raise ValueError("Each scene manifest layer must include style_id.")


def load_scene_manifest(path: str | Path) -> dict[str, object]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_scene_manifest(document)
    return document


def _load_replay(path_or_document: str | Path | ReplayDocument | None) -> ReplayDocument | None:
    if path_or_document is None:
        return None
    if isinstance(path_or_document, Mapping):
        validate_replay_document(path_or_document)
        return dict(path_or_document)
    return load_replay_document(str(path_or_document))


def _copy_replay_document(output_dir: Path, replay_document: ReplayDocument | None) -> str | None:
    if replay_document is None:
        return None
    replay_path = output_dir / "replay" / "replay.json"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text(json.dumps(replay_document, indent=2), encoding="utf-8")
    return str(Path("replay") / "replay.json")


def _build_manifest(
    *,
    scene_id: str,
    bounds_xy_m: Bounds2D,
    runtime_crs: Mapping[str, object],
    source_crs_id: str,
    layers: Sequence[dict[str, object]],
    replay_path: str | None,
    source_kind: str,
    environment_path: str,
    style_path: str,
    provenance: Mapping[str, object],
) -> dict[str, object]:
    manifest = {
        "format_version": SCENE_FORMAT_VERSION,
        "scene_id": scene_id,
        "bounds_xy_m": bounds_xy_m.to_metadata(),
        "runtime_crs": dict(runtime_crs),
        "source_crs_id": source_crs_id,
        "layers": list(layers),
        "replay": {"path": replay_path} if replay_path else None,
        "metadata": {
            "environment": environment_path,
            "style": style_path,
        },
        "build": {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_kind": source_kind,
            **dict(provenance),
        },
    }
    validate_scene_manifest(manifest)
    return manifest


def _build_landcover_meshes_from_environment(environment) -> Sequence[SceneLayerMesh]:
    class_styles = {
        int(LandCoverClass.OPEN): "landcover-open",
        int(LandCoverClass.FOREST): "landcover-forest",
        int(LandCoverClass.WATER): "landcover-water",
        int(LandCoverClass.URBAN): "landcover-urban",
    }
    grouped = defaultdict(list)
    style = default_style()

    for tile in environment.land_cover._tiles.values():
        rows, cols = tile.classes.shape
        for row in range(rows):
            for col in range(cols):
                class_id = int(tile.classes[row, col])
                if class_id == int(LandCoverClass.OPEN):
                    continue
                style_id = class_styles.get(class_id)
                if style_id is None:
                    continue
                x0 = tile.x_min_m + (col * tile.cell_size_m)
                x1 = x0 + tile.cell_size_m
                y0 = tile.y_min_m + (row * tile.cell_size_m)
                y1 = y0 + tile.cell_size_m
                quad = np.asarray([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float32)
                grouped[style_id].append(
                    mesh_from_polygon(
                        quad,
                        height_at=environment.terrain.height_at,
                        color_rgba=style[style_id]["color_rgba"],
                        z_offset_m=0.2,
                    )
                )

    return [
        SceneLayerMesh(
            layer_id=style_id,
            semantic_kind=str(style[style_id]["semantic_kind"]),
            mesh=merge_meshes(meshes),
            style_id=style_id,
        )
        for style_id, meshes in grouped.items()
    ]


def _build_obstacle_meshes(
    occluding_objects: Sequence[Mapping[str, object]],
    *,
    height_at,
) -> Sequence[SceneLayerMesh]:
    style = default_style()
    grouped = defaultdict(list)

    for object_meta in occluding_objects:
        blocker_type = str(object_meta.get("blocker_type", "building"))
        layer_id = (
            "walls"
            if blocker_type == "wall"
            else "vegetation"
            if blocker_type == "vegetation"
            else "buildings"
        )
        if isinstance(object_meta.get("footprint_xy_m"), list):
            footprint = np.asarray(object_meta["footprint_xy_m"], dtype=np.float32)
            min_tz, max_tz = _terrain_height_span_under_footprint(footprint, height_at)
            base_z, top_z = _solid_mesh_elevations(object_meta, min_tz, max_tz)
            grouped[layer_id].append(
                mesh_from_extrusion(
                    footprint,
                    base_z_m=base_z,
                    top_z_m=top_z,
                    color_rgba=style[layer_id]["color_rgba"],
                )
            )
        elif object_meta.get("kind") == "cylinder-v1":
            radius = float(object_meta["radius_m"])
            cx = float(object_meta["center_x_m"])
            cy = float(object_meta["center_y_m"])
            perimeter_angles = np.linspace(0.0, math.tau, num=8, endpoint=False, dtype=np.float32)
            perimeter_heights = [
                height_at(
                    cx + (radius * float(math.cos(float(angle)))),
                    cy + (radius * float(math.sin(float(angle)))),
                )
                for angle in perimeter_angles
            ]
            perimeter_heights.append(height_at(cx, cy))
            min_tz = float(min(perimeter_heights))
            max_tz = float(max(perimeter_heights))
            base_z, top_z = _solid_mesh_elevations(object_meta, min_tz, max_tz)
            angles = np.linspace(0.0, math.tau, num=18, endpoint=False, dtype=np.float32)
            ring = np.stack(
                [
                    cx + (np.cos(angles) * radius),
                    cy + (np.sin(angles) * radius),
                ],
                axis=-1,
            )
            grouped[layer_id].append(
                mesh_from_extrusion(
                    ring,
                    base_z_m=base_z,
                    top_z_m=top_z,
                    color_rgba=style[layer_id]["color_rgba"],
                )
            )

    return [
        SceneLayerMesh(
            layer_id=layer_id,
            semantic_kind=str(style[layer_id]["semantic_kind"]),
            mesh=merge_meshes(meshes),
            style_id=layer_id,
        )
        for layer_id, meshes in grouped.items()
    ]


def _dedupe_style_ids(layer_documents: Sequence[Mapping[str, object]]) -> Sequence[str]:
    seen = set()
    ordered = []
    for layer in layer_documents:
        style_id = str(layer["style_id"])
        if style_id in seen:
            continue
        seen.add(style_id)
        ordered.append(style_id)
    return ordered


def _landcover_style_id(properties: Mapping[str, object]) -> str:
    style = default_style()
    direct_id = properties.get("style_id")
    if isinstance(direct_id, str) and direct_id in style:
        return direct_id

    for key in ("landcover", "land_cover", "class", "kind", "type", "category"):
        value = properties.get(key)
        if not isinstance(value, str):
            continue
        normalized = value.strip().lower()
        if any(token in normalized for token in ("forest", "wood", "tree")):
            return "landcover-forest"
        if any(token in normalized for token in ("water", "river", "lake", "stream", "wetland")):
            return "landcover-water"
        if any(
            token in normalized
            for token in ("urban", "built", "residential", "commercial", "industrial")
        ):
            return "landcover-urban"
    return "landcover-open"


def build_scene_from_replay(
    replay: str | Path | ReplayDocument,
    output_dir: str | Path,
    *,
    scene_id: str | None = None,
    environment_bundle: str | Path | None = None,
) -> dict[str, object]:
    replay_document = _load_replay(replay)
    if replay_document is None:
        raise ValueError("Replay scene compilation requires a replay document.")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    meta = replay_document.get("meta", {})
    active_scene_id = _normalize_scene_id(
        scene_id or str(meta.get("scenario_name", "synthetic-scene"))
    )

    terrain_meta = meta.get("terrain")
    if environment_bundle:
        environment = load_environment_bundle(str(environment_bundle))
        terrain_meta = environment.terrain.to_metadata()
        bounds_xy_m = environment.bounds_xy_m
        runtime_crs = _runtime_crs_metadata(environment.crs)
        landcover_layers = _build_landcover_meshes_from_environment(environment)
        occluding_objects = environment.obstacles.to_metadata()
    else:
        environment = None
        bounds_xy_m = _as_bounds(
            meta.get("environment_bounds_m") or terrain_meta.get("xy_bounds_m")
        )
        runtime_crs = {
            "source_crs_id": str(meta.get("crs_id", "local-synthetic")),
            "runtime_crs_id": str(meta.get("crs_id", "local-enu")),
            "origin_geodetic": {"lat_deg": None, "lon_deg": None, "h_m": None},
            "xy_units": "meters",
            "z_datum": "local",
        }
        landcover_layers = []
        occluding_objects = list(meta.get("occluding_objects", []))

    if not isinstance(terrain_meta, Mapping) or not isinstance(
        terrain_meta.get("viewer_mesh"), Mapping
    ):
        raise ValueError("Replay metadata does not include terrain.viewer_mesh.")

    terrain_mesh = terrain_mesh_from_viewer_mesh(terrain_meta["viewer_mesh"])
    height_at = height_function_from_mesh(terrain_meta["viewer_mesh"])
    layer_specs = [
        SceneLayerMesh(
            layer_id="terrain-base",
            semantic_kind="terrain",
            mesh=terrain_mesh,
            style_id="terrain-base",
        ),
        *landcover_layers,
        *_build_obstacle_meshes(occluding_objects, height_at=height_at),
    ]
    manifest_layers = write_scene_layers(output_path, layer_specs)
    replay_path = _copy_replay_document(output_path, replay_document)

    style_ids = [*_dedupe_style_ids(manifest_layers), "tracks", "truths", "nodes"]
    _write_json(output_path / "metadata" / "style.json", style_document(style_ids))
    _write_json(
        output_path / "metadata" / "environment.json",
        {
            "source_kind": "synthetic",
            "scene_id": active_scene_id,
            "source_crs_id": runtime_crs["source_crs_id"],
            "runtime_crs": runtime_crs,
            "bounds_xy_m": bounds_xy_m.to_metadata(),
            "terrain_summary": {
                "kind": terrain_meta.get("kind"),
                "base_resolution_m": terrain_meta.get("base_resolution_m"),
                "min_height_m": terrain_meta.get("min_height_m"),
                "max_height_m": terrain_meta.get("max_height_m"),
            },
            "occluding_objects": to_jsonable(occluding_objects),
            "replay_meta": to_jsonable(meta),
        },
    )

    manifest = _build_manifest(
        scene_id=active_scene_id,
        bounds_xy_m=bounds_xy_m,
        runtime_crs=runtime_crs,
        source_crs_id=str(runtime_crs["source_crs_id"]),
        layers=manifest_layers,
        replay_path=replay_path,
        source_kind="synthetic",
        environment_path="metadata/environment.json",
        style_path="metadata/style.json",
        provenance={
            "tool": "argusnet.world.scene_loader.build_scene_from_replay",
            "replay_source": str(replay) if not isinstance(replay, Mapping) else "<in-memory>",
            "environment_bundle": None if environment_bundle is None else str(environment_bundle),
        },
    )
    _write_json(output_path / "scene_manifest.json", manifest)
    return manifest


def build_scene_from_gis(
    dem_path: str | Path,
    output_dir: str | Path,
    *,
    scene_id: str | None = None,
    source_crs: str | None = None,
    replay: str | Path | ReplayDocument | None = None,
    overlay_paths: Mapping[str, Sequence[str | Path]] | None = None,
) -> dict[str, object]:
    raster = project_dem_to_runtime(dem_path, source_crs=source_crs)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    active_scene_id = _normalize_scene_id(scene_id or Path(dem_path).stem)
    style = default_style()

    layer_specs = chunked_height_grid_layers(
        layer_id="terrain-base",
        style_id="terrain-base",
        x_values=raster.x_values_m,
        y_values=raster.y_values_m,
        heights_m=raster.heights_m,
    )
    overlay_stats: dict[str, int] = {}
    overlay_source_crs: dict[str, Sequence[str]] = {}

    for semantic_kind, paths in (overlay_paths or {}).items():
        grouped_meshes = defaultdict(list)
        feature_count = 0
        source_crs_ids = []
        for path in paths:
            layer = load_geojson_layer(path, default_source_crs_id=raster.source_crs_id)
            source_crs_ids.append(layer.source_crs_id)
            for feature in layer.features:
                for geometry_kind, geometry_xy in geojson_geometry_to_runtime(
                    feature["geometry"],
                    source_crs_id=layer.source_crs_id,
                    runtime_crs=raster.runtime_crs,
                ):
                    feature_count += 1
                    if semantic_kind == "buildings" and geometry_kind == "polygon":
                        height_m = float(
                            feature["properties"].get("height_m")
                            or feature["properties"].get("height")
                            or 24.0
                        )
                        base_m = float(
                            feature["properties"].get("base_z_m")
                            or raster.height_at(float(geometry_xy[0][0]), float(geometry_xy[0][1]))
                        )
                        grouped_meshes["buildings"].append(
                            mesh_from_extrusion(
                                geometry_xy,
                                base_z_m=base_m,
                                top_z_m=base_m + max(height_m, 2.0),
                                color_rgba=style["buildings"]["color_rgba"],
                            )
                        )
                    elif semantic_kind == "roads" and geometry_kind == "line":
                        grouped_meshes["roads"].append(
                            mesh_from_polyline(
                                geometry_xy,
                                height_at=raster.height_at,
                                color_rgba=style["roads"]["color_rgba"],
                                width_m=float(style["roads"].get("line_width_m", 6.0)),
                            )
                        )
                    elif semantic_kind == "water" and geometry_kind == "line":
                        grouped_meshes["water"].append(
                            mesh_from_polyline(
                                geometry_xy,
                                height_at=raster.height_at,
                                color_rgba=style["water"]["color_rgba"],
                                width_m=float(style["water"].get("line_width_m", 8.0)),
                            )
                        )
                    elif semantic_kind == "water" and geometry_kind == "polygon":
                        grouped_meshes["water"].append(
                            mesh_from_polygon(
                                geometry_xy,
                                height_at=raster.height_at,
                                color_rgba=style["water"]["color_rgba"],
                                z_offset_m=0.28,
                            )
                        )
                    elif semantic_kind == "zones" and geometry_kind == "polygon":
                        grouped_meshes["zones"].append(
                            mesh_from_polygon(
                                geometry_xy,
                                height_at=raster.height_at,
                                color_rgba=style["zones"]["color_rgba"],
                                z_offset_m=0.28,
                            )
                        )
                    elif semantic_kind == "landcover" and geometry_kind == "polygon":
                        style_id = _landcover_style_id(feature["properties"])
                        grouped_meshes[style_id].append(
                            mesh_from_polygon(
                                geometry_xy,
                                height_at=raster.height_at,
                                color_rgba=style[style_id]["color_rgba"],
                                z_offset_m=0.28,
                            )
                        )
        overlay_stats[semantic_kind] = feature_count
        overlay_source_crs[semantic_kind] = sorted(set(source_crs_ids))
        for style_id, meshes in grouped_meshes.items():
            if not meshes:
                continue
            layer_specs.append(
                SceneLayerMesh(
                    layer_id=style_id,
                    semantic_kind=str(style[style_id]["semantic_kind"]),
                    mesh=merge_meshes(meshes),
                    style_id=style_id,
                )
            )

    manifest_layers = write_scene_layers(output_path, layer_specs)
    replay_document = _load_replay(replay)
    replay_path = _copy_replay_document(output_path, replay_document)
    style_ids = [*_dedupe_style_ids(manifest_layers), "tracks", "truths", "nodes"]
    _write_json(output_path / "metadata" / "style.json", style_document(style_ids))
    _write_json(
        output_path / "metadata" / "environment.json",
        {
            "source_kind": "gis",
            "scene_id": active_scene_id,
            "source_crs_id": raster.source_crs_id,
            "runtime_crs": raster.runtime_crs.to_metadata(),
            "bounds_xy_m": raster.bounds_xy_m.to_metadata(),
            "terrain_summary": {
                "kind": "geotiff-dem-v1",
                "rows": int(raster.heights_m.shape[0]),
                "cols": int(raster.heights_m.shape[1]),
                "min_height_m": float(np.min(raster.heights_m)),
                "max_height_m": float(np.max(raster.heights_m)),
                "source_bounds_xy": {
                    "x_min": raster.source_bounds_xy[0],
                    "x_max": raster.source_bounds_xy[1],
                    "y_min": raster.source_bounds_xy[2],
                    "y_max": raster.source_bounds_xy[3],
                },
            },
            "overlay_counts": overlay_stats,
            "overlay_source_crs": overlay_source_crs,
            "dem_path": str(dem_path),
        },
    )

    manifest = _build_manifest(
        scene_id=active_scene_id,
        bounds_xy_m=raster.bounds_xy_m,
        runtime_crs=raster.runtime_crs.to_metadata(),
        source_crs_id=raster.source_crs_id,
        layers=manifest_layers,
        replay_path=replay_path,
        source_kind="gis",
        environment_path="metadata/environment.json",
        style_path="metadata/style.json",
        provenance={
            "tool": "argusnet.world.scene_loader.build_scene_from_gis",
            "dem_path": str(dem_path),
            "overlay_paths": {
                key: [str(path) for path in value] for key, value in (overlay_paths or {}).items()
            },
        },
    )
    _write_json(output_path / "scene_manifest.json", manifest)
    return manifest


def build_scene_package(
    output_dir: str | Path,
    *,
    replay: str | Path | ReplayDocument | None = None,
    environment_bundle: str | Path | None = None,
    dem_path: str | Path | None = None,
    source_crs: str | None = None,
    overlay_paths: Mapping[str, Sequence[str | Path]] | None = None,
    scene_id: str | None = None,
) -> dict[str, object]:
    if dem_path is not None:
        return build_scene_from_gis(
            dem_path=dem_path,
            output_dir=output_dir,
            scene_id=scene_id,
            source_crs=source_crs,
            replay=replay,
            overlay_paths=overlay_paths,
        )
    if replay is None:
        raise ValueError("Synthetic scene compilation requires a replay document or --dem.")
    return build_scene_from_replay(
        replay=replay,
        output_dir=output_dir,
        scene_id=scene_id,
        environment_bundle=environment_bundle,
    )


def build_temp_scene_from_replay(replay_path: str | Path) -> tempfile.TemporaryDirectory[str]:
    temp_dir = tempfile.TemporaryDirectory(prefix="smartscene-")
    build_scene_from_replay(replay_path, temp_dir.name)
    return temp_dir


__all__ = [
    "SCENE_FORMAT_VERSION",
    "STYLE_FORMAT_VERSION",
    "build_scene_from_gis",
    "build_scene_from_replay",
    "build_scene_package",
    "build_temp_scene_from_replay",
    "load_scene_manifest",
    "validate_scene_manifest",
]
