from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from argusnet.core.frames import ENUOrigin, wgs84_to_enu

from .environment import Bounds2D, EnvironmentCRS


def lazy_import_tifffile():
    try:
        import tifffile
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
        raise RuntimeError(
            "Scene compilation from GeoTIFF requires `tifffile`. "
            "Install project dependencies with `python3 -m pip install --user -e .`."
        ) from exc
    return tifffile


def lazy_import_pyproj():
    try:
        from pyproj import CRS, Transformer
    except ImportError as exc:  # pragma: no cover - exercised only when dependency is missing.
        raise RuntimeError(
            "Scene compilation from GIS data requires `pyproj`. "
            "Install project dependencies with `python3 -m pip install --user -e .`."
        ) from exc
    return CRS, Transformer


@dataclass(frozen=True)
class RasterReference:
    heights_m: np.ndarray
    x_values_m: np.ndarray
    y_values_m: np.ndarray
    source_bounds_xy: tuple[float, float, float, float]
    runtime_crs: EnvironmentCRS
    source_crs_id: str

    @property
    def bounds_xy_m(self) -> Bounds2D:
        return Bounds2D(
            x_min_m=float(np.min(self.x_values_m)),
            x_max_m=float(np.max(self.x_values_m)),
            y_min_m=float(np.min(self.y_values_m)),
            y_max_m=float(np.max(self.y_values_m)),
        )

    def height_at(self, x_m: float, y_m: float) -> float:
        tx = np.clip(
            np.interp(
                float(x_m), self.x_values_m, np.arange(self.x_values_m.size, dtype=np.float32)
            ),
            0.0,
            self.x_values_m.size - 1,
        )
        ty = np.clip(
            np.interp(
                float(y_m), self.y_values_m, np.arange(self.y_values_m.size, dtype=np.float32)
            ),
            0.0,
            self.y_values_m.size - 1,
        )
        col = min(int(math.floor(float(tx))), self.x_values_m.size - 2)
        row = min(int(math.floor(float(ty))), self.y_values_m.size - 2)
        ax = float(tx) - col
        ay = float(ty) - row
        z00 = float(self.heights_m[row, col])
        z10 = float(self.heights_m[row, col + 1])
        z01 = float(self.heights_m[row + 1, col])
        z11 = float(self.heights_m[row + 1, col + 1])
        z0 = z00 + ((z10 - z00) * ax)
        z1 = z01 + ((z11 - z01) * ax)
        return z0 + ((z1 - z0) * ay)


@dataclass(frozen=True)
class GeoJSONLayer:
    features: list[dict[str, object]]
    source_crs_id: str


def extract_geotiff_transform(
    tags: Mapping[int, object],
) -> tuple[float, float, float, float, tuple[float, ...] | None]:
    transform_tag = tags.get(34264)
    if transform_tag is not None:
        return 0.0, 0.0, 0.0, 0.0, tuple(float(value) for value in transform_tag.value)

    scale_tag = tags.get(33550)
    tiepoint_tag = tags.get(33922)
    if scale_tag is None or tiepoint_tag is None:
        raise ValueError(
            "DEM GeoTIFF must include ModelPixelScaleTag and ModelTiepointTag "
            "or ModelTransformationTag."
        )
    scale = tuple(float(value) for value in scale_tag.value)
    tiepoints = tuple(float(value) for value in tiepoint_tag.value)
    if len(scale) < 2 or len(tiepoints) < 6:
        raise ValueError("DEM GeoTIFF georeferencing tags are incomplete.")
    return scale[0], scale[1], tiepoints[3], tiepoints[4], None


def _extract_nodata_value(page: object) -> float | None:
    nodata = getattr(page, "nodata", None)
    if nodata is not None:
        try:
            return float(nodata)
        except (TypeError, ValueError):
            pass
    tag = getattr(page, "tags", {}).get(42113)  # GDAL_NODATA
    if tag is not None:
        value = getattr(tag, "value", None)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="ignore")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    return None


def _fill_nodata(heights: np.ndarray) -> np.ndarray:
    filled = np.asarray(heights, dtype=np.float32).copy()
    if not np.isnan(filled).any():
        return filled

    for row_index in range(filled.shape[0]):
        row = filled[row_index]
        valid = np.isfinite(row)
        if not valid.any():
            continue
        invalid = ~valid
        if invalid.any():
            row[invalid] = np.interp(
                np.flatnonzero(invalid),
                np.flatnonzero(valid),
                row[valid],
            )
            filled[row_index] = row

    for col_index in range(filled.shape[1]):
        column = filled[:, col_index]
        valid = np.isfinite(column)
        if not valid.any():
            continue
        invalid = ~valid
        if invalid.any():
            column[invalid] = np.interp(
                np.flatnonzero(invalid),
                np.flatnonzero(valid),
                column[valid],
            )
            filled[:, col_index] = column

    if np.isnan(filled).any():
        finite = filled[np.isfinite(filled)]
        replacement = float(np.mean(finite)) if finite.size else 0.0
        filled[~np.isfinite(filled)] = replacement
    return filled


def project_dem_to_runtime(
    dem_path: str | Path, *, source_crs: str | None = None
) -> RasterReference:
    tifffile = lazy_import_tifffile()
    CRS, Transformer = lazy_import_pyproj()

    with tifffile.TiffFile(str(dem_path)) as handle:
        page = handle.pages[0]
        heights = page.asarray().astype(np.float32)
        nodata_value = _extract_nodata_value(page)
        if nodata_value is not None:
            heights = np.where(np.isclose(heights, nodata_value), np.nan, heights)
        heights = _fill_nodata(heights)
        scale_x, scale_y, tie_x, tie_y, transform_values = extract_geotiff_transform(page.tags)

        source_crs_id = source_crs
        if source_crs_id is None:
            geo_key_tag = page.tags.get(34735)
            if geo_key_tag is not None:
                values = tuple(int(value) for value in geo_key_tag.value)
                for index in range(0, len(values), 4):
                    if index + 3 >= len(values):
                        break
                    key_id, _, _, value_offset = values[index : index + 4]
                    if key_id in {2048, 3072} and value_offset:
                        source_crs_id = f"EPSG:{value_offset}"
                        break
        if source_crs_id is None:
            raise ValueError(
                "Could not determine source CRS from DEM. Pass --source-crs, "
                "for example EPSG:32611."
            )

    source = CRS.from_user_input(source_crs_id)
    to_wgs84 = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)

    rows, cols = heights.shape
    if transform_values is not None:
        matrix = np.asarray(transform_values, dtype=float).reshape(4, 4)
        source_x_values = np.array(
            [matrix[0, 0] * col + matrix[0, 3] for col in range(cols)], dtype=np.float64
        )
        source_y_values = np.array(
            [matrix[1, 1] * row + matrix[1, 3] for row in range(rows)], dtype=np.float64
        )
    else:
        source_x_values = np.array(
            [tie_x + (col * scale_x) for col in range(cols)], dtype=np.float64
        )
        source_y_values = np.array(
            [tie_y - (row * scale_y) for row in range(rows)], dtype=np.float64
        )

    x_min = float(np.min(source_x_values))
    x_max = float(np.max(source_x_values))
    y_min = float(np.min(source_y_values))
    y_max = float(np.max(source_y_values))
    lon_center, lat_center = to_wgs84.transform((x_min + x_max) * 0.5, (y_min + y_max) * 0.5)
    origin = ENUOrigin(
        latitude_deg=float(lat_center), longitude_deg=float(lon_center), altitude_m=0.0
    )

    runtime_x_values = []
    for x_value in source_x_values:
        lon, lat = to_wgs84.transform(float(x_value), float(source_y_values[0]))
        runtime_x_values.append(float(wgs84_to_enu(float(lat), float(lon), 0.0, origin)[0]))

    runtime_y_values = []
    reference_x = float(source_x_values[0])
    for y_value in source_y_values:
        lon, lat = to_wgs84.transform(reference_x, float(y_value))
        runtime_y_values.append(float(wgs84_to_enu(float(lat), float(lon), 0.0, origin)[1]))

    return RasterReference(
        heights_m=heights,
        x_values_m=np.asarray(runtime_x_values, dtype=np.float32),
        y_values_m=np.asarray(runtime_y_values, dtype=np.float32),
        source_bounds_xy=(x_min, x_max, y_min, y_max),
        runtime_crs=EnvironmentCRS(
            source_crs_id=str(source_crs_id),
            runtime_crs_id="local-enu",
            origin_lat_deg=float(lat_center),
            origin_lon_deg=float(lon_center),
            origin_h_m=0.0,
            xy_units="meters",
            z_datum="ellipsoidal-height",
        ),
        source_crs_id=str(source_crs_id),
    )


def load_geojson_features(path: str | Path) -> list[dict[str, object]]:
    return load_geojson_layer(path).features


def _extract_geojson_crs(document: Mapping[str, object]) -> str | None:
    crs = document.get("crs")
    if not isinstance(crs, Mapping):
        return None
    properties = crs.get("properties")
    if not isinstance(properties, Mapping):
        return None
    name = properties.get("name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return None


def load_geojson_layer(
    path: str | Path, *, default_source_crs_id: str | None = None
) -> GeoJSONLayer:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if document.get("type") == "FeatureCollection":
        features = document.get("features", [])
    elif document.get("type") == "Feature":
        features = [document]
    else:
        features = [{"type": "Feature", "geometry": document, "properties": {}}]
    normalized = []
    for feature in features:
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        normalized.append(
            {
                "geometry": geometry,
                "properties": feature.get("properties") or {},
            }
        )
    return GeoJSONLayer(
        features=normalized,
        source_crs_id=_extract_geojson_crs(document) or str(default_source_crs_id or "EPSG:4326"),
    )


def geojson_geometry_to_runtime(
    geometry: Mapping[str, object],
    *,
    source_crs_id: str,
    runtime_crs: EnvironmentCRS,
) -> list[tuple[str, np.ndarray]]:
    CRS, Transformer = lazy_import_pyproj()
    source = CRS.from_user_input(source_crs_id)
    to_wgs84 = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)
    origin = ENUOrigin(
        latitude_deg=float(runtime_crs.origin_lat_deg or 0.0),
        longitude_deg=float(runtime_crs.origin_lon_deg or 0.0),
        altitude_m=float(runtime_crs.origin_h_m or 0.0),
    )

    def project(coord: Sequence[float]) -> np.ndarray:
        lon, lat = to_wgs84.transform(float(coord[0]), float(coord[1]))
        enu = wgs84_to_enu(float(lat), float(lon), 0.0, origin)
        return np.asarray([enu[0], enu[1]], dtype=np.float32)

    geometry_type = str(geometry.get("type"))
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list):
        return []
    if geometry_type == "Polygon":
        return [
            ("polygon", np.asarray([project(point) for point in coordinates[0]], dtype=np.float32))
        ]
    if geometry_type == "MultiPolygon":
        return [
            ("polygon", np.asarray([project(point) for point in polygon[0]], dtype=np.float32))
            for polygon in coordinates
        ]
    if geometry_type == "LineString":
        return [("line", np.asarray([project(point) for point in coordinates], dtype=np.float32))]
    if geometry_type == "MultiLineString":
        return [
            ("line", np.asarray([project(point) for point in line], dtype=np.float32))
            for line in coordinates
        ]
    return []


__all__ = [
    "GeoJSONLayer",
    "RasterReference",
    "geojson_geometry_to_runtime",
    "load_geojson_layer",
    "load_geojson_features",
    "project_dem_to_runtime",
]
