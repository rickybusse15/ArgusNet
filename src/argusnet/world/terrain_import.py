"""Real terrain for a run (R5): a GeoTIFF DEM to `argusnet.terrain`.

Offline tooling, like the buildings importer. It reads a DEM GeoTIFF and
writes an `argusnet.terrain` v1 document. The native builder's
`terrain_path` option builds the run's terrain from it in place of the
procedural one.

The DEM is resampled onto the run's own height grid: the cells a map preset
gives (`--map-preset`, the same bounds and resolution the procedural terrain
uses), in the local east-north-up frame around a geodetic origin. Use the
same origin as the buildings (`import_buildings.py --origin`) and the
layers line up: the builder refuses a buildings file whose origin differs.

Georeferencing works cell by cell, never by axis. Each grid cell's ENU
position goes to latitude and longitude (the inverse of the transform the
buildings importer uses), then to the DEM's CRS, then through the inverse of
the GeoTIFF's affine to a fractional pixel. The DEM is sampled there
bilinearly. A rotated or sheared raster, or a CRS whose grid north is not
true north, is therefore placed correctly. Pixel centres follow the file's
raster type: `PixelIsArea` (the default) puts a pixel's value at its centre,
half a pixel in from the tie point; `PixelIsPoint` puts it on the tie point.

Heights: `z = elevation - origin_h_m`, where `origin_h_m` is the lowest
sampled elevation rounded down, less 2 m. So the lowest ground stands 2 m
above the runtime's ground plane, as procedural terrain does, and nothing is
water by height alone: water comes from the data, not from the DEM's low
points. The offset is recorded in the georeference, with the DEM's vertical
datum (`z_datum: "dem"`, as the file gives it).

Refused, never guessed: a DEM that does not cover every cell of the map, a
CRS that cannot be found (pass `source_crs`), or a raster over the pixel
limit. Nodata pixels are filled from their neighbours and counted.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from argusnet.core.frames import ENUOrigin, enu_to_wgs84, wgs84_to_enu

from ._scene_gis import (
    _MAX_DEM_PIXELS,
    _extract_nodata_value,
    _fill_nodata,
    lazy_import_pyproj,
    lazy_import_tifffile,
)

FORMAT = "argusnet.terrain"
VERSION = 1
# The lowest ground's height above the runtime's ground plane, metres.
GROUND_MARGIN_M = 2.0
# Heights are written to the millimetre.
HEIGHT_DECIMALS = 3
_PIXEL_IS_POINT = 2


def _geo_keys(tags: Mapping[int, Any]) -> dict[int, int]:
    """The GeoKeyDirectory's short-valued keys, by id."""
    tag = tags.get(34735)
    if tag is None:
        return {}
    values = [int(v) for v in getattr(tag, "value", ())]
    keys: dict[int, int] = {}
    for index in range(4, len(values) - 3, 4):
        key_id, location, _, value = values[index : index + 4]
        if location == 0:
            keys[key_id] = value
    return keys


def raster_affine(tags: Mapping[int, Any]) -> np.ndarray:
    """The 2x3 affine from raster space (column, row) to the DEM's CRS.

    From ModelTransformationTag when the file has one, otherwise from the
    first tie point and the pixel scale. Raster space puts a pixel's corner
    at integers; see `pixel_offset` for where its value is.
    """
    transform = tags.get(34264)
    if transform is not None:
        m = [float(v) for v in transform.value]
        if len(m) != 16:
            raise ValueError("ModelTransformationTag must hold 16 values")
        matrix: np.ndarray = np.array([[m[0], m[1], m[3]], [m[4], m[5], m[7]]], dtype=np.float64)
        return matrix
    scale_tag, tie_tag = tags.get(33550), tags.get(33922)
    if scale_tag is None or tie_tag is None:
        raise ValueError(
            "DEM GeoTIFF must include ModelPixelScaleTag and ModelTiepointTag "
            "or ModelTransformationTag."
        )
    scale = [float(v) for v in scale_tag.value]
    tie = [float(v) for v in tie_tag.value]
    if len(scale) < 2 or len(tie) < 6:
        raise ValueError("DEM GeoTIFF georeferencing tags are incomplete.")
    if scale[0] <= 0.0 or scale[1] <= 0.0:
        raise ValueError("DEM GeoTIFF pixel scale must be positive.")
    i, j, x, y = tie[0], tie[1], tie[3], tie[4]
    affine: np.ndarray = np.array(
        [[scale[0], 0.0, x - i * scale[0]], [0.0, -scale[1], y + j * scale[1]]],
        dtype=np.float64,
    )
    return affine


def pixel_offset(keys: Mapping[int, int]) -> float:
    """Where a pixel's value sits in raster space past its corner: the
    centre (0.5) for `PixelIsArea`, the default, or the corner itself for
    `PixelIsPoint`."""
    return 0.0 if keys.get(1025) == _PIXEL_IS_POINT else 0.5


def _source_crs(keys: Mapping[int, int], source_crs: str | None) -> str:
    if source_crs:
        return source_crs
    for key in (3072, 2048):  # ProjectedCSType, GeographicType
        code = keys.get(key)
        if code and code != 32767:  # user-defined
            return f"EPSG:{code}"
    raise ValueError("Could not determine the DEM's CRS. Pass source_crs, for example EPSG:32611.")


def _bilinear(heights: np.ndarray, col: np.ndarray, row: np.ndarray) -> np.ndarray:
    rows, cols = heights.shape
    c0 = np.clip(np.floor(col).astype(np.int64), 0, max(cols - 2, 0))
    r0 = np.clip(np.floor(row).astype(np.int64), 0, max(rows - 2, 0))
    c1 = np.minimum(c0 + 1, cols - 1)
    r1 = np.minimum(r0 + 1, rows - 1)
    ax = col - c0
    ay = row - r0
    z00, z10 = heights[r0, c0], heights[r0, c1]
    z01, z11 = heights[r1, c0], heights[r1, c1]
    z0 = z00 + (z10 - z00) * ax
    z1 = z01 + (z11 - z01) * ax
    sampled: np.ndarray = z0 + (z1 - z0) * ay
    return sampled


def import_terrain(
    dem_path: str | Path,
    *,
    map_preset: str,
    origin: tuple[float, float] | None = None,
    source_crs: str | None = None,
) -> dict[str, object]:
    """The `argusnet.terrain` document for `dem_path` over `map_preset`'s
    grid. `origin` is the local frame's (latitude, longitude) in degrees;
    the centre of the DEM when not given."""
    from argusnet.simulation.sim import (
        map_bounds_for_scale,
        map_bounds_object,
        map_scale_for_preset,
        terrain_resolution_for_scale,
    )

    from .procedural import _grid_values

    tifffile = lazy_import_tifffile()
    CRS, Transformer = lazy_import_pyproj()
    source_bytes = Path(dem_path).read_bytes()
    with tifffile.TiffFile(str(dem_path)) as handle:
        page = handle.pages[0]
        if math.prod(page.shape) > _MAX_DEM_PIXELS:
            raise ValueError(f"DEM {dem_path} has more than {_MAX_DEM_PIXELS} pixels.")
        if len(page.shape) != 2:
            raise ValueError(f"DEM {dem_path} must be a single band, got shape {page.shape}.")
        raw = page.asarray().astype(np.float64)
        nodata = _extract_nodata_value(page)
        keys = _geo_keys(page.tags)
        affine = raster_affine(page.tags)
    if nodata is not None:
        raw = np.where(np.isclose(raw, nodata), np.nan, raw)
    raw[~np.isfinite(raw)] = np.nan
    nodata_pixels = int(np.count_nonzero(np.isnan(raw)))
    if nodata_pixels == raw.size:
        raise ValueError(f"DEM {dem_path} holds no elevations.")
    heights: np.ndarray = _fill_nodata(raw).astype(np.float64)
    offset = pixel_offset(keys)
    source_crs_id = _source_crs(keys, source_crs)
    source = CRS.from_user_input(source_crs_id)
    to_source = Transformer.from_crs(CRS.from_epsg(4326), source, always_xy=True)
    to_wgs84 = Transformer.from_crs(source, CRS.from_epsg(4326), always_xy=True)

    linear = affine[:, :2]
    if abs(np.linalg.det(linear)) < 1e-18:
        raise ValueError("DEM GeoTIFF affine is singular.")
    inverse = np.linalg.inv(linear)

    if origin is None:
        # The middle of the samples, wherever the raster type puts them:
        # (cols / 2, rows / 2) for PixelIsArea, half a pixel less for
        # PixelIsPoint, whose samples span 0..cols - 1.
        rows, cols = heights.shape
        centre = affine @ np.array([offset + (cols - 1) * 0.5, offset + (rows - 1) * 0.5, 1.0])
        lon, lat = to_wgs84.transform(float(centre[0]), float(centre[1]))
        origin = (round(float(lat), 7), round(float(lon), 7))
    enu_origin = ENUOrigin(latitude_deg=origin[0], longitude_deg=origin[1], altitude_m=0.0)

    scale = map_scale_for_preset(map_preset)
    bounds = map_bounds_for_scale(scale)
    resolution_m = terrain_resolution_for_scale(scale)
    x_values, y_values = _grid_values(map_bounds_object(bounds), resolution_m)
    xx: np.ndarray
    yy: np.ndarray
    xx, yy = np.meshgrid(x_values, y_values)
    flat_x, flat_y = xx.ravel(), yy.ravel()

    lats = np.empty(flat_x.size)
    lons = np.empty(flat_x.size)
    worst_roundtrip_m = 0.0
    for k in range(flat_x.size):
        lat, lon, _ = enu_to_wgs84(np.array([flat_x[k], flat_y[k], 0.0]), enu_origin)
        lats[k], lons[k] = lat, lon
        back = wgs84_to_enu(lat, lon, 0.0, enu_origin)
        worst_roundtrip_m = max(
            worst_roundtrip_m, math.hypot(back[0] - flat_x[k], back[1] - flat_y[k])
        )
    sx, sy = to_source.transform(lons, lats)
    model = np.vstack([np.asarray(sx) - affine[0, 2], np.asarray(sy) - affine[1, 2]])
    col, row = inverse @ model
    col, row = col - offset, row - offset
    rows, cols = heights.shape
    tolerance = 1e-6
    inside = (
        (col >= -tolerance)
        & (col <= cols - 1 + tolerance)
        & (row >= -tolerance)
        & (row <= rows - 1 + tolerance)
    )
    if not inside.all():
        covered = float(np.mean(inside))
        raise ValueError(
            f"DEM {dem_path} covers {covered:.1%} of the {map_preset} map around "
            f"{origin[0]:.7f},{origin[1]:.7f}: it must cover all of it. Use a larger "
            "DEM, a smaller map preset or another origin."
        )
    elevations = _bilinear(heights, np.clip(col, 0, cols - 1), np.clip(row, 0, rows - 1))
    origin_h_m = math.floor(float(np.min(elevations))) - GROUND_MARGIN_M
    local = np.round(elevations - origin_h_m, HEIGHT_DECIMALS)
    return {
        "format": FORMAT,
        "version": VERSION,
        "georeference": {
            "source_crs_id": source_crs_id,
            "runtime_crs_id": "local-enu",
            "origin_lat_deg": origin[0],
            "origin_lon_deg": origin[1],
            "origin_h_m": origin_h_m,
            "z_datum": "dem",
        },
        "map_preset": map_preset,
        "bounds_xy_m": bounds,
        "resolution_m": resolution_m,
        "rows": len(y_values),
        "cols": len(x_values),
        "heights_m": [float(v) for v in local.tolist()],
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "summary": {
            "dem_shape": [rows, cols],
            "nodata_pixels_filled": nodata_pixels,
            "pixel_is_point": offset == 0.0,
            "elevation_min_m": float(np.min(elevations)),
            "elevation_max_m": float(np.max(elevations)),
            "max_roundtrip_error_m": worst_roundtrip_m,
        },
    }


def write_terrain(document: Mapping[str, object], path: str | Path) -> None:
    Path(path).write_text(json.dumps(document, separators=(",", ":")) + "\n", encoding="utf-8")
