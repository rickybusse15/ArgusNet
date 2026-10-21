"""The DEM importer (R5): GeoTIFF to `argusnet.terrain` on the run's grid.

The DEMs here are planes in UTM zone 11 (EPSG:32611), which bilinear
sampling reproduces exactly, so every imported height can be checked
against the plane at the cell's true position. The origin is 2 degrees
east of the zone's central meridian, where grid north is about 1.1 degrees
off true north: an importer that projected each axis on its own would miss
by decimetres at the map's edge.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
pyproj = pytest.importorskip("pyproj")

from argusnet.core.frames import ENUOrigin, enu_to_wgs84  # noqa: E402
from argusnet.world.terrain_import import (  # noqa: E402
    GROUND_MARGIN_M,
    import_terrain,
)

ORIGIN = (34.2, -115.0)
TO_UTM = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32611", always_xy=True)
CENTRE = TO_UTM.transform(ORIGIN[1], ORIGIN[0])


def plane(e: np.ndarray, n: np.ndarray) -> np.ndarray:
    return 500.0 + 0.04 * (e - CENTRE[0]) + 0.015 * (n - CENTRE[1])


def write_dem(
    path: Path,
    *,
    affine: tuple[float, float, float, float, float, float],
    shape: tuple[int, int],
    point: bool = False,
    tie_scale: bool = True,
    crs_key: int | None = 32611,
    nodata_cells: tuple[tuple[int, int], ...] = (),
) -> Path:
    """A DEM of `plane`, sampled at pixel values' positions under `affine`
    (X = a*col + b*row + c, Y = d*col + e*row + f, raster space)."""
    a, b, c, d, e, f = affine
    rows, cols = shape
    shift = 0.0 if point else 0.5
    col, row = np.meshgrid(np.arange(cols) + shift, np.arange(rows) + shift)
    data = plane(a * col + b * row + c, d * col + e * row + f).astype(np.float32)
    for r, k in nodata_cells:
        data[r, k] = -9999.0
    keys = [1, 1, 0, 0]
    entries = [(1024, 0, 1, 1), (1025, 0, 1, 2 if point else 1)]
    if crs_key is not None:
        entries.append((3072, 0, 1, crs_key))
    keys[3] = len(entries)
    for entry in entries:
        keys.extend(entry)
    tags = [(34735, "H", len(keys), keys, True)]
    if tie_scale:
        assert b == 0.0 and d == 0.0 and e < 0.0
        tags += [
            (33550, "d", 3, (a, -e, 0.0), True),
            (33922, "d", 6, (0.0, 0.0, 0.0, c, f, 0.0), True),
        ]
    else:
        matrix = (a, b, 0.0, c, d, e, 0.0, f, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)
        tags.append((34264, "d", 16, matrix, True))
    if nodata_cells:
        tags.append((42113, "s", 0, "-9999", True))
    tifffile.imwrite(path, data, extratags=tags)
    return path


def aligned(pixel_m: float = 10.0, size: int = 120) -> tuple:
    """An axis-aligned raster centred on the origin, `size` pixels a side."""
    half = pixel_m * size * 0.5
    return (pixel_m, 0.0, CENTRE[0] - half, 0.0, -pixel_m, CENTRE[1] + half)


def expected(document: dict) -> np.ndarray:
    """The plane at each grid cell's true UTM position, less the offset."""
    bounds = document["bounds_xy_m"]
    res = document["resolution_m"]
    xs = np.arange(bounds["x_min_m"], bounds["x_max_m"] + res * 0.5, res)
    ys = np.arange(bounds["y_min_m"], bounds["y_max_m"] + res * 0.5, res)
    origin = ENUOrigin(latitude_deg=ORIGIN[0], longitude_deg=ORIGIN[1], altitude_m=0.0)
    out = np.empty((ys.size, xs.size))
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            lat, lon, _ = enu_to_wgs84(np.array([x, y, 0.0]), origin)
            e, n = TO_UTM.transform(lon, lat)
            out[i, j] = plane(np.array(e), np.array(n))
    return out - document["georeference"]["origin_h_m"]


def grid(document: dict) -> np.ndarray:
    return np.array(document["heights_m"]).reshape(document["rows"], document["cols"])


def test_the_dem_is_sampled_at_each_cells_true_position(tmp_path: Path) -> None:
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    document = import_terrain(dem, map_preset="small", origin=ORIGIN)
    assert document["format"] == "argusnet.terrain" and document["version"] == 1
    assert document["georeference"]["source_crs_id"] == "EPSG:32611"
    assert document["georeference"]["runtime_crs_id"] == "local-enu"
    # The small preset's grid: x -360..360, y -330..360 at 5 m.
    assert (document["rows"], document["cols"]) == (139, 145)
    assert document["resolution_m"] == 5.0
    np.testing.assert_allclose(grid(document), expected(document), atol=2e-3)
    # Back through the buildings importer's transform, to well under 1 mm.
    assert document["summary"]["max_roundtrip_error_m"] < 1e-4


def test_axis_by_axis_projection_would_miss(tmp_path: Path) -> None:
    """The check above is sensitive: grid north here is ~1.1 degrees off
    true north, so taking the UTM offsets as the ENU ones misses by
    decimetres at the map's corners."""
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    document = import_terrain(dem, map_preset="small", origin=ORIGIN)
    bounds = document["bounds_xy_m"]
    xs = np.arange(bounds["x_min_m"], bounds["x_max_m"] + 2.5, 5.0)
    ys = np.arange(bounds["y_min_m"], bounds["y_max_m"] + 2.5, 5.0)
    xx, yy = np.meshgrid(xs, ys)
    naive = plane(CENTRE[0] + xx, CENTRE[1] + yy) - document["georeference"]["origin_h_m"]
    assert np.max(np.abs(naive - grid(document))) > 0.2


def test_a_rotated_raster_gives_the_same_terrain(tmp_path: Path) -> None:
    """A ModelTransformation raster turned 30 degrees, with another pixel
    size, samples the same plane: the terrain is the same."""
    straight = import_terrain(
        write_dem(tmp_path / "a.tif", affine=aligned(), shape=(120, 120)),
        map_preset="small",
        origin=ORIGIN,
    )
    turn, pixel, size = math.radians(30.0), 8.0, 220
    ca, sa = math.cos(turn) * pixel, math.sin(turn) * pixel
    # Columns run along the turned east, rows along the turned south; the
    # raster's centre is on the origin.
    half = size * 0.5
    c = CENTRE[0] - (ca * half + sa * half)
    f = CENTRE[1] - (sa * half - ca * half)
    rotated = import_terrain(
        write_dem(
            tmp_path / "b.tif",
            affine=(ca, sa, c, sa, -ca, f),
            shape=(size, size),
            tie_scale=False,
        ),
        map_preset="small",
        origin=ORIGIN,
    )
    assert rotated["georeference"]["origin_h_m"] == straight["georeference"]["origin_h_m"]
    np.testing.assert_allclose(grid(rotated), grid(straight), atol=2e-3)


def test_pixel_is_point_is_honoured(tmp_path: Path) -> None:
    """PixelIsPoint puts each value on its pixel's corner, half a pixel
    from where PixelIsArea puts it: each file, read by its own raster
    type, is the plane at the true positions."""
    area = import_terrain(
        write_dem(tmp_path / "a.tif", affine=aligned(), shape=(120, 120)),
        map_preset="small",
        origin=ORIGIN,
    )
    point = import_terrain(
        write_dem(tmp_path / "p.tif", affine=aligned(), shape=(120, 120), point=True),
        map_preset="small",
        origin=ORIGIN,
    )
    assert point["summary"]["pixel_is_point"] and not area["summary"]["pixel_is_point"]
    # Both are the plane at the true positions.
    np.testing.assert_allclose(grid(point), expected(point), atol=2e-3)
    np.testing.assert_allclose(grid(area), expected(area), atol=2e-3)


def test_heights_start_just_above_the_ground_plane(tmp_path: Path) -> None:
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    document = import_terrain(dem, map_preset="small", origin=ORIGIN)
    heights = grid(document)
    lowest = document["summary"]["elevation_min_m"]
    assert document["georeference"]["origin_h_m"] == math.floor(lowest) - GROUND_MARGIN_M
    assert GROUND_MARGIN_M <= heights.min() < GROUND_MARGIN_M + 1.0


def test_a_dem_that_does_not_cover_the_map_is_refused(tmp_path: Path) -> None:
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(size=60), shape=(60, 60))
    with pytest.raises(ValueError, match="must cover all of it"):
        import_terrain(dem, map_preset="small", origin=ORIGIN)


def test_the_crs_is_read_or_given_never_guessed(tmp_path: Path) -> None:
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120), crs_key=None)
    with pytest.raises(ValueError, match="CRS"):
        import_terrain(dem, map_preset="small", origin=ORIGIN)
    document = import_terrain(dem, map_preset="small", origin=ORIGIN, source_crs="EPSG:32611")
    np.testing.assert_allclose(grid(document), expected(document), atol=2e-3)


def test_nodata_is_filled_and_counted(tmp_path: Path) -> None:
    holes = ((60, 60), (61, 60), (10, 100))
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120), nodata_cells=holes)
    document = import_terrain(dem, map_preset="small", origin=ORIGIN)
    assert document["summary"]["nodata_pixels_filled"] == len(holes)
    # Filled along its row from its neighbours: on a plane, still the plane.
    np.testing.assert_allclose(grid(document), expected(document), atol=2e-3)


def test_the_default_origin_is_the_dems_centre(tmp_path: Path) -> None:
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    document = import_terrain(dem, map_preset="small")
    georeference = document["georeference"]
    assert georeference["origin_lat_deg"] == pytest.approx(ORIGIN[0], abs=1e-6)
    assert georeference["origin_lon_deg"] == pytest.approx(ORIGIN[1], abs=1e-6)


def test_a_run_stands_on_the_imported_dem(tmp_path: Path) -> None:
    """End to end: the importer's document, run by `argusnet-rs sim`. The
    replay records the import and its origin, and its terrain is the DEM's:
    the viewer mesh's corners are the document's corner heights."""
    import hashlib
    import json
    import os
    import subprocess

    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120))
    document = import_terrain(dem, map_preset="small", origin=ORIGIN)
    terrain_path = tmp_path / "terrain.json"
    terrain_path.write_text(json.dumps(document), encoding="utf-8")
    config = tmp_path / "cfg.json"
    config.write_text(
        json.dumps(
            {
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 7,
                "duration_s": 4.0,
                "options": {
                    "map_preset": "small",
                    "terrain_preset": "urban_flat",
                    "terrain_path": str(terrain_path),
                },
            }
        ),
        encoding="utf-8",
    )
    replay = tmp_path / "replay.json"
    subprocess.run(
        [str(binary), "sim", "--quiet", "--config", str(config), "--replay", str(replay)],
        check=True,
        capture_output=True,
    )
    meta = json.loads(replay.read_text(encoding="utf-8"))["meta"]
    record = meta["terrain_import"]
    assert record["source_sha256"] == hashlib.sha256(dem.read_bytes()).hexdigest()
    assert record["georeference"] == document["georeference"]
    assert meta["enu_origin"]["latitude_deg"] == ORIGIN[0]
    assert meta["enu_origin"]["altitude_m"] == document["georeference"]["origin_h_m"]
    assert meta["terrain_summary"]["source"]["source"] == "dem-import"
    mesh = np.array(meta["terrain"]["viewer_mesh"]["heights_m"])
    heights = grid(document)
    for (i, j), (r, c) in {
        (0, 0): (0, 0),
        (0, -1): (0, -1),
        (-1, 0): (-1, 0),
        (-1, -1): (-1, -1),
    }.items():
        assert mesh[i, j] == pytest.approx(heights[r, c], abs=1e-6)


def test_a_point_registered_dems_default_origin_is_its_samples_centre(tmp_path: Path) -> None:
    """PixelIsPoint samples span 0..cols - 1 in raster space, so their
    middle is half a pixel (5 m here) west and north of the corner extent's."""
    dem = write_dem(tmp_path / "dem.tif", affine=aligned(), shape=(120, 120), point=True)
    document = import_terrain(dem, map_preset="small")
    lon, lat = pyproj.Transformer.from_crs("EPSG:32611", "EPSG:4326", always_xy=True).transform(
        CENTRE[0] - 5.0, CENTRE[1] + 5.0
    )
    georeference = document["georeference"]
    assert georeference["origin_lat_deg"] == pytest.approx(lat, abs=2e-7)
    assert georeference["origin_lon_deg"] == pytest.approx(lon, abs=2e-7)
