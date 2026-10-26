"""Regression tests for the attack paths left after the first security pass.

Covers environment-bundle traversal and decompression bombs, scene
``land_cover`` paths, GeoJSON size caps, YAML alias expansion, and ``.npy``
headers that name a huge array.
"""

from __future__ import annotations

import json
import unittest
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import yaml

from argusnet.core.config import load_config_mapping
from argusnet.core.io_limits import FileSizeLimitError
from argusnet.mapping.world_model import _load_npy
from argusnet.world._scene_gis import load_geojson_layer
from argusnet.world.environment import (
    Bounds2D,
    EnvironmentCRS,
    EnvironmentModel,
    LandCoverClass,
    LandCoverLayer,
    ObstacleLayer,
    TerrainLayer,
)
from argusnet.world.environment_io import (
    _MAX_DECOMPRESSED_BLOB_BYTES,
    _array_from_blob,
    load_environment_bundle,
    write_environment_bundle,
)
from argusnet.world.scene_loader import validate_scene_manifest


def _bounds() -> Bounds2D:
    return Bounds2D(x_min_m=0.0, y_min_m=0.0, x_max_m=20.0, y_max_m=20.0)


def _bundle(root: Path) -> Path:
    bounds = _bounds()
    terrain = TerrainLayer.from_height_grid(
        environment_id="bundle",
        bounds_xy_m=bounds,
        heights_m=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=float),
        resolution_m=10.0,
        tile_size_cells=2,
    )
    classes = np.array([[int(LandCoverClass.OPEN)]], dtype=np.uint8)
    land_cover = LandCoverLayer.from_rasters(
        bounds_xy_m=bounds,
        classes=classes,
        density=None,
        resolution_m=20.0,
        tile_size_cells=1,
    )
    environment = EnvironmentModel(
        environment_id="bundle-scene",
        crs=EnvironmentCRS(runtime_crs_id="local-enu"),
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=20.0, primitives=()),
        land_cover=land_cover,
    )
    bundle = root / "map"
    write_environment_bundle(str(bundle), environment)
    return bundle


def _manifest() -> dict:
    return {
        "format_version": "smartscene-v1",
        "scene_id": "scene-under-test",
        "bounds_xy_m": {"x_min_m": 0.0, "y_min_m": 0.0, "x_max_m": 1.0, "y_max_m": 1.0},
        "layers": [{"id": "l1", "asset_path": "terrain/mesh.glb", "style_id": "s1"}],
        "metadata": {"environment": "environment.json", "style": "style.json"},
    }


class TestEnvironmentBundleContainment(unittest.TestCase):
    def test_tile_path_dotdot_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            manifest_path = bundle / "environment_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["terrain"]["tiles"][0]["path"] = "../outside.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            (root / "outside.json").write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_environment_bundle(str(bundle))

    def test_tile_symlink_escape_is_rejected(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle = _bundle(root)
            manifest = json.loads(
                (bundle / "environment_manifest.json").read_text(encoding="utf-8")
            )
            relative = manifest["terrain"]["tiles"][0]["path"]
            tile = bundle / relative
            outside = root / "outside.json"
            tile.replace(outside)
            tile.symlink_to(outside)
            with self.assertRaises(ValueError):
                load_environment_bundle(str(bundle))

    def test_round_trip_still_loads(self):
        with TemporaryDirectory() as tmp:
            bundle = _bundle(Path(tmp))
            loaded = load_environment_bundle(str(bundle))
            self.assertEqual(loaded.environment_id, "bundle-scene")


class TestArrayBlobLimits(unittest.TestCase):
    def test_decompression_over_the_cap_is_rejected(self):
        raw = b"\x00" * 4000
        blob = {
            "shape": [4000],
            "dtype": "uint8",
            "compression": "zlib",
            "payload_hex": zlib.compress(raw).hex(),
        }
        with (
            patch("argusnet.world.environment_io._MAX_DECOMPRESSED_BLOB_BYTES", 64),
            self.assertRaises(ValueError),
        ):
            _array_from_blob(blob)
        # The production cap still accepts this small blob.
        self.assertGreater(_MAX_DECOMPRESSED_BLOB_BYTES, len(raw))
        decoded = _array_from_blob(blob)
        self.assertEqual(int(decoded[0]), 0)
        self.assertEqual(decoded.shape, (4000,))

    def test_object_dtype_is_rejected(self):
        blob = {
            "shape": [1],
            "dtype": "O",
            "compression": "zlib",
            "payload_hex": zlib.compress(b"\x00").hex(),
        }
        with self.assertRaises(ValueError):
            _array_from_blob(blob)


class TestSceneLandCoverPath(unittest.TestCase):
    def test_land_cover_dotdot_is_rejected(self):
        manifest = _manifest()
        manifest["metadata"]["land_cover"] = "../outside.json"
        with self.assertRaises(ValueError):
            validate_scene_manifest(manifest)


class TestGeoJsonSizeCap(unittest.TestCase):
    def test_oversized_geojson_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "features.geojson"
            path.write_text('{"type":"FeatureCollection","features":[]}', encoding="utf-8")

            def tiny(p, **kwargs):
                from argusnet.core.io_limits import read_text_capped

                return read_text_capped(p, max_bytes=8)

            with (
                patch("argusnet.world._scene_gis.read_text_capped", tiny),
                self.assertRaises(FileSizeLimitError),
            ):
                load_geojson_layer(path)


class TestYamlAliases(unittest.TestCase):
    def test_alias_expansion_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "constants.yaml"
            path.write_text('a: &a ["x", "x"]\nb: [*a, *a, *a, *a]\n', encoding="utf-8")
            with self.assertRaises(yaml.YAMLError):
                load_config_mapping(str(path))

    def test_plain_mapping_still_loads(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "constants.yaml"
            path.write_text("sensor:\n  note: 1\n", encoding="utf-8")
            loaded = load_config_mapping(str(path))
            self.assertEqual(loaded["sensor"]["note"], 1)


class TestNpyHeaderCap(unittest.TestCase):
    def test_normal_array_loads(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "grid.npy"
            np.save(path, np.arange(6, dtype=np.float64))
            loaded = _load_npy(path)
            self.assertEqual(loaded.shape, (6,))

    def test_declared_huge_shape_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bomb.npy"
            with path.open("wb") as handle:
                np.lib.format.write_array_header_1_0(
                    handle,
                    {"descr": "<f8", "fortran_order": False, "shape": (10**9,)},
                )
            with self.assertRaises(ValueError):
                _load_npy(path)

    def test_object_array_is_rejected(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "objects.npy"
            np.save(path, np.array(["secret"], dtype=object))
            with self.assertRaises(ValueError):
                _load_npy(path)
