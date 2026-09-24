from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from argusnet.world.scene_loader import (
    _build_obstacle_meshes,
    build_scene_from_gis,
    build_scene_from_replay,
    load_scene_manifest,
)

try:
    import tifffile
except ImportError:  # pragma: no cover - dependency gating only
    tifffile = None


def _sample_replay_document() -> dict:
    return {
        "meta": {
            "scenario_name": "demo-scene",
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": 1,
            "dt_s": 0.5,
            "seed": 7,
            "crs_id": "local-enu",
            "environment_bounds_m": {
                "x_min_m": -100.0,
                "x_max_m": 100.0,
                "y_min_m": -80.0,
                "y_max_m": 80.0,
            },
            "terrain": {
                "kind": "analytic-terrain",
                "base_resolution_m": 20.0,
                "min_height_m": 100.0,
                "max_height_m": 140.0,
                "viewer_mesh": {
                    "rows": 3,
                    "cols": 3,
                    "x_min_m": -100.0,
                    "x_max_m": 100.0,
                    "y_min_m": -80.0,
                    "y_max_m": 80.0,
                    "heights_m": [
                        [100.0, 108.0, 112.0],
                        [102.0, 116.0, 130.0],
                        [104.0, 120.0, 140.0],
                    ],
                },
            },
            "occluding_objects": [],
        },
        "summary": {},
        "frames": [
            {
                "timestamp_s": 0.0,
                "tracks": [
                    {
                        "track_id": "track-1",
                        "timestamp_s": 0.0,
                        "position": [10.0, 5.0, 120.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "covariance": [1.0] * 9,
                        "measurement_std_m": 1.0,
                        "update_count": 1,
                        "stale_steps": 0,
                    }
                ],
                "truths": [
                    {
                        "target_id": "truth-1",
                        "timestamp_s": 0.0,
                        "position": [8.0, 4.0, 118.0],
                        "velocity": [0.0, 0.0, 0.0],
                    }
                ],
                "nodes": [
                    {
                        "node_id": "node-1",
                        "timestamp_s": 0.0,
                        "position": [0.0, 0.0, 115.0],
                        "velocity": [0.0, 0.0, 0.0],
                        "is_mobile": False,
                    }
                ],
                "observations": [],
                "rejected_observations": [],
                "metrics": {
                    "active_track_count": 1,
                    "observation_count": 0,
                    "accepted_observation_count": 0,
                    "rejected_observation_count": 0,
                },
                "generation_rejections": [],
            }
        ],
    }


class SceneObstacleMeshTest(unittest.TestCase):
    def test_ground_contact_obstacle_mesh_spans_terrain_relief(self) -> None:
        footprint = np.array(
            [
                [-20.0, -15.0],
                [18.0, -10.0],
                [22.0, 14.0],
                [-16.0, 18.0],
            ],
            dtype=np.float32,
        )

        def height_at(x_m: float, y_m: float) -> float:
            return 120.0 + 0.08 * x_m + 0.14 * y_m

        layers = _build_obstacle_meshes(
            [
                {
                    "object_id": "warehouse-a",
                    "blocker_type": "building",
                    "footprint_xy_m": footprint.tolist(),
                    "base_elevation_m": 123.5,
                    "top_elevation_m": 151.5,
                }
            ],
            height_at=height_at,
        )

        self.assertEqual(1, len(layers))
        z_values = layers[0].mesh.positions[:, 2]
        sampled_heights = [height_at(float(x_m), float(y_m)) for x_m, y_m in footprint]
        self.assertAlmostEqual(min(sampled_heights), float(np.min(z_values)), places=4)
        self.assertGreaterEqual(float(np.max(z_values)), max(sampled_heights) + 3.0 - 1.0e-6)

    def test_vegetation_mesh_is_grounded_to_local_terrain_relief(self) -> None:
        footprint = np.array(
            [
                [-32.0, -10.0],
                [20.0, -16.0],
                [28.0, 12.0],
                [-18.0, 24.0],
            ],
            dtype=np.float32,
        )

        def height_at(x_m: float, y_m: float) -> float:
            return 96.0 + 0.11 * x_m + 0.19 * y_m

        layers = _build_obstacle_meshes(
            [
                {
                    "object_id": "forest-a",
                    "kind": "forest-stand-v1",
                    "blocker_type": "vegetation",
                    "footprint_xy_m": footprint.tolist(),
                    "base_elevation_m": 118.0,
                    "top_elevation_m": 130.0,
                }
            ],
            height_at=height_at,
        )

        self.assertEqual(1, len(layers))
        z_values = layers[0].mesh.positions[:, 2]
        sampled_heights = [height_at(float(x_m), float(y_m)) for x_m, y_m in footprint]
        self.assertAlmostEqual(min(sampled_heights), float(np.min(z_values)), places=4)
        self.assertGreaterEqual(float(np.max(z_values)), max(sampled_heights) + 3.0 - 1.0e-6)


@unittest.skipUnless(tifffile is not None, "requires tifffile")
class SceneCompilationTest(unittest.TestCase):
    def test_build_scene_from_replay_writes_standard_bundle(self) -> None:
        replay_document = _sample_replay_document()
        with tempfile.TemporaryDirectory(prefix="smartscene-replay-") as temp_dir:
            manifest = build_scene_from_replay(replay_document, temp_dir)
            scene_root = Path(temp_dir)

            self.assertEqual("smartscene-v1", manifest["format_version"])
            self.assertTrue((scene_root / "scene_manifest.json").exists())
            self.assertTrue((scene_root / "terrain" / "terrain-base.glb").exists())
            self.assertTrue((scene_root / "metadata" / "style.json").exists())
            self.assertTrue((scene_root / "replay" / "replay.json").exists())

            loaded_manifest = load_scene_manifest(scene_root / "scene_manifest.json")
            self.assertEqual("demo-scene", loaded_manifest["scene_id"])
            self.assertEqual("terrain-base", loaded_manifest["layers"][0]["style_id"])

            style_document = json.loads(
                (scene_root / "metadata" / "style.json").read_text(encoding="utf-8")
            )
            style_ids = {layer["id"] for layer in style_document["layers"]}
            self.assertTrue({"terrain-base", "tracks", "truths", "nodes"}.issubset(style_ids))

    def test_build_scene_from_gis_chunks_terrain_and_classifies_landcover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="smartscene-gis-") as temp_dir:
            temp_root = Path(temp_dir)
            dem_path = temp_root / "dem.tif"
            landcover_path = temp_root / "landcover.geojson"
            buildings_path = temp_root / "buildings.geojson"
            output_path = temp_root / "scene"

            heights = np.arange(70 * 70, dtype=np.float32).reshape(70, 70) * 0.5 + 100.0
            tifffile.imwrite(
                dem_path,
                heights,
                extratags=[
                    (33550, "d", 3, (30.0, 30.0, 0.0), False),
                    (33922, "d", 6, (0.0, 0.0, 0.0, 500000.0, 3800000.0, 0.0), False),
                    (34735, "H", 8, (1, 1, 0, 1, 3072, 0, 1, 32611), False),
                ],
            )
            landcover_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "crs": {"type": "name", "properties": {"name": "EPSG:32611"}},
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"class": "forest"},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [500000.0, 3800000.0],
                                            [500180.0, 3800000.0],
                                            [500180.0, 3799820.0],
                                            [500000.0, 3799820.0],
                                            [500000.0, 3800000.0],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            buildings_path.write_text(
                json.dumps(
                    {
                        "type": "FeatureCollection",
                        "features": [
                            {
                                "type": "Feature",
                                "properties": {"height_m": 18.0},
                                "geometry": {
                                    "type": "Polygon",
                                    "coordinates": [
                                        [
                                            [500030.0, 3799970.0],
                                            [500090.0, 3799970.0],
                                            [500090.0, 3799910.0],
                                            [500030.0, 3799910.0],
                                            [500030.0, 3799970.0],
                                        ]
                                    ],
                                },
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )

            manifest = build_scene_from_gis(
                dem_path,
                output_path,
                overlay_paths={"landcover": [landcover_path], "buildings": [buildings_path]},
            )

            terrain_layers = [layer for layer in manifest["layers"] if layer["kind"] == "terrain"]
            self.assertGreater(len(terrain_layers), 1)
            self.assertTrue(
                any(layer["style_id"] == "landcover-forest" for layer in manifest["layers"])
            )
            self.assertTrue(any(layer["style_id"] == "buildings" for layer in manifest["layers"]))

            environment_document = json.loads(
                (output_path / "metadata" / "environment.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                ["EPSG:32611"], environment_document["overlay_source_crs"]["landcover"]
            )
            self.assertEqual(1, environment_document["overlay_counts"]["landcover"])


class SceneObstacleRoundTripTest(unittest.TestCase):
    """A packaged scene's obstacle metadata must rebuild through its own loader."""

    @staticmethod
    def _height_at(x_m: float, y_m: float) -> float:
        return 110.0 + 0.05 * x_m + 0.09 * y_m

    @staticmethod
    def _replay_with_obstacles() -> dict:
        replay_document = _sample_replay_document()
        replay_document["meta"]["occluding_objects"] = [
            {
                "object_id": "warehouse-north",
                "kind": "polygon-prism-v1",
                "blocker_type": "building",
                "footprint_xy_m": [[-60.0, -40.0], [40.0, -40.0], [40.0, 30.0], [-60.0, 30.0]],
                "base_elevation_m": 100.0,
                "top_elevation_m": 128.0,
            },
            {
                "object_id": "pine-stand-west",
                "kind": "forest-stand-v1",
                "blocker_type": "vegetation",
                "footprint_xy_m": [[-90.0, 10.0], [-50.0, 4.0], [-44.0, 52.0], [-84.0, 60.0]],
                "base_elevation_m": 104.0,
                "top_elevation_m": 122.0,
                "density": 0.58,
            },
        ]
        return replay_document

    def test_packaged_metadata_keeps_footprints_as_xy_pairs(self) -> None:
        replay_document = self._replay_with_obstacles()
        with tempfile.TemporaryDirectory() as temp_dir:
            build_scene_from_replay(replay_document, temp_dir)
            environment_document = json.loads(
                (Path(temp_dir) / "metadata" / "environment.json").read_text(encoding="utf-8")
            )

        packaged = environment_document["occluding_objects"]
        self.assertEqual(2, len(packaged))
        for source, written in zip(
            replay_document["meta"]["occluding_objects"], packaged, strict=True
        ):
            self.assertEqual(source["footprint_xy_m"], written["footprint_xy_m"])

        # The nested replay_meta copy must agree with the top-level one, and with
        # the canonical replay document the package also carries.
        for written in environment_document["replay_meta"]["occluding_objects"]:
            self.assertTrue(all(len(pair) == 2 for pair in written["footprint_xy_m"]))

    def test_obstacle_meshes_rebuild_from_packaged_metadata(self) -> None:
        replay_document = self._replay_with_obstacles()
        with tempfile.TemporaryDirectory() as temp_dir:
            build_scene_from_replay(replay_document, temp_dir)
            environment_document = json.loads(
                (Path(temp_dir) / "metadata" / "environment.json").read_text(encoding="utf-8")
            )

        layers = _build_obstacle_meshes(
            environment_document["occluding_objects"], height_at=self._height_at
        )

        self.assertEqual({"buildings", "vegetation"}, {layer.layer_id for layer in layers})
        for layer in layers:
            self.assertGreater(len(layer.mesh.positions), 0)
            self.assertGreater(len(layer.mesh.indices), 0)

    def test_flat_footprints_from_existing_packages_still_build(self) -> None:
        """Scene packages written before the fix store footprints flattened."""
        nested = [[-20.0, -15.0], [18.0, -10.0], [22.0, 14.0], [-16.0, 18.0]]
        flat = [value for pair in nested for value in pair]

        def build(footprint):
            return _build_obstacle_meshes(
                [
                    {
                        "object_id": "warehouse-a",
                        "blocker_type": "building",
                        "footprint_xy_m": footprint,
                        "base_elevation_m": 123.5,
                        "top_elevation_m": 151.5,
                    }
                ],
                height_at=self._height_at,
            )

        nested_layers = build(nested)
        flat_layers = build(flat)

        self.assertEqual(1, len(flat_layers))
        np.testing.assert_allclose(nested_layers[0].mesh.positions, flat_layers[0].mesh.positions)

    def test_odd_length_flat_footprint_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            _build_obstacle_meshes(
                [
                    {
                        "object_id": "warehouse-a",
                        "blocker_type": "building",
                        "footprint_xy_m": [0.0, 0.0, 10.0, 0.0, 10.0],
                        "base_elevation_m": 100.0,
                        "top_elevation_m": 110.0,
                    }
                ],
                height_at=self._height_at,
            )

    @unittest.skipUnless(
        Path("scene.smartscene/metadata/environment.json").exists(),
        "checked-in scene package not present",
    )
    def test_checked_in_scene_package_rebuilds(self) -> None:
        environment_document = json.loads(
            Path("scene.smartscene/metadata/environment.json").read_text(encoding="utf-8")
        )
        layers = _build_obstacle_meshes(
            environment_document["occluding_objects"], height_at=self._height_at
        )
        self.assertTrue(any(len(layer.mesh.positions) for layer in layers))


if __name__ == "__main__":
    unittest.main()


class LandCoverDocumentTest(unittest.TestCase):
    """P2.1: a scene built from an environment carries its land-cover classes
    for the terrain material, as the bundle's own `land_cover` payload."""

    def _environment(self):
        from argusnet.world.environment import (
            Bounds2D,
            EnvironmentCRS,
            EnvironmentModel,
            LandCoverClass,
            LandCoverLayer,
            ObstacleLayer,
            TerrainLayer,
        )

        bounds = Bounds2D(-100.0, 100.0, -80.0, 80.0)
        terrain = TerrainLayer.from_height_grid(
            environment_id="lc",
            bounds_xy_m=bounds,
            heights_m=np.full((17, 21), 110.0),
            resolution_m=10.0,
            tile_size_cells=8,
        )
        rows, cols = 16, 20
        classes = np.zeros((rows, cols), dtype=np.uint8)
        classes[:, 10:] = int(LandCoverClass.FOREST)
        classes[4:8, :] = int(LandCoverClass.WATER)
        classes[12, 3] = int(LandCoverClass.ROAD)
        land_cover = LandCoverLayer.from_rasters(
            bounds_xy_m=bounds,
            classes=classes,
            density=None,
            resolution_m=10.0,
            tile_size_cells=8,
        )
        return EnvironmentModel(
            environment_id="lc",
            crs=EnvironmentCRS(runtime_crs_id="local-enu"),
            bounds_xy_m=bounds,
            terrain=terrain,
            obstacles=ObstacleLayer(bounds_xy_m=bounds, tile_size_m=20.0, primitives=()),
            land_cover=land_cover,
        )

    def test_a_scene_built_from_an_environment_carries_its_land_cover(self) -> None:
        from argusnet.world.environment_io import write_environment_bundle

        environment = self._environment()
        with tempfile.TemporaryDirectory(prefix="smartscene-lc-") as temp_dir:
            bundle = Path(temp_dir) / "map"
            write_environment_bundle(str(bundle), environment)
            scene = Path(temp_dir) / "scene"
            manifest = build_scene_from_replay(
                _sample_replay_document(), scene, environment_bundle=bundle
            )
            self.assertEqual("metadata/land_cover.json", manifest["metadata"]["land_cover"])
            document = json.loads((scene / "metadata" / "land_cover.json").read_text())

        self.assertEqual(("argusnet.land-cover", 1), (document["format"], document["version"]))
        layer = environment.land_cover
        self.assertEqual(layer.tile_size_cells, document["tile_size_cells"])
        self.assertEqual(layer.base_resolution_m, document["resolution"])
        tiles = {(t["tx"], t["ty"]): t for t in document["tiles"]}
        self.assertEqual(set(layer._tiles), set(tiles))
        for key, tile in layer._tiles.items():
            np.testing.assert_array_equal(tile.classes, np.asarray(tiles[key]["classes"]))
            self.assertEqual(tile.x_min_m, tiles[key]["x_min_m"])
        self.assertEqual(9, len(document["legend"]))
        self.assertEqual(set(document["legend"]), set(document["palette"]))
        self.assertEqual(4, len(document["palette"]["forest"]))

    def test_a_scene_built_without_an_environment_declares_no_land_cover(self) -> None:
        with tempfile.TemporaryDirectory(prefix="smartscene-nolc-") as temp_dir:
            manifest = build_scene_from_replay(_sample_replay_document(), temp_dir)
            self.assertNotIn("land_cover", manifest["metadata"])
            self.assertFalse((Path(temp_dir) / "metadata" / "land_cover.json").exists())
