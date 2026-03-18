from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from smart_tracker.scene import _build_obstacle_meshes, build_scene_from_gis, build_scene_from_replay, load_scene_manifest

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
                "metrics": {},
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

            style_document = json.loads((scene_root / "metadata" / "style.json").read_text(encoding="utf-8"))
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
            self.assertTrue(any(layer["style_id"] == "landcover-forest" for layer in manifest["layers"]))
            self.assertTrue(any(layer["style_id"] == "buildings" for layer in manifest["layers"]))

            environment_document = json.loads((output_path / "metadata" / "environment.json").read_text(encoding="utf-8"))
            self.assertEqual(["EPSG:32611"], environment_document["overlay_source_crs"]["landcover"])
            self.assertEqual(1, environment_document["overlay_counts"]["landcover"])


if __name__ == "__main__":
    unittest.main()
