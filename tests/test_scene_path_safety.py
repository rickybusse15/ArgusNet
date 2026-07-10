"""Tests for scene manifest path-traversal protections."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from argusnet.world.scene_loader import (
    load_scene_manifest,
    resolve_scene_asset_path,
    validate_scene_manifest,
)

_BOUNDS = {"x_min_m": 0.0, "y_min_m": 0.0, "x_max_m": 1.0, "y_max_m": 1.0}


def _manifest(asset_path: str = "terrain/mesh.glb") -> dict:
    return {
        "format_version": "smartscene-v1",
        "scene_id": "scene-under-test",
        "bounds_xy_m": _BOUNDS,
        "layers": [{"id": "l1", "asset_path": asset_path, "style_id": "s1"}],
        "metadata": {"environment": "environment.json", "style": "style.json"},
    }


class TestValidateSceneManifestPaths(unittest.TestCase):
    def test_normal_relative_path_is_accepted(self):
        validate_scene_manifest(_manifest("terrain/mesh.glb"))  # should not raise

    def test_absolute_asset_path_rejected(self):
        with self.assertRaises(ValueError):
            validate_scene_manifest(_manifest("/etc/passwd"))

    def test_dotdot_asset_path_rejected(self):
        with self.assertRaises(ValueError):
            validate_scene_manifest(_manifest("../../../../etc/passwd"))

    def test_dotdot_in_middle_of_path_rejected(self):
        with self.assertRaises(ValueError):
            validate_scene_manifest(_manifest("terrain/../../escape.glb"))

    def test_windows_style_absolute_path_rejected(self):
        with self.assertRaises(ValueError):
            validate_scene_manifest(_manifest("C:\\Windows\\System32\\config"))

    def test_environment_and_style_paths_are_checked(self):
        manifest = _manifest()
        manifest["metadata"]["environment"] = "../outside.json"
        with self.assertRaises(ValueError):
            validate_scene_manifest(manifest)

    def test_replay_path_is_checked_when_present(self):
        manifest = _manifest()
        manifest["replay"] = {"path": "../../outside_replay.json"}
        with self.assertRaises(ValueError):
            validate_scene_manifest(manifest)


class TestResolveSceneAssetPath(unittest.TestCase):
    def test_contained_path_resolves(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "terrain").mkdir()
            resolved = resolve_scene_asset_path(root, "terrain/mesh.glb")
            self.assertEqual(resolved, (root / "terrain" / "mesh.glb").resolve())

    def test_escaping_path_raises_even_without_dotdot_string_check_bypassed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "scene"
            root.mkdir()
            with self.assertRaises(ValueError):
                resolve_scene_asset_path(root, "../outside.glb")


class TestLoadSceneManifestRejectsTraversal(unittest.TestCase):
    def test_load_scene_manifest_blocks_asset_path_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            scene_dir.mkdir()
            manifest_path = scene_dir / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest("../../../../etc/passwd")))
            with self.assertRaises(ValueError):
                load_scene_manifest(manifest_path)

    def test_load_scene_manifest_accepts_well_formed_package(self):
        with tempfile.TemporaryDirectory() as tmp:
            scene_dir = Path(tmp) / "scene"
            scene_dir.mkdir()
            manifest_path = scene_dir / "manifest.json"
            manifest_path.write_text(json.dumps(_manifest("terrain/mesh.glb")))
            document = load_scene_manifest(manifest_path)
            self.assertEqual(document["scene_id"], "scene-under-test")


if __name__ == "__main__":
    unittest.main()
