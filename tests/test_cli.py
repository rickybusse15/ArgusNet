from __future__ import annotations

import unittest
from unittest.mock import patch

import argusnet.cli.main as cli


class CliImportBehaviorTest(unittest.TestCase):
    def test_build_scene_args_parse_without_sim_runtime_import(self) -> None:
        with patch(
            "argusnet.cli.main._import_sim_module",
            side_effect=RuntimeError(cli.PYTHON_DEPENDENCY_INSTALL_HINT),
        ):
            args = cli.parse_args(["build-scene", "--replay", "demo.json", "--output", "scene-out"])

        self.assertEqual("build-scene", args.command)
        self.assertEqual("demo.json", args.replay)
        self.assertEqual("scene-out", args.output)

    def test_sim_main_exits_with_clear_dependency_hint(self) -> None:
        with patch(
            "argusnet.cli.main._import_sim_module",
            side_effect=RuntimeError(
                "Simulation requires the Python dependency `numpy`. "
                + cli.PYTHON_DEPENDENCY_INSTALL_HINT
            ),
        ):
            with self.assertRaises(SystemExit) as context:
                cli.main(["sim", "--duration-s", "5"])

        self.assertIn("python3 -m pip install --user -e .", str(context.exception))
        self.assertIn("numpy", str(context.exception))

    def test_build_scene_main_dispatches_to_scene_module(self) -> None:
        scene_module = unittest.mock.Mock()
        with patch("argusnet.cli.main._import_scene_module", return_value=scene_module):
            cli.main(["build-scene", "--replay", "demo.json", "--output", "scene-out"])

        scene_module.build_scene_package.assert_called_once()

    def test_batch_export_args_parse(self) -> None:
        args = cli.parse_args(
            [
                "batch-export",
                "--replay",
                "demo.json",
                "--enu-origin",
                "47.0,8.0,400.0",
                "--formats",
                "geojson,kmz",
                "--output-dir",
                "exports",
            ]
        )
        self.assertEqual("batch-export", args.command)
        self.assertEqual("geojson,kmz", args.formats)
        self.assertEqual("exports", args.output_dir)

    def test_export_shapefile_requires_directory_output(self) -> None:
        replay_module = unittest.mock.Mock()
        replay_module.load_replay_document.return_value = {"meta": {}, "frames": []}
        with patch.dict("sys.modules", {"argusnet.evaluation.replay": replay_module}):
            with self.assertRaises(SystemExit) as context:
                cli.main(
                    [
                        "export",
                        "--replay",
                        "demo.json",
                        "--format",
                        "shapefile",
                        "--enu-origin",
                        "47.0,8.0,400.0",
                        "--output",
                        "tracks.shp",
                    ]
                )
        self.assertIn("directory", str(context.exception))


if __name__ == "__main__":
    unittest.main()
