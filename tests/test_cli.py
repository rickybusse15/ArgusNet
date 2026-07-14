from __future__ import annotations

import unittest
from unittest.mock import patch

import argusnet.cli.main as cli
from argusnet.security.identity import DeviceRegistry


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
        with (
            patch(
                "argusnet.cli.main._import_sim_module",
                side_effect=RuntimeError(
                    "Simulation requires the Python dependency `numpy`. "
                    + cli.PYTHON_DEPENDENCY_INSTALL_HINT
                ),
            ),
            self.assertRaises(SystemExit) as context,
        ):
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
        with (
            patch.dict("sys.modules", {"argusnet.evaluation.replay": replay_module}),
            self.assertRaises(SystemExit) as context,
        ):
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


class IngestMqttDeviceRegistryTest(unittest.TestCase):
    """`ingest --mqtt-broker` must be able to load a DeviceRegistry, since
    MQTTIngestionAdapter now refuses a non-loopback broker without one."""

    def _run_ingest(self, argv, env=None):
        adapter_cls = unittest.mock.Mock()
        with (
            patch("argusnet.sensing.ingestion.frame_stream.MQTTIngestionAdapter", adapter_cls),
            patch("argusnet.sensing.ingestion.frame_stream.LiveIngestionRunner"),
            patch("argusnet.adapters.argusnet_grpc.TrackingService"),
            patch.object(
                DeviceRegistry, "from_directory", return_value=unittest.mock.sentinel.registry
            ) as from_directory,
            patch.dict("os.environ", env or {}, clear=False),
        ):
            args = cli.parse_args(argv)
            cli._run_ingest(args)
        return adapter_cls, from_directory

    def test_cli_flag_loads_device_registry(self):
        adapter_cls, from_directory = self._run_ingest(
            [
                "ingest",
                "--mqtt-broker",
                "mqtt.example.com",
                "--enu-origin",
                "0,0,0",
                "--device-registry",
                "/some/registry/dir",
            ]
        )
        from_directory.assert_called_once_with("/some/registry/dir")
        self.assertIs(
            adapter_cls.call_args.kwargs["device_registry"], unittest.mock.sentinel.registry
        )

    def test_env_var_fallback_loads_device_registry(self):
        adapter_cls, from_directory = self._run_ingest(
            ["ingest", "--mqtt-broker", "mqtt.example.com", "--enu-origin", "0,0,0"],
            env={"ARGUSNET_MQTT_DEVICE_REGISTRY": "/env/registry/dir"},
        )
        from_directory.assert_called_once_with("/env/registry/dir")
        self.assertIs(
            adapter_cls.call_args.kwargs["device_registry"], unittest.mock.sentinel.registry
        )

    def test_cli_flag_takes_precedence_over_env_var(self):
        adapter_cls, from_directory = self._run_ingest(
            [
                "ingest",
                "--mqtt-broker",
                "mqtt.example.com",
                "--enu-origin",
                "0,0,0",
                "--device-registry",
                "/flag/registry/dir",
            ],
            env={"ARGUSNET_MQTT_DEVICE_REGISTRY": "/env/registry/dir"},
        )
        from_directory.assert_called_once_with("/flag/registry/dir")

    def test_no_registry_configured_passes_none(self):
        adapter_cls, from_directory = self._run_ingest(
            ["ingest", "--mqtt-broker", "mqtt.example.com", "--enu-origin", "0,0,0"]
        )
        from_directory.assert_not_called()
        self.assertIsNone(adapter_cls.call_args.kwargs["device_registry"])


if __name__ == "__main__":
    unittest.main()
