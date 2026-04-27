"""Tests for CLI usability improvements (Phase 4B).

Covers: validate-replay, info, dump-config, validate-scene, --dry-run,
        --verbose/--quiet, normalize_argv with new commands, and
        the expanded export format choices (kml, gpx).
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from argusnet.cli.main import (
    ALL_COMMANDS,
    COMMAND_DUMP_CONFIG,
    COMMAND_INFO,
    COMMAND_VALIDATE_REPLAY,
    COMMAND_VALIDATE_SCENE,
    build_parser,
    normalize_argv,
)


def _minimal_valid_replay_document() -> dict:
    return {
        "meta": {
            "scenario_name": "unit-test",
            "generated_at_utc": "2025-01-15T12:00:00+00:00",
            "frame_count": 1,
            "dt_s": 0.25,
            "seed": 42,
            "node_ids": [],
            "track_ids": [],
        },
        "frames": [
            {
                "timestamp_s": 0.0,
                "nodes": [],
                "observations": [],
                "rejected_observations": [],
                "tracks": [],
                "truths": [],
                "metrics": {
                    "mean_error_m": None,
                    "max_error_m": None,
                    "active_track_count": 0,
                    "observation_count": 0,
                    "accepted_observation_count": 0,
                    "rejected_observation_count": 0,
                    "mean_measurement_std_m": None,
                    "track_errors_m": {},
                    "rejection_counts": {},
                    "accepted_observations_by_target": {},
                    "rejected_observations_by_target": {},
                },
                "generation_rejections": [],
                "mapping_state": None,
                "localization_state": None,
                "inspection_events": [],
            }
        ],
    }


def _write_scene_bundle(
    bundle_root: Path,
    *,
    manifest_name: str = "scene_manifest.json",
    include_asset: bool = True,
    include_metadata: bool = True,
) -> None:
    if include_asset:
        terrain_dir = bundle_root / "terrain"
        terrain_dir.mkdir(parents=True, exist_ok=True)
        (terrain_dir / "terrain-base.glb").write_bytes(b"glb")
    if include_metadata:
        metadata_dir = bundle_root / "metadata"
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (metadata_dir / "environment.json").write_text("{}", encoding="utf-8")
        (metadata_dir / "style.json").write_text("{}", encoding="utf-8")

    manifest = {
        "format_version": "smartscene-v1",
        "scene_id": "test-scene",
        "bounds_xy_m": {
            "x_min_m": 0.0,
            "x_max_m": 10.0,
            "y_min_m": 0.0,
            "y_max_m": 10.0,
        },
        "runtime_crs": {"runtime_crs_id": "local-enu"},
        "source_crs_id": "local-synthetic",
        "layers": [
            {
                "id": "terrain-base",
                "kind": "terrain",
                "asset_path": "terrain/terrain-base.glb",
                "style_id": "terrain-base",
            }
        ],
        "metadata": {
            "environment": "metadata/environment.json",
            "style": "metadata/style.json",
        },
    }
    (bundle_root / manifest_name).write_text(json.dumps(manifest), encoding="utf-8")


class TestNormalizeArgv(unittest.TestCase):
    def test_new_commands_recognized(self):
        for cmd in (
            COMMAND_VALIDATE_SCENE,
            COMMAND_VALIDATE_REPLAY,
            COMMAND_INFO,
            COMMAND_DUMP_CONFIG,
        ):
            result = normalize_argv([cmd, "arg"])
            self.assertEqual(result[0], cmd)

    def test_unknown_args_default_to_sim(self):
        result = normalize_argv(["--duration-s", "30"])
        self.assertEqual(result[0], "sim")

    def test_all_commands_constant(self):
        self.assertIn("validate-scene", ALL_COMMANDS)
        self.assertIn("validate-replay", ALL_COMMANDS)
        self.assertIn("info", ALL_COMMANDS)
        self.assertIn("dump-config", ALL_COMMANDS)


class TestBuildParser(unittest.TestCase):
    def test_parser_has_new_subcommands(self):
        parser = build_parser()
        # Parsing should not fail for new subcommands
        args = parser.parse_args(["validate-replay", "/tmp/test.json"])
        self.assertEqual(args.command, "validate-replay")
        self.assertEqual(args.path, "/tmp/test.json")

    def test_info_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["info", "/tmp/replay.json"])
        self.assertEqual(args.command, "info")
        self.assertEqual(args.path, "/tmp/replay.json")

    def test_validate_scene_subcommand(self):
        parser = build_parser()
        args = parser.parse_args(["validate-scene", "/tmp/scene_dir"])
        self.assertEqual(args.command, "validate-scene")
        self.assertEqual(args.path, "/tmp/scene_dir")

    def test_dump_config_default_format(self):
        parser = build_parser()
        args = parser.parse_args(["dump-config"])
        self.assertEqual(args.command, "dump-config")
        self.assertEqual(args.format, "json")

    def test_dump_config_yaml_format(self):
        parser = build_parser()
        args = parser.parse_args(["dump-config", "--format", "yaml"])
        self.assertEqual(args.format, "yaml")

    def test_export_accepts_kml_gpx(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "export",
                "--replay",
                "test.json",
                "--format",
                "kml",
                "--enu-origin",
                "34.0,-118.0",
                "--output",
                "out.kml",
            ]
        )
        self.assertEqual(args.format, "kml")

    def test_export_time_range_arg(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "export",
                "--replay",
                "test.json",
                "--format",
                "geojson",
                "--enu-origin",
                "34.0,-118.0",
                "--output",
                "out.json",
                "--time-range",
                "10.0,60.0",
            ]
        )
        self.assertEqual(args.time_range, "10.0,60.0")

    def test_sim_accepts_config_file_and_weather_preset(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "sim",
                "--config-file",
                "/tmp/sim.json",
                "--weather-preset",
                "fog",
                "--target-motion",
                "loiter",
            ]
        )
        self.assertEqual(args.config_file, "/tmp/sim.json")
        self.assertEqual(args.weather_preset, "fog")
        self.assertEqual(args.target_motion, "loiter")

    def test_verbose_flag(self):
        parser = build_parser()
        args = parser.parse_args(["info", "/tmp/f.json", "-v"])
        self.assertTrue(args.verbose)
        self.assertFalse(args.quiet)

    def test_quiet_flag(self):
        parser = build_parser()
        args = parser.parse_args(["info", "/tmp/f.json", "-q"])
        self.assertFalse(args.verbose)
        self.assertTrue(args.quiet)


class TestValidateReplay(unittest.TestCase):
    def test_valid_replay(self):
        """A valid replay document should pass validation via the CLI function."""
        import argparse

        from argusnet.cli.main import _run_validate_replay

        doc = _minimal_valid_replay_document()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            f.flush()
            path = f.name

        try:
            args = argparse.Namespace(path=path, verbose=False, quiet=False)
            # Should not raise
            _run_validate_replay(args)
        finally:
            os.unlink(path)

    def test_invalid_replay_missing_frames(self):
        import argparse

        from argusnet.cli.main import _run_validate_replay

        doc = {"meta": {"dt_s": 0.25}}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            f.flush()
            path = f.name

        try:
            args = argparse.Namespace(path=path, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_replay(args)
        finally:
            os.unlink(path)

    def test_invalid_replay_missing_required_frame_fields(self):
        import argparse

        from argusnet.cli.main import _run_validate_replay

        doc = _minimal_valid_replay_document()
        doc["frames"] = [{"timestamp_s": 0.0}]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            f.flush()
            path = f.name

        try:
            args = argparse.Namespace(path=path, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_replay(args)
        finally:
            os.unlink(path)

    def test_missing_file(self):
        import argparse

        from argusnet.cli.main import _run_validate_replay

        args = argparse.Namespace(
            path="/tmp/nonexistent_replay_12345.json", verbose=False, quiet=False
        )
        with self.assertRaises(SystemExit):
            _run_validate_replay(args)


class TestValidateScene(unittest.TestCase):
    def test_valid_scene(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_scene_bundle(Path(tmpdir))
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            _run_validate_scene(args)  # should not raise

    def test_valid_scene_legacy_manifest_filename(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_scene_bundle(Path(tmpdir), manifest_name="manifest.json")
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            _run_validate_scene(args)  # should not raise

    def test_missing_manifest(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_scene(args)

    def test_not_a_directory(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        args = argparse.Namespace(path="/tmp/nonexistent_dir_12345", verbose=False, quiet=False)
        with self.assertRaises(SystemExit):
            _run_validate_scene(args)

    def test_missing_layer_asset(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_scene_bundle(Path(tmpdir), include_asset=False)
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_scene(args)

    def test_missing_metadata_files(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            _write_scene_bundle(Path(tmpdir), include_metadata=False)
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_scene(args)

    def test_invalid_manifest_shape(self):
        import argparse

        from argusnet.cli.main import _run_validate_scene

        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "scene_manifest.json").write_text(
                json.dumps({"format_version": "smartscene-v1", "scene_id": "broken-scene"}),
                encoding="utf-8",
            )
            args = argparse.Namespace(path=tmpdir, verbose=False, quiet=False)
            with self.assertRaises(SystemExit):
                _run_validate_scene(args)


class TestInfoCommand(unittest.TestCase):
    def test_info_output(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from argusnet.cli.main import _run_info

        doc = {
            "meta": {
                "scenario_name": "test-scenario",
                "dt_s": 0.25,
                "frame_count": 10,
                "seed": 42,
                "node_ids": ["s1", "s2"],
                "track_ids": ["t1"],
                "generated_at_utc": "2026-01-01T00:00:00+00:00",
            },
            "summary": {
                "mean_error_m": 5.2,
                "peak_error_m": 12.1,
                "mean_observations_per_frame": 3.5,
                "observation_rejection_rate": 0.15,
            },
            "frames": [{"timestamp_s": i * 0.25} for i in range(10)],
        }
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(doc, f)
            f.flush()
            path = f.name

        try:
            args = argparse.Namespace(path=path, verbose=False, quiet=False)
            buf = io.StringIO()
            with redirect_stdout(buf):
                _run_info(args)
            output = buf.getvalue()
            self.assertIn("test-scenario", output)
            self.assertIn("10", output)  # frame count
            self.assertIn("s1", output)
            self.assertIn("t1", output)
            self.assertIn("5.20", output)  # mean error
        finally:
            os.unlink(path)


class TestDumpConfig(unittest.TestCase):
    def test_dump_config_json_stdout(self):
        import argparse
        import io
        from contextlib import redirect_stdout

        from argusnet.cli.main import _run_dump_config

        args = argparse.Namespace(format="json", output=None, verbose=False, quiet=False)
        buf = io.StringIO()
        with redirect_stdout(buf):
            _run_dump_config(args)
        output = buf.getvalue()
        parsed = json.loads(output)
        self.assertIn("sensor", parsed)
        self.assertIn("dynamics", parsed)

    def test_dump_config_to_file(self):
        import argparse

        from argusnet.cli.main import _run_dump_config

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = f.name

        try:
            args = argparse.Namespace(format="json", output=path, verbose=False, quiet=False)
            _run_dump_config(args)
            with open(path) as f:
                parsed = json.loads(f.read())
            self.assertIn("sensor", parsed)
        finally:
            os.unlink(path)


if __name__ == "__main__":
    unittest.main()
