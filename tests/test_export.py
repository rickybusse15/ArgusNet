"""Tests for export module (GeoJSON, CZML, Foxglove stub)."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import types
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from argusnet.core.frames import ENUOrigin
from argusnet.evaluation.export import (
    export_czml,
    export_foxglove,
    export_geojson,
    export_geopackage,
    export_kmz,
    export_shapefile,
    suggested_output_path,
)


def _make_replay_doc(n_frames=4, n_tracks=2, include_obs=True, include_nodes=True):
    """Build a minimal replay document for testing."""
    frames = []
    for i in range(n_frames):
        t = float(i) * 0.25
        tracks = []
        for j in range(n_tracks):
            tracks.append({
                "track_id": f"T{j}",
                "timestamp_s": t,
                "position": [100.0 * j + i, 200.0 * j + i, 50.0],
                "velocity": [1.0, 0.0, 0.0],
                "covariance_row_major": [1.0] * 9,
                "measurement_std_m": 5.0,
                "update_count": i + 1,
                "stale_steps": 0,
            })
        observations = []
        if include_obs:
            observations.append({
                "node_id": "N0",
                "target_id": "T0",
                "origin": [0.0, 0.0, 0.0],
                "direction": [1.0, 0.0, 0.0],
                "bearing_std_rad": 0.01,
                "timestamp_s": t,
                "confidence": 0.9,
            })
        nodes = []
        if include_nodes:
            nodes.append({
                "node_id": "N0",
                "position": [0.0, 0.0, 0.0],
                "velocity": [0.0, 0.0, 0.0],
                "is_mobile": False,
                "timestamp_s": t,
                "health": 1.0,
            })
        frames.append({
            "timestamp_s": t,
            "tracks": tracks,
            "observations": observations,
            "nodes": nodes,
            "rejected_observations": [],
            "truths": [],
            "metrics": {},
        })
    return {
        "meta": {
            "scenario_name": "test",
            "frame_count": n_frames,
            "dt_s": 0.25,
            "generated_at_utc": "2025-01-01T00:00:00+00:00",
        },
        "frames": frames,
    }


ORIGIN = ENUOrigin(latitude_deg=47.3769, longitude_deg=8.5417, altitude_m=408.0)


class TestExportGeoJSON(unittest.TestCase):
    def test_basic_geojson_structure(self):
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertIsInstance(data["features"], list)

    def test_track_linestrings(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
        self.assertEqual(len(lines), 2)
        for line in lines:
            self.assertEqual(len(line["geometry"]["coordinates"]), 4)
            self.assertIn("track_id", line["properties"])

    def test_single_point_track_excluded(self):
        doc = _make_replay_doc(n_frames=1, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
        self.assertEqual(len(lines), 0)

    def test_observations_included(self):
        doc = _make_replay_doc(n_frames=3, include_obs=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path, include_observations=True)
            with open(path) as f:
                data = json.load(f)
        obs = [f for f in data["features"] if f["properties"].get("type") == "observation"]
        self.assertEqual(len(obs), 3)

    def test_observations_excluded_by_default(self):
        doc = _make_replay_doc(n_frames=3, include_obs=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        obs = [f for f in data["features"] if f["properties"].get("type") == "observation"]
        self.assertEqual(len(obs), 0)

    def test_nodes_included(self):
        doc = _make_replay_doc(n_frames=3, include_nodes=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path, include_nodes=True)
            with open(path) as f:
                data = json.load(f)
        nodes = [f for f in data["features"] if f["properties"].get("type") == "node"]
        self.assertEqual(len(nodes), 1)

    def test_coordinates_are_wgs84(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        lines = [f for f in data["features"] if f["geometry"]["type"] == "LineString"]
        self.assertTrue(len(lines) >= 1)
        coord = lines[0]["geometry"]["coordinates"][0]
        lon, lat, alt = coord
        self.assertAlmostEqual(lat, 47.3769, delta=0.1)
        self.assertAlmostEqual(lon, 8.5417, delta=0.1)

    def test_creates_parent_directories(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "sub", "dir", "out.geojson")
            export_geojson(doc, ORIGIN, path)
            self.assertTrue(os.path.exists(path))

    def test_time_range_filters_track_positions(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path, start_time_s=0.25, end_time_s=0.5)
            with open(path) as f:
                data = json.load(f)
        line = [feature for feature in data["features"] if feature["geometry"]["type"] == "LineString"][0]
        self.assertEqual(len(line["geometry"]["coordinates"]), 2)
        self.assertAlmostEqual(line["properties"]["start_time_s"], 0.25)
        self.assertAlmostEqual(line["properties"]["end_time_s"], 0.5)

    def test_time_range_uses_latest_node_within_window(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=0, include_nodes=True)
        for index, frame in enumerate(doc["frames"]):
            frame["nodes"][0]["position"] = [float(index), 0.0, 0.0]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(
                doc,
                ORIGIN,
                path,
                include_nodes=True,
                start_time_s=0.25,
                end_time_s=0.5,
            )
            with open(path) as f:
                data = json.load(f)
        node_feature = [feature for feature in data["features"] if feature["properties"].get("type") == "node"][0]
        lon, lat, alt = node_feature["geometry"]["coordinates"]
        expected_lat, expected_lon, expected_alt = ORIGIN.latitude_deg, ORIGIN.longitude_deg, ORIGIN.altitude_m
        self.assertNotEqual(lon, expected_lon)
        self.assertAlmostEqual(lat, expected_lat, delta=0.1)
        self.assertAlmostEqual(alt, expected_alt, delta=5.0)


class TestExportCZML(unittest.TestCase):
    def test_basic_czml_structure(self):
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        self.assertIsInstance(data, list)
        self.assertGreaterEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "document")

    def test_document_header(self):
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        header = data[0]
        self.assertEqual(header["id"], "document")
        self.assertIn("clock", header)
        self.assertIn("interval", header["clock"])

    def test_track_packets(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        track_packets = [p for p in data if p["id"] != "document"]
        self.assertEqual(len(track_packets), 2)
        for pkt in track_packets:
            self.assertIn("position", pkt)
            self.assertIn("cartographicDegrees", pkt["position"])
            self.assertIn("point", pkt)
            self.assertIn("path", pkt)

    def test_cartographic_degrees_length(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        track_pkt = [p for p in data if p["id"] != "document"][0]
        # 4 frames * 4 values (time, lon, lat, alt) = 16
        self.assertEqual(len(track_pkt["position"]["cartographicDegrees"]), 16)

    def test_custom_start_time(self):
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path, start_time_utc="2025-06-01T12:00:00+00:00")
            with open(path) as f:
                data = json.load(f)
        header = data[0]
        self.assertIn("2025-06-01", header["clock"]["interval"])

    def test_time_range_filters_track_samples(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path, start_time_s=0.25, end_time_s=0.5)
            with open(path) as f:
                data = json.load(f)
        track_pkt = [packet for packet in data if packet["id"] != "document"][0]
        self.assertEqual(len(track_pkt["position"]["cartographicDegrees"]), 8)

    def test_time_range_updates_document_interval(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path, start_time_s=0.25, end_time_s=0.5)
            with open(path) as f:
                data = json.load(f)
        interval = data[0]["clock"]["interval"]
        start_iso, end_iso = interval.split("/")
        self.assertEqual(start_iso, "2025-01-01T00:00:00.250000+00:00")
        self.assertEqual(end_iso, "2025-01-01T00:00:00.500000+00:00")


class _FakeMcapWriter:
    instances: list["_FakeMcapWriter"] = []

    def __init__(self, handle):
        self.handle = handle
        self.messages = []
        _FakeMcapWriter.instances.append(self)

    def start(self):
        return None

    def register_schema(self, **kwargs):
        self.schema = kwargs
        return 1

    def register_channel(self, **kwargs):
        self.channel = kwargs
        return 2

    def add_message(self, **kwargs):
        self.messages.append(kwargs)

    def finish(self):
        return None


class TestExportFoxglove(unittest.TestCase):
    def test_time_range_filters_messages_and_uses_absolute_anchor(self):
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        _FakeMcapWriter.instances.clear()
        fake_mcap = types.ModuleType("mcap")
        fake_mcap_writer = types.ModuleType("mcap.writer")
        fake_mcap_writer.Writer = _FakeMcapWriter

        with patch.dict(sys.modules, {"mcap": fake_mcap, "mcap.writer": fake_mcap_writer}):
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "out.mcap")
                export_foxglove(doc, ORIGIN, path, start_time_s=0.25, end_time_s=0.5)

        writer = _FakeMcapWriter.instances[-1]
        self.assertEqual(len(writer.messages), 2)
        anchor = datetime.fromisoformat(doc["meta"]["generated_at_utc"].replace("Z", "+00:00"))
        expected_times = [
            int((anchor + timedelta(seconds=value)).timestamp() * 1e9)
            for value in (0.25, 0.5)
        ]
        self.assertEqual([message["log_time"] for message in writer.messages], expected_times)
        self.assertEqual([message["publish_time"] for message in writer.messages], expected_times)


class TestAdditionalExportFormats(unittest.TestCase):
    def test_kmz_wraps_single_doc_kml(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kmz")
            export_kmz(doc, ORIGIN, path)
            import zipfile

            with zipfile.ZipFile(path) as archive:
                self.assertEqual(archive.namelist(), ["doc.kml"])
                self.assertIn(b"<kml", archive.read("doc.kml"))

    def test_geopackage_requires_fiona(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpkg")
            with self.assertRaises(RuntimeError) as context:
                export_geopackage(doc, ORIGIN, path)
        self.assertIn("fiona", str(context.exception).lower())

    def test_shapefile_requires_fiona(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            output_dir = os.path.join(tmp, "shape")
            with self.assertRaises(RuntimeError) as context:
                export_shapefile(doc, ORIGIN, output_dir)
        self.assertIn("fiona", str(context.exception).lower())

    def test_suggested_output_path_uses_expected_suffixes(self):
        self.assertEqual(
            suggested_output_path("demo.replay.json", "kmz", "exports").name,
            "demo.replay.kmz",
        )
        self.assertEqual(
            suggested_output_path("demo.replay.json", "shapefile", "exports").name,
            "demo.replay-shapefile",
        )


class TestExportEmptyDoc(unittest.TestCase):
    def test_empty_frames_geojson(self):
        doc = {"meta": {}, "frames": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.geojson")
            export_geojson(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(data["type"], "FeatureCollection")
        self.assertEqual(len(data["features"]), 0)

    def test_empty_frames_czml(self):
        doc = {"meta": {}, "frames": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.czml")
            export_czml(doc, ORIGIN, path)
            with open(path) as f:
                data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["id"], "document")


if __name__ == "__main__":
    unittest.main()
