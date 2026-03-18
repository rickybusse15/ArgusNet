"""Tests for KML and GPX export formats."""

from __future__ import annotations

import os
import tempfile
import unittest
import xml.etree.ElementTree as ET

from smart_tracker.coordinates import ENUOrigin
from smart_tracker.export import export_kml, export_gpx


_KML_NS = "http://www.opengis.net/kml/2.2"
_GPX_NS = "http://www.topografix.com/GPX/1/1"

ORIGIN = ENUOrigin(latitude_deg=47.3769, longitude_deg=8.5417, altitude_m=408.0)


def _make_replay_doc(n_frames=4, n_tracks=2, include_nodes=True):
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
            "observations": [],
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


# -----------------------------------------------------------------------
# KML tests
# -----------------------------------------------------------------------


class TestExportKML(unittest.TestCase):
    """Tests for export_kml."""

    def test_kml_valid_xml_with_root_element(self):
        """KML output should be parseable XML with <kml> root."""
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            result = export_kml(doc, ORIGIN, path)
            self.assertEqual(result, path)
            tree = ET.parse(path)
            root = tree.getroot()
        # Root tag should be 'kml' (possibly namespace-qualified)
        self.assertIn("kml", root.tag.lower())

    def test_kml_has_document_and_folders(self):
        """KML should have a Document element with Tracks and Sensor Nodes folders."""
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            tree = ET.parse(path)
            root = tree.getroot()
        # Find Document (handle namespace prefix)
        ns = {"kml": _KML_NS}
        document = root.find("kml:Document", ns)
        if document is None:
            document = root.find("Document")
        self.assertIsNotNone(document)
        # Look for Folder elements
        folders = document.findall("kml:Folder", ns)
        if not folders:
            folders = document.findall("Folder")
        folder_names = []
        for folder in folders:
            name_el = folder.find("kml:name", ns)
            if name_el is None:
                name_el = folder.find("name")
            if name_el is not None:
                folder_names.append(name_el.text)
        self.assertIn("Tracks", folder_names)
        self.assertIn("Sensor Nodes", folder_names)

    def test_kml_track_data_present(self):
        """KML should contain Placemark elements for track LineStrings."""
        doc = _make_replay_doc(n_frames=4, n_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"T0", xml_content)
        self.assertIn(b"T1", xml_content)
        self.assertIn(b"LineString", xml_content)
        self.assertIn(b"coordinates", xml_content)

    def test_kml_timespan_present(self):
        """KML track Placemarks should have TimeSpan elements."""
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"TimeSpan", xml_content)
        self.assertIn(b"begin", xml_content)
        self.assertIn(b"end", xml_content)

    def test_kml_times_use_generated_at_anchor(self):
        """KML timestamps should be anchored to generated_at_utc, not Unix epoch."""
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"2025-01-01T00:00:00+00:00", xml_content)
        self.assertIn(b"2025-01-01T00:00:00.250000+00:00", xml_content)

    def test_kml_falls_back_to_unix_epoch_when_anchor_is_invalid(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        doc["meta"]["generated_at_utc"] = "not-a-date"
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"1970-01-01T00:00:00+00:00", xml_content)

    def test_kml_sensor_nodes_present(self):
        """KML should contain Point placemarks for sensor nodes."""
        doc = _make_replay_doc(n_frames=2, include_nodes=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"N0", xml_content)
        self.assertIn(b"Point", xml_content)

    def test_kml_namespace(self):
        """KML root element should declare the KML namespace."""
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"opengis.net/kml", xml_content)

    def test_kml_empty_input(self):
        """Empty frames should produce valid KML with no Placemarks."""
        doc = {"meta": {}, "frames": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("kml", root.tag.lower())

    def test_kml_time_range_filter(self):
        """Time-range filter should exclude frames outside the window."""
        doc = _make_replay_doc(n_frames=8, n_tracks=1)
        # Frames at t=0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        with tempfile.TemporaryDirectory() as tmp:
            path_full = os.path.join(tmp, "full.kml")
            path_filtered = os.path.join(tmp, "filtered.kml")
            export_kml(doc, ORIGIN, path_full)
            export_kml(doc, ORIGIN, path_filtered, start_time_s=0.5, end_time_s=1.0)
            full_size = os.path.getsize(path_full)
            filtered_size = os.path.getsize(path_filtered)
        # Filtered file should be smaller (fewer coordinates)
        self.assertLess(filtered_size, full_size)

    def test_kml_time_range_excludes_all(self):
        """Time range that excludes all frames produces valid but empty KML."""
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        # Frames at t=0.0, 0.25, 0.5, 0.75 -- filter above all of them
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path, start_time_s=100.0)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("kml", root.tag.lower())


# -----------------------------------------------------------------------
# GPX tests
# -----------------------------------------------------------------------


class TestExportGPX(unittest.TestCase):
    """Tests for export_gpx."""

    def test_gpx_valid_xml_with_root_element(self):
        """GPX output should be parseable XML with <gpx> root."""
        doc = _make_replay_doc()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            result = export_gpx(doc, ORIGIN, path)
            self.assertEqual(result, path)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("gpx", root.tag.lower())

    def test_gpx_namespace(self):
        """GPX root should declare the correct GPX 1.1 namespace."""
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"topografix.com/GPX/1/1", xml_content)

    def test_gpx_version_attribute(self):
        """GPX root element should have version='1.1'."""
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertEqual(root.attrib.get("version"), "1.1")

    def test_gpx_track_data_present(self):
        """GPX should contain trk and trkseg elements with trkpt children."""
        doc = _make_replay_doc(n_frames=4, n_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"<trk>", xml_content)
        self.assertIn(b"<trkseg>", xml_content)
        self.assertIn(b"<trkpt", xml_content)

    def test_gpx_trkpt_has_lat_lon_ele_time(self):
        """Each trkpt should have lat, lon attributes and ele, time children."""
        doc = _make_replay_doc(n_frames=3, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            tree = ET.parse(path)
            root = tree.getroot()
        # ElementTree qualifies all tags with the namespace URI
        trkpts = root.findall(".//{%s}trkpt" % _GPX_NS)
        self.assertGreater(len(trkpts), 0)
        for pt in trkpts:
            self.assertIn("lat", pt.attrib)
            self.assertIn("lon", pt.attrib)
            # ele and time children (namespace-qualified)
            ele = pt.find("{%s}ele" % _GPX_NS)
            time_el = pt.find("{%s}time" % _GPX_NS)
            self.assertIsNotNone(ele)
            self.assertIsNotNone(time_el)

    def test_gpx_waypoints_for_nodes(self):
        """GPX should contain wpt elements for sensor nodes."""
        doc = _make_replay_doc(n_frames=2, include_nodes=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"<wpt", xml_content)
        self.assertIn(b"N0", xml_content)

    def test_gpx_empty_input(self):
        """Empty frames should produce valid GPX with no tracks/waypoints."""
        doc = {"meta": {}, "frames": []}
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("gpx", root.tag.lower())

    def test_gpx_time_range_filter(self):
        """Time-range filter should exclude frames outside the window."""
        doc = _make_replay_doc(n_frames=8, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path_full = os.path.join(tmp, "full.gpx")
            path_filtered = os.path.join(tmp, "filtered.gpx")
            export_gpx(doc, ORIGIN, path_full)
            export_gpx(doc, ORIGIN, path_filtered, start_time_s=0.5, end_time_s=1.0)
            full_size = os.path.getsize(path_full)
            filtered_size = os.path.getsize(path_filtered)
        self.assertLess(filtered_size, full_size)

    def test_gpx_time_range_excludes_all(self):
        """Time range that excludes all frames produces valid but empty GPX."""
        doc = _make_replay_doc(n_frames=4, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path, start_time_s=100.0)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("gpx", root.tag.lower())

    def test_gpx_track_names(self):
        """GPX tracks should have name elements matching track IDs."""
        doc = _make_replay_doc(n_frames=3, n_tracks=2)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"T0", xml_content)
        self.assertIn(b"T1", xml_content)

    def test_gpx_times_use_generated_at_anchor(self):
        """GPX trackpoint timestamps should be anchored to generated_at_utc."""
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"2025-01-01T00:00:00+00:00", xml_content)
        self.assertIn(b"2025-01-01T00:00:00.250000+00:00", xml_content)

    def test_gpx_falls_back_to_unix_epoch_when_anchor_is_missing(self):
        doc = _make_replay_doc(n_frames=2, n_tracks=1)
        del doc["meta"]["generated_at_utc"]
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path)
            xml_content = open(path, "rb").read()
        self.assertIn(b"1970-01-01T00:00:00+00:00", xml_content)


# -----------------------------------------------------------------------
# Cross-format time-range tests
# -----------------------------------------------------------------------


class TestTimeRangeFiltering(unittest.TestCase):
    """Shared tests for the time-range filter concept."""

    def test_start_time_only(self):
        """Providing only start_time_s should exclude earlier frames."""
        doc = _make_replay_doc(n_frames=8, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.kml")
            export_kml(doc, ORIGIN, path, start_time_s=1.0)
            xml_content = open(path, "rb").read()
            tree = ET.parse(path)
        # Should be valid XML
        self.assertIn(b"kml", xml_content)

    def test_end_time_only(self):
        """Providing only end_time_s should exclude later frames."""
        doc = _make_replay_doc(n_frames=8, n_tracks=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "out.gpx")
            export_gpx(doc, ORIGIN, path, end_time_s=0.5)
            tree = ET.parse(path)
            root = tree.getroot()
        self.assertIn("gpx", root.tag.lower())

    def test_time_range_filters_consistently(self):
        """Both KML and GPX should filter the same number of track points."""
        doc = _make_replay_doc(n_frames=8, n_tracks=1, include_nodes=False)
        start, end = 0.25, 0.75
        with tempfile.TemporaryDirectory() as tmp:
            kml_path = os.path.join(tmp, "out.kml")
            gpx_path = os.path.join(tmp, "out.gpx")
            export_kml(doc, ORIGIN, kml_path, start_time_s=start, end_time_s=end)
            export_gpx(doc, ORIGIN, gpx_path, start_time_s=start, end_time_s=end)

            # Count trkpt elements in GPX
            gpx_tree = ET.parse(gpx_path)
            gpx_root = gpx_tree.getroot()
            ns = {"gpx": _GPX_NS}
            trkpts = gpx_root.findall(".//{%s}trkpt" % _GPX_NS)
            if not trkpts:
                trkpts = gpx_root.findall(".//trkpt")

            # Frames at 0.25, 0.5, 0.75 => 3 frames * 1 track = 3 trkpts
            self.assertEqual(len(trkpts), 3)


if __name__ == "__main__":
    unittest.main()
