"""Tests for filesystem size guards on untrusted inputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from argusnet.core.io_limits import FileSizeLimitError, check_file_size, read_text_capped
from argusnet.evaluation.replay import load_replay_document


class TestReadTextCapped(unittest.TestCase):
    def test_reads_file_under_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "small.json"
            path.write_text('{"a": 1}', encoding="utf-8")
            self.assertEqual(read_text_capped(path, max_bytes=1024), '{"a": 1}')

    def test_rejects_file_over_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "big.json"
            path.write_text("x" * 1000, encoding="utf-8")
            with self.assertRaises(FileSizeLimitError):
                read_text_capped(path, max_bytes=100)

    def test_check_file_size_returns_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.txt"
            path.write_text("12345", encoding="utf-8")
            self.assertEqual(check_file_size(path, max_bytes=1024), 5)


class TestLoadReplayDocumentSizeCap(unittest.TestCase):
    def _valid_replay_document(self) -> dict:
        frame = {
            "timestamp_s": 0.0,
            "tracks": [],
            "truths": [],
            "observations": [],
            "rejected_observations": [],
            "nodes": [],
            "metrics": {},
        }
        return {
            "meta": {"dt_s": 0.1, "frame_count": 1},
            "frames": [frame],
        }

    def test_oversized_replay_document_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(self._valid_replay_document()), encoding="utf-8")

            def tiny_cap(p, **kwargs):
                return read_text_capped(p, max_bytes=10)

            with (
                patch("argusnet.evaluation.replay.read_text_capped", tiny_cap),
                self.assertRaises(FileSizeLimitError),
            ):
                load_replay_document(str(path))

    def test_normal_replay_document_loads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "replay.json"
            path.write_text(json.dumps(self._valid_replay_document()), encoding="utf-8")
            document = load_replay_document(str(path))
            self.assertEqual(len(document["frames"]), 1)


class TestDemPixelCap(unittest.TestCase):
    """`project_dem_to_runtime` must reject an oversized raster before decoding it.

    Uses a fake tifffile module (rather than writing a real multi-gigapixel
    GeoTIFF fixture) so the pixel-count guard is exercised without ever
    materializing a huge array; `page.asarray()` must not be reached.
    """

    def test_oversized_dem_rejected_before_decode(self):
        import argusnet.world._scene_gis as scene_gis

        class FakePage:
            shape = (20_000, 20_000)  # 400M pixels, over the 256M cap
            tags: dict = {}

            def asarray(self):
                raise AssertionError("asarray() must not be called for an oversized DEM")

        class FakeHandle:
            pages = [FakePage()]

            def __enter__(self):
                return self

            def __exit__(self, *exc_info):
                return False

        class FakeTifffile:
            @staticmethod
            def TiffFile(path):
                return FakeHandle()

        with tempfile.TemporaryDirectory() as tmp:
            dem_path = Path(tmp) / "huge.tif"
            dem_path.write_bytes(b"")
            with (
                patch.object(scene_gis, "lazy_import_tifffile", lambda: FakeTifffile),
                self.assertRaises(ValueError),
            ):
                scene_gis.project_dem_to_runtime(dem_path, source_crs="EPSG:32611")


if __name__ == "__main__":
    unittest.main()
