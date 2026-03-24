"""Tests for health monitoring dataclasses and integration."""

from __future__ import annotations

import unittest

from argusnet.core.types import HealthReport, NodeHealthMetrics


class TestNodeHealthMetrics(unittest.TestCase):
    def test_construction(self):
        m = NodeHealthMetrics(
            node_id="sensor-01",
            last_seen_s=10.0,
            observation_rate_hz=4.0,
            mean_latency_s=0.01,
            accepted_count=40,
            rejected_count=2,
            health_score=0.95,
        )
        self.assertEqual(m.node_id, "sensor-01")
        self.assertAlmostEqual(m.observation_rate_hz, 4.0)
        self.assertEqual(m.accepted_count, 40)

    def test_frozen(self):
        m = NodeHealthMetrics("a", 0.0, 0.0, 0.0, 0, 0, 0.0)
        with self.assertRaises(AttributeError):
            m.node_id = "b"


class TestHealthReport(unittest.TestCase):
    def test_construction(self):
        report = HealthReport(
            status="SERVING",
            started_at_utc="2026-03-13T00:00:00Z",
            processed_frame_count=100,
            node_health=[
                NodeHealthMetrics("a", 10.0, 4.0, 0.01, 40, 2, 0.95),
                NodeHealthMetrics("b", 9.5, 3.8, 0.02, 38, 5, 0.88),
            ],
            mean_frame_rate_hz=4.0,
            mean_ingest_latency_s=0.005,
            active_node_count=2,
            stale_node_count=0,
        )
        self.assertEqual(report.status, "SERVING")
        self.assertEqual(len(report.node_health), 2)
        self.assertEqual(report.active_node_count, 2)

    def test_defaults(self):
        report = HealthReport(
            status="SERVING",
            started_at_utc="2026-03-13T00:00:00Z",
            processed_frame_count=0,
        )
        self.assertEqual(len(report.node_health), 0)
        self.assertAlmostEqual(report.mean_frame_rate_hz, 0.0)
        self.assertEqual(report.stale_node_count, 0)


if __name__ == "__main__":
    unittest.main()
