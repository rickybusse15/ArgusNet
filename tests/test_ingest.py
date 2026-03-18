"""Tests for live sensor ingestion adapters."""

from __future__ import annotations

import math
import unittest

import numpy as np

from smart_tracker.coordinates import ENUOrigin, wgs84_to_enu
from smart_tracker.ingest import (
    MQTTIngestionAdapter,
    parse_mqtt_node_state,
    parse_mqtt_observation,
)


class TestParseMQTTObservation(unittest.TestCase):
    def _sample_payload(self, **overrides):
        base = {
            "node_id": "sensor-01",
            "target_id": "target-A",
            "lat_deg": 47.3769,
            "lon_deg": 8.5417,
            "alt_m": 408.0,
            "azimuth_deg": 90.0,
            "elevation_deg": 0.0,
            "bearing_std_rad": 0.005,
            "confidence": 0.85,
            "timestamp_unix_s": 1710000000.0,
        }
        base.update(overrides)
        return base

    def test_basic_parse(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        obs = parse_mqtt_observation(self._sample_payload(), origin)
        self.assertEqual(obs.node_id, "sensor-01")
        self.assertEqual(obs.target_id, "target-A")
        self.assertAlmostEqual(obs.bearing_std_rad, 0.005)
        self.assertAlmostEqual(obs.confidence, 0.85)
        self.assertAlmostEqual(obs.timestamp_s, 1710000000.0)

    def test_origin_position_is_enu(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        obs = parse_mqtt_observation(self._sample_payload(), origin)
        # Observer is at the ENU origin
        np.testing.assert_allclose(obs.origin, [0.0, 0.0, 0.0], atol=1e-3)

    def test_direction_east(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        obs = parse_mqtt_observation(
            self._sample_payload(azimuth_deg=90.0, elevation_deg=0.0), origin
        )
        # Azimuth 90 = East → direction should be [1, 0, 0]
        np.testing.assert_allclose(obs.direction, [1.0, 0.0, 0.0], atol=1e-10)

    def test_direction_north(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        obs = parse_mqtt_observation(
            self._sample_payload(azimuth_deg=0.0, elevation_deg=0.0), origin
        )
        # Azimuth 0 = North → direction should be [0, 1, 0]
        np.testing.assert_allclose(obs.direction, [0.0, 1.0, 0.0], atol=1e-10)

    def test_direction_with_elevation(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        obs = parse_mqtt_observation(
            self._sample_payload(azimuth_deg=0.0, elevation_deg=45.0), origin
        )
        expected_z = math.sin(math.radians(45.0))
        expected_y = math.cos(math.radians(45.0))
        np.testing.assert_allclose(
            obs.direction, [0.0, expected_y, expected_z], atol=1e-10
        )

    def test_default_target_id(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        payload = self._sample_payload()
        del payload["target_id"]
        obs = parse_mqtt_observation(payload, origin)
        self.assertEqual(obs.target_id, "unknown")

    def test_default_confidence(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        payload = self._sample_payload()
        del payload["confidence"]
        obs = parse_mqtt_observation(payload, origin)
        self.assertAlmostEqual(obs.confidence, 1.0)

    def test_missing_required_key_raises(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        payload = self._sample_payload()
        del payload["node_id"]
        with self.assertRaises(ValueError):
            parse_mqtt_observation(payload, origin)


class TestParseMQTTNodeState(unittest.TestCase):
    def test_basic_parse(self):
        origin = ENUOrigin(47.3769, 8.5417, 408.0)
        payload = {
            "node_id": "sensor-01",
            "lat_deg": 47.3769,
            "lon_deg": 8.5417,
            "alt_m": 408.0,
            "timestamp_unix_s": 1710000000.0,
        }
        node = parse_mqtt_node_state(payload, origin)
        self.assertEqual(node.node_id, "sensor-01")
        np.testing.assert_allclose(node.position, [0.0, 0.0, 0.0], atol=1e-3)
        np.testing.assert_allclose(node.velocity, [0.0, 0.0, 0.0], atol=1e-10)
        self.assertTrue(node.is_mobile)
        self.assertAlmostEqual(node.health, 1.0)

    def test_velocity_and_mobile_flag(self):
        origin = ENUOrigin(0.0, 0.0, 0.0)
        payload = {
            "node_id": "sensor-02",
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "alt_m": 0.0,
            "vx_ms": 1.0,
            "vy_ms": 2.0,
            "vz_ms": 3.0,
            "is_mobile": False,
            "timestamp_unix_s": 1710000000.0,
            "health": 0.75,
        }
        node = parse_mqtt_node_state(payload, origin)
        np.testing.assert_allclose(node.velocity, [1.0, 2.0, 3.0])
        self.assertFalse(node.is_mobile)
        self.assertAlmostEqual(node.health, 0.75)


class TestMQTTAdapterConstruction(unittest.TestCase):
    def test_default_topics(self):
        adapter = MQTTIngestionAdapter(broker="localhost")
        self.assertEqual(adapter.observation_topic, "smart_tracker/observations")
        self.assertEqual(adapter.node_topic, "smart_tracker/nodes")
        self.assertEqual(adapter.port, 1883)

    def test_flush_empty(self):
        adapter = MQTTIngestionAdapter(broker="localhost")
        nodes, observations = adapter.flush_pending()
        self.assertEqual(len(nodes), 0)
        self.assertEqual(len(observations), 0)


if __name__ == "__main__":
    unittest.main()
