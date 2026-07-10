"""Tests for MQTT payload validation: size caps and numeric range/finiteness checks."""

from __future__ import annotations

import base64
import json
import math
import time
import unittest

from argusnet.core.errors import ValidationError
from argusnet.core.frames import ENUOrigin
from argusnet.security.identity import (
    DeviceRegistry,
    EnvelopeVerifier,
    generate_keypair,
    public_key_bytes,
    sign_payload,
)
from argusnet.sensing.ingestion.frame_stream import (
    MQTTIngestionAdapter,
    parse_mqtt_node_state,
    parse_mqtt_observation,
)

_ORIGIN = ENUOrigin(0.0, 0.0, 0.0)


def _obs_payload(**overrides):
    base = {
        "node_id": "sensor-01",
        "target_id": "target-A",
        "lat_deg": 0.0,
        "lon_deg": 0.0,
        "alt_m": 0.0,
        "azimuth_deg": 10.0,
        "elevation_deg": 5.0,
        "bearing_std_rad": 0.01,
        "confidence": 0.9,
        "timestamp_unix_s": 1000.0,
    }
    base.update(overrides)
    return base


class TestObservationNumericValidation(unittest.TestCase):
    def test_valid_payload_parses(self):
        parse_mqtt_observation(_obs_payload(), _ORIGIN)  # should not raise

    def test_nan_azimuth_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(azimuth_deg=math.nan), _ORIGIN)

    def test_inf_elevation_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(elevation_deg=math.inf), _ORIGIN)

    def test_nonpositive_bearing_std_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(bearing_std_rad=0.0), _ORIGIN)
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(bearing_std_rad=-0.01), _ORIGIN)

    def test_out_of_range_confidence_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(confidence=1.5), _ORIGIN)
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(confidence=-0.1), _ORIGIN)

    def test_nan_observer_position_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_observation(_obs_payload(lat_deg=math.nan), _ORIGIN)


class TestNodeStateNumericValidation(unittest.TestCase):
    def _node_payload(self, **overrides):
        base = {
            "node_id": "sensor-01",
            "lat_deg": 0.0,
            "lon_deg": 0.0,
            "alt_m": 0.0,
            "timestamp_unix_s": 1000.0,
        }
        base.update(overrides)
        return base

    def test_valid_payload_parses(self):
        parse_mqtt_node_state(self._node_payload(), _ORIGIN)  # should not raise

    def test_inf_velocity_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_node_state(self._node_payload(vx_ms=math.inf), _ORIGIN)

    def test_nan_health_rejected(self):
        with self.assertRaises(ValidationError):
            parse_mqtt_node_state(self._node_payload(health=math.nan), _ORIGIN)


class TestMqttPayloadSizeCap(unittest.TestCase):
    def test_oversized_payload_is_dropped_before_parsing(self):
        adapter = MQTTIngestionAdapter(broker="127.0.0.1", payload_max_bytes=64)

        class Msg:
            topic = adapter.observation_topic
            payload = b"x" * 128

        adapter._on_message(None, None, Msg())
        nodes, observations = adapter.flush_pending()
        self.assertEqual(len(nodes), 0)
        self.assertEqual(len(observations), 0)

    def test_non_object_payload_is_dropped(self):
        adapter = MQTTIngestionAdapter(broker="127.0.0.1")

        class Msg:
            topic = adapter.observation_topic
            payload = json.dumps([1, 2, 3]).encode()

        adapter._on_message(None, None, Msg())
        nodes, observations = adapter.flush_pending()
        self.assertEqual(len(observations), 0)


class TestMqttSignedEnvelopeIntegration(unittest.TestCase):
    def test_accepts_valid_signed_observation_end_to_end(self):
        private_key, public_key = generate_keypair()
        registry = DeviceRegistry.from_mapping({"drone-1": public_key_bytes(public_key)})
        adapter = MQTTIngestionAdapter(broker="127.0.0.1", device_registry=registry)
        adapter._verifier = EnvelopeVerifier(registry)

        body = _obs_payload(node_id="drone-1", timestamp_unix_s=time.time())
        signature = sign_payload(
            private_key,
            device_id="drone-1",
            sequence=1,
            timestamp_s=body["timestamp_unix_s"],
            payload=body,
        )
        envelope = dict(
            body, device_id="drone-1", sequence=1, signature=base64.b64encode(signature).decode()
        )

        class Msg:
            topic = adapter.observation_topic
            payload = json.dumps(envelope).encode()

        adapter._on_message(None, None, Msg())
        _, observations = adapter.flush_pending()
        self.assertEqual(len(observations), 1)

    def test_drops_unsigned_observation_when_registry_configured(self):
        _, public_key = generate_keypair()
        registry = DeviceRegistry.from_mapping({"drone-1": public_key_bytes(public_key)})
        adapter = MQTTIngestionAdapter(broker="127.0.0.1", device_registry=registry)
        adapter._verifier = EnvelopeVerifier(registry)

        class Msg:
            topic = adapter.observation_topic
            payload = json.dumps(_obs_payload(node_id="drone-1")).encode()

        adapter._on_message(None, None, Msg())
        _, observations = adapter.flush_pending()
        self.assertEqual(len(observations), 0)


if __name__ == "__main__":
    unittest.main()
