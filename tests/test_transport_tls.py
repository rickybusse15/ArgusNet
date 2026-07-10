"""Tests for the TLS/mTLS transport guard: loopback detection and the
refuse-plaintext-off-loopback behavior of the gRPC client and MQTT adapter.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from argusnet.adapters.argusnet_grpc import TrackingService
from argusnet.security.transport import (
    TLSConfig,
    TransportSecurityError,
    is_loopback_endpoint,
    is_loopback_host,
)
from argusnet.sensing.ingestion.frame_stream import MQTTIngestionAdapter


class TestLoopbackDetection(unittest.TestCase):
    def test_loopback_hosts(self):
        for host in ("127.0.0.1", "localhost", "::1"):
            self.assertTrue(is_loopback_host(host), host)

    def test_non_loopback_hosts(self):
        for host in ("10.0.0.5", "example.com", "192.168.1.1", "2001:db8::1"):
            self.assertFalse(is_loopback_host(host), host)

    def test_loopback_endpoints(self):
        for endpoint in ("127.0.0.1:50051", "localhost:50051", "[::1]:50051"):
            self.assertTrue(is_loopback_endpoint(endpoint), endpoint)

    def test_non_loopback_endpoints(self):
        for endpoint in ("10.0.0.5:50051", "example.com:50051", "0.0.0.0:50051"):
            self.assertFalse(is_loopback_endpoint(endpoint), endpoint)


class TestGrpcClientRefusesPlaintextOffLoopback(unittest.TestCase):
    def test_non_loopback_without_tls_raises(self):
        with self.assertRaises(TransportSecurityError):
            TrackingService(
                endpoint="203.0.113.10:50051",
                spawn_local=False,
                tls_config=TLSConfig(),
            )

    def test_loopback_without_tls_does_not_raise_transport_error(self):
        # No daemon is running on this port, so this fails on the readiness
        # wait (RuntimeError), not on the transport-security check — proving
        # the TLS guard doesn't fire for loopback endpoints.
        with self.assertRaises(RuntimeError) as ctx:
            TrackingService(
                endpoint="127.0.0.1:1",
                spawn_local=False,
                startup_timeout_s=0.2,
            )
        self.assertNotIsInstance(ctx.exception, TransportSecurityError)


class TestMqttAdapterRefusesPlaintextOffLoopback(unittest.TestCase):
    def test_non_loopback_broker_without_tls_raises(self):
        adapter = MQTTIngestionAdapter(broker="mqtt.example.com", tls_config=TLSConfig())
        with self.assertRaises(TransportSecurityError):
            adapter.start(on_frame=lambda *_: None)

    def test_non_loopback_broker_with_tls_but_no_registry_raises(self):
        tls_config = TLSConfig(ca_path=Path(__file__))  # any path; only `.configured` is checked
        adapter = MQTTIngestionAdapter(broker="mqtt.example.com", tls_config=tls_config)
        with self.assertRaises(TransportSecurityError):
            adapter.start(on_frame=lambda *_: None)


if __name__ == "__main__":
    unittest.main()
