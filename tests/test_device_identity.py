"""Tests for Ed25519 device identity and signed-envelope verification."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from argusnet.security.identity import (
    DeviceRegistry,
    EnvelopeRejectionReason,
    EnvelopeVerifier,
    IdentityError,
    generate_keypair,
    public_key_bytes,
    sign_payload,
    verify_signature,
)


def _sample_payload(**overrides):
    base = {"azimuth_deg": 12.0, "elevation_deg": 3.0, "target_id": "tgt-a"}
    base.update(overrides)
    return base


class TestSignAndVerify(unittest.TestCase):
    def test_valid_signature_verifies(self):
        private_key, public_key = generate_keypair()
        payload = _sample_payload()
        signature = sign_payload(
            private_key, device_id="drone-1", sequence=1, timestamp_s=100.0, payload=payload
        )
        self.assertTrue(
            verify_signature(
                public_key,
                signature,
                device_id="drone-1",
                sequence=1,
                timestamp_s=100.0,
                payload=payload,
            )
        )

    def test_tampered_payload_fails_verification(self):
        private_key, public_key = generate_keypair()
        payload = _sample_payload()
        signature = sign_payload(
            private_key, device_id="drone-1", sequence=1, timestamp_s=100.0, payload=payload
        )
        tampered = dict(payload, azimuth_deg=999.0)
        self.assertFalse(
            verify_signature(
                public_key,
                signature,
                device_id="drone-1",
                sequence=1,
                timestamp_s=100.0,
                payload=tampered,
            )
        )

    def test_wrong_key_fails_verification(self):
        private_key, _ = generate_keypair()
        _, other_public_key = generate_keypair()
        payload = _sample_payload()
        signature = sign_payload(
            private_key, device_id="drone-1", sequence=1, timestamp_s=100.0, payload=payload
        )
        self.assertFalse(
            verify_signature(
                other_public_key,
                signature,
                device_id="drone-1",
                sequence=1,
                timestamp_s=100.0,
                payload=payload,
            )
        )

    def test_sequence_or_device_id_is_covered_by_signature(self):
        # Signature must not verify if device_id or sequence is swapped after
        # signing (both are part of the signed envelope, not just payload).
        private_key, public_key = generate_keypair()
        payload = _sample_payload()
        signature = sign_payload(
            private_key, device_id="drone-1", sequence=1, timestamp_s=100.0, payload=payload
        )
        self.assertFalse(
            verify_signature(
                public_key,
                signature,
                device_id="drone-1",
                sequence=2,  # sequence changed post-signing
                timestamp_s=100.0,
                payload=payload,
            )
        )
        self.assertFalse(
            verify_signature(
                public_key,
                signature,
                device_id="drone-2",  # device_id changed post-signing
                sequence=1,
                timestamp_s=100.0,
                payload=payload,
            )
        )


class TestDeviceRegistry(unittest.TestCase):
    def test_from_mapping_round_trips(self):
        _, public_key = generate_keypair()
        registry = DeviceRegistry.from_mapping({"drone-1": public_key_bytes(public_key)})
        self.assertIn("drone-1", registry)
        self.assertIsNotNone(registry.public_key("drone-1"))
        self.assertIsNone(registry.public_key("unknown"))
        self.assertEqual(len(registry), 1)

    def test_from_directory_loads_pub_files(self):
        _, public_key = generate_keypair()
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "drone-1.pub").write_bytes(public_key_bytes(public_key))
            registry = DeviceRegistry.from_directory(tmp)
            self.assertIn("drone-1", registry)

    def test_from_directory_missing_dir_raises(self):
        with self.assertRaises(IdentityError):
            DeviceRegistry.from_directory("/nonexistent/path/for/argusnet/test")

    def test_from_directory_rejects_invalid_key_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "bad.pub").write_bytes(b"not-a-valid-ed25519-key")
            with self.assertRaises(IdentityError):
                DeviceRegistry.from_directory(tmp)


class TestEnvelopeVerifier(unittest.TestCase):
    def setUp(self):
        self.private_key, self.public_key = generate_keypair()
        self.registry = DeviceRegistry.from_mapping({"drone-1": public_key_bytes(self.public_key)})
        self.verifier = EnvelopeVerifier(
            self.registry, timestamp_window_s=5.0, rate_limit_per_s=10.0, rate_limit_window_s=1.0
        )

    def _sign_and_verify(self, *, sequence, timestamp_s, payload=None, now_s=None):
        payload = payload if payload is not None else _sample_payload()
        signature = sign_payload(
            self.private_key,
            device_id="drone-1",
            sequence=sequence,
            timestamp_s=timestamp_s,
            payload=payload,
        )
        return self.verifier.verify(
            device_id="drone-1",
            sequence=sequence,
            timestamp_s=timestamp_s,
            payload=payload,
            signature=signature,
            now_s=now_s if now_s is not None else timestamp_s,
        )

    def test_valid_envelope_accepted(self):
        result = self._sign_and_verify(sequence=1, timestamp_s=1000.0)
        self.assertTrue(result.accepted)
        self.assertIsNone(result.reason)

    def test_unknown_device_rejected(self):
        payload = _sample_payload()
        signature = sign_payload(
            self.private_key, device_id="ghost", sequence=1, timestamp_s=1000.0, payload=payload
        )
        result = self.verifier.verify(
            device_id="ghost",
            sequence=1,
            timestamp_s=1000.0,
            payload=payload,
            signature=signature,
            now_s=1000.0,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.UNKNOWN_DEVICE)

    def test_bad_signature_rejected(self):
        result = self.verifier.verify(
            device_id="drone-1",
            sequence=1,
            timestamp_s=1000.0,
            payload=_sample_payload(),
            signature=b"\x00" * 64,
            now_s=1000.0,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.BAD_SIGNATURE)

    def test_tampered_payload_rejected(self):
        payload = _sample_payload()
        signature = sign_payload(
            self.private_key, device_id="drone-1", sequence=1, timestamp_s=1000.0, payload=payload
        )
        result = self.verifier.verify(
            device_id="drone-1",
            sequence=1,
            timestamp_s=1000.0,
            payload=dict(payload, azimuth_deg=999.0),
            signature=signature,
            now_s=1000.0,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.BAD_SIGNATURE)


if __name__ == "__main__":
    unittest.main()
