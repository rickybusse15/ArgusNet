"""Tests for anti-replay / rate-limit gating in EnvelopeVerifier."""

from __future__ import annotations

import unittest

from argusnet.security.identity import (
    DeviceRegistry,
    EnvelopeRejectionReason,
    EnvelopeVerifier,
    generate_keypair,
    public_key_bytes,
    sign_payload,
)


class TestAntiReplay(unittest.TestCase):
    def setUp(self):
        self.private_key, public_key = generate_keypair()
        self.registry = DeviceRegistry.from_mapping({"drone-1": public_key_bytes(public_key)})

    def _verifier(self, **kwargs):
        defaults = dict(timestamp_window_s=5.0, rate_limit_per_s=1000.0, rate_limit_window_s=1.0)
        defaults.update(kwargs)
        return EnvelopeVerifier(self.registry, **defaults)

    def _send(self, verifier, *, sequence, timestamp_s, now_s=None, payload=None):
        payload = payload if payload is not None else {"x": 1}
        signature = sign_payload(
            self.private_key,
            device_id="drone-1",
            sequence=sequence,
            timestamp_s=timestamp_s,
            payload=payload,
        )
        return verifier.verify(
            device_id="drone-1",
            sequence=sequence,
            timestamp_s=timestamp_s,
            payload=payload,
            signature=signature,
            now_s=now_s if now_s is not None else timestamp_s,
        )

    def test_duplicate_sequence_rejected(self):
        verifier = self._verifier()
        first = self._send(verifier, sequence=5, timestamp_s=1000.0)
        second = self._send(verifier, sequence=5, timestamp_s=1000.0)
        self.assertTrue(first.accepted)
        self.assertFalse(second.accepted)
        self.assertEqual(second.reason, EnvelopeRejectionReason.SEQUENCE_REPLAY)

    def test_sequence_regression_rejected(self):
        verifier = self._verifier()
        self.assertTrue(self._send(verifier, sequence=10, timestamp_s=1000.0).accepted)
        result = self._send(verifier, sequence=9, timestamp_s=1000.1)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.SEQUENCE_REPLAY)

    def test_increasing_sequence_accepted(self):
        verifier = self._verifier()
        for sequence in range(1, 6):
            result = self._send(verifier, sequence=sequence, timestamp_s=1000.0 + sequence)
            self.assertTrue(result.accepted, f"sequence {sequence} should be accepted")

    def test_out_of_window_future_timestamp_rejected(self):
        verifier = self._verifier(timestamp_window_s=5.0)
        result = self._send(verifier, sequence=1, timestamp_s=1100.0, now_s=1000.0)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.TIMESTAMP_OUT_OF_WINDOW)

    def test_out_of_window_stale_timestamp_rejected(self):
        verifier = self._verifier(timestamp_window_s=5.0)
        result = self._send(verifier, sequence=1, timestamp_s=900.0, now_s=1000.0)
        self.assertFalse(result.accepted)
        self.assertEqual(result.reason, EnvelopeRejectionReason.TIMESTAMP_OUT_OF_WINDOW)

    def test_within_window_accepted(self):
        verifier = self._verifier(timestamp_window_s=5.0)
        result = self._send(verifier, sequence=1, timestamp_s=997.0, now_s=1000.0)
        self.assertTrue(result.accepted)

    def test_rate_limit_trips(self):
        verifier = self._verifier(rate_limit_per_s=3.0, rate_limit_window_s=1.0)
        accepted = 0
        rejected_reasons = []
        for sequence in range(1, 6):
            result = self._send(verifier, sequence=sequence, timestamp_s=1000.0)
            if result.accepted:
                accepted += 1
            else:
                rejected_reasons.append(result.reason)
        self.assertEqual(accepted, 3)
        self.assertTrue(all(r == EnvelopeRejectionReason.RATE_LIMITED for r in rejected_reasons))

    def test_rate_limit_window_slides(self):
        verifier = self._verifier(rate_limit_per_s=2.0, rate_limit_window_s=1.0)
        self.assertTrue(self._send(verifier, sequence=1, timestamp_s=1000.0, now_s=1000.0).accepted)
        self.assertTrue(self._send(verifier, sequence=2, timestamp_s=1000.0, now_s=1000.1).accepted)
        self.assertFalse(
            self._send(verifier, sequence=3, timestamp_s=1000.0, now_s=1000.2).accepted
        )
        # After the 1s window has fully elapsed, budget replenishes.
        self.assertTrue(self._send(verifier, sequence=4, timestamp_s=1000.0, now_s=1001.3).accepted)

    def test_rejected_message_does_not_advance_sequence_state(self):
        """A forged/invalid message must not desync the replay guard for the real device."""
        verifier = self._verifier()
        payload = {"x": 1}
        bad_result = verifier.verify(
            device_id="drone-1",
            sequence=100,
            timestamp_s=1000.0,
            payload=payload,
            signature=b"\x00" * 64,  # invalid signature
            now_s=1000.0,
        )
        self.assertFalse(bad_result.accepted)
        # A legitimately signed, lower sequence number must still be accepted.
        good_result = self._send(verifier, sequence=1, timestamp_s=1000.0)
        self.assertTrue(good_result.accepted)


if __name__ == "__main__":
    unittest.main()
