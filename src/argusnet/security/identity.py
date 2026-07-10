"""Ed25519 device identity, signed observation envelopes, and anti-replay gating.

Threat model: any party on the MQTT broker or network can publish a
``node_id``/``target_id`` — those are unauthenticated strings. This module adds
a cryptographic identity layer on top: each device signs a canonical envelope
(``device_id`` ‖ ``sequence`` ‖ ``timestamp_s`` ‖ payload) with an Ed25519
private key; the fusion side only ever holds public keys (:class:`DeviceRegistry`)
and verifies. :class:`EnvelopeVerifier` additionally rejects replayed/duplicate
sequences, out-of-window timestamps, and per-device floods, so a valid signature
alone is not enough to inject data.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from argusnet.core.errors import ArgusNetError

__all__ = [
    "DeviceRegistry",
    "EnvelopeRejectionReason",
    "EnvelopeVerifier",
    "VerificationResult",
    "canonical_envelope_bytes",
    "envelope_signing_bytes",
    "generate_keypair",
    "public_key_bytes",
    "sign_payload",
    "verify_signature",
]


class EnvelopeRejectionReason(str, Enum):
    UNKNOWN_DEVICE = "unknown_device"
    BAD_SIGNATURE = "bad_signature"
    SEQUENCE_REPLAY = "sequence_replay"
    TIMESTAMP_OUT_OF_WINDOW = "timestamp_out_of_window"
    RATE_LIMITED = "rate_limited"


class IdentityError(ArgusNetError):
    """Raised for device-registry / key-material problems (not per-message rejections)."""


# ---------------------------------------------------------------------------
# Signing primitives
# ---------------------------------------------------------------------------


def generate_keypair() -> tuple[Ed25519PrivateKey, Ed25519PublicKey]:
    private_key = Ed25519PrivateKey.generate()
    return private_key, private_key.public_key()


def public_key_bytes(public_key: Ed25519PublicKey) -> bytes:
    return bytes(public_key.public_bytes(Encoding.Raw, PublicFormat.Raw))


def canonical_envelope_bytes(
    device_id: str, sequence: int, timestamp_s: float, payload: bytes
) -> bytes:
    """Deterministic bytes covered by the signature.

    Uses a unit-separator-delimited header so no field can be confused with
    another regardless of its contents (device IDs are not restricted to a
    delimiter-safe charset).
    """
    header = f"{device_id}\x1f{int(sequence)}\x1f{float(timestamp_s)!r}\x1f".encode()
    return header + payload


def envelope_signing_bytes(
    device_id: str, sequence: int, timestamp_s: float, payload: dict
) -> bytes:
    """Canonicalize a JSON-able payload dict and wrap it in the signed envelope."""
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return canonical_envelope_bytes(device_id, sequence, timestamp_s, canonical_payload)


def sign_payload(
    private_key: Ed25519PrivateKey,
    *,
    device_id: str,
    sequence: int,
    timestamp_s: float,
    payload: dict,
) -> bytes:
    """Reference signer used by simulated devices and tests."""
    return bytes(
        private_key.sign(envelope_signing_bytes(device_id, sequence, timestamp_s, payload))
    )


def verify_signature(
    public_key: Ed25519PublicKey,
    signature: bytes,
    *,
    device_id: str,
    sequence: int,
    timestamp_s: float,
    payload: dict,
) -> bool:
    try:
        signed_bytes = envelope_signing_bytes(device_id, sequence, timestamp_s, payload)
        public_key.verify(signature, signed_bytes)
        return True
    except InvalidSignature:
        return False


# ---------------------------------------------------------------------------
# Authorized-device registry
# ---------------------------------------------------------------------------


@dataclass
class DeviceRegistry:
    """Maps ``device_id`` -> Ed25519 public key. Holds no private key material."""

    _keys: dict[str, Ed25519PublicKey] = field(default_factory=dict)

    @classmethod
    def from_directory(cls, directory: str | Path) -> DeviceRegistry:
        """Load ``<device_id>.pub`` files (raw 32-byte Ed25519 public keys) from a directory."""
        registry = cls()
        root = Path(directory)
        if not root.is_dir():
            raise IdentityError(f"device registry directory does not exist: {root}")
        for path in sorted(root.glob("*.pub")):
            raw = path.read_bytes()
            try:
                key = Ed25519PublicKey.from_public_bytes(raw)
            except ValueError as exc:
                raise IdentityError(f"invalid Ed25519 public key in {path}: {exc}") from exc
            registry._keys[path.stem] = key
        return registry

    @classmethod
    def from_mapping(cls, mapping: dict[str, bytes]) -> DeviceRegistry:
        registry = cls()
        for device_id, raw in mapping.items():
            registry._keys[device_id] = Ed25519PublicKey.from_public_bytes(raw)
        return registry

    def public_key(self, device_id: str) -> Ed25519PublicKey | None:
        return self._keys.get(device_id)

    def __contains__(self, device_id: str) -> bool:
        return device_id in self._keys

    def __len__(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Signature + anti-replay verification
# ---------------------------------------------------------------------------


@dataclass
class VerificationResult:
    accepted: bool
    reason: EnvelopeRejectionReason | None = None
    detail: str = ""


@dataclass
class _DeviceReplayState:
    last_sequence: int | None = None
    recent_event_times: deque[float] = field(default_factory=deque)


class EnvelopeVerifier:
    """Verifies signed envelopes and enforces anti-replay / rate-limit gating.

    State only advances (sequence high-water mark, rate-limit window) when a
    message is fully accepted, so a forged or malformed message cannot be used
    to desynchronize the replay guard for a legitimate device.
    """

    def __init__(
        self,
        registry: DeviceRegistry,
        *,
        timestamp_window_s: float = 5.0,
        rate_limit_per_s: float = 50.0,
        rate_limit_window_s: float = 1.0,
    ) -> None:
        if timestamp_window_s <= 0.0:
            raise IdentityError("timestamp_window_s must be > 0.")
        if rate_limit_per_s <= 0.0 or rate_limit_window_s <= 0.0:
            raise IdentityError("rate_limit_per_s and rate_limit_window_s must be > 0.")
        self._registry = registry
        self._timestamp_window_s = timestamp_window_s
        self._rate_limit_per_s = rate_limit_per_s
        self._rate_limit_window_s = rate_limit_window_s
        self._devices: dict[str, _DeviceReplayState] = {}

    def verify(
        self,
        *,
        device_id: str,
        sequence: int,
        timestamp_s: float,
        payload: dict,
        signature: bytes,
        now_s: float | None = None,
    ) -> VerificationResult:
        public_key = self._registry.public_key(device_id)
        if public_key is None:
            return VerificationResult(False, EnvelopeRejectionReason.UNKNOWN_DEVICE, device_id)

        if not verify_signature(
            public_key,
            signature,
            device_id=device_id,
            sequence=sequence,
            timestamp_s=timestamp_s,
            payload=payload,
        ):
            return VerificationResult(False, EnvelopeRejectionReason.BAD_SIGNATURE, device_id)

        now = time.time() if now_s is None else now_s
        if not (now - self._timestamp_window_s <= timestamp_s <= now + self._timestamp_window_s):
            return VerificationResult(
                False,
                EnvelopeRejectionReason.TIMESTAMP_OUT_OF_WINDOW,
                f"timestamp_s={timestamp_s} now={now}",
            )

        state = self._devices.setdefault(device_id, _DeviceReplayState())
        if state.last_sequence is not None and sequence <= state.last_sequence:
            return VerificationResult(
                False,
                EnvelopeRejectionReason.SEQUENCE_REPLAY,
                f"sequence={sequence} last_sequence={state.last_sequence}",
            )

        cutoff = now - self._rate_limit_window_s
        while state.recent_event_times and state.recent_event_times[0] < cutoff:
            state.recent_event_times.popleft()
        max_events = self._rate_limit_per_s * self._rate_limit_window_s
        if len(state.recent_event_times) >= max_events:
            return VerificationResult(False, EnvelopeRejectionReason.RATE_LIMITED, device_id)

        state.last_sequence = sequence
        state.recent_event_times.append(now)
        return VerificationResult(True)
