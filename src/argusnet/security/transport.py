"""TLS/mTLS transport configuration shared by the gRPC client and MQTT ingestion.

Certificates and keys are always sourced from filesystem paths (env vars by
default) — never embedded in code or committed to the repo. Connecting to a
non-loopback endpoint without TLS configured is refused (raises), not warned
about: the daemon and MQTT broker have no other authentication, so plaintext
off-loopback traffic is a false-data-injection / eavesdropping risk, not a
style issue.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass
from pathlib import Path

from argusnet.core.errors import ArgusNetError

__all__ = [
    "TLSConfig",
    "TransportSecurityError",
    "grpc_channel_credentials",
    "is_loopback_endpoint",
    "is_loopback_host",
    "mqtt_tls_kwargs",
]


class TransportSecurityError(ArgusNetError):
    """Raised when a non-loopback endpoint lacks the TLS material it requires."""


def is_loopback_host(host: str) -> bool:
    host = host.strip("[]")
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def is_loopback_endpoint(endpoint: str) -> bool:
    """``endpoint`` is ``host:port``, with IPv6 hosts written as ``[::1]:50051``."""
    if endpoint.startswith("["):
        host = endpoint[1 : endpoint.index("]")]
    else:
        host = endpoint.rsplit(":", 1)[0] if ":" in endpoint else endpoint
    return is_loopback_host(host)


@dataclass(frozen=True)
class TLSConfig:
    """Filesystem paths to PEM-encoded TLS material.

    ``ca_path`` verifies the peer (server verifying client certs, or client
    verifying the server). ``cert_path``/``key_path`` are this side's own
    identity, required for mTLS (client cert auth / server identity).
    """

    ca_path: Path | None = None
    cert_path: Path | None = None
    key_path: Path | None = None

    @property
    def configured(self) -> bool:
        return self.ca_path is not None or self.cert_path is not None

    def read_ca(self) -> bytes | None:
        return self.ca_path.read_bytes() if self.ca_path else None

    def read_cert(self) -> bytes | None:
        return self.cert_path.read_bytes() if self.cert_path else None

    def read_key(self) -> bytes | None:
        return self.key_path.read_bytes() if self.key_path else None

    @classmethod
    def from_env(cls, prefix: str) -> TLSConfig:
        def _path(name: str) -> Path | None:
            value = os.environ.get(f"{prefix}_{name}")
            return Path(value) if value else None

        return cls(ca_path=_path("CA"), cert_path=_path("CERT"), key_path=_path("KEY"))


def grpc_channel_credentials(config: TLSConfig):  # noqa: ANN201 - grpc.ChannelCredentials, optional dep
    import grpc

    if config.cert_path and not config.key_path:
        raise TransportSecurityError("TLS cert_path is set without a matching key_path.")
    return grpc.ssl_channel_credentials(
        root_certificates=config.read_ca(),
        private_key=config.read_key(),
        certificate_chain=config.read_cert(),
    )


def mqtt_tls_kwargs(config: TLSConfig) -> dict[str, str]:
    """Keyword arguments for ``paho.mqtt.client.Client.tls_set(**kwargs)``."""
    kwargs: dict[str, str] = {}
    if config.ca_path:
        kwargs["ca_certs"] = str(config.ca_path)
    if config.cert_path:
        kwargs["certfile"] = str(config.cert_path)
    if config.key_path:
        kwargs["keyfile"] = str(config.key_path)
    return kwargs
