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
import socket
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


def _ip_is_loopback(address: str) -> bool:
    """True for a loopback address, including an IPv4-mapped IPv6 loopback.

    ``ipaddress`` reports ``::ffff:127.0.0.1`` as not loopback. That address
    still dials 127.0.0.1, including when a resolver returns it for
    ``localhost``, so it is treated as loopback.
    """
    try:
        ip = ipaddress.ip_address(address.split("%", 1)[0])
    except ValueError:
        return False
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None:
        return bool(mapped.is_loopback)
    return bool(ip.is_loopback)


def _localhost_resolves_only_to_loopback(host: str) -> bool:
    """``localhost`` is loopback only when every resolved address is.

    Trusting the name without resolving it lets a poisoned hosts file or DNS
    answer send plaintext to a non-local address. Any other name is not
    treated as loopback even if it currently resolves to 127.0.0.1: that
    would widen the plaintext exemption to attacker-controlled DNS.
    """
    if host.lower() != "localhost":
        return False
    try:
        infos = socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)
    except socket.gaierror:
        return False
    for info in infos:
        address = info[4][0]
        # getaddrinfo's sockaddr is a union; only the host string is an address.
        if not isinstance(address, str) or not _ip_is_loopback(address):
            return False
    return bool(infos)


def is_loopback_host(host: str) -> bool:
    host = host.strip().strip("[]")
    if _localhost_resolves_only_to_loopback(host):
        return True
    return _ip_is_loopback(host)


def endpoint_host(endpoint: str) -> str:
    """Host a URL parser would dial, not the substring before the first colon.

    ``127.0.0.1:80@evil.example:443`` is userinfo ``127.0.0.1:80`` and host
    ``evil.example``. Splitting on the first colon reports the loopback
    address and would exempt a remote plaintext connection.
    """
    if "://" in endpoint:
        endpoint = endpoint.split("://", 1)[1]
    for separator in "/?#":
        endpoint = endpoint.split(separator, 1)[0]
    if "@" in endpoint:
        endpoint = endpoint.rsplit("@", 1)[1]
    if endpoint.startswith("["):
        end = endpoint.find("]")
        if end == -1:
            return endpoint
        return endpoint[1:end]
    if ":" in endpoint:
        return endpoint.rsplit(":", 1)[0]
    return endpoint


def is_loopback_endpoint(endpoint: str) -> bool:
    """``endpoint`` is ``host:port``, with IPv6 hosts written as ``[::1]:50051``."""
    return is_loopback_host(endpoint_host(endpoint))


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
