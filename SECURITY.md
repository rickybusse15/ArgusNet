# Security

ArgusNet is a **local-first research/simulation platform**: a Python simulation +
CLI, a Rust sensor-fusion daemon (`argusnetd`), and a Bevy 3D viewer. There is no
web frontend, no database, no user accounts, and no cloud service. This document
describes the actual trust boundaries, what is enforced at each of them, and what
is deliberately left as the operator's responsibility.

## Threat model

The realistic attack surface is:

1. **The gRPC daemon** (`argusnetd`) — Python and any other client talk to it
   over tonic/gRPC, default bind `127.0.0.1:50051`.
2. **The MQTT ingestion path** — drones/sensors publish observation and
   node-state JSON to a broker; `MQTTIngestionAdapter` consumes it.
3. **Untrusted files** — replay JSON, scene manifests/packages, GeoTIFF DEMs,
   GLB assets that may come from a shared dataset or scene package.
4. **CI/CD & supply chain.**

Classic web-app vulnerability classes (XSS, CSRF, SQL injection, IDOR, JWT/session
handling, CORS, cookies) do not apply to this codebase and are not treated as
findings here.

## What is enforced today

### Device identity & signed envelopes (drone↔device, drone↔drone)

`node_id`/`target_id`/`sensor_type` on the wire are plain strings — on their own
they authenticate nothing. `src/argusnet/security/identity.py` adds a
cryptographic identity layer on top:

- Each device holds an **Ed25519 private key**; the fusion side only ever holds
  the corresponding **public key**, via `DeviceRegistry` (loaded from
  `<device_id>.pub` raw-key files in an operator-provisioned directory — never
  committed to the repo).
- A device signs a canonical envelope — `device_id ‖ sequence ‖ timestamp_s ‖
  payload` — over its observation/node-state JSON before publishing.
- `EnvelopeVerifier` checks the signature against the registry, a monotonic
  per-device **sequence number** (rejects replays/duplicates), a bounded
  **timestamp window** (rejects stale/future-dated messages independent of the
  fusion engine's own timestamp-skew quality gate), and a **per-device rate
  limit** (one compromised/authenticated device can't flood fusion).
- This is wired into `MQTTIngestionAdapter`: if a `device_registry` is
  configured, every message must carry a valid, non-replayed, in-window,
  correctly-signed envelope or it is dropped and logged with a rejection
  reason. If no registry is configured, MQTT behaves as before (legacy/local
  path) — see "Loopback/local fast path" below.
- `proto/argusnet/v1/world_model.proto` carries `device_id`, `sequence`,
  `signature`, `signer_pubkey_id` as additive fields on `BearingObservation`
  and `NodeState` so a signed transport can round-trip the envelope; an empty
  `signature` means "legacy/unsigned path."

### Transport encryption (mTLS)

- **gRPC client** (`TrackingService`, `src/argusnet/adapters/argusnet_grpc.py`):
  refuses (raises `TransportSecurityError`) to connect to a non-loopback
  endpoint unless TLS is configured via `ARGUSNET_GRPC_TLS_CA` (+
  `ARGUSNET_GRPC_TLS_CERT`/`_KEY` for mTLS) or an explicit `TLSConfig`.
- **gRPC server** (`argusnetd`, `rust/argusnet-server`): refuses to bind a
  non-loopback address unless `--tls-cert`/`--tls-key` (and optionally
  `--tls-client-ca` for mandatory mTLS) are supplied — same flags are also
  readable from `ARGUSNET_TLS_CERT`/`ARGUSNET_TLS_KEY`/`ARGUSNET_TLS_CLIENT_CA`.
- **MQTT**: `MQTTIngestionAdapter` refuses to connect to a non-loopback broker
  without TLS (`ARGUSNET_MQTT_TLS_CA`/`_CERT`/`_KEY`) and without a
  `device_registry` configured; broker credentials are read from
  `ARGUSNET_MQTT_USERNAME`/`ARGUSNET_MQTT_PASSWORD`.
- Certificates/keys are always filesystem paths from env/config — never
  embedded in code or committed.

### Loopback/local fast path

Single-machine simulation and tests do not require keys or certificates:
loopback gRPC (`127.0.0.1`/`localhost`/`::1`) and a loopback MQTT broker keep
working unauthenticated and in plaintext, matching the existing local
simulation workflow. The moment an endpoint is non-loopback, the checks above
apply and plaintext/unauthenticated connections are refused rather than merely
warned about.

### Daemon resource limits

`argusnetd` (`rust/argusnet-server`) bounds a single unauthenticated client's
ability to exhaust the daemon: a 16 MiB cap on decoded/encoded gRPC message
size, a per-connection concurrency limit, and a per-request timeout. Binding a
non-loopback address without TLS is refused, and a warning is printed if the
listen address is loopback-only but still reachable from other local users.

### Untrusted file handling

- **Scene manifests** (`src/argusnet/world/scene_loader.py`): every path field
  (`layers[].asset_path`, `metadata.environment`/`style`, `replay.path`) is
  checked for absolute paths and `..` traversal, then re-verified with a
  filesystem containment check (`resolve()` + `relative_to()`) against the
  scene package's own directory before it is ever read. A malicious scene
  package cannot reference files outside itself.
- **Replay JSON / recording index files** (`src/argusnet/evaluation/replay.py`,
  `recording.py`): capped via `src/argusnet/core/io_limits.py` before
  `json.loads` (default 256 MB) to bound memory use on an oversized or
  maliciously large file.
- **GeoTIFF DEMs** (`src/argusnet/world/_scene_gis.py`): pixel count is checked
  against a cap before the raster is decoded into memory.
- **MQTT payloads**: capped in size before `json.loads`; numeric fields
  (angles, standard deviations, confidence, position/velocity, timestamps) are
  validated finite/in-range via `src/argusnet/core/validation.py` before being
  turned into observations.

### Supply chain / CI

- CI workflows run with `permissions: contents: read` by default.
- Third-party GitHub Actions are pinned to commit SHAs, not floating tags.
- A `security` CI job runs `pip-audit`, `bandit`, `cargo audit`, and
  `gitleaks` (currently non-blocking while the first pass of findings is
  triaged — see the job in `.github/workflows/ci.yml`).
- Dependabot (`.github/dependabot.yml`) watches pip, cargo, and GitHub Actions
  for updates.
- `Cargo.lock` is committed (this workspace only ships binaries/services, not
  a published library, so a reproducible lockfile is the right default).

### Reviewed and already safe (informational)

- `src/argusnet/evaluation/export.py` only **writes** KML/GPX XML via
  `xml.etree.ElementTree`; it never parses untrusted XML, so there is no XXE
  surface.
- Subprocess calls (`cli/main.py`, `adapters/argusnet_grpc.py`) always use
  argument-list `subprocess.Popen(...)`, never `shell=True`.
- The generated tracker-config YAML interpolates `data_association_mode`, but
  `TrackerConfig.__post_init__` already restricts it to a fixed allowlist
  (`labeled`/`gnn`/`jpda`), and the YAML renderer re-checks that allowlist
  immediately before interpolating, as defense in depth.

## What is *not* authenticated — read this before deploying off one machine

**`argusnetd` has no authentication model beyond TLS/mTLS.** Anyone who can
reach a loopback-bound daemon on the same machine can ingest arbitrary frames.
Binding it to a non-loopback address requires TLS, but TLS alone is transport
security, not authorization — anyone holding a valid client certificate (for
mTLS) or reaching the endpoint (for server-only TLS) can call every RPC.

**Do not expose `argusnetd` on an untrusted network without mTLS, and do not
treat mTLS as authorization** — it authenticates the transport, not what a
connected client is allowed to ingest. For genuine multi-device deployments,
combine non-loopback TLS with the signed-envelope device identity layer
described above so *individual observations* are attributable to a specific,
registered device, independent of which client happened to relay them.

## Deployment checklist

- [ ] Keep `argusnetd` bound to `127.0.0.1` unless you have a specific reason
      not to.
- [ ] If binding non-loopback: provision `--tls-cert`/`--tls-key` (and
      `--tls-client-ca` for mTLS) before starting the daemon; it will refuse
      to start otherwise.
- [ ] If ingesting from real devices over MQTT: provision an Ed25519 keypair
      per device, populate a `DeviceRegistry` directory with each device's
      public key, and pass `device_registry=...` to `MQTTIngestionAdapter`.
- [ ] If the MQTT broker is not on loopback: configure
      `ARGUSNET_MQTT_TLS_CA`/`_CERT`/`_KEY` and broker credentials via
      `ARGUSNET_MQTT_USERNAME`/`_PASSWORD`.
- [ ] Only load scene packages, replay files, and DEMs from sources you trust
      — the size/path-traversal guards reduce blast radius but are not a
      substitute for provenance.
- [ ] Rotate/revoke a device's key by removing its `.pub` file from the
      registry directory; there is no automated revocation or rotation
      tooling (see "Non-goals" below).

## Non-goals

- **Automated PKI / certificate issuance / key rotation.** This repo provides
  the mechanism (Ed25519 device registry, mTLS config loading) and documents
  manual provisioning above; automating issuance and rotation is a follow-up,
  not implemented here.
- **Authorization beyond "is this device registered."** The identity layer
  answers "did a registered device sign this," not "is this device allowed to
  claim this `target_id`" or any finer-grained policy.
- **Detection evasion, DoS mitigation beyond basic resource caps,** or
  hardening against a compromised host — out of scope for a local-first
  research tool.

## Reporting a vulnerability

Please open a private security advisory on the repository (GitHub Security
Advisories) rather than a public issue. Include reproduction steps and the
affected component (Python package, `argusnetd`, or the viewer).
