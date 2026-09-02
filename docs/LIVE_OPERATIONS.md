# Live Operations

ArgusNet live mode separates sensing platforms, operational targets, fused tracks, and
simulation truth. Operator mode hides truth by default.

## Start a live session

Build the live-enabled Rust binaries once, then run `argusnet live --scene scene.smartscene`.
The command owns the daemon, continuous wall-clock simulation, rotating recording, and viewer,
and shuts down its children when the viewer closes. Use `--connect-only` to attach the viewer to
an existing endpoint.

The viewer uses `WatchFramesV2`, automatically reconnects, retains the newest 10,000 frames,
and displays sequence and drop counters. Legacy `WatchFrames` clients remain supported.

## Connecting to a remote daemon

`argusnet-viewer --live` streams in plaintext only over loopback. Subscribing to any other
endpoint requires the CA that signed the daemon certificate, and fails before the window opens
otherwise:

```bash
argusnet-viewer --scene scene.smartscene \
  --live tracker.internal:50051 \
  --live-tls-ca /etc/argusnet/ca.pem \
  --live-tls-cert /etc/argusnet/viewer.pem \
  --live-tls-key /etc/argusnet/viewer.key
```

`--live-tls-cert`/`--live-tls-key` are only needed when the daemon was started with
`--tls-client-ca` (mandatory mTLS). Add `--live-tls-domain` when the certificate name differs
from the endpoint host. All four also read from `ARGUSNET_TLS_CLIENT_CA`, `ARGUSNET_TLS_CERT`,
and `ARGUSNET_TLS_KEY`, matching `argusnetd`'s own flags. See `SECURITY.md`.

## Operator semantics

- Targets are scenario/operator-described objects of interest.
- Observations are individual sensor bearings.
- Tracks are fused estimates produced by the Rust tracking engine.
- Truth is simulation-only evaluation data and is hidden from operator presets.
- Safety events show command clamps and blocking Abort decisions.

## Troubleshooting

- A rising dropped counter means the subscriber or requested rate cannot keep up.
- `reconnecting` means the daemon is unavailable; the viewer retries every two seconds.
- Missing target metadata indicates a legacy ingest client.
- Safety Abort at startup usually means the scenario altitude is below the selected role's
  minimum terrain clearance.
