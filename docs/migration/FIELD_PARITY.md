# G0 field parity matrix

Guardrail **G0** in [`DELIVERY_PLAN.md`](DELIVERY_PLAN.md) asks that every payload
be "preserved, normalized explicitly or rejected; no silent missing-field
success". The failure it guards against is not a dropped field — several are
dropped deliberately — but a field dropped *without anyone having said so*.

This file records what is supposed to happen to each of the 171 fields in
`proto/argusnet/v1/world_model.proto`.
[`scripts/check_field_parity.py`](../../scripts/check_field_parity.py) fails if a
proto field has no row here, if a row survives its field, or if a row carries a
disposition outside the vocabulary below. It deliberately does **not** judge
whether the handling is correct — only that it was stated.

## Dispositions

| Value | Meaning |
| --- | --- |
| `carried` | Reaches every consumer that asks for it. |
| `dropped-intentional` | Not represented in the Rust runtime type on purpose. The note says where it does survive. |
| `inert` | On the wire and in the generated bindings, but nothing reads or writes a meaningful value on either side. |
| `grafted` | Preserved only because one code path copies it around the runtime type, so a different RPC about the same frame may omit it. |
| `consumer-gap` | Produced by one side, never consumed by the other. Nothing is lost on the wire; the far side ignores it. |
| `derived` | Computed from another field when encoding and never read back, so it carries no information of its own. |

## What the inventory found

Of 171 fields: **134 `carried`**, 16 `consumer-gap`, 16 `inert`,
3 `dropped-intentional`, 2 `derived`. No `grafted` rows remain: P4.0 closed the
only two by giving `PlatformFrame` real fields.

### How much the `carried` rows are worth

`carried` means both sides reference the field. That was established partly by
searching for field names, which cannot tell `TrackState.confidence` from
`BearingObservation.confidence`, and cannot see a field assigned a constant
rather than passed through. The all-field round trips in
`rust/argusnet-proto/tests/roundtrip.rs` are what turn the claim into evidence,
and running them corrected three rows this file originally got wrong:
`TrackState.acceleration_mps2` is a hardcoded zero vector rather than a
`consumer-gap`, and `confidence` and `last_seen_s` are aliases of
`quality_score` and `timestamp_s` rather than `carried`.

The round trips now cover `NodeState`, `BearingObservation`, `TruthState`,
`TrackState`, `PlatformMetrics`, `TrackerConfig`, `ObservationRejection`,
`TargetMetadata`, `SafetyEvent` and `PlatformFrame`, in both the all-present and
all-absent shapes, plus invalid covariance lengths and missing required
submessages. What still rests on inspection alone is the service-level messages
that have no runtime counterpart to round-trip through — `HealthResponse`,
`MissionStatus*`, `WatchFramesV2*` — and the mission-zone messages, which finding
3 below records as unused.

Three findings are worth a reader's attention. All were verified by reading the
conversion code, not inferred from a name search.

**0. The covariance matrix was transposed on the wire — fixed here.**
`covariance_to_vec` emitted `matrix.iter()`, which is nalgebra's column-major
order, into a field the proto names `covariance_row_major`. Every reader honours
that name: `covariance_from_vec` uses `from_row_slice`, and the Python adapter
uses numpy's C-order `reshape(6, 6)`. So Rust wrote the transpose of what it
meant. A covariance matrix is symmetric, which is why nothing ever noticed — but
the Kalman update uses the Joseph form without an explicit symmetrization step,
so rounding leaves small asymmetries, and any genuinely non-symmetric matrix
would have been corrupted silently. The fix transposes before flattening and is
a no-op for symmetric data; the cross-language parity fixtures pass unchanged.

**1. `PlatformFrame.target_metadata` and `.safety_events` — fixed in P4.0.**
These used to answer differently depending on which RPC asked. The runtime
`PlatformFrame` had no fields for either, so `frame_to_pb_ref` emitted
`Vec::new()`, and the server compensated in the ingest path only by grafting
the request's copies onto the response. Watchers saw that patched response;
`LatestFrame`, rebuilt from the runtime type, returned both empty, while the
viewer's replay path modelled both and expected them.

P4.0 gave `PlatformFrame` real fields, threaded them through `ingest_frame`,
and deleted the graft. `target_metadata_and_safety_events_reach_ingest_and_latest_frame_alike`
now asserts the symmetry it once pinned the asymmetry of.

**1b. `TrackState.mode_probability_cv` did not survive being unset — fixed in
P4.0.** The runtime type models it as `Option<f64>`; the wire field is a bare
`double`, so "unset" needs an in-band value. Both decoders already read `0.0` as
absent — `track_from_pb`, and the viewer's replay reader — and the same proto
uses that convention for `NodeState.max_range_m`. Only the *encoder* disagreed:
it sent `1.0` for `None`, so `None` came back as `Some(1.0)`, reporting total
confidence in the constant-velocity mode where the truth was "no IMM estimate at
all" — a fabricated value, not a lost one, which is the worse of the two.

The fix is one constant, `UNSET_MODE_PROBABILITY_CV = 0.0`, used by both
directions, so the encoder now agrees with the decoders that were already there.
No proto field is renumbered or reinterpreted; the interpretation was already
the readers'. The residual cost is that a genuine `Some(0.0)` cannot be told
from absent, and that is acceptable because the only producer cannot emit one:
`argusnet-core`'s IMM mode probability is a normalized posterior whose numerator
is floored at `1.0e-30`, so it is strictly positive. That claim is asserted by
running the filter (`the_imm_filter_never_emits_a_zero_mode_probability`) rather
than by reading it.

**2. The attestation fields are inert.** `device_id`, `signature` and
`signer_pubkey_id` on both `NodeState` and `BearingObservation` are always
written as an empty string by the Rust conversions and never read back; the
runtime types have no equivalent. A client that signs its observations has that
signature discarded silently, and nothing in the schema says so. Six fields.

**3. The protobuf mission-zone messages are unused.** Nothing anywhere
constructs a `pb::MissionZone`; `MissionStatusResponse` hardcodes `zones: vec![]`
and `active_zone_count: 0`. Zones do reach the viewer, but through replay JSON
using a *separate* viewer-side struct that happens to share the name — an easy
thing to mistake for the same type.

`NodeState.sensor_type`, `fov_half_angle_deg` and `max_range_m` were held up
here as the model to follow: dropped from the runtime type on purpose, with a
comment in `argusnet-proto/src/lib.rs` saying why and where they survive
instead.

**That reading did not survive P4.8d-i, and the paragraph is kept only so the
correction is visible.** The comment was accurate about the *tracker*, which
genuinely does not need them, and wrong about the conversion, which is shared
with every other consumer. The encoder did not merely drop them: it
*substituted* `"optical" / 180.0 / 0.0` on every call, so a thermal sensor
round-tripped as an optical one and the substitution was indistinguishable
from a producer that had said so. "Dropped on purpose, documented" is only a
defensible disposition when the field is absent on the way out; a field that
comes back holding a value nobody supplied is `lossy`, whatever the comment
says. All three are now `carried`, along with the new `battery_fraction`.

## Scope

This covers the protobuf wire contract. The replay JSON schema
(`docs/replay-schema.json`) is a separate contract, already enforced by
`validate_replay_document`. G0's other half — the installed-wheel contract job —
landed separately in #53, corrected by #55.

## Matrix

| Field | Disposition | Note |
| --- | --- | --- |
| `BearingObservation.bearing_std_rad` | `carried` |  |
| `BearingObservation.confidence` | `carried` |  |
| `BearingObservation.device_id` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `BearingObservation.direction` | `carried` |  |
| `BearingObservation.node_id` | `carried` |  |
| `BearingObservation.origin` | `carried` |  |
| `BearingObservation.sequence` | `carried` |  |
| `BearingObservation.signature` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `BearingObservation.signer_pubkey_id` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `BearingObservation.target_id` | `carried` |  |
| `BearingObservation.timestamp_s` | `carried` |  |
| `GetConfigResponse.config` | `carried` |  |
| `HealthResponse.active_node_count` | `carried` |  |
| `HealthResponse.ingest_latency` | `carried` |  |
| `HealthResponse.mean_frame_rate_hz` | `carried` |  |
| `HealthResponse.mean_ingest_latency_s` | `carried` |  |
| `HealthResponse.node_health` | `carried` |  |
| `HealthResponse.processed_frame_count` | `carried` |  |
| `HealthResponse.stale_node_count` | `carried` |  |
| `HealthResponse.started_at_utc` | `carried` |  |
| `HealthResponse.status` | `carried` |  |
| `IngestFrameRequest.node_states` | `carried` |  |
| `IngestFrameRequest.observations` | `carried` |  |
| `IngestFrameRequest.safety_events` | `carried` |  |
| `IngestFrameRequest.target_metadata` | `carried` |  |
| `IngestFrameRequest.timestamp_s` | `carried` |  |
| `IngestFrameRequest.truths` | `carried` |  |
| `IngestFrameResponse.frame` | `carried` |  |
| `LatencyHistogram.max_s` | `carried` |  |
| `LatencyHistogram.p50_s` | `carried` |  |
| `LatencyHistogram.p95_s` | `carried` |  |
| `LatencyHistogram.p99_s` | `carried` |  |
| `LatencyHistogram.sample_count` | `carried` |  |
| `LatestFrameResponse.frame` | `carried` |  |
| `MissionStatusRequest.scenario_name` | `consumer-gap` | Populated in Rust; the Python adapter does not read it. |
| `MissionStatusResponse.active_track_count` | `carried` |  |
| `MissionStatusResponse.active_track_ids` | `consumer-gap` | Populated in Rust; the Python adapter does not read it. |
| `MissionStatusResponse.active_zone_count` | `inert` | Server hardcodes `0`, for the same reason as `zones`. |
| `MissionStatusResponse.elapsed_s` | `consumer-gap` | Populated in Rust; the Python adapter does not read it. |
| `MissionStatusResponse.scenario_name` | `consumer-gap` | Populated in Rust; the Python adapter does not read it. |
| `MissionStatusResponse.zones` | `inert` | Server hardcodes `vec![]` -- zones are pushed from Python and not stored server-side. |
| `MissionZone.center` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZone.label` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZone.priority` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZone.radius_m` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZone.zone_id` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZone.zone_type` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `MissionZoneUpdate.timestamp_s` | `carried` |  |
| `MissionZoneUpdate.zones` | `inert` | Nothing constructs the protobuf `MissionZone`. Zones reach the viewer through replay JSON, via a separate viewer-side struct that happens to share the name. |
| `NodeHealthMetrics.accepted_count` | `carried` |  |
| `NodeHealthMetrics.health_score` | `carried` |  |
| `NodeHealthMetrics.last_seen_s` | `carried` |  |
| `NodeHealthMetrics.mean_latency_s` | `carried` |  |
| `NodeHealthMetrics.node_id` | `carried` |  |
| `NodeHealthMetrics.observation_rate_hz` | `carried` |  |
| `NodeHealthMetrics.rejected_count` | `carried` |  |
| `NodeState.battery_fraction` | `carried` | `optional` on the wire, `Option<f64>` in the runtime type. Unlike the two sensor fields below it does **not** use the `0.0`-means-unset convention: `Some(0.0)` is a flat battery and `None` is a platform with no battery, and a ground station must not read as a drone at zero percent. Python's `-1.0` not-applicable sentinel converts to and from wire absence in `adapters/argusnet_grpc.py`. |
| `NodeState.device_id` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `NodeState.fov_half_angle_deg` | `carried` | `Option<f64>` in the runtime type, a bare `double` on the wire where `0` means unset. A genuine `Some(0.0)` is therefore indistinguishable from absent, which costs nothing: a zero-width field of view is not a reading. Degrees, not radians, deliberately — a verbatim passthrough of the wire and replay field of the same name. |
| `NodeState.health` | `carried` |  |
| `NodeState.is_mobile` | `carried` |  |
| `NodeState.max_range_m` | `carried` | `Option<f64>` in the runtime type, a bare `double` on the wire where `0` means unset. Previously the replay path deserialized the sentinel to `Some(0.0)`, giving a zero-radius radar ring where the 400 m default was meant. |
| `NodeState.node_id` | `carried` |  |
| `NodeState.position` | `carried` |  |
| `NodeState.sensor_type` | `carried` | proto3 strings have no presence, so `""` on the wire means "unspecified" and decodes to `argusnet_types::DEFAULT_SENSOR_TYPE`. Not merely a label: `argusnet-sim`'s observation stage classifies it into a `SensorBand` that moves both the noise multiplier and the confidence. |
| `NodeState.sequence` | `carried` |  |
| `NodeState.signature` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `NodeState.signer_pubkey_id` | `inert` | Rust writes an empty string and never reads it back; the runtime types have no such field, so a signed payload is discarded silently. |
| `NodeState.timestamp_s` | `carried` |  |
| `NodeState.velocity` | `carried` |  |
| `ObservationRejection.attempted_point` | `carried` |  |
| `ObservationRejection.blocker_type` | `carried` |  |
| `ObservationRejection.closest_point` | `carried` |  |
| `ObservationRejection.detail` | `carried` |  |
| `ObservationRejection.first_hit_range_m` | `carried` |  |
| `ObservationRejection.node_id` | `carried` |  |
| `ObservationRejection.origin` | `carried` |  |
| `ObservationRejection.reason` | `carried` |  |
| `ObservationRejection.target_id` | `carried` |  |
| `ObservationRejection.timestamp_s` | `carried` |  |
| `PlatformFrame.generation_rejections` | `carried` |  |
| `PlatformFrame.metrics` | `carried` |  |
| `PlatformFrame.nodes` | `carried` |  |
| `PlatformFrame.observations` | `carried` |  |
| `PlatformFrame.rejected_observations` | `carried` |  |
| `PlatformFrame.safety_events` | `carried` | Real field on the runtime `PlatformFrame` since P4.0. Round-tripped by `frame_annotations_round_trip_every_field`. |
| `PlatformFrame.target_metadata` | `carried` | Real field on the runtime `PlatformFrame` since P4.0. Round-tripped by `frame_annotations_round_trip_every_field`. |
| `PlatformFrame.timestamp_s` | `carried` |  |
| `PlatformFrame.tracks` | `carried` |  |
| `PlatformFrame.truths` | `carried` |  |
| `PlatformMetrics.accepted_observation_count` | `carried` |  |
| `PlatformMetrics.accepted_observations_by_target` | `carried` |  |
| `PlatformMetrics.active_track_count` | `carried` |  |
| `PlatformMetrics.max_error_m` | `carried` |  |
| `PlatformMetrics.mean_error_m` | `carried` |  |
| `PlatformMetrics.mean_measurement_std_m` | `carried` |  |
| `PlatformMetrics.observation_count` | `carried` |  |
| `PlatformMetrics.rejected_observation_count` | `carried` |  |
| `PlatformMetrics.rejected_observations_by_target` | `carried` |  |
| `PlatformMetrics.rejection_counts` | `carried` |  |
| `PlatformMetrics.track_errors_m` | `carried` |  |
| `SafetyEvent.blocked` | `carried` |  |
| `SafetyEvent.command_clamped` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `SafetyEvent.event_id` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `SafetyEvent.reason` | `carried` |  |
| `SafetyEvent.state` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `SafetyEvent.subject_id` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `SafetyEvent.timestamp_s` | `carried` |  |
| `SafetyEvent.violations` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.behavior` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.confidence` | `carried` |  |
| `TargetMetadata.first_seen_s` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.last_seen_s` | `carried` |  |
| `TargetMetadata.notes` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.operator_label` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.priority` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TargetMetadata.status` | `carried` |  |
| `TargetMetadata.target_id` | `carried` |  |
| `TargetMetadata.target_type` | `consumer-gap` | Handled in Python; no Rust runtime representation. |
| `TrackState.acceleration_mps2` | `inert` | `track_to_pb` assigns a hardcoded `vec![0.0, 0.0, 0.0]`; no runtime field backs it. |
| `TrackState.confidence` | `derived` | An alias of `quality_score` (`quality_score.unwrap_or(0.0)`), never read back. |
| `TrackState.contributing_nodes` | `carried` |  |
| `TrackState.covariance_row_major` | `carried` |  |
| `TrackState.last_seen_s` | `derived` | An alias of `timestamp_s`, never read back. |
| `TrackState.lifecycle_state` | `carried` |  |
| `TrackState.measurement_std_m` | `carried` |  |
| `TrackState.mode_probability_cv` | `carried` | `Option<f64>` in the runtime type, a bare `double` on the wire. `0.0` means unset — the same convention `NodeState.max_range_m` uses. A genuine `Some(0.0)` is therefore indistinguishable from absent; the IMM filter cannot emit one. |
| `TrackState.position` | `carried` |  |
| `TrackState.quality_score` | `carried` |  |
| `TrackState.stale_steps` | `carried` |  |
| `TrackState.timestamp_s` | `carried` |  |
| `TrackState.track_id` | `carried` |  |
| `TrackState.update_count` | `carried` |  |
| `TrackState.velocity` | `carried` |  |
| `TrackerConfig.adaptive_measurement_noise` | `carried` |  |
| `TrackerConfig.chi_squared_gate_threshold` | `carried` |  |
| `TrackerConfig.cluster_distance_threshold_m` | `carried` |  |
| `TrackerConfig.confirmation_m` | `carried` |  |
| `TrackerConfig.confirmation_n` | `carried` |  |
| `TrackerConfig.ct_process_accel_std` | `carried` |  |
| `TrackerConfig.ct_turn_rate_std` | `carried` |  |
| `TrackerConfig.cv_process_accel_std` | `carried` |  |
| `TrackerConfig.data_association_mode` | `carried` |  |
| `TrackerConfig.innovation_max_scale` | `carried` |  |
| `TrackerConfig.innovation_scale_factor` | `carried` |  |
| `TrackerConfig.innovation_window` | `carried` |  |
| `TrackerConfig.max_bearing_std_rad` | `carried` |  |
| `TrackerConfig.max_coast_frames` | `carried` |  |
| `TrackerConfig.max_coast_seconds` | `carried` |  |
| `TrackerConfig.max_stale_steps` | `carried` |  |
| `TrackerConfig.max_timestamp_skew_s` | `carried` |  |
| `TrackerConfig.min_confidence` | `carried` |  |
| `TrackerConfig.min_intersection_angle_deg` | `carried` |  |
| `TrackerConfig.min_observations` | `carried` |  |
| `TrackerConfig.min_quality_score` | `carried` |  |
| `TrackerConfig.near_parallel_rejection_angle_deg` | `carried` |  |
| `TrackerConfig.retain_history` | `carried` |  |
| `TruthState.position` | `carried` |  |
| `TruthState.target_id` | `carried` |  |
| `TruthState.timestamp_s` | `carried` |  |
| `TruthState.velocity` | `carried` |  |
| `Vector3.x_m` | `carried` |  |
| `Vector3.y_m` | `carried` |  |
| `Vector3.z_m` | `carried` |  |
| `WatchFramesV2Request.include_truth` | `carried` |  |
| `WatchFramesV2Request.max_rate_hz` | `carried` |  |
| `WatchFramesV2Request.resume_after_sequence` | `carried` |  |
| `WatchFramesV2Response.dropped_since_last` | `carried` |  |
| `WatchFramesV2Response.frame` | `carried` |  |
| `WatchFramesV2Response.sequence` | `carried` |  |
| `WatchFramesV2Response.session_id` | `carried` |  |
| `WatchFramesV2Response.source_timestamp_s` | `consumer-gap` | Populated in Rust; the Python adapter does not read it. |
