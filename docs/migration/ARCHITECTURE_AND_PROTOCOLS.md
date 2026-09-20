# Runtime Architecture and Communication Protocols

This is the implementation contract for the remaining Rust migration, adopted
with ADR-007 and ADR-008. **Planned interfaces below are not implemented APIs.**
The implemented baseline is recorded in [the migration index](../RUST_RUNTIME_MIGRATION.md).
Requirements use MUST for an invariant, SHOULD for a default with a documented
exception, and MAY for an optional capability. Changes to a MUST require an ADR
or an explicit compatibility amendment in the implementing PR.

## 1. Authority and dependency boundaries

| Concern | Owning crate / package | Public boundary | Forbidden dependency or behavior |
| --- | --- | --- | --- |
| Runtime values, IDs, units, serializable enum vocabulary | `argusnet-types` | Validated values and records | Bevy, Tokio, protobuf-generated types, Python |
| Terrain/obstacle kernels, procedural generation, spatial queries | `argusnet-world` | Immutable world views and builders | Reading viewer geometry or mutating mission state |
| Coverage, reconstructed map, belief revisions | `argusnet-mapping` | `BeliefSnapshot`, coverage patches and query views | Undeclared truth samples |
| Pose estimation, GridLocalizer, VIO, GNSS and relocalization | planned `argusnet-localization` | Pose estimate plus status/covariance | Mutating mapping or silently declaring timeout convergence |
| Synthetic sensors | planned `argusnet-sensing` | Versioned observation source and batch | Writing fusion or mission state |
| Fused object states | `argusnet-core` | Direct update/reset/snapshot | Transport ownership or Python fusion math |
| Route proposals, frontier/POI/battery/deconfliction | planned `argusnet-planning` | Intent to route/trajectory proposal | Applying motion or querying a simulator truth handle |
| Limits, validation, escalation and accepted clamp decisions | `safety-engine` | Typed decision with input revisions | Mutating unrelated planner/controller state |
| Tick ordering, scenario instances and mission state | planned `argusnet-sim` | `Simulation`, commands and immutable snapshots | UI, file watching, Python invocation, network timing in the tick |
| Replay DTOs, schema validation and artifact conversion | planned `argusnet-replay` | Versioned documents and readers/writers | New simulation math or silent field loss |
| Transport adapters | `argusnet-proto`, `argusnet-server` | Versioned protobuf service | A second simulation or fusion engine |
| Operator presentation | `argusnet-viewer` | Read snapshots; submit commands | Owning authoritative mission/world state |
| Transitional Python calls | `argusnet-native` | C ABI with owned handles/buffers | A fallback evaluator or subprocess per query |
| Offline evaluation/GIS/reporting | Python | Replay/config/artifact inputs | Required imports on the interactive Rust boot path |

`argusnet-localization` separates estimator state from mapping ownership; the
existing footprint kernel remains in `argusnet-mapping` and is reused. It is not
a second implementation. `argusnet-replay` isolates compatibility DTOs so core
simulation does not depend on the viewer. Avoid a general-purpose event bus or
plugin framework until an actual consumer requires one.

Dependency direction is consumer -> owner. `argusnet-sim` composes the runtime
crates; CLI/server/viewer/native depend on it, never the reverse. World geometry
can evaluate either truth or belief data, but that shared algorithm does not
make the two datasets interchangeable. Domain crates MUST build without Bevy;
headless simulation MUST build without the viewer feature set. Add a dependency
check to CI before the first `argusnet-sim` release.

## 2. Types, coordinates and compatibility DTOs

Runtime spatial/time calculations use f64. The renderer converts to f32 in one
adapter, with a scene-local render origin for large coordinates. Moving that
origin MUST NOT change runtime positions or exported coordinates. Runtime axes
remain local ENU: X east, Y north, Z up; units are metres, seconds and radians.

Typed IDs distinguish platform, truth, track, zone, mission, run and command;
text encodings stay compatible with existing replay/proto IDs. Types MUST NOT
use an entity's label as identity. Define typed tick/run/revision values instead
of passing unrelated integers interchangeably. `Bounds2d` remains horizontal;
a grid adds resolution, dimensions, axis order, origin and coverage validity.
Do not infer a grid's dimensions or cell-center convention from a mesh.

Before unifying a DTO, inventory every field in Python, core, proto and viewer:
name, meaning, units, absent/null/default behavior, writer, consumers and loss
risk. In particular preserve sensor configuration, battery fraction, track
covariance, IMM probability, contributing nodes, health, mapping/pose/evidence
payloads and safety provenance. Do not reduce a rich replay type to today's
smaller core type merely to eliminate a struct.

Matrices have explicit dimensions and row-major interchange encoding. Track
covariance requires 36 finite elements; pose covariance has its own named
shape. Validate once at ingress with field-specific errors. Optional values
remain optional; legacy zero-as-unset fields are normalized only in their
versioned adapter. Unknown fields can be tolerated where the schema permits;
unknown required capabilities or unsupported schema majors fail visibly.

New glTF exporters MUST emit spec-conforming Y-up assets. The coordinate mapping
from runtime ENU to glTF is `(x, y, z) -> (x, z, -y)`; the loader applies its
inverse exactly once. Add explicit asset-axis metadata for new packages; absent
metadata means the existing ArgusNet legacy Z-up convention. Test round trips
with asymmetric geometry, normals, winding, lights and bounds. Do not reinterpret
old packages globally or rotate replay coordinates.

## 3. Truth, prior, belief and display capabilities

- `TruthWorld` is available only to synthetic sensor generation, physical
  environment interaction and evaluation. It MUST NOT be passed to new planners.
- A `PriorWorld` is imported, attributed data. It initializes belief with source,
  uncertainty and validity metadata; import does not make it certain ground truth.
- Mapping alone creates `BeliefSnapshot { revision, as_of_tick, grid_layout,
  cells/tiles, provenance }`. Cells distinguish unknown from measured zero height
  or zero occupancy. Evidence source, observation time and uncertainty are stored.
- Planning takes a belief view and a pose estimate with status. Unknown space has
  an explicit mission-policy response; a failed query MUST NOT become zero height
  or free space. New physical-mode behavior defaults to hold/replan when required
  evidence is absent. Preserve historical simulation behavior under an explicitly
  named compatibility profile during migration.
- Safety validates proposals using the profile's declared knowledge view. The
  simulator's physical collision/terrain floor is a separate world interaction:
  it may prevent penetration of actual terrain, but its correction is recorded
  and MUST NOT feed hidden truth back into planner inputs.
- The operator view uses belief and observed tracks by default. Truth rendering
  is a separately labelled simulation/debug capability, stripped at the transport
  and snapshot boundary when not requested or authorized. Hiding a mesh is not
  sufficient isolation.

Tests hold sensor/pose/belief inputs fixed, perturb hidden truth, and require
identical planning/knowledge-based safety outputs. Physical collision outcomes
and synthetic observations can legitimately change when their truth inputs do.
Timeout is a distinct localization status. The existing forced-confidence path
is ported as legacy behavior first, then changed in a separately reviewed
behavioral step; new inspection gates MUST NOT mistake it for observed convergence.

## 4. Deterministic tick transaction

One worker owns a `Simulation` instance and all mutable runtime subsystems.
Public conceptual entry points are `new(config, world, sources)`,
`apply_command(envelope)`, `step() -> Result<Arc<RuntimeSnapshot>, TickError>`,
`checkpoint()` and `restore(checkpoint)`. Exact Rust signatures are finalized in
P4.1; this document fixes their semantics. No async await or wall-clock sampling
belongs inside `step`. Simulation time is derived from integer tick and a fixed,
validated timestep; timestamps at interchange boundaries are derived seconds.

The first Rust port MUST preserve the Python causal order, including decisions
that take effect on the next tick. It MUST NOT silently "improve" that order:

| Order | Stage and state writes | Compatibility constraint |
| --- | --- | --- |
| 0 | Apply validated commands scheduled for this boundary | Log accepted/rejected command and effective tick |
| 1 | Sample nominal controllers and account battery using the prior positions | Preserve existing battery accounting before overrides; any physical-accounting correction is a separate behavior change |
| 2 | Apply frontier, then POI, then egress overrides | Preserve precedence and mission-phase applicability |
| 3 | Cooperative deconfliction | Stable platform ordering and tie-breaking |
| 4 | Opt-in hard safety monitor/clamp | Runs after deconfliction; no accepted motion below the applicable terrain floor |
| 5 | Generate truth/synthetic IMU features, update VIO and select observation pose | Preserve the current pose substitution and source provenance; do not silently couple VIO to different trajectories |
| 6 | Generate the observation batch | Preserve rejection stages/reasons and attempted/accepted counters |
| 7 | Update fusion directly in Rust | Same adapter-normalized inputs as legacy gRPC |
| 8 | Update coverage/belief and localization | Deduplicate footprint work while retaining weighting and masking semantics |
| 9 | Evaluate mission/frontier/POI/egress transitions and events | Record which future tick controller changes take effect |
| 10 | Freeze snapshot, metrics, event journal position and checkpoint eligibility | Consumers see one committed revision set |

Create a machine-readable stage trace against the Python reference before
extracting the orchestrator. Audit branch-dependent ordering in scan, inspection,
VIO and egress paths; the table is a migration constraint, not a claim that every
Python nested loop has already been extracted.

Internal stages mutate owned state without cloning every platform list. External
readers see only the last committed snapshot. Invalid commands are rejected
before mutation. On an unexpected mid-tick failure, publish no partial snapshot,
mark the run faulted and require reset or restore from the last complete
checkpoint. Do not continue a half-updated run or promise automatic rollback
without retaining sufficient state. An exporter failure follows section 6 and
does not masquerade as a physics failure.

## 5. In-process communication and operator commands

Use a dedicated simulation worker for interactive mode, with a typed bounded
command channel and immutable snapshots. A future Bevy `SimulationSchedule`
represents this boundary/driver; it does not put expensive runtime work on the
render thread. CLI batch mode invokes the same step method synchronously.

A command envelope contains `protocol_version`, `command_id`, `run_id`,
`expected_epoch`, optional `expected_revision`, command payload and requested
effective tick. Initial commands: load/start, pause/resume, single-step, set
speed, reset, seek, select mission parameters supported by the current engine,
and cancel pending preparation. UI camera/selection changes remain local.

Commands receive an acknowledgement with ID, accepted/rejected status, effective
tick, resulting epoch/revision and a typed reason. Queue-full returns Busy to the
UI; commands MUST NOT be silently dropped. Invalid mode, stale run/epoch,
unsupported command and unavailable rewind are explicit outcomes. Duplicate IDs
return the original acknowledgement within the current run's bounded dedup
window; expired retry IDs require resynchronization, not blind replay.

Initial configurable budgets: 256 queued commands, 4,096 dedup entries, and two
adjacent published frame snapshots. They are resource limits, not performance
claims; benchmark before increasing them. Pause/stop is checked between ticks
and world-build chunks. Long builders MUST support cancellation; otherwise the
UI displays that cancellation is pending and remains responsive. Never interrupt
a mutating tick halfway and call it a valid pause.

Snapshots use `Arc` and copy-on-write/deduplicated map tiles where appropriate.
Rendering may replace an unconsumed frame with a newer one. Durable mission and
safety events MUST NOT rely on that lossy path: use a separate ordered journal
and an event cursor. Expose dropped presentation frames, queue occupancy,
simulation lag and exporter backlog to diagnostics.

Playback speed scales desired wall-clock pacing, never physics dt. Bound catch-up
work; if the worker is behind, display slower-than-real-time state instead of
dropping simulation ticks. Render interpolation uses the two adjacent committed
snapshots, source timestamps and a clamped blend. Never interpolate across epochs,
teleports, missing IDs, discrete mission transitions or incompatible world
revisions. Selection follows typed IDs; removed IDs clear selection deliberately.

### Command lifecycle clarification for the next implementation

Distinguish admission (`Accepted`) from execution (`Applied`) and terminal
`Rejected`/`Cancelled` outcomes. A retry flag accompanies the original result;
it must not erase the original refusal reason. Scope identity by run/client/ID,
bind it to a payload fingerprint and define the bounded retry horizon. Preserve
existing wire fields; introduce compatible extensions with regeneration/tests.

Validate payloads, finite positive speed and requested tick policy at admission.
Revalidate applicable epoch/revision preconditions when scheduled work executes.
A reset is an epoch barrier: cancel remaining old-epoch work, including commands
already drained into a local batch. Drain eligible recovery controls even while
faulted/completed without running domain stages. SingleStep advances exactly one
tick and returns to paused. Report actual effective tick/revision, not merely
the originally requested value. UI submission or queue acceptance is not motion.

## 6. Snapshot, checkpoint and replay are different contracts

| Artifact | Intended contents | Persistence/compatibility |
| --- | --- | --- |
| `RuntimeSnapshot` | Public platform/track/pose/mission state, world/belief revision refs, metrics and event cursor | Immutable in-process presentation contract; truth capability filtered |
| `SimulationCheckpoint` | Full controller, fusion, mapping, estimator, safety, battery, mission, RNG and command/journal state | Restore contract tied to runtime algorithm/config/world versions |
| Replay document | Public frame/event sequence and run metadata | Long-lived versioned interchange for Python and viewer |

A replay is not a checkpoint. A fused-state frame alone omits IMM filters,
controller integrators, random-stream state, safety escalation and planner
caches that affect future decisions.

Checkpoint header: format version, runtime algorithm version, seed version,
run/epoch/tick, config hash, world content hash, belief revision and journal
cursor. Persist all causal state, including pending commands and any counters
that affect ordering. Exclude render entities/assets, wall-clock timers and
reconstructible caches; prove that rebuilding an excluded cache is behaviorally
neutral. Publish/checksum a checkpoint atomically after the tick commits.

Initial rewind policy: checkpoint every 20 ticks, configurable memory ceiling
256 MiB and disk spill opt-in. This is a starting budget to measure, not a promise
of a particular history duration. Evict oldest complete checkpoints under the
budget and report the available seek interval. A seek restores the nearest
checkpoint and deterministically replays logged inputs. A changed command after
seek creates a new epoch/branch; it does not overwrite the historical replay.
Live observation streams cannot rewind the external world; expose historical
playback as read-only unless all inputs support deterministic replay.

Recording must preserve every committed frame/event selected by its declared
recording policy. A bounded writer queue may throttle a recording/batch worker;
it MUST NOT silently lose frames. In interactive mode a full recorder queue may
pause recording with an explicit incomplete-artifact error according to the
selected policy. Render remains responsive. Large terrain blobs are referenced
by immutable content hash, not embedded into every frame. Write temporary files
and finalize manifests atomically; partial output is labelled and never reported
as a complete replay.

### Checkpoint completeness and presentation epoch

Concrete subsystem state belongs to the Simulation owner; stateful handlers
cannot sit outside reset/checkpoint/restore. Persist idempotency history needed
for resumed command decisions and declare whether pause/pacing are restored or
reset. Verify content hashes (not revision numbers), header/tick consistency,
versions, payload dimensions and queue bounds before any mutation. Set the clock
in O(1). A restore into an active session gets a fresh publication epoch to
invalidate interpolation; retained checkpoint provenance and a new branch's
identity are separate concepts.

## 7. External protobuf transport

Current service is `argusnet.v1.WorldModelService`, not the old conceptual
`TrackingService` name. Preserve existing RPCs and fields. `WatchFramesV2` carries
sequence, session ID, source time and dropped count; its current resume parameter
filters new events and is **not a durable historical replay guarantee**. The
server currently uses a 64-frame broadcast channel. Existing `MissionZone` and
`MissionZoneUpdate` messages do not mean zones are stored or streamed: current
`MissionStatus` returns no zones.

Add an explicitly negotiated world-aware stream (working name `WatchRuntimeV3`)
and snapshot/world fetch RPCs without repurposing V2. Do not pick protobuf field
numbers in prose; allocate unused numbers in the proto PR and run regeneration
and compatibility checks. The new service vocabulary is:

| Message | Required semantics |
| --- | --- |
| `RuntimeHello` | Protocol major/minor, feature flags, session/run/epoch, coordinate frame, supported schema versions and resource limits |
| `RuntimeSnapshotEnvelope` | Sequence, tick/time, epoch, world and belief revisions, frame payload, event cursor |
| `WorldManifest` | World ID/revision, CRS/axis metadata, tile layout, layer descriptors, content hashes and mission zones |
| `WorldChunk` | Manifest revision, chunk ID/offset/count, encoding, byte size and checksum |
| `BeliefTilePatch` | Tile key, base/new revision, observed mask, confidence/staleness data and provenance |
| `ResyncRequired` | Reason, latest snapshot/world revision and resume instructions |
| `MissionState` | Actual active zones/tasks/status at a stated revision |

Handshake sequence: client advertises supported capabilities; server responds
with selected protocol and manifest; client fetches/verifies required chunks;
client installs a complete revision atomically; frame subscription begins against
that snapshot cursor. Frames referring to unavailable required world data are
held within a bounded window or show an explicit world-unavailable state, never
quietly projected onto a different terrain revision.

Within `(session_id, epoch)`, sequence increases monotonically. A new session or
epoch clears interpolation, pending deltas and stale selections. Duplicates are
ignored; a gap or base-revision mismatch triggers resync. Resume includes session,
epoch and last consumed sequence; return retained data only if available,
otherwise `ResyncRequired`. Initial server retention: 256 frame envelopes or
32 MiB, whichever is reached first; it is a bounded convenience buffer, not a
recording system. Preserve drop telemetry under client rate limiting and lag.

World transfer is separate from high-rate frames. Start with raw little-endian
numeric chunks, at most 1 MiB per chunk, checksummed and capped at a configurable
256 MiB per assembled revision. Reject oversized declarations before allocation.
Compression is negotiated later; decoders must enforce expanded-size limits.
Retry chunks by content ID, cancel obsolete revision downloads, and limit
concurrent fetches. A manifest cannot make the viewer fetch arbitrary remote
asset URLs: use the configured authenticated service or explicitly selected
local imports. Keep old world revisions until readers release them, subject to
memory limits and explicit resync.

TLS and non-loopback guards remain mandatory for every client/server path.
Truth inclusion is capability-checked; it is not authorized solely by a boolean
request. Existing signed ingest identity/sequence checks remain intact. Initial
runtime command protocol is local; remote operator control is a separate feature
requiring authenticated command authorization and auditing, not implied by the
read stream. This migration does not authorize physical actuation.

## 8. Native ABI and Python transition

Current C ABI version is 1, with JSON control/owned integer handles and typed
batch buffers. Extend it additively; do not change an exported function's pointer
layout in place. Add a startup capability/version handshake before Python uses a
new symbol, with an actionable package/library mismatch error. Keep explicit
pointer extent/alignment/non-alias requirements and checked dimensions.

Handles have one owner, explicit release/clone rules and no pickle support;
serialize metadata or an explicitly versioned checkpoint instead. A rejected
call must not poison the registry. New stateful estimator handles must document
thread affinity/concurrency and snapshot lifetimes. Avoid holding a global
registry mutex over expensive independent queries: measure contention, then use
per-object ownership/locks with immutable read handles. Do not remove synchronization
without a concurrency test. Panic containment/error conversion must be defined for
every exported function; unwinding across the C ABI is forbidden.

Source installs build native artifacts; wheels ship them. Imports MUST NOT run
Cargo or discover conda paths. During coexistence Python calls native hot paths;
as runtime ownership moves, remove their Python evaluators rather than keeping
fallbacks. Keep `environment.py` re-exports until their callers migrate. Standalone
Python analysis must ultimately work without Rust unless it explicitly requests
native simulation or world generation.

## 9. Versioning, errors and observability

Separate protocol major/minor, replay schema version, checkpoint format version,
runtime algorithm version, world-generation version and seed-derivation version.
A single package version cannot express all these compatibility boundaries.
Every run records resolved config, content hashes, code/build identifier and
algorithm versions. Canonical hashes sort map keys/IDs and exclude wall-clock
metadata, transport sequence and renderer state.

Errors use stable codes plus human-readable details and context: invalid config,
unsupported capability/version, missing world coverage, stale revision, invalid
observation, queue full, world transfer failure, checkpoint mismatch, recorder
failure and tick failure. Do not encode failures as empty successful frames,
zero heights, zero zones or confidence 1.0. Compatibility exceptions are explicitly
named and tested at adapters.

Correlate logs with run ID, epoch, tick, command ID and revision; use structured
stage durations, counters and queue metrics. Trace parentage crosses native and
transport boundaries. Keep wall-clock profiling outside deterministic state.
Operational diagnostics distinguish time-to-first-world, steady render CPU/GPU,
simulation tick latency, native transfer cost and recording backlog. Logs are
rate-limited; they are not an unbounded per-frame buffer.

## 10. Domain and product integration refinements

The [capability plan](../CAPABILITY_DELIVERY.md) extends the runtime with typed
scorecards, scenario failures, explanations, comparisons, energy contingencies,
spatial memory and adaptive inspection. Feature state stays with its domain owner.
A single revisioned belief owner supplies consistent read views and dirty tiles;
persistent site history stores immutable revisions rather than mutating a second
copy. Task/route/trajectory/safety/execution records carry causality IDs and the
input revisions necessary to explain them in the viewer.

Physical-profile localization preserves timeout as a degraded outcome, validates
freshness/status, deduplicates platforms and accounts for shared evidence before
claiming fusion confidence. Legacy parity remains a separate named profile.
Evaluation-only truth matching and calibration are separate from operational
fusion. Truth capability filtering covers derived metrics as well as entity lists.
Canonical spatial interchange remains f64; covariance compatibility readers
normalize supported shapes and reject invalid numeric data explicitly.

[VIEWER_UI.md](../VIEWER_UI.md) is the consumer specification: Operate, Plan and
Review share source adapters, coherent snapshot application and durable events.
Plan submits validated intent; Review reads history/evidence; neither becomes
an alternate simulation or safety authority. Remote actuation remains outside
the read-stream upgrade.

## 11. Additive 3D presentation and qualification contracts

The [autonomous system plan](../AUTONOMOUS_SYSTEM_PLAN.md) extends world manifests
from surface tiles to structure, sparse occupancy, clearance and evidence chunks.
Each declares 3D bounds, frame, resolution/LOD, time, revision, hash and provenance.
A surface-only client/source explicitly lacks volume data; unrepresented volume
is never assumed free. Add capability negotiation and reader-first compatibility
before defaulting writers to those payloads. No new RPC or field number is
established by this prose.

Trajectory records retain orientation, time, envelope, clearance, energy,
localization requirement and validity. Localization integrity, health/fault/
contingency and SIL/HIL qualification records are producer-owned versioned inputs,
not UI-derived status. Evaluation/review and command capabilities remain distinct.
[The feature UI contract](../VIEWER_FEATURE_INTEGRATION.md) maps these inputs to
inspectors, three projections, client outputs and explicit unavailable states.
