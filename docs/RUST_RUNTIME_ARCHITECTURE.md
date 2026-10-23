# Rust runtime target: independently scheduled subsystems

**Design updated:** 2026-09-23. **Implementation reviewed:** 2026-10-23.
**Status:** agreed target, partially implemented. Rust composition and the viewer
worker ship; opt-in lane rates, tick-aligned sensor rates and delayed delivery
are implemented. Full independent scheduling and pending-job history remain open.
Implementation evidence is tracked in [state ownership](STATE_OWNERSHIP.md) and
[known gaps](KNOWN_GAPS.md). This design extends ADR-005/007 through
[ADR-009](adr/009-independently-scheduled-runtime.md). Package IDs and acceptance
remain in the [delivery plan](migration/DELIVERY_PLAN.md).

## 1. What the finished system is

Rust owns starting, running, controlling, viewing and recording a simulation.
Python remains optional for offline experiments, analysis, GIS and exports; it is
not called inside the production runtime or required to boot a synthetic world.
Existing domain crates are reused. This is a new execution model in the existing
repository, not a second implementation of sensing, fusion or planning.

The key change is independent scheduling: control runs on its deadline, sensing
on each sensor's schedule, estimation on arrivals and prediction deadlines,
mapping on accumulated evidence, and planning on relevant changes or a deadline.
Rendering follows its own frame clock. These are coordinated through time and
revisions, not forced through one global all-subsystems tick.

| Deployment | Owner and boundary |
| --- | --- |
| Embedded desktop simulation | Bevy owns presentation; a dedicated Rust worker owns the runtime. Closing this session stops its worker after recording flush or an explicit incomplete-recording result. |
| Headless simulation | A Rust CLI drives the same runtime library, with no Bevy dependency in domain crates. |
| Remote session | A Rust service owns the runtime. The viewer attaches through capability-declared transport; disconnecting the viewer does not stop the service. |
| Future aircraft | An onboard autonomy runtime uses qualified sensor/autopilot adapters; stabilization and independent safety remain onboard. Ground/cloud availability is not a control-loop dependency. |

Only one host owns a given session. Embedded and remote are deployment choices,
not two writers to one world. Real-time hardware qualification is separate from
the simulation migration; Rust and async I/O alone provide no hard deadline guarantee.

## 2. Runtime modules and owned state

| Concern | Rust placement | Target responsibility |
| --- | --- | --- |
| Session and scheduler | `argusnet-sim` for simulation; shared contracts may be extracted when a real hardware consumer exists | Lifecycle, clock, schedules, command admission, commit boundaries, resource budgets |
| World and physical state | `argusnet-world` plus simulation-owned moving state | Immutable environment, integration, physical collision; separate from estimated poses |
| Sensor sources | `argusnet-sensing`, simulation generators and future adapters | Captures, calibration, timestamps, bias/noise state, delivery queues |
| Pose estimation | `argusnet-localization` and existing VIO components | Per-vehicle estimates, integrity, history and elapsed-time propagation |
| Object fusion | `argusnet-core` | Fused tracks, association, prediction and track lifecycle; direct calls internally |
| Belief and spatial products | `argusnet-mapping` | One writer per belief store; immutable revisions, dirty tiles, evidence provenance |
| Planning and mission | `argusnet-planning`, mission composition in `argusnet-sim` | Task ownership, affected-route invalidation, trajectory proposals, energy/availability decisions |
| Safety and control | `safety-engine` plus runtime controller state | Validate proposals, enforce profile limits and apply qualified fallback behavior |
| Persistence | `argusnet-replay` contracts and runtime recording/checkpoint components | Versioned artifacts, causal history, complete resume state; not a second world owner |
| Presentation and transport | `argusnet-viewer`, `argusnet-server`, `argusnet-proto` | Projection/selection, external adapters and capability negotiation; no duplicate domain math |

The session owns stateful subsystem instances. Replace transitional shared
mutable stage handles with owned state and narrow borrowed inputs as composition
lands. Workers receive immutable inputs and return proposals; they do not mutate
the live world. Production construction rejects missing required subsystems.

Truth is a capability of physics, synthetic sensing and independent evaluation.
Estimated vehicle poses, object tracks and belief are distinct state. Planning
never averages different vehicle positions into a supposed shared vehicle pose.
Shared localization exchanges qualified constraints with provenance/correlation
handling; this redesign does not itself implement multi-vehicle SLAM.

## 3. Time and scheduling

Keep `legacy_stages_v1` for port acceptance. Introduce `scheduled_v1` as an opt-in
execution model with its own behavior version; promote it only after P4.10f.
The eleven-stage order remains a compatibility contract, not the final scheduler.

Use checked integer nanoseconds for internal time. Periodic schedules specify an
integer period or a rational frequency. Compute occurrence deadlines from the
epoch and occurrence index, using checked wide intermediate arithmetic and one
documented rounding rule (ceiling to the next nanosecond). Do not repeatedly add
a rounded period, and do not require a camera rate to divide the control rate.
Reject rates whose distinct deadlines cannot be represented or whose resource
budget is unsupported. Floating seconds remain derived interchange values.

| Work | Trigger and data contract |
| --- | --- |
| Physics | Fixed integration boundaries. Off-boundary synthetic captures use the physical model's declared sampling/interpolation capability; never reuse an incorrectly timestamped pose. |
| Control/safety | Fixed deadlines with bounded essential work and current qualified estimates. Safety is not delayed until a slow planner finishes. |
| Sensor capture/delivery | Per-source acquisition schedule and modeled availability; acquisition and arrival are distinct. |
| Estimation/fusion | Available measurements plus prediction/expiry deadlines, including empty-observation intervals. |
| Mapping | Bounded evidence batches or maximum-age deadline; retain every required observation between updates. |
| Planning | Mission/availability/reserve changes, affected geometry, or periodic deadline; unchanged map regions do not invalidate unrelated routes. |
| Publication | Independent display cadence and important state changes; source timestamps accompany every component. |
| Meshing/export/optimization | Dirty input or explicit request on bounded background workers. |

At one timestamp, execute physical propagation to that time if required, boundary
commands, due captures, due deliveries, estimator updates, due mapping, mission
and planning decisions, control/safety, then publication. At reset, discard the
old epoch's remaining events. Commands apply only at declared safe boundaries;
Stop/Pause remain responsive between bounded work units. Control outputs affect
future physical intervals, never the already integrated interval.

Break equal-time ties by phase, stable source identity and sequence. A newly
created event cannot run in a phase already closed at that timestamp; defer it
to the next eligible boundary and record its effective time. Initialization
establishes physical and controller state before the first interval.

A synthetic sensor profile must name how it samples between integration steps:
exact model evaluation or declared interpolation for supported quantities. If
the physical model cannot supply that sample, reject the profile. Do not claim
accurate high-rate IMU output by interpolating a low-fidelity position-only model.

Each subsystem receives its actual elapsed simulated time, not a universal dt.
Prediction, battery accounting, timeouts and mission dwell are reviewed when
rates change. Accumulate physical energy over executed motion, with changes to
legacy accounting isolated as explicit algorithm changes.

## 4. Virtual time versus live time

Deterministic simulation advances a virtual clock through due events and physical
boundaries. Real-time pacing and accelerated/headless execution use the same
causal order. Never drop physics steps to meet wall-clock speed. Overload reduces
achieved speed and is visible. Rendering and disk timing cannot change outcomes.

Hardware operation uses monotonic host deadlines and timestamped external inputs.
Device adapters preserve raw clock domain, conversion/calibration identity and
clock uncertainty. UTC is metadata, not the integration clock. A device reboot
starts a new source epoch; it must not appear as time moving backward silently.

Critical control/estimator work has reserved execution capacity and bounded work.
I/O waits use async facilities; CPU-heavy work uses bounded workers. No task may
hold runtime state locks across I/O. Background tasks have cooperative
cancellation and bounded work units; a worker pool is not a real-time guarantee.

Deterministic causal jobs have recorded application boundaries. If a result is
not ready, wall time waits while commands stay serviceable; simulated time does
not advance. Live jobs instead have qualified deadlines and fallback policies.
Purely visual work may complete in any order if stale outputs are rejected.

## 5. Inputs, outputs and overload

A measurement envelope carries source/sequence/epoch, acquisition time,
availability and receipt times, frame identity, calibration identity, uncertainty,
clock quality and payload. Fusion uses acquisition time, not processing time.
Adapters declare supported lateness and ordering. Unsupported delayed profiles
fail before a run; late live data is rejected with a reason unless the estimator
supports a tested historical correction policy. No silent retimestamping.

Typed direct calls serve the local hot path. Serialization and gRPC occur at real
process boundaries. Wire additions update Rust and Python bindings together.

| Path | Required policy |
| --- | --- |
| Commands | Bounded admission, explicit Busy/rejection, deduplicated receipts and eventual applied/cancelled result; never silent loss |
| Presentation | Latest coherent revision replaces older display state; consumers can skip intermediate snapshots |
| Sensor evidence | Per-source bounded queues with explicit overflow/loss events; any coalescing must be justified by that sensor/estimator contract |
| Events/recording | Ordered evidence and decision journal separate from presentation; explicit durable cursor and incomplete-output state |
| Background jobs | Bounded queue; cancel superseded requests; reject old epochs and revalidate relevant revisions and trajectory safety |

For deterministic simulation, required-recording pressure suspends advancement
at a committed boundary. For live inputs, storage pressure cannot stop onboard
control; use the qualified priority/retention policy and record gaps. Safety and
command events must never be silently treated as lossy display telemetry.

A coherent snapshot is a consistent selection of committed component revisions,
not a claim that every component updated simultaneously. Include each component's
as-of time, age/status, provenance and dependency revisions. The viewer can show
an old map beside a fresh pose if the age is explicit; it cannot combine mismatched
epochs or coordinate frames. Only presentation converts f64 world coordinates
to local f32 render coordinates.

## 6. Lifecycle, persistence and operator behavior

Start validates capabilities/configuration and constructs a complete session.
Reset rebuilds all causal subsystem state from the resolved scenario. Pause
stops virtual advancement at a boundary; Step advances one configured control
interval and processes its intervening events, then pauses. Speed changes only
pacing. External aircraft time is never paused or rewound by viewer playback.

Checkpoint physical state, estimators, controllers, sensor bias/RNG state,
mission/safety/battery state, belief, event queues, next occurrence indices,
pending jobs/inputs, command history and journal position. Rebuildable caches may
be excluded only after equivalence is demonstrated. Store actual config/world
content digests, not revision numbers masquerading as content hashes.

Validate and construct restored state before replacing the active runtime.
Interactive restore/branching gets a fresh publication epoch; late jobs and
snapshots from the previous epoch are rejected. Historical replay viewing is
different from resuming execution. Older incomplete checkpoints cannot resume a
full mission by silently defaulting missing state.

An unexpected partial update faults the session and publishes no partial revision.
Reset/restore is required before further advancement; automatic rollback is not
claimed. Export/recording faults are separately visible. Shutdown cancels workers
and flushes recording within its configured budget or marks output incomplete.

The viewer uses Operate/Plan/Review and Perspective/Top-down/Section. It shows
source capability, estimated-state freshness, achieved speed, pending commands,
recording status and localization integrity. Diagnostics adds scheduler, queue
and deadline detail without placing implementation controls in routine missions.
See [viewer specification](VIEWER_UI.md) for the visual design.

## 6a. Model, world and persistence completeness

[Simulation fidelity](SIMULATION_FIDELITY.md) separates execution model, time
policy, physical/sensor model and visual quality. Paced simulation may slow down;
deadline-coupled SIL/HIL/live operation must instead detect and handle missed
external timing. Sensor render/capture is independent of the spectator camera.

[World generation](TERRAIN.md#world-generation-and-publication-contract-target)
records generator/config/asset identity, physical semantics, stable tile generation
and atomic publication. Imported visual assets are not automatically collision
or sensor geometry. A world revision and a content digest have different meanings.

[Track contracts](FUSION.md#track-identity-lifecycle-and-time-contract-target)
separate scenario targets, observations and estimates, with source-scoped stable
identity, elapsed-time expiry, explicit deletion and measurement age independent
of prediction/publication. Rich association or smoothing capabilities are not
implied by a faster scheduler.

[Backend recovery](migration/ARCHITECTURE_AND_PROTOCOLS.md#session-configuration-admission-and-recovery-target)
distinguishes immutable run configuration, volatile command receipts, committed
state and durable journal position. A crash cannot turn an incomplete artifact
into a complete run. [Viewer consistency](VIEWER_UI.md#112a-display-time-frame-and-resource-consistency)
requires versioned transforms, bounded staged uploads and selection against the
displayed revision. These close design contracts; they do not add another port.

## 7. Delivery and evidence

Finish production ownership/reset/full checkpoint work in P4.8, baseline in P4.9,
and ship P5.0 basic viewer controls early. P4.10a–f then introduce independent
deadlines, measurement envelopes, incremental work, bounded jobs and complete
scheduler persistence before promotion. P6 removes old execution paths only
after consumer and installed-package acceptance.

Balanced simulation is the first workload. Advanced flight dynamics, historical
measurement smoothing, full cooperative SLAM and hardware actuation are separate
capabilities, not implied by a scheduler implementation. Hardware progression is
defined in [hardware integration](HARDWARE_INTEGRATION.md).

The [performance standard](PERFORMANCE_AND_BENCHMARKING.md) records separate
latency, work, memory, accuracy and usability outcomes. The
[verification plan](migration/VERIFICATION_AND_ROLLOUT.md) requires cross-rate,
cross-worker, restore, stale-data, overload, truth-isolation and cross-platform
evidence. No speedup or hardware-readiness claim follows from documentation alone.
