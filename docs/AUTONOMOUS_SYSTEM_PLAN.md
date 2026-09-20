# ArgusNet autonomous system plan

## Status and scope

This document defines the integrated product plan for the next ArgusNet system:
an autonomous, three-dimensional site-intelligence platform that is qualified in
simulation and hardware-in-the-loop before promotion to physical aircraft.

It extends, rather than renumbers, the existing plans:

- migration packages `P2`–`P6` remain defined by
  [the delivery plan](migration/DELIVERY_PLAN.md);
- product capabilities remain described by [the product roadmap](ROADMAP.md);
- vertical features `F1`–`F7` remain defined by
  [capability delivery](CAPABILITY_DELIVERY.md); and
- presentation behavior remains governed by
  [the viewer specification](VIEWER_UI.md).

The `S0`–`S8` releases below describe system-level outcomes. They are dependency
gates, not calendar estimates and not replacements for migration package IDs.

## 1. Product definition

ArgusNet plans, validates, executes, records and explains autonomous civilian
mapping and inspection missions. Normal operation does not require a person to
manually fly an aircraft. An authorized person defines intent and constraints,
approves deployment where required, supervises health and outcomes, and may
request hold, return, divert, land or abort. Immediate stabilization and
contingency behavior remain onboard and do not depend on the viewer, network or
cloud.

The system must produce a defensible spatial record answering:

1. what mission was authorized and what constraints applied;
2. where each platform estimated itself to be and with what integrity;
3. what was observed, reconstructed, inferred and still unknown;
4. which task, route, trajectory and safety decisions were made and why;
5. which faults occurred and whether the required contingency completed;
6. what evidence supports each inspection or change claim; and
7. which data, configuration, calibration, firmware and model versions produced
   the result.

Simulation is part of the product, not merely a developer demo. It is the first
execution environment for mission logic, flight software, hardware faults,
navigation failures, RF loss and autonomous recovery.

## 2. Non-negotiable invariants

1. **No routine manual flight dependency.** Manual control may exist for
   controlled testing or an operation-specific emergency requirement, but it is
   not the normal controller or the primary safety mitigation.
2. **Onboard safety survives disconnection.** Loss of the viewer, ground station,
   network or cloud cannot remove stabilization, containment, energy protection
   or the configured safe-landing behavior.
3. **Plan from belief, not simulation truth.** Truth may generate synthetic
   observations, enforce the physical collision floor and score evaluation. It
   is not available to production planning, localization or safety decisions.
4. **One mutable authority per concern.** Flight control, mission execution,
   localization, belief mapping, fusion, safety and presentation have declared
   owners. The viewer is never a second controller or map authority.
5. **Three-dimensional semantics end to end.** Poses, trajectories, obstacles,
   airspace, sensor volumes, uncertainty and reconstructions retain altitude,
   orientation, reference frame and time. A 2D projection cannot silently become
   the analytic source.
6. **Unknown is not safe.** Missing, stale, uncertain and observed-free are
   separate states in storage, planning, safety and presentation.
7. **Every consequential action is replayable.** Inputs, revisions, proposals,
   decisions, acknowledgements, actuator intents, faults and outcomes are
   versioned and attributable.
8. **Promotion is evidence based.** A capability moves from deterministic tests
   to SIL, HIL and aircraft only after its predeclared gate passes. A successful
   demonstration is not qualification.

## 3. Client products and external outputs

Clients receive approved outcomes, not an undifferentiated telemetry dump.

| Product | Required content | Principal presentation / export |
| --- | --- | --- |
| Site package | Site boundary, coordinate reference, altitude datum, prior world, assets, privacy/exclusion zones and permissions | Site catalog and 3D Plan workspace; versioned manifest |
| Mission package | Objectives, evidence requirements, platforms, operating envelope, contingencies and authorization | Plan validation and signed mission manifest |
| Live operating picture | Platform pose/integrity, task, trajectory, health, energy, link, safety and active contingency | Operate workspace and capability-negotiated stream |
| Belief-world revision | Surface, structures, free/occupied/unknown volume, semantics, age, uncertainty and provenance | 3D world, top-down/section projections and spatial chunks |
| Evidence package | Original captures, sensor/platform pose, calibration, localization integrity, quality and asset association | Review workspace and immutable evidence manifest |
| Reconstruction product | Point cloud, mesh, orthomosaic/elevation where applicable, alignment and quality | Synchronized 3D Review; COG/GeoTIFF, LAS/LAZ, glTF/3D Tiles |
| Change set | Added/removed/changed geometry or condition, confidence, affected region and supporting evidence | Prior/current 3D comparison and GeoJSON/GeoPackage findings |
| Mission assurance report | Completion, gaps, faults, safety interventions, localization integrity, energy and landing outcome | Human-readable report plus machine-readable JSON |
| Audit bundle | Input journal, events, decisions, approvals, configuration and component versions | Replay/Audit view and signed content manifest |

Every externally meaningful artifact declares site and mission identity,
coordinate frame, time basis, world/belief revision, provenance, content hash,
runtime algorithm version, calibration version and applicable ML model version.

## 4. Runtime and authority architecture

### 4.1 Aircraft boundary

The aircraft-side stack is layered:

1. The flight controller owns attitude/rate stabilization, actuators and
   platform-native emergency modes.
2. The onboard autonomy executive owns mission state, trajectory following,
   local replanning and contingency selection.
3. An independent safety supervisor validates commands and may replace them with
   hold, return, divert or land behavior.
4. The health monitor evaluates compute, sensors, propulsion, power, storage,
   thermal state and link quality.
5. The recorder preserves sensor, estimate, command, actuator, fault and version
   history needed for diagnosis and replay.

ArgusNet should initially integrate a proven autopilot through a narrow,
versioned adapter and SITL/HIL interface. Reimplementing low-level stabilization
is a separate program requiring its own control-law and airworthiness evidence.

### 4.2 Ground and edge boundary

The ground application owns mission authoring, fleet supervision, durable
spatial memory, reconstruction review, reporting and non-time-critical compute.
It may submit authenticated mission intents or mode requests, but it does not
stream continuous actuator values. Immediate flight safety is not conditional on
ground connectivity.

Telemetry and command paths are separate capabilities with separate identity,
authorization, sequencing, audit and failure policies. Adding a read stream never
implicitly enables actuation.

## 5. Three-dimensional world contract

The current terrain raster and surface tile model remain useful but are not a
complete spatial representation. The target revision contains:

- surface tiles: elevation, normals, surface class and uncertainty;
- structure chunks: meshes, point clouds and stable asset geometry;
- occupancy chunks: sparse free/occupied/unknown voxels or equivalent octree;
- distance/clearance fields used by planning and safety;
- semantic hypotheses with observation provenance and uncertainty;
- dynamic objects with time-indexed state and prediction envelopes;
- three-dimensional mission, privacy, geofence and exclusion volumes;
- localization landmarks, keyframes, constraints and frame transforms; and
- evidence anchors linking captures to spatial geometry.

Each chunk carries a 3D bounding box, frame, resolution/LOD, revision, timestamp,
content hash and provenance. The runtime publishes immutable complete revisions
plus base-revision-checked patches. The renderer may use simplified geometry but
must display the same analytic revision used by planning and safety.

Trajectories are time-parameterized 3D objects containing position, orientation,
velocity, acceleration, vehicle envelope, clearance, localization requirement,
energy state and validity interval. Proposed, safety-modified, accepted,
executing, completed and contingency trajectories remain distinct.

## 6. Core capability workstreams

### Simulation and flight-software qualification

- deterministic analytic simulation for unit and regression tests;
- higher-fidelity 6-DoF vehicle and actuator models where required;
- actual autopilot and autonomy executables through SITL;
- sensor timing, calibration, vibration, bias, saturation and dropout;
- wind, gusts, precipitation, visibility and ground effect;
- propulsion, power, compute, thermal, storage and payload failures;
- RF latency, jitter, interference, asymmetric link and complete loss; and
- seeded scenario manifests, fault schedules and comparable outcome reports.

Simulation models are calibrated with measured bench/flight data and evaluated
against held-out data. Tuning a model against the same flight used for acceptance
does not demonstrate simulation credibility.

### Localization and GNSS-denied operation

The localization service fuses appropriate combinations of IMU, VIO, barometer,
terrain-relative altitude, LiDAR/depth odometry, prior-map landmarks, loop
closures, surveyed anchors and optional peer ranging.

Its public result includes pose, frame, covariance/containment, integrity state,
drift estimate, anchor source and age, matched evidence, failure reason and
recommended recovery. Required states are `initializing`, `localized`,
`degraded`, `lost` and `recovered`.

Mission policy responds to integrity rather than a manufactured confidence:

- initializing: collect safe observations; block precision work;
- degraded: reduce speed, enlarge clearance and favor known/landmark-rich space;
- lost: hold, relocalize, return by an independently valid method or land;
- recovered: invalidate and revalidate the remaining trajectory before resuming.

Qualification includes darkness, low texture, repeated structures, changed maps,
occlusion, dust, vibration, clock error, kidnapped-platform relocalization,
incorrect peer data, GNSS outage and spoof-like inconsistency.

### Mapping, reconstruction and spatial memory

- incremental live occupancy and surface updates;
- progressive sparse then dense reconstruction;
- uncertainty-aware map merge and conflict handling;
- observation aging and explicit stale/unknown state;
- immutable site/world/evidence revisions across missions;
- prior/current alignment and reviewed change claims; and
- bounded edge caches with resumable synchronization to durable storage.

### Mission autonomy, planning and safety

- task selection from mission value, information gain and evidence requirements;
- 3D route, viewpoint, speed, altitude and gimbal proposals;
- kinematic, energy, communication and localization feasibility;
- time-aware multi-platform separation;
- onboard-safe hold, return, divert and landing alternatives; and
- structured decision reasons linking inputs, revisions, constraints and result.

### Data governance and security

- immutable raw observations and separately versioned derived products;
- device identity, signed envelopes, sequence checks and replay resistance;
- authenticated and authorized command capabilities;
- encryption in transit and at rest plus credential rotation/revocation;
- site/mission/tenant access, privacy zones and redaction;
- retention, deletion, export and legal-hold policies;
- calibration, firmware, runtime and model registries; and
- offline edge operation with conflict-aware, auditable synchronization.

### Machine learning

ML first supports reviewable perception: semantics, anomaly candidates, place
recognition, evidence quality, reconstruction assistance and change candidates.
It is not initially the sole authority for stabilization, containment or flight
safety.

Promotion is offline evaluation, recorded replay, simulation, shadow mode,
advisory use and then explicitly bounded operational use. Every result records
model/data version, calibrated confidence, out-of-distribution status and source
evidence. Online learning does not modify an executing aircraft's operational
model.

## 7. Viewer and interaction plan

The viewer is a 3D operational instrument with derived 2D projections. Its
persistent global workspaces are:

- **Operate:** autonomous execution, health, trajectories, contingencies and
  live belief world;
- **Plan:** site volumes, constraints, tasks, viewpoints and validated proposals;
- **Review:** replay, evidence, reconstruction, change and mission assurance.

Within a workspace, camera projections are named `Perspective`, `Top-down` and
`Section`. Semantic layer presets are `Operational`, `Reconstruction` and
`Safety`. These are deliberately different labels and scopes: a workspace is not
duplicated as a camera button or layer button.

The central 3D world shows terrain, structures, occupancy, aircraft pose,
time-indexed trajectories, uncertainty volumes, sensor frusta, airspace,
reconstruction and evidence anchors. Side rails contain indexed mission state and
the selected object's explanation. The timeline is the single control for source
time; play/pause is not duplicated in the canvas toolbar.

Viewer delivery slices:

1. replay fixture, design tokens and typed selection;
2. persistent 3D ECS, camera modes, depth-correct picking and world revision;
3. surface/volume belief, trajectories, sensors, zones and uncertainty;
4. Operate shell plus durable event explanations;
5. Plan authoring with 3D volumes and vertical/time validation;
6. Review evidence, reconstruction and prior/current comparison;
7. simulation worker and acknowledged commands;
8. read-only HIL/live sources; and
9. separately approved autonomous command capability.

### Feature-level viewer implementation handoff

[VIEWER_FEATURE_INTEGRATION.md](VIEWER_FEATURE_INTEGRATION.md) translates this
system plan into concrete Operate/Plan/Review workflows, F1–F7 and broader
capability surfaces, nine client-product previews, data-producer contracts and
SIL/HIL acceptance fixtures. The [viewer specification](VIEWER_UI.md) incorporates
Perspective / Top-down / Section, a single source-time timeline, 3D spatial chunks,
localization integrity and onboard-contingency presentation. These are target
requirements; source support and control authority remain capability-declared.

## 8. System releases and gates

### S0 — Complete the coherent Rust runtime

Finish P4.3 world construction, P4.6 belief publication, P4.8 tick composition,
P4.9 CLI parity and the P5 worker boundary. Exit with a nonempty representative
mission owned by Rust, direct fusion, immutable snapshots, complete checkpoints,
durable events and no Python callback inside a tick.

### S1 — Canonical contracts and client output skeleton

Land site, mission, platform/sensor, localization, spatial revision, trajectory,
safety, evidence, reconstruction, finding, change and report contracts. Produce
one replay-derived client package with explicit unavailable fields.

### S2 — Three-dimensional belief world and client review

Publish surface plus volumetric spatial chunks, progressive reconstruction,
evidence anchors and uncertainty. Deliver synchronized perspective, top-down and
section views plus initial exports. Exit when planning, safety and presentation
resolve the same spatial revision.

### S3 — Autonomous flight software in SIL

Integrate the selected autopilot and vehicle adapter. Execute actual flight and
mission software against nominal and injected-fault simulations. Exit when the
required scenario matrix produces bounded outcomes without manual control.

### S4 — HIL and operator product

Run the production flight computer, autopilot, radios and payload interfaces in
HIL. Complete Operate/Plan/Review, black-box recording, security, data governance
and mission assurance reporting. Exit when hardware faults, link loss and ground
application loss invoke the declared onboard contingencies.

### S5 — Contained autonomous aircraft

Progress through propulsion stand, restrained, netted/tethered where appropriate,
and bounded low-energy autonomous takeoff/hover/land tests. Humans supervise and
may abort; they do not normally fly. Exit through repeated agreement with HIL
predictions and complete post-run evidence.

### S6 — Autonomous field mission and GNSS-denied envelope

Expand range, altitude, wind, payload and environment only inside qualified
envelopes. Qualify GNSS-denied initialization, drift containment, relocalization,
return and landing. Exit when localization integrity detects loss before the
declared containment bound is exceeded.

### S7 — Coordinated fleet and advanced operations

Add distributed identity/time, peer observations, trajectory reservations,
network partitions, coordinator loss and per-aircraft independent contingencies.
Regulatory/operational approval is a separate gate and cannot be inferred from
technical readiness.

### S8 — Intelligence and scale

Promote qualified ML capabilities, multi-site spatial memory, distributed edge
synchronization, plugin SDKs and collaborative review without weakening the
authority, provenance or safety boundaries established earlier.

## 9. Required verification ladder

Every operational capability advances through:

1. unit, property and fuzz tests;
2. deterministic component and cross-platform fixtures;
3. scenario replay and truth-isolation tests;
4. SIL nominal, boundary, adversarial and Monte Carlo campaigns;
5. HIL with production timing, buses, radios and compute limits;
6. contained physical tests;
7. bounded field-envelope tests; and
8. operational release with monitoring, rollback and incident review.

The fault matrix covers at least software deadline/crash, propulsion degradation,
actuator faults, battery sag/thermal state, sensor dropout/freeze/bias, GNSS
loss/inconsistency, VIO or map-match divergence, clock error, CPU overload,
watchdog reset, storage exhaustion, RF delay/loss/reconnect, stale or replayed
commands, incorrect world revision, severe weather inputs and multi-platform
conflicts.

Each case predeclares detection deadline, allowed transient, containment bound,
fallback, expected terminal state and required evidence. “No crash” alone is not
an acceptance criterion.

### Current phase integration

At `ba163ac`, P4.3g procedural landscape builders and P4.6d belief publication
are merged, as are P4.8a–c mission state, TickWorld and stage bodies. S0 remains
open until complete mission composition, checkpoint/replay and CLI/worker gates
pass. The [delivery plan](migration/DELIVERY_PLAN.md) tracks this distinction.

## 10. Immediate execution queue

1. Integrate the merged P4.3g Rust landscape builders into the complete scenario
   and viewer boot path.
2. Consume merged P4.6d immutable belief revisions and dirty patches, then define
   the additive volumetric spatial-chunk contract.
3. Complete P4.8 mission/controller composition around the merged mission state,
   TickWorld and stage bodies; qualify decisions, full checkpoints and replay.
4. Define S1 client-output manifests before expanding Plan/Review UI so the UI
   cannot invent unavailable backend state.
5. Implement the viewer against replay fixtures using the interaction taxonomy
   in section 7, then bind the runtime worker.
6. Select the first supported autopilot/vehicle class and freeze its adapter,
   timing, frame and failure contracts.
7. Build the SIL fault-injection harness and mission assurance report before any
   aircraft integration.
8. Add production hardware only through HIL, then promote a frozen release bundle
   to contained autonomous testing.

Work should continue as small vertical slices: one contract, one fault or client
outcome, one operator explanation and one measurable acceptance scenario.
