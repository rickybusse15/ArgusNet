# ArgusNet Product and Capability Roadmap

## Purpose

This roadmap describes the long-term evolution of ArgusNet from a capable
simulation/replay foundation into a safety-bounded, human-supervised platform for
civilian aerial mapping, inspection, localization, survey, and persistent spatial
memory.

It is a sequencing and architecture document, not a claim that roadmap-only
capabilities are already implemented. When this document conflicts with running
code, the code and its tests describe current behavior. See
[architecture.md](architecture.md) and [KNOWN_GAPS.md](KNOWN_GAPS.md) for the
current implementation boundary.

## Runtime migration track

The [Rust runtime migration plan](RUST_RUNTIME_MIGRATION.md) tracks the separate
engineering migration, with detailed [work packages](migration/DELIVERY_PLAN.md),
[protocols](migration/ARCHITECTURE_AND_PROTOCOLS.md) and
[acceptance gates](migration/VERIFICATION_AND_ROLLOUT.md). Rust world/safety/fusion
authority is active, and localization, sensing, mapping and planning libraries
exist. Rust now includes mission state, TickWorld, populated snapshots and several
composed stage bodies, and `argusnet-rs` runs every canonical scenario with no
Python, qualified over 30 seeds plus a held-out set (A3.2). The Python simulation
remains the reference the Rust replays are checked against. Scene packages are
built in Rust; the viewer's Tools → Scenario simulation still runs Python. See the checkout-grounded
[ownership inventory](STATE_OWNERSHIP.md). Migration phase numbers do not
correspond to the product phases below.

The integrated [autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md) adds the
system-level path from the current migration through client outputs, a true 3D
belief world, flight-software SIL/HIL, GNSS-denied localization, contained
autonomous aircraft and coordinated operations. Its S0–S8 release gates do not
renumber the migration or feature packages in this document.

## Scheduled Rust runtime and operator delivery

The target is [independently scheduled Rust execution](RUST_RUNTIME_ARCHITECTURE.md),
not merely the existing Python loop translated to Rust. Complete and baseline
P4.8/P4.9; deliver basic P5.0 viewer controls early; add P4.10a–f independent rates,
measurement timing, incremental work, bounded jobs and causal history. Promote
scheduled execution only after named-workload accuracy and performance gates.
P6 retires superseded execution paths after installed-package/consumer acceptance.

The visual target remains the flat Operate/Plan/Review instrument interface,
Perspective/Top-down/Section, explicit uncertainty/unknown space and one source-time
timeline. Add source freshness, pending commands, achieved speed and recording
health; detailed deadlines/queues belong in Diagnostics. Visual polish cannot
substitute for coherent domain outputs or hide unavailable capabilities.

S0 establishes a usable Rust-owned runtime; scheduled-profile qualification is
tracked independently in P4.10. S3/S4 require actual flight-software/hardware
execution and recorded timing evidence under [hardware integration](HARDWARE_INTEGRATION.md).
A passed desktop benchmark is not an onboard readiness gate.

Architecture completion also includes reproducible world manifests, explicit
track identity/expiry, versioned frame corrections, bounded viewer uploads and
verified journal recovery. [Simulation fidelity](SIMULATION_FIDELITY.md) separates
model accuracy, visual quality and real-time pacing. These refine P4/P5 gates;
advanced flight dynamics and sensor rendering remain separately qualified work.

## Next delivery priorities

The [refactor review](NEXT_REFACTOR_REVIEW.md) supplies source findings;
[capability delivery](CAPABILITY_DELIVERY.md) supplies F1–F7 owners, dependencies,
viewer surfaces and acceptance criteria. Both are planned work, not shipped claims.

| Order | Deliverable | Product phases / migration dependencies |
| --- | --- | --- |
| 1 | Correct command lifecycle and complete checkpoints; explicit subsystem ownership; precision/validation contracts | Product 0; repair P4.1 before P5 controls |
| 2 | Compose and qualify a complete mission using merged landscape builders, belief publication and stage bodies | Product 0–2; P4.3/P4.6/P4.8/P4.9 |
| 3 | Truth/belief separation, validated task-to-execution pipeline, physical localization semantics and independent accuracy evaluation | Product 2–3; P4 integration plus separately versioned physical profile |
| Alongside 1–3 | F1 scorecards, F2 failure scenarios, F3 explanations; F4 paired scenario comparisons | Product 0–1 and early operator slices; reuse current replay/evaluation first |
| 4 (basic controls early) | Viewer Operate/Plan/Review, world-aware live protocol, responsive worker, acknowledged controls and durable events | P2 and P5; product 5 is final operator acceptance, not the start of UI work |
| 5 | F5 continuous battery contingencies and task handoff | Product 3; validated execution and current belief/localization |
| 6 | F6 cross-mission spatial memory and F7 evidence-driven adaptive inspection | Product 4; single-mission F7 may precede F6 |
| Throughout / after migration | Measure startup/tick/render/storage costs, share coverage work, bound snapshots/recordings, then retire unused Python paths | Migration P6 only after consumer and installed-package gates |

The [viewer specification](VIEWER_UI.md) is the presentation contract. Backend
features and their operator views ship as vertical slices; a plausible display
must never stand in for missing belief, evidence or mission state.

## Latest system and viewer integration

The [autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md) extends delivery to
onboard-independent contingencies, true 3D belief/trajectory semantics, qualified
SIL/HIL, localization integrity and client outcome packages. The
[viewer feature plan](VIEWER_FEATURE_INTEGRATION.md) maps those requirements and
F1–F7 to concrete workflows, data owners and acceptance fixtures.

Operate / Plan / Review remain the global workspaces. Perspective / Top-down /
Section are projections; Operational / Reconstruction / Safety are layer presets.
The timeline alone controls source time. Plan validates altitude and time as well
as horizontal geometry. Review connects evidence, reconstruction and change to
mission assurance and audit products. HIL/live supervision begins read-only;
telemetry access cannot grant autonomous command capability.

S0 supplies runtime coherence, S1 canonical client-output contracts, S2 volumetric
belief/reconstruction presentation, S3/S4 flight-software and hardware qualification,
and S5–S8 progressively qualified aircraft/fleet/intelligence capabilities. These
are dependency gates, not renumbered product or migration phases, and do not
prevent replay-first UI work with explicit unavailable fields.

## Product North Star

An authorized operator should be able to define a bounded site and objective,
run a mission in realistic simulation or through an approved live adapter, and
receive a reusable world record that explains:

- what was observed, reconstructed, and remains unknown;
- where each platform was, and how certain that estimate was;
- which mission decisions were made and why;
- which safety constraints applied or blocked an action;
- which inspection evidence satisfies the requested quality criteria; and
- what changed since a previous inspection and whether a revisit is warranted.

ArgusNet is intentionally not an unsafe autonomy, surveillance, or weaponization
platform. Human supervision, site authorization, privacy protections, and
constraint-bounded operation are product requirements rather than optional
features.

## Strategic Outcome

The eventual system operates as a closed loop:

```text
mission and site constraints
  -> prior-world loading and sensor ingest
  -> localization and belief-world update
  -> spatial-memory/index update
  -> task and viewpoint selection
  -> feasible 3D trajectory proposal
  -> blocking safety/deconfliction validation
  -> simulated or approved execution
  -> replay, evidence, metrics, and persistent mission record
```

Ground truth is permitted through explicit simulation capabilities for synthetic
observations, physical collision and evaluation. It must never become a hidden
input to planning, safety decisions, localization or operator-mode visualizations.
The physical collision floor is separate from the belief-based safety decision.

## Architectural Principles

1. **Plan from belief, not truth.** The reconstructed belief world is the source
   of planning and safety decisions. Unknown space is not safe space.
2. **Safety gates execution.** A mission intent becomes executable only after it
   has a feasible trajectory and a recorded passing safety decision.
3. **Human supervision is explicit.** Operators define mission boundaries,
   review critical transitions, can hold/return/abort, and can reconstruct why
   decisions were made.
4. **State has a single owner.** Rust remains the runtime authority for fused
   object-state output. New subsystems must not create hidden parallel sources of
   truth.
5. **Every consequential action is replayable.** Inputs, uncertainty, plans,
   safety decisions, execution results, and operator overrides are evented and
   versioned.
6. **The visual world reflects the analytic world.** The viewer can simplify
   geometry for performance, but it must display the same belief state that
   planners and safety checks use.
7. **Simulation and field paths share contracts.** Deterministic simulation,
   recording replay, hardware-in-the-loop, and approved adapters should use the
   same mission, observation, and safety interfaces.
8. **Schema changes are additive by default.** Protobuf, replay, scene, and
   evidence formats need compatible migration paths.

## State Authority Model

The platform should make these distinct representations explicit.

| State | Authority and purpose | Permitted planner use |
|---|---|---|
| Ground Truth World | Simulation-only terrain, objects, weather, and poses used to synthesize measurements and score results | Never |
| Prior World | Imported site boundary, rough DEM, existing map, asset list, and operator constraints | Yes, as an initial prior |
| Belief World | Continuously reconstructed terrain, occupancy, semantics, coverage, uncertainty, and temporal state | Yes; primary world authority |
| Localization State | Pose, covariance, confidence, status, and matched map evidence | Yes; required for map-relative actions |
| Mission State | Tasks, routes, trajectories, operator approval, safety and execution outcomes | Yes; operational authority |
| Evidence Store | Immutable captures, calibration, quality, reconstruction, and review results | Yes, for inspection/revisit work |
| Visual World | Viewer-friendly representation of current/replayed belief state | No direct authority |

## Capability Pillars

### 1. System contracts and runtime foundations

The first priority is making the many existing modules composable rather than
adding disconnected capability.

Required domain contracts:

- `MissionDefinition` and `MissionConstraints`;
- `PlatformProfile` and `SensorRig`;
- `ObservationSource` and sensor envelopes;
- `BeliefWorld` and read-only `WorldBeliefQuery`;
- `LocalizationEstimate` with covariance, confidence, and lifecycle state;
- `MissionTask`, `RouteCandidate`, and `TrajectoryProposal`;
- `SafetyDecision`, `ExecutionCommand`, and fallback policy;
- `InspectionRequest`, `InspectionEvidence`, `InspectionReconstruction`, and
  `InspectionResult`; and
- versioned replay and event records.

Each contract should define ownership, IDs, coordinate frame, timestamp source,
uncertainty semantics, serialization, migration policy, and whether it is
simulation-only, operator-visible, or valid for approved live operation.

Implementation priorities:

- Decompose `simulation/sim.py` into scenario construction, dynamics, observation
  synthesis, mission loop, and replay assembly.
- Decompose large viewer surfaces into state, rendering, interaction, and panel
  modules with stable UI-facing state.
- Put observation synthesis behind a versioned `ObservationSource` interface.
- Record task, route, trajectory, safety, and execution decisions as typed events.
- Enforce state ownership and truth isolation in tests.

### 2. Realistic digital-twin simulation

Simulation realism should be built in tiers so deterministic tests stay fast and
high-fidelity runs remain optional.

#### Tier A: mission-realistic dynamics

- Multirotor and fixed-wing platform profiles.
- Acceleration, jerk, turning, bank/yaw limits, climb/descent, terrain following,
  launch/landing, hold, return-home, and abort behavior.
- Battery and payload models driven by speed, climb, hover, sensor use, wind, and
  reserve policy.
- Wind fields, gusts, turbulence, precipitation, fog, temperature, and visibility.
- Terrain- and structure-dependent RF range, latency, packet loss, and lost-link
  behavior.
- Dynamic civilian obstacles, temporary construction hazards, and changing site
  conditions.

#### Tier B: sensor-realistic observation generation

- Camera calibration, distortion, exposure, rolling shutter, motion blur, dynamic
  range, lighting, and low-light behavior.
- Thermal contrast, emissivity, atmospheric attenuation, sun/time-of-day effects,
  and calibration drift.
- Depth/lidar incidence-angle effects, point density, range limits, dropout, and
  multipath.
- IMU bias, drift, saturation, vibration, and calibration error.
- GNSS occlusion, multipath, correction quality, and constellation degradation.
- Sensor clock drift, delayed delivery, packet loss, and out-of-order arrival.

#### Tier C: environmental and visual realism

- DEM, LiDAR, mesh, BIM/CAD, GIS, and orthophoto scene ingestion.
- Procedural urban, industrial, agricultural, coastal, forest, mountain, and
  construction scenarios.
- Semantic materials and land-cover: roof, road, vegetation, water, soil, metal,
  glass, solar panel, utility asset, and restricted area.
- Seasonal vegetation, weather, light, and surface-state variation.

The deterministic analytic backend should remain the default for unit tests.
Geometry/physics and high-fidelity visual-sensor backends should be selectable
through the observation-source contract rather than entangled with core mission
logic.

### 3. Sensing, ingestion, and perception

ArgusNet needs one ingestion path for synthetic, recorded, and supported live
sensor streams.

- Typed sensor envelopes with device identity, sequence, timestamp source,
  calibration version, integrity status, and provenance.
- Clock synchronization, bounded buffering, and explicit late/out-of-order policy.
- Sensor-health state: rate, latency, dropout, saturation, calibration validity,
  and confidence.
- Raw media stored by reference, with strongly versioned derived perception output.
- A perception plugin interface for detections, keypoints, semantics, depth,
  landmarks, anomalies, and tracks.
- Dataset import/export, annotation manifests, and reproducible lineage from raw
  capture through calibration and perception version to map/evidence artifact.

### 4. Belief-world mapping and spatial representation

Mapping must evolve from coverage accounting into a persistent, uncertainty-aware
world representation that planning can query safely.

Each tile or cell should ultimately include:

- terrain elevation and uncertainty;
- surface normal and terrain/flight risk;
- obstacle probability and static/dynamic classification;
- semantic and land-cover labels;
- observation count, quality, age, and provenance;
- confidence and change probability;
- geofence, operational-zone, and privacy-zone membership; and
- links to relevant keyframes, evidence, and map version.

Roadmap sequence:

1. Introduce persistent, multi-resolution tiled storage and spatial indexing.
2. Fuse elevation and occupancy measurements with uncertainty propagation.
3. Fuse semantic evidence without treating a single classifier output as truth.
4. Age stale observations and distinguish unknown from known-safe space.
5. Add temporal layers and change candidates.
6. Support tile snapshots, replay reconstruction, map merge, and conflict handling.
7. Export GeoTIFF, GIS features, point clouds, meshes, and scene packages.

The viewer must show belief terrain, coverage, confidence, obstacle probability,
staleness, and frontier state from the same map representation used by the
planner.

### 5. Localization and relocalization

Localization should become a confidence-driven operational service rather than a
simulation phase gate.

Inputs:

- IMU propagation;
- GNSS and corrected GNSS priors;
- barometric and terrain-relative altitude;
- visual odometry and VIO;
- depth/lidar/camera landmark matching;
- terrain and structure alignment;
- keyframe retrieval, loop closure, and prior-world matching.

Outputs:

- local, global, and map-relative pose;
- covariance or compact uncertainty representation;
- confidence and lifecycle status;
- matched map region/landmarks;
- failure reason and recovery recommendation.

| Localization state | Required mission behavior |
|---|---|
| Initializing | Collect safe observations; do not perform precision map-relative work |
| Localized | Allow normal planning subject to other constraints |
| Degraded | Reduce speed, increase clearance, favor known space and landmarks |
| Lost | Hold, relocalize, use an approved fallback, or return home |
| Recovered | Revalidate planned path before resuming work |

The key acceptance scenario is a restarted or replacement platform, with only a
weak GNSS prior, safely recovering map-relative pose against a prior ArgusNet
world and revisiting a stored site.

### 6. Closed-loop mission executive, planning, and trajectories

The target mission pipeline is:

```text
mission intent
  -> candidate task
  -> candidate route/viewpoint
  -> speed/altitude/gimbal profile
  -> feasible trajectory
  -> safety and deconfliction decision
  -> executable command or explicit rejection
```

Task classes should include mapping frontier, uncertainty reduction,
localization-support observation, inspection, evidence capture, revisit, change
verification, relay/coverage support, hold, return home, and operator review.

Task scoring should consider mission value, information gain, inspection priority,
localization gain, energy, weather, sensor quality, communications, safety risk,
deadline, and guaranteed return-home feasibility.

Planning progression:

1. Wire current frontier and claimed-cell helpers into the active runtime.
2. Replace 2D-only routing with terrain/obstacle/uncertainty-aware 3D route
   candidates.
3. Add kinematic and energy feasibility.
4. Add time-aware corridor and platform deconfliction.
5. Add sensor-viewpoint and multi-view capture planning.
6. Replan when belief-world, localization, weather, link, or platform health
   state changes materially.
7. Always retain a safe hold or return-home contingency.

### 7. Safety and responsible operation

Safety must be a blocking execution gate. It should validate:

- mission geofence, privacy zone, and exclusion-zone compliance;
- terrain and obstacle clearance;
- vehicle speed, turn, acceleration, climb, descent, and gimbal limits;
- battery reserve and return-home feasibility;
- localization confidence and uncertainty-dependent margins;
- link quality and lost-link policy;
- weather and sensor-operability limits;
- platform health and data freshness; and
- horizontal, vertical, and time-based multi-platform separation.

Every rejected or modified command needs a structured reason, the relevant
constraint values, the selected fallback, and a replay-visible event. Safety
logic should be shared across simulation, replay evaluation, HIL, and approved
live adapters.

### 8. Multi-platform coordination

Coordinate teams conservatively and degrade safely under partial connectivity.

1. Deterministic shared coverage claims.
2. Battery-aware task allocation.
3. Route/corridor/altitude deconfliction.
4. Delayed messages and disconnected-operation handling.
5. Tile synchronization and map-conflict resolution.
6. Localization/communications support tasks.
7. Heterogeneous platform/sensor roles.
8. Safe reassignment when a platform returns, fails, or loses link.

No team-level optimization may weaken a single platform's safety constraint.

### 9. Inspection, evidence, reconstruction, and change detection

Inspection should graduate from POI dwell to evidence-quality completion.

#### Site and request model

- Stable asset/site IDs and map-relative geometry.
- Sensor, resolution, standoff, view-angle, overlap, lighting, and timing
  requirements.
- Planned and alternate viewpoints with safety/localization thresholds.
- Repeat inspection policies and priority/deadline constraints.

#### Evidence pipeline

- Immutable evidence manifests with capture, camera/platform pose, calibration,
  localization quality, and file references.
- Quality checks for sharpness, exposure, blur, occlusion, thermal contrast,
  resolution, overlap, and coverage.
- Operator review, rejection, approval, and reason codes.
- Links between evidence, maps, POIs, keyframes, missions, and change candidates.

#### Reconstruction and comparison

- Local point clouds, meshes, orthomosaics, depth fusion, and thermal overlays.
- Alignment confidence against the world model.
- Prior/current synchronized review and difference overlays.
- Change confidence, severity, affected area, and false-positive review.
- Revisit recommendations based on asset policy, detected change, and evidence
  insufficiency.

### 10. Operator experience and visual system

The Rust/Bevy viewer should become the native mission-control application. It
should support focused operator workflows, not just render replay data.

The [viewer system specification](VIEWER_UI.md) defines the shell, visual
language, source modes, controls and acceptance gates. Its workspace hierarchy is:

| Workspace | Operator objective |
|---|---|
| Operate | Supervise belief world, platforms, task execution, safety and contingencies |
| Plan | Author bounded mission intent, priors, zones, tasks and validated proposals |
| Review | Reconstruct events, evaluate outcomes, inspect evidence and compare missions |

Diagnostics, Scenario Studio, Reporting and Site catalog are secondary
destinations. Live 3D World and Replay Timeline are shared surfaces, not separate
mission authorities. F1–F7 surface assignments are in
[capability delivery](CAPABILITY_DELIVERY.md).

Essential interactions:

- Select a platform to inspect its task, route, confidence, battery, link,
  sensor state, and contingency action.
- Select a map cell to inspect coverage, uncertainty, provenance, staleness,
  obstacle probability, and safe/unsafe rationale.
- Select a POI to compare requested versus achieved viewpoints and evidence.
- Select a safety event to see the geometry, constraints, rejected command, and
  fallback.
- Compare previous/current evidence and reconstruction with split view,
  synchronized camera, blending, and difference layers.
- Pause/replay events without losing live-operation context.

Use a consistent visual language for safe, unknown, caution, blocked, degraded,
and critical state. Provide a low-clutter operational mode and a rich diagnostic
mode. A future web companion can serve reporting and asynchronous review, but it
must consume shared contracts and never fork mission logic.

### 11. Persistent data platform and spatial memory

Support cross-mission value with a durable data architecture:

- site and mission catalog;
- tiled belief-world store;
- keyframe, landmark, and observation index;
- evidence and reconstruction object store;
- spatial/temporal retrieval index;
- replay/event store;
- model/calibration/version registry; and
- retention, privacy, export, and deletion policy.

Representative retrieval questions:

- Which thermal captures cover this asset region?
- What was the last high-confidence inspection of this site?
- Which viewpoints are repeatable and well localized?
- What changed since the previous approved inspection?
- Which missions were affected by a calibration or perception-model version?

### 12. Integrations, security, and release engineering

Maintain narrow, audited boundaries for ROS 2, MAVLink, PX4, file replay,
hardware-in-the-loop, and other approved adapters. Required platform controls:

- device identity, signed envelopes, sequence validation, and replay resistance;
- TLS/mTLS and transport hardening;
- role-based access for operators, supervisors, reviewers, and administrators;
- auditable geofence edits, mission edits, overrides, and evidence approvals;
- privacy zones, redaction, retention, and export policy;
- simulation-only, HIL, and live-operation mode separation; and
- no external execution path that bypasses safety validation.

## Delivery Phases

The phases below describe dependency order, not fixed calendar estimates.

### Phase crosswalk for the integrated plan

These are scope mappings, not claims that any complete phase has passed.

| Product phase | Features and viewer delivery | Migration / system gate |
| --- | --- | --- |
| 0 — Contracts | Command/checkpoint corrections, typed decisions, client-output skeleton, replay-first shell | P4.1/P4.8, P2.0/P2.5; S0/S1 |
| 1 — Simulation | F1 assurance, F2 fault campaigns, F3 explanations, F4 paired comparisons; Scenario Studio and Review | P4.9 qualification; S3 SIL, S4 hardware timing |
| 2 — Belief/localization | Reuse P4.3g/P4.6d; add 3D chunks and integrity; Perspective/Top-down/Section | P4.6/P5.1; S2 spatial revision, later S6 GNSS-denied envelope |
| 3 — Planning/safety | F5 reserve/handoff; validated 3D/time proposals and onboard contingencies | P4.7/P4.8, acknowledged P5 controls; S3–S6 promotion |
| 4 — Inspection/memory | F6 site history, F7 evidence/revisit, reconstruction/change/client outputs | S1/S2 contracts; Plan/Review; later S8 scale |
| 5 — Operator/integrations | Complete Operate/Plan/Review, read-only HIL/live, reporting/audit; separately qualified command capability | P2/P5 acceptance; S4–S6; no control authority from telemetry |
| 6 — Expansion | Fleet reservations/partitions, qualified ML, edge sync and multi-site review | S7/S8; preserve earlier safety/provenance gates |

Migration P6 remains consumer-driven Python retirement, independent of product
Phase 6. Early replay UI and scorecard work need not wait for aircraft milestones.

### Phase 0 — Architecture, contracts, and hardening

Deliver:

- state-authority matrix and schema/migration policy;
- observation-source and mission-decision event contracts;
- decomposition plan for simulation and viewer monoliths;
- deterministic test and replay baseline;
- baseline performance and visual snapshots;
- safety-gate interface around all new motion paths.

Exit condition: new planning or UI work no longer needs direct simulation-truth
access or bypasses typed decisions.

### Phase 1 — Credible mission simulation

Deliver:

- platform dynamics, wind/weather, battery, and link models;
- sensor timing, noise, failure, and calibration behavior;
- scene and scenario realism matrix;
- mission-realism metrics and scenario controls.

Exit condition: mission outcomes change for physically credible reasons, not only
scripted scenario logic.

### Phase 2 — Belief-world and localization authority

Deliver:

- persistent tiled belief world;
- elevation/occupancy/semantic uncertainty updates;
- localization covariance/status contract;
- GNSS/IMU/VIO/map-relative recovery path;
- belief and localization layers in the viewer.

Exit condition: planning works from incomplete reconstructed state, and a later
mission can recover a map-relative pose.

### Phase 3 — Closed-loop planning and safety

Deliver:

- active frontier/next-best-view tasking;
- 3D route and trajectory proposals;
- blocking safety/deconfliction gate;
- energy/comms/uncertainty-aware contingencies;
- deterministic multi-platform coordination.

Exit condition: every executed action has a task, trajectory, safety decision,
reason, and fallback.

### Phase 4 — Inspection intelligence and spatial memory

Deliver:

- persistent sites, requests, evidence, and quality checks;
- local reconstruction and world alignment;
- index-backed evidence retrieval;
- prior/current comparison, change review, and revisit workflow.

Exit condition: an inspection result is defensible through evidence quality,
provenance, reconstruction, and recorded review outcome.

### Phase 5 — Operator product and controlled integrations

Deliver:

- polished mission-control, site-authoring, inspection-review, and reporting UI;
- live-operation hardening and HIL workflow;
- security, audit, retention, and privacy controls;
- deployment/release packaging and supported adapter documentation.

Exit condition: an authorized operator can plan, supervise, review, and replay a
complete bounded mission without developer-only tooling.

### Phase 6 — Advanced expansion

Potential future work after the operational core is stable:

- learned semantic perception and anomaly candidates;
- large-scale multi-site spatial memory;
- multi-modal reconstruction and temporal change analysis;
- distributed edge/map synchronization;
- plugin SDKs for sensors, planners, scene sources, and exporters; and
- collaborative review and site-management workflows.

## Evaluation and Release Gates

Every major capability needs scenario-based, measurable acceptance criteria.

| Area | Representative measures |
|---|---|
| Mapping | coverage, map completeness, terrain error, obstacle precision/recall, uncertainty calibration |
| Localization | position/orientation error, covariance calibration, time-to-relocalize, recovery success |
| Planning | task completion, information gain, energy use, replanning latency, route feasibility |
| Safety | prevented violations, false blocks, clearance margin, contingency success |
| Inspection | view completion, quality score, reconstruction quality, change-detection precision/recall |
| Viewer | frame rate, live latency, visual regressions, operator task-completion time |
| Platform | ingest throughput, dropped frames, replay determinism, API compatibility, storage performance |

Required scenario families include urban occlusion, vegetation, canyon terrain,
low-light thermal inspection, high wind, degraded GNSS, intermittent comms,
dynamic obstacles, low battery, lost localization, multi-platform conflicts,
repeat inspection with known change, and delayed/corrupted sensor streams.

Release gates should include unit, integration, replay, truth-isolation,
property/fuzz, end-to-end scenario, performance, visual-regression, security,
and hardware-in-the-loop checks appropriate to the capability's risk.

## Immediate Implementation Backlog

Use this queue with the stable migration packages, not as a second phase system:

1. Repair command/reset/single-step/dedup semantics and stateful checkpoints.
2. Maintain existing proto-regeneration and field-parity CI; finish semantic
   conversion validation, f64 interchange and representative renderer baselines.
3. Integrate the merged world builders and belief revisions; qualify one Rust-owned mission
   with direct fusion, traced safety ordering and nonempty snapshots.
4. Require belief inputs and typed task → route → trajectory → safety → execution
   decisions for new mission paths; qualify the physical localization profile.
5. Deliver F1/F2/F3 using current replay contracts where possible, then F4 paired
   comparisons. Carry explicit unavailable states until richer contracts land.
6. Implement the viewer spec against replay fixtures, then bind the repaired
   worker/commands and coherent live/world protocol. Separate lossy display from
   durable events and recording.
7. Add F5 reserve-aware replanning, F6 spatial memory and F7 adaptive inspection
   using the feature dependency/acceptance matrix.
8. Re-profile, optimize demonstrated hot paths, and delete migrated Python paths
   only after callers, packaging and exports pass compatibility checks.

## Governance

Maintain architecture decision records for consequential choices, especially:

- belief world as the planning/safety authority;
- mission intent through feasibility and safety gates;
- localization confidence requirements for precision inspection;
- persistent evidence and spatial-memory ownership;
- simulator backend boundaries and truth isolation; and
- approved external execution adapter policy.

Roadmap implementation should favor small vertical slices: one state contract,
one scenario, one operator visualization, one safety condition, and one
measurable acceptance test at a time. This preserves the existing deterministic
simulation/replay strengths while steadily increasing realism and operational
value.
