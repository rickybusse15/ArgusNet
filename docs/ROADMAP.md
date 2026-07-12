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

Ground truth is permitted in simulation only to generate synthetic observations
and score results. It must never become a hidden input to production planning,
safety, localization, or operator-mode visualizations.

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

| Workspace | Operator objective |
|---|---|
| Mission Control | Monitor mission status, platforms, task execution, safety, and live map |
| Site and Map Editor | Define priors, geofences, zones, POIs, constraints, and asset metadata |
| Live 3D World | Understand belief terrain, uncertainty, paths, sensor footprints, and hazards |
| Inspection Review | Review evidence, reconstruction, coverage, quality, and changes |
| Replay Timeline | Reconstruct decisions, state transitions, captures, and safety events |
| Diagnostics | Diagnose sensor health, localization, fusion, streaming, and performance |
| Scenario Studio | Build deterministic scenes, profiles, and benchmark runs |
| Reporting | Produce map, mission, evidence, and audit exports |

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

## Engineering Rules and Definition of Done

The Architectural Principles above state intent. This section states the enforceable
rules that carry that intent into day-to-day work. A change that violates one of
these is a defect regardless of the feature it delivers, and reviewers should block
on it. These rules operationalize `docs/STATE_OWNERSHIP.md`,
`docs/ARCHITECTURE_DECISIONS.md`, and `docs/PERFORMANCE_AND_BENCHMARKING.md`.

### Universal Definition of Done

A change is *done* only when all of the following hold:

- Behavior is covered by at least one test at the appropriate level (unit,
  integration, replay, truth-isolation, or scenario).
- Determinism is preserved: a fixed seed + config yields the same replay hash, or
  the hash change is intentional, reviewed, and the baseline is updated in the same
  commit with a one-line reason.
- State ownership is unchanged, or the `docs/STATE_OWNERSHIP.md` table is updated in
  the same commit.
- No ground-truth value became a hidden input to planning, safety, localization,
  belief-mapping, or operator-mode visualization (truth-isolation still passes).
- Schema changes (proto, replay, scene, evidence) are additive, or a migration plus
  version bump plus reader/writer/viewer update ship together.
- The `docs/KNOWN_GAPS.md` status label and the relevant subsystem doc are updated
  to match reality.
- Any public Python symbol that moved keeps a re-export from its old module (the
  `environment.py` backward-compatibility rule).

### Vertical-slice rule

Favor one state contract + one scenario + one visualization + one safety condition +
one measurable acceptance test over broad horizontal edits. A pillar item is not
"started" until it has a runtime wiring path. A module that exists but sits on no
execution path is tracked as **Planned** in KNOWN_GAPS, never **Implemented**.

### Truth-isolation rule (hard gate)

Simulation ground truth (`TruthState`, true poses, true object/weather state) may be
read only by (a) observation synthesis and (b) evaluation/metrics. Any read of a
truth-typed value from planning, safety, localization, belief-mapping, or
operator-mode viewer code fails review. Every new motion-producing path routes
through the safety gate before execution — there is no "temporary" bypass.

### Event-first rule

Every consequential decision — task selection, route/trajectory choice, safety
accept/reject/modify, operator override, execution outcome — is recorded as a typed,
versioned replay event *as* it takes effect, not reconstructed afterward. If it is
not in the event stream, it did not happen for audit purposes.

### Cross-language boundary rule

Per ADR Decision 1: Rust owns fused object-state, safety math, and latency-critical
runtime; Python owns scenario, orchestration, analysis, and export. Do not
reimplement fusion or safety math in the Python hot loop except as a clearly labeled
test/reference utility. New math that will run per-frame in production belongs in
Rust behind a typed boundary.

### Codebase organization rules

**Dependency direction is one-way.** `core/types` (contracts) ← subsystems (`world`,
`sensing`, `mapping`, `localization`, `planning`, `mission`) ← `simulation` /
`adapters` (orchestration) ← `cli`. Contracts never import subsystems; subsystems
never import the simulation loop; nothing imports `cli`. An import that reverses this
is a defect.

**Where new code goes.**

- New shared runtime/replay state → a frozen dataclass in `core/types.py`.
- New domain logic → the owning subsystem package, never `simulation/sim.py`.
- New observation generation → behind the `ObservationSource` contract, not inline in
  `build_observations()`.
- New viewer feature → the matching viewer module (state / render / interaction /
  panel), not `app.rs` or `ui.rs`.

**Monolith decomposition targets (active debt — do not grow these).**

| File | Approx. size | Split into |
|---|---|---|
| `src/argusnet/simulation/sim.py` | ~7,100 lines | scenario construction, dynamics, observation synthesis, mission loop, replay assembly |
| `rust/argusnet-viewer/src/ui.rs` | ~2,900 | panel modules with stable UI-facing state |
| `rust/argusnet-viewer/src/app.rs` | ~2,700 | state / render / interaction |
| `rust/argusnet-core/src/lib.rs` | ~2,300 | filter, association, health, service boundary |
| `src/argusnet/planning/inspection.py` | ~1,500 | request model, viewpoint planning, evidence/quality |
| `src/argusnet/simulation/behaviors.py` | ~1,300 | per-behavior modules |

Guideline: a new Python module over ~800 lines or a Rust module over ~1,000 lines
warrants a decomposition note in its PR; adding to a listed monolith requires
justification and must not deepen its responsibilities.

**Naming and units.** Meters for distance, radians for internal angles (CLI degree
flags labeled), Z above datum. When a frame or unit is ambiguous, encode it in the
name or a typed wrapper. IDs are stable and typed, not positional indices.

### Performance rules and budgets

These make `docs/PERFORMANCE_AND_BENCHMARKING.md` enforceable.

- **Benchmark before merge.** Any change to a hot path — per-frame sim step, fusion
  ingest, terrain query, LOS/occlusion, planner cost, replay assembly, viewer frame —
  reports Level-0 or Level-1 numbers (median / p95 / p99, input size, seed, commit
  SHA) before and after. No numbers, no merge for hot-path changes.
- **Complexity budget.** Per-frame cost must not scale worse than linearly in
  drones × objects without an explicit, documented justification.
- **Terrain access.** Batched terrain reads go through the cached `height_at_many()`
  path; no new per-point Python loop over terrain on a hot path.
- **No hidden state.** Hot paths make no unbounded per-frame heap allocation and hold
  no hidden global mutable state.
- **Viewer parity.** The viewer holds its interactive frame-rate target; geometry may
  be simplified for display but must reflect the same belief state (Principle 6).
- **Scale before claiming a win.** Profile on the large-map (theater/operational)
  scenarios, not only the default small scene.
- **Determinism is a CI gate.** The replay-hash check and a headless reconstruction
  render run in CI. The baseline has drifted and been hand-updated repeatedly; treat
  an unexplained hash change as a failure to investigate, not a baseline to bump.

### Review and merge gate checklist

Every nontrivial PR confirms: tests at the right level; determinism preserved or
baseline intentionally updated; truth-isolation intact; schema additive or migrated;
state-ownership doc current; `KNOWN_GAPS.md` / subsystem doc updated; hot-path
benchmarks attached when relevant; backward-compat re-exports kept; and a safety
decision recorded for any new motion path.

## Delivery Phases

The phases below describe dependency order, not fixed calendar estimates.

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

The first implementation tranche should be deliberately architectural:

1. Publish the state-authority and schema-compatibility rules.
2. Extract observation generation from `simulation/sim.py` behind
   `ObservationSource`.
3. Add typed task, route, trajectory, safety, and execution replay events.
4. Wire `MissionExecutor` into one non-legacy mission execution path.
5. Make `WorldBeliefQuery` the mandatory input for new planning work.
6. Prototype persistent tiled belief-world storage.
7. Add a localization covariance/status contract and corresponding viewer panel.
8. Couple wind, battery, link degradation, and sensor timing into simulation.
9. Add deterministic headless viewer snapshots to CI.
10. Build a mission-control UI slice showing map, platform, task, and safety
    explanations together.
11. Establish the scenario matrix and physical-mode truth-isolation gate.
12. Integrate the in-progress transport/security work as a protected foundation,
    not an isolated feature branch.

### Standing engineering debt (repay alongside feature work)

These are known, verifiable gaps that the rules above should stop from recurring:

- Wire CI regression gates for the replay-determinism hash and a headless
  reconstruction render. `ci.yml` and `nightly-bench.yml` exist but do not currently
  gate on either; the reconstruction baseline has been hand-updated repeatedly, which
  is exactly the drift the determinism gate is meant to catch.
- Repair the shipped `scene.smartscene`, which fails `validate-scene`, and add a test
  that keeps checked-in scene packages valid.
- Land the standing performance budgets: profile the large-map
  (theater/operational) scenarios and record golden Level-1 numbers so regressions
  are detectable rather than anecdotal.

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
