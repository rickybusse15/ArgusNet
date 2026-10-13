# Viewer feature integration and interaction contract

**Design alignment updated: 2026-09-23.** The
[independently scheduled Rust runtime](RUST_RUNTIME_ARCHITECTURE.md) and
[hardware qualification](HARDWARE_INTEGRATION.md) govern source timing,
freshness and control authority. These remain target requirements, not a claim
that the current viewer implements the new UI or scheduler.

## Status, sources and analysis

Planned UI behavior, not a claim of implementation. The supplied autonomous
system plan and product roadmap matched the repository copies byte-for-byte
when reviewed for this update. This document incorporates their requirements
into [VIEWER_UI.md](VIEWER_UI.md), retaining its typography, flat instrument
layout, accessibility, source boundaries and quality tiers.

Authority is divided deliberately: [AUTONOMOUS_SYSTEM_PLAN.md](AUTONOMOUS_SYSTEM_PLAN.md)
defines system outcomes and S0–S8 qualification;
[ROADMAP.md](ROADMAP.md) sequences capabilities;
[CAPABILITY_DELIVERY.md](CAPABILITY_DELIVERY.md) owns F1–F7 acceptance;
[VIEWER_UI.md](VIEWER_UI.md) owns presentation; this document specifies feature
workflows and their data dependencies. P2/P4/P5/P6 migration IDs remain unchanged.

The newer plan materially expands the previous surface-map/replay UI:

| Design gap | Required resolution |
| --- | --- |
| Terrain and 2D cells cannot represent underpasses, overhangs or vertical separation | Surface, structure and occupancy chunks share one spatial revision; Perspective, Top-down and Section are projections of that revision |
| A scalar confidence cannot explain navigation integrity | Show initializing/localized/degraded/lost/recovered, pose uncertainty, drift, anchor age and recovery; distinguish legacy confidence from physical integrity |
| A ground command panel can imply ground-dependent flight safety | Show onboard executive, safety supervisor and active contingency; loss of ground connectivity makes telemetry stale and never disables onboard safety |
| F1–F7 omit qualification and client-product presentation | Add SIL/HIL case evidence, reconstruction/change review, manifest-based exports and release-bundle traceability |
| More features can overload the shell | Keep three workspaces, contextual inspectors, a single timeline and secondary destinations; do not add a permanent panel per feature |
| Live history and simulation pause can be confused | Persist source, execution environment and time context; historical play/pause never pauses a physical aircraft |

## 1. Shell and interaction taxonomy

The application header selects mission/site, workspace and source. The status
line shows freshness, recording completeness, critical integrity/contingency
state and control capability. The left rail is an index of fleet/tasks/alerts
in Operate, mission entities in Plan, and results/evidence/events in Review.
The center is the spatial canvas; the right inspector explains the typed selection.
The bottom timeline is the only source-time scrub/play/pause/step/speed surface.
Keyboard shortcuts invoke the same action; they do not create another clock.

| Control scope | Labels / content | Must not do |
| --- | --- | --- |
| Workspace | Operate, Plan, Review | Change source, command capability or mission state implicitly |
| Projection | Perspective, Top-down, Section | Turn a 2D drawing into the analytic world or duplicate workspace labels |
| Camera behavior | Orbit, free-fly, follow, chase, focus, bookmarks | Advance simulation time or bypass occlusion |
| Layer preset | Operational, Reconstruction, Safety | Hide missing or incompatible data behind plausible geometry |
| Source context | Simulation, SIL, HIL, Live, Replay, Evaluation; current/history status | Grant control from read access; expose truth outside Evaluation |
| Timeline | Source time, available history, events, playback/step where supported | Rewind the external world or imply a live flight is paused |

Use Perspective as the initial 3D overview; Top-down remains a fast overview and
Section makes vertical relationships inspectable. Preserve the user's projection
per workspace. Switching projections preserves selection, revision and time.
Follow/chase are camera behaviors rather than alternate workspaces.

Section specifies a plane, frame, offset, thickness and clip extent. Show its
orientation in a small context view and identify clipped versus absent geometry.
Top-down overlap is never evidence of a collision: selecting a potential conflict
opens vertical clearance and a time-indexed Section. Picking in every projection
resolves the same typed object and full 3D coordinates.

## Timing, freshness and visual delivery

A source snapshot selects compatible committed revisions; its components need
not have the same acquisition time. Operate shows pose/map/plan freshness and
integrity; Review aligns capture, delivery, decisions and gaps on source time;
Diagnostics owns rate/deadline/queue detail. The source bar distinguishes paused
simulation, paused historical playback and a live vehicle still executing.

Keep the flat typography-led visual design, shared 3D entity selection, one
timeline, and independent projection/layer controls. Unknown occupancy,
uncertainty and safety semantics survive every quality tier. Source capabilities
control whether Run/Step/Seek or control intent is available, not the selected
workspace. A stale worker result cannot replace a newer reconstruction or route.

Deliver P5.0 basic controls early, then incremental P2 visual layers and P4.10
freshness/overload states. Acceptance includes delayed sensor evidence, stale map
with fresh pose, pending/rejected command, exhausted history, incomplete recording,
clock reset, source disconnect and a historical HIL view with no actuation.

## 2. Feature-to-view matrix

| Feature / outcome | Primary surface and visible information | Interaction / recorded result | Required data and unavailable behavior |
| --- | --- | --- | --- |
| F1 Scorecards / mission assurance | Review results table: objective status, gaps, evidence, localization integrity, energy, interventions, faults and terminal/landing outcome | Select a failed objective to focus its region and causal events; export report | Evaluator-owned report, objective/event/evidence IDs and metric definitions; incomplete recording or unmatched evaluation is flagged |
| F2 Failure scenarios | Scenario Studio case editor; Operate fault/degradation alerts; Review expected versus observed response | Configure simulation fault onset/duration/seed; select detection → response → recovery markers | Scenario manifest and scheduled/observed fault records; never infer a real fault solely from a test schedule or expose live injection controls |
| F3 Explanations | Platform/task inspector: objective, selected proposal, constraints, uncertainty, next trigger and contingency | Expand Why this action; navigate task → route → trajectory → safety → applied command | Typed decision IDs, reasons, input revisions and outcomes; missing links show incomplete explanation rather than generated rationale |
| F4 Comparisons | Review paired-run table and distributions; Scenario Studio setup | Select matching objective/event across runs, compare spatial outcome, export comparison | Paired seed/config/world/algorithm manifests, failed-run counts and sampling definition; unlike worlds/frames require explicit alignment or independent cameras |
| F5 Energy contingencies | Operate battery/reserve/return-feasibility fields and active return/divert/land trajectory | Inspect limiting constraint, handoff and unassigned tasks; Plan previews backend-validated feasibility | Planner/safety-owned energy prediction, reserve policy, freshness and alternatives; UI never extrapolates a safe-return promise |
| F6 Spatial memory | Site catalog revision/history index; Plan prior selection; Review prior/current geometry | Choose two revisions, inspect changed region and supporting captures | Site/frame/revision IDs, alignment uncertainty, provenance and stale masks; poor alignment blocks a confident change claim |
| F7 Adaptive inspection | POI inspector: requested versus achieved viewpoints, quality failures, unmet criteria and revisit budget | Select missing view to focus capture frustum; follow planned revisit or explicit unmet result | Evidence quality and mission task records with safety/energy validation; route completion alone never marks evidence accepted |
| 3D belief and reconstruction | Canvas surface/structures/occupancy/clearance plus uncertainty and observed mask | Inspect voxel/chunk/asset; clip Section; switch sparse/dense reconstruction without changing analytic revision | Chunk bounds, frame, LOD, resolution, content identity, age and provenance; distinguish not observed, unavailable and clipped |
| Localization / GNSS-denied integrity | Pose uncertainty volume, trail and inspector integrity/recovery fields | Select matched landmark/anchor or loss event; inspect trajectory revalidation after recovery | Estimate frame/time, covariance or containment, integrity, drift, anchor age, failure/recovery and matched evidence; heuristic confidence is labeled legacy |
| Onboard health and autonomous contingency | Fleet index exception state; selected platform compute/sensor/power/thermal/storage/link health | Follow fault → safety decision → contingency → terminal outcome; display last acknowledged mode | Authoritative health and executive/supervisor records; disconnected is not landed, safe, or fault-free |
| Fleet coordination | Ownership/claims, time-indexed trajectory reservations, peer-data freshness and handoff | Select conflict and inspect each platform's constraint/fallback independently | Reservation/claim revision and validity, coordinator/partition state, peer provenance; unknown coordination is explicit |
| SIL/HIL qualification | Diagnostics and Review case matrix: build, environment, detection deadline, allowed transient, containment and outcome | Jump to failing sample/event and export qualification bundle | Frozen software/hardware/config IDs, assertions and measured results; successful playback is not a passed gate |
| ML-assisted results | Evidence/change inspector: candidate class, model version, confidence, out-of-distribution and review state | Inspect source captures; accept/reject a finding through an authorized recorded review path when available | Versioned inference and review records; candidate is not verified fact; never update an executing model through a UI toggle |
| Offline edge and synchronization | Status-line sync summary; Diagnostics backlog/conflicts; Site catalog local availability | Review conflicting revisions and explicit retry/resolution outcome | Local/remote revision IDs, durable queue/error and conflict policy; reconnect does not silently replace an active mission world |
| Governance and audit | Review/Audit and Reporting: approvals, lineage, redaction/retention status and access capability | Resolve artifact lineage or submit authorized export/review request | Service-owned permission, audit and policy results; hiding a control is not authorization enforcement |

## 3. End-to-end operator workflows

### Operate: explain and supervise an autonomous mission

1. Select a mission/source; show execution environment and capabilities before
   enabling controls. Display only a coherent world and snapshot pair.
2. Select a platform in the fleet index or canvas. Keep pose/integrity, current
   task, actual versus commanded mode, energy and contingency above diagnostics.
3. Open Why this action to see the selected proposal, constraints and resulting
   safety/command outcome. Focus the same trajectory in Perspective or Section.
4. On degradation, show severity, freshness, detecting subsystem, active onboard
   response and the next expected outcome. Acknowledging an alert only records
   that it was seen; it never clears the fault or changes the vehicle mode.
5. During lost link, retain a labeled last-known state and last acknowledged
   contingency policy. Do not animate a predicted landing as observed completion.
6. Inspect retained history with a persistent HISTORY label and a Return to
   current action. Real-time ingest continues; source-time controls affect only
   historical presentation.

Read-only is the initial HIL/live capability. Future hold/return/divert/land/abort
requests require a separately adopted command contract and backend authorization.
Never add routine stick/actuator controls as a dependency of normal operation.

### Plan: turn intent into a validated 3D mission

1. Load a versioned site prior and mission constraints, showing CRS, altitude
   datum, source age and unknown volumes.
2. Author/select 3D zones, tasks, assets, sensor viewpoints and evidence criteria.
   Top-down editing always pairs with explicit altitude bounds; Section supports
   vertical placement. A draft has a distinct identity from the active mission.
3. Request backend proposals with speed/time/vehicle envelope, reserve,
   localization requirements and contingency routes. Show invalid or missing
   constraints before any deployment request.
4. Compare proposed and safety-modified trajectories and conflict times. A change
   to the world, constraint, platform or localization basis invalidates approval
   until the backend revalidates that proposal revision.
5. Produce the mission package with declared versions and authorization state.
   Save, validate, authorize, submit and applied are distinct outcomes. Unsupported
   authoring/submission sources remain read-only with a reason.

### Review: convert mission history into defensible client outputs

1. Open the assurance report or an event, resolving the exact world/reconstruction
   revision it references. Missing artifacts remain visible as missing.
2. Select an objective or evidence anchor to inspect original capture, pose,
   calibration, quality, review outcome and source lineage.
3. Compare prior/current geometry with linked cameras in a validated common frame.
   Offer split, blend and difference views; label alignment uncertainty and
   unseen areas so lack of observation cannot appear as removed geometry.
4. Compare scenarios using matched events or mission elapsed time explicitly;
   independent runs do not necessarily share wall-clock timestamps.
5. Export a reviewed product from its immutable manifest. Show complete,
   incomplete, restricted or unavailable state and report generation failures.
   Do not claim signature verification when only a hash is available.

## 4. Client products and presentation

| Output from system plan | Entry point | Minimum preview and export evidence |
| --- | --- | --- |
| Site package | Site catalog → Plan | Boundary/volumes, prior/assets, CRS/datum, permissions and version |
| Mission package | Plan → validation/authorization | Objectives, evidence criteria, platforms, contingencies, proposal revision and manifest |
| Live operating picture | Operate | Pose/integrity, task, health, energy, trajectory, contingency and freshness |
| Belief-world revision | Canvas → chunk inspector | Surface/volume layers, observed/unknown state, revision/LOD and provenance |
| Evidence package | Review → evidence index | Originals, capture pose, calibration, quality and asset association |
| Reconstruction product | Review → reconstruction | Sparse/dense geometry, alignment, completeness and declared export format |
| Change set | Review → prior/current | Candidate/reviewed status, affected 3D region, uncertainty and supporting evidence |
| Mission assurance report | Review → outcomes → Reporting | Completion/gaps, integrity, energy, faults, interventions and terminal/landing outcome |
| Audit bundle | Review → Audit → Reporting | Input/events/decisions/approvals and config/calibration/firmware/model lineage |

Exports follow the system plan's declared formats where implemented. Display
available formats from capabilities; do not imply COG, LAS/LAZ, 3D Tiles or signed
bundles exist before an exporter and compatibility tests land. Every product
retains site/mission, frame/datum, time basis, revisions, content identity and
applicable component/model/calibration versions.

## 5. Presentation contracts and scaling

These are proposed semantic additions, not existing RPCs or assigned proto fields.

| Contract | Authoritative producer | Viewer behavior |
| --- | --- | --- |
| Spatial revision/chunk manifest | World/belief owner | Surface/structure/occupancy/clearance/evidence references, 3D bounds and hashes; atomically install required coherent content |
| Trajectory and validation record | Planner / safety / executive | Position/orientation/time, envelope, constraints, energy and validity; preserve proposal-to-execution identity |
| Localization integrity record | Localization | Distinguish operational lifecycle from legacy estimator status; show validity, age and matched evidence |
| Health/fault/contingency record | Onboard health / safety / executive | Show detection, selected response, progress and observed terminal result |
| Decision/evidence/report record | Respective domain or evaluator | Resolve typed causal links; no UI-derived mission confidence or quality |
| Qualification manifest/result | SIL/HIL runner | Present predeclared limits versus measured outcomes and frozen component identities |
| Capability and governance result | Authorized adapter / data service | Gate mode requests, review/export and artifact access at both service and UI boundaries |

Use immutable shared revisions and bounded asynchronous decoding, spatial
culling, chunk LOD and changed-data uploads. Section sampling and picking must
consult declared data/coordinates; visual simplification cannot change safety
meaning. Raw captures and dense geometry load on demand with explicit progress,
byte budgets, cancellation and errors. Optional detail may refine progressively
within the same revision; required missing layers remain unavailable. Do not
block the render loop on reports, reconstruction, map fetches or recording.

## 6. Delivery and acceptance

| Slice | Dependencies | Demonstration and required gate |
| --- | --- | --- |
| Replay-first shell and F1/F3 | P2 fixtures and S1 output skeleton | Typed selection, coherent time, outcome → event → trajectory navigation; unsupported fields shown honestly |
| 3D projections and belief | P2 presentation + P4.6/S2 spatial contracts | Same object/revision in all three projections; overhang and vertically separated paths remain distinguishable |
| Plan and Review depth | S1/S2 + domain validation/evidence; F4–F7 as their owners land | Invalid vertical/time proposal rejected; insufficient capture remains incomplete; aligned two-mission change review |
| Interactive Rust simulation | Corrected P4.1, P4.8/P4.9, P5 worker/checkpoints | Pause/single-step/seek preserve command and epoch semantics with nonempty mission state |
| SIL/HIL supervision | S3/S4 + P5.3 source contract | Fault schedule versus actual response, lost-link stale state, assurance package and build provenance |
| Aircraft mode requests | Separately authorized command design and S5/S6 gates | No actuation from telemetry capability; explicit command lifecycle; onboard containment survives viewer loss |
| Fleet / ML / scale | S7/S8 qualification | Partition/peer freshness, independent contingencies, reviewed model candidates and auditable offline synchronization |

Functional fixtures must include: overhang/underpass occupancy; vertical and
time-separated trajectories; lost/recovered localization; disconnected platform
with unobserved landing; stale prior map; rejected/expired command; insufficient
inspection evidence; misaligned change comparison; recording gaps; model OOD;
and unavailable client export. Verify keyboard-equivalent selection and non-color
status cues. Retain the viewer spec's named-hardware frame/resource gates.

Before implementation, freeze numeric per-scenario integrity/containment limits,
reference hardware and chunk budgets. Autopilot choice, aircraft command policy,
volumetric encoding, reconstruction formats and model qualification are backend
product decisions; the UI exposes declared capabilities instead of guessing them.

## 7. Console parity: live viewer and the reference replay console

Planned work, not a claim of implementation. A browser replay console
(single-file HTML/Three.js, one baked replay) demonstrated the target
usability: a status strip, a fleet rail, a spatial canvas, a typed inspector and
one docked timeline. It is a **reference prototype and acceptance target only**;
the native Bevy/egui viewer is the product and shares no code with it. This
section records what the native viewer must match and what the live path must
supply first. It refines, and does not replace, the shell in section 1 and
[VIEWER_UI.md](VIEWER_UI.md).

### 7.1 Parity checklist

| Reference console element | Native requirement | Data it reads |
| --- | --- | --- |
| Status strip: sim time, tracks, mean error now, bearings accepted, run mean error, fusion rejection rate, map known, contacts | One strip above the canvas in every workspace; a value with no source reads "unavailable", never zero | `FrameMetrics`; `MappingState.coverage_fraction`; contacts from mapping/world state |
| Entity rail grouped Targets (fused tracks) / Drones / Ground stations, click to select | Operate rail index; selection is the same typed selection as the canvas | `tracks`, `nodes` (`is_mobile`), `target_metadata` |
| Inspector for the selected entity plus a "Why bearings were not generated" breakdown | Typed inspector; rejection bars by reason (dropout, terrain, vegetation occlusion, noise, low confidence) | `generation_rejections`, `rejected_observations`, `FrameMetrics.rejection_counts` |
| Primary belief/truth toggle ("Drones' map" / "True world") ahead of secondary layer chips | Belief first, truth only under Evaluation (VIEWER_UI 2.2); chips for scans, bearings, 2σ, trails, zones, sensor range | `mapping_state`, `scan_mission_state`, `truths` (Evaluation only) |
| 3D / Map toggle, Reset view, scale bar | Perspective / Top-down projection with reset and scale; Section remains planned | none (camera state) |
| One timeline with mean-error and map-known curves, 1×/4×/10× | Single bottom timeline (section 1); curves from per-frame metrics; live shows a history window, never a paused aircraft | `FrameMetrics.mean_error_m`, `MappingState.coverage_fraction` |
| Task per drone and handoff history (not in the prototype; added by planning-efficiency P0) | Rail row and inspector section for current task, reason for last change | per-drone assignment history (P0b) |

### 7.2 Live path audit (2026-10-12)

`argusnet-viewer/src/live.rs` receives `WatchFramesV2` and converts each
`PlatformFrame` with `frame_from_pb` (`argusnet-replay/src/lib.rs`). That
conversion fills tracks, truths, nodes, observations, rejected and generation
rejections, metrics, target metadata and safety events. It sets these to empty
or `None` because the wire message has no field for them:

- `mapping_state` (map known %, covered cells, the drones' map)
- `scan_mission_state` (phase, coverage threshold, POI status, team localization,
  egress progress)
- `localization_state`, `inspection_events`, `deconfliction_events`
- the document `meta`, `summary` and `mission_assurance`

Consequence: a live no-prior-map run (`argusnet live --mission-mode
scan_map_inspect --cooperative-search`) streams tracks and bearings but cannot
drive the map-known, contacts, team-localization or assurance elements above.
`proto/argusnet/v1/world_model.proto` `PlatformFrame` carries fields 1–10 only.
This is the main gap, and it is a data gap, not a styling gap. Replay JSON
already carries all of these fields, so replay and live currently differ in
semantics, against VIEWER_UI 2.3 (same semantics in simulation, replay and
live).

### 7.3 Slices

| Slice | Content | Gate |
| --- | --- | --- |
| V0 Shared frame contract | Add the missing frame content to the live protocol so `frame_from_pb` yields the same `ReplayFrame` as replay JSON. Additive proto fields; update Python bindings (`src/argusnet/v1/`) and Rust (`argusnet-proto`, `argusnet-core`, `argusnet-server`, `argusnet-replay`) together. Rust remains the source of truth for fused state; mapping state is published by the stage that owns it. | A recorded run replayed through the daemon and the same run loaded from JSON produce equal `ReplayFrame`s for every field in 7.2; fields still unavailable stay `None` |
| V1 Shell and strip | Status strip; split `ui.rs` (~3000 lines) into strip, rail, inspector, dock and canvas modules before restyling; no behaviour change in the split commit | `headless.rs` snapshot of each module; existing viewer tests unchanged |
| V2 Rail and inspector | Entity rail, typed selection shared with canvas picking, rejection-reason breakdown, per-source "unavailable" states | Keyboard-equivalent selection; non-colour status cues; selection survives workspace and projection changes |
| V3 Layers and timeline | Belief/truth primary toggle, layer chips, projection toggle with reset and scale, error and map-known curves on the timeline, live history window | Truth hidden outside Evaluation; scrubbing never implies a live flight is paused |
| V4 Planned-feature surfaces | P0b assignment history and efficiency metrics, P1 allocation lines, R0–R4 world layers, scheduler lane rates in Diagnostics | Each ships with the backend slice that produces its data; no panel without a source |

V0 is the prerequisite for everything that makes live look like the console;
V1–V3 can proceed on replay in parallel and are verified there first, then
re-verified on live once V0 lands. V4 items are scheduled by their owning
backend phases rather than by the viewer track.
