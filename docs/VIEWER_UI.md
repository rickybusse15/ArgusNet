# ArgusNet Next-Generation Viewer System Specification

**Design alignment updated: 2026-09-23.** The
[independently scheduled Rust runtime](RUST_RUNTIME_ARCHITECTURE.md) and
[hardware qualification](HARDWARE_INTEGRATION.md) govern source timing,
freshness and control authority. These remain target requirements, not a claim
that the current viewer implements the new UI or scheduler.

## Document status

This document defines the target viewer system at the end of the Rust runtime
migration. It is an implementation specification, not a description of the
current `bevy_egui` interface. Current behavior remains defined by the code,
tests, [known gaps](KNOWN_GAPS.md), and [state ownership map](STATE_OWNERSHIP.md).

The specification is governed by:

- [the autonomous system plan](AUTONOMOUS_SYSTEM_PLAN.md), including 3D semantics,
  onboard authority and S0–S8 qualification;
- [feature interaction and data handoffs](VIEWER_FEATURE_INTEGRATION.md), including
  F1–F7 workflows, client products and SIL/HIL presentation;

- [the product roadmap](ROADMAP.md), especially the operator experience and
  visual-system outcome;
- [the Rust runtime migration](RUST_RUNTIME_MIGRATION.md), especially migration
  Phases 2 and 5;
- [the delivery plan](migration/DELIVERY_PLAN.md), especially P2.0-P2.6 and
  P5.0-P5.5;
- [the runtime and protocol architecture](migration/ARCHITECTURE_AND_PROTOCOLS.md);
- [ADR-007](adr/007-runtime-composition-and-snapshots.md) for snapshot and
  command boundaries; and
- [ADR-008](adr/008-world-aware-streaming-and-compatibility.md) for coherent live
  world state.

If this document conflicts with an authority, safety, protocol, or compatibility
rule in those documents, the governing architecture wins.

## Scheduled-runtime presentation amendment

The visual target remains a flat, typography-led Swiss/Braun instrument layout:
restrained monochrome with purposeful accent, readable labels, no decorative
card wall, neon or glass. Operate / Plan / Review are workspaces; Perspective /
Top-down / Section are projections; Operational / Reconstruction / Safety are
layer presets. Keep these controls distinct and preserve selection across them.

Terrain, structures, unknown/free/occupied volumes, uncertainty, trajectories,
sensor volumes and evidence anchors retain their 3D meaning in every projection.
A low quality tier may simplify rendering, not remove a hazard or turn unknown
space into empty space. Truth overlays remain a labeled evaluation capability.

Independent rates add these required states:

| Surface | Required behavior |
| --- | --- |
| Source bar and timeline | Simulation/SIL/HIL/Live/Replay, current versus historical time, achieved versus requested simulation speed, available history and recording health |
| Operate inspector | Pose/map/plan as-of time, stale/degraded/unavailable reason, localization integrity, active contingency and command pending/applied/rejected state |
| Plan | Show capability/rate incompatibility and stale input revisions before accepting a proposal; revalidation is backend-owned |
| Review | Correlate acquisition/delivery/decision times and evidence gaps; distinguish render-skipped snapshots from lost measurements |
| Diagnostics | Per-subsystem rate, execution/queue latency, deadline misses, clock quality, rejected stale jobs, recorder backlog and resource ceilings |
| Scenario Studio | Configure only supported source rates/delays and execution profiles; unsupported timing fails visibly before start |

A committed revision may contain a fresh pose and an older map; label both ages.
Interpolate compatible presentation poses only; do not invent fresh map evidence,
extrapolate stale tracks indefinitely or interpolate across epochs/frame changes.
Skip superseded display updates without dropping decisions from the event journal.

P5.0 delivers basic controls against the qualified legacy Rust model while
P4.10 is opt-in. Pause/Step/Speed affect virtual simulation; Step processes one
control interval including intervening sensor events. Playback of external
history cannot pause an aircraft. Disable unsupported controls with a reason.
Large world loads, mesh jobs and recorder waits must not block the render thread.

Acceptance fixtures include mixed-rate fresh/stale components, no observations,
clock reset, overloaded workers, full recorder, branch/seek and disconnect.
Check flat-layout typography, keyboard/text-entry behavior, picking and semantic
layers in overview, close perspective, top-down and section on named hardware.

## 1. Product definition

The next-generation viewer is the native ArgusNet mission-control application.
It is the operator's coherent view of a simulated, recorded, SIL/HIL or authorized
live autonomous mission. It presents the belief world, platform state, mission intent, safety
decisions, evidence, and system health without becoming an independent runtime
authority.

The viewer must answer, at a glance:

1. What is the mission doing now?
2. What does the system believe about the world, and what remains unknown?
3. Where is each platform, how certain is that state, and how fresh is it?
4. What is each platform doing, what was commanded, and what will happen next?
5. Which constraint, degradation, or safety decision needs operator attention?
6. What evidence or event explains the current state?
7. Is the operator looking at live state, simulation state, or historical state?

The viewer is not a decorative scene renderer, an engineering debug overlay, or
a second mission planner. It is a safety-conscious information system whose 3D
world and UI are two coordinated views of the same committed runtime snapshot.

## 2. Non-negotiable system principles

### 2.1 One authority per concern

The viewer reads immutable state and submits typed commands. It never owns
authoritative terrain, belief, localization, fusion, mission, safety, or vehicle
state. UI-local selection, camera, layout, filters, and presentation preferences
are the only state that the viewer may own independently.

### 2.2 Belief first; truth isolated

Operator views use the belief world, observed platform states, and fused tracks.
Unknown space must remain visibly unknown. Simulation truth is unavailable at
the presentation boundary unless an explicit evaluation capability authorizes
it. Hiding a truth layer after it entered normal viewer state is insufficient.

### 2.3 Same semantics in simulation, replay, and live operation

The shell, selection model, layer meanings, status vocabulary, and event model
must remain consistent across sources. Source-specific capabilities may be
unavailable, but a term must not change meaning between modes.

### 2.4 Complete revisions only

The viewer presents only committed runtime snapshots paired with compatible
world and belief revisions. It must not combine a new frame with stale terrain,
interpolate across an epoch boundary, or silently render missing world data as a
valid empty world.

### 2.5 Human authority is explicit

Actual state, commanded state, proposed intent, safety result, and execution
acknowledgement are distinct. A proposal is never styled as an active command.
Commands are not considered effective until a typed acknowledgement identifies
the effective tick and resulting revision.

### 2.6 Every consequential state is explainable

Mission transitions, rejected or modified commands, safety decisions, operator
overrides, evidence captures, and faults link to durable typed events. The UI
must offer a direct path from a visible outcome to its cause.

### 2.7 Rendering never controls simulation time

Render cadence, camera motion, visibility, window focus, and playback speed do
not change deterministic simulation results. Simulation, rendering, recording,
and historical review have independent clocks and explicit lag indicators.

## 3. Operating contexts

Every window must expose its source, execution environment and temporal context persistently. The source
indicator may not disappear when a drawer closes or a platform is selected.

| Context | State source | Permitted time operations | Command behavior |
| --- | --- | --- | --- |
| Simulation | In-process `argusnet-sim` worker | Run, pause, step, speed, seek within available checkpoint history, branch after seek | Local acknowledged runtime commands |
| Replay | Versioned replay plus referenced world assets | Play, pause, step, scrub, loop, jump to event | Read-only; annotations remain presentation data |
| SIL / HIL | Flight-software or hardware adapter with explicit qualification identity | Follow execution or review retained history; pause only if declared supported | Read-only supervision initially; test controls require explicit non-live capabilities |
| Live | World-aware stream and fetched immutable revisions | Follow live or inspect retained history | Read-only by default; remote actuation is outside this migration |
| Evaluation | Authorized simulation-derived data | Replay controls plus evaluation comparisons | Read-only; truth may appear only under the explicit capability |

The visible context state is one of:

- `LIVE` — following the newest coherent live snapshot;
- `LIVE / HISTORY` — inspecting retained live history while new live data
  continues to arrive;
- `SIM / RUNNING`, `SIM / PAUSED`, or `SIM / FAULTED`;
- `SIL` / `HIL`, qualified by current or history context;
- `REPLAY`; or
- `EVALUATION / TRUTH ENABLED`.

The viewer must never imply that live history can rewind the external world.

## 4. Information architecture

### 4.1 Primary workspaces

The product uses three primary operator verbs in the persistent shell:

| Workspace | Purpose | Principal content |
| --- | --- | --- |
| Operate | Supervise current mission execution | Live/belief world, platform state, task progress, safety, contingencies |
| Plan | Define or revise bounded mission intent | Site bounds, zones, assets, tasks, route/trajectory proposals, validation |
| Review | Reconstruct and assess what happened | Replay journal, inspection evidence, prior/current comparison, approvals |

These are user workflows, not technical subsystems. Diagnostics, Scenario Studio,
Reporting, and administrative tools are secondary destinations reached from the
application menu or context-specific links. They must not compete equally with
the three primary verbs in the mission shell.

### 4.2 Secondary destinations

- **Diagnostics:** sensor, localization, fusion, streaming, renderer, worker,
  queue, and recording health.
- **Scenario Studio:** deterministic simulation configuration and benchmark
  setup; never shown in an authorized live session.
- **Reporting:** mission, map, evidence, and audit export.
- **Site catalog:** site, prior-world, asset, and mission history selection.
- **Preferences:** keybindings, quality tier, units, accessibility, and local
  presentation settings.

### 4.3 Operate workspace anatomy

The desktop shell is an integrated instrument surface, not a collection of
floating cards:

```text
┌────────────── application identity / mission / workspace / source ──────────────┐
├────────────── system status and freshness line ─────────────────────────────────┤
│ mission rail │                                                                  │
│ task phase   │               dominant 2D / 3D world canvas         selection     │
│ fleet index  │                                                     inspector     │
│ alerts       │                                                                  │
├────────────── mission clock / event scale / selected event ─────────────────────┤
└──────────────────────────────────────────────────────────────────────────────────┘
```

Default reference proportions at 1440×900 logical pixels:

- application header: 64 px;
- status line: 28 px;
- mission rail: 194 px;
- selection inspector: 252 px;
- timeline: 100 px;
- world canvas: all remaining space.

These are starting dimensions, not fixed pixel requirements. The world canvas
must remain the dominant region. Rails are structural surfaces separated by
hairlines; they are not elevated, rounded, translucent, or detached from the
window edge.

The composition should be intentionally asymmetric. The mission rail is compact
and index-like. The inspector may be denser and wider. Alerts may sit directly on
the canvas edge when they point to spatial state, but must use an opaque backing
when needed for legibility.

### 4.4 Responsive behavior

The operational target is a desktop or control-station display. At reduced
widths:

1. preserve the source/mode indicator, world, selected platform identity,
   safety state, and timeline;
2. collapse the mission rail before compressing the canvas below useful size;
3. reduce inspector fields to actual state, commanded state, localization,
   battery, link, and safety;
4. move secondary layer controls into one menu; and
5. never hide a critical alert solely because of window size.

Below the supported operational viewport, the application may enter a compact
monitoring layout. It must not pretend that mission planning or safety review is
usable on an undersized surface.

## 5. Visual system

### 5.1 Design intent

The visual language combines Swiss modernist information design, Braun-era
industrial restraint, airport wayfinding, professional telemetry, and subtle
aerospace instrumentation. The result must feel engineered, quiet, and mature.

Hierarchy is created with type, alignment, spacing, rules, density, and contrast.
Decoration is not a hierarchy tool. Every line, tick, label, and color must carry
information or improve spatial orientation.

### 5.2 Typography

The normative type direction is Helvetica/Neue Haas Grotesk. The shipping build
must bundle or legally depend on a licensed face with the required glyphs and
tabular numerals; it must not assume that a particular operating system font is
installed. Font loading failure must use a documented metric-compatible fallback
rather than silently returning to the default egui face.

Required styles:

| Role | Size | Weight | Tracking | Use |
| --- | ---: | ---: | ---: | --- |
| Object identity | 28-32 px | 300 | +0.05 em | Selected platform or track ID |
| Workspace title | 16-18 px | 400 | -0.01 em | Mission or review context |
| Section title | 13-15 px | 400-500 | 0 | Named content group |
| Body/readout | 10-12 px | 400 | +0.01 em | Values and concise explanation |
| Utility label | 8-9 px | 400-500 | +0.12 to +0.18 em | Uppercase metadata and axes |
| Numeric instrument | 9-21 px | 300-400 | +0.03 to +0.08 em | Time, coordinates, measurements |

Rules:

- use tabular numerals for time, coordinates, telemetry, revisions, counters,
  and measurements;
- sentence case is preferred for human-readable content; uppercase is reserved
  for compact utility labels and status vocabulary;
- no oversized marketing typography exists inside the application;
- long identifiers truncate visually but remain available through selection,
  focus, or copy; and
- weights above 500 are exceptional.

### 5.3 Color

Base reference palette:

| Token | Reference | Purpose |
| --- | --- | --- |
| Window | `#0A0C0D` | Deep neutral application ground |
| Chrome | `#101315` | Rails, header, and timeline |
| Canvas | `#0C1012` | World viewport ground |
| Rule strong | `#363B3E` | Structural boundaries |
| Rule quiet | `#282E31` | Local grouping and table rows |
| Text primary | `#E2E2DF` | Primary identities and values |
| Text secondary | `#A5AAAC` | Normal readouts |
| Text muted | `#687175` | Labels and passive metadata |
| Functional accent | `#D2AD62` | Focus, selection, active index, actionable caution |

The default presentation is monochrome plus the functional accent. Additional
semantic colors are permitted only when they communicate safety meaning. Critical
red is reserved for blocked execution, fault, or immediate safety action. Safe,
unknown, degraded, and critical states must also differ by label, shape, pattern,
or value; color is never the only distinction.

No bloom, neon glow, purple/cyan gradient, glass effect, lens flare, or colored
ambient shadow is permitted in the UI. Scene lighting and post-processing must
not wash out overlays or turn safety colors into mood lighting.

### 5.4 Geometry, spacing, and elevation

- 1 px rules define hierarchy and ownership boundaries.
- Corners are square by default. Radius is zero for panels, tables, controls,
  timelines, and inspectors.
- Controls may use a 1-2 px focus treatment but no pill shape.
- Spacing follows a 4 px base with primary intervals of 8, 12, 16, 20, 24, and
  32 px.
- Structural regions are opaque and flat. Elevation is reserved for temporary
  menus, dialogs, and safety confirmations.
- Shadows are limited to temporary overlays where separation cannot be achieved
  by contrast and a rule.
- Empty metric cards, decorative containers, and card-per-value layouts are
  prohibited.

### 5.5 HUD and instrumentation language

HUD elements are sparse orientation and measurement aids:

- coordinate readouts and CRS indicator;
- north/heading reference;
- altitude or range tape when relevant to the active view;
- scale and render-origin readout;
- selection corner marks bounded to the selected glyph;
- path ticks, waypoint indexes, uncertainty boundaries, and time annotations;
- revision, epoch, and freshness metadata in diagnostic contexts.

HUD elements must not form a decorative background grid, imitate circuitry, glow,
or obscure the world. They appear only where the operator can read a quantity or
make a spatial judgement from them.

### 5.6 Interaction states

Controls use crisp state changes with no ornamental animation:

- idle: muted label and quiet rule;
- hover: one contrast step brighter;
- keyboard focus: visible 1-2 px outline with adequate offset;
- active/selected: functional accent rule plus text or shape change;
- unavailable: reduced contrast plus explicit reason on focus/hover;
- pending: visible command ID or concise `PENDING ACK` state;
- rejected/faulted: semantic treatment plus reason; and
- stale: timestamp/freshness treatment that cannot be confused with valid live
  state.

Motion is limited to world movement, timeline playhead, direct state transition,
and short spatial continuity cues. Respect reduced-motion preferences.

## 6. World presentation

### 6.1 Default operating view

Operate opens in a Perspective 3D overview sized to the mission extent, preserving
an explicitly saved projection preference. Projection controls are Perspective,
Top-down and Section; workspace controls remain Operate, Plan and Review, and
layer presets remain Operational, Reconstruction and Safety. These labels have
different scopes and must not be duplicated as interchangeable view buttons.

Section exposes plane/frame, offset, thickness and clip extent with orientation
context. A clipped region is not missing or observed-free space. All projections
resolve the same typed selection, full 3D coordinates, revision and source time.
Follow/chase changes camera behavior; global context remains one action away.

Available camera behaviors:

- orbit;
- damped free-fly;
- selected-entity follow;
- selected-entity chase;
- focus selection (`F`); and
- bookmarks (`1`-`9`).

Camera motion uses render delta time. Loss of a followed entity returns to a
stable last-known context and reports the loss; it never snaps to the origin.

### 6.2 Visual layers

Layer controls represent meaning, not rendering implementation. The initial
operator presets are:

| Preset | Default content |
| --- | --- |
| Operational | Belief terrain, observed platforms, fused tracks, active task, route, mission zones, relevant alerts |
| Reconstruction | Belief terrain, observed mask, confidence, staleness, localization, reconstruction, provenance |
| Safety | Terrain/obstacle clearance, zones, proposed trajectory, deconfliction, constraint geometry, fallbacks |

Individual layers remain available for advanced use, but the default shell does
not expose every renderer toggle at once.

### 6.3 Belief-world rendering

The viewer renders the same belief representation used by planning and safety:

- belief elevation and slope;
- observed versus unknown mask;
- confidence or uncertainty;
- staleness/last-observed time;
- occupancy or obstacle probability;
- land-cover or semantics where available;
- coverage and revisit state; and
- frontier/candidate regions.

Unknown, unavailable, and stale are distinct. Missing belief fields produce an
explicit unavailable state. Shaders must not infer confidence from truth terrain,
distance to a platform, or visual color.

Terrain, belief, and mission revisions are installed atomically. During revision
transfer, retain the last coherent world and label it stale, or show a bounded
loading state. Never project new platform state onto an incompatible world.

### 6.3a Volumetric world and time-indexed trajectories

Belief presentation includes surface tiles, structure meshes/point clouds, sparse
free/occupied/unknown volumes, clearance fields, semantics, landmarks and evidence
anchors. Each spatial chunk names its 3D bounds, frame, resolution/LOD, revision,
time, content hash and provenance. Legacy surface-only sources show volumetric
state as unavailable; never infer free airspace above a terrain raster.

Trajectories retain orientation, time, velocity/acceleration, vehicle envelope,
clearance, localization requirement, energy and validity interval. Show proposed,
safety-modified, accepted, executing, completed and contingency paths distinctly.
Use Section and the timeline to explain vertical/time separation. Rendering
simplification does not change the analytic revision or validated envelope.

### 6.4 Platforms, tracks, and truth

Platform, fused track, target, observation, and simulation truth are different
types with different glyphs and selection details. A shared label does not merge
identities.

Platform glyphs:

- retain screen-space readability at mission overview scale;
- encode heading and broad platform class without miniature photorealism;
- use a bounded selection outline that respects terrain occlusion;
- expose vehicle geometry at close range when a model is available;
- do not allow trails, sensor volumes, zones, or transparent effects to steal
  platform picks; and
- remain stable across interpolation through typed IDs.

Fused tracks show uncertainty and contributing observation context without being
styled as known truth. Simulation truth appears only in evaluation mode and must
be visually and textually marked as truth.

### 6.5 Sensors, trails, zones, and trajectories

- Sensor frusta and footprints derive from the same pose and limits as runtime
  sensing; unknown range is explicit.
- Trails are time/distance sampled, bounded, and reset at teleport or epoch
  change.
- Mission zones are world-sized volumes or boundaries with type, authority,
  effective revision, and status.
- Proposed, safety-modified, accepted, executing, and completed trajectories are
  visually distinct and linked to command/safety events.
- Diagnostic gizmos are off in normal operator presets.

## 7. Selection and inspection

### 7.1 Selection model

Selection uses typed identity: platform ID, track ID, zone ID, POI/site ID,
evidence ID, map cell/tile key, task ID, command ID, or event ID. Selection
survives ordinary frame updates but clears deliberately when the ID disappears,
the run/epoch changes, or the active source cannot resolve it.

World picking uses occlusion-aware mesh raycasting in the active viewport. UI
capture prevents world picking. Every pointer selection path has an equivalent
keyboard or indexed-list path.

### 7.2 Platform inspector

The default platform inspector presents, in order:

1. typed ID, platform class/profile, role, and adapter/source;
2. actual state and freshness;
3. commanded state and acknowledgement;
4. current task, route/trajectory, and next transition;
5. pose/localization integrity (initializing, localized, degraded, lost, recovered),
   covariance/containment, drift, anchor age/source, reference frame and recovery;
6. altitude, velocity, heading, and relevant dynamics;
7. battery, reserve, and return-home feasibility;
8. link state, latency, loss, and lost-link policy;
9. sensor rig and active capture state;
10. safety envelope and active constraints; and
11. onboard executive/safety-supervisor state, active contingency and observed
    outcome, separate from a pending operator request.

Fields that are unknown or unavailable display those states explicitly. Zero is
never used as a substitute for missing data.

Legacy estimator confidence and timeout remain labeled as legacy fields; they
must not be mapped to physical integrity without a validated adapter. Recovered
localization includes the remaining trajectory's revalidation outcome. A lost
link displays last-known state, never inferred successful return or landing.

### 7.3 Map-cell inspector

Selecting a belief-world location shows coverage, last observation, confidence,
uncertainty, provenance, obstacle probability, semantic hypotheses, and the
planner/safety rationale for known, unknown, safe, or unsafe classification.

### 7.4 Safety-event inspector

A safety event shows the proposed command, evaluated geometry, applicable
constraints and values, decision, modifications, fallback, acknowledgement, and
effective tick. The operator can focus the relevant spatial geometry and jump to
the preceding and following committed snapshots.

### 7.5 Inspection evidence

An inspection site shows requested versus achieved viewpoints, captures,
localization/calibration quality, evidence checks, reconstruction alignment,
review status, and previous/current comparison when available.

## 8. Time, events, and causality

The bottom timeline is the single source-time control surface. Do not duplicate
play/pause, step, speed or scrubbing in the canvas toolbar. Keyboard shortcuts
invoke the same actions. Live-history playback never pauses a physical aircraft.
It contains:

- source-aware time and mission elapsed time;
- current tick/frame and epoch;
- available history extent;
- play/pause/step controls when supported;
- event marks grouped by mission, safety, command, evidence, and fault class;
- recording state and completeness; and
- selected event summary.

Replay scrubbing uses source timestamps, not frame index. Interpolation uses two
adjacent compatible committed snapshots and a clamped blend. It is disabled
across epochs, teleports, missing IDs, discrete mission transitions, and
incompatible world revisions.

Durable events come from the ordered event journal and its cursor. They must not
depend on the lossy presentation-frame path. When the renderer drops intermediate
frames, the UI may remain smooth but the event history remains complete.

Simulation seek restores the nearest checkpoint and deterministically replays
logged inputs. A new command after seek creates a new epoch/branch and preserves
the original history. Replay seek and live-history review never mutate a runtime.

## 9. Commands and operator authority

### 9.1 Command lifecycle

Every runtime command passes through:

```text
operator intent
  -> local validation
  -> command envelope
  -> queued / pending acknowledgement
  -> accepted for scheduling or rejected with reason
  -> applied at effective tick/revision, or cancelled with reason
  -> execution/safety event
```

Admission acknowledgement resolves submission, but the UI retains scheduled/pending
execution state until Applied, Rejected or Cancelled. Duplicate responses preserve
the original result and reason with a retry flag. Queue full,
stale epoch, stale revision, invalid mode, unsupported command, unavailable
rewind, and runtime fault are explicit outcomes. The viewer never retries a
non-idempotent command blindly.

Initial in-process commands are load/start, pause/resume, single-step, set speed,
stop, reset, seek, supported mission-parameter changes, and cancellation of
pending world preparation. Camera, selection, layer visibility, and layout remain
local presentation actions.

### 9.2 Physical aircraft boundary

Normal operation supervises autonomous missions rather than manual flight. The
onboard flight controller, executive and safety supervisor retain stabilization
and contingencies without the viewer or ground link. The UI never streams routine
actuator values or treats alert acknowledgement as fault clearance.

Connecting real aircraft or drones does not grant control authority. The viewer
must first support read-only world and vehicle state with clear source identity,
freshness, authentication, and capability reporting.

Any future remote actuation requires a separate adopted design covering:

- authenticated and authorized operator roles;
- signed commands, sequence validation, and replay resistance;
- explicit arming/control authority and handoff;
- safety validation that cannot be bypassed by the UI;
- two-person or supervisor approval where policy requires it;
- lost-link and local-autopilot precedence;
- complete audit records; and
- unambiguous simulation, HIL, and live-mode separation.

ADR-008's read-stream upgrade must not be used as an implicit command channel.

## 10. Data and protocol contracts

### 10.1 Presentation inputs

| Input | Viewer use | Viewer obligation |
| --- | --- | --- |
| `RuntimeSnapshot` | Current public platform, track, localization, mission, metrics, revision refs, event cursor | Treat as immutable; capability-filter truth |
| `WorldManifest` and chunks | CRS, axes, tiles, layers, hashes, zones | Verify, bound, and install a complete revision atomically |
| `BeliefRenderSnapshot` / tile patches | Observed mask, confidence, staleness, provenance, belief revision | Apply only to the declared base; resync on mismatch |
| Event journal | Consequential ordered decisions and outcomes | Preserve order and expose causality independently of dropped display frames |
| Command acknowledgements | Accepted/rejected state, reason, effective tick, revision | Resolve pending UI by command ID; never infer success from motion |
| Replay document | Long-lived public frame/event interchange | Honor schema compatibility and explicit unsupported states |
| Presentation preferences | Camera, layout, quality, units, keybindings | Keep local; exclude from deterministic state and canonical hashes |

A `SimulationCheckpoint` is never a renderer input or a replay substitute.

### 10.2 Identity and revision keys

Viewer state is correlated with typed IDs and the tuple:

```text
session_id / run_id / epoch / tick / world_revision / belief_revision / sequence
```

A new session or epoch clears interpolation, pending deltas, unresolved command
assumptions, and stale selections. Sequence gaps or belief base mismatches trigger
an explicit resynchronization state.

### 10.3 Coordinate handling

Runtime spatial calculations remain f64; scheduled source time is integer
nanoseconds, with derived seconds at interchange boundaries. One adapter converts to f32 for
Bevy using a scene-local render origin. The viewer always exposes the active CRS,
axis convention, origin, units, and altitude datum where relevant. New packages
follow the explicit glTF axis contract; legacy Z-up behavior remains versioned at
the loader boundary.

## 11. Viewer architecture

### 11.1 Process and thread model

- Bevy owns rendering, input, presentation ECS, and UI-local state.
- Interactive simulation runs in a dedicated `argusnet-sim` worker, not on the
  render thread.
- Commands use a typed bounded channel and receive acknowledgements.
- Published snapshots are immutable and may replace an unconsumed presentation
  frame; the event journal is separately ordered and lossless under its policy.
- World construction, tile decode, checkpoint restore, and recording I/O are
  cancellable/off-thread where applicable.
- The viewer must remain responsive while work is pending or faulted.

### 11.2 Presentation pipeline

Systems execute in explicit order:

```text
source input
  -> protocol/revision validation
  -> snapshot adapter
  -> persistent ECS apply
  -> render presentation
  -> picking/interaction
  -> UI composition
  -> diagnostics
```

Persistent entities update transforms and changed quality data in place. They
spawn/despawn only on membership change. Visibility toggles gate optional
presentation work but do not disable resource cleanup.

### 11.2a Display-time, frame and resource consistency

Choose one displayed source time and a compatible revision set before ECS apply.
Validate on workers, then apply bounded deltas without exposing a half-installed
required world revision. Large uploads are staged across frames; show the prior
valid revision/loading state until required resources are ready. Optional tiles
may arrive progressively with explicit unavailable regions.

Frame transforms are versioned domain outputs. Map alignment/localization
corrections must update poses, maps, trails and evidence through compatible
transform revisions; clear interpolation where that continuity is not known.
A render-origin shift only changes local presentation coordinates. It must not
appear as physical motion, change exported positions or contaminate sensor input.
Picking resolves an entity ID and the displayed revision; commands are checked
against current backend state rather than blindly executing an old screen pick.

Bound trail history, GPU uploads, decoded tiles, outstanding requests and retained
revisions by configured byte/work budgets. Cancel obsolete camera/epoch work and
reject late GPU/decode results. Quality reduction may defer cosmetic detail but
must preserve uncertainty, unknown regions and safety cues. Release assets only
when their readers are done, and expose memory pressure rather than growing
without limit. A desktop context/device loss rebuilds presentation resources from
a committed source; it does not reset the simulation.

Sensor imagery uses the dedicated capture path in
[simulation fidelity](SIMULATION_FIDELITY.md), not the operator viewport. Test
headless versus visible runs, origin shifts, map corrections, picking on old
revisions, lifecycle cleanup and interrupted world uploads.

### 11.3 Required module boundaries

The current large UI/render surfaces should converge on these responsibilities:

- `source`: replay, in-process worker, and live-stream adapters;
- `snapshot`: validation, revision state, interpolation eligibility, and ECS
  application;
- `presentation`: terrain/belief, platforms/tracks, sensors, zones, trails,
  trajectories, evidence, and selection outline;
- `interaction`: picking, camera, keybindings, focus, and local filters;
- `shell`: header, status line, rails, timeline, workspace routing, dialogs;
- `inspectors`: typed platform, map, event, POI/evidence, and diagnostics views;
- `commands`: local envelope creation, pending state, acknowledgements, and
  reason mapping; and
- `diagnostics`: renderer, worker, streaming, world transfer, queues, recording,
  and lifecycle counters.

No module may query simulation truth or mutate runtime state merely because it
renders or inspects it.

## 12. Failure, loading, and degraded states

The shell must define first-class presentations for:

- no source selected;
- loading manifest;
- loading/verifying world revision;
- world unavailable or incompatible;
- belief unavailable, stale, or awaiting base revision;
- reconnecting;
- live source lag or presentation drops;
- simulation paused or slower than real time;
- command pending, busy, rejected, or stale;
- recording backpressure, paused, incomplete, or disk failure;
- simulation fault requiring reset/restore;
- TLS/authentication/capability failure; and
- unsupported replay/protocol/package version.

Each state preserves the last coherent presentation when safe, names what is
stale, and offers a bounded next action. Errors are never encoded as empty frames,
zero terrain, zero zones, or fully confident state.

## 13. Accessibility and input

- All core operations are keyboard accessible and remappable.
- Focus state is always visible and does not depend on color.
- Status is conveyed with text and structure as well as color.
- UI text and essential map annotations meet WCAG AA contrast against their
  actual rendered background.
- The viewer supports UI scale independently of world scale.
- Reduced motion disables nonessential transitions.
- Alerts are ordered by operational severity and announced without repeating on
  every frame.
- Dense metadata remains readable at supported scaling; utility labels do not
  shrink below the tested minimum for the target display.

Default bindings retained or added:

| Input | Action |
| --- | --- |
| Space | Play/pause replay or submit pause/resume to simulation |
| Left / Right | Previous/next committed replay state |
| Home / End | First/latest available state |
| `[` / `]` | Previous/next event |
| `F` | Focus selected entity |
| `L` / `C` | Follow / chase the selected entity |
| `G` | Free-fly (`W A S D`, `E`/`Q`, `Shift`, right-drag to look) |
| `1`-`9` | Recall camera bookmark (`Ctrl`/`Cmd`: save) |
| Right drag | Orbit |
| Middle drag or Shift+right drag | Pan |
| Scroll | Zoom / scale movement speed |
| Left click | Select visible selectable entity |
| Escape | Clear transient mode, menu, or selection in that order |

Bindings do not fire while entering text.

## 14. Performance and resource budgets

On the named reference hardware and representative fixed workloads:

- warmed p95 render frame time is at most 16.7 ms;
- warmed p99 render frame time is at most 25 ms;
- no unexplained stall exceeds 100 ms;
- compatibility-mode p95 tick processing remains below its fixed dt at 1×;
- scheduled mode measures per-subsystem release-to-start, execution and
  completion latency against its own deadline; simulation overload reduces
  achieved speed rather than silently dropping physical work;
- catch-up work is bounded and simulation ticks are not dropped;
- selection, pause acknowledgement, and local layer changes remain responsive
  during world loading and recording;
- entity, material, tile, trail, snapshot, checkpoint, and native-handle counts
  remain within declared ceilings through long reload/seek/toggle runs; and
- world, belief, history, and recording memory have observable eviction or
  backpressure behavior.

CPU presentation time, GPU time, simulation step/task time, transport/native cost,
world-build time, shader warmup, and recording backlog are measured separately.
Lavapipe smoke validates shader execution, not interactive frame rate.

## 15. Quality tiers

Low, standard, and high quality tiers share semantic meaning. A lower tier may
reduce geometric detail, shadow quality, atmospheric effects, and trail density;
it may not remove safety geometry, unknown-space encoding, selection identity,
or essential uncertainty information.

Bloom is off by default. Fog, SSAO, shadows, color grading, and environment fill
must be justified against real scenes and scaled to scene span. Shader warmup and
material compilation are visible startup stages rather than hidden stalls.

## 16. Verification and acceptance

### 16.1 Functional acceptance

1. A platform selected in the world and fleet index resolves to the same typed
   identity and inspector.
2. Selection survives compatible updates and clears on removal or epoch change.
3. The inspector distinguishes actual, commanded, pending, accepted, rejected,
   and executing state.
4. Belief, localization, zone, trajectory, and safety overlays match the exact
   revisions named by the current snapshot.
5. Unknown/unavailable belief data is visible and never fabricated.
6. A safety event reconstructs proposal, geometry, constraints, decision,
   fallback, acknowledgement, and effective tick.
7. Replay, simulation history, and live history expose their different mutation
   semantics.
8. Frames may be skipped for presentation without losing durable mission or
   safety events.
9. Session/epoch change resets interpolation and stale command assumptions.
10. Live mode never exposes truth without capability authorization.

### 16.2 Rendering and lifecycle acceptance

- Fixed overview, close, low-pitch, split, and high-entity fixtures have reviewed
  baseline images and perceptual/semantic diff thresholds.
- Terrain/belief pixel probes agree with CPU queries at observed/unknown borders,
  tile edges, and LOD seams.
- Picking tests cover depth, hidden truth, transparent overlays, split viewports,
  same-label/different-kind IDs, camera motion, and despawn.
- Repeated seek, reconnect, reload, source switch, layer toggle, and world-revision
  replacement do not cause monotonic resource growth.
- The UI passes supported-width, scaling, keyboard, contrast, and reduced-motion
  checks.

### 16.3 End-state migration acceptance

From a clean installed build, an operator can:

1. open the viewer without Python, Cargo, runtime GLB baking, or source-tree path
   assumptions;
2. start an in-process Rust simulation and observe the coherent world;
3. pause, step, change speed, seek, branch, stop, restart, and record through
   acknowledged commands;
4. inspect mission, platform, belief, localization, safety, and event state;
5. open a replay through the same presentation contracts;
6. connect to a world-aware live endpoint and receive actual mission/world
   metadata with explicit resynchronization behavior; and
7. sustain the declared performance and memory budgets without losing events or
   presenting partial state.

## 17. Delivery sequence

The UI redesign follows architecture, not the reverse:

1. **Foundation:** publish font/license decision, palette/spacing tokens,
   semantic status vocabulary, fixed UI fixtures, and baseline screenshots.
2. **Snapshot presentation:** complete the typed snapshot-to-ECS adapter,
   persistent selection, ordered system sets, and revision invalidation.
3. **World language:** finish belief terrain/materials, platform/track glyphs,
   trails, sensor geometry, zones, trajectories, and occlusion-correct picking.
4. **Instrument shell:** implement the header, status line, mission rail, world
   canvas, typed inspector, and timeline against replay snapshots first.
5. **Interaction:** add camera modes, event navigation, layer presets,
   accessibility, keybinding/remapping, and low-clutter/diagnostic modes.
6. **Runtime integration:** bind simulation controls to the acknowledged command
   channel and immutable snapshots; surface loading, cancellation, faults,
   backpressure, and available history.
7. **World-aware live:** consume negotiated manifests, chunks, belief patches,
   mission state, epochs, sequence, and resync without adding remote actuation.
8. **Product workspaces:** expand Plan and Review into site/map authoring,
   inspection evidence, comparison, and reporting as their contracts land.
9. **Acceptance:** pass visual, lifecycle, performance, compatibility, packaging,
   truth-isolation, and operator-workflow gates before replacing the transitional
   interface.

## 17a. System feature handoffs

[VIEWER_FEATURE_INTEGRATION.md](VIEWER_FEATURE_INTEGRATION.md) expands this matrix
into concrete interactions, 3D projections, integrity/contingency presentation,
SIL/HIL qualification, client outputs, ML provenance and offline governance.
S0–S8 are system release gates; they do not replace P2/P4/P5 or F1–F7 IDs.

The [capability delivery plan](CAPABILITY_DELIVERY.md) assigns backend ownership,
prerequisites and acceptance for these product surfaces:

| Feature | Viewer responsibility | Backend contract |
| --- | --- | --- |
| F1 Mission scorecards | Review outcome/reason summary; Reporting export | Evaluation results with event/evidence references and unavailable metrics |
| F2 Failure scenarios | Scenario Studio schedules; Operate degraded status; Review recovery timeline | Versioned deterministic injections and mission fallback events |
| F3 Explanations | Platform/map inspectors show why a task/route was chosen, constraints and next replanning trigger | Typed decision chain, input revisions, reason codes and execution outcome |
| F4 Comparisons | Paired scenario/seed comparison in Review, setup in Scenario Studio | Reproducible evaluation manifests, distributions and failed-run counts |
| F5 Energy contingencies | Reserve/return feasibility, handoff and unmet tasks | Battery/planning/safety decisions; no UI-computed reserve authority |
| F6 Spatial memory | Site catalog and prior/current synchronized Review | Immutable site/map/evidence revisions, alignment uncertainty and provenance |
| F7 Adaptive inspection | Requested/achieved evidence, unmet criteria and revisit reason | Quality results and safely validated bounded revisit tasks |

Replay fixtures can support early shell delivery. Missing contracts display
unavailable states; UI work never invents confidence, evidence or explanations.
P2 owns presentation, P4 owns runtime/domain state and P5 connects worker/live
sources. The feature IDs do not renumber these migration packages.

## 18. Explicit non-goals

The next-generation viewer does not:

- own or duplicate fusion, planning, safety, mapping, localization, or mission
  execution logic;
- treat the visual world as planner authority;
- expose simulation truth in normal operator workflows;
- infer remote-control authorization from stream access;
- use replay frames as simulation checkpoints;
- require Python or exported GLB assets to start an interactive Rust simulation;
- hide missing data behind plausible graphics;
- present debugging implementation details as the primary operator experience;
- reproduce generic card dashboards, decorative HUDs, glassmorphism, neon
  cyberpunk effects, or component-library aesthetics; or
- sacrifice information meaning for a cinematic scene.

The target is a coherent mission instrument: typographically disciplined,
operationally explicit, snapshot-correct, and ready to incorporate real aircraft
without confusing visualization with authority.
