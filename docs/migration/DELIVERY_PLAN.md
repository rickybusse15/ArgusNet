# Remaining Migration: Delivery Plan

This plan covers every open migration phase. Product roadmap numbering in
[ROADMAP.md](../ROADMAP.md) is independent. Status is grounded in `84af500` plus
GitHub main `a7817d9`; a planned crate or API below is not yet shipping.

[Architecture and protocols](ARCHITECTURE_AND_PROTOCOLS.md) is the cross-cutting
contract. [Verification and rollout](VERIFICATION_AND_ROLLOUT.md) specifies gates.
Work package IDs are stable references for PRs and handoffs; they are not dates.
An implementation PR should normally close one package or a clearly identified
sub-slice, with its prerequisites and remaining work stated.

## 1. Baseline and scope

| Phase | Delivered | Still open |
| --- | --- | --- |
| 0 | Profile, meaningful work counters, replay parity/schema foundations, TLS guards and toolchain pins | Reproducible proto regeneration, frame-time budget, complete scanner triage and contract inventory |
| 1 | Bevy 0.19, shared dependencies/features, real headless renderer and CI smoke | Ongoing compatibility maintenance; no migration redo |
| 2 | Interpolation, typed selection, stage gating, HDR/bloom/fog/sky, persistent typed mesh markers and holographic shader | Full ECS snapshot path, terrain/belief material, sensor/zone/trail geometry, picking, camera/time/UI, visual regression and frame budget |
| 3 | Rust world and safety query/policy authority; native Python bridge; shared viewer sampling | Operational hardening carried below; procedural shaping remains Python and moves in P4.3 |
| 4 | Shared core/world types and native particle-footprint scoring | DTO completeness, estimator state/VIO, sensing/mapping/planning/mission, direct fusion composition, replay and Rust CLI |
| 5 | No in-process viewer simulation yet | Worker/snapshots/commands, world boot/LOD, live world protocol, rewind and removal of Python boot dependencies |
| 6 | No broad Python reduction yet | Caller-driven deletions, analysis packaging, entry-point ownership and documentation |

The accepted native kernel speedup is not evidence of a complete localizer port.
The current marker mesh shader is not a terrain shader. The current CI colored
pixel assertion is not image regression coverage. Phase 3 completion does not
mean procedural generation, the Python tick or GLB boot dependencies are gone.

## 2. Dependency graph and workstream coordination

```mermaid
flowchart TD
    G[Guardrails G0-G3] --> T[P4.0 Types and adapters]
    T --> H[P4.1 Runtime interfaces and stage trace]
    H --> L[P4.2 Localization state]
    H --> W[P4.3 World construction]
    L --> V[P4.4 VIO and pose]
    H --> S[P4.5 Sensing]
    H --> M[P4.6 Mapping]
    M --> P[P4.7 Planning and safety]
    V --> R[P4.8 Simulation and replay]
    S --> R
    P --> R
    W --> R
    R --> C[P4.9 Rust CLI and parity]
    G --> E[P2.0-P2.6 Graphics and controls]
    E --> I[P5.0-P5.4 Interactive runtime]
    C --> I
    I --> X[P5.5 Acceptance]
    X --> D[P6.0-P6.4 Python reduction]
```

Parallel work is possible after shared contracts are frozen, but integration
order stays explicit. Graphics owns renderer files; runtime owns domain crates;
protocol/DTO changes are reviewed jointly. Every PR names a responsible owner
and a reviewer for affected consumers. This does not require multiple agents
or concurrent edits to shared files.

Before a work package starts, publish a short handoff: base commit, package ID,
input/output contract versions, files/crates affected, test command/fixtures,
known limitations and downstream consumers. Interface-changing work includes a
sample payload and migration example. Update this plan's package status and
STATE_OWNERSHIP when a slice lands; do not mark an entire phase complete because
one new crate exists. Resolve semantic questions through an ADR instead of
inconsistent Slack/comment-only agreements. Reports must distinguish locally
verified, pushed, PR checks passing and merged.

## 3. Guardrails before larger ports

| ID | Deliverable | Exit evidence |
| --- | --- | --- |
| G0 | Inventory replay/proto/Python/core/viewer field parity, enum semantics and unsupported schema versions; add installed-wheel and clean-checkout contract jobs | Every payload is preserved, normalized explicitly or rejected; no silent missing-field success |
| G1 | One pinned proto regeneration command and CI no-diff check; compatible `grpcio-tools`/protobuf dependencies | Python generated bindings regenerate from committed proto; Rust generation still builds; no hand-edited generated files |
| G2 | Representative renderer fixtures and warm-up-aware CPU/GPU frame instrumentation; split software smoke from hardware timing | Baseline artifacts for overview, close view, low pitch, split mode and high entity count, with hardware/profile/viewport recorded |
| G3 | Triage informational scanners, retain pins and branch checks, add artifact/provenance checks | Findings resolved or documented with owner/expiry; required checks enabled individually after triage, never blanket ignored |

G0/G1 precede the first wire change. G2 precedes turning expensive effects on by
default. G3 can run alongside early ports but is a release dependency. Re-profile
a representative full run after each major port: the original 40% localization
figure predates native scoring and must not be treated as today's allocation.

## 4. Phase 2 — complete the viewer engine

### P2.0 — snapshot adapter and persistent presentation state

Replace per-frame `CurrentRuntimeMarkers` reconstruction/string cloning with a
single snapshot-to-ECS adapter. Keep typed platform/track/truth IDs and add typed
selectability/trail state. Stable IDs survive pause, seek, filtering and live
reconnect only within a valid run/epoch. Spawn/despawn on membership changes;
update transforms/quality only on changed data. Clear selection on removal and
invalidate incompatible interpolation when epoch/world changes.

Inputs: current replay adapter first, future `RuntimeSnapshot` later. Outputs:
persistent domain entities and versioned renderer resources. Use explicitly
ordered SystemSets (input, snapshot apply, presentation, picking, UI) and gate
optional work by visibility/dirty revisions. Keep resource cleanup systems active
when their overlays are off. Test entity/asset counts across long seek/reload
loops and compare allocation counts before/after. Do not replace `argusnet-sim`
with an ECS-owned second world.

### P2.1 — terrain and belief material

Author an embedded WGSL material using Z-up elevation/slope bands, anti-aliased
contours and land-cover tint. Supply a typed `BeliefRenderSnapshot` containing
exact grid origin/resolution/dimensions, observed mask, confidence, last-observed
tick and belief revision. Sampling/packing lives in one adapter; shaders MUST
NOT infer coverage from truth terrain color or distance to a drone.

Begin with a synthetic belief fixture and replay fields that actually exist.
When a replay lacks belief data, render an explicit unavailable/unknown state;
do not fabricate planner confidence. P4.6 supplies full live dirty-tile patches.
Separate class/observed discrete channels from confidence/time numeric channels;
use nearest sampling for categorical values, documented interpolation for other
channels and halo/edge handling across tiles. Upload only changed tiles, cap
texture dimensions/memory and expose failed/unavailable uploads. Validate pixel
probes against CPU belief queries at boundaries, stale/unknown cells and LOD
seams. Contour style is presentation, never another terrain evaluator.

### P2.2 — sensor, mission-zone and trail geometry

Keep the existing hologram material, but move sensor FOV into cone/frustum meshes,
mission zones into extruded boundary/volume geometry and trails into persistent
ribbons. Physical FOV/range volumes remain world-sized; platform glyph-like meshes
retain screen-size readability. Derive sensor axes from the same sensing pose
and limits as the runtime. Unknown/unset range keeps adapter semantics.

Cache topology/materials by sensor/zone shape; update transforms/parameters on
change. Trail sampling is time/distance-based with a bounded memory budget,
restart on teleport/epoch changes, and shrink/remove when IDs disappear. Define
transparency ordering, depth-test behavior and selection exclusions deliberately.
Use LOD/instancing only where measured instance counts justify it. Review visual
appearance at low pitch and extreme zoom; no effect may enclose the camera and
make the world invisible. Move remaining diagnostic gizmos behind a toggle once
an equivalent readable mesh path exists.

### P2.3 — picking, hover and focus identity

Replace the 18-pixel proximity picker with mesh raycasting filtered to visible,
selectable entities in the active viewport. Terrain/buildings must occlude picks;
transparent sensor volumes, trails, sky and zones must not steal a platform pick
unless explicitly selected in that interaction mode. Decide pick proxy geometry
for thin platform arms; it must be bounded by the visible glyph footprint.

Add hover feedback and a selection outline/material pass independent of the
normal hologram material, with no through-terrain outline by default. UI-captured
pointer input does not pick the world. Tests cover depth ordering, hidden truth,
split viewport coordinates, camera movement, despawn and same-label/different-kind
IDs. Inspect both pointer and keyboard selection workflows.

### P2.4 — camera and time controls

Retain orbit and add damped free-fly, selected-entity follow/chase, F-to-focus and
1-9 bookmarks. Camera controls use render delta time, not simulation dt. Scale
speed to altitude/scene size, clamp pitch/far planes for the Z-up world and apply
movement within the focused viewport. Follow loss returns to a stable camera
rather than snapping to origin. Remappable bindings cannot fire while typing.

Use time-based scrubbing over irregular replay timestamps, with interpolation,
loop and jump-to-event. Pause/step acts on the runtime command channel in Phase 5;
replay mode implements the same UI affordances locally. Display unavailable seek
ranges and live/read-only history separately from simulation rewind. Save camera
bookmarks as presentation preferences, not deterministic runtime state.

### P2.5 — authored C2 UI and quality tiers

Specify palette tokens, typography/license, spacing, semantic colors and readable
contrast. Status must not rely on color alone. Provide a keybinding reference and
conflict-checked remapping. Avoid exposing transport/kernel implementation details
in operator flows; show actionable loading, stale-world, paused, recording and
fault states. Keep debug metrics in a diagnostic panel.

Evaluate environment fill, SSAO and color grading on actual scenes. Define low,
standard and high quality tiers with shared meaning; software CI can use a known
supported tier. Shadows and fog scale with camera/scene span. Do not enable every
post-process by default merely because Bevy supports it. Material compilation,
asset warmup and glyph visibility are part of visual QA.

### P2.6 — graphics acceptance

Run fixed replay/world fixtures through the real renderer in all quality tiers.
Publish visual diffs and timing/allocation evidence. Exit: runtime entities and
terrain/belief are readable at overview and close zoom; controls/picking work;
gizmos are optional diagnostics; no lifetime leak; designated hardware meets the
frame budget. Lavapipe smoke remains a shader correctness gate, not a 60 fps gate.

## 5. Phase 3 — retain one authority while extending consumers

No second terrain/safety engine is planned. Add ABI capability negotiation,
registry concurrency/lifetime tests and native error containment as new consumers
arrive. Preserve Python metadata/re-exports until callers migrate. Fix any newly
found geometry error with a minimal parity fixture and explicit baseline review.
Move procedural shaping in P4.3, not by declaring the existing Python generator
an offline-only tool before the Rust runtime can construct its own worlds.

## 6. Phase 4 — finish the Rust simulation runtime

### P4.0 — complete shared types and adapters

Start from `argusnet-types`, not another value crate. Complete the G0 field matrix;
extract rich replay DTOs/normalizers into `argusnet-replay`; keep runtime types free
of Bevy/prost. Viewer conversion to f32 happens once. Consolidate proto conversion
logic while preserving efficient ownership: move owned strings/vectors when owned,
borrow/clone only where needed, and share field policy instead of doubling it.

Add runtime/DTO conversion round trips covering all optional payloads and invalid
covariance lengths. Python-to-Rust and Rust-to-Python replay fixtures remain green.
No proto field is removed, renumbered or reinterpreted. Exit: one semantic
runtime type definition, documented compatibility DTOs, complete field ownership.

### P4.1 — simulation API, trace and deterministic execution substrate

Introduce a small `argusnet-sim` shell with config validation, integer clock,
explicit stages, command application and snapshot interfaces. Keep it runnable
with fixture sources; do not present it as the production replacement yet. Capture
Python stage traces for default, scan-map-inspect, opt-in safety, VIO, frontier,
POI and egress paths. Identify every mutable controller/cache and future-tick
side effect before moving code.

Implement deterministic keyed seed derivation and RNG/math conformance fixtures
before changing stochastic stages. See verification policy: choose/pin algorithm
and normal sampler in a dedicated ADR-backed PR. Caller-provided fixed offsets
remain a deterministic test input, not a permanent second RNG authority. Exit:
trace harness, reset semantics, fault behavior and snapshot headers are testable
without viewer/network/file output.

### P4.2 — full GridLocalizer state

Move estimates, per-platform step counts, timeout state, particle generation,
annealing, weighting, confidence and inverse-variance fusion to
`argusnet-localization`; reuse native coverage scoring. First validate all
non-random formulas using recorded offsets. Then migrate PRNG explicitly under
ADR-006 with predeclared statistical envelopes. Preserve reset behavior, no-map
early returns, elevation handling and duplicate/missing platform semantics.

Expose a transitional native estimator handle so the Python tick uses Rust state
while other stages remain Python. Remove the Python estimator math after parity;
retain only config/result adapters. Add `LocalizationStatus` to the runtime/DTO
contract so timeout, unavailable map and actual convergence are distinguishable.
Keep legacy forced confidence only under the compatibility profile until a
separate mission-gating change is reviewed. Re-profile full localization and
record ABI/state-transfer cost rather than repeating the scoring-only benchmark.

### P4.3 — runtime world construction

Port procedural height shaping, smoothing/detail, hydrology/semantic masks,
seasonal land cover and obstacle construction into `argusnet-world` builders.
Parameter/preset data becomes a versioned data asset consumable by Rust and
Python tooling. Validate units/ranges and generation version; use one seed scheme.
Batch terrain and categorical construction; avoid serializing full grids during
each builder stage.

DEM/GIS decoding may stay in offline import tools, but the Rust runtime accepts
a documented preprocessed world bundle directly. Pure synthetic defaults MUST
need neither Python nor GLB baking. World builds have cancellation/progress,
content hashes and bounded caches. Tests cover all presets, tile seams, missing
coverage and exported metadata consistency. Exit: representative worlds construct
in Rust with recorded startup timing and no Python process.

### P4.4 — VIO and pose pipeline

Port odometry/VIO/GNSS fusion, relevant transforms and implemented relocalization
or pose-graph behavior into the localization crate. Inventory which helper
modules are actually wired before porting stubs. Separate synthetic IMU/features
from the estimator so recorded/real observations can use the same estimator.
Preserve frame transforms, covariance units, initialization, reset and pose
substitution into sensing/fusion.

Test zero motion, turning, timestamp gaps, bias drift, GNSS loss/recovery,
singular/invalid covariance and replayed observation fixtures. Define invalid or
stale pose behavior explicitly. No manufactured confidence on a failed estimator;
legacy behavior changes require separate validation. Exit: Python delegates the
wired estimator path and statistical/numerical envelopes pass.

### P4.5 — sensing and versioned sources

Port SensorModel, FOV/range/elevation gates, bias drift/noise, LOS/transmittance,
clutter behavior actually used and the rejection cascade into `argusnet-sensing`.
Preserve the newly merged `argusnet.observation-source/v1` compatibility boundary:
its Python request includes synthetic truth and RNG, but new generic Rust ingest
must not require truth. Define a synthetic request carrying restricted truth and
a separate observed-data source that produces the same batch contract.

`ObservationBatch` records attempted count, accepted observations, generation
rejections and counts by node/target/reason. Generation rejection remains distinct
from fusion rejection. Sources declare capabilities and version; unsupported
capabilities fail before the run. Test reordered nodes/targets, missing sensors,
weather/vegetation, threshold edges and occlusion-aware truth isolation. Exit:
Python generator becomes an adapter and acceptance/rejection envelopes pass.

### P4.6 — coverage and belief mapping

Move mutable CoverageMap/WorldMap state into `argusnet-mapping`. Introduce one
footprint rasterizer whose indices/weights feed both coverage and belief updates,
while preserving their distinct statistical meanings. Reuse the same grid layout
as localization. Replace repeated full-grid stats with revision-aware aggregates;
prove incremental totals against full reductions, including reset and removal.

Publish immutable belief revisions and dirty tile patches for planners and P2.1.
Carry observed/unknown, uncertainty, timestamp and evidence provenance. Define
out-of-bounds/missing-tile behavior and deterministic update order when multiple
sensors touch a cell. Test commutative updates where promised, deterministic
ordered updates where not, tile seams and no-truth-access. Exit: mapping is
load-bearing in the transitional Python loop and shared with the renderer.

### P4.7 — planning, battery, deconfliction and mission intents

Port visibility-graph routing, frontier claims, POI assignment/dwell, battery
accounting/reserves, egress and cooperative deconfliction as small modules.
Planner input is belief/pose/mission constraints; output is a proposal with
world/belief/pose revisions, validity interval and reason. Do not let a planner
apply transforms. Cache routes against actual input revisions and quantized
request keys; never reuse a route across a changed exclusion zone blindly.

Integrate the existing Rust safety engine after deconfliction. Keep mission
policy, hard physical limits and simulation collision correction distinguishable.
Test crossing routes, low battery, unreachable POIs, stale/unknown belief,
coordinator/claim changes and conflicting overrides. Log proposed/accepted motion
and clamp reason. Exit: ordering matches trace fixtures and the mission loop no
longer recomputes multiple identical routes per platform per tick.

### P4.8 — compose tick, mission state and replay

Replace Python orchestration with the P4.1 stages and explicit mission state.
Consume typed events once; preserve scan/localize/inspect/egress transitions and
the implemented mission-executor hooks without claiming the entire product
mission architecture is complete. Eliminate repeated dataclass-style list rebuilds;
freeze once at the snapshot boundary. Call `argusnet-core` directly.

Add complete checkpoints and a versioned Rust replay writer. Python evaluation
reads its output through strict schema validation. Compare direct-core results
with the gRPC adapter using identical fixture inputs. Export failures and tick
failures have distinct states. Exit: a full representative run is owned by Rust,
with no Python callbacks inside a tick, and all relevant fields survive replay.

### P4.9 — Rust CLI and phase acceptance

Add `argusnet-cli` with a Rust `argusnet sim` command using the same library as the
viewer/server. During coexistence package it under an unambiguous development
binary such as `argusnet-rs`; do not let pip and Cargo overwrite each other's
`argusnet` executable. Versioned config files are the canonical input; adapters
translate legacy CLI flags with explicit unsupported-option errors.

Run the canonical scenario suite, edge profiles and multi-seed statistical tests.
Emit resolved config/version/provenance and record startup/tick distributions.
Preserve the frozen gRPC contract fixture unless its own contract actually
changes; new simulation goldens do not automatically require changing transport
fixtures. Exit: Rust CLI reproduces supported scenarios within approved envelopes,
no Python runtime dependency, direct fusion parity and meaningful perf gates.

## 7. Phase 5 — run the world in the viewer

### P5.0 — worker and command integration

Link `argusnet-sim` into the viewer with the worker/command/snapshot protocol.
Wire Run, pause/resume, speed, step and stop to acknowledged commands. World
preparation is cancellable and off the render thread. Loading a second scenario
must release old workers/handles/assets only when readers are finished. Invalid
configuration remains editable; a failed load must not show a half-new world.
Test rapid Run/Stop, repeated reload, queue saturation and close-during-load.

### P5.1 — Rust world boot and terrain LOD

Mesh terrain directly from immutable `argusnet-world` views. Use camera-driven
LOD with skirts/stitching and deterministic tile bounds; scheduling/upload can be
asynchronous, but world revision publication is atomic. Cache by content hash,
LOD and render-origin transform; cap CPU/GPU memory and evict unused tiles.
Near terrain must not disappear while a coarse tile is being replaced.

`.smartscene` remains import/export, not required boot state. Keep support for
legacy Z-up packages while new exporters follow the explicit glTF axis contract.
Test synthetic boot with no Python executable and imported preprocessed worlds
with no exporter installed. Raster/mesh sampling parity is tested at LOD seams.

### P5.2 — time, checkpoints and recording

Implement bounded snapshot/checkpoint history, deterministic seek/replay and
branching after new input. UI uses available-history bounds and reports recording
backpressure/failure. Test equivalence of uninterrupted execution and
checkpoint-restore-to-same-tick, including safety escalation, RNG, pending
commands, VIO and mapping. Rate changes must not change final deterministic state.
A live external stream offers historical viewing, not reversal of external state.

### P5.3 — world-aware live protocol

Implement ADR-008 additively: capability negotiation, world manifest/chunks,
revisioned frames/belief patches, resync and bounded retention. Update Python
bindings and Rust conversion/server/viewer together. Store mission zones/status
at the server authority; eliminate the hardcoded empty zone response. A zones
message definition alone does not close this package.

Test old V2/new client combinations, reordered/missing chunks, corrupt hashes,
restarts, replayed sequences, base-revision gaps, slow readers and non-loopback TLS.
Do not add remote control implicitly. Exit: live mode renders a coherent world
and actual mission state rather than floating markers with absent metadata.

### P5.4 — remove subprocess/runtime export path

Delete `simulation_python`, `spawn_simulation_child` and
`spawn_build_scene_child` after P5.0/P5.1 pass. Remove Python daemon auto-build from
library imports and replace temporary-port probing with server-selected port 0
plus an explicit readiness/address handshake where local daemon launching remains
needed for standalone clients. Rust viewer startup must not invoke Python, Cargo
or GLB export. Preserve offline commands under explicit analysis/export tooling.

### P5.5 — interactive acceptance

From a clean installed build, press Run on the designated performance scenario,
observe advancing world state at the chosen fixed dt and responsive 60 fps render
on reference hardware. Inspect the process tree: no Python descendant; unrelated
system Python processes are irrelevant. Verify pause/step/rewind, recording,
world/mission overlays, picking, camera modes and restart under load. Record
render CPU/GPU timings separately from sim ticks and catch-up lag. Exit includes
memory stability and graceful cancellation, not only a screenshot.

## 8. Phase 6 — reduce Python without breaking analysis

### P6.0 — caller inventory and deletion map

For every candidate module, list runtime callers, analysis/export callers, tests,
CLI entry points and public imports. Port used behaviors; delete unused stubs
with evidence. Resolve duplicate trajectory builders by behavior name and test,
not by keeping whichever file is shorter. Do not remove a whole directory while
an exporter or evaluator still imports a useful data type from it.

### P6.1 — pure analysis/config packaging

Separate lightweight configuration/replay value readers from the native bridge.
Python evaluation, reporting, GIS/export and benchmark sweeps must install and
import without building Rust when they do not request native functionality.
Retain optional native world/simulation tooling as a separate extra/package only
if there is a demonstrated caller. Publish supported wheel/platform policy and
an explicit source-build path; test sdist completeness.

### P6.2 — entry-point ownership and deprecations

Rust owns the production `argusnet` executable. Python provides
`argusnet-analysis` (and `python -m argusnet` compatibility where feasible) for
evaluation/export. Maintain a command migration table for every existing CLI
subcommand, flags, output format and exit code. A compatibility wrapper may
forward an explicit runtime request to an installed Rust binary; it MUST NOT
build the binary or search hardcoded interpreter paths. Unknown commands and
missing binaries fail with an installation instruction.

Deprecations require release notes and at least one documented compatibility
release where consumers exist. If an API has no supported external users and all
repo callers migrate atomically, record that evidence rather than keeping a stub
forever. Check Python exports/`__all__`, package data and schema paths after removal.

### P6.3 — module removal and documentation convergence

Delete migrated implementations in world/simulation/sensing/mapping/localization/
planning/safety as their last runtime callers disappear. Relocate exporter-only
GLB/GIS code first. Remove empty adapters/UI/indexing stubs only after import
inventory. Shrink mypy ignore lists and remove unused dependencies, dev tools and
orphan Rust/native compatibility paths. Retain one CLAUDE/AGENTS source.

Rewrite README module tree, installation, usage/CLI table, VIEWER_UI, architecture,
STATE_OWNERSHIP, KNOWN_GAPS, performance docs, changelog and release notes against
the shipped system. Preserve historical ADRs; supersede them explicitly where
needed. No "implemented" label may describe a proposed interface.

### P6.4 — migration acceptance and maintenance handoff

Test Rust-only runtime installation, Python-only analysis installation, supported
legacy replay/scene import, exports in a conformant glTF consumer, and upgrade
from the compatibility release. All required CI gates pass from a fresh checkout.
Publish final ownership/dependency map, before/after performance artifacts,
compatibility matrix and remaining product-roadmap gaps. Archive migration
checklists only after the release contains the work; a merged port is not an
installed release by itself.

## 9. Immediate execution queue

1. Land/review the current Phase 3 plus Phase 2/4 slices and this plan; require
   branch CI before merge. Do not overwrite main to publish a plan.
2. G0/G1 contract inventory/regeneration and G2 renderer measurement fixtures.
3. P4.0 DTO completeness plus P2.0 snapshot adapter (shared ID/revision agreement).
4. P4.1 stage trace/seed contract, then P4.2 estimator state; independently P2.1
   terrain/belief material using declared unavailable states until P4.6 exists.
5. Continue the dependency graph; use measured profiles to prioritize within
   available packages, never to skip a compatibility or truth-isolation gate.
