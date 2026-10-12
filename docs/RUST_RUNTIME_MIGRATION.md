# Rust Runtime Migration and Game-Engine Viewer

## Reviewed state and target (2026-09-23)

The reviewed code baseline is `ec9c2f7` (including main through `45390b8`), not a
claim about the latest remote or release. Rust stage bodies, direct fusion,
belief publication and replay writing exist. Command recovery/dedup and format-2
control-state checkpoints exist locally in this baseline. Complete production
composition, subsystem reset/restore, full causal checkpoints and viewer-worker
acceptance remain open. No scheduler or aircraft qualification is claimed.

**Update 2026-10-03 (`main` at `0b1032c`).** P4.8d to P4.9d and viewer P2.0 to
P2.3 are merged (#122, `badb1e1`): the controller stage, world-backed stage 1,
the scenario bundle and native builder, the mission driver, the causal-state
audit of every assembled stateful stage, and an `argusnet-rs` CLI that builds
and runs all seven canonical scenarios with no Python (six within every
declared envelope; see [migration status](migration/STATUS.md)), owned
subsystem state in every stage (A1.2) and a production constructor that
rejects a missing required stage by name (A1.3). Reset equal to fresh and full
restore (A2), a durable event journal with crash recovery (A4.1), 30-seed
statistical qualification (A3.2) and the viewer's in-process simulation worker
(D1.1, D1.2) are merged too. Still open: the independent scheduler (P4.10:
scheduler not yet wired into `Simulation`, pending-job restore), cross-platform
determinism (A3.3) and the rest of D5 on the Python side (daemon auto-build on import). The viewer itself runs and builds scenes with no Python (D2, D5). No scheduler or
aircraft qualification is claimed.

The [Rust-only runtime target](RUST_RUNTIME_ARCHITECTURE.md) now specifies
independent sensor/task deadlines, one session owner, timestamped measurements,
bounded workers and separate virtual/live time. P4.10a–f adds this after P4.9;
P5.0 basic controls ships alongside it. [Hardware integration](HARDWARE_INTEGRATION.md)
remains a separate S3–S7 qualification track.

## Historical integration status at `ba163ac`

This plan now includes the merged P4.3g landscape builders, P4.6d belief
publication and P4.8a–c mission/world/stage work. Evidence is recorded in
[landscape builders](migration/LANDSCAPE_BUILDERS.md),
[belief revisions](migration/BELIEF_REVISIONS.md),
[mission state](migration/MISSION_STATE.md),
[tick world](migration/TICK_WORLD.md) and [stage bodies](migration/TICK_STAGES.md).
Rust can publish populated platform/track snapshots from `TickWorld`; it is no
longer accurate to describe all snapshots as necessarily empty. The default
constructor still registers no domain handlers. Complete production scenario
composition, remaining controllers, full causal checkpoints, replay/CLI
qualification and viewer integration remain acceptance work. Existing slice
test reports are not a fresh full-system test result.

## Detailed migration plan

The remaining migration is governed by these companion documents:

- [Delivery plan](migration/DELIVERY_PLAN.md): work packages, dependencies,
  consumer handoffs and exit criteria for partial Phases 2/4 and remaining 5/6.
- [Architecture and communication protocols](migration/ARCHITECTURE_AND_PROTOCOLS.md):
  crate ownership, truth/belief access, tick order, commands, snapshots,
  checkpoints, streaming, native ABI and versioning.
- [Verification and rollout](migration/VERIFICATION_AND_ROLLOUT.md): deterministic
  seeds/math, statistical qualification, test matrices, performance/visual gates,
  compatibility, release/rollback and open calibration decisions.
- [ADR-007](adr/007-runtime-composition-and-snapshots.md) and
  [ADR-008](adr/008-world-aware-streaming-and-compatibility.md) record the adopted
  architecture. Accepted design does not mean the planned interfaces exist yet.

These documents supersede the abbreviated ordering below where more specific.
They do not change the separate product roadmap's phase numbering.

## Current reading guide

The historical milestone narrative below records earlier implementation states;
use [STATE_OWNERSHIP.md](STATE_OWNERSHIP.md) for the reviewed checkout inventory.
Localization, sensing, mapping and planning libraries now exist, but P4.8 complete
mission composition, world assembly and revisioned belief ownership remain open.
Proto regeneration and field-parity checks are present in CI. Do not restart
completed library ports or infer full runtime readiness from their existence.

The [next refactor review](NEXT_REFACTOR_REVIEW.md) adds command/checkpoint
correctness and physical-confidence gates to existing packages. The
[viewer specification](VIEWER_UI.md) governs P2 presentation and P5 integration;
[capability delivery](CAPABILITY_DELIVERY.md) adds F1–F7 product slices without
renumbering migration phases. The viewer is a consumer of state, never a second
simulation, map or mission owner.

## Historical milestone record

## Status after Phase 3

Implementation snapshot: `8fbf7c2` on `codex/finish-phase3`, based on GitHub
`main` at `5f86094` (the latest fetched main when Phase 3 was completed).
Phase 3 is implemented and verified on this branch; merge/release status must be
checked in GitHub rather than inferred from this implementation snapshot.
This plan uses **migration phases**, distinct from the product capability phases
in [ROADMAP.md](ROADMAP.md).

The governing rule remains **one authority per concern**. Rust owns world
queries, safety policy and fusion. Python still drives the simulation; the
viewer still loads batch replay/scene artifacts. Completing Phase 3 does not
complete the real-time runtime migration.

## Subsequent implemented slices

The branch now also incorporates GitHub main `a7817d9` (PR #43 observation-source
contract). Phase 2 has persistent typed mesh markers, a holographic material and
a marker-aware CI smoke fixture. Phase 4 has begun: shared core/world value types
live in `argusnet-types`, and the hot GridLocalizer footprint-scoring batch runs
in `argusnet-mapping` through the native bridge. See the
[implementation and measurement report](performance/phase2-phase4-first-slices.md).
Neither phase is complete; the simulation loop and estimator state remain Python.

## Historical Phase 3 system state

| Concern | Current authority | Remaining transition |
| --- | --- | --- |
| Terrain evaluation, tiles/LOD/mips, obstacles/index, land cover, LOS/detection | `argusnet-world` | Move remaining procedural world generation from Python before the viewer can boot without Python. |
| Safety limits and simulation safety policy | `safety-engine`, invoked through `argusnet-native` | Keep opt-in blocking and clamp ordering while porting the tick loop. |
| Fused tracks and fusion health | `argusnet-core`, served by `argusnet-server` | Replace the simulation's per-tick gRPC call with a direct Rust library call. |
| Scenario/preset configuration and procedural scene shaping | Python | Port runtime construction; retain analysis and exporter-facing configuration where useful. |
| Sensing, mapping, GridLocalizer/VIO, planning, mission state and replay assembly | Python simulation | Phase 4. |
| Shared runtime value types | `argusnet-types` for core/world/proto runtime values; Python and viewer DTOs remain | Finish replay DTO alignment and conversion consolidation. |
| Interactive and headless rendering | Bevy 0.19 viewer | Finish Phase 2 materials, geometry and operator controls; integrate simulation in Phase 5. |
| Viewer terrain height queries | `argusnet-world::TerrainRaster` | Mesh runtime terrain directly in Phase 5. |
| Scene/GLB export, GIS and offline evaluation | Python | Keep as offline tooling; remove export from the interactive boot path. |

Current execution path:

```text
Python simulation tick
  -> in-process argusnet-native -> argusnet-world / safety-engine
  -> blocking gRPC -> argusnet-server -> argusnet-core
  -> replay JSON + Python scene/GLB export -> Bevy viewer

Bevy viewer terrain queries -> argusnet-world::TerrainRaster
```

The native bridge owns persistent world handles and uses batched f64 buffers for
terrain queries. Source installation builds the library; wheels include it.
There is no Python evaluator fallback or runtime Cargo auto-build. World handles
hold snapshots: replace layers through `EnvironmentModel` to update the world
and invalidate caches; mutating exported NumPy arrays is not a runtime update.
The compatibility re-exports in `environment.py` remain available.

## Earlier phase status and completion plan

| Migration phase | Status | Next action / exit |
| --- | --- | --- |
| 0 — Measurement, contracts and guardrails | Foundation landed; verification gaps remain | Retain strict replay and deterministic work gates. Add a protobuf regeneration comparison and a steady-state viewer frame budget bench; neither was verified as present in this checkout. |
| 1 — Bevy 0.19 and one renderer | Landed | Keep real headless Bevy smoke coverage. CPU rasterization is no longer the CI render path. |
| 2 — Viewer engine, visuals and controls | Partial | Preserve interpolation, typed selection, ordered/gated systems, HDR/bloom/fog/gradient sky and screen-scaled camera-facing symbols. Persistent typed marker meshes and the holographic material now exist. Finish terrain/belief material, sensor/zone/trail geometry, picking, camera/time controls and authored UI. Exit with gizmos diagnostic-only, image regression coverage and a measured frame budget. |
| 3 — One world authority | Implemented and verified locally | Review/merge the branch containing `8fbf7c2`; no remaining Phase 3 implementation items identified. See the evidence below. |
| 4 — Rust simulation runtime | In progress: shared types and localization scoring | Finish shared DTO alignment and localization state, then decompose the loop into library crates. Rust CLI must reproduce scenarios within documented statistical envelopes. |
| 5 — Real-time viewer | Depends on Phases 2 and 4 | Link simulation in-process, fixed timestep, pause/step/speed/rewind; remove Python subprocesses and runtime GLB baking, add live world metadata. Verify sustained render performance. |
| 6 — Python reduction | After runtime consumers migrate | Delete migrated runtime modules/bridge paths as callers disappear; retain evaluation, reporting, GIS and export tooling. Update ownership and usage docs. |

Phase 2 can continue independently of Phase 4. Phase 3's Python wrappers are
transitional API/metadata adapters, not independent terrain or safety engines.

### Phase 3 delivered

- Renamed `terrain-engine` to `argusnet-world`; both the world crate and
  `safety-engine` now have shipping consumers.
- Centralized obstacle geometry/indexing, analytic terrain evaluation, tiled
  bilinear sampling/mip pyramids, land-cover sampling and two-level DDA LOS with
  vegetation attenuation and detection probability.
- Added owned native handles and Python bindings, and removed
  `src/argusnet/safety/checker.py`. Rust safety policy preserves the existing
  simulation telemetry contract, opt-in blocking and clamp ordering.
- Removed the viewer's independent height sampler; replay rasters use the world
  crate's interpolation kernel.
- Added cross-language analytic fixtures and tests for collision/terrain/LOS
  semantics, packaging, handle lifecycle and safety integration.

A verified reverse-direction DDA correction changes occlusion and therefore some
observation counts. The deterministic work goldens were explicitly re-baselined,
not widened. This follows [ADR-006](adr/006-determinism-rebaseline.md); runtime
ownership follows [ADR-005](adr/005-rust-runtime-authority.md). These are the
actual ADR numbers, superseding the earlier draft plan's numbering.

### Phase 4 implementation order

1. Extract `argusnet-types` and consolidate core/proto/viewer conversions while
   preserving replay and protobuf compatibility.
2. Port `GridLocalizer` first using the Phase 0 profile (about 40% of that measured
   tick loop), then remaining terrain construction costs and VIO. Reuse
   `argusnet-world`; terrain evaluation and sampling are already migrated.
3. Build `argusnet-sensing`, `argusnet-mapping` and `argusnet-planning` around
   explicit inputs/state. Merge duplicate coverage footprint rasterization and
   avoid repeated full-grid reductions and redundant route planning.
4. Compose `argusnet-sim` from scenario construction, dynamics, sensing, mapping,
   planning, mission and replay stages. Preserve safety/deconfliction ordering;
   do not transcribe the monolithic Python function or repeated state rebuilding.
5. Specify the Rust PRNG and stable seed derivation under ADR-006 before changing
   stochastic behavior. Compare acceptance/rejection distributions and track
   error against Python; retain frozen cross-language transport/replay fixtures.
6. Expose a Rust `argusnet sim` entry point and call fusion directly in-process.
   Compare scenario throughput and statistical parity before removing Python
   runtime consumers.

Removing gRPC is an architectural prerequisite for Phase 5, not the primary
performance case: Phase 0 measured it at 5.9% of tick time.

### Phase 5 dependencies to keep explicit

`simulation_python`, `spawn_simulation_child` and `spawn_build_scene_child`
are gone (D2, D5): the viewer builds scenes and runs Tools → Scenario
simulations through `argusnet_sim` as a library, and a test fails if any viewer
source starts a process. Generate terrain meshes from the world
crate with LOD; replay JSON and `.smartscene` become export/interchange artifacts.
Live mode also needs additive world/zone metadata and real mission status.
Fixed-step simulation, snapshot rewind and rendering must have independent clocks.

## Verification and measured limits

For `8fbf7c2`: 192 Rust workspace tests passed; 940 Python tests passed with
10 skipped and 65 subtests. Replay-only viewer tests: 93 passed. Workspace build,
Clippy, formatting, Ruff, mypy and benchmarks passed. Editable installation and a
wheel were exercised, including native safety and packaged schema loading. A real
headless Bevy still visibly showed terrain and geometry. Archived-commit tests
confirmed the new world fixtures were tracked.

Local measurements: 10,000 analytic height queries improved from 197.70 ms to
0.915 ms; 257x257 tiled construction from 42.80 ms to 24.47 ms. Representative
mean simulation tick time was effectively unchanged (4.00 ms to 3.94 ms).
This is a world-query improvement, not evidence of sustained 60 fps. Python
localization, VIO and orchestration remain substantial work.

See [Phase 3 measurements and verification](performance/phase3-world-migration.md)
for methodology, semantic changes and limitations. Future phases must retain
workspace/Python checks, replay/schema and gRPC parity, strict work-counter gates,
truth-isolation tests and installed-package checks. Add steady-state viewer frame
measurement and fixed-scene image regression checks; the current non-empty-image
smoke check alone does not establish either property.
