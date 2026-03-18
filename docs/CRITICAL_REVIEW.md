# Critical Review — Architecture Update Plan for 6-Subsystem Multi-Drone Autonomy Platform

**Reviewer role:** Worker 4 — Critical Reviewer (Stage 0 mandatory review)
**Review date:** 2026-03-15
**Codebase examined:** `/Users/rickybusse/Downloads/Smart Trajectory tracker`
**Status:** Stage 0 blocking review

---

## Executive Summary

The proposed transformation is technically achievable but the plan contains several structural contradictions with the existing codebase that will cause integration failures if not resolved before Stage 1 begins. The central problem is that the plan treats the codebase as a clean-slate starting point when in fact it has a deep, well-tested simulation kernel (`sim.py`, 4123 lines), a live Python/Rust dual-tracking stack, and a frozen-fixture parity test that will break the moment core tracking or proto boundaries are altered. The most dangerous assumption in the plan is that "Rust is source of truth for tracking" — the code currently runs a full parallel Kalman filter stack in Python (`fusion.py`) and relies on it for observation routing and acceptance decisions inside the simulation loop itself.

---

## 1. Feasibility Assessment

### What the codebase actually has today

The current system is not a skeleton. It already implements:

- A 4123-line simulation engine (`src/smart_tracker/sim.py`) with cooperative multi-drone orbit controllers (`FollowPathController`, `ObservationTriggeredFollowController`), slot-based angular assignment for drone formations, terrain-following flight, obstacle-aware planning, weather-adjusted trajectory wrappers, and launch controllers.
- A complete IMM (Interacting Multiple Model) Kalman filter in Python (`fusion.py`, ~740 lines) implementing CV + CT models, M-of-N track lifecycle, adaptive process noise, and quality scoring.
- The same tracker duplicated in Rust (`rust/tracker-core/src/lib.rs`) with a gRPC boundary (`TrackerService.ingest_frame()`). Both implementations are kept in sync through the proto contract and a frozen-fixture parity test (`tests/test_runtime.py`).
- A 2D obstacle-aware path planner (`planning.py`) using visibility-graph A* with obstacle polygon expansion.
- A layered environment model: tiled terrain heightmaps, obstacle layers, land-cover layers, LOS queries, seasonal variation, and GeoTIFF import (`environment.py`, `terrain.py`, `obstacles.py`, `visibility.py`).
- An existing proto contract (`proto/smarttracker/v1/tracker.proto`) with 23 serialized fields in `TrackerConfig`, 6 RPC methods, and bidirectional streaming.
- 26 test files covering the whole stack including a Rust runtime parity fixture test.

### What "6-subsystem multi-drone autonomy" actually requires on top of this

The plan proposes adding: a command/mission planner subsystem, a fleet coordinator, a communication link model, a threat model, a conflict-resolution layer, and a live replay/visualization update path. The plan further proposes migrating the tracker "source of truth" entirely to Rust and removing or demoting the Python Kalman stack.

**Realistic verdict:** Stages 1–3 (adding mission zones, fleet-level coordination, and a conflict resolver) are feasible as Python-layer extensions without breaking the existing stack. Stage 4 onward (changing the Rust tracker interface, adding new proto RPCs, adding fleet-level state to the gRPC boundary) carries substantial integration risk that is currently underestimated in the plan. The 8-stage schedule is aggressive against a codebase that already has >26 test files that must remain green throughout.

---

## 2. Contradiction Detection

### 2.1 "Rust is source of truth for tracking" vs. Python's full Kalman stack

**Conflict:** `docs/architecture.md` line 83 states "Rust is the source of truth for tracking output." The plan repeats this as a design principle. However, `src/smart_tracker/fusion.py` contains:

- `KalmanTrack3D` (lines 47–147): full 6-state constant-velocity Kalman filter with predict/update/snapshot
- `CoordinatedTurnTrack3D` (lines 225–398): full 7-state coordinated-turn model
- `IMMTrack3D` (lines 404–610): the complete IMM filter combining both models with innovation-history-based adaptive Q scaling
- `ManagedTrack` (lines 657–744): M-of-N lifecycle management (tentative/confirmed/coasting/deleted)

These are **not dead code**. `service.py` references `TrackerConfig` fields (`cv_process_accel_std`, `ct_process_accel_std`, `ct_turn_rate_std`, `innovation_window`, `confirmation_m`, etc.) that map directly to the Python fusion parameters and are serialized to the Rust daemon via proto. The plan does not address which layer runs first, how they are kept in sync, or what happens if they diverge.

**Risk:** Removing or subordinating Python fusion before the Rust tracker exposes equivalent APIs (especially for multi-target fleet-level commands) will break `tests/test_fusion_advanced.py`, `tests/test_runtime.py`, and any test that exercises `TrackingService.ingest_frame()` with truth-state comparison.

### 2.2 "Waypoints are not executable commands" vs. sim.py's direct waypoint execution

**Conflict:** The plan proposes that waypoints be advisory intent, not imperative commands. The existing `FollowPathController.__call__()` (sim.py lines 411–690) executes waypoints directly and imperatively on every timestep. It:

1. Computes a lead position from the target trajectory with a configurable `lead_s` parameter.
2. Selects an orbit candidate via `PathPlanner2D.plan_route()`.
3. Advances position along a planned polyline at `max_speed_mps * delta_t_s`.
4. Applies acceleration limiting and orbit-blend smoothing.
5. Returns the next position and velocity as the drone's physical state.

`ScenarioDefinition` (sim.py lines 206–298) carries `drone_roles`, `adaptive_drone_controllers`, and `launchable_controllers` — all of which are bound to specific waypoint-execution callables. The plan's intent/command distinction does not map onto any existing abstraction.

**Risk:** Introducing an intent layer over the top of an already-imperative controller will require either a full controller refactor or a shim layer that converts intents into controller seeds — neither of which is scoped in the plan's stages.

### 2.3 "Visual terrain must not be authoritative" vs. terrain.py serving both roles

**Conflict:** The plan asserts that the visual (viewer) terrain should be non-authoritative. The existing `TerrainLayer` class (`environment.py` lines 205–454) serves all three roles simultaneously:

- **Physical authority:** `TerrainLayer.clamp_altitude()` and `TerrainLayer.height_at()` are called during drone position computation inside `compose_air_state()` (sim.py) to enforce the ground constraint.
- **LOS authority:** `EnvironmentQuery.los()` (via `visibility.py`) samples `TerrainLayer.height_at()` for terrain clearance checks that determine observation acceptance/rejection.
- **Visual output:** `TerrainLayer.viewer_mesh()` (environment.py line 433) and `TerrainLayer.to_metadata()` export the same height data to the replay JSON and the Bevy viewer.

A single `TerrainLayer` instance is created once in `ScenarioDefinition.__post_init__()` (sim.py lines 274–281) and shared across all three roles. There is no separation between "physical terrain" and "visual terrain" in the type system. Decoupling these would require refactoring `TerrainLayer`, `EnvironmentModel`, and the `from_legacy()` adapter, and regenerating all fixture data.

### 2.4 Plan's "6 subsystems" vs. existing `ScenarioDefinition` doing all coordination

**Conflict:** The plan's six subsystems (sensor fusion, path planning, mission management, fleet coordination, communication model, threat model) are presently all embedded in a single `ScenarioDefinition` dataclass with no subsystem boundaries:

- `adaptive_drone_controllers` — mix of `FollowPathController` and `ObservationTriggeredFollowController`
- `launchable_controllers` — launch events dispatched directly from sim loop
- `mission_zones` — `MissionZone` tuples embedded in `ScenarioDefinition`
- `drone_roles` / `drone_target_assignments` — flat dicts, no typed subsystem

The plan assumes clean boundaries exist. They do not. Extracting these into typed subsystem objects will require changing `build_default_scenario()` (sim.py lines 2901–3085+), all callers in the test suite, and the `ScenarioDefinition` frozen dataclass.

### 2.5 Proto contract completeness

The existing proto (`tracker.proto`) defines only a **per-frame ingest contract** — it carries per-frame bearing observations, node states, and track outputs. It has no messages for:

- Fleet-level commands (assign drone to target, change role)
- Mission zone updates
- Threat alerts
- Multi-drone conflict notifications
- Communication link state

The plan's proposed new subsystems all require new proto messages and RPCs. Since `TrackerConfig` already occupies fields 1–23 in the proto message, adding fields requires careful field number assignment to preserve backward compatibility. Adding new top-level messages is backward-compatible; adding fields to existing messages requires caution and proto3 default-value semantics must be handled explicitly in both Python and Rust.

---

## 3. Missing Dependencies

### 3.1 Rust workspace additions

The plan likely proposes new Rust crates (e.g., a fleet coordinator crate, a mission planner crate). The workspace `Cargo.toml` currently defines exactly 4 members:

```
rust/tracker-core
rust/tracker-proto
rust/tracker-server
rust/tracker-viewer
```

Adding a new crate requires: creating the crate directory and `Cargo.toml`, adding it to the workspace `members` array, wiring its dependencies against the workspace-level `[workspace.dependencies]` block (which currently pins `nalgebra = "0.33"`, `tonic = "0.14"`, `prost = "0.14"`), and ensuring it is included in CI. None of this is explicitly scoped in the plan's stage definitions.

### 3.2 Proto regeneration chain

The proto binding regeneration chain is currently: `proto/smarttracker/v1/tracker.proto` → `tonic-prost-build` (Rust, via `build.rs` in `tracker-proto`) → `tracker_pb2.py` + `tracker_pb2_grpc.py` (Python, generated into `src/smarttracker/v1/`). Any proto change requires:

1. Editing the `.proto` file.
2. Rebuilding the Rust proto bindings (triggers on `cargo build`).
3. Re-running `protoc` for Python bindings and committing the generated files.
4. Updating `service.py` serializers/deserializers (`_tracker_config_to_proto`, `_tracker_config_from_proto`, `_node_to_proto`, etc.).
5. Updating `rust/tracker-server/src/lib.rs` conversion helpers.

The plan does not include an explicit proto-change workflow task in any stage. This is a multi-step change that typically takes 2–4 hours per round trip when crossing language boundaries.

### 3.3 Python module refactoring prerequisites

Before new subsystem modules can be added cleanly, the following refactoring is needed:

- `sim.py` (4123 lines) must be decomposed. At minimum the controller classes (`FollowPathController`, `ObservationTriggeredFollowController`, `LaunchController`) should move to a `controllers.py` module. The `build_default_scenario()` function (sim.py lines 2901+) must be refactored to accept subsystem-level arguments rather than flat scalar parameters. This is a prerequisite for Stage 2 (fleet coordination) because the plan cannot add a fleet coordinator as an independent module if scenario construction is monolithic.
- `fusion.py` needs a clear ownership designation: either it becomes the authoritative filter (Python path) or it becomes a test-only shadow of the Rust implementation. Currently it is neither — it is imported by `service.py` indirectly through `TrackerConfig` and its field names directly mirror the Rust struct in `tracker-core/src/lib.rs`. Ambiguous ownership here will cause silent divergence bugs.
- `environment.py` currently does a cascade of cross-module imports at the bottom of the file (lines 572–661, importing from `obstacles.py`, `visibility.py`, and `environment_io.py`) that creates a fragile import chain. Before adding new environment-related subsystem modules, this must be resolved or new modules will encounter circular import errors.

### 3.4 Test infrastructure gaps

The current test suite relies heavily on:
- The Rust daemon being built and available (spawned by `TrackingService.__init__()` via `subprocess.Popen` if `spawn_local=True`).
- A frozen fixture file at `tests/fixtures/runtime_parity_fixture.json` for parity testing.

There is no mock-based tracker stub, no fixture regeneration script documented, and no integration test that exercises multi-drone coordination at the subsystem level. Any new subsystem tests will need either a mocking layer for the Rust daemon or a fixture update workflow, neither of which exists.

---

## 4. Risk Assessment

### Risk 1: Scope Creep — HIGH

The current codebase already implements substantial portions of what the plan calls "new subsystems." `FollowPathController` (sim.py lines 349–690) is a fully featured drone path controller with cooperative slot assignment, orbit blending, terrain following, and obstacle avoidance. `ObservationTriggeredFollowController` (sim.py lines 693+) is a state-machine-based coordination controller. The plan does not acknowledge these implementations and risks re-implementing them in new module structures that conflict with the existing ones, creating two parallel implementations with no clear deprecation path.

**Mitigation required:** The plan should map each proposed subsystem to existing code and explicitly designate which existing functions are promoted, refactored, or deprecated before new code is written.

### Risk 2: Breaking Existing Tests — HIGH

The test suite (`tests/`, 26 files) exercises the full stack including:
- `tests/test_runtime.py`: Rust-parity test with a frozen JSON fixture. Any change to tracker output fields, proto field semantics, or filter parameters will fail this test deterministically.
- `tests/test_service.py`, `tests/test_sim.py`: Integration tests that construct full `ScenarioDefinition` instances through `build_default_scenario()`. Refactoring the scenario-building pipeline will break these unless changes are made simultaneously.
- `tests/test_fusion_advanced.py`: Tests `IMMTrack3D`, `CoordinatedTurnTrack3D`, and `ManagedTrack` directly. If Python fusion is demoted, this test file either becomes dead weight or needs replacement.

There is no staged build gate in the plan to keep tests green between stages. The plan must treat test-green status as a mandatory gate for each stage exit.

### Risk 3: Performance Regression — MEDIUM

The simulation runs at `dt_s = 0.25` seconds with N drones calling `FollowPathController.__call__()` per step, each potentially invoking `PathPlanner2D.plan_route()` (visibility graph A*) and `EnvironmentQuery.los()` (ray-march LOS check). Adding a fleet coordinator that runs between the controller and the scenario loop will add a synchronization point per timestep. If the fleet coordinator is implemented in Python with any form of global state or locking, performance will degrade non-linearly with drone count. The plan does not include a performance baseline test or a regression threshold.

### Risk 4: Complexity Debt — HIGH

`sim.py` is already 4123 lines and contains deeply nested closures (the `build_default_scenario()` function uses inner closures `_zone_from_center`, `_try_place`, `_sample_objective`, `_sample_exclusion`, `_sample_surveillance`, `_sample_patrol`). The `ScenarioDefinition` dataclass has 17 fields, including three `Mapping[str, object]` catch-alls. Adding new subsystems to this dataclass rather than creating typed subsystem objects will worsen maintainability. The plan's 8-stage delivery will make this worse at each stage unless explicit decomposition is enforced before Stage 1.

### Risk 5: Integration Gaps at the Python/Rust Boundary — HIGH

The current gRPC boundary is a per-frame stateless contract: Python pushes observations, Rust returns tracks. The plan's fleet coordinator and mission manager need **bidirectional, persistent state** across frames (e.g., drone assignments, threat escalations, zone violations). The existing proto contract has no messages for this. Implementing fleet-state as a per-frame sidecar in the proto (adding it to `PlatformFrame`) is feasible but means the Rust daemon must carry and return fleet state it has no business computing. Implementing it as a separate gRPC service is the correct architecture but requires a new proto service definition, a new Rust server crate, and new Python client code — none of which are prerequisites scoped in Stage 1.

---

## 5. Recommended Adjustments

### 5.1 Stage ordering

**Current assumed ordering (inferred):** Add subsystems → wire coordination → add proto messages → add Rust crates → add viewer support.

**Recommended ordering:**

1. **Stage 0 (current):** Critical review. Done.
2. **Stage 0.5 (prerequisite, not in plan):** Decompose `sim.py`. Extract `controllers.py` (move `FollowPathController`, `ObservationTriggeredFollowController`, launch controller). This is a prerequisite for all later stages. Estimated: 2–3 days. No behavior change.
3. **Stage 1:** Introduce typed subsystem interfaces (Protocol classes or abstract dataclasses) for each proposed subsystem. Wire to existing implementations. This establishes seams without yet replacing behavior. Keep all tests green.
4. **Stage 2:** Implement fleet coordinator as a Python-layer coordinator that delegates to existing controllers. No Rust changes. No proto changes. This tests the architecture without incurring boundary risk.
5. **Stage 3:** Proto additions (new service for fleet state). Requires: proto file edit, Rust proto crate rebuild, Python bindings regeneration. Treat this as a dedicated integration milestone with its own test gate.
6. **Stage 4+:** Rust additions and viewer updates.

### 5.2 Explicit decisions required before Stage 1

The following questions must be answered in writing before Stage 1 begins:

1. **Python fusion vs. Rust tracker:** Is `fusion.py` authoritative, a shadow, or a test fixture? The answer determines whether `IMMTrack3D` stays in the hot path or is removed. Currently it is in neither and both simultaneously.
2. **TerrainLayer unification:** Does the plan accept that visual and physical terrain remain unified, or is decoupling a Stage N item? If decoupling is required, it must be explicitly staged.
3. **`build_default_scenario()` ownership:** Who calls it in the new architecture? Does it become a factory method on a `ScenarioBuilder` object? The current flat function signature (14 parameters) is incompatible with the proposed subsystem model.
4. **Proto backward compatibility policy:** All existing replay JSON and fixture files will need to be regenerated or versioned when proto fields are added. Is this acceptable?

### 5.3 Scope reduction recommendation

The plan as described takes the system from its current state to a 6-subsystem autonomy platform in 8 stages. Given the existing depth of the codebase, stages 1–3 alone represent 3–5 weeks of careful refactoring work. The plan should explicitly descope or defer the following from early stages:

- Communication link modeling (no existing primitives; requires new infrastructure from scratch)
- Viewer integration of new subsystem state (Bevy code changes are isolated but slow to test)
- Any new Rust crates before Stages 4–5

---

## 6. Definition of Done Review

Without seeing Section 14 of the plan directly, I can assess the enforceability of likely DoD criteria against the current test infrastructure:

### "All tests pass" — PARTIALLY ENFORCEABLE

The existing test suite can be run with `python3 -m pytest tests/ -q` and `cargo test`. However:
- `tests/test_runtime.py` requires a live Rust build and will fail in environments without Cargo.
- `tests/test_service.py` spawns a subprocess (the `smart-trackerd` binary). CI environments must have the binary pre-built.
- There is no test for multi-drone fleet coordination behavior, so passing tests does not imply the new subsystems are correct.

### "Simulation is deterministic for fixed seed" — ENFORCED

This invariant is tested implicitly by the replay schema tests (`tests/test_replay.py`, `tests/test_replay_schema.py`) and the runtime parity fixture. Any change that alters determinism will break these tests.

### "Rust is source of truth for tracking" — NOT ENFORCEABLE AS STATED

There is no test that verifies Python fusion is not used in the primary execution path. Both `IMMTrack3D` (Python) and the Rust tracker process the same observations, but through different code paths. The DoD criterion would need an explicit test that the Python fusion code is not called during `run_simulation()`, or an explicit code removal.

### "Physical collision never pushes entities below terrain" — ENFORCED

`collision_aware_position()` (called in `FollowPathController.__call__()`) enforces this. Tested in `tests/test_collision.py`. This invariant survives any refactoring that preserves the call.

### "Proto changes update both Python and Rust bindings" — NOT ENFORCED

There is no CI check that verifies the generated Python bindings match the current `.proto` file. It is possible to edit `tracker.proto` and commit without regenerating `tracker_pb2.py`. The DoD criterion needs a `protoc --check` step or a generated-file hash in CI.

### "Replay metadata changes are additive" — PARTIALLY ENFORCED

The replay schema JSON (`docs/replay-schema.json`) is tested against in `tests/test_replay_schema.py`. Additive changes will not break the schema test. Breaking changes will. But the test only validates structure, not semantic correctness of new fields.

---

## Summary of Blockers for Stage 1 Entry

| Blocker | Severity | Resolution Required |
|---|---|---|
| `sim.py` monolith prevents clean subsystem seams | High | Extract `controllers.py` before Stage 1 |
| Python fusion ownership ambiguous | High | Explicit decision document required |
| Proto has no fleet-state messages | High | New proto service scoped before Stage 3 |
| No mock layer for Rust daemon in tests | Medium | Required before any subsystem unit tests |
| `TerrainLayer` serves visual and physical roles simultaneously | Medium | Accept or defer decoupling explicitly |
| No generated-binding check in CI | Medium | Add proto sync check before proto changes |
| `build_default_scenario()` is a 14-parameter flat function | Medium | Refactor to subsystem-accepting builder pattern |
| No performance baseline | Low | Establish before adding coordination overhead |

None of these blockers are individually fatal, but proceeding to Stage 1 without resolving the top three will result in Stage 2 work being built on foundations that require later teardown.
