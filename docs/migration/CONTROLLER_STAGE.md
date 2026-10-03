# P4.8d.3: the controller stage — the Rust tick moves the fleet

**Status:** delivered 2026-09-22, locally verified, then merged 2026-10-01 in #122 (`badb1e1`). Package ID
`P4.8d.3`. Built on PR #119 (`codex/runtime-command-checkpoints`, the P4.1
command/checkpoint repair) plus the P4.8d.1–2b motion stack
(`claude/p48d-controllers`), which it carries as four cherry-picked commits.

Until this slice `Stage::SampleControllersAndBattery` ran only `BatteryStage`.
Platform positions were written into `TickWorld` by whoever drove the run, so
the Rust tick composed fusion, mapping, deconfliction and mission state around
a fleet it could not move. P4.8's exit criterion — *"a full representative run
is owned by Rust"* — was unreachable. This slice closes stage 1:

| Piece | Where | Reference |
| --- | --- | --- |
| Handler state contract on #119's checkpoint | `simulation.rs`, `snapshot.rs` | P4.1 corrective gate, §6 |
| `BatteryStage` checkpoints its ledger | `battery.rs` | `_battery_states`, prev positions, warned latch |
| `looped_waypoint_path` (the search lane) | `waypoint_path.rs` | `sim.py` `looped_waypoint_path` |
| `ObservationTriggeredFollowController` | `observation_follow.rs` | `sim.py` class of that name |
| `ControllerStage`: controllers, then battery | `controller.rs` | `scenario.node_states(t)` + battery loop |
| Stage 6 → stage 1 engagement feed | `observe.rs` `feed_controllers` | `controller.update_from_frame(frame, batch)` |

Every controller the default scenario composes now exists in Rust and runs
inside the tick: ground stations, `Launch(Follow)` inspect/follow drones and
`Launch(ObservationTriggered(search, follow))` search drones.

## The handler state contract, ported onto #119

#119 persists pause, speed, the command queue and retry history, but a stage
handler could still only `run()`: nothing a handler accumulated reached the
checkpoint, and `reset()` left it in place. A controller carries state across
ticks, so without this a resumed mission flies a different path.

`StageHandler` gains `reset`, `checkpoint_state`, `check_state` and
`restore_state`. The defaults *declare* "I hold nothing that survives a tick";
a handler handed state it never writes refuses it. `check_state` is separate
from `restore_state` so that `Simulation::restore` validates **every**
handler's state before mutating anything — with two stateful handlers, taking
each one's state as it validated would half-restore on a later refusal. The
checkpoint gains `stage_states` (format 3). A handler with no entry in the
checkpoint is reset rather than left holding the restoring object's state.
Refusals are typed: `UnknownStageState`, `StageState { stage, detail }`.

This is the P4.1d contract from `claude/p41d-checkpoint-completeness`, re-done
against #119's checkpoint shape rather than merged, since the two P4.1 branches
conflict. The rest of that branch (epoch branching on restore, FNV world
content hash) is not ported; see *Next*.

## Findings

**1. Checkpoints were not bit-exact through JSON.** `serde_json` without
`float_roundtrip` does not parse floats correctly rounded, so an f64 written to
a checkpoint can come back one ULP away. That affects #119's own persisted
`speed` as much as anything added here. `argusnet-sim` now enables
`float_roundtrip`. See the verification table for whether the differential
test catches its removal.

**2. The follow controller steers off ground truth.** The reference hands
`FollowPathController` the *target's trajectory function*: pursuit reads true
target state, not a track. §3 makes truth a capability and stage 1 is not
admitted to it, so the port cannot do the same thing silently. It takes a
`PursuitOracle`, which the run driver installs explicitly, and the trace
records a `legacy pursuit read N true target states` note on every tick one is
consulted. Parity is preserved and the truth read is visible. Moving pursuit to
fused tracks is a behaviour change that needs its own validation.

**3. The reference's `reset_state` is not a reset.** `FollowPathController.
reset_state` leaves the acceleration limiter's last velocity and the smoothed
terrain height behind, so running one scenario object twice would start the
second run with a velocity from the first. The reference's CLI never does
that. `StageHandler::reset` promises a freshly constructed handler, so the
stage resets controllers to their constructed state instead of calling the
partial reset. `a_reset_run_is_a_fresh_run` pins this.

**4. Relaunch on reset.** The reference's `run_simulation` launches every
drone at `t = 0` after resetting runtime state; that is driver code, not
controller code. `LaunchProfile::auto_launch_at_s` carries it so a reset run
relaunches the way a fresh one does.

## Preserved rather than corrected

* **Launch drift is always integrated over 1.0 s**, whatever the timestep (the
  reference passes no `dt` on the launch path). The stage test runs at
  `dt = 0.5` precisely so a port that "fixed" this fails.
* **An observation before the first sample engages without seeding**, and
  **reset resets only the follow half** (`hasattr(…, "reset_state")`).
* **Engagement is a next-tick effect.** The reference applies it after fusion
  and Rust applies it inside stage 6. Nothing between the two reads it, and it
  is first flown by the next tick's stage 1, so the difference cannot be
  observed.
* `looped_waypoint_path`: `searchsorted(side="right")` (a distance exactly on a
  vertex takes the next segment), the 0.5 m/s speed floor, `phase % tau` with
  Python's modulo, `np.linalg.norm` rather than `hypot`, and numpy's
  left-to-right `amplitude * frequency * cos` order.

## Verification

Fixtures record the reference; nothing recomputes it.
`scripts/capture_controller_fixture.py` → `tests/fixtures/controllers.json`:
7 `looped_waypoint_path` cases (terrain and obstacle reads as panic-on-miss
oracles, reusing `capture_follow_path_fixture.py`'s recorders) and 5
`ObservationTriggeredFollowController` scripts, compared as exact call logs.
Its own `check_coverage` asserts the obstacle push fires, min-AGL binds, the
speed floor applies, the vertex case resolves to the second segment, and each
wrapper quirk shows up in the log.

`rust/argusnet-sim/tests/controller_stage.rs` runs the real pipeline: stage 1
with battery chained, and stage 6 with a real sensor feeding engagement. It
checks:

* the fleet moves every tick after launch, nothing is ever below terrain,
  engagement arrives mid-run through stage 6, and battery is drawn;
* **uninterrupted run equals resumed run**: checkpoints taken mid-climb,
  mid-search, on the tick an observation engaged the follow controller (before
  it had flown), and deep in the follow are each serialised to JSON text and
  restored into a different `Simulation` that was first sent down a wrong
  history. Every platform's position, velocity, timestamp and battery fraction
  must then match bit for bit on every remaining tick;
* a reset run equals a fresh run;
* a checkpoint for a different fleet is refused and changes nothing; one
  naming an absent stage is refused.

### Mutation round

31 mutants, **31 caught** after one strengthening pass. The first pass caught
26. Each of the 5 survivors pointed at a missing test, and all are now covered:

| Survivor | Why it survived | Fix |
| --- | --- | --- |
| restore skips launch state | every drone auto-launched at t=0, so a wrong-history object already held the right launch state | `drone-late`, launched by hand at tick 30; the wrong history launches it at tick 2 |
| a handler absent from the checkpoint keeps its state | no checkpoint ever lacked stage-1 state | `a_checkpoint_without_stage_state_resets_the_stage` |
| platform timestamp not written | nothing checked the absolute time | per-platform timestamp assertion |
| launch drift over the run's `dt` | the differential compares the port with itself | `launch_drift_is_integrated_over_the_reference_one_second` at `dt = 0.5` |
| checkpoint order not checked | the swap put different kinds side by side, so the kind check refused first | two same-kind ground stations swapped |
| `hypot` for segment length | every fixture segment had an exactly representable length | `diagonal_lengths_use_numpy_norm`, a segment found by search where the two round apart |

Caught on the first pass, among others: removing `float_roundtrip` (the
resume differential diverges, which confirms finding 1), dropping follow state,
engagement, the nested follow state, the battery ledger or its previous
positions from the checkpoint, the reference's partial reset in place of fresh
construction, skipping the relaunch on reset, `Simulation::reset` not reaching
handlers, running battery before the controllers, cutting the stage 6 → 1
feed, validate-after-mutate in `restore`, and every wrapper and waypoint
quirk listed above.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` (includes viewer) | 1015 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean (one `large_enum_variant` allowed on `Operational`, with the reason at the site) |
| `cargo fmt --all -- --check` | clean |
| `pytest tests/` | 1001 passed, 9 skipped |
| CI | **not run**: nothing is pushed |

## Not in this slice

* **No scenario builder.** Waypoints, follow configs, station assignment and
  the sensor table are built by the test here and by `sim.py` in production.
  Building them in Rust from a versioned scenario config is P4.9 work.
* **No owned `argusnet-world` adapter for the stage.** `WorldTerrain` and
  `WorldObstacles` borrow; `ControllerEnvironment` needs owned `Send`
  capabilities. It is a thin wrapper over an `Arc`'d world and is the first
  item under *Next*.
* **No planner adapter.** `planner: None`, as the reference's default
  scenario passes for follow controllers. Search-lane routing around obstacles
  happens at construction time (`search_path`), so it belongs to the scenario
  builder.
* **`TickWorld` is not in the checkpoint.** Stage 1 rewrites every controlled
  platform's kinematics before any other stage reads them, so the resume
  differential holds. A driver-prescribed platform, or a stage-2+ mutation
  that persists across ticks, would still need the world persisted.
* **`ObserveStage` holds uncheckpointed state**: the per-sensor bias random
  walk. The differential passes because the bias is applied to a bearing
  *after* acceptance (`observe.rs`, per observation), the rig's sensor is
  drift-free by default, and stage 7 is not installed. Once fusion is in the loop that
  drift is causal, and `ObserveStage` needs the same contract. Fusion, mapping,
  deconfliction, overrides and mission stages have not been audited against
  it either.
* `EnvelopeLimitedTrajectory` (opt-in `enforce_flight_envelope`) and the target
  trajectory generators (`sinusoid_target`, `racetrack_target`,
  `orbital_path`) are not ported. Targets reach the tick as prescribed truth.

## Next

**`P4.8d.4` — world-backed stage and a stage-1 scenario parity run.**
Owned `argusnet-world` adapters for `ControllerEnvironment`, then a fixture
from `sim.py` running a small default scenario for N ticks. It records node
states per tick and the scenario's built waypoints, follow configs and target
trajectories, and the Rust stage must reproduce the node states. This is the
first Python-vs-Rust comparison of stage 1 as a whole, not of its parts.

Then **the remaining causal-state audit**: `ObserveStage` bias drift, fusion
tracker state, mapping, deconfliction latches and mission state, each on the
`StageHandler` contract, each with a differential like the one here. After
that, `TickWorld` in the checkpoint, and the P4.1d pieces #119 does not have
(epoch branching on restore, real world content hash).

Then **P4.9** (`argusnet-rs` CLI) can drive a scenario whose stage 1 is Rust.
