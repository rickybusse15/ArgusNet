# P4.8e.1: scenario bundle v1 and runtime assembly

**Status:** delivered 2026-09-23, locally verified only (not pushed). Package ID
`P4.8e.1`. See [STATUS.md](STATUS.md) for where it sits in the pre-P5.0 queue.

Every tick stage had a Rust body, and stage 1 moved the fleet (P4.8d.3). But
nothing **composed** a scenario into a configured `Simulation`: nothing
installed each stage, lent it the world, gave each node its sensor and
controller, drove the targets' truth, or recorded a replay. This slice adds
that assembly, for target-tracking scenarios, and the versioned document it
runs from.

| Piece | Where |
| --- | --- |
| Exporter: a built Python scenario → `argusnet.scenario-bundle/v1` | `src/argusnet/simulation/scenario_bundle.py` |
| `bundle_spec` on `static_ground_point` / `looped_waypoint_path` closures | `sim.py` (data only; no arithmetic changed) |
| Bundle types, world build, assembly, replay-producing runner | `rust/argusnet-sim/src/scenario.rs` |
| One owned world shared by stages 1, 3 and 6 | `rust/argusnet-sim/src/runtime_world.rs` |
| Motion weather (`WindModel`, `flight_speed_penalty`) | `rust/argusnet-sim/src/weather.rs` |
| Deconfliction clamps against real terrain | `rust/argusnet-sim/src/deconflict.rs` (`RoleTable::with_terrain`) |

## The format

Resolved data, never a preset name that could be re-resolved differently:

* **world**: the `terrain_new`, `obstacles_new` (with each primitive's
  `obstacle_new` metadata) and `land_cover_new` payloads, exactly as the
  native ABI already decodes them on every Python run; the reference's optical
  visibility profile; the sensing LOS clearance.
* **nodes**: sensor (the reference's own `SensorErrorConfig`, from
  `_sensor_model_for_node`), health, and controller: `static`, or `launched`
  around `search`, `follow` or `search_then_follow`.
* **targets**: truth sampled at every tick timestamp and at every
  `timestamp + lead_s` a follow controller asks for. Each key is computed with
  the single addition both implementations perform, so the Rust lookup is by
  bits and a miss is an error, never an interpolation. Exact because every
  reference target trajectory is a pure function of time.
* **weather** (motion half), **tracker** config, **drone roles**, **mission zones**.

`ScenarioBundle::from_json` checks the format and version envelope before
anything else, so a v2 bundle is refused as a version, not as whichever field
v2 renamed. Unsupported features are refused by name on both sides: the
exporter raises `UnsupportedScenario` and `assemble` returns
`BundleError::Unsupported`. v1 refused `scan_map_inspect` until the mission
driver landed (P4.8e.3). It still refuses the flight envelope, frontier and
occlusion-aware mapping, adaptive search, VIO position substitution, and a
follow controller with a route planner. Blocking safety is carried since
P4.9d, as `"safety_blocking": true`, written only when set so older bundles
read as off ([SAFETY_AND_PROFILES.md](SAFETY_AND_PROFILES.md)).

## What is compared, and how exactly

`scripts/capture_scenario_bundle_fixture.py` exports a bundle and runs
`run_simulation` on the same built scenario, recording per tick every node's
kinematics, the truths, the deconfliction participants and each
search-then-follow drone's engagement state.

**`tt_follow`, all drones truth-fed (7 nodes × 80 ticks): every node on every
tick matches bit for bit**. That includes launch, climb, weather drift,
truth-fed pursuit with line of sight over the real world, and **61 frames of
deconfliction** resolved and clamped against the real terrain. No observation
can move anything in this scenario, so nothing may differ, and nothing does.

**`tt`, one search-then-follow drone**: stage 6 draws from keyed ChaCha
streams (ADR-006), not NumPy's, so the two runs accept different observations
and the drone engages on a different tick (tick 4 in the reference). Every node
matches bit for bit until then; stations match for the whole run; each follow
drone matches until deconfliction couples it to the diverged drone (tick 25).
That is 294 exact node-ticks. Accepted observations over the run: Python 227,
Rust 217. Proper statistical envelopes are P4.9c.

## Findings

1. **Deconfliction clamped against a constant.** The stage's floor was a flat
   `Option<f64>` (its own doc said "until the world is wired in"). The
   reference calls `terrain.clamp_altitude(xy, z, min_agl)`. `RoleTable`
   now carries the real terrain, and the all-follow run's 61 resolved frames
   match exactly with it.
2. **Stage 6 senses from the flown position; the reference senses from the
   nominal one.** `build_observations` is handed `scenario.nodes` and calls
   `node.state(t)`, so observations originate at the *pre-override,
   pre-deconfliction* controller position. Rust stage 6 reads `TickWorld`.
   → P4.8e.2.
3. **Directional sensors look the wrong way.** The reference looks along the
   nominal velocity, tilted down by `drone_look_down_angle_deg` (30°), and
   only above 0.5 m/s of horizontal speed. Rust looks along the raw flown
   velocity above 1e-6 m/s. Drones carry a 55° directional sensor by
   default, so this affects every drone observation. → P4.8e.2.
4. **Tracker association mode spelling.** Python writes `labeled` / `gnn` /
   `jpda`; `argusnet-core`'s serde form is the Rust variant name. Mapped at
   the bundle boundary, as the gRPC adapter maps it, rather than by changing
   the core's serialized form under its other consumers.
5. **`float_roundtrip` must be workspace-wide.** Enabling it only on
   `argusnet-sim` (P4.8d.3) meant `cargo build -p argusnet-native` parsed
   floats differently from a workspace build, because Cargo unifies features
   per build. It is now on the workspace dependency.

## Not in this slice

* **Mission stages 2, 8 and 9**, and the mission driver that consumes
  `MissionStage` events (POI assignment, egress, frontier). → P4.8e.3.
* **Stage 5 VIO** is not installed. The reference always runs VIO for mobile
  nodes, but in target tracking it only reaches the replay's VIO fields.
* **Battery reserve wind** is not evaluated per tick. The reference asks
  `wind_at(z, t)` for each drone every tick; `BatteryLink` holds one wind
  vector. That affects the reserve warning, not kinematics. → P4.8e.4.
* **Generation rejections** are not written into replay frames yet.
* **Replay coverage/localization/mission fields** do not exist for target
  tracking in v1.

## Verification

Fixtures: `scripts/capture_scenario_bundle_fixture.py` (two scenarios, plus 340
reference `query.los` sight lines over the bundle's world) and
`scripts/capture_weather_fixture.py` (14 weather cases). The reference is always
recorded, never recomputed.

| Test | What it proves |
| --- | --- |
| `an_all_follow_scenario_matches_on_every_tick` | 7 nodes × 80 ticks: position, velocity, health, battery fraction, all bit-exact, through 61 deconfliction frames |
| `a_python_scenario_runs_in_the_rust_tick_and_matches_where_it_must` | 294 exact node-ticks up to engagement divergence; Rust engages through its own stage 6 → 1 feed |
| `line_of_sight_matches_the_reference` | 340 sight lines, visibility, blocker, transmittance, detection and noise multipliers bit-exact; follow clearance equals the reference's default |
| `the_assembly_wires_every_capability` | visibility oracle, sensor FOV specs, tracker mode, battery model, roles, terrain floor, stage-3 slot |
| `every_preset_matches_the_reference_bit_for_bit` | wind and flight-speed penalty, every preset and clamp |
| `test_scenario_bundle.py` (6) | exporter refusals by name, truth sampled at every tick and lead, closure specs |

### Mutation round

26 mutants: **25 caught, 1 proven equivalent**, after one strengthening pass.
The first pass caught 15. All 11 survivors were things the exact kinematic
comparison cannot reach, because they only change observation-fed behaviour or
were never exercised by the scenario: LOS clearance, land cover at the midpoint,
the FOV threshold, the battery model, follow-controller obstacles, the
visibility oracle, the deconfliction slot, the engagement feed, roles, health
and the tracker mode. Each was closed by recording more of the reference (health
and battery fraction per node, 340 LOS queries) or by a direct wiring assertion.
The battery model is now caught behaviourally: every drone's battery fraction
matches Python exactly across both runs.

The equivalent mutant is the gust gate. Dropping `gust_amplitude_mps > 0` from
`period > 0 && amplitude > 0` changes nothing, because a zero amplitude makes
the gust `0 · sin(·) = 0`.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1023 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `pytest tests/` | 1004 passed, 9 skipped, 3 failed; all 3 are the pre-existing `argusnetd` startup timeout below |
| `ruff check` (new and touched files) | clean |
| CI | **not run**: nothing is pushed |

**Pre-existing flakiness, not from this slice:** Python tests that spawn
`argusnetd` intermittently fail with `Timed out waiting for argusnetd`, 1–4
per run on a different set of tests each time. Untouched `origin/main`
`8dfab06` reproduces it (3 and 4 failures over two runs of
`tests/test_service.py`). A stale `target/debug/argusnetd` makes it worse, and
this slice's workspace-wide feature change forces that rebuild. Flagged as its
own task.

## Next

**P4.8e.2**: a nominal-position source (the controller stage's own output
before overrides) for stages 5 and 6, and the reference's look direction for
directional sensors. Prerequisite: this slice. Then **P4.8e.3**: the mission
driver and `scan_map_inspect` assembly.
