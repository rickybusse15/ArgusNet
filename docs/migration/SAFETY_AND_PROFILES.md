# P4.9d: blocking safety and mission profiles, native

**Status:** local, 2026-09-23. See [STATUS.md](STATUS.md).

P4.9c's acceptance run refused two of the seven canonical scenarios,
`sar_person_search` and `industrial_asset_survey`. Both are mission profiles,
and a profile does only four things beyond setting ordinary options:

1. It turns on `safety_blocking`, stage 4's hard safety monitor.
2. It replaces the generated mission zones with the profile's zones, lifted
   onto the terrain.
3. For inspection, it replaces the generated POIs with the profile's POIs, at
   hover height, the first `poi_count` of them.
4. It names the scenario after itself and adds a `mission_profile` block to
   the replay's `meta`.

Stage 4 was ported and unit-tested in P4.8c but never assembled from a
bundle. This slice assembles it, carries the flag through both bundle
writers, and builds profiles natively.

## Stage 4, as the reference runs it

Comparing the unassembled body against the reference tick for tick found
three arguments that were not the reference's:

| Argument | Was | Reference |
| --- | --- | --- |
| `dt_s` | the tick's **timestamp** | `simulation_config.dt_s` |
| caution fraction | 0.85 | 0.9, `SafetyMonitor.__init__`'s default, which the reference never overrides |
| terrain height | one constant | `scenario.terrain.height_at(x, y)` under each platform, after deconfliction |

The monitor derives acceleration from `dt_s`, so the first defect made every
acceleration check wrong. Caution reaches no event and no clamp, only the
monitor's recorded state. A test therefore pins it through that state: a
tracker at 31 m/s is nominal at 0.9 and would be caution at 0.85.

What else assembly needed:

* **Checkpoint state.** Each platform's escalation record (consecutive
  violating frames, and state) and the velocity the monitor last allowed are
  causal. `SimulationSafetyMonitor` gains `record` and `restore_record`. The
  latter refuses a pair `process` could not have left, such as a violation
  count on `nominal` or none on `abort`. The monitors are rebuilt from the
  roles on restore, because the limits are configuration, not state.
* **Events into fusion.** The reference sends the tick's validations as
  `safety_events` on its ingest request, and the daemon echoes them on the
  fused frame. `SafetyLink::feed_fusion` does the same in process.
  `SafetyValidation::to_event` is `_safety_event_to_proto`.
* **The replay.** Frames carry `safety_events`, read back as
  `_safety_event_from_proto` reads them: `accepted = not blocked`, the
  validator named by the reader, `reason or state or None`.
* **Not installed unless asked for.** A bundle without the flag assembles
  exactly as before. The stage-4 slot keeps its placeholder, and no
  checkpoint gains a key.

### The bundle flag

`"safety_blocking": true`, written by `export_bundle` and the native builder
**only when set**. Every bundle written before this slice therefore still
reads as off, and every committed bundle fixture is byte-identical.

## Mission profiles in the native builder

`scripts/export_builder_tables.py` exports each profile from the reference:
its domain, workflow and mission mode, and its zones and POIs **on a unit
map**. `build_profile_mission_zones` and `build_profile_inspection_pois` scale
each coordinate and radius by the map extent in a single multiplication, so at
extent 1.0 they produce the coefficients themselves. The builder's `extent *
coefficient` then equals the reference's product exactly, the negated
exclusion centre included.

The builder checks a profile as `ScenarioOptions.__post_init__` does: a known
id, and the domain, workflow and mission mode that profile declares. A
mismatch is `Invalid`, not `Unsupported`. Profile zones skip the zone
generator's draws. Those are the last draws the reference's generator makes,
so skipping them moves no other value.

## Verification

### Parity with the reference

| Fixture | What must match | Result |
| --- | --- | --- |
| `scenario_bundle_tt_safety` (all-follow, interceptor envelopes, 80 ticks) | every node's position, velocity, health and battery, and every safety event, **bit for bit on every tick** | 242 events (35 warnings, 207 clamped) all match |
| `scenario_bundle_scan_safety` (scan mission, tracker envelopes, 400 ticks, localization replayed) | the same, plus the mission phase, through `complete` | all match; six violation kinds, `max_descent_rate` included |
| `native_bundles.json`: both profiles, industrial with `poi_count=2`, and a blocking run | the native builder writes `export_bundle`'s bundle: structure exact, floats within 1e-12 m | all match; the 11 earlier cases unchanged |
| whole bundles `tt_safety`, `scan_safety` | the native builder rebuilds the committed bundles whole | match |

`assembled_resume.rs` adds a blocking-safety run, checkpointed at five ticks,
mid-warning and mid-abort, and a reset differential. Every later tick is
identical.

### Mutation rounds

| Area | Mutants | Result |
| --- | --- | --- |
| stage 4 (arguments, wiring, events, replay, checkpoint, reset) | 16 | 10 caught at first. The six survivors were the caution fraction, the battery wiring, the replay's reason fallback, two `restore_record` consistency rules, and the checkpoint and reset paths. Tests were added for each: the caution-state test, a drained-battery assembly test, the event read-back test, the `restore_record` table, the resume and reset differentials, and the `check_state` refusals. All 16 are now caught |
| native builder profiles and the exporter flag | 15 | 15 caught |

### Acceptance

The full canonical suite, all seven scenarios, was rerun on the final release
build (`scripts/run_phase_acceptance.py`, envelopes unchanged from P4.9c).
Every exact check passes in all seven, 20 of 20 each. Every performance gate
passes: the worst tick p95 is 0.213 ms and the worst startup 43 ms.
**Six of the seven scenarios pass every envelope.**

| Scenario | Result | Detail |
| --- | --- | --- |
| baseline_coverage, intercept_stress, persistent_long, search_acquisition, planner_adversarial | pass | as in P4.9c, identically (the run is deterministic) |
| industrial_asset_survey | **pass** | every envelope; 24.1x |
| sar_person_search | **fail**: `covariance_reduction_mean` | reference −1.875, Rust −0.181, \|Δ\| 1.69 against 0.10 allowed. Every other envelope passes: RMSE 39.4 / 33.0 m (≤ 9.9), continuity, reacquisition, completion and energy, and 5/5 verdicts |

**Why `covariance_reduction_mean` fails, and why it is not widened.** The
metric compares a track's first covariance with its covariance at its **last
appearance**, for tracks with at least five updates. In this scenario only
one such track exists per seed at most. The reference has one on one seed
(42). Rust has one on three seeds (42, 137, 31415). So the two means are over
different seeds, and each is decided by one track. Three of those four
tracks' last appearances fall four steps into a coast, just before deletion,
where the value measures the coast rather than the filter. The fourth, Rust's
on seed 31415, ends on a fresh update, and it scores +0.89. On seed 42, the only seed
where both sides report it, it is the same target's track on both sides, in
the same final coast: −1.88 against −1.49. The difference comes from
draw-dependent histories (first trace 321 against 297 m², final 923 against
737 m²).

Fusion is proven identical given its inputs (P4.9c), and stage 4's kinematics
and events are proven identical tick for tick above. No defect was found. The
envelope stays declared and failed. What would make the metric meaningful,
such as the covariance at the last update rather than the last appearance, or
a minimum number of qualifying tracks, is a change to the benchmark's
definition. That is a decision, not a fix, and it is recorded in
[KNOWN_GAPS.md](../KNOWN_GAPS.md).

**Decided (D-1, 2026-10-03):** measure at the last update, and score only
with at least five qualifying tracks a side. On the rerun
([PHASE_ACCEPTANCE.md](PHASE_ACCEPTANCE.md#after-d-1-the-covariance-metric-redefined-2026-10-03))
`sar_person_search` passes `covariance_reduction_mean` (0.544 / 0.495). That
run, on Linux, fails the scenario's RMSE envelope by 0.11 m instead, for a
platform reason described there.

### Gate

| Check | Result |
| --- | --- |
| `cargo test --workspace`, `cargo clippy --all-targets -D warnings`, `cargo fmt --check` | 1096 passed, 0 failed; clean; clean |
| `ruff check`, `ruff format --check`; `export_builder_tables.py --check` | clean; current |
| `pytest tests/` (`ARGUSNET_REQUIRE_RS=1`) | 1047 passed, 9 skipped, 0 failed |
| CI | **not run**: nothing is pushed |

## Findings

* **The reference's blocking monitor latches drones into `abort`.** An abort
  zeroes the velocity it allows, and that zero is stored as
  `previous_velocity`. On the next tick the controller commands its normal
  speed again, which reads as an acceleration far past the limit, so it
  aborts again. In the scan fixture, both drones abort on all 400 ticks, and
  `max_horizontal_accel` is among the violations on 735 of those 800. The
  clamp does not accumulate, because each tick starts again from the
  controller's pose, so the mission still completes. But every event is an
  abort. The port reproduces this, as parity requires. Whether the reference
  policy should compare against the *commanded* rather than the *allowed*
  velocity after an abort is a policy question, recorded in
  [KNOWN_GAPS.md](../KNOWN_GAPS.md).
* **`covariance_reduction_mean` measures the coast before deletion.** See
  the acceptance results.
