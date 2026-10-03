# P4.8e.2: stage 6 senses the way the reference does

**Status:** delivered 2026-09-23, locally verified, then merged 2026-10-01 in #122 (`badb1e1`). Package ID
`P4.8e.2`. See [STATUS.md](STATUS.md).

P4.8e.1 ran a Python-built scenario in the Rust tick and matched it bit for bit
wherever observations cannot reach. It also found that stage 6 (observation
generation) differed from the reference in two named ways. Fixing those
exposed a third, larger one. The mis-aimed field-of-view gate had been
hiding it.

| | Reference (`build_observations`) | Rust before | Rust now |
| --- | --- | --- | --- |
| Sensor origin | the **nominal** controller pose: `node.state(t)`, before overrides and deconfliction | the flown pose in `TickWorld` | the nominal pose, from stage 1 |
| Look direction | nominal velocity, vertical replaced by `-tan(30°)·speed_xy`, only above 0.5 m/s horizontal | raw flown velocity above 1e-6 m/s | the reference's |
| Detection probability | `p_d × sensor_pd × cloud`, where `p_d` is the world's: path loss, transmittance, weather visibility, land-cover multiplier | `sensor_pd` alone | the reference's |
| Bearing-noise multiplier | `effective_noise_multiplier` (land cover × range growth × weather) | the raw LOS multiplier | the reference's |
| Dominant loss (vegetation vs dropout) | the world's `dominant_loss_factor` | a local rule on sensor transmittance | the reference's |

## How the third one was found

Fixing the look direction made stage 6 **worse** on the face of it:
accepted observations over the mixed scenario rose from 217 to 433 against the
reference's 227. The draw-free gates (range, then FOV, both before the pair's
first random draw) now matched Python exactly on every tick, so the surplus had
to come from a later gate. Per-reason totals on the all-follow scenario, where
kinematics are identical throughout, located it:

| | accepted | dropout | outside FOV | terrain |
| --- | --- | --- | --- | --- |
| Python | 185 | 151 | 125 | 84 |
| Rust, look fixed only | 303 | 35 | 125 | 84 |
| Rust, this slice | 177 | 153 | 125 | 84 |

The detection draw was missing the world's `p_d`. Stage 6 was ported (P4.8c)
from the sensor model's detection probability alone, and the horizontal,
untilted look had been rejecting enough pairs at the FOV gate to make the totals
look plausible.

## What changed

* `SensorPoseSource` (stage 6) and `NominalTrajectory` (stage 5, already in
  place) are both implemented by `ControllerLink`, which records each tick's
  controller output **before** overrides and deconfliction move the platform. A
  request for any other timestamp is refused, not answered with a stale pose.
* `sensor_look_direction` / `fov_angle_deg`: the reference's arithmetic, in its
  order (normalise the sight line, then dot). `SensorSpec` gains
  `look_down_angle_deg`, carried in the bundle.
* `VisibilityOracle::sight_and_detection`: one LOS per pair, as the reference
  computes it, plus the world's detection factors. `SharedVisibility`
  implements it through `argusnet_world::visibility::detection`; oracles
  without a world (`AlwaysVisible`, unit-test stubs) return `None`, and stage 6
  falls back to the sensor model as before.
* Weather's sensing half: `visibility_at_range`, `bearing_noise_scale`, cloud
  layers. The bundle's weather document carries atmosphere and clouds.
* Python: `sensor_look_direction`, `fov_angle_deg` and `cloud_detection_factor`
  are lifted verbatim out of `build_observations` so fixtures can call them.
  Recapturing the committed scenario references after the lift reproduced them
  **identically**, which proves the lift preserved behaviour.

## Verification

| Test | What it proves |
| --- | --- |
| `the_look_direction_matches_the_reference` | 27 look directions (the 0.5 m/s gate from both sides, a 3-4-5 case exactly on it, a pure climb) and 72 FOV angles bit-exact; the default tilt is the reference's constant |
| `an_all_follow_scenario_matches_on_every_tick` | now also: every tick's range/FOV rejections equal the reference's **exactly** (125 FOV rejections), and the draw-dependent totals sit within a 3σ envelope (accepted 177 vs 185, dropout 153 vs 151) |
| `detection_matches_the_reference` | 396 detections under three weathers and two sensor ranges, including sight lines through a full cloud slab: `p_d`, effective noise, dominant loss and cloud factor bit-exact |
| `every_preset_matches_the_reference_bit_for_bit` | adds `visibility_at_range`, `bearing_noise_scale` and the cloud factor for 17 weathers |
| `the_assembly_wires_every_capability` | the assembled stage 6 senses through the bundle's world **and weather** (probed against the reference) |
| `observe::tests::the_world_detection_probability_gates_the_draw_and_names_the_loss`, `..._effective_noise_multiplier_scales_the_bearing_noise` | stage 6 consumes each detection field, with a fixed oracle |
| `nominal_poses_answer_only_for_their_own_tick` | a stale nominal pose is refused |

### Mutation round

21 mutants: **21 caught**, after one strengthening pass. The first pass caught
13. The survivors were the detection-model details a clear-weather,
short-range scenario cannot distinguish statistically (clouds, weather factor,
weather noise, the range norm), stage 6's consumption of each detection field,
the stale-pose guard, the default tilt, and the assembly's weather wiring. Each
is now pinned exactly: by the detection fixture, by fixed-oracle unit tests,
or by probing the assembled oracle.

One mutant confirmed the slice's premise directly. Dropping the pose source
(sensing from the flown position) fails the all-follow scenario's exact
draw-free comparison.

### Gate

Pinned toolchain 1.98.0-aarch64, exit codes read directly:

| Check | Result |
| --- | --- |
| `cargo test --workspace` | 1028 passed, 0 failed |
| `cargo clippy --workspace --all-targets -- -D warnings` | clean |
| `cargo fmt --all -- --check` | clean |
| `ruff check src/argusnet scripts tests` | clean |
| `pytest tests/` | 1007 passed, 9 skipped, 0 failed (no daemon timeouts on this run) |
| CI | **not run**: nothing is pushed |

## Not in this slice

* The bias random walk. The reference passes `pair_rng` to
  `sensor_model.apply_bias`; the Rust stage keeps a per-sensor stream (P4.8c).
  Its effect is on bearings, after acceptance. It is left for the statistical
  qualification in P4.9c.
* Stage 5 is still not assembled. Its nominal source now exists
  (`ControllerLink` implements `NominalTrajectory`).

## Next

**P4.8e.3**: the mission driver (POI assignment, egress, frontier) and
`scan_map_inspect` assembly, the default run.
