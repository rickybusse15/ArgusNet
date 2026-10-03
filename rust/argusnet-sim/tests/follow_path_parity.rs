//! Drive `follow_path.rs` against what the Python reference returned.
//!
//! Fixture from `scripts/capture_follow_path_fixture.py`: 26 cases, 129
//! samples, ~7500 recorded reads. Terrain, obstacles, line of sight and the
//! planner are all **oracles** — replayed from the recording, panicking on
//! anything the reference never asked for.
//!
//! That is doing more work here than in any earlier slice. This controller
//! *chooses* between candidates using those reads, so the call pattern is
//! the decision: a port that tested visibility before the obstacle, or
//! skipped the per-tick planner check on the stored goal, asks about points
//! the reference never mentioned and fails on a lookup rather than on a
//! number that happens to agree.
//!
//! # Cases are sequences, not samples
//!
//! `FollowPathState` carries a memo, the previous position, heading, route
//! and smoothed terrain height across ticks. Most of the interesting
//! behaviour is unreachable in one call — acceleration limiting needs a
//! previous velocity, the orbit blend a previous position — so each case
//! replays a whole sequence through one controller and checks every sample.
//!
//! Bit for bit on the platform the fixture was captured on; elsewhere within
//! `common::parity::MAX_ULPS`, lookups included.

mod common;

use std::collections::HashMap;

use argusnet_sim::follow_path::{
    FollowPathConfig, FollowPathContext, FollowPathController, MissionZone, VisibilitySampler,
};
use argusnet_sim::motion::{BoundsXy, ObstacleSampler, TerrainSampler};
use common::parity::{parity_of, Parity, ReplayTable};
use serde_json::Value;

/// The fixture's two target trajectories, boxed so a case can pick one.
type TargetFn = Box<dyn FnMut(f64) -> ([f64; 3], [f64; 3])>;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/follow_path.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_follow_path_fixture.py")
    });
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

fn num(value: &Value) -> f64 {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected a stringified float, got {value}"))
        .parse()
        .expect("stringified float parses")
}

fn vec2(value: &Value) -> [f64; 2] {
    let items = value.as_array().expect("array");
    [num(&items[0]), num(&items[1])]
}

fn vec3(value: &Value) -> [f64; 3] {
    let items = value.as_array().expect("array");
    [num(&items[0]), num(&items[1]), num(&items[2])]
}

/// Python's `repr`-stringified floats parse back to the same double.
fn nums(entry: &Value, fields: &[&str]) -> Vec<f64> {
    fields.iter().map(|field| num(&entry[*field])).collect()
}

/// An absent optional argument (the reference's default `delta_m`).
const DEFAULT: f64 = f64::NAN;

struct ReplayTerrain {
    heights: ReplayTable<f64>,
    gradients: ReplayTable<[f64; 2]>,
    clamps: ReplayTable<f64>,
}

impl ReplayTerrain {
    fn from_case(case: &Value, parity: Parity) -> Self {
        let oracle = &case["terrain_oracle"];
        let mut heights = ReplayTable::new(parity);
        for entry in oracle["heights"].as_array().expect("heights") {
            heights.insert(nums(entry, &["x_m", "y_m"]), num(&entry["height_m"]));
        }
        let mut gradients = ReplayTable::new(parity);
        for entry in oracle["gradients"].as_array().expect("gradients") {
            let delta = match entry["delta_m"].as_str().unwrap() {
                "default" => DEFAULT,
                _ => num(&entry["delta_m"]),
            };
            gradients.insert(
                vec![num(&entry["x_m"]), num(&entry["y_m"]), delta],
                vec2(&entry["gradient"]),
            );
        }
        let mut clamps = ReplayTable::new(parity);
        for entry in oracle["clamps"].as_array().expect("clamps") {
            clamps.insert(
                nums(entry, &["x_m", "y_m", "z_m", "min_agl_m"]),
                num(&entry["clamped_z_m"]),
            );
        }
        Self {
            heights,
            gradients,
            clamps,
        }
    }
}

impl TerrainSampler for ReplayTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        *self
            .heights
            .get(&[x_m, y_m])
            .unwrap_or_else(|| panic!("height_at({x_m}, {y_m}) was never recorded"))
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2] {
        *self
            .gradients
            .get(&[x_m, y_m, delta_m.unwrap_or(DEFAULT)])
            .unwrap_or_else(|| panic!("gradient_at({x_m}, {y_m}, {delta_m:?}) was never recorded"))
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        *self
            .clamps
            .get(&[x_m, y_m, z_m, min_agl_m])
            .unwrap_or_else(|| {
                panic!("clamp_altitude({x_m}, {y_m}, {z_m}, {min_agl_m}) was never recorded")
            })
    }
}

struct ReplayObstacles {
    probes: ReplayTable<bool>,
    pushes: ReplayTable<[f64; 2]>,
}

impl ReplayObstacles {
    fn from_case(case: &Value, parity: Parity) -> Self {
        let mut probes = ReplayTable::new(parity);
        for entry in case["obstacle_oracle"].as_array().expect("obstacle_oracle") {
            probes.insert(
                nums(entry, &["x_m", "y_m", "z_m"]),
                entry["collides"].as_bool().expect("collides"),
            );
        }
        let mut pushes = ReplayTable::new(parity);
        for entry in case["obstacle_push_oracle"]
            .as_array()
            .expect("obstacle_push_oracle")
        {
            pushes.insert(
                nums(entry, &["x_m", "y_m", "margin_m"]),
                vec2(&entry["pushed_xy"]),
            );
        }
        Self { probes, pushes }
    }
}

impl ObstacleSampler for ReplayObstacles {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        let collides = *self
            .probes
            .get(&point_m)
            .unwrap_or_else(|| panic!("point_collides{point_m:?} was never recorded"));
        if !collides {
            return None;
        }
        Some(
            *self
                .pushes
                .get(&[point_m[0], point_m[1], margin_m])
                .unwrap_or_else(|| {
                    panic!("push_outside_xy({point_m:?}, {margin_m}) was never recorded")
                }),
        )
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        *self.probes.get(&point_m).unwrap_or_else(|| {
            panic!(
                "point_collides{point_m:?} was never recorded -- the port probed a candidate \
                 the reference did not"
            )
        })
    }
}

struct ReplayVisibility {
    results: ReplayTable<bool>,
}

impl ReplayVisibility {
    fn from_case(case: &Value, parity: Parity) -> Self {
        let mut results = ReplayTable::new(parity);
        for entry in case["los_oracle"].as_array().expect("los_oracle") {
            let mut key = vec3(&entry["observer"]).to_vec();
            key.extend(vec3(&entry["target"]));
            results.insert(key, entry["visible"].as_bool().expect("v"));
        }
        Self { results }
    }
}

impl VisibilitySampler for ReplayVisibility {
    fn is_visible(&self, observer_m: [f64; 3], target_m: [f64; 3]) -> bool {
        let query = [observer_m, target_m].concat();
        *self.results.get(&query).unwrap_or_else(|| {
            panic!(
                "los({observer_m:?} -> {target_m:?}) was never recorded -- the port asked about \
                 a candidate position the reference never evaluated"
            )
        })
    }
}

fn config_from(case: &Value, bounds: BoundsXy) -> FollowPathConfig {
    let c = &case["config"];
    let opt = |name: &str| c[name].as_str().map(|v| v.parse::<f64>().expect("float"));
    FollowPathConfig {
        map_bounds_m: bounds,
        base_agl_m: num(&c["base_agl_m"]),
        standoff_radius_m: num(&c["standoff_radius_m"]),
        map_scale: num(&c["map_scale"]),
        max_speed_mps: num(&c["max_speed_mps"]),
        phase: num(&c["phase"]),
        lead_s: num(&c["lead_s"]),
        candidate_count: num(&c["candidate_count"]) as usize,
        rotation_rate_rad_s: num(&c["rotation_rate_rad_s"]),
        min_agl_m: num(&c["min_agl_m"]),
        vertical_amplitude_m: num(&c["vertical_amplitude_m"]),
        vertical_frequency_rad_s: num(&c["vertical_frequency_rad_s"]),
        target_altitude_offset_m: opt("target_altitude_offset_m"),
        max_accel_mps2: opt("max_accel_mps2"),
        terrain_following: c["terrain_following"].as_bool().expect("flag"),
        terrain_following_agl_m: num(&c["terrain_following_agl_m"]),
        terrain_following_smoothing_s: num(&c["terrain_following_smoothing_s"]),
        slot_index: num(&c["slot_index"]) as usize,
        slot_count: num(&c["slot_count"]) as usize,
    }
}

/// The level target most cases drove: `[200 + 6t, 100 + 2t, 140]`, velocity
/// `[6, 2, 0]`. Reproduced rather than recorded because it is the test's own
/// input, not the reference's output.
fn level_target(timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
    (
        [200.0 + 6.0 * timestamp_s, 100.0 + 2.0 * timestamp_s, 140.0],
        [6.0, 2.0, 0.0],
    )
}

/// A target losing 30 m/s, used only by the follow-floor case.
///
/// The floor compares the composed altitude against
/// `current_target_z + offset * 0.3`, while the composition uses the *lead*
/// sample — so with a level target the two agree and the floor is
/// unreachable at any offset. Separating them is the only way to reach it.
fn descending_target(timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
    (
        [
            200.0 + 6.0 * timestamp_s,
            100.0 + 2.0 * timestamp_s,
            400.0 - 30.0 * timestamp_s,
        ],
        [6.0, 2.0, -30.0],
    )
}

#[test]
fn follow_path_matches_the_reference() {
    let fixture = fixture();
    let raw = &fixture["bounds"];
    let bounds = BoundsXy {
        x_min_m: num(&raw["x_min_m"]),
        x_max_m: num(&raw["x_max_m"]),
        y_min_m: num(&raw["y_min_m"]),
        y_max_m: num(&raw["y_max_m"]),
    };

    let parity = parity_of(&fixture);
    let cases = fixture["cases"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the fixture recorded no cases");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        assert!(
            !case["has_planner"].as_bool().expect("flag"),
            "{name}: the fixture recorded a planner, which this test cannot replay yet"
        );

        let terrain = ReplayTerrain::from_case(case, parity);
        let obstacles = ReplayObstacles::from_case(case, parity);
        let visibility = ReplayVisibility::from_case(case, parity);
        let zones: Vec<MissionZone> = case["zones"]
            .as_array()
            .expect("zones")
            .iter()
            .map(|zone| MissionZone {
                zone_id: zone["zone_id"].as_str().unwrap().to_string(),
                zone_type: zone["zone_type"].as_str().unwrap().to_string(),
                center_m: vec3(&zone["center"]),
                radius_m: num(&zone["radius_m"]),
            })
            .collect();

        let context = FollowPathContext {
            terrain: &terrain,
            obstacles: &obstacles,
            visibility: &visibility,
            planner: None,
            zones: &zones,
        };

        let mut target: TargetFn = match case["target_kind"].as_str().expect("target_kind") {
            "descending" => Box::new(descending_target),
            "level" => Box::new(level_target),
            other => panic!("{name}: unknown target kind {other:?}"),
        };

        let mut controller = FollowPathController::new(config_from(case, bounds));
        if let Some(seed) = case["seed_with"].as_object() {
            controller.seed(
                vec3(&seed["position"]),
                vec3(&seed["velocity"]),
                num(&seed["timestamp_s"]),
            );
        }

        for (index, sample) in case["samples"]
            .as_array()
            .expect("samples")
            .iter()
            .enumerate()
        {
            let timestamp_s = num(&sample["timestamp_s"]);
            let (position, velocity) = controller.sample(timestamp_s, target.as_mut(), &context);
            let (want_position, want_velocity) =
                (vec3(&sample["position"]), vec3(&sample["velocity"]));
            assert!(
                parity.same_all(&position, &want_position),
                "{name}: position at sample {index} (t = {timestamp_s}): \
                 {position:?} against {want_position:?}"
            );
            assert!(
                parity.same_all(&velocity, &want_velocity),
                "{name}: velocity at sample {index} (t = {timestamp_s}): \
                 {velocity:?} against {want_velocity:?}"
            );
        }
    }
}

/// The fixture is only evidence if it still reaches what it claims to.
///
/// These re-check, against the committed file, the handful of properties
/// that make the cases discriminating. The capture script asserts them too,
/// but the fixture is committed and the script is not run in CI — so a
/// stale or hand-edited fixture could otherwise turn the parity test above
/// into a formality.
#[test]
fn the_fixture_still_covers_what_it_claims() {
    let fixture = fixture();
    let by_name: HashMap<&str, &Value> = fixture["cases"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();

    let positions = |name: &str| -> Vec<[f64; 3]> {
        by_name[name]["samples"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| vec3(&s["position"]))
            .collect()
    };

    // The seeded controller flies.
    let orbit = positions("plain_orbit");
    assert!(
        orbit.first() != orbit.last(),
        "the seeded orbit never moved"
    );

    // The unseeded one does not, and reports a changing velocity while it
    // sits still. Both halves of the finding.
    let frozen = positions("unseeded_never_flies");
    assert!(
        frozen.windows(2).all(|w| w[0] == w[1]),
        "the unseeded freeze no longer holds"
    );
    let frozen_velocities: Vec<[f64; 3]> = by_name["unseeded_never_flies"]["samples"]
        .as_array()
        .unwrap()
        .iter()
        .map(|s| vec3(&s["velocity"]))
        .collect();
    assert!(
        frozen_velocities.windows(2).any(|w| w[0] != w[1]),
        "the frozen controller now reports a constant velocity too"
    );

    // The dead smoothing filter: opposite ends of its range agree exactly.
    assert_eq!(
        positions("terrain_following_fast_filter"),
        positions("terrain_following_slow_filter"),
        "the smoothing constant now changes the output -- the EMA has been \
         wired up and the port must consume it"
    );

    // Each capability actually changed a decision somewhere.
    assert_ne!(
        positions("obstacle_rejects_candidates"),
        orbit[..4].to_vec(),
        "the obstacle no longer changes the chosen path"
    );
    assert_ne!(
        positions("blocked_line_of_sight"),
        orbit[..4].to_vec(),
        "blocked line of sight no longer changes the chosen path"
    );
    assert_ne!(
        positions("exclusion_zone"),
        orbit[..4].to_vec(),
        "the exclusion zone is no longer enforced"
    );
    assert_eq!(
        positions("surveillance_zone_is_not_enforced"),
        orbit[..3].to_vec(),
        "a soft zone is now being enforced as an exclusion"
    );
    assert_ne!(
        positions("cooperative_slot_0_of_4"),
        positions("cooperative_slot_1_of_4"),
        "two cooperative slots now fly the same path"
    );

    // The follow floor binds exactly at `target_z + 0.3 * offset`.
    for sample in by_name["follow_floor_raises_the_platform"]["samples"]
        .as_array()
        .unwrap()
    {
        let t = num(&sample["timestamp_s"]);
        let expected = (400.0 - 30.0 * t) + (-50.0 * 0.3);
        assert!(
            (num(&sample["position"][2]) - expected).abs() < 1e-9,
            "the follow floor no longer binds at t = {t}"
        );
    }

    // Overlapping zones leave the platform outside both.
    for position in positions("overlapping_exclusion_zones") {
        for (cx, cy) in [(380.0, 200.0), (470.0, 260.0)] {
            let distance = (position[0] - cx).hypot(position[1] - cy);
            assert!(
                distance >= 150.0 - 1e-6,
                "a sample stayed inside the zone at ({cx}, {cy})"
            );
        }
    }

    // Every exclusion sample really is outside the zone.
    for position in positions("exclusion_zone") {
        let distance = (position[0] - 380.0).hypot(position[1] - 200.0);
        assert!(
            distance >= 150.0 - 1e-6,
            "a sample stayed inside the exclusion zone, {distance} m from its centre"
        );
    }
}
