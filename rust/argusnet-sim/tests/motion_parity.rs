//! Drive `motion.rs` against what the Python reference actually returned.
//!
//! The fixture comes from `scripts/capture_motion_fixture.py`, which calls
//! `sim.py` and records the results. Nothing here recomputes a reference
//! value: a test that did would agree with a port that shared its author's
//! misreading while both disagreed with Python.
//!
//! # The terrain is a replay, and the keys are half the test
//!
//! Terrain sampling is already Rust (P4.3) and so is the obstacle push
//! (Phase 3); re-checking them here would test other people's slices. So the
//! fixture records every read the reference made — keyed by `(x, y)` for
//! heights, `(x, y, delta)` for gradients, `(x, y, z, min_agl)` for clamps —
//! and [`ReplayTerrain`] answers from that table and **panics on a miss**.
//!
//! That makes the keys part of the assertion. A port that asked for the
//! gradient at the wrong finite-difference step, or that called
//! `clamp_altitude` on the path where the reference takes a plain `max`,
//! fails on a lookup even though its arithmetic would have produced a
//! perfectly plausible number.
//!
//! # Exact comparison
//!
//! Values are compared bit-for-bit against the reference's `repr(float(…))`.
//! These are compositions of reads, not iterative numerics, so every
//! operation is a multiply, an add or a comparison on values Python and Rust
//! both hold as f64 — there is no room for a legitimate 1-ULP disagreement,
//! and admitting a tolerance would hide a real divergence.

use std::collections::HashMap;

use argusnet_sim::motion::{
    advance_along_polyline, collision_aware_position, compose_air_state, compose_terrain_state,
    BoundsXy, ObstacleSampler, TargetAltitudeHold, TerrainSampler,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/motion.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_motion_fixture.py")
    });
    serde_json::from_str(&text).expect("fixture is valid JSON")
}

/// Parse the reference's `repr(float(x))` rendering back to the same f64.
fn num(value: &Value) -> f64 {
    value
        .as_str()
        .unwrap_or_else(|| panic!("expected a stringified float, got {value}"))
        .parse()
        .expect("stringified float parses")
}

fn vec2(value: &Value) -> [f64; 2] {
    let items = value.as_array().expect("array");
    assert_eq!(items.len(), 2, "expected a 2-vector, got {value}");
    [num(&items[0]), num(&items[1])]
}

fn vec3(value: &Value) -> [f64; 3] {
    let items = value.as_array().expect("array");
    assert_eq!(items.len(), 3, "expected a 3-vector, got {value}");
    [num(&items[0]), num(&items[1]), num(&items[2])]
}

/// A float's exact key. Reusing the reference's own string rendering means
/// the lookup agrees with the recording by construction, with no rounding
/// policy of its own to get wrong.
fn key(value: f64) -> String {
    let mut text = format!("{value:?}");
    // Rust renders integral floats as "1.0" and so does Python's repr, but
    // Rust writes "1e-9" where Python writes "1e-09". The fixture's keys come
    // from Python, so normalise toward it.
    if let Some(index) = text.find('e') {
        let (mantissa, exponent) = text.split_at(index);
        let exponent = &exponent[1..];
        let (sign, digits) = match exponent.strip_prefix('-') {
            Some(rest) => ("-", rest),
            None => ("+", exponent),
        };
        text = format!("{mantissa}e{sign}{digits:0>2}");
    }
    text
}

/// Answers terrain reads from the recording, and panics on anything else.
struct ReplayTerrain {
    heights: HashMap<(String, String), f64>,
    gradients: HashMap<(String, String, String), [f64; 2]>,
    clamps: HashMap<(String, String, String, String), f64>,
}

impl ReplayTerrain {
    fn from_fixture(fixture: &Value) -> Self {
        let oracle = &fixture["terrain_oracle"];
        let mut heights = HashMap::new();
        for entry in oracle["heights"].as_array().expect("heights") {
            heights.insert(
                (
                    entry["x_m"].as_str().unwrap().to_string(),
                    entry["y_m"].as_str().unwrap().to_string(),
                ),
                num(&entry["height_m"]),
            );
        }
        let mut gradients = HashMap::new();
        for entry in oracle["gradients"].as_array().expect("gradients") {
            gradients.insert(
                (
                    entry["x_m"].as_str().unwrap().to_string(),
                    entry["y_m"].as_str().unwrap().to_string(),
                    entry["delta_m"].as_str().unwrap().to_string(),
                ),
                vec2(&entry["gradient"]),
            );
        }
        let mut clamps = HashMap::new();
        for entry in oracle["clamps"].as_array().expect("clamps") {
            clamps.insert(
                (
                    entry["x_m"].as_str().unwrap().to_string(),
                    entry["y_m"].as_str().unwrap().to_string(),
                    entry["z_m"].as_str().unwrap().to_string(),
                    entry["min_agl_m"].as_str().unwrap().to_string(),
                ),
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
            .get(&(key(x_m), key(y_m)))
            .unwrap_or_else(|| panic!("height_at({x_m}, {y_m}) was never recorded"))
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2] {
        let delta_key = delta_m.map_or_else(|| "default".to_string(), key);
        *self
            .gradients
            .get(&(key(x_m), key(y_m), delta_key.clone()))
            .unwrap_or_else(|| {
                panic!(
                    "gradient_at({x_m}, {y_m}, {delta_key}) was never recorded -- the reference \
                     asked for a different finite-difference step"
                )
            })
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        *self
            .clamps
            .get(&(key(x_m), key(y_m), key(z_m), key(min_agl_m)))
            .unwrap_or_else(|| {
                panic!(
                    "clamp_altitude({x_m}, {y_m}, {z_m}, {min_agl_m}) was never recorded -- the \
                     reference did not clamp here"
                )
            })
    }
}

/// Replays the recorded probe sequence, in order, and checks the port asked
/// for the points the reference asked for.
struct ReplayObstacles {
    probes: Vec<(f64, f64, f64, Option<[f64; 2]>)>,
    next: std::cell::Cell<usize>,
}

impl ReplayObstacles {
    fn from_case(case: &Value) -> Self {
        let probes = case["probes"]
            .as_array()
            .expect("probes")
            .iter()
            .map(|probe| {
                let point = vec3(&probe["point"]);
                let pushed = if probe["collides"].as_bool().expect("collides") {
                    assert_eq!(
                        num(&probe["margin_m"]),
                        1.0,
                        "the fixture recorded a push at a margin other than 1.0"
                    );
                    Some(vec2(&probe["pushed_xy"]))
                } else {
                    None
                };
                (point[0], point[1], point[2], pushed)
            })
            .collect();
        Self {
            probes,
            next: std::cell::Cell::new(0),
        }
    }

    fn probes_consumed(&self) -> usize {
        self.next.get()
    }
}

impl ObstacleSampler for ReplayObstacles {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        assert_eq!(
            margin_m, 1.0,
            "the port pushed with a margin other than 1.0"
        );
        let index = self.next.get();
        let probe = self.probes.get(index).unwrap_or_else(|| {
            panic!(
                "the port probed {point_m:?} on iteration {index}, but the reference made only {} \
                 probes -- the loop ran too long",
                self.probes.len()
            )
        });
        assert_eq!(
            [probe.0, probe.1, probe.2],
            point_m,
            "probe {index} was made at the wrong point"
        );
        self.next.set(index + 1);
        probe.3
    }
}

#[test]
fn compose_terrain_state_matches_the_reference() {
    let fixture = fixture();
    let terrain = ReplayTerrain::from_fixture(&fixture);
    let cases = fixture["compose_terrain_state"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the fixture recorded no cases");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let (position, velocity) = compose_terrain_state(
            &terrain,
            vec2(&case["xy_m"]),
            vec2(&case["xy_velocity_mps"]),
            num(&case["agl_m"]),
            num(&case["agl_rate_mps"]),
            num(&case["min_agl_m"]),
            case["terrain_speed_adaptation"].as_bool().expect("flag"),
        );
        assert_eq!(position, vec3(&case["position"]), "{name}: position");
        assert_eq!(velocity, vec3(&case["velocity"]), "{name}: velocity");
    }
}

#[test]
fn compose_air_state_matches_the_reference() {
    let fixture = fixture();
    let terrain = ReplayTerrain::from_fixture(&fixture);
    let cases = fixture["compose_air_state"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the fixture recorded no cases");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        // The reference's guard is an `or` over two separately-optional
        // arguments; the port folds them into one `Option`, so the fixture's
        // "position but no offset" and "offset but no position" cases both
        // have to map to `None` here. That mapping is the thing under test as
        // much as the arithmetic is.
        let hold = match (
            case["target_position"].as_array(),
            case["target_altitude_offset_m"].as_str(),
        ) {
            (Some(_), Some(_)) => Some(TargetAltitudeHold {
                position_m: vec3(&case["target_position"]),
                velocity_mps: case["target_velocity"]
                    .as_array()
                    .map(|_| vec3(&case["target_velocity"])),
                offset_m: num(&case["target_altitude_offset_m"]),
                rate_offset_mps: case["target_altitude_rate_offset_mps"]
                    .as_str()
                    .map(|_| num(&case["target_altitude_rate_offset_mps"])),
            }),
            _ => None,
        };

        let (position, velocity) = compose_air_state(
            &terrain,
            vec2(&case["xy_m"]),
            vec2(&case["xy_velocity_mps"]),
            num(&case["min_agl_m"]),
            num(&case["nominal_agl_m"]),
            num(&case["agl_rate_mps"]),
            hold,
        );
        assert_eq!(position, vec3(&case["position"]), "{name}: position");
        assert_eq!(velocity, vec3(&case["velocity"]), "{name}: velocity");
    }
}

#[test]
fn collision_aware_position_matches_the_reference() {
    let fixture = fixture();
    let terrain = ReplayTerrain::from_fixture(&fixture);
    let cases = fixture["collision_aware_position"]
        .as_array()
        .expect("cases");
    assert!(!cases.is_empty(), "the fixture recorded no cases");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let has_layer = case["has_obstacle_layer"].as_bool().expect("flag");
        let obstacles = has_layer.then(|| ReplayObstacles::from_case(case));

        let resolution = collision_aware_position(
            &terrain,
            obstacles
                .as_ref()
                .map(|sampler| sampler as &dyn ObstacleSampler),
            vec3(&case["position"]),
            num(&case["min_agl_m"]),
        );

        assert_eq!(
            resolution.position_m,
            vec3(&case["result_position"]),
            "{name}: position"
        );
        // Every recorded probe was consumed: a port that returned early would
        // leave some unused, and the sampler itself catches running over.
        if let Some(sampler) = &obstacles {
            assert_eq!(
                sampler.probes_consumed(),
                sampler.probes.len(),
                "{name}: the port made fewer probes than the reference"
            );
        }
        // None of the fixture's cases wedge, so none should report unresolved.
        assert!(!resolution.unresolved, "{name}: unexpectedly unresolved");
    }
}

#[test]
fn advance_along_polyline_matches_the_reference() {
    let fixture = fixture();
    let cases = fixture["advance_along_polyline"].as_array().expect("cases");
    assert!(!cases.is_empty(), "the fixture recorded no cases");

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let points: Vec<[f64; 2]> = case["points_xy"]
            .as_array()
            .expect("points")
            .iter()
            .map(vec2)
            .collect();
        let expected_remaining: Vec<[f64; 2]> = case["remaining_xy"]
            .as_array()
            .expect("remaining")
            .iter()
            .map(vec2)
            .collect();

        let advance = advance_along_polyline(&points, num(&case["travel_distance_m"]))
            .unwrap_or_else(|e| panic!("{name}: {e}"));

        assert_eq!(advance.cursor_xy, vec2(&case["new_xy"]), "{name}: cursor");
        assert_eq!(
            advance.direction,
            vec2(&case["direction"]),
            "{name}: direction"
        );
        assert_eq!(
            advance.remaining_xy, expected_remaining,
            "{name}: remaining polyline"
        );
    }
}

#[test]
fn bounds_match_the_reference() {
    let fixture = fixture();
    let raw = &fixture["bounds"]["bounds"];
    let bounds = BoundsXy {
        x_min_m: num(&raw["x_min_m"]),
        x_max_m: num(&raw["x_max_m"]),
        y_min_m: num(&raw["y_min_m"]),
        y_max_m: num(&raw["y_max_m"]),
    };

    for case in fixture["bounds"]["within"].as_array().expect("within") {
        let name = case["name"].as_str().expect("name");
        assert_eq!(
            bounds.contains(vec2(&case["xy_m"]), num(&case["margin_m"])),
            case["within"].as_bool().expect("within"),
            "{name}"
        );
    }

    for case in fixture["bounds"]["clamp"].as_array().expect("clamp") {
        let name = case["name"].as_str().expect("name");
        assert_eq!(
            bounds.clamp(vec2(&case["xy_m"]), num(&case["margin_m"])),
            vec2(&case["clamped_xy_m"]),
            "{name}"
        );
    }
}

/// The fixture is only evidence if it still reaches the branches it claims to.
///
/// `capture_motion_fixture.py` asserts this at capture time, but the fixture
/// is committed and the script is not run in CI. Re-checking the handful of
/// properties that make the cases discriminating means a hand-edited or
/// stale fixture cannot quietly turn these tests into a formality.
#[test]
fn the_fixture_still_covers_what_it_claims() {
    let fixture = fixture();

    let terrain: HashMap<&str, &Value> = fixture["compose_terrain_state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();
    assert_eq!(
        num(&terrain["clamped_and_descending"]["velocity"][2]),
        0.0,
        "the descent guard is no longer exercised"
    );
    assert!(
        num(&terrain["downhill_descent_unclamped"]["velocity"][2]) < 0.0,
        "no case now descends without the clamp firing"
    );
    assert!(
        num(&terrain["clamped_and_ascending"]["velocity"][2]) > 0.0,
        "no case now ascends while clamped"
    );

    let air: HashMap<&str, &Value> = fixture["compose_air_state"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();
    assert!(
        num(&air["target_term_wins_max"]["position"][2]) > 300.0,
        "the target term no longer wins the max anywhere"
    );

    let collision: HashMap<&str, &Value> = fixture["collision_aware_position"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();
    assert!(
        collision["inside_obstacle_pushed_out"]["probes"]
            .as_array()
            .unwrap()
            .len()
            >= 2,
        "the push loop no longer iterates in any case"
    );
    assert!(
        collision["no_layer_returns_inputs"]["probes"]
            .as_array()
            .unwrap()
            .is_empty(),
        "the no-layer path now probes"
    );

    let polyline: HashMap<&str, &Value> = fixture["advance_along_polyline"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();
    assert_eq!(
        vec2(&polyline["exactly_on_vertex"]["direction"]),
        [1.0, 0.0],
        "the inclusive-boundary case no longer lands on the first segment"
    );
    assert_eq!(
        vec2(&polyline["all_degenerate_zero_direction"]["direction"]),
        [0.0, 0.0],
        "the zero-direction case no longer returns zero"
    );
}
