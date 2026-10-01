//! Drive `trajectory.rs` against what the Python reference returned.
//!
//! Fixture from `scripts/capture_trajectory_fixture.py`. Same technique as
//! `motion_parity.rs`: terrain and weather are **oracles**, keyed by the
//! arguments the reference passed, replayed through stubs that panic on an
//! unrecorded read.
//!
//! That matters more here than it did for the motion substrate, because the
//! call *pattern* carries behaviour. `apply_weather_adjustments` reads the
//! terrain twice — before the speed penalty so it sees a true AGL, and again
//! after the wind has moved the platform over different ground — and asks the
//! weather for a penalty at one altitude and a wind at another time. A port
//! that clamped once, or that passed the current timestamp to the penalty,
//! produces entirely plausible numbers and replays the wrong reads.

use std::cell::RefCell;
use std::collections::HashMap;

use argusnet_sim::motion::{BoundsXy, TerrainSampler};
use argusnet_sim::trajectory::{
    apply_weather_adjustments, LaunchController, Trajectory, WeatherAdjusted, WeatherContext,
    WeatherSampler,
};
use serde_json::Value;

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/trajectory.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_trajectory_fixture.py")
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

fn vec3(value: &Value) -> [f64; 3] {
    let items = value.as_array().expect("array");
    assert_eq!(items.len(), 3, "expected a 3-vector, got {value}");
    [num(&items[0]), num(&items[1]), num(&items[2])]
}

/// Python's `repr(float(x))`, reproduced so lookups agree by construction.
fn key(value: f64) -> String {
    let mut text = format!("{value:?}");
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

/// Answers height reads from the recording, records the order, panics on a miss.
struct ReplayTerrain {
    heights: HashMap<(String, String), f64>,
    calls: RefCell<Vec<(String, String)>>,
}

impl ReplayTerrain {
    fn from_case(case: &Value) -> Self {
        let mut heights = HashMap::new();
        for entry in case["terrain_oracle"].as_array().expect("terrain_oracle") {
            heights.insert(
                (
                    entry["x_m"].as_str().unwrap().to_string(),
                    entry["y_m"].as_str().unwrap().to_string(),
                ),
                num(&entry["height_m"]),
            );
        }
        Self {
            heights,
            calls: RefCell::new(Vec::new()),
        }
    }

    /// The reads the port made, in order, for comparison with the recorded
    /// ones. The count is the assertion that catches a single clamp.
    fn calls(&self) -> Vec<(String, String)> {
        self.calls.borrow().clone()
    }
}

impl TerrainSampler for ReplayTerrain {
    fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        let lookup = (key(x_m), key(y_m));
        self.calls.borrow_mut().push(lookup.clone());
        *self
            .heights
            .get(&lookup)
            .unwrap_or_else(|| panic!("height_at({x_m}, {y_m}) was never recorded"))
    }

    fn gradient_at(&self, x_m: f64, y_m: f64, _delta_m: Option<f64>) -> [f64; 2] {
        panic!("the weather path asked for a gradient at ({x_m}, {y_m}); the reference does not");
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, _z_m: f64, _min_agl_m: f64) -> f64 {
        panic!(
            "the weather path called clamp_altitude at ({x_m}, {y_m}); the reference takes a \
             plain max against height_at"
        );
    }
}

struct ReplayWeather {
    penalties: HashMap<String, f64>,
    winds: HashMap<(String, String), [f64; 3]>,
}

impl ReplayWeather {
    fn from_case(case: &Value) -> Self {
        let mut penalties = HashMap::new();
        for entry in case["penalty_oracle"].as_array().expect("penalty_oracle") {
            penalties.insert(
                entry["altitude_m"].as_str().unwrap().to_string(),
                num(&entry["penalty"]),
            );
        }
        let mut winds = HashMap::new();
        for entry in case["wind_oracle"].as_array().expect("wind_oracle") {
            winds.insert(
                (
                    entry["altitude_m"].as_str().unwrap().to_string(),
                    entry["time_s"].as_str().unwrap().to_string(),
                ),
                vec3(&entry["wind"]),
            );
        }
        Self { penalties, winds }
    }
}

impl WeatherSampler for ReplayWeather {
    fn flight_speed_penalty(&self, altitude_m: f64) -> f64 {
        *self.penalties.get(&key(altitude_m)).unwrap_or_else(|| {
            panic!(
                "flight_speed_penalty({altitude_m}) was never recorded -- the reference asked \
                 about a different altitude, so the clamp ordering differs"
            )
        })
    }

    fn wind_at(&self, altitude_m: f64, time_s: f64) -> [f64; 3] {
        *self
            .winds
            .get(&(key(altitude_m), key(time_s)))
            .unwrap_or_else(|| {
                panic!(
                    "wind_at({altitude_m}, {time_s}) was never recorded -- the reference asked \
                     at a different altitude or a different time"
                )
            })
    }
}

fn bounds_from(fixture: &Value) -> BoundsXy {
    let raw = &fixture["bounds"];
    BoundsXy {
        x_min_m: num(&raw["x_min_m"]),
        x_max_m: num(&raw["x_max_m"]),
        y_min_m: num(&raw["y_min_m"]),
        y_max_m: num(&raw["y_max_m"]),
    }
}

/// Replays recorded samples, and records which timestamps it was asked for.
struct ReplayInner {
    calls: Vec<f64>,
}

impl ReplayInner {
    fn new() -> Self {
        Self { calls: Vec::new() }
    }
}

impl Trajectory for ReplayInner {
    fn sample(&mut self, timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
        self.calls.push(timestamp_s);
        // The same deterministic stand-in the capture script used.
        (
            [300.0 + timestamp_s, 120.0 - timestamp_s, 260.0],
            [7.0, -3.0, 0.5],
        )
    }
}

#[test]
fn apply_weather_adjustments_matches_the_reference() {
    let fixture = fixture();
    let bounds = bounds_from(&fixture);
    let cases = fixture["apply_weather_adjustments"]
        .as_array()
        .expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let terrain = case["has_terrain"]
            .as_bool()
            .expect("flag")
            .then(|| ReplayTerrain::from_case(case));
        let weather = case["has_weather"]
            .as_bool()
            .expect("flag")
            .then(|| ReplayWeather::from_case(case));

        let context = WeatherContext {
            weather: weather.as_ref().map(|w| w as &dyn WeatherSampler),
            terrain: terrain.as_ref().map(|t| t as &dyn TerrainSampler),
            bounds: case["has_bounds"]
                .as_bool()
                .expect("flag")
                .then_some(bounds),
            min_agl_m: num(&case["min_agl_m"]),
            wind_scale: num(&case["wind_scale"]),
            dt_s: num(&case["dt_s"]),
        };

        let (position, velocity) = apply_weather_adjustments(
            vec3(&case["position"]),
            vec3(&case["velocity"]),
            num(&case["timestamp_s"]),
            &context,
        );

        assert_eq!(
            position,
            vec3(&case["adjusted_position"]),
            "{name}: position"
        );
        assert_eq!(
            velocity,
            vec3(&case["adjusted_velocity"]),
            "{name}: velocity"
        );

        // The reads themselves, in order. This is what separates one clamp
        // from two, and it cannot be inferred from the outputs when the
        // second clamp happens not to raise anything.
        if let Some(terrain) = &terrain {
            let recorded: Vec<(String, String)> = case["terrain_calls"]
                .as_array()
                .expect("terrain_calls")
                .iter()
                .map(|call| {
                    let pair = call.as_array().expect("pair");
                    (
                        pair[0].as_str().unwrap().to_string(),
                        pair[1].as_str().unwrap().to_string(),
                    )
                })
                .collect();
            assert_eq!(terrain.calls(), recorded, "{name}: terrain read pattern");
        }
    }
}

#[test]
fn the_launch_controller_matches_the_reference() {
    let fixture = fixture();
    let bounds = bounds_from(&fixture);
    let cases = fixture["launch_controller"].as_array().expect("cases");
    assert!(!cases.is_empty());

    for case in cases {
        let name = case["name"].as_str().expect("name");
        let has_weather = case["has_weather"].as_bool().expect("flag");
        let terrain = has_weather.then(|| ReplayTerrain::from_case(case));
        let weather = has_weather.then(|| ReplayWeather::from_case(case));
        let context = WeatherContext {
            weather: weather.as_ref().map(|w| w as &dyn WeatherSampler),
            terrain: terrain.as_ref().map(|t| t as &dyn TerrainSampler),
            bounds: has_weather.then_some(bounds),
            min_agl_m: 18.0,
            wind_scale: WeatherContext::DEFAULT_WIND_SCALE,
            // The launch path never passes one, so it gets the default.
            dt_s: WeatherContext::DEFAULT_DT_S,
        };

        let mut controller = LaunchController::new(
            vec3(&case["station_position"]),
            ReplayInner::new(),
            num(&case["climb_duration_s"]),
            num(&case["operational_altitude_agl"]),
        );
        if let Some(launch_at) = case["launch_at"].as_str() {
            controller.trigger_launch(launch_at.parse().expect("float"), "t-1");
        }
        if let Some(again) = case["trigger_twice_at"].as_str() {
            controller.trigger_launch(again.parse().expect("float"), "t-2");
        }

        let (position, velocity) = controller.sample(num(&case["sample_at"]), &context);

        assert_eq!(position, vec3(&case["position"]), "{name}: position");
        assert_eq!(velocity, vec3(&case["velocity"]), "{name}: velocity");
        assert_eq!(
            controller.is_launched(),
            case["launched"].as_bool().expect("launched"),
            "{name}: launched"
        );
        assert_eq!(
            controller.launch_target_id(),
            case["launch_target_id"].as_str(),
            "{name}: a second trigger_launch overwrote the first"
        );
    }
}

/// The wrapped trajectory must not be consulted before the platform flies.
///
/// A separate test because it asserts on something that did *not* happen,
/// which no output value records. The inner trajectory may be stateful, so a
/// sample the platform is not flying is a side effect rather than waste.
#[test]
fn the_inner_trajectory_is_consulted_only_when_operational() {
    let fixture = fixture();
    let bounds = bounds_from(&fixture);

    for case in fixture["launch_controller"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let has_weather = case["has_weather"].as_bool().expect("flag");
        let terrain = has_weather.then(|| ReplayTerrain::from_case(case));
        let weather = has_weather.then(|| ReplayWeather::from_case(case));
        let context = WeatherContext {
            weather: weather.as_ref().map(|w| w as &dyn WeatherSampler),
            terrain: terrain.as_ref().map(|t| t as &dyn TerrainSampler),
            bounds: has_weather.then_some(bounds),
            min_agl_m: 18.0,
            wind_scale: WeatherContext::DEFAULT_WIND_SCALE,
            dt_s: WeatherContext::DEFAULT_DT_S,
        };

        let mut controller = LaunchController::new(
            vec3(&case["station_position"]),
            ReplayInner::new(),
            num(&case["climb_duration_s"]),
            num(&case["operational_altitude_agl"]),
        );
        if let Some(launch_at) = case["launch_at"].as_str() {
            controller.trigger_launch(launch_at.parse().expect("float"), "t-1");
        }
        controller.sample(num(&case["sample_at"]), &context);

        let expected: Vec<f64> = case["inner_calls"]
            .as_array()
            .expect("inner_calls")
            .iter()
            .map(|t| t.as_str().unwrap().parse().expect("float"))
            .collect();
        // `trigger_twice_at` is skipped above, which does not change who was
        // consulted -- the relaunch is ignored either way.
        if case["trigger_twice_at"].is_null() {
            assert_eq!(
                controller_inner_calls(&controller),
                expected,
                "{name}: the operational trajectory was consulted at the wrong times"
            );
        }
    }
}

/// Reach the stand-in's call log through the controller.
fn controller_inner_calls(controller: &LaunchController<ReplayInner>) -> Vec<f64> {
    controller.operational().calls.clone()
}

#[test]
fn the_weather_wrapper_matches_the_reference() {
    let fixture = fixture();
    let bounds = bounds_from(&fixture);

    for case in fixture["weather_adjusted_trajectory"]
        .as_array()
        .expect("cases")
    {
        let name = case["name"].as_str().expect("name");
        let terrain = ReplayTerrain::from_case(case);
        let weather = ReplayWeather::from_case(case);
        let context = WeatherContext {
            weather: Some(&weather),
            terrain: Some(&terrain),
            bounds: Some(bounds),
            min_agl_m: 18.0,
            wind_scale: WeatherContext::DEFAULT_WIND_SCALE,
            dt_s: num(&case["dt_s"]),
        };

        let mut wrapper = WeatherAdjusted::new(ReplayInner::new());
        let (position, velocity) = wrapper.sample(num(&case["sample_at"]), &context);

        assert_eq!(position, vec3(&case["position"]), "{name}: position");
        assert_eq!(velocity, vec3(&case["velocity"]), "{name}: velocity");

        let expected: Vec<f64> = case["inner_calls"]
            .as_array()
            .expect("inner_calls")
            .iter()
            .map(|t| t.as_str().unwrap().parse().expect("float"))
            .collect();
        assert_eq!(wrapper.inner().calls, expected, "{name}: inner calls");
    }
}

/// `reset_state` reaches the wrapped trajectory through both wrappers.
///
/// The reference does this with `hasattr(…, "reset_state")`, so a port that
/// forgot it fails silently: the run keeps working and the inner trajectory
/// carries the previous epoch's state into the new one.
#[test]
fn a_reset_reaches_the_wrapped_trajectory() {
    #[derive(Default)]
    struct Counting {
        resets: usize,
    }

    impl Trajectory for Counting {
        fn sample(&mut self, _timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
            ([0.0; 3], [0.0; 3])
        }
        fn reset_state(&mut self) {
            self.resets += 1;
        }
    }

    let mut wrapper = WeatherAdjusted::new(Counting::default());
    wrapper.reset_state();
    assert_eq!(wrapper.inner().resets, 1, "the wrapper swallowed the reset");

    let mut launch = LaunchController::new([0.0; 3], Counting::default(), 8.0, 230.0);
    launch.trigger_launch(5.0, "t-1");
    launch.reset_state();
    assert_eq!(
        launch.operational().resets,
        1,
        "the launch controller swallowed the reset"
    );
    assert!(!launch.is_launched(), "the platform stayed launched");
    assert_eq!(launch.launch_target_id(), None);

    // And it can launch again afterwards, against a new target.
    launch.trigger_launch(9.0, "t-2");
    assert_eq!(launch.launch_target_id(), Some("t-2"));
}

/// The fixture is only evidence if it still reaches what it claims to.
#[test]
fn the_fixture_still_covers_what_it_claims() {
    let fixture = fixture();
    let weather: HashMap<&str, &Value> = fixture["apply_weather_adjustments"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();

    let full = weather["full_path"];
    assert_eq!(
        full["terrain_calls"].as_array().unwrap().len(),
        2,
        "the terrain is no longer read twice, so one clamp would pass"
    );
    assert_ne!(
        full["terrain_calls"][0], full["terrain_calls"][1],
        "both reads are at the same point"
    );
    assert_eq!(
        weather["below_floor_is_clamped_before_the_penalty"]["penalty_oracle"][0]["altitude_m"]
            .as_str()
            .unwrap(),
        "18.0",
        "the penalty no longer sees a clamped altitude"
    );
    assert!(
        weather["no_weather_is_a_passthrough"]["terrain_calls"]
            .as_array()
            .unwrap()
            .is_empty(),
        "the passthrough now reads terrain"
    );

    let launch: HashMap<&str, &Value> = fixture["launch_controller"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| (c["name"].as_str().unwrap(), c))
        .collect();
    assert_eq!(
        num(&launch["quarter_through_climb"]["velocity"][2]),
        num(&launch["three_quarters_through_climb"]["velocity"][2]),
        "the smooth-step rate is no longer symmetric"
    );
    assert_eq!(
        num(&launch["at_launch_instant"]["velocity"][2]),
        0.0,
        "the climb no longer starts at zero rate"
    );
    assert!(
        launch["exactly_at_climb_end_is_operational"]["inner_calls"]
            .as_array()
            .unwrap()
            .len()
            == 1,
        "the strict climb boundary is no longer exercised"
    );
}
