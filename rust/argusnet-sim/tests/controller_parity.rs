//! Drive `waypoint_path.rs` and `observation_follow.rs` against the reference.
//!
//! Fixture from `scripts/capture_controller_fixture.py`. Terrain and obstacle
//! reads are oracles that panic on a miss, exactly as in
//! `follow_path_parity.rs`; the observation-triggered wrapper is compared as a
//! call log, because what it does is *choose* -- which half to sample, when to
//! seed, and from what -- and a log is the only place those choices show.

use std::cell::RefCell;
use std::collections::{BTreeMap, HashMap};
use std::rc::Rc;

use argusnet_sim::motion::{ObstacleSampler, TerrainSampler};
use argusnet_sim::observation_follow::{ObservationTriggeredFollow, Seedable};
use argusnet_sim::waypoint_path::{LoopedWaypointPath, WaypointPathConfig};
use serde_json::{json, Value};

fn fixture() -> Value {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/controllers.json"
    );
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_controller_fixture.py")
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

/// Python's `repr(float(x))`.
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

fn s(value: &Value) -> String {
    value.as_str().expect("string key").to_string()
}

struct ReplayTerrain {
    heights: HashMap<(String, String), f64>,
    gradients: HashMap<(String, String, String), [f64; 2]>,
    clamps: HashMap<(String, String, String, String), f64>,
}

impl ReplayTerrain {
    fn from_case(case: &Value) -> Self {
        let oracle = &case["terrain_oracle"];
        let heights = oracle["heights"]
            .as_array()
            .expect("heights")
            .iter()
            .map(|e| ((s(&e["x_m"]), s(&e["y_m"])), num(&e["height_m"])))
            .collect();
        let gradients = oracle["gradients"]
            .as_array()
            .expect("gradients")
            .iter()
            .map(|e| {
                (
                    (s(&e["x_m"]), s(&e["y_m"]), s(&e["delta_m"])),
                    vec2(&e["gradient"]),
                )
            })
            .collect();
        let clamps = oracle["clamps"]
            .as_array()
            .expect("clamps")
            .iter()
            .map(|e| {
                (
                    (s(&e["x_m"]), s(&e["y_m"]), s(&e["z_m"]), s(&e["min_agl_m"])),
                    num(&e["clamped_z_m"]),
                )
            })
            .collect();
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
            .unwrap_or_else(|| panic!("gradient_at({x_m}, {y_m}, {delta_key}) was never recorded"))
    }

    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
        *self
            .clamps
            .get(&(key(x_m), key(y_m), key(z_m), key(min_agl_m)))
            .unwrap_or_else(|| {
                panic!("clamp_altitude({x_m}, {y_m}, {z_m}, {min_agl_m}) was never recorded")
            })
    }
}

struct ReplayObstacles {
    probes: HashMap<(String, String, String), bool>,
    pushes: HashMap<(String, String, String), [f64; 2]>,
}

impl ReplayObstacles {
    fn from_case(case: &Value) -> Self {
        let probes = case["obstacle_oracle"]
            .as_array()
            .expect("obstacle_oracle")
            .iter()
            .map(|e| {
                (
                    (s(&e["x_m"]), s(&e["y_m"]), s(&e["z_m"])),
                    e["collides"].as_bool().expect("collides"),
                )
            })
            .collect();
        let pushes = case["obstacle_push_oracle"]
            .as_array()
            .expect("obstacle_push_oracle")
            .iter()
            .map(|e| {
                (
                    (s(&e["x_m"]), s(&e["y_m"]), s(&e["margin_m"])),
                    vec2(&e["pushed_xy"]),
                )
            })
            .collect();
        Self { probes, pushes }
    }

    fn probe(&self, point_m: [f64; 3]) -> bool {
        *self
            .probes
            .get(&(key(point_m[0]), key(point_m[1]), key(point_m[2])))
            .unwrap_or_else(|| panic!("point_collides{point_m:?} was never recorded"))
    }
}

impl ObstacleSampler for ReplayObstacles {
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]> {
        if !self.probe(point_m) {
            return None;
        }
        Some(
            *self
                .pushes
                .get(&(key(point_m[0]), key(point_m[1]), key(margin_m)))
                .unwrap_or_else(|| {
                    panic!("push_outside_xy({point_m:?}, {margin_m}) was never recorded")
                }),
        )
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        self.probe(point_m)
    }
}

/// A stub that fails the test if the path probes obstacles at all.
struct NoProbes;

impl ObstacleSampler for NoProbes {
    fn push_clear(&self, point_m: [f64; 3], _margin_m: f64) -> Option<[f64; 2]> {
        panic!("probed {point_m:?} on a path the reference never gave an obstacle layer")
    }

    fn collides(&self, point_m: [f64; 3]) -> bool {
        panic!("probed {point_m:?} on a path the reference never gave an obstacle layer")
    }
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|v| v.to_bits()).collect()
}

#[test]
fn looped_waypoint_path_matches_the_reference_exactly() {
    let fixture = fixture();
    let cases = fixture["looped_waypoint_path"].as_array().expect("cases");
    assert!(cases.len() >= 7, "the fixture lost cases");
    let mut samples_checked = 0;
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let config = WaypointPathConfig {
            waypoints_xy_m: case["waypoints_xy_m"]
                .as_array()
                .unwrap()
                .iter()
                .map(vec2)
                .collect(),
            speed_mps: num(&case["speed_mps"]),
            base_agl_m: num(&case["base_agl_m"]),
            vertical_amplitude_m: num(&case["vertical_amplitude_m"]),
            vertical_frequency_rad_s: num(&case["vertical_frequency_rad_s"]),
            phase: num(&case["phase"]),
            min_agl_m: num(&case["min_agl_m"]),
            avoid_obstacles: case["avoid_obstacles"].as_bool().unwrap(),
        };
        let path = LoopedWaypointPath::new(config).expect("the reference accepted these");
        let terrain = ReplayTerrain::from_case(case);
        let replay = ReplayObstacles::from_case(case);
        // Hand the world's layer over whether or not the reference had one:
        // the port must decline to use it when `avoid_obstacles` is false.
        let obstacles: &dyn ObstacleSampler = if case["avoid_obstacles"].as_bool().unwrap() {
            &replay
        } else {
            &NoProbes
        };
        for sample in case["samples"].as_array().unwrap() {
            let t = num(&sample["t"]);
            let (position, velocity) = path.sample(t, &terrain, Some(obstacles));
            assert_eq!(
                bits(&position),
                bits(&vec3(&sample["position"])),
                "{name} t={t}: position {position:?} != {}",
                sample["position"]
            );
            assert_eq!(
                bits(&velocity),
                bits(&vec3(&sample["velocity"])),
                "{name} t={t}: velocity {velocity:?} != {}",
                sample["velocity"]
            );
            samples_checked += 1;
        }
    }
    assert!(
        samples_checked >= 30,
        "only {samples_checked} samples checked"
    );
}

/// A fake half that answers from the recording and logs like the Python fake.
struct ReplayHalf {
    label: &'static str,
    returns: HashMap<String, ([f64; 3], [f64; 3])>,
    log: Rc<RefCell<Vec<Value>>>,
}

impl ReplayHalf {
    fn new(label: &'static str, returns: &Value, log: Rc<RefCell<Vec<Value>>>) -> Self {
        let returns = returns
            .as_object()
            .expect("returns")
            .iter()
            .map(|(t, r)| (t.clone(), (vec3(&r["position"]), vec3(&r["velocity"]))))
            .collect();
        Self {
            label,
            returns,
            log,
        }
    }

    fn sample(&mut self, t: f64) -> ([f64; 3], [f64; 3]) {
        self.log
            .borrow_mut()
            .push(json!({"call": self.label, "t": key(t)}));
        *self.returns.get(&key(t)).unwrap_or_else(|| {
            panic!(
                "{} was sampled at {t}, which the reference never did",
                self.label
            )
        })
    }
}

impl Seedable for ReplayHalf {
    fn seed(&mut self, position_m: [f64; 3], velocity_mps: [f64; 3], timestamp_s: f64) {
        self.log.borrow_mut().push(json!({
            "call": "seed",
            "position": position_m.iter().map(|v| key(*v)).collect::<Vec<_>>(),
            "velocity": velocity_mps.iter().map(|v| key(*v)).collect::<Vec<_>>(),
            "t": key(timestamp_s),
        }));
    }

    fn reset_state(&mut self) {
        self.log.borrow_mut().push(json!({"call": "reset_follow"}));
    }
}

#[test]
fn observation_triggered_follow_makes_the_reference_calls() {
    let fixture = fixture();
    let cases = fixture["observation_triggered_follow"]
        .as_array()
        .expect("cases");
    assert!(cases.len() >= 5, "the fixture lost cases");
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let log = Rc::new(RefCell::new(Vec::new()));
        let search = ReplayHalf::new("search", &case["search_returns"], Rc::clone(&log));
        let follow = ReplayHalf::new("follow", &case["follow_returns"], Rc::clone(&log));
        let mut controller = ObservationTriggeredFollow::new("drone-1", "target-a", search, follow);

        let mut outputs = Vec::new();
        for event in case["script"].as_array().unwrap() {
            match event["event"].as_str().unwrap() {
                "sample" => {
                    let (position, velocity) = controller.sample(
                        num(&event["t"]),
                        |search, t| search.sample(t),
                        |follow, t| follow.sample(t),
                    );
                    outputs.push(json!({
                        "event": "sample",
                        "position": position.iter().map(|v| key(*v)).collect::<Vec<_>>(),
                        "velocity": velocity.iter().map(|v| key(*v)).collect::<Vec<_>>(),
                    }));
                }
                "observe" => {
                    let counts: BTreeMap<(String, String), u64> = event["counts"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(pair, count)| {
                            let (node, target) = pair.split_once('|').expect("node|target");
                            (
                                (node.to_string(), target.to_string()),
                                count.as_u64().unwrap(),
                            )
                        })
                        .collect();
                    controller.update_from_observations(&counts);
                    outputs.push(json!({"event": "observe", "engaged": controller.is_engaged()}));
                }
                "reset" => {
                    controller.reset_state();
                    outputs.push(json!({"event": "reset", "engaged": controller.is_engaged()}));
                }
                other => panic!("unknown event {other}"),
            }
        }
        assert_eq!(
            Value::Array(log.borrow().clone()),
            case["calls"],
            "{name}: call log differs"
        );
        assert_eq!(
            Value::Array(outputs),
            case["outputs"],
            "{name}: outputs differ"
        );
    }
}
