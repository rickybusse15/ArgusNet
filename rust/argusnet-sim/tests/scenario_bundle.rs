//! A Python-built scenario, run by the Rust tick, compared with the reference.
//!
//! Fixtures from `scripts/capture_scenario_bundle_fixture.py`: the bundle, and
//! `run_simulation` on the same built scenario. See that script for what can
//! be compared exactly and what cannot. In short: stage 6 draws differ by
//! design (ADR-006), so the comparison is exact for every platform until the
//! two runs first disagree about an engagement, and after that exact for the
//! platforms no observation can reach -- stations, and each truth-fed follow
//! drone until deconfliction first couples it to one that diverged.

use std::collections::BTreeMap;

use argusnet_core::AssociationMode;
use argusnet_sim::scenario::{
    assemble, build_world, reference_battery_model, BundleError, ScenarioBundle,
    FOLLOW_LOS_CLEARANCE_M,
};
use argusnet_sim::stage::{Stage, StageOutcome};
use serde_json::Value;

fn read(name: &str) -> String {
    let path = format!("{}/../../tests/fixtures/{name}", env!("CARGO_MANIFEST_DIR"));
    std::fs::read_to_string(&path).unwrap_or_else(|e| {
        panic!("reading {path}: {e}; regenerate with scripts/capture_scenario_bundle_fixture.py")
    })
}

fn bundle_of(case: &str) -> ScenarioBundle {
    ScenarioBundle::from_json(&read(&format!("scenario_bundle_{case}.json")))
        .expect("bundle parses")
}

fn bundle() -> ScenarioBundle {
    bundle_of("tt")
}

fn reference_of(case: &str) -> Value {
    serde_json::from_str(&read(&format!("scenario_bundle_{case}_reference.json")))
        .expect("reference parses")
}

fn reference() -> Value {
    reference_of("tt")
}

fn num(value: &Value) -> f64 {
    value
        .as_str()
        .expect("stringified float")
        .parse()
        .expect("parses")
}

fn vec3(value: &Value) -> [f64; 3] {
    let items = value.as_array().expect("array");
    [num(&items[0]), num(&items[1]), num(&items[2])]
}

/// One node at one tick, as bits: position, velocity, and
/// `[health, battery_fraction]` (the reference's `-1.0` for "no battery").
type Kinematics = ([u64; 3], [u64; 3], [u64; 2]);

fn bits(v: [f64; 3]) -> [u64; 3] {
    v.map(f64::to_bits)
}

struct RustRun {
    frames: Vec<BTreeMap<String, Kinematics>>,
    engaged: Vec<BTreeMap<String, bool>>,
    accepted: Vec<usize>,
    tracks: Vec<usize>,
}

fn run_rust(case: &str, steps: usize, search_drones: &[String]) -> RustRun {
    let bundle = bundle_of(case);
    let mut run = assemble(&bundle).expect("bundle assembles");
    let mut out = RustRun {
        frames: Vec::new(),
        engaged: Vec::new(),
        accepted: Vec::new(),
        tracks: Vec::new(),
    };
    for _ in 0..steps {
        let snapshot = run.step().expect("tick runs");
        out.frames.push(
            snapshot
                .platforms
                .iter()
                .map(|p| {
                    (
                        p.node_id.clone(),
                        (
                            bits([p.position.x, p.position.y, p.position.z]),
                            bits([p.velocity.x, p.velocity.y, p.velocity.z]),
                            [
                                p.health.to_bits(),
                                p.battery_fraction.unwrap_or(-1.0).to_bits(),
                            ],
                        ),
                    )
                })
                .collect(),
        );
        out.engaged.push(
            search_drones
                .iter()
                .map(|id| (id.clone(), run.controllers.is_engaged(id).unwrap_or(false)))
                .collect(),
        );
        let frame = run.fusion.frame().expect("stage 7 ran");
        out.accepted.push(frame.observations.len());
        out.tracks.push(frame.tracks.len());
    }
    out
}

fn reference_kinematics(frame: &Value, node_id: &str) -> Kinematics {
    let node = &frame["nodes"][node_id];
    (
        bits(vec3(&node["position"])),
        bits(vec3(&node["velocity"])),
        [
            num(&node["health"]).to_bits(),
            num(&node["battery_fraction"]).to_bits(),
        ],
    )
}

#[test]
fn the_bundle_parses_and_refuses_what_it_does_not_know() {
    let text = read("scenario_bundle_tt.json");
    let bundle = ScenarioBundle::from_json(&text).expect("parses");
    assert_eq!(bundle.version, 1);
    assert!(bundle.nodes.len() >= 6);

    let v2 = text.replacen("\"version\":1", "\"version\":2", 1);
    assert_eq!(
        ScenarioBundle::from_json(&v2).err(),
        Some(BundleError::Version { found: 2 })
    );
    let other = text.replacen(
        "\"format\":\"argusnet.scenario-bundle\"",
        "\"format\":\"something-else\"",
        1,
    );
    assert!(matches!(
        ScenarioBundle::from_json(&other),
        Err(BundleError::Format { .. })
    ));

    let mut scan = bundle.clone();
    scan.mission_mode = "scan_map_inspect".into();
    assert!(matches!(assemble(&scan), Err(BundleError::Unsupported(_))));
}

#[test]
fn a_python_scenario_runs_in_the_rust_tick_and_matches_where_it_must() {
    let reference = reference();
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle();
    let search_drones: Vec<String> = frames[0]["engaged"]
        .as_object()
        .expect("engaged")
        .keys()
        .cloned()
        .collect();
    assert!(!search_drones.is_empty(), "no search-then-follow drone");
    let rust = run_rust("tt", frames.len(), &search_drones);

    // The first tick on which the runs disagree about any engagement.
    let divergence = (0..frames.len())
        .find(|&i| {
            search_drones.iter().any(|id| {
                frames[i]["engaged"][id].as_bool().unwrap() != rust.engaged[i][id.as_str()]
            })
        })
        .unwrap_or(frames.len());

    let stations: Vec<&str> = bundle
        .nodes
        .iter()
        .filter(|n| !n.is_mobile)
        .map(|n| n.node_id.as_str())
        .collect();
    let follow_drones: Vec<String> = bundle
        .nodes
        .iter()
        .filter(|n| n.is_mobile && !search_drones.contains(&n.node_id))
        .map(|n| n.node_id.clone())
        .collect();

    let mut exact_samples = 0;
    let mut decoupled: BTreeMap<String, usize> = BTreeMap::new();
    for (i, frame) in frames.iter().enumerate() {
        let comparable: Vec<String> = if i < divergence {
            bundle.nodes.iter().map(|n| n.node_id.clone()).collect()
        } else {
            // After divergence: stations always; a follow drone until the
            // first deconfliction event that couples it to anything.
            for id in &follow_drones {
                let involved = frame["deconfliction_drones"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|d| d.as_str() == Some(id));
                if involved {
                    decoupled.entry(id.clone()).or_insert(i);
                }
            }
            stations
                .iter()
                .map(|s| s.to_string())
                .chain(
                    follow_drones
                        .iter()
                        .filter(|id| !decoupled.contains_key(*id))
                        .cloned(),
                )
                .collect()
        };
        for id in comparable {
            let expected = reference_kinematics(frame, &id);
            let actual = rust.frames[i][&id];
            assert_eq!(
                actual,
                expected,
                "{id} at tick {i} (engagement divergence at {divergence}): rust {:?} python {:?}",
                (
                    actual.0.map(f64::from_bits),
                    actual.1.map(f64::from_bits),
                    actual.2.map(f64::from_bits)
                ),
                (
                    expected.0.map(f64::from_bits),
                    expected.1.map(f64::from_bits),
                    expected.2.map(f64::from_bits)
                ),
            );
            exact_samples += 1;
        }
    }
    eprintln!(
        "engagement divergence at tick {divergence}; {exact_samples} exact node-ticks; \
         decoupled {decoupled:?}"
    );
    assert!(
        exact_samples >= 200,
        "only {exact_samples} node-ticks compared exactly"
    );
    // The Rust run engages too, through its own stage 6 -> stage 1 feed. The
    // divergence above is found by comparing engagement; if Rust never
    // engaged, that comparison alone would not notice a broken feed.
    assert!(
        rust.engaged.iter().any(|e| e.values().any(|&v| v)),
        "the Rust search-then-follow drone never engaged"
    );

    // Statistical sanity on the observation-fed half: both runs see targets
    // and both fuse tracks. Envelopes proper are P4.9c.
    let python_accepted: u64 = frames
        .iter()
        .map(|f| f["accepted_observations"].as_u64().unwrap())
        .sum();
    let rust_accepted: usize = rust.accepted.iter().sum();
    assert!(python_accepted > 0 && rust_accepted > 0);
    assert!(
        rust.tracks.iter().any(|&t| t > 0),
        "the Rust run never fused a track"
    );
    eprintln!("accepted observations: python {python_accepted}, rust {rust_accepted}");
}

#[test]
fn the_whole_run_produces_a_replay() {
    let bundle = bundle();
    let steps = bundle.steps as usize;
    let replay = assemble(&bundle)
        .expect("assembles")
        .run_to_replay()
        .expect("runs");
    assert_eq!(replay.frames.len(), steps);
    assert_eq!(replay.frames[0].nodes.len(), bundle.nodes.len());
    assert_eq!(replay.frames[0].truths.len(), bundle.targets.len());
}

/// Every drone truth-fed, four on one target: no observation can move
/// anything, so every node must match on every tick -- through 61 frames of
/// deconfliction resolutions clamped against the real terrain.
#[test]
fn an_all_follow_scenario_matches_on_every_tick() {
    let reference = reference_of("tt_follow");
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle_of("tt_follow");
    let rust = run_rust("tt_follow", frames.len(), &[]);
    let mut resolved_ticks = 0;
    for (i, frame) in frames.iter().enumerate() {
        if !frame["deconfliction_drones"].as_array().unwrap().is_empty() {
            resolved_ticks += 1;
        }
        for node in &bundle.nodes {
            let expected = reference_kinematics(frame, &node.node_id);
            let actual = rust.frames[i][&node.node_id];
            assert_eq!(
                actual,
                expected,
                "{} at tick {i}: rust {:?} python {:?}",
                node.node_id,
                (
                    actual.0.map(f64::from_bits),
                    actual.1.map(f64::from_bits),
                    actual.2.map(f64::from_bits)
                ),
                (
                    expected.0.map(f64::from_bits),
                    expected.1.map(f64::from_bits),
                    expected.2.map(f64::from_bits)
                ),
            );
        }
    }
    assert!(
        resolved_ticks >= 20,
        "only {resolved_ticks} ticks exercised deconfliction"
    );
}

/// Wiring the exact comparisons cannot reach: things that only change
/// observation-fed behaviour, or that this scenario never exercises.
#[test]
fn the_assembly_wires_every_capability() {
    let bundle = bundle();
    let mut run = assemble(&bundle).expect("assembles");

    assert!(
        run.observe.visibility_is_modelled(),
        "stage 6 is running on AlwaysVisible"
    );
    for node in &bundle.nodes {
        let spec = run
            .observe
            .sensor(&node.node_id)
            .expect("every node has a sensor");
        if node.is_mobile {
            assert_eq!(
                spec.fov_half_angle_deg,
                Some(node.sensor.fov_half_angle_deg)
            );
        } else {
            assert_eq!(
                spec.fov_half_angle_deg, None,
                "a 180-degree station is directional"
            );
        }
        assert_eq!(spec.max_range_m, node.sensor.max_range_m);
    }
    assert_eq!(
        run.fusion.config().data_association_mode,
        AssociationMode::Labeled
    );
    assert_eq!(run.battery.model(), reference_battery_model());

    let summary = run.controllers.environment_summary().expect("installed");
    assert!(summary.has_obstacles && summary.has_weather && summary.has_pursuit);
    assert!(summary.has_map_bounds && !summary.has_planner);

    assert!(
        run.deconflict.has_terrain(),
        "deconfliction clamps against a constant"
    );
    let roles = run.deconflict.roles();
    for (drone, role) in &bundle.drone_roles {
        assert_eq!(&roles.role_of(drone), role);
    }

    // Past the reference's first resolution (tick 25): stage 3 must have
    // done real work, in its own slot, and nothing in stage 4's.
    let mut stage_three_worked = false;
    for _ in 0..40 {
        run.step().expect("tick");
        for record in run.sim.trace().records() {
            if record.outcome != StageOutcome::NoWork {
                stage_three_worked |= record.stage == Stage::Deconflict;
                assert_ne!(
                    record.stage,
                    Stage::SafetyMonitor,
                    "something ran in stage 4"
                );
            }
        }
    }
    assert!(stage_three_worked, "stage 3 never did any work");
}

/// Line of sight over the bundle's world against the reference's `query.los`.
#[test]
fn line_of_sight_matches_the_reference() {
    let reference = reference();
    let los = &reference["los"];
    assert_eq!(
        FOLLOW_LOS_CLEARANCE_M,
        num(&los["default_clearance_m"]),
        "the follow controller's clearance is not the reference's default"
    );
    let world = build_world(&bundle().world).expect("world builds");
    let queries = los["queries"].as_array().expect("queries");
    assert!(queries.len() >= 300);
    for (i, q) in queries.iter().enumerate() {
        let result = world
            .visibility
            .with_clearance(num(&q["clearance_m"]))
            .los(vec3(&q["from"]), vec3(&q["to"]));
        let context = format!("query {i}: {q}");
        assert_eq!(result.visible, q["visible"].as_bool().unwrap(), "{context}");
        assert_eq!(
            result.blocker_type,
            q["blocker_type"].as_str().unwrap(),
            "{context}"
        );
        for (field, value) in [
            ("transmittance", result.transmittance),
            ("detection_multiplier", result.detection_multiplier),
            ("noise_multiplier", result.noise_multiplier),
        ] {
            assert_eq!(
                value.to_bits(),
                num(&q[field]).to_bits(),
                "{field}, {context}"
            );
        }
    }
}
