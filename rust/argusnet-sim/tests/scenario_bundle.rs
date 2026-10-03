//! A Python-built scenario, run by the Rust tick, compared with the reference.
//!
//! Fixtures from `scripts/capture_scenario_bundle_fixture.py`: the bundle, and
//! `run_simulation` on the same built scenario. See that script for what can
//! be compared exactly and what cannot. In short: stage 6 draws differ by
//! design (ADR-006), so the comparison is exact for every platform until the
//! two runs first disagree about an engagement, and after that exact for the
//! platforms no observation can reach -- stations, and each truth-fed follow
//! drone until deconfliction first couples it to one that diverged.
//! "Exact" is bit for bit on the platform the fixture was captured on, and
//! within a few ULPs elsewhere (`common::parity`).

mod common;

use std::collections::BTreeMap;

use argusnet_core::AssociationMode;
use argusnet_sensing::cascade::{Outcome, RejectReason};
use argusnet_sim::scenario::{
    assemble, build_world, reference_battery_model, BundleError, ScenarioBundle,
    FOLLOW_LOS_CLEARANCE_M,
};
use argusnet_sim::stage::{Stage, StageOutcome};
use common::parity::{parity_of, Parity};
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

/// Bit for bit on the capture platform, within the parity bound elsewhere.
fn same_kinematics(parity: Parity, a: Kinematics, b: Kinematics) -> bool {
    let flat = |k: Kinematics| -> Vec<f64> {
        k.0.iter()
            .chain(&k.1)
            .chain(&k.2)
            .map(|v| f64::from_bits(*v))
            .collect()
    };
    parity.same_all(&flat(a), &flat(b))
}

struct RustRun {
    frames: Vec<BTreeMap<String, Kinematics>>,
    engaged: Vec<BTreeMap<String, bool>>,
    accepted: Vec<usize>,
    tracks: Vec<usize>,
    /// Per tick, sorted `[node, target, reason]` for every pair rejected by a
    /// gate that runs before the pair's first draw.
    draw_free: Vec<Vec<[String; 3]>>,
    /// Every generation rejection over the run, by reason.
    rejections: BTreeMap<String, u64>,
}

fn run_rust(case: &str, steps: usize, search_drones: &[String]) -> RustRun {
    let bundle = bundle_of(case);
    let mut run = assemble(&bundle).expect("bundle assembles");
    let mut out = RustRun {
        frames: Vec::new(),
        engaged: Vec::new(),
        accepted: Vec::new(),
        tracks: Vec::new(),
        draw_free: Vec::new(),
        rejections: BTreeMap::new(),
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
                .map(|id| {
                    (
                        id.clone(),
                        run.controllers().is_engaged(id).unwrap_or(false),
                    )
                })
                .collect(),
        );
        let frame = run.fusion().frame().expect("stage 7 ran");
        out.accepted.push(frame.observations.len());
        out.tracks.push(frame.tracks.len());
        let mut draw_free: Vec<[String; 3]> = run
            .observe()
            .records()
            .into_iter()
            .filter_map(|record| match record.outcome {
                Outcome::Rejected(
                    reason @ (RejectReason::OutOfRange | RejectReason::OutsideFov),
                ) => Some([
                    record.node_id,
                    record.target_id,
                    reason.as_str().to_string(),
                ]),
                _ => None,
            })
            .collect();
        draw_free.sort();
        for record in run.observe().records() {
            if let Outcome::Rejected(reason) = record.outcome {
                *out.rejections
                    .entry(reason.as_str().to_string())
                    .or_default() += 1;
            }
        }
        out.draw_free.push(draw_free);
    }
    out
}

fn reference_draw_free(frame: &Value) -> Vec<[String; 3]> {
    frame["draw_free_rejections"]
        .as_array()
        .expect("draw_free_rejections")
        .iter()
        .map(|r| {
            let r = r.as_array().expect("triple");
            [0, 1, 2].map(|i| r[i].as_str().expect("string").to_string())
        })
        .collect()
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

    // A scan bundle must carry its mission section; a mode this build does
    // not know is refused by name.
    let mut scan = bundle.clone();
    scan.mission_mode = "scan_map_inspect".into();
    assert!(matches!(assemble(&scan), Err(BundleError::Invalid(_))));
    let mut unknown = bundle.clone();
    unknown.mission_mode = "patrol".into();
    assert!(matches!(
        assemble(&unknown),
        Err(BundleError::Unsupported(_))
    ));
}

#[test]
fn a_python_scenario_runs_in_the_rust_tick_and_matches_where_it_must() {
    let reference = reference();
    let parity = parity_of(&reference);
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
    let mut gate_samples = 0;
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
        // The draw-free gates, for pairs whose sensor is still comparable:
        // range and field of view from the nominal pose (P4.8e.2).
        let keep = |r: &[String; 3]| comparable.contains(&r[0]);
        let expected_gates: Vec<_> = reference_draw_free(frame)
            .into_iter()
            .filter(keep)
            .collect();
        let actual_gates: Vec<_> = rust.draw_free[i]
            .iter()
            .filter(|r| keep(r))
            .cloned()
            .collect();
        assert_eq!(
            actual_gates, expected_gates,
            "draw-free rejections at tick {i}"
        );
        gate_samples += expected_gates.len();
        for id in comparable {
            let expected = reference_kinematics(frame, &id);
            let actual = rust.frames[i][&id];
            assert!(
                same_kinematics(parity, actual, expected),
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
    // Few in this scenario (its engagement diverges within ticks, and stations
    // never reject); the all-follow test carries the strict requirement.
    assert!(
        gate_samples >= 5,
        "only {gate_samples} draw-free rejections compared"
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
    let parity = parity_of(&reference);
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle_of("tt_follow");
    let rust = run_rust("tt_follow", frames.len(), &[]);
    let mut resolved_ticks = 0;
    let mut fov_rejections = 0;
    for (i, frame) in frames.iter().enumerate() {
        let expected_gates = reference_draw_free(frame);
        assert_eq!(
            rust.draw_free[i], expected_gates,
            "draw-free rejections at tick {i}"
        );
        fov_rejections += expected_gates
            .iter()
            .filter(|r| r[2] == "outside_fov")
            .count();
        if !frame["deconfliction_drones"].as_array().unwrap().is_empty() {
            resolved_ticks += 1;
        }
        for node in &bundle.nodes {
            let expected = reference_kinematics(frame, &node.node_id);
            let actual = rust.frames[i][&node.node_id];
            assert!(
                same_kinematics(parity, actual, expected),
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
    assert!(
        fov_rejections >= 50,
        "only {fov_rejections} FOV rejections compared"
    );
    let mut python_rejections: BTreeMap<String, u64> = BTreeMap::new();
    for frame in frames {
        for (reason, count) in frame["rejection_counts"].as_object().unwrap() {
            *python_rejections.entry(reason.clone()).or_default() += count.as_u64().unwrap();
        }
    }
    let python_accepted: u64 = frames
        .iter()
        .map(|f| f["accepted_observations"].as_u64().unwrap())
        .sum();
    let rust_accepted = rust.accepted.iter().sum::<usize>() as u64;
    eprintln!(
        "all-follow: python accepted {python_accepted} rejections {python_rejections:?}; \
         rust accepted {rust_accepted} rejections {:?}",
        rust.rejections
    );
    // The draw-dependent gates, statistically. Kinematics are identical, so
    // the only difference is the stream (ADR-006): about 340 pairs reach the
    // detection draw at p ~ 0.55, a standard deviation near 9. The envelope
    // is three of them. Before P4.8e.2, stage 6 ignored the world's detection
    // probability and accepted 303 against the reference's 185 -- the
    // regression this bound exists to catch. Formal multi-seed envelopes are
    // P4.9c.
    let within = |name: &str, python: u64, rust: u64| {
        let gap = python.abs_diff(rust);
        assert!(
            gap <= 27,
            "{name}: python {python}, rust {rust} (gap {gap} > 27)"
        );
    };
    within("accepted", python_accepted, rust_accepted);
    within(
        "dropout",
        python_rejections.get("dropout").copied().unwrap_or(0),
        rust.rejections.get("dropout").copied().unwrap_or(0),
    );
}

/// Wiring the exact comparisons cannot reach: things that only change
/// observation-fed behaviour, or that this scenario never exercises.
#[test]
fn the_assembly_wires_every_capability() {
    let bundle = bundle();
    let mut run = assemble(&bundle).expect("assembles");

    assert!(
        run.observe().visibility_is_modelled(),
        "stage 6 is running on AlwaysVisible"
    );
    for node in &bundle.nodes {
        let spec = run
            .observe()
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
        run.fusion().config().data_association_mode,
        AssociationMode::Labeled
    );
    assert_eq!(run.battery().model(), reference_battery_model());

    let summary = run.controllers().environment_summary().expect("installed");
    assert!(summary.has_obstacles && summary.has_weather && summary.has_pursuit);
    assert!(summary.has_map_bounds && !summary.has_planner);

    assert!(
        run.deconflict().has_terrain(),
        "deconfliction clamps against a constant"
    );
    let roles = run.deconflict().roles();
    for (drone, role) in &bundle.drone_roles {
        assert_eq!(&roles.role_of(drone), role);
    }

    // Stage 6 senses through the bundle's world *and weather*: probe the
    // installed oracle with the reference's scenario-weather detections.
    let reference = reference();
    let scenario_weather: Vec<&Value> = reference["los"]["detections"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|d| d["weather"] == "scenario")
        .collect();
    assert!(!scenario_weather.is_empty());
    for d in scenario_weather {
        let (_, detection) =
            run.observe()
                .probe(vec3(&d["from"]), vec3(&d["to"]), num(&d["max_range_m"]));
        let detection = detection.expect("stage 6 has a world detection model");
        assert_eq!(detection.p_d.to_bits(), num(&d["p_d"]).to_bits(), "{d}");
        assert_eq!(
            detection.effective_noise_multiplier.to_bits(),
            num(&d["effective_noise_multiplier"]).to_bits(),
            "{d}"
        );
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

/// The world's detection model -- what stage 6 multiplies into its detection
/// draw and its bearing noise -- against the reference's
/// `compute_detection_probability`, under three weathers and two sensor
/// ranges, including sight lines that cross a whole cloud slab.
#[test]
fn detection_matches_the_reference() {
    use argusnet_sim::observe::VisibilityOracle;
    use argusnet_sim::weather::MotionWeather;
    use std::sync::Arc;

    let reference = reference();
    let los = &reference["los"];
    let world = build_world(&bundle().world).expect("world builds");
    let weathers: BTreeMap<String, MotionWeather> = los["weathers"]
        .as_object()
        .expect("weathers")
        .iter()
        .map(|(name, doc)| {
            (
                name.clone(),
                serde_json::from_value(doc.clone()).expect("weather document"),
            )
        })
        .collect();
    let detections = los["detections"].as_array().expect("detections");
    assert!(detections.len() >= 300);
    for (i, d) in detections.iter().enumerate() {
        let weather = weathers[d["weather"].as_str().unwrap()].clone();
        let oracle = world.visibility.with_weather(Some(Arc::new(weather)));
        let (_, detection) =
            oracle.sight_and_detection(vec3(&d["from"]), vec3(&d["to"]), num(&d["max_range_m"]));
        let detection = detection.expect("a world oracle always reports detection");
        let context = format!("detection {i}: {d}");
        assert_eq!(
            detection.p_d.to_bits(),
            num(&d["p_d"]).to_bits(),
            "p_d, {context}"
        );
        assert_eq!(
            detection.effective_noise_multiplier.to_bits(),
            num(&d["effective_noise_multiplier"]).to_bits(),
            "noise, {context}"
        );
        assert_eq!(
            Some(detection.dominant_loss_factor),
            d["dominant_loss_factor"].as_str(),
            "dominant loss, {context}"
        );
        assert_eq!(
            detection.cloud_factor.to_bits(),
            num(&d["cloud_factor"]).to_bits(),
            "cloud, {context}"
        );
    }
}

/// The default mission mode, end to end: scan, localize, inspect, egress,
/// complete. Every node, the coverage fraction and the phase match the
/// reference bit for bit until the first tick the two runs disagree about the
/// phase -- which localization (ADR-006 streams) decides -- and stations match
/// for the whole run. Past that point the Rust run must still finish the
/// mission on its own.
#[test]
fn a_scan_map_inspect_run_matches_until_localization_decides() {
    use argusnet_sim::mission::MissionPhase;

    let reference = reference_of("scan");
    let parity = parity_of(&reference);
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle_of("scan");
    let mut run = assemble(&bundle).expect("scan bundle assembles");

    let phase_name = |phase: MissionPhase| phase.to_string();
    let mut divergence = None;
    let mut exact_node_ticks = 0;
    let mut phases_seen = std::collections::BTreeSet::new();
    for (i, frame) in frames.iter().enumerate() {
        let snapshot = run.step().expect("tick runs");
        let rust_phase = phase_name(run.mission().expect("a scan run has a mission").phase());
        phases_seen.insert(rust_phase.clone());
        let python_phase = frame["mission"]["phase"].as_str().unwrap().to_string();
        if divergence.is_none() && rust_phase != python_phase {
            divergence = Some(i);
        }
        for platform in &snapshot.platforms {
            let comparable = divergence.is_none() || !platform.is_mobile;
            if !comparable {
                continue;
            }
            let expected = reference_kinematics(frame, &platform.node_id);
            let actual: Kinematics = (
                bits([
                    platform.position.x,
                    platform.position.y,
                    platform.position.z,
                ]),
                bits([
                    platform.velocity.x,
                    platform.velocity.y,
                    platform.velocity.z,
                ]),
                [
                    platform.health.to_bits(),
                    platform.battery_fraction.unwrap_or(-1.0).to_bits(),
                ],
            );
            assert!(
                same_kinematics(parity, actual, expected),
                "{} at tick {i} ({rust_phase}): rust {:?} python {:?}",
                platform.node_id,
                (actual.0.map(f64::from_bits), actual.1.map(f64::from_bits)),
                (
                    expected.0.map(f64::from_bits),
                    expected.1.map(f64::from_bits)
                ),
            );
            exact_node_ticks += 1;
        }
        if divergence.is_none() {
            assert_eq!(
                run.mapping()
                    .expect("a scan run has mapping")
                    .coverage_fraction()
                    .to_bits(),
                num(&frame["mission"]["coverage_fraction"]).to_bits(),
                "coverage fraction at tick {i}"
            );
        }
    }
    eprintln!(
        "scan: phase divergence at {divergence:?}; {exact_node_ticks} exact node-ticks; \
         rust phases {phases_seen:?}"
    );
    assert!(
        exact_node_ticks >= 400,
        "only {exact_node_ticks} exact node-ticks"
    );
    for phase in ["scanning", "inspecting", "egress", "complete"] {
        assert!(
            phases_seen.contains(phase),
            "the Rust run never reached {phase}"
        );
    }
}

/// With the reference's localization replayed in, the mission has no
/// stochastic input left, so the whole run must match: every node on every
/// tick -- lanes, POI redirects, dwell, handoffs, egress -- and the phase and
/// coverage fraction on every tick, through `complete`.
#[test]
fn a_scan_map_inspect_run_with_replayed_localization_matches_throughout() {
    use argusnet_sim::mission_driver::RecordedLocalization;

    let reference = reference_of("scan");
    let parity = parity_of(&reference);
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle_of("scan");
    let mut run = assemble(&bundle).expect("scan bundle assembles");
    assert!(run.load_localization_replay(
        frames
            .iter()
            .map(|f| RecordedLocalization {
                confidences: f["mission"]["confidences"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(num)
                    .collect(),
                timed_out: f["mission"]["localization_timed_out"].as_bool().unwrap(),
            })
            .collect(),
    ));

    let mut phases = Vec::new();
    for (i, frame) in frames.iter().enumerate() {
        let snapshot = run.step().expect("tick runs");
        let phase = run.mission().expect("mission").phase().to_string();
        assert_eq!(
            phase,
            frame["mission"]["phase"].as_str().unwrap(),
            "phase at tick {i}"
        );
        assert_eq!(
            run.mission()
                .expect("mission")
                .state()
                .localization_timed_out(),
            frame["mission"]["localization_timed_out"]
                .as_bool()
                .unwrap(),
            "localization-timeout latch at tick {i}"
        );
        if phases.last() != Some(&phase) {
            phases.push(phase.clone());
        }
        assert_eq!(
            run.mapping()
                .expect("mapping")
                .coverage_fraction()
                .to_bits(),
            num(&frame["mission"]["coverage_fraction"]).to_bits(),
            "coverage fraction at tick {i}"
        );
        for platform in &snapshot.platforms {
            let expected = reference_kinematics(frame, &platform.node_id);
            let actual: Kinematics = (
                bits([
                    platform.position.x,
                    platform.position.y,
                    platform.position.z,
                ]),
                bits([
                    platform.velocity.x,
                    platform.velocity.y,
                    platform.velocity.z,
                ]),
                [
                    platform.health.to_bits(),
                    platform.battery_fraction.unwrap_or(-1.0).to_bits(),
                ],
            );
            assert!(
                same_kinematics(parity, actual, expected),
                "{} at tick {i} ({phase}): rust {:?} python {:?}",
                platform.node_id,
                (actual.0.map(f64::from_bits), actual.1.map(f64::from_bits)),
                (
                    expected.0.map(f64::from_bits),
                    expected.1.map(f64::from_bits)
                ),
            );
        }
    }
    eprintln!("scan with replayed localization: phases {phases:?}");
    assert_eq!(phases.last().map(String::as_str), Some("complete"));
}

/// The scan assembly hands the driver exactly what the bundle says: every
/// threshold and speed, and every POI with its priority. Values a single
/// scenario may never make decisive, so they are checked at the wiring.
#[test]
fn the_scan_assembly_wires_the_mission_as_the_bundle_describes() {
    let bundle = bundle_of("scan");
    let doc = bundle
        .mission
        .clone()
        .expect("scan bundle has a mission section");
    let run = assemble(&bundle).expect("assembles");
    let (config, pois) = run.mission_setup.clone().expect("a scan run has a driver");
    assert_eq!(
        config.mission.scan_coverage_threshold,
        doc.scan_coverage_threshold
    );
    assert_eq!(
        config.mission.localization_confidence_threshold,
        doc.localization_confidence_threshold
    );
    assert_eq!(
        config.mission.gap_fill_min_fraction,
        doc.gap_fill_min_fraction
    );
    assert_eq!(
        config.mission.egress_arrival_radius_m,
        doc.egress_arrival_radius_m
    );
    assert_eq!(config.redirect_speed_mps, doc.redirect_speed_mps);
    assert_eq!(pois.len(), doc.pois.len());
    for (poi, expected) in pois.iter().zip(&doc.pois) {
        assert_eq!(poi.poi_id, expected.poi_id);
        assert_eq!(poi.priority, expected.priority);
        assert_eq!(poi.position, expected.position);
        assert_eq!(poi.required_dwell_s, expected.required_dwell_s);
    }
}

// ---------------------------------------------------------------------
// Stage 4: the hard safety floor (P4.9d)
// ---------------------------------------------------------------------

/// This tick's safety events as the replay writes them, and as the reference
/// did.
fn safety_events(run: &argusnet_sim::scenario::AssembledRun) -> Value {
    let frame = run.fusion().frame().expect("stage 7 ran");
    serde_json::to_value(
        frame
            .safety_events
            .iter()
            .map(argusnet_sim::replay::ReplaySafetyEvent::from)
            .collect::<Vec<_>>(),
    )
    .expect("events serialise")
}

fn reference_safety_events(frame: &Value) -> Value {
    match &frame["replay"]["safety_events"] {
        Value::Null => Value::Array(Vec::new()),
        events => events.clone(),
    }
}

/// Every platform's kinematics at one tick, from a snapshot.
fn snapshot_kinematics(platform: &argusnet_types::NodeState) -> Kinematics {
    (
        bits([
            platform.position.x,
            platform.position.y,
            platform.position.z,
        ]),
        bits([
            platform.velocity.x,
            platform.velocity.y,
            platform.velocity.z,
        ]),
        [
            platform.health.to_bits(),
            platform.battery_fraction.unwrap_or(-1.0).to_bits(),
        ],
    )
}

/// The all-follow fleet under `safety_blocking`: interceptor envelopes over
/// real terrain, `dt` from the config, a caution fraction of 0.9. No
/// observation moves anything, so every clamp and every event must match the
/// reference on every tick.
#[test]
fn a_blocking_safety_run_matches_on_every_tick() {
    let reference = reference_of("tt_safety");
    let parity = parity_of(&reference);
    let frames = reference["frames"].as_array().expect("frames");
    let bundle = bundle_of("tt_safety");
    assert!(bundle.safety_blocking);
    let mut run = assemble(&bundle).expect("bundle assembles");
    assert!(
        run.safety().is_some(),
        "stage 4 is assembled when asked for"
    );
    let (mut events, mut warnings, mut clamped) = (0, 0, 0);
    for (i, frame) in frames.iter().enumerate() {
        let snapshot = run.step().expect("tick runs");
        for platform in &snapshot.platforms {
            let expected = reference_kinematics(frame, &platform.node_id);
            let actual = snapshot_kinematics(platform);
            assert!(
                same_kinematics(parity, actual, expected),
                "{} at tick {i}: rust {:?} python {:?}",
                platform.node_id,
                (actual.0.map(f64::from_bits), actual.1.map(f64::from_bits)),
                (
                    expected.0.map(f64::from_bits),
                    expected.1.map(f64::from_bits)
                ),
            );
        }
        let expected = reference_safety_events(frame);
        assert_eq!(safety_events(&run), expected, "safety events at tick {i}");
        for event in expected.as_array().unwrap() {
            events += 1;
            warnings += usize::from(event["reason"] == "warning");
            clamped += usize::from(event["clamped"] == true);
        }
    }
    eprintln!("blocking safety: {events} events, {warnings} warnings, {clamped} clamped");
    assert!(events >= 100 && warnings >= 10 && clamped >= 50);
}

/// The scan fleet under `safety_blocking`, with localization replayed: the
/// tracker envelope, and the whole mission through `complete`.
#[test]
fn a_blocking_safety_scan_run_matches_throughout() {
    use argusnet_sim::mission_driver::RecordedLocalization;

    let reference = reference_of("scan_safety");
    let parity = parity_of(&reference);
    let frames = reference["frames"].as_array().expect("frames");
    let mut run = assemble(&bundle_of("scan_safety")).expect("assembles");
    assert!(run.load_localization_replay(
        frames
            .iter()
            .map(|f| RecordedLocalization {
                confidences: f["mission"]["confidences"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(num)
                    .collect(),
                timed_out: f["mission"]["localization_timed_out"].as_bool().unwrap(),
            })
            .collect(),
    ));
    let mut violations: BTreeMap<String, usize> = BTreeMap::new();
    for (i, frame) in frames.iter().enumerate() {
        let snapshot = run.step().expect("tick runs");
        assert_eq!(
            run.mission().expect("mission").phase().to_string(),
            frame["mission"]["phase"].as_str().unwrap(),
            "phase at tick {i}"
        );
        for platform in &snapshot.platforms {
            assert!(
                same_kinematics(
                    parity,
                    snapshot_kinematics(platform),
                    reference_kinematics(frame, &platform.node_id),
                ),
                "{} at tick {i}",
                platform.node_id
            );
        }
        let expected = reference_safety_events(frame);
        assert_eq!(safety_events(&run), expected, "safety events at tick {i}");
        for event in expected.as_array().unwrap() {
            for violation in event["violations"].as_array().unwrap() {
                *violations
                    .entry(violation.as_str().unwrap().to_string())
                    .or_default() += 1;
            }
        }
    }
    assert_eq!(
        run.mission().expect("mission").phase().to_string(),
        "complete"
    );
    // `max_descent_rate` only the tracker envelope reaches here.
    assert!(violations.len() >= 6, "{violations:?}");
}

/// The replay carries the events, and a run without `safety_blocking` has no
/// stage 4 at all: not a quiet one, none.
#[test]
fn safety_events_reach_the_replay_and_an_unblocked_run_has_no_stage_four() {
    let run = assemble(&bundle_of("tt_safety")).expect("assembles");
    let replay = run.run_to_replay().expect("runs");
    let value = serde_json::to_value(&replay).expect("serialises");
    let written: usize = value["frames"]
        .as_array()
        .unwrap()
        .iter()
        .map(|f| f["safety_events"].as_array().map_or(0, Vec::len))
        .sum();
    assert!(written >= 100, "only {written} events reached the replay");

    // Without it, the placeholder holds the slot: no monitor, no events.
    let mut unblocked = assemble(&bundle_of("tt_follow")).expect("assembles");
    assert!(unblocked.safety().is_none());
    unblocked.step().expect("tick runs");
    let record = unblocked
        .sim
        .trace()
        .records()
        .iter()
        .rev()
        .find(|r| r.stage == Stage::SafetyMonitor)
        .expect("the slot is traced")
        .clone();
    assert_eq!(record.outcome, StageOutcome::NoWork);
    assert!(record.notes.is_empty());
    assert!(unblocked.fusion().frame().unwrap().safety_events.is_empty());
}

/// Stage 4 reads energy from stage 1's ledger, as the reference passes
/// `ns.battery_fraction`: a drained platform violates its reserve. (The
/// fixture run never drains that far, so the wiring is checked here.)
#[test]
fn the_assembled_safety_monitor_reads_the_battery_ledger() {
    let mut run = assemble(&bundle_of("tt_safety")).expect("assembles");
    let drone = "drone-north";
    assert!(run.battery_mut().set_remaining_wh(drone, 0.5));
    run.step().expect("tick runs");
    let frame = run.fusion().frame().expect("stage 7 ran");
    let event = frame
        .safety_events
        .iter()
        .find(|e| e.subject_id == drone)
        .expect("the drained drone violated something");
    assert!(
        event.violations.iter().any(|v| v == "min_energy_reserve"),
        "{:?}",
        event.violations
    );
}
