//! `field_v3`: `field_v2` on estimated positions, and what a drone that does
//! not know where it is does.
//!
//! Expectations, declared before the runs: the profile is recorded as itself;
//! the composed run resumes exactly, touches nothing and completes. A drone
//! without GNSS stops where it believes it is while it is lost, goes on when
//! it knows again without being put anywhere, and goes home for good when that
//! lasts.

use argusnet_planning::route3d::ReplanConfig;
use argusnet_sim::belief_world::{FollowUpConfig, LostPositionConfig};
use argusnet_sim::compute::ComputePool;
use argusnet_sim::evidence::EvidenceConfig;
use argusnet_sim::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};
use argusnet_sim::mission_driver::{MissionDriverStage, SensorLost};
use argusnet_sim::navigation::NavigationConfig;
use argusnet_sim::run_options::{RunOptions, RunProfile};
use argusnet_sim::scenario::{assemble, assemble_with, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::stage::Stage;
use serde_json::Value;

const DRONE: &str = "drone-east";
/// Past what the fixture's fixes leave, and soon reached without them.
const HOLD_SIGMA_M: f64 = 0.08;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run_with(options: &RunOptions) -> AssembledRun {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    options.apply(&mut run, &bundle).expect("options apply");
    run
}

fn replay(options: &RunOptions) -> (Value, Vec<Value>) {
    let document = run_with(options).run_to_replay().expect("runs");
    let frames = document.frames.iter();
    (
        serde_json::to_value(&document.meta).unwrap(),
        frames.map(|f| serde_json::to_value(f).unwrap()).collect(),
    )
}

/// The profile, with a scan gate the 400-tick fixture reaches and, for a lost
/// drone, limits its outages reach.
fn whole_mission(outage_s: Option<(f64, f64)>, lost: Option<LostPositionConfig>) -> RunOptions {
    let mut options = RunOptions {
        scan_coverage_threshold: Some(0.3),
        profile: None,
        faults: outage_s.map(|(start_s, end_s)| FaultManifest {
            format: MANIFEST_FORMAT.into(),
            version: MANIFEST_VERSION,
            faults: vec![Fault::GnssOutage {
                node_id: DRONE.into(),
                start_s,
                end_s,
            }],
        }),
        ..RunOptions::field_v3()
    };
    let exploration = options.exploration.as_mut().unwrap();
    exploration.fault_return.as_mut().unwrap().lost_position = lost;
    options
}

fn soon_lost(home_after_s: f64) -> Option<LostPositionConfig> {
    Some(LostPositionConfig {
        hold_sigma_m: HOLD_SIGMA_M,
        home_after_s,
    })
}

fn phase(frame: &Value) -> &str {
    frame["scan_mission_state"]["phase"].as_str().unwrap()
}

fn estimate(frame: &Value) -> &Value {
    frame["navigation"]
        .as_array()
        .unwrap()
        .iter()
        .find(|pose| pose["node_id"] == DRONE)
        .expect("a pose")
}

fn lost(frame: &Value) -> bool {
    estimate(frame)["horizontal_sigma_m"].as_f64().unwrap() > HOLD_SIGMA_M
}

fn position(frame: &Value) -> [f64; 3] {
    let node = frame["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["node_id"] == DRONE)
        .expect("node in frame");
    std::array::from_fn(|k| node["position"][k].as_f64().unwrap())
}

fn step_m(frames: &[Value], k: usize) -> f64 {
    let (a, b) = (position(&frames[k - 1]), position(&frames[k]));
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn widest_step(frames: &[Value]) -> f64 {
    (1..frames.len())
        .map(|k| step_m(frames, k))
        .fold(0.0, f64::max)
}

fn contacts(frames: &[Value]) -> usize {
    frames
        .iter()
        .map(|frame| frame["world_contacts"].as_array().map_or(0, Vec::len))
        .sum()
}

#[test]
fn the_profile_is_field_v2_and_what_it_adds() {
    assert_eq!(RunProfile::parse("field_v3"), Ok(RunProfile::FieldV3));
    let v2 = RunProfile::FieldV2.options();
    let v3 = RunProfile::FieldV3.options();
    let mut stripped = v3.clone();
    stripped.profile = v2.profile;
    let exploration = stripped.exploration.as_mut().unwrap();
    exploration.navigation = None;
    exploration.quality.as_mut().unwrap().max_horizontal_sigma_m = None;
    exploration.fault_return.as_mut().unwrap().lost_position = None;
    exploration.separation_surface = false;
    exploration.follow_up = None;
    exploration.route_3d = None;
    exploration.flown_handovers = false;
    exploration.end_when_all_home = false;
    assert_eq!(stripped, v2);

    let (meta, _) = replay(&v3);
    assert_eq!(meta["run_profile"]["id"], "field_v3");
    // On another receiver it is not the profile, nor is v2 with one.
    for profile in [RunProfile::FieldV3, RunProfile::FieldV2] {
        let mut other = profile.options();
        other.exploration.as_mut().unwrap().navigation = Some(NavigationConfig::standalone());
        let (meta, _) = replay(&other);
        assert_eq!(
            meta["run_profile"]["id"],
            format!("{}+navigation", profile.as_str())
        );
    }
}

#[test]
fn the_whole_mission_touches_nothing_and_completes() {
    let (_, frames) = replay(&whole_mission(None, Some(LostPositionConfig::default())));
    assert_eq!(phase(frames.last().unwrap()), "complete");
    assert_eq!(contacts(&frames), 0);
    let quality = &frames.last().unwrap()["mapping_state"]["belief"]["quality"];
    assert!(quality["met_fraction"].as_f64().unwrap() > 0.0);
}

/// It stops while it is lost, and goes on when it knows where it is: flown
/// back to where its controller has it, not put there.
#[test]
fn a_lost_drone_stops_and_goes_on() {
    let outage = Some((20.0, 60.0));
    let (_, held) = replay(&whole_mission(outage, soon_lost(1.0e6)));
    let (_, flown_on) = replay(&whole_mission(outage, None));
    let first = held.iter().position(lost).expect("the outage loses it");
    let last = held.iter().rposition(lost).unwrap();
    assert!(last > first + 25, "lost for 25 s and more: {first} {last}");
    // Its believed way there, once it has braked.
    let believed = |frames: &[Value], k: usize| -> f64 {
        let at = |k: usize, i: usize| estimate(&frames[k])["position_m"][i].as_f64().unwrap();
        (at(k, 0) - at(k - 1, 0)).hypot(at(k, 1) - at(k - 1, 1))
    };
    let braked = first + 12;
    let way = |frames: &[Value]| (braked..last).map(|k| believed(frames, k)).sum::<f64>();
    assert!(
        way(&held) < 0.1 * way(&flown_on),
        "{} of {}",
        way(&held),
        way(&flown_on)
    );
    // Going on again: moving, and no faster than it catches up, 4 m/s of
    // the error it has just learned of beside.
    assert!((last + 1..last + 30).any(|k| step_m(&held, k) > 1.0));
    assert!(
        widest_step(&held) <= 1.25 * widest_step(&flown_on) + 4.0,
        "{} against {}",
        widest_step(&held),
        widest_step(&flown_on)
    );
    assert_eq!(contacts(&held), 0);
}

#[test]
fn a_drone_lost_for_long_goes_home_for_good() {
    let options = whole_mission(Some((20.0, 1.0e6)), soon_lost(15.0));
    let mut run = run_with(&options);
    let sent = (0..bundle().steps)
        .find(|_| {
            run.step().unwrap();
            let driver = run
                .sim
                .stage_mut::<MissionDriverStage>(Stage::EvaluateMissionTransitions)
                .expect("a mission driver");
            driver.homeward_faults().get(DRONE) == Some(&SensorLost::Position)
        })
        .expect("sent home");
    assert!((34..=38).contains(&sent), "15 s after its last fix: {sent}");
    // At the pad in the end, as well as it knows where that is.
    let (_, frames) = replay(&options);
    let home = position(&frames[0]);
    let end = frames.len() - 1;
    let at = position(&frames[end]);
    let away_m = (at[0] - home[0]).hypot(at[1] - home[1]);
    let drift_m = estimate(&frames[end])["horizontal_sigma_m"]
        .as_f64()
        .unwrap();
    let radius_m = bundle().mission.unwrap().egress_arrival_radius_m;
    assert!(
        away_m <= radius_m + 3.0 * drift_m,
        "{away_m} m from home, knowing it to {drift_m}"
    );
    assert_eq!(contacts(&frames), 0);
}

#[test]
fn a_held_run_resumes_exactly() {
    let options = whole_mission(Some((20.0, 60.0)), soon_lost(30.0));
    let (_, straight) = replay(&options);
    let first = straight.iter().position(lost).unwrap();
    let last = straight.iter().rposition(lost).unwrap();
    for at in [first + 2, first + 60, last + 2] {
        let mut original = run_with(&options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&options);
        resumed.restore(&checkpoint).expect("restore accepted");
        for (k, frame) in straight.iter().enumerate().skip(at) {
            resumed.step().unwrap();
            let now = serde_json::to_value(resumed.last_frame().unwrap()).unwrap();
            assert_eq!(&now, frame, "resumed at {at}, tick {k}");
        }
    }
}

#[test]
fn what_needs_navigation_is_refused_without_it() {
    let mut options = RunOptions::field_v3();
    options.exploration.as_mut().unwrap().navigation = None;
    options
        .exploration
        .as_mut()
        .unwrap()
        .quality
        .as_mut()
        .unwrap()
        .max_horizontal_sigma_m = None;
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let error = options.apply(&mut run, &bundle).unwrap_err();
    assert!(error.contains("lost position"), "{error}");
}

/// Whether stage 9 recorded a decision of `kind` for `reason`, by tick.
fn decided(options: &RunOptions, kind: &str) -> Vec<(u64, String)> {
    let mut run = run_with(options);
    let mut found = Vec::new();
    for tick in 0..bundle().steps {
        run.step().unwrap();
        let driver = run
            .sim
            .stage_mut::<MissionDriverStage>(Stage::EvaluateMissionTransitions)
            .expect("a mission driver");
        for decision in driver.decisions() {
            if decision.kind == kind {
                found.push((tick, decision.reason.to_string()));
            }
        }
    }
    found
}

fn met(frame: &Value) -> u64 {
    frame["mapping_state"]["belief"]["quality"]["met_cells"]
        .as_u64()
        .expect("a met count")
}

/// Once the points are inspected the fleet goes back over what falls short,
/// then home; a run checkpointed in it finishes as the uninterrupted one.
#[test]
fn the_fleet_follows_up_on_the_map_before_it_goes_home() {
    let with = |follow_up: Option<FollowUpConfig>| {
        let mut options = whole_mission(None, None);
        options.exploration.as_mut().unwrap().follow_up = follow_up;
        options
    };
    let follow_up = FollowUpConfig {
        max_s: 40.0,
        window_s: 10.0,
        min_cells_per_drone_min: 0.0,
        enough_fraction: 0.0,
    };
    let options = with(Some(follow_up.clone()));
    let decisions = decided(&options, "follow_up");
    assert_eq!(
        decisions
            .iter()
            .map(|(_, r)| r.as_str())
            .collect::<Vec<_>>(),
        ["map_falls_short", "time_up"]
    );
    let (start, end) = (decisions[0].0, decisions[1].0);
    assert_eq!(end - start, 40, "the time it was given");
    let (_, followed) = replay(&options);
    let (_, straight_home) = replay(&with(None));
    let (a, b) = (followed.last().unwrap(), straight_home.last().unwrap());
    assert_eq!((phase(a), phase(b)), ("complete", "complete"));
    assert!(met(a) > met(b), "{} against {}", met(a), met(b));
    assert_eq!(contacts(&followed), 0);
    // Nothing gained is nothing to stay for.
    let idle = with(Some(FollowUpConfig {
        min_cells_per_drone_min: 1.0e9,
        ..follow_up
    }));
    let ended = decided(&idle, "follow_up");
    assert_eq!(ended[1].1, "little_gained");
    assert_eq!(ended[1].0 - ended[0].0, 10, "its window");

    let at = start as usize + 15;
    let mut original = run_with(&options);
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&options);
    resumed.restore(&checkpoint).expect("restore accepted");
    for (k, frame) in followed.iter().enumerate().skip(at) {
        resumed.step().unwrap();
        let now = serde_json::to_value(resumed.last_frame().unwrap()).unwrap();
        assert_eq!(&now, frame, "tick {k}");
    }
}

/// A scan whose every drone has gone home for good is over, short of its
/// gate: without that, the mission scans on with nothing to scan.
#[test]
fn a_scan_with_every_drone_home_ends() {
    let mut options = whole_mission(None, None);
    options.scan_coverage_threshold = Some(1.0);
    options.exploration.as_mut().unwrap().follow_up = None;
    let drones: Vec<String> = replay(&whole_mission(None, None)).1[0]["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|node| node["is_mobile"] == true)
        .map(|node| node["node_id"].as_str().unwrap().to_string())
        .collect();
    options.faults = Some(FaultManifest {
        format: MANIFEST_FORMAT.into(),
        version: MANIFEST_VERSION,
        faults: drones
            .into_iter()
            .map(|node_id| Fault::MappingOutage {
                node_id,
                start_s: 20.0,
                end_s: 1.0e6,
            })
            .collect(),
    });
    let (_, ended) = replay(&options);
    assert_eq!(phase(ended.last().unwrap()), "complete");
    let last = &ended.last().unwrap()["scan_mission_state"];
    let threshold = last["scan_coverage_threshold"].as_f64().unwrap();
    assert!(
        last["scan_coverage_fraction"].as_f64().unwrap() < threshold,
        "and is said to be short of its gate"
    );
    options.exploration.as_mut().unwrap().end_when_all_home = false;
    let (_, open) = replay(&options);
    assert_eq!(phase(open.last().unwrap()), "scanning");
}

fn routed(scan_threshold: f64, routed: bool) -> RunOptions {
    let mut options = whole_mission(None, None);
    options.scan_coverage_threshold = Some(scan_threshold);
    let exploration = options.exploration.as_mut().unwrap();
    exploration.follow_up = None;
    if !routed {
        exploration.route_3d = None;
    }
    options
}

/// The fleet's energy left at the end, watt-hours.
fn left_wh(frames: &[Value]) -> f64 {
    frames.last().unwrap()["energy_reserves"]
        .as_object()
        .unwrap()
        .values()
        .map(|reserve| reserve["remaining_wh"].as_f64().unwrap())
        .sum()
}

/// Mission flights routed round and over the fixture's hills touch nothing,
/// spend less, and a run checkpointed among them finishes as it would have.
#[test]
fn routes_in_three_dimensions_spend_less_and_touch_nothing() {
    let options = routed(0.2, true);
    let (_, with) = replay(&options);
    let (_, without) = replay(&routed(0.2, false));
    assert_eq!(phase(with.last().unwrap()), "complete");
    assert_eq!(contacts(&with), 0);
    assert!(
        left_wh(&with) > left_wh(&without),
        "{} against {}",
        left_wh(&with),
        left_wh(&without)
    );
    let first = with
        .iter()
        .position(|frame| phase(frame) == "inspecting")
        .unwrap();
    let at = first + 10;
    let mut original = run_with(&options);
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&options);
    resumed.restore(&checkpoint).expect("restore accepted");
    for (k, frame) in with.iter().enumerate().skip(at) {
        resumed.step().unwrap();
        let now = serde_json::to_value(resumed.last_frame().unwrap()).unwrap();
        assert_eq!(&now, frame, "tick {k}");
    }
}

/// The measurement behind docs/migration/ROUTES_3D.md: seven missions, which
/// differ by when the scan ends and so by where the drones are then.
#[test]
#[ignore = "seven missions twice: run it to measure"]
fn routes_in_three_dimensions_over_seven_missions() {
    let mut saved_wh = 0.0;
    for threshold in [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5] {
        let runs = [false, true].map(|on| replay(&routed(threshold, on)).1);
        let done = runs.each_ref().map(|frames| {
            frames
                .iter()
                .position(|frame| phase(frame) == "complete")
                .expect("completes")
        });
        let left = runs.each_ref().map(|frames| left_wh(frames));
        println!("{threshold}: done at {done:?}, {left:?} Wh left");
        saved_wh += left[1] - left[0];
    }
    assert!(saved_wh > 0.0, "{saved_wh} Wh");
}

/// Flights planned again as the map changes under them: the same run whatever
/// the workers plan them, and a run stopped with planning pending goes on as it
/// would have.
#[test]
fn flights_planned_again_do_not_depend_on_who_plans_them() {
    let mut options = whole_mission(None, None);
    let exploration = options.exploration.as_mut().unwrap();
    exploration.follow_up = None;
    exploration.route_3d.as_mut().unwrap().replan = Some(ReplanConfig {
        check_s: 1.0,
        latency_s: 3.0,
        capacity: 4,
    });
    options.scan_coverage_threshold = Some(0.2);
    let run = |workers: usize| {
        let bundle = bundle();
        let pool = std::sync::Arc::new(ComputePool::new(workers).unwrap());
        let mut run = assemble_with(&bundle, pool).unwrap();
        options.apply(&mut run, &bundle).expect("options apply");
        run
    };
    let frames = |run: AssembledRun| -> Vec<Value> {
        let document = run.run_to_replay().expect("runs");
        let frames = document.frames.iter();
        frames.map(|f| serde_json::to_value(f).unwrap()).collect()
    };
    let straight = frames(run(1));
    assert_eq!(straight, frames(run(4)), "four workers");
    assert_eq!(phase(straight.last().unwrap()), "complete");
    assert_eq!(contacts(&straight), 0);
    let mut stepped = run(2);
    let mut pending_at = Vec::new();
    for tick in 0..bundle().steps as usize {
        let checkpoint = stepped.sim.checkpoint().unwrap();
        let text = serde_json::to_string(&checkpoint).unwrap();
        if text.contains("\"input_revision\"") && pending_at.len() < 2 {
            let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
            let mut resumed = run(1);
            resumed.restore(&checkpoint).expect("restore accepted");
            for (k, frame) in straight.iter().enumerate().skip(tick) {
                resumed.step().unwrap();
                let now = serde_json::to_value(resumed.last_frame().unwrap()).unwrap();
                assert_eq!(&now, frame, "resumed at {tick}, tick {k}");
            }
            pending_at.push(tick);
        }
        stepped.step().unwrap();
    }
    assert!(!pending_at.is_empty(), "a flight was planned again");
}

/// Every point dwelt at is settled on its views, met or short with a reason,
/// before the fleet goes home; and a run stopped on the way to a viewpoint goes on
/// as it would have.
#[test]
fn points_are_settled_on_their_views_before_the_fleet_goes_home() {
    let mut options = whole_mission(None, None);
    let exploration = options.exploration.as_mut().unwrap();
    exploration.follow_up = None;
    let without = replay(&options);
    assert!(without.0.get("inspection_evidence").is_none());
    options.exploration.as_mut().unwrap().inspection_evidence = Some(EvidenceConfig {
        max_s: 100.0,
        ..EvidenceConfig::default()
    });
    let (meta, frames) = replay(&options);
    assert_eq!(phase(frames.last().unwrap()), "complete");
    assert_eq!(contacts(&frames), 0);
    let points = meta["inspection_evidence"].as_object().unwrap();
    let asked = frames.last().unwrap()["scan_mission_state"]["poi_statuses"]
        .as_array()
        .unwrap()
        .len();
    assert_eq!(points.len(), asked);
    let mut met = 0;
    for (id, point) in points {
        let views = point["views"].as_array().unwrap();
        let verdict = point["verdict"].as_str().expect("settled");
        assert_eq!(verdict == "met", views.len() >= 3, "{id}: {point}");
        met += usize::from(verdict == "met");
        for view in views {
            let range = view["range_m"].as_f64().unwrap();
            let elevation = view["elevation_deg"].as_f64().unwrap();
            assert!(
                range <= 60.0 && (15.0..=75.0).contains(&elevation),
                "{view}"
            );
        }
    }
    assert!(met > 0, "{points:?}");

    let first = {
        let mut run = run_with(&options);
        (0..bundle().steps)
            .find(|_| {
                run.step().unwrap();
                let driver = run
                    .sim
                    .stage_mut::<MissionDriverStage>(Stage::EvaluateMissionTransitions)
                    .expect("a mission driver");
                driver
                    .inspection_evidence()
                    .is_some_and(|evidence| !evidence.tours.is_empty())
            })
            .expect("a drone goes back")
    };
    for at in [first as usize + 3, first as usize + 20] {
        let mut original = run_with(&options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&options);
        resumed.restore(&checkpoint).expect("restore accepted");
        for (k, frame) in frames.iter().enumerate().skip(at) {
            resumed.step().unwrap();
            let now = serde_json::to_value(resumed.last_frame().unwrap()).unwrap();
            assert_eq!(&now, frame, "resumed at {at}, tick {k}");
        }
    }
}
