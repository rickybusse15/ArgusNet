//! W2: wires fitted from what was measured.
//!
//! The gates: tracking needs measured wires and is refused otherwise; fitted
//! tracks lie on the true conductors though nothing tells the tracker which
//! wire a point came from; the map's fitted spans are named by track, never
//! by the true line, and keep more than the base clearance; flying them
//! comes within 10 m of a conductor less often than the height band alone;
//! the run is deterministic and resumes exactly; and without the tracker
//! nothing is recorded.
//!
//! Truth appears only here, in the tests.

use argusnet_mapping::wires::{WireEstimate, WireTrackerConfig};
use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::rig::RigProfile;
use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, build_world, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::wire_sensing::WireSensorModel;
use argusnet_world::powerline::{closest_on_segment, PowerLine};

/// The scan fixture with a double conductor line across the map, its top
/// 260 m above the datum, near the height drones fly.
fn bundle() -> ScenarioBundle {
    bundle_at(0.0, 260.0)
}

/// The same line at `x_m`, its top `top_m` above the datum.
fn bundle_at(x_m: f64, top_m: f64) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    let support = |y: f64| serde_json::json!({ "xy_m": [x_m, y], "base_z_m": 100.0, "height_m": top_m - 100.0 });
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "across",
        "supports": [support(-300.0), support(0.0), support(300.0)],
        "conductor_offsets_m": [-2.0, 2.0],
        "sag_ratio": 0.02,
        "conductor_radius_m": 0.015,
        "clearance_m": 10.0
    }]);
    ScenarioBundle::from_json(&document.to_string()).unwrap()
}

fn config(track: bool) -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(RigProfile::provisional()),
        measured_wires: Some(WireSensorModel::default()),
        wire_tracker: track.then(WireTrackerConfig::default),
        ..ExplorationConfig::default()
    }
}

/// Under efficient_v1, the only policy that plans round wires.
fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        world_knowledge: WorldKnowledge::NoPrior,
        exploration: Some(config),
        ..Default::default()
    }
    .apply(&mut run, bundle)
    .expect("assembles");
    run
}

fn true_lines(bundle: &ScenarioBundle) -> Vec<PowerLine> {
    build_world(&bundle.world).unwrap().power_lines.to_vec()
}

/// How far `point_m` is from the nearest true conductor.
fn off_conductor(lines: &[PowerLine], point_m: [f64; 3]) -> f64 {
    lines
        .iter()
        .flat_map(|line| line.pieces())
        .map(|piece| {
            let on = closest_on_segment(point_m, piece.start_m, piece.end_m);
            (0..3)
                .map(|k| (on[k] - point_m[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

/// How many platform-ticks came within 10 m of a true conductor.
fn ticks_near(bundle: &ScenarioBundle, track: bool) -> usize {
    let lines = true_lines(bundle);
    let mut run = run_with(bundle, config(track));
    let mut last = std::collections::BTreeMap::new();
    let mut near = 0;
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|k| node["position"][k].as_f64().unwrap());
            let id = node["node_id"].as_str().unwrap().to_string();
            let from = last.insert(id, p).unwrap_or(p);
            if lines
                .iter()
                .any(|line| line.clearance_along(from, p) < 10.0)
            {
                near += 1;
            }
        }
    }
    near
}

fn confirmed(run: &AssembledRun) -> Vec<WireEstimate> {
    let grid = run.mapping().unwrap().belief_grid().unwrap();
    let tracker = grid.wire_tracker().expect("tracking");
    tracker
        .estimates()
        .into_iter()
        .filter(|estimate| estimate.confirmed)
        .collect()
}

fn frames(run: AssembledRun) -> Vec<serde_json::Value> {
    run.run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

#[test]
fn tracking_needs_measured_wires() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                measured_wires: None,
                ..config(true)
            },
        )
        .unwrap_err();
    assert!(error.contains("measured"), "{error}");
}

/// Every confirmed track is one conductor, with both ends and its sagged
/// middle on the true wire. Nothing told it which wire a point came from.
#[test]
fn fitted_tracks_lie_on_the_true_conductors() {
    let bundle = bundle();
    let lines = true_lines(&bundle);
    let mut run = run_with(&bundle, config(true));
    for _ in 0..bundle.steps {
        run.step().unwrap();
    }
    let tracks = confirmed(&run);
    assert!(tracks.len() >= 2, "{} confirmed", tracks.len());
    for track in &tracks {
        let middle: [f64; 3] = std::array::from_fn(|k| (track.from_m[k] + track.to_m[k]) / 2.0);
        let middle = [middle[0], middle[1], middle[2] - track.sag_m];
        assert!(
            off_conductor(&lines, middle) < 1.0,
            "track {} is {:.2} m off a conductor at its middle",
            track.id,
            off_conductor(&lines, middle)
        );
        assert!(track.sigma_lateral_m < 1.0 && track.sigma_vertical_m < 1.0);
    }
}

#[test]
fn the_map_names_fitted_spans_by_track_and_widens_their_clearance() {
    let bundle = bundle();
    let frames = frames(run_with(&bundle, config(true)));
    let spans: Vec<&serde_json::Value> = frames
        .iter()
        .flat_map(|frame| frame["mapped_wires"].as_array().into_iter().flatten())
        .collect();
    assert!(!spans.is_empty(), "the map gained fitted spans");
    for span in spans {
        let id = span["id"].as_str().unwrap();
        assert!(id.starts_with("wire-track-"), "{id}");
        assert!(span["clearance_m"].as_f64().unwrap() > 10.0);
    }
    let last = frames.last().unwrap();
    assert!(
        last["mapping_state"]["belief"]["wire_tracks"]
            .as_u64()
            .unwrap()
            >= 2
    );
}

/// Measured 2026-11-04 on this scenario: 13 platform-ticks within 10 m with
/// the height band alone, 5 with fitted wires.
#[test]
fn flying_fitted_wires_comes_near_a_conductor_less_often() {
    let bundle = bundle();
    let (band, fitted) = (ticks_near(&bundle, false), ticks_near(&bundle, true));
    assert!(fitted < band, "fitted {fitted} against band {band}");
}

/// The most ticks in a row any platform spends within 10 m of a conductor.
fn longest_stay(bundle: &ScenarioBundle, config: ExplorationConfig) -> usize {
    let lines = true_lines(bundle);
    let mut run = run_with(bundle, config);
    let mut stays = std::collections::BTreeMap::new();
    let mut longest = 0;
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        for node in frame["nodes"].as_array().unwrap() {
            if node["is_mobile"] != true {
                continue;
            }
            let p: [f64; 3] = std::array::from_fn(|k| node["position"][k].as_f64().unwrap());
            let stay = stays.entry(node["node_id"].to_string()).or_insert(0);
            let near = lines.iter().any(|line| line.clearance_along(p, p) < 10.0);
            *stay = if near { *stay + 1 } else { 0 };
            longest = longest.max(*stay);
        }
    }
    longest
}

/// A line whose top is near flight height, over ground that rises to meet
/// it. Stopped there, a drone once stayed for 33 ticks, up against the wire
/// with the ground below, while its explorer flew on. It now gets away, with
/// or without fitted wires.
#[test]
fn a_drone_stopped_at_a_wire_gets_away() {
    for (x_m, top_m) in [(0.0, 340.0), (-120.0, 340.0)] {
        let bundle = bundle_at(x_m, top_m);
        for track in [false, true] {
            let longest = longest_stay(&bundle, config(track));
            assert!(longest <= 8, "x {x_m} track {track}: {longest} ticks");
        }
    }
}

#[test]
fn a_tracked_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let straight = frames(run_with(&bundle, config(true)));
    assert_eq!(straight, frames(run_with(&bundle, config(true))));
    let at = 150usize;
    let mut original = run_with(&bundle, config(true));
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&bundle, config(true));
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);
}

/// Applying the frames gives the wires the map holds, merged tracks gone.
#[test]
fn the_replay_holds_the_spans_the_map_holds() {
    let bundle = bundle();
    let mut run = run_with(&bundle, config(true));
    let mut held = std::collections::BTreeSet::new();
    for _ in 0..bundle.steps {
        run.step().unwrap();
        let frame = serde_json::to_value(run.last_frame().unwrap()).unwrap();
        for span in frame["mapped_wires"].as_array().into_iter().flatten() {
            held.insert(span["id"].as_str().unwrap().to_string());
        }
        for id in frame["retired_wires"].as_array().into_iter().flatten() {
            assert!(held.remove(id.as_str().unwrap()), "{id} was never held");
        }
    }
    let map: std::collections::BTreeSet<String> = run
        .mapping()
        .unwrap()
        .belief_wires()
        .iter()
        .cloned()
        .collect();
    assert_eq!(held, map);
}

#[test]
fn without_the_tracker_nothing_is_recorded() {
    let bundle = bundle();
    let frames = frames(run_with(&bundle, config(false)));
    assert!(frames.iter().all(|frame| frame["mapping_state"]["belief"]
        .get("wire_tracks")
        .is_none()));
}
