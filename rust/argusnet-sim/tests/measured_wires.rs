//! W1: power lines measured, not looked up.
//!
//! The gates: with the model, no sensor hands over a span and the map holds
//! none; LiDAR points land on the true wires within the noise, though they
//! carry no id; camera lines point at the true wires, with no depth;
//! evidence builds a hazard band avoidance reads; the run is deterministic
//! and resumes exactly; and without the model nothing changes.
//!
//! Truth appears only here, in the tests: the sensors' output never names a
//! wire.

use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
use argusnet_sim::rig::RigProfile;
use argusnet_sim::scenario::{assemble, build_world, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;
use argusnet_sim::wire_sensing::WireSensorModel;
use argusnet_world::powerline::{closest_on_segment, PowerLine};

/// The scan fixture with a line across the middle of the map, 120 m above
/// the highest ground it crosses, so drones fly near it.
fn bundle(height_m: f64) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).expect("fixture");
    let mut document: serde_json::Value = serde_json::from_str(&text).unwrap();
    let support =
        |y: f64| serde_json::json!({ "xy_m": [0.0, y], "base_z_m": 100.0, "height_m": height_m });
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

fn config() -> ExplorationConfig {
    ExplorationConfig {
        rig: Some(RigProfile::provisional()),
        measured_wires: Some(WireSensorModel::default()),
        ..ExplorationConfig::default()
    }
}

fn run_with(bundle: &ScenarioBundle, config: ExplorationConfig) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, config)
        .expect("assembles");
    run
}

fn frames(run: AssembledRun) -> Vec<serde_json::Value> {
    run.run_to_replay()
        .expect("runs")
        .frames
        .iter()
        .map(|frame| serde_json::to_value(frame).unwrap())
        .collect()
}

/// How far `point` is from the nearest true conductor.
fn from_true_wire(lines: &[PowerLine], point: [f64; 3]) -> f64 {
    lines
        .iter()
        .flat_map(|line| line.pieces())
        .map(|piece| {
            let on = closest_on_segment(point, piece.start_m, piece.end_m);
            (0..3)
                .map(|k| (on[k] - point[k]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
        .fold(f64::INFINITY, f64::min)
}

#[test]
fn no_span_is_handed_over_and_the_points_land_on_the_wires() {
    let bundle = bundle(160.0);
    let world = build_world(&bundle.world).unwrap();
    let mut run = run_with(&bundle, config());
    let mut points = Vec::new();
    for _ in 0..bundle.steps {
        run.step().unwrap();
        for scan in run.mapping().unwrap().fused_scans() {
            assert!(scan.wires.is_empty(), "no sensor reports a span");
            points.extend(scan.wire_points.iter().copied());
        }
    }
    let mapping = run.mapping().unwrap();
    assert!(mapping.belief_wires().is_empty(), "the map holds no span");
    assert!(points.len() > 20, "{} points", points.len());
    let mut errors: Vec<f64> = points
        .iter()
        .map(|p| from_true_wire(&world.power_lines, *p))
        .collect();
    errors.sort_by(f64::total_cmp);
    // Noise, boresight bias and attitude error, at up to 250 m range: well
    // under a metre for most points, and no point is off in the field.
    let median = errors[errors.len() / 2];
    assert!(median < 1.0, "median {median} m");
    assert!(
        *errors.last().unwrap() < 5.0,
        "worst {}",
        errors.last().unwrap()
    );
}

#[test]
fn camera_lines_point_at_the_wire_with_no_depth() {
    let bundle = bundle(160.0);
    let world = build_world(&bundle.world).unwrap();
    let mut run = run_with(&bundle, config());
    let mut checked = 0;
    for _ in 0..bundle.steps {
        run.step().unwrap();
        for scan in run.mapping().unwrap().fused_scans() {
            for line in &scan.wire_lines {
                for dir in [line.from_dir, line.to_dir] {
                    // Some range along this sight line meets a true wire.
                    let nearest = (1..50)
                        .map(|r| {
                            let at: [f64; 3] =
                                std::array::from_fn(|k| scan.origin_m[k] + dir[k] * r as f64);
                            from_true_wire(&world.power_lines, at)
                        })
                        .fold(f64::INFINITY, f64::min);
                    assert!(nearest < 3.0, "{nearest} m from any wire");
                }
                checked += 1;
            }
        }
    }
    assert!(checked > 0, "the camera saw the line at least once");
}

#[test]
fn evidence_builds_a_hazard_band_in_the_replay() {
    let bundle = bundle(160.0);
    let frames = frames(run_with(&bundle, config()));
    let last = &frames.last().unwrap()["mapping_state"]["belief"];
    assert!(last["wire_points"].as_u64().unwrap() > 20, "{last}");
    assert!(last["wire_hazard_cells"].as_u64().unwrap() > 0, "{last}");
    let sampled = frames
        .iter()
        .flat_map(|f| f["range_scans"].as_array().cloned().unwrap_or_default())
        .any(|scan| scan.get("wire_points").is_some());
    assert!(sampled, "the replay carries a sample of the points");
    // Nothing in a frame names a wire.
    let text = serde_json::to_string(&frames).unwrap();
    assert!(!text.contains("across"), "no true line id in the frames");
}

/// Under `efficient_v1`, a line in the flight band moves the aircraft: the
/// hazard band is read by avoidance, not only reported. Flights over a line
/// well above them never meet it, so the evidence changes nothing.
#[test]
fn the_hazard_band_moves_flights_only_where_the_wire_is() {
    use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
    let positions = |bundle: &ScenarioBundle, measured: Option<WireSensorModel>| {
        let mut run = assemble(bundle).unwrap();
        RunOptions {
            planning_policy: PlanningPolicy::EfficientV1,
            world_knowledge: WorldKnowledge::NoPrior,
            exploration: Some(ExplorationConfig {
                rig: Some(RigProfile::provisional()),
                measured_wires: measured,
                ..ExplorationConfig::default()
            }),
            ..RunOptions::default()
        }
        .apply(&mut run, bundle)
        .unwrap();
        frames(run)
            .iter()
            .map(|frame| frame["nodes"].clone())
            .collect::<Vec<_>>()
    };
    let in_band = bundle(160.0);
    assert_ne!(
        positions(&in_band, None),
        positions(&in_band, Some(WireSensorModel::default()))
    );
    let above = bundle(240.0);
    assert_eq!(
        positions(&above, None),
        positions(&above, Some(WireSensorModel::default()))
    );
}

#[test]
fn a_measured_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle(160.0);
    let straight = frames(run_with(&bundle, config()));
    assert_eq!(straight, frames(run_with(&bundle, config())));
    let at = 150usize;
    let mut original = run_with(&bundle, config());
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    assert!(text.contains("wire_count"), "the evidence is checkpointed");
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&bundle, config());
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, straight[at..]);
}

#[test]
fn without_the_model_the_oracle_stays_as_it_was() {
    let bundle = bundle(160.0);
    let mut run = run_with(
        &bundle,
        ExplorationConfig {
            rig: Some(RigProfile::provisional()),
            ..ExplorationConfig::default()
        },
    );
    for _ in 0..bundle.steps {
        run.step().unwrap();
        for scan in run.mapping().unwrap().fused_scans() {
            assert!(scan.wire_points.is_empty() && scan.wire_lines.is_empty());
        }
    }
    assert!(
        !run.mapping().unwrap().belief_wires().is_empty(),
        "the oracle maps the span"
    );
}

#[test]
fn an_invalid_model_is_refused_before_the_run() {
    let bundle = bundle(160.0);
    let mut run = assemble(&bundle).unwrap();
    let error = run
        .set_world_knowledge(
            WorldKnowledge::NoPrior,
            ExplorationConfig {
                measured_wires: Some(WireSensorModel {
                    points_per_s: -1.0,
                    ..WireSensorModel::default()
                }),
                ..ExplorationConfig::default()
            },
        )
        .unwrap_err();
    assert!(error.contains("points_per_s"), "{error}");
}
