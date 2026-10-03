//! The loop P4.8 is really after: stage 8 measures coverage, stage 9 decides
//! what the mission does about it.
//!
//! Each half has its own unit tests. What neither can show on its own is that
//! they compose — that a drone flying a lawnmower pattern raises the coverage
//! fraction until the mission machine's own gate trips, with no test reaching
//! in to fake the number.
//!
//! This is the first place two ported subsystems drive each other through the
//! real tick transaction.

use argusnet_mapping::GridBounds;
use argusnet_sim::config::SimulationConfig;
use argusnet_sim::mapping::MappingStage;
use argusnet_sim::mission::{MissionConfig, MissionInputs, MissionPhase, MissionStage};
use argusnet_sim::simulation::Simulation;
use argusnet_sim::stage::Stage;
use argusnet_types::{NodeState, Vector3};

fn bounds() -> GridBounds {
    GridBounds {
        x_min_m: 0.0,
        x_max_m: 1_000.0,
        y_min_m: 0.0,
        y_max_m: 1_000.0,
        resolution_m: 50.0,
    }
}

fn drone(id: &str) -> NodeState {
    NodeState {
        node_id: id.to_string(),
        position: Vector3::new(100.0, 100.0, 80.0),
        velocity: Vector3::new(0.0, 0.0, 0.0),
        is_mobile: true,
        timestamp_s: 0.0,
        health: 1.0,
        ..Default::default()
    }
}

/// Stage 8's coverage, read through the session that owns it.
fn coverage_fraction(sim: &Simulation) -> f64 {
    sim.stage::<MappingStage>(Stage::UpdateMappingAndLocalization)
        .expect("stage 8 is installed")
        .coverage_fraction()
}

#[test]
fn flying_a_lawnmower_pattern_trips_the_missions_coverage_gate() {
    let mut sim = Simulation::new(SimulationConfig {
        seed: 21,
        dt_s: 1.0,
        steps: Some(4_000),
        run_id: "loop".to_string(),
    })
    .expect("config is valid");
    sim.load_world(vec![drone("drone-1")], Vec::new());

    let mut mapping = MappingStage::new(bounds()).expect("bounds are valid");
    mapping.set_footprint_radius_m(150.0);
    sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
        .unwrap();

    let mission = MissionStage::new(MissionConfig::default());
    let mission_link = mission.link();
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
        .unwrap();

    assert_eq!(mission_link.phase(), MissionPhase::Scanning);

    // A lawnmower over the 1 km square, 100 m steps.
    let mut x = 100.0;
    let mut y = 100.0;
    let mut phase_at = None;
    for step in 0..400 {
        {
            let platform = sim.world_mut().platform_mut("drone-1").unwrap();
            platform.position.x = x;
            platform.position.y = y;
        }

        // Stage 8 runs before stage 9 in the pipeline, but the mission inputs
        // are pushed in before the tick starts -- so they carry the coverage
        // as it stood at the END of the previous tick. That is the reference's
        // own ordering: `scan_frac` is read from the map after stage 8 has
        // updated it, and the phase chain consumes it on the same tick.
        let mut inputs = MissionInputs::at(sim.tick() as f64);
        inputs.mobile_count = 1;
        // Converged, so the gate that trips is the coverage one.
        inputs.localization_confidences = vec![0.99];
        inputs.gap_cell_fraction = Some(0.0);
        inputs.coverage_fraction = coverage_fraction(&sim);
        mission_link.set_inputs(inputs);

        sim.step().expect("no stage fails");

        if mission_link.phase() != MissionPhase::Scanning && phase_at.is_none() {
            phase_at = Some((step, coverage_fraction(&sim)));
            break;
        }

        x += 100.0;
        if x > 950.0 {
            x = 100.0;
            y += 100.0;
        }
        if y > 950.0 {
            y = 100.0;
        }
    }

    let (step, fraction) = phase_at.expect("the mission never left Scanning");
    assert_eq!(mission_link.phase(), MissionPhase::Inspecting);
    assert!(
        fraction >= MissionConfig::default().scan_coverage_threshold,
        "the gate tripped at {fraction}, below its own threshold"
    );
    // It must take real flying, not trip on the first tick.
    assert!(step > 0, "coverage cannot cross 70% on one footprint");
}

/// And the negative: a drone that never moves cannot cover enough ground, so
/// the mission stays in `Scanning` however long it runs. Without this the test
/// above would pass just as well against a coverage fraction that climbed for
/// the wrong reason.
#[test]
fn a_stationary_drone_never_trips_the_gate() {
    let mut sim = Simulation::new(SimulationConfig {
        seed: 22,
        dt_s: 1.0,
        steps: Some(500),
        run_id: "loop-negative".to_string(),
    })
    .expect("config is valid");
    sim.load_world(vec![drone("drone-1")], Vec::new());

    let mut mapping = MappingStage::new(bounds()).expect("bounds are valid");
    mapping.set_footprint_radius_m(150.0);
    sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
        .unwrap();

    let mission = MissionStage::new(MissionConfig::default());
    let mission_link = mission.link();
    sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
        .unwrap();

    for _ in 0..200 {
        let mut inputs = MissionInputs::at(sim.tick() as f64);
        inputs.mobile_count = 1;
        inputs.localization_confidences = vec![0.99];
        inputs.coverage_fraction = coverage_fraction(&sim);
        mission_link.set_inputs(inputs);
        sim.step().unwrap();
    }

    assert_eq!(mission_link.phase(), MissionPhase::Scanning);
    let fraction = coverage_fraction(&sim);
    assert!(fraction > 0.0, "it did cover its own footprint");
    assert!(
        fraction < MissionConfig::default().scan_coverage_threshold,
        "one footprint is {fraction}, which must not reach the gate"
    );
}
