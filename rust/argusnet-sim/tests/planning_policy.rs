//! P1: `efficient_v1` allocates the scan mission's POIs fleet-wide. Opt-in:
//! the reference run is untouched, and the replay records the policy only
//! when it is not the reference.

use argusnet_sim::run_options::{PlanningPolicy, RunOptions};
use argusnet_sim::scenario::{assemble, AssembledRun, RunError, ScenarioBundle};
use argusnet_sim::simulation::TickError;

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run(case: &str, policy: PlanningPolicy) -> AssembledRun {
    let bundle = bundle(case);
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        planning_policy: policy,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    run
}

/// Every POI assignment decision the run makes, as `(drone, poi, reason)`.
fn assignments(mut run: AssembledRun) -> Vec<(String, String, String)> {
    let mut made = Vec::new();
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let body = run.journal_body();
        for decision in body["decisions"].as_array().into_iter().flatten() {
            if decision["kind"] == "poi_assignment" {
                made.push((
                    decision["subject"].as_str().unwrap().to_string(),
                    decision["selected"].as_str().unwrap().to_string(),
                    decision["reason"].as_str().unwrap().to_string(),
                ));
            }
        }
    }
    made
}

#[test]
fn the_replay_records_a_non_reference_policy_only() {
    let meta = |policy| {
        let replay = run("scan", policy).run_to_replay().expect("runs");
        serde_json::to_value(&replay.meta).expect("meta")
    };
    assert!(meta(PlanningPolicy::Reference)
        .get("planning_policy")
        .is_none());
    assert_eq!(
        meta(PlanningPolicy::EfficientV1)["planning_policy"],
        "efficient_v1"
    );
    assert!(!RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        ..RunOptions::default()
    }
    .is_legacy());
}

#[test]
fn efficient_v1_assigns_the_scan_missions_pois_fleet_wide() {
    let made = assignments(run("scan", PlanningPolicy::EfficientV1));
    assert!(!made.is_empty(), "the scan mission reaches its POIs");
    assert!(made
        .iter()
        .all(|(_, _, reason)| reason == "fleet_min_cost" || reason == "fleet_team_join"));
    let reference = assignments(run("scan", PlanningPolicy::Reference));
    assert!(reference
        .iter()
        .all(|(_, _, reason)| reason == "energy_aware_priority"));
}

#[test]
fn a_reset_efficient_run_is_a_fresh_one() {
    let fresh = serde_json::to_string(
        &run("scan", PlanningPolicy::EfficientV1)
            .run_to_replay()
            .expect("runs"),
    )
    .unwrap();
    let mut reset = run("scan", PlanningPolicy::EfficientV1);
    for _ in 0..60 {
        reset.step().expect("steps");
    }
    reset.reset();
    let again = serde_json::to_string(&reset.run_to_replay().expect("runs")).unwrap();
    assert!(fresh == again, "a reset keeps the policy and starts over");
}

/// A checkpoint taken under `efficient_v1` carries runtime algorithm 3, so a
/// build that predates the policy (which accepts only 2) refuses it instead
/// of resuming it under the reference planner. It resumes into an
/// `efficient_v1` run; a reference run refuses it on the driver's policy. A
/// reference checkpoint is unchanged at 2.
#[test]
fn an_efficient_checkpoint_is_versioned_and_resumes_only_as_efficient() {
    use argusnet_sim::snapshot::CheckpointHeader;
    let mut source = run("scan", PlanningPolicy::EfficientV1);
    for _ in 0..40 {
        source.step().expect("steps");
    }
    let checkpoint = source.sim.checkpoint().expect("not faulted");
    assert_eq!(
        checkpoint.header.runtime_algorithm_version,
        CheckpointHeader::RUNTIME_ALGORITHM_VERSION_POLICY
    );
    assert!(run("scan", PlanningPolicy::EfficientV1)
        .restore(&checkpoint)
        .is_ok());
    let refused = run("scan", PlanningPolicy::Reference)
        .restore(&checkpoint)
        .unwrap_err();
    assert!(format!("{refused}").contains("efficient_v1"), "{refused}");

    let mut reference = run("scan", PlanningPolicy::Reference);
    reference.step().expect("steps");
    assert_eq!(
        reference
            .sim
            .checkpoint()
            .expect("not faulted")
            .header
            .runtime_algorithm_version,
        CheckpointHeader::RUNTIME_ALGORITHM_VERSION
    );
}

/// P2 under a mission lane slower than the boundary: a drone that reaches its
/// reserve on a boundary the lane sits out turns for home on that boundary.
/// The egress the driver queues there reaches stage 2, which flies it from
/// the next boundary on.
#[test]
fn a_drone_at_its_reserve_between_mission_runs_turns_for_home() {
    use argusnet_sim::overrides::{MissionOverrideStage, OverrideSource};
    use argusnet_sim::scheduler::{parse_rate, LaneRates, PhasedLaneExecutor};
    use argusnet_sim::stage::Stage;
    let bundle = bundle("scan");
    let dt_ms = (bundle.dt_s * 1000.0).round() as u64;
    let rates = LaneRates {
        mission: Some(parse_rate(&format!("1000/{}", 10 * dt_ms)).unwrap()),
        ..LaneRates::default()
    };
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        planning_policy: PlanningPolicy::EfficientV1,
        lane_rates: rates,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    let mut lanes = rates
        .apply(PhasedLaneExecutor::new::<RunError>(bundle.dt_s, 0).expect("exact timestep"))
        .expect("rates");
    // The lane runs at boundaries 0, 10, 20, ...: stop after 11, so the
    // next boundary is one it sits out.
    for _ in 0..12 {
        lanes.step(&mut run).expect("steps");
    }
    let source = |run: &AssembledRun| {
        run.sim
            .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
            .expect("installed")
            .applied_source("drone-east")
    };
    assert_ne!(source(&run), Some(OverrideSource::Egress));
    assert!(run.battery_mut().set_remaining_wh("drone-east", 0.5));
    lanes
        .step(&mut run)
        .expect("a boundary the mission lane sits out");
    lanes.step(&mut run).expect("the next boundary");
    assert_eq!(source(&run), Some(OverrideSource::Egress));
}

/// P3: under `efficient_v1` deconfliction's lateral push is kept out of the
/// ground and out of structures, as in a run with no prior map; the
/// reference keeps its unguarded push.
#[test]
fn efficient_v1_guards_deconfliction_and_the_reference_does_not() {
    use argusnet_sim::deconflict::DeconflictStage;
    use argusnet_sim::stage::Stage;
    let guarded = |policy| {
        run("scan", policy)
            .sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .expect("installed")
            .is_guarded()
    };
    assert!(guarded(PlanningPolicy::EfficientV1));
    assert!(!guarded(PlanningPolicy::Reference));
}

/// The guard belongs to the policy, not to the mission: a target-follow run
/// with no mission driver is guarded too, and going back to the reference
/// before the first tick takes the guard away again.
#[test]
fn the_deconfliction_guard_follows_the_policy_alone() {
    use argusnet_sim::deconflict::DeconflictStage;
    use argusnet_sim::stage::Stage;
    let guarded = |run: &AssembledRun| {
        run.sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .expect("installed")
            .is_guarded()
    };
    let mut follow = assemble(&bundle("tt")).expect("assembles");
    assert!(follow.driver().is_none(), "no mission driver");
    assert!(!follow
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap());
    assert!(guarded(&follow));

    let mut scan = assemble(&bundle("scan")).expect("assembles");
    scan.set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    scan.set_planning_policy(PlanningPolicy::Reference).unwrap();
    assert!(!guarded(&scan));

    // A driverless checkpoint resumes only under its own policy: the guard
    // is the difference, and no driver state is there to refuse it.
    let mut source = assemble(&bundle("tt")).expect("assembles");
    source
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    for _ in 0..5 {
        source.sim.step().expect("steps");
    }
    let checkpoint = source.sim.checkpoint().expect("not faulted");
    let mut reference = assemble(&bundle("tt")).expect("assembles");
    let refused = reference.restore(&checkpoint).unwrap_err();
    assert!(
        format!("{refused}").contains("planning policy"),
        "{refused}"
    );
    let mut efficient = assemble(&bundle("tt")).expect("assembles");
    efficient
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    assert!(efficient.restore(&checkpoint).is_ok());
    // A checkpoint from an older efficient algorithm version is refused
    // for its version, not as a different policy.
    let mut older = checkpoint.clone();
    older.header.runtime_algorithm_version -= 1;
    let mut fresh = assemble(&bundle("tt")).expect("assembles");
    fresh
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    assert!(matches!(
        fresh.restore(&older),
        Err(argusnet_sim::snapshot::RestoreError::AlgorithmVersion { .. })
    ));
    // A restored run has begun, even at tick 0 with an empty trace.
    assert!(efficient
        .set_planning_policy(PlanningPolicy::Reference)
        .is_err());
    assert!(guarded(&efficient));

    // Once tick 0 is open (stage 0 run, clock still at 0) the policy is
    // fixed.
    let mut opened = assemble(&bundle("tt")).expect("assembles");
    let _open = opened.sim.begin_tick().expect("begins");
    assert_eq!(opened.sim.tick(), 0);
    assert!(opened
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .is_err());
    assert!(!guarded(&opened));

    // After the first tick the policy is fixed, with or without a driver.
    follow.sim.step().expect("steps");
    assert!(follow
        .set_planning_policy(PlanningPolicy::Reference)
        .is_err());
    assert!(guarded(&follow));
    scan.sim.step().expect("steps");
    assert!(scan
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .is_err());
    assert!(!guarded(&scan));
}

/// R3: under `efficient_v1` deconfliction's pushes and the rejoins keep
/// the world's power lines' clearance, and the physical guard keeps off the
/// wires; the reference knows no lines, and a run with no prior map forgets
/// them with the rest of the map, whichever is set first.
#[test]
fn efficient_v1_keeps_power_lines_clear_and_the_reference_does_not() {
    use argusnet_sim::belief_world::WorldKnowledge;
    use argusnet_sim::deconflict::DeconflictStage;
    use argusnet_sim::overrides::MissionOverrideStage;
    use argusnet_sim::stage::Stage;
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).unwrap();
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "line",
        "supports": [
            {"xy_m": [-300.0, 0.0], "base_z_m": 0.0, "height_m": 30.0},
            {"xy_m": [300.0, 0.0], "base_z_m": 0.0, "height_m": 30.0}
        ],
        "conductor_offsets_m": [0.0],
        "sag_ratio": 0.03,
        "conductor_radius_m": 0.02,
        "clearance_m": 10.0
    }]);
    let bundle = ScenarioBundle::from_json(&document.to_string()).expect("bundle parses");
    let lines = |policy, knowledge| {
        let mut run = assemble(&bundle).expect("assembles");
        RunOptions {
            planning_policy: policy,
            world_knowledge: knowledge,
            ..RunOptions::default()
        }
        .apply(&mut run, &bundle)
        .expect("applies");
        let decon = run
            .sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .expect("installed")
            .power_lines()
            .len();
        let rejoins = run
            .sim
            .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
            .expect("installed")
            .power_lines()
            .count();
        // Physics keeps off the wires wherever there is a guard.
        let physics = run
            .sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .unwrap()
            .physical_guard()
            .map(|guard| guard.power_line_count());
        (decon, rejoins, physics)
    };
    assert_eq!(
        lines(PlanningPolicy::EfficientV1, WorldKnowledge::Prior),
        (1, 1, Some(1))
    );
    assert_eq!(
        lines(PlanningPolicy::Reference, WorldKnowledge::Prior),
        (0, 0, None)
    );
    assert_eq!(
        lines(PlanningPolicy::EfficientV1, WorldKnowledge::NoPrior),
        (0, 0, Some(1))
    );
    assert_eq!(
        lines(PlanningPolicy::Reference, WorldKnowledge::NoPrior),
        (0, 0, Some(1))
    );
    // The other order: the policy first, then a run with no prior map.
    let mut run = assemble(&bundle).expect("assembles");
    run.set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    run.set_world_knowledge(WorldKnowledge::NoPrior, Default::default())
        .unwrap();
    assert!(run
        .sim
        .stage::<DeconflictStage>(Stage::Deconflict)
        .unwrap()
        .power_lines()
        .is_empty());
    assert_eq!(
        run.sim
            .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
            .unwrap()
            .power_lines()
            .count(),
        0
    );
    // And back to the reference: the lines go with the policy's guard.
    let mut run = assemble(&bundle).expect("assembles");
    run.set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    run.set_planning_policy(PlanningPolicy::Reference).unwrap();
    assert!(run
        .sim
        .stage::<DeconflictStage>(Stage::Deconflict)
        .unwrap()
        .power_lines()
        .is_empty());
}

/// R3: a run without a prior map plans round the wires it maps, under
/// `efficient_v1` only, whichever of the policy and the world knowledge is
/// set first; with the wire-hazard prior its map holds every span from the
/// start.
#[test]
fn a_run_without_a_prior_map_plans_round_the_wires_it_maps() {
    use argusnet_sim::belief_world::{ExplorationConfig, WorldKnowledge};
    use argusnet_sim::deconflict::DeconflictStage;
    use argusnet_sim::mapping::MappingStage;
    use argusnet_sim::overrides::MissionOverrideStage;
    use argusnet_sim::stage::Stage;
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut document: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("fixture")).unwrap();
    document["world"]["power_lines"] = serde_json::json!([{
        "id": "line",
        "supports": [
            {"xy_m": [-300.0, 0.0], "base_z_m": 0.0, "height_m": 30.0},
            {"xy_m": [0.0, 0.0], "base_z_m": 0.0, "height_m": 30.0},
            {"xy_m": [300.0, 0.0], "base_z_m": 0.0, "height_m": 30.0}
        ],
        "conductor_offsets_m": [0.0],
        "sag_ratio": 0.03,
        "conductor_radius_m": 0.02,
        "clearance_m": 10.0
    }]);
    let bundle = ScenarioBundle::from_json(&document.to_string()).expect("bundle parses");
    let follows = |run: &AssembledRun| {
        let driver = run.driver().unwrap().follows_mapped_power_lines();
        let decon = run
            .sim
            .stage::<DeconflictStage>(Stage::Deconflict)
            .unwrap()
            .follows_mapped_power_lines();
        let rejoins = run
            .sim
            .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
            .unwrap()
            .follows_mapped_power_lines();
        [driver, decon, rejoins]
    };
    let none = || ExplorationConfig::default();
    let mut efficient = assemble(&bundle).expect("assembles");
    efficient
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    efficient
        .set_world_knowledge(WorldKnowledge::NoPrior, none())
        .unwrap();
    assert_eq!(follows(&efficient), [true; 3]);
    let mut other_order = assemble(&bundle).expect("assembles");
    other_order
        .set_world_knowledge(WorldKnowledge::NoPrior, none())
        .unwrap();
    assert_eq!(follows(&other_order), [false; 3]);
    other_order
        .set_planning_policy(PlanningPolicy::EfficientV1)
        .unwrap();
    assert_eq!(follows(&other_order), [true; 3]);
    other_order
        .set_planning_policy(PlanningPolicy::Reference)
        .unwrap();
    assert_eq!(follows(&other_order), [false; 3]);
    let prior = run("scan", PlanningPolicy::EfficientV1);
    assert_eq!(
        follows(&prior),
        [false; 3],
        "a prior run knows the lines already"
    );

    // The scanner looks for both spans.
    use argusnet_sim::observe::ObserveStage;
    assert_eq!(
        efficient
            .sim
            .stage::<ObserveStage>(Stage::GenerateObservations)
            .unwrap()
            .range_scanner()
            .unwrap()
            .wire_span_count(),
        2
    );

    // The prior: both spans known before any scan.
    let mapped = |config: ExplorationConfig| {
        let mut run = assemble(&bundle).expect("assembles");
        run.set_world_knowledge(WorldKnowledge::NoPrior, config)
            .unwrap();
        run.sim
            .stage::<MappingStage>(Stage::UpdateMappingAndLocalization)
            .unwrap()
            .belief_wires()
    };
    assert!(mapped(none()).is_empty());
    assert_eq!(
        mapped(ExplorationConfig {
            wire_prior: true,
            ..none()
        }),
        ["line#0", "line#1"]
    );
}
