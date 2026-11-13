//! `field_v1`: the composed field mission as one named set of options.
//!
//! The gates: the profile is one thing to ask for and is recorded as such;
//! the composed run is deterministic and resumes exactly; mission flights
//! fly over what the map believes is there, touch nothing, and wait rather
//! than jump; every drone is at its station once the mission is complete
//! and stays there; and the scan ends on the declared share of the map.

use argusnet_sim::belief_world::{ExplorationConfig, MissionFlightConfig, WorldKnowledge};
use argusnet_sim::run_options::{
    RunOptions, RunProfile, FIELD_V1_QUALITY, FIELD_V1_SCAN_THRESHOLD,
};
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::snapshot::SimulationCheckpoint;

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_scan.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

fn run_with(bundle: &ScenarioBundle, options: &RunOptions) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    options.apply(&mut run, bundle).expect("options apply");
    run
}

fn replay(run: AssembledRun) -> (serde_json::Value, Vec<serde_json::Value>) {
    let document = run.run_to_replay().expect("runs");
    (
        serde_json::to_value(&document.meta).unwrap(),
        document
            .frames
            .iter()
            .map(|frame| serde_json::to_value(frame).unwrap())
            .collect(),
    )
}

/// The profile's sensing and flight, with a scan gate the 400-tick fixture
/// reaches at its one capture a second: the whole mission, scan to home.
fn whole_mission() -> RunOptions {
    RunOptions {
        scan_coverage_threshold: Some(0.3),
        profile: None,
        ..RunOptions::field_v1()
    }
}

/// A no-prior run flying the legacy scanner, with or without mission
/// flights over the believed surface.
fn scanner_run(belief_flights: bool) -> RunOptions {
    RunOptions {
        world_knowledge: WorldKnowledge::NoPrior,
        energy_profile: argusnet_sim::run_options::EnergyProfile::FlownV1,
        planning_policy: argusnet_sim::run_options::PlanningPolicy::ContingencyV1,
        exploration: Some(ExplorationConfig {
            mission_flights: belief_flights.then(MissionFlightConfig::default),
            ..ExplorationConfig::default()
        }),
        ..RunOptions::default()
    }
}

fn phase(frame: &serde_json::Value) -> &str {
    frame["scan_mission_state"]["phase"].as_str().unwrap()
}

fn contacts(frames: &[serde_json::Value]) -> usize {
    frames
        .iter()
        .map(|frame| frame["world_contacts"].as_array().map_or(0, Vec::len))
        .sum()
}

fn position(frame: &serde_json::Value, node_id: &str) -> [f64; 3] {
    let node = frame["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|node| node["node_id"] == node_id)
        .expect("node in frame");
    let at = |k: usize| node["position"][k].as_f64().unwrap();
    [at(0), at(1), at(2)]
}

fn mobile(frame: &serde_json::Value) -> Vec<String> {
    frame["nodes"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|node| node["is_mobile"] == true)
        .map(|node| node["node_id"].as_str().unwrap().to_string())
        .collect()
}

#[test]
fn the_profile_names_itself_and_everything_it_composes() {
    assert_eq!(RunProfile::parse("field_v1"), Ok(RunProfile::FieldV1));
    assert!(RunProfile::parse("field_v2").is_err());
    let options = RunProfile::FieldV1.options();
    assert!(!options.is_legacy());
    let bundle = bundle();
    let (meta, frames) = replay(run_with(&bundle, &options));
    assert_eq!(meta["run_profile"]["id"], "field_v1");
    assert_eq!(meta["energy_profile"], "flown_v1");
    assert_eq!(meta["planning_policy"], "contingency_v1");
    let knowledge = &meta["world_knowledge"];
    assert_eq!(knowledge["knowledge"], "none");
    let exploration = &knowledge["exploration"];
    for part in [
        "rig",
        "measured_wires",
        "wire_tracker",
        "poles",
        "wire_planning",
        "mission_flights",
        "quality",
    ] {
        assert!(
            exploration.get(part).is_some_and(|v| !v.is_null()),
            "the profile records its {part}"
        );
    }
    assert_eq!(
        exploration["quality"],
        serde_json::to_value(FIELD_V1_QUALITY).unwrap()
    );
    assert!(
        exploration["rig"]["sensors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|sensor| sensor["lidar"].get("returns").is_some_and(|r| !r.is_null())),
        "the rig's LiDAR models its returns: {}",
        exploration["rig"]["sensors"]
    );
    // The scan ends on the profile's share, not the scenario's.
    assert_eq!(
        frames[0]["scan_mission_state"]["scan_coverage_threshold"],
        FIELD_V1_SCAN_THRESHOLD
    );
    assert_ne!(
        bundle.mission.as_ref().unwrap().scan_coverage_threshold,
        FIELD_V1_SCAN_THRESHOLD
    );
    // Changed options are no longer the profile, and do not say they are.
    let (meta, _) = replay(run_with(&bundle, &whole_mission()));
    assert!(meta.get("run_profile").is_none());
}

#[test]
fn the_composed_run_is_deterministic_and_resumes_exactly() {
    let bundle = bundle();
    let options = whole_mission();
    let (_, straight) = replay(run_with(&bundle, &options));
    let (_, again) = replay(run_with(&bundle, &options));
    assert_eq!(straight, again);

    // Checkpoints taken while scanning, on a mission flight and at home.
    let first = |name: &str| {
        straight
            .iter()
            .position(|frame| phase(frame) == name)
            .unwrap_or_else(|| panic!("the mission reaches {name}"))
    };
    for at in [15, first("inspecting") + 3, first("egress") + 3] {
        let mut original = run_with(&bundle, &options);
        for _ in 0..at {
            original.step().unwrap();
        }
        let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
        let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
        let mut resumed = run_with(&bundle, &options);
        resumed.restore(&checkpoint).expect("restore accepted");
        let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
            .map(|_| {
                resumed.step().unwrap();
                serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
            })
            .collect();
        assert_eq!(rest, straight[at..], "resumed at tick {at}");
    }

    // A reset run is a fresh one.
    let mut run = run_with(&bundle, &options);
    for _ in 0..40 {
        run.step().unwrap();
    }
    run.reset();
    assert_eq!(replay(run).1, straight);
}

#[test]
fn the_whole_mission_touches_nothing_and_ends_at_home() {
    let bundle = bundle();
    let (_, frames) = replay(run_with(&bundle, &whole_mission()));
    let last = frames.last().unwrap();
    assert_eq!(phase(last), "complete", "the mission completes");
    assert_eq!(contacts(&frames), 0, "physics never had to stop a platform");

    // Every drone is at its station when the mission completes, and still
    // there at the end: it does not go back to its lanes.
    let complete = frames
        .iter()
        .position(|frame| phase(frame) == "complete")
        .unwrap();
    assert!(
        complete + 20 < frames.len(),
        "the run goes on after completion"
    );
    let radius_m = bundle.mission.as_ref().unwrap().egress_arrival_radius_m;
    for drone in mobile(last) {
        let home = position(&frames[0], &drone);
        for frame in &frames[complete..] {
            let at = position(frame, &drone);
            let away = (at[0] - home[0]).hypot(at[1] - home[1]);
            assert!(
                away <= radius_m,
                "{drone} is {away:.1} m from its station after completion"
            );
        }
    }
    let pois = last["scan_mission_state"]["poi_statuses"]
        .as_array()
        .unwrap();
    assert!(!pois.is_empty());
    assert!(
        pois.iter().all(|poi| poi["status"] == "complete"),
        "{pois:?}"
    );
}

/// A mission flight is flown continuously: it never moves further in a
/// tick than its speed allows, where a flight physics held used to be put
/// back on its plan hundreds of metres on.
#[test]
fn mission_flights_never_jump() {
    let bundle = bundle();
    let (_, frames) = replay(run_with(&bundle, &whole_mission()));
    let first = frames
        .iter()
        .position(|frame| phase(frame) == "inspecting")
        .expect("the mission reaches its POIs");
    let speed_mps = bundle.mission.as_ref().unwrap().redirect_speed_mps;
    // A tick's horizontal flight at the redirect speed, and the governor's
    // climb on top; a little slack for the tick a flight is handed over.
    let reach_m = (speed_mps * bundle.dt_s).hypot(15.0 * bundle.dt_s) * 1.5;
    for pair in frames[first + 1..].windows(2) {
        for drone in mobile(&pair[0]) {
            let (a, b) = (position(&pair[0], &drone), position(&pair[1], &drone));
            let moved =
                ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt();
            assert!(
                moved <= reach_m,
                "{drone} moved {moved:.1} m in one tick at t = {}",
                pair[1]["timestamp_s"]
            );
        }
    }
}

/// With the legacy scanner alone: flying mission flights over the believed
/// surface leaves physics less to stop than flying them straight.
#[test]
fn belief_flights_meet_less_than_straight_ones() {
    let bundle = bundle();
    let (_, straight) = replay(run_with(&bundle, &scanner_run(false)));
    let (_, governed) = replay(run_with(&bundle, &scanner_run(true)));
    assert!(
        contacts(&straight) > 0,
        "the fixture's straight flights do meet the world"
    );
    assert!(
        contacts(&governed) < contacts(&straight),
        "{} contacts governed, {} straight",
        contacts(&governed),
        contacts(&straight)
    );
    assert_eq!(phase(governed.last().unwrap()), "complete");
}

#[test]
fn a_run_without_the_profile_is_the_run_it_was() {
    let bundle = bundle();
    let plain = scanner_run(false);
    let (meta, frames) = replay(run_with(&bundle, &plain));
    assert!(meta.get("run_profile").is_none());
    assert!(meta["world_knowledge"]["exploration"]
        .get("mission_flights")
        .is_none());
    // The scenario's own scan share stands.
    assert_eq!(
        frames[0]["scan_mission_state"]["scan_coverage_threshold"],
        bundle.mission.as_ref().unwrap().scan_coverage_threshold
    );
}

#[test]
fn a_bad_threshold_or_a_run_with_no_scan_is_refused() {
    let bundle = bundle();
    for bad in [0.0, -0.1, 1.5, f64::NAN] {
        let mut run = assemble(&bundle).unwrap();
        let options = RunOptions {
            scan_coverage_threshold: Some(bad),
            ..RunOptions::default()
        };
        assert!(options.apply(&mut run, &bundle).is_err(), "{bad}");
    }
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_tt.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let tracking = ScenarioBundle::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
    let mut run = assemble(&tracking).unwrap();
    let options = RunOptions {
        scan_coverage_threshold: Some(0.9),
        ..RunOptions::default()
    };
    let refused = options.apply(&mut run, &tracking).unwrap_err();
    assert!(refused.contains("no scan mission"), "{refused}");
}

/// Truth isolation for the mission's inputs: a POI that declares its hover
/// height is flown, allocated and reported at that height over what the
/// platforms know. The absolute height its scenario took from the true
/// ground can be anything and nothing changes.
#[test]
fn a_declared_poi_height_is_all_that_is_read_of_a_pois_height() {
    let declared = |lift_m: f64| {
        let mut bundle = bundle();
        for poi in &mut bundle.mission.as_mut().unwrap().pois {
            poi.hover_agl_m = Some(40.0);
            poi.position[2] += lift_m;
        }
        bundle
    };
    let options = whole_mission();
    let (meta, frames) = replay(run_with(&declared(0.0), &options));
    assert_eq!(
        meta["world_knowledge"]["mission_setup"]["poi_hover_agl"],
        serde_json::json!({ "declared": 3, "recovered_from_true_ground": 0 })
    );
    assert_eq!(phase(frames.last().unwrap()), "complete");
    for lift_m in [-35.0, 250.0] {
        let (_, lifted) = replay(run_with(&declared(lift_m), &options));
        assert_eq!(lifted, frames, "POI heights off by {lift_m} m");
    }

    // A bundle that declares none says where its heights came from.
    let (meta, plain) = replay(run_with(&bundle(), &options));
    assert_eq!(
        meta["world_knowledge"]["mission_setup"]["poi_hover_agl"],
        serde_json::json!({ "declared": 0, "recovered_from_true_ground": 3 })
    );
    assert_eq!(phase(plain.last().unwrap()), "complete");
}

/// A blind setup plans nothing on the true world: the search lanes and the
/// POIs are the same whatever it holds. The setup a prior run gets places
/// its POIs on the true ground, and differs with it.
#[test]
fn a_blind_setup_is_the_same_whatever_the_world_holds() {
    use argusnet_sim::builder::{build_bundle, ScenarioConfig};
    let build = |terrain: &str, blind: bool| -> serde_json::Value {
        let config: ScenarioConfig = serde_json::from_value(serde_json::json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "seed": 11,
            "duration_s": 5.0,
            "options": {
                "mission_mode": "scan_map_inspect",
                "terrain_preset": terrain,
                "blind_setup": blind,
            },
        }))
        .expect("config");
        build_bundle(&config).expect("bundle builds")
    };
    // What the mission is given before it flies: each drone's lanes, and
    // the POIs' places and declared heights.
    let setup = |bundle: &serde_json::Value| {
        let lanes: Vec<serde_json::Value> = bundle["nodes"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|node| {
                node.pointer("/controller/operational/search/waypoints_xy_m")
                    .cloned()
            })
            .collect();
        let pois: Vec<serde_json::Value> = bundle["mission"]["pois"]
            .as_array()
            .unwrap()
            .iter()
            .map(|poi| {
                serde_json::json!([poi["position"][0], poi["position"][1], poi["hover_agl_m"]])
            })
            .collect();
        (lanes, pois)
    };
    let (flat, alpine) = (build("urban_flat", true), build("alpine", true));
    let (lanes, pois) = setup(&flat);
    assert!(!lanes.is_empty() && !pois.is_empty());
    assert!(pois.iter().all(|poi| poi[2].is_number()), "{pois:?}");
    assert_eq!(setup(&alpine), (lanes, pois));
    assert_eq!(
        flat["presentation"]["scenario_options"]["blind_setup"],
        true
    );

    // The control: planned on the world, the POIs are placed by its ground
    // and differ with it, and none declares a height.
    let (flat, alpine) = (build("urban_flat", false), build("alpine", false));
    assert_ne!(
        flat["mission"]["pois"], alpine["mission"]["pois"],
        "POIs placed on the true ground"
    );
    assert!(flat["mission"]["pois"][0].get("hover_agl_m").is_none());
    assert!(flat["presentation"]["scenario_options"]
        .get("blind_setup")
        .is_none());
}

/// A blind setup sites every station, and so every pad a drone climbs
/// from, outside each power line's clearance with room to spare. (West
/// Pleasant View is the case that needs it: its layout puts a station 12 m
/// from the line, and the qualification's wire-clearance gate measures it.)
#[test]
fn a_blind_setup_sites_stations_clear_of_power_lines() {
    use argusnet_sim::builder::{build_bundle, ScenarioConfig};
    let build = |blind: bool| -> serde_json::Value {
        let config: ScenarioConfig = serde_json::from_value(serde_json::json!({
            "format": "argusnet.scenario-config",
            "version": 1,
            "seed": 11,
            "duration_s": 5.0,
            "options": {
                "mission_mode": "scan_map_inspect",
                "terrain_preset": "urban_flat",
                "map_preset": "medium",
                "power_lines": true,
                "ground_station_count": 12,
                "blind_setup": blind,
            },
        }))
        .expect("config");
        build_bundle(&config).expect("bundle builds")
    };
    // How near the nearest station stands to the line's centre.
    let nearest = |bundle: &serde_json::Value| -> f64 {
        let lines = bundle["world"]["power_lines"].as_array().expect("a line");
        let mut nearest = f64::INFINITY;
        for node in bundle["nodes"].as_array().unwrap() {
            let Some(at) = node.pointer("/controller/position_m") else {
                continue;
            };
            let (x, y) = (at[0].as_f64().unwrap(), at[1].as_f64().unwrap());
            for line in lines {
                let supports = line["supports"].as_array().unwrap();
                for pair in supports.windows(2) {
                    let end = |s: &serde_json::Value, k: usize| s["xy_m"][k].as_f64().unwrap();
                    let (ax, ay) = (end(&pair[0], 0), end(&pair[0], 1));
                    let (dx, dy) = (end(&pair[1], 0) - ax, end(&pair[1], 1) - ay);
                    let t = (((x - ax) * dx + (y - ay) * dy) / (dx * dx + dy * dy)).clamp(0.0, 1.0);
                    nearest = nearest.min((x - ax - dx * t).hypot(y - ay - dy * t));
                }
            }
        }
        nearest
    };
    let line = &build(true)["world"]["power_lines"][0];
    let widest = line["conductor_offsets_m"]
        .as_array()
        .unwrap()
        .iter()
        .map(|o| o.as_f64().unwrap().abs())
        .fold(0.0, f64::max);
    let keep = widest + line["clearance_m"].as_f64().unwrap() + 15.0;
    assert!(
        nearest(&build(true)) >= keep - 1e-6,
        "{} m from the line, {keep} m kept",
        nearest(&build(true))
    );
}

// --- the profile under the scheduler ---

use argusnet_sim::compute::ComputePool;
use argusnet_sim::scenario::{assemble_with, RunError};
use argusnet_sim::scheduler::{parse_rate, ExecutorError, LaneRates, PhasedLaneExecutor};
use argusnet_sim::simulation::TickError;

/// The whole mission driven by the phased-lane executor at `rates`, on
/// `workers` workers; with `resume_at`, checkpointed there and finished by
/// a freshly assembled run, as a recovered process would.
fn scheduled(rates: LaneRates, workers: usize, resume_at: Option<u64>) -> Vec<serde_json::Value> {
    let bundle = bundle();
    let options = RunOptions {
        lane_rates: rates,
        ..whole_mission()
    };
    let assembled = || {
        let pool = std::sync::Arc::new(ComputePool::new(workers).expect("pool"));
        let mut run = assemble_with(&bundle, pool).expect("assembles");
        options.apply(&mut run, &bundle).expect("options apply");
        run
    };
    let executor = |tick: u64| {
        rates
            .apply(PhasedLaneExecutor::new::<RunError>(bundle.dt_s, tick).expect("exact timestep"))
            .expect("rates")
    };
    let mut run = assembled();
    let mut lanes = executor(0);
    let mut frames = Vec::new();
    loop {
        if resume_at == Some(run.sim.tick()) {
            let text = serde_json::to_string(&run.sim.checkpoint().unwrap()).unwrap();
            let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
            run = assembled();
            run.restore(&checkpoint).expect("restore accepted");
            lanes = executor(run.sim.tick());
        }
        match lanes.step(&mut run) {
            Ok(_) => frames.push(serde_json::to_value(run.last_frame().unwrap()).unwrap()),
            Err(ExecutorError::Tick(RunError::Tick(TickError::Completed { .. }))) => break,
            Err(error) => panic!("{error:?}"),
        }
    }
    frames
}

fn rates(mission: Option<&str>, mapping: Option<&str>) -> LaneRates {
    LaneRates {
        mission: mission.map(|rate| parse_rate(rate).unwrap()),
        mapping: mapping.map(|rate| parse_rate(rate).unwrap()),
        fusion: None,
    }
}

/// The phased lanes and the worker count change how the composed run is
/// executed, not what it does.
#[test]
fn the_composed_run_is_the_same_under_the_lane_executor_and_any_worker_count() {
    let (_, straight) = replay(run_with(&bundle(), &whole_mission()));
    assert_eq!(scheduled(LaneRates::default(), 1, None), straight);
    assert_eq!(scheduled(LaneRates::default(), 4, None), straight);
}

/// With the mission and mapping lanes at their own rates (`scheduled_v1`)
/// the composed mission still completes, touching nothing, and a run
/// checkpointed while scanning, on a mission flight or on the way home
/// finishes exactly as the uninterrupted one: the map evidence a lane is
/// holding, the flights under way and their delays all come back.
#[test]
fn the_composed_run_at_its_own_lane_rates_completes_and_resumes_exactly() {
    // The fixture ticks once a second: the mission decides every 2 s, the
    // map integrates every 3 s.
    let rated = rates(Some("1/2"), Some("1/3"));
    let straight = scheduled(rated, 1, None);
    assert_eq!(phase(straight.last().unwrap()), "complete");
    assert_eq!(contacts(&straight), 0);
    let first = |name: &str| {
        straight
            .iter()
            .position(|frame| phase(frame) == name)
            .unwrap_or_else(|| panic!("the mission reaches {name}")) as u64
    };
    for at in [16, first("inspecting") + 4, first("egress") + 2] {
        assert_eq!(scheduled(rated, 1, Some(at)), straight, "resumed at {at}");
    }
    assert_eq!(scheduled(rated, 4, None), straight, "on four workers");
}

// --- a fault the composed mission has to live with ---

use argusnet_sim::faults::{Fault, FaultManifest, MANIFEST_FORMAT, MANIFEST_VERSION};

fn mapping_outage(node_id: &str, start_s: f64, end_s: f64) -> FaultManifest {
    FaultManifest {
        format: MANIFEST_FORMAT.into(),
        version: MANIFEST_VERSION,
        faults: vec![Fault::MappingOutage {
            node_id: node_id.into(),
            start_s,
            end_s,
        }],
    }
}

/// F2 against the composed mission: one drone's mapping LiDAR returns
/// nothing for the first 80 s.
///
/// - Detection: the fault is in every frame of its interval, and the map
///   holds less at its end than the unfaulted run's.
/// - Containment: nothing is flown into and nothing jumps; the faulted
///   drone's navigation camera still sees for it.
/// - Terminal state: the mission completes, later, with every POI done and
///   every drone home. The scan is not ended early on a map it did not
///   make.
/// - A run checkpointed inside the outage finishes exactly as the
///   uninterrupted one.
#[test]
fn a_mapping_outage_slows_the_mission_and_nothing_else() {
    let bundle = bundle();
    let clean = whole_mission();
    let faulted = RunOptions {
        faults: Some(mapping_outage("drone-east", 0.0, 80.0)),
        ..whole_mission()
    };
    let (_, clean_frames) = replay(run_with(&bundle, &clean));
    let (meta, frames) = replay(run_with(&bundle, &faulted));
    assert_eq!(
        meta["fault_manifest"]["faults"][0]["kind"],
        "mapping_outage"
    );

    // Detection.
    let active = |frame: &serde_json::Value| {
        frame["active_faults"]
            .as_array()
            .is_some_and(|faults| faults.iter().any(|f| f["kind"] == "mapping_outage"))
    };
    assert!(frames[..80].iter().all(active));
    assert!(!frames[80..].iter().any(active));
    let known = |frame: &serde_json::Value| {
        frame["mapping_state"]["belief"]["known_fraction"]
            .as_f64()
            .unwrap()
    };
    assert!(
        known(&frames[79]) < 0.8 * known(&clean_frames[79]),
        "{} against {} known at the outage's end",
        known(&frames[79]),
        known(&clean_frames[79])
    );

    // Containment.
    assert_eq!(contacts(&frames), 0);

    // Terminal state.
    let complete_at = |frames: &[serde_json::Value]| {
        frames
            .iter()
            .position(|frame| phase(frame) == "complete")
            .expect("the mission completes")
    };
    assert!(complete_at(&frames) > complete_at(&clean_frames));
    let last = frames.last().unwrap();
    let pois = last["scan_mission_state"]["poi_statuses"]
        .as_array()
        .unwrap();
    assert!(pois.iter().all(|poi| poi["status"] == "complete"));
    let radius_m = bundle.mission.as_ref().unwrap().egress_arrival_radius_m;
    for drone in mobile(last) {
        let (home, at) = (position(&frames[0], &drone), position(last, &drone));
        assert!((at[0] - home[0]).hypot(at[1] - home[1]) <= radius_m);
    }

    // Resumed inside the outage.
    let at = 40usize;
    let mut original = run_with(&bundle, &faulted);
    for _ in 0..at {
        original.step().unwrap();
    }
    let text = serde_json::to_string(&original.sim.checkpoint().unwrap()).unwrap();
    let checkpoint: SimulationCheckpoint = serde_json::from_str(&text).unwrap();
    let mut resumed = run_with(&bundle, &faulted);
    resumed.restore(&checkpoint).expect("restore accepted");
    let rest: Vec<serde_json::Value> = (at..bundle.steps as usize)
        .map(|_| {
            resumed.step().unwrap();
            serde_json::to_value(resumed.last_frame().unwrap()).unwrap()
        })
        .collect();
    assert_eq!(rest, frames[at..]);
}

/// A run with a prior map flies no mapping payload: the fault is refused
/// before the run starts, as is one that names no platform.
#[test]
fn a_mapping_outage_is_refused_where_nothing_maps() {
    let bundle = bundle();
    let mut run = assemble(&bundle).unwrap();
    let prior = RunOptions {
        faults: Some(mapping_outage("drone-east", 0.0, 80.0)),
        ..RunOptions::default()
    };
    let refused = prior.apply(&mut run, &bundle).unwrap_err();
    assert!(refused.contains("mapping_outage"), "{refused}");

    let mut run = assemble(&bundle).unwrap();
    let unknown = RunOptions {
        faults: Some(mapping_outage("drone-nowhere", 0.0, 80.0)),
        ..whole_mission()
    };
    assert!(unknown.apply(&mut run, &bundle).is_err());
}
