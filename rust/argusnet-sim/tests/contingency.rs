//! F5: `contingency_v1` plans energy contingencies continuously. A drone
//! gives a POI up when it can no longer afford the dwell left and the trip
//! home, not at a fixed battery fraction; new work needs a margin above
//! that, so work just taken is not dropped (hysteresis); and a POI no drone
//! can afford is reported unassigned and no longer holds the mission open.

use argusnet_planning::battery::EnergyProfile;
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

fn run(policy: PlanningPolicy) -> AssembledRun {
    let bundle = bundle("scan");
    let mut run = assemble(&bundle).expect("assembles");
    let options = RunOptions {
        planning_policy: policy,
        energy_profile: EnergyProfile::PhysicalV1,
        ..RunOptions::default()
    };
    options.apply(&mut run, &bundle).expect("applies");
    run
}

const DRONES: [&str; 2] = ["drone-east", "drone-north"];

/// One decision, as the journal records it.
#[derive(Debug, Clone)]
struct Made {
    tick: u64,
    kind: String,
    reason: String,
    subject: Option<String>,
    selected: Option<String>,
}

/// What a run did: its decisions in order, its last scan-mission state and
/// the lowest charge any drone held.
struct Outcome {
    decisions: Vec<Made>,
    phase: String,
    poi_statuses: Vec<serde_json::Value>,
    lowest_wh: f64,
    meta: serde_json::Value,
}

impl Outcome {
    fn with(&self, kind: &str) -> Vec<&Made> {
        self.decisions
            .iter()
            .filter(|made| made.kind == kind)
            .collect()
    }

    /// Decisions in which a drone gave work up.
    fn give_ups(&self) -> Vec<&Made> {
        self.decisions
            .iter()
            .filter(|made| made.kind == "poi_handoff" || made.kind == "poi_release")
            .collect()
    }

    fn statuses(&self) -> Vec<(String, String, Option<String>)> {
        self.poi_statuses
            .iter()
            .map(|status| {
                (
                    status["poi_id"].as_str().unwrap().to_string(),
                    status["status"].as_str().unwrap().to_string(),
                    status["unassigned_reason"].as_str().map(str::to_string),
                )
            })
            .collect()
    }
}

/// Step `run` to its end. `seed` is called on every tick from the first the
/// mission is inspecting on, with the number of such ticks before it.
fn finish(mut run: AssembledRun, mut seed: impl FnMut(u64, &mut AssembledRun)) -> Outcome {
    let mut decisions = Vec::new();
    let mut inspecting: Option<u64> = None;
    let mut tick = 0;
    let mut lowest_wh = f64::INFINITY;
    loop {
        match run.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
        let body = run.journal_body();
        for decision in body["decisions"].as_array().into_iter().flatten() {
            let text = |key: &str| decision[key].as_str().map(str::to_string);
            decisions.push(Made {
                tick,
                kind: text("kind").unwrap(),
                reason: text("reason").unwrap(),
                subject: text("subject"),
                selected: text("selected"),
            });
        }
        for drone in DRONES {
            lowest_wh = lowest_wh.min(run.battery().remaining_wh(drone).unwrap());
        }
        if inspecting.is_none() && body["scan_mission"]["phase"] == "inspecting" {
            inspecting = Some(0);
        }
        if let Some(count) = inspecting.as_mut() {
            seed(*count, &mut run);
            *count += 1;
        }
        tick += 1;
    }
    assert!(inspecting.is_some(), "the mission reaches its POIs");
    let replay = serde_json::to_value(run.run_to_replay().expect("replay")).expect("serialises");
    let state = replay["frames"]
        .as_array()
        .unwrap()
        .iter()
        .rev()
        .find_map(|frame| frame.get("scan_mission_state").cloned())
        .expect("a scan mission");
    Outcome {
        decisions,
        phase: state["phase"].as_str().unwrap().to_string(),
        poi_statuses: state["poi_statuses"].as_array().unwrap().clone(),
        lowest_wh,
        meta: replay["meta"].clone(),
    }
}

fn set_all(run: &mut AssembledRun, wh: f64) {
    for drone in DRONES {
        assert!(run.battery_mut().set_remaining_wh(drone, wh));
    }
}

/// Both drones start inspecting with `wh` (of 500 Wh).
fn starting_with(policy: PlanningPolicy, wh: f64) -> Outcome {
    finish(run(policy), |count, run| {
        if count == 0 {
            set_all(run, wh)
        }
    })
}

/// Hysteresis. At 152 Wh (30.4 %) both drones can afford their POIs and the
/// trip home. `efficient_v1` gives them to them, then takes them back as
/// they pass its fixed 30 % threshold a few seconds later, and inspects
/// nothing. `contingency_v1` judges by what is left to do: it never gives up
/// work it took, and inspects every POI.
#[test]
fn work_just_taken_is_not_dropped() {
    let efficient = starting_with(PlanningPolicy::EfficientV1, 152.0);
    let taken = efficient.with("poi_assignment");
    assert!(!taken.is_empty());
    for given_up in efficient.give_ups() {
        let took = taken
            .iter()
            .find(|made| made.subject == given_up.subject)
            .expect("it gave up what it took");
        assert!(
            given_up.tick - took.tick <= 15,
            "{given_up:?} after {took:?}"
        );
    }
    assert!(
        !efficient.give_ups().is_empty(),
        "efficient_v1 thrashes here"
    );
    assert_ne!(efficient.phase, "complete");

    let contingency = starting_with(PlanningPolicy::ContingencyV1, 152.0);
    assert!(
        contingency.give_ups().is_empty(),
        "{:?}",
        contingency.give_ups()
    );
    assert_eq!(contingency.phase, "complete");
    assert!(contingency
        .statuses()
        .iter()
        .all(|(_, status, reason)| status == "complete" && reason.is_none()));
    assert!(contingency.lowest_wh > 0.0);
}

/// At 117.5 Wh no drone can afford any POI with the margin new work needs
/// above its reserve. `contingency_v1` says so, once per POI, reports each
/// unassigned in the replay, and the mission goes home instead of waiting
/// for work no one can do. `efficient_v1` leaves them pending, and the
/// mission is still inspecting when the run ends.
#[test]
fn a_poi_no_drone_can_afford_is_reported_unassigned_and_the_mission_ends() {
    let contingency = starting_with(PlanningPolicy::ContingencyV1, 117.5);
    let unassigned = contingency.with("poi_unassigned");
    assert_eq!(unassigned.len(), 3, "{unassigned:?}");
    assert!(unassigned
        .iter()
        .all(|made| made.reason == "no_affordable_drone"));
    let mut named: Vec<_> = unassigned
        .iter()
        .map(|made| made.selected.clone().unwrap())
        .collect();
    named.sort();
    assert_eq!(named, ["poi-00", "poi-01", "poi-02"]);
    assert!(contingency.with("poi_assignment").is_empty());
    assert_eq!(contingency.phase, "complete");
    for (poi, status, reason) in contingency.statuses() {
        assert_eq!(status, "pending", "{poi}");
        assert_eq!(reason.as_deref(), Some("no_affordable_drone"), "{poi}");
    }
    assert!(contingency.lowest_wh > 0.0);
    assert_eq!(contingency.meta["planning_policy"], "contingency_v1");

    let efficient = starting_with(PlanningPolicy::EfficientV1, 117.5);
    assert_eq!(efficient.phase, "inspecting");
    assert!(efficient
        .statuses()
        .iter()
        .all(|(_, _, reason)| reason.is_none()));
}

/// A drone that can no longer afford the dwell left and the trip home hands
/// its POI on at once, before it reaches its reserve and turns home; the POI
/// goes to the other drone, which finishes it; and the mission's recall
/// ending does not cancel the reserve return.
///
/// Its charge is not checked to the end of the run: stage 1 accounts energy
/// on the nominal controller positions before overrides (the preserved
/// causal order, ARCHITECTURE_AND_PROTOCOLS stage 1), so a drone held at
/// home is still charged for the lanes it would have flown.
#[test]
fn a_holder_that_can_no_longer_finish_hands_its_poi_on() {
    let mut cut = None;
    let outcome = finish(run(PlanningPolicy::ContingencyV1), |count, run| {
        if count == 0 {
            set_all(run, 160.0);
        }
        // drone-east takes poi-02 at the 21st inspecting tick; drop it to
        // just above its 100 Wh reserve floor on the way there.
        if count == 24 && cut.is_none() {
            assert!(run.battery_mut().set_remaining_wh("drone-east", 100.5));
            cut = Some(count);
        }
    });
    let given_up: Vec<_> = outcome
        .give_ups()
        .into_iter()
        .filter(|made| made.subject.as_deref() == Some("drone-east"))
        .collect();
    assert_eq!(given_up.len(), 1, "{:?}", outcome.decisions);
    let given_up = given_up[0];
    assert!(
        given_up.reason == "cannot_finish_handoff" || given_up.reason == "no_affordable_receiver",
        "{given_up:?}"
    );
    let home = outcome
        .with("return_home")
        .into_iter()
        .find(|made| made.subject.as_deref() == Some("drone-east"))
        .map(|made| made.tick);
    let home = home.expect("it turns home on its reserve");
    assert!(home > given_up.tick, "{home} {given_up:?}");
    let kept = outcome
        .with("return_home")
        .into_iter()
        .find(|made| made.reason == "reserve_return_kept")
        .expect("the recall's end keeps its return");
    assert_eq!(kept.subject.as_deref(), Some("drone-east"));
    assert!(outcome
        .statuses()
        .iter()
        .all(|(_, status, _)| status == "complete"));
}

/// The unassigned POIs are checkpointed: a run resumed mid-mission finishes
/// as the straight run does, and a run under another policy refuses the
/// checkpoint.
#[test]
fn a_contingency_checkpoint_resumes_exactly_and_only_as_contingency() {
    let straight = |stop: Option<u64>| {
        let mut run = run(PlanningPolicy::ContingencyV1);
        let mut inspecting = 0;
        let mut tick = 0;
        loop {
            if Some(tick) == stop {
                return (None, Some(run.sim.checkpoint().expect("not faulted")));
            }
            match run.step() {
                Ok(_) => {}
                Err(RunError::Tick(TickError::Completed { .. })) => break,
                Err(error) => panic!("{error:?}"),
            }
            if run.journal_body()["scan_mission"]["phase"] == "inspecting" {
                if inspecting == 0 {
                    set_all(&mut run, 117.5);
                }
                inspecting += 1;
            }
            tick += 1;
        }
        let replay = serde_json::to_value(run.run_to_replay().expect("replay")).unwrap();
        (Some(replay), None)
    };
    let (Some(whole), None) = straight(None) else {
        unreachable!()
    };
    let (None, Some(checkpoint)) = straight(Some(60)) else {
        unreachable!()
    };
    let mut resumed = run(PlanningPolicy::ContingencyV1);
    resumed.restore(&checkpoint).expect("restores");
    loop {
        match resumed.step() {
            Ok(_) => {}
            Err(RunError::Tick(TickError::Completed { .. })) => break,
            Err(error) => panic!("{error:?}"),
        }
    }
    let resumed = serde_json::to_value(resumed.run_to_replay().expect("replay")).unwrap();
    let whole = whole["frames"].as_array().unwrap();
    let resumed = resumed["frames"].as_array().unwrap();
    assert!(!resumed.is_empty() && resumed.len() < whole.len());
    assert!(
        whole[whole.len() - resumed.len()..] == resumed[..],
        "the resumed run's frames are the straight run's"
    );
    let refused = run(PlanningPolicy::EfficientV1)
        .restore(&checkpoint)
        .unwrap_err();
    assert!(format!("{refused}").contains("contingency_v1"), "{refused}");
}
