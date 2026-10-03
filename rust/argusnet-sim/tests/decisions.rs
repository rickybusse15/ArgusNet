//! F3: the mission's decisions are typed, and they trace to what was flown.
//!
//! The scan mission is run tick by tick. Every decision stage 9 records is
//! checked against what actually happened: a queued POI assignment is flown by
//! stage 2 on the next tick; a released redirect stops being flown; phase
//! changes are the mission's real transitions, each with the inputs that
//! decided it.

use argusnet_sim::mission_driver::{DecisionStatus, MissionDriverStage};
use argusnet_sim::overrides::{MissionOverrideStage, OverrideSource};
use argusnet_sim::scenario::{assemble, ScenarioBundle};
use argusnet_sim::stage::Stage;

fn bundle(case: &str) -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_{case}.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

#[test]
fn every_mission_decision_traces_to_what_was_flown() {
    let mut run = assemble(&bundle("scan")).expect("assembles");
    let mut decisions = Vec::new();
    let mut phases = vec![run.mission().expect("scan").phase().to_string()];
    let mut expect_flown: Vec<(u64, String)> = Vec::new();
    let mut expect_released: Vec<(u64, String)> = Vec::new();
    while run.step().is_ok() {
        let tick = run.sim.tick() - 1;
        let overrides = run
            .sim
            .stage::<MissionOverrideStage>(Stage::ApplyMissionOverrides)
            .expect("scan runs stage 2");
        // What the previous tick queued is flown by this tick's stage 2.
        for (due, drone) in expect_flown.drain(..) {
            assert_eq!(due, tick);
            assert_eq!(
                overrides.applied_source(&drone),
                Some(OverrideSource::Poi),
                "tick {tick}: {drone}'s queued POI assignment was not flown"
            );
        }
        for (due, drone) in expect_released.drain(..) {
            assert_eq!(due, tick);
            assert_ne!(
                overrides.applied_source(&drone),
                Some(OverrideSource::Poi),
                "tick {tick}: {drone}'s released redirect is still flown"
            );
        }
        let phase = run.mission().expect("scan").phase().to_string();
        if phases.last() != Some(&phase) {
            phases.push(phase.clone());
        }
        let driver = run.driver().expect("scan has a driver");
        for decision in MissionDriverStage::decisions(driver) {
            match decision.kind {
                "poi_assignment" => {
                    assert_eq!(decision.status, DecisionStatus::QueuedForNextTick);
                    assert_eq!(decision.command, Some("set_poi"));
                    let poi = decision.selected.clone().expect("a POI");
                    assert!(
                        decision.candidates.contains(&poi),
                        "chosen from the pending"
                    );
                    expect_flown.push((tick + 1, decision.subject.clone().expect("a drone")));
                }
                "poi_complete" => {
                    assert_eq!(decision.command, Some("clear_poi"));
                    expect_released.push((tick + 1, decision.subject.clone().expect("a drone")));
                }
                "phase_change" => {
                    assert_eq!(decision.selected.as_deref(), Some(phase.as_str()));
                    assert!(decision.inputs.get("coverage_fraction").is_some());
                }
                _ => {}
            }
            decisions.push((tick, decision.clone()));
        }
    }

    let kinds = |kind: &str| decisions.iter().filter(|(_, d)| d.kind == kind).count();
    assert_eq!(kinds("coordinator_election"), 1);
    let elected = decisions
        .iter()
        .find(|(_, d)| d.kind == "coordinator_election")
        .and_then(|(_, d)| d.selected.clone());
    assert_eq!(
        elected,
        run.driver()
            .and_then(|driver| driver.report().coordinator_drone_id.clone())
    );
    assert!(kinds("poi_assignment") >= 3, "every POI was assigned");
    assert_eq!(kinds("poi_complete"), 3, "every POI completed");
    assert_eq!(kinds("poi_rescore"), 1);
    let changes: Vec<String> = decisions
        .iter()
        .filter(|(_, d)| d.kind == "phase_change")
        .map(|(_, d)| d.selected.clone().unwrap())
        .collect();
    assert_eq!(
        changes,
        phases[1..],
        "every transition, and only those, explained"
    );
    // Queued and applied decisions are told apart.
    assert!(decisions
        .iter()
        .any(|(_, d)| d.status == DecisionStatus::Applied));
}
