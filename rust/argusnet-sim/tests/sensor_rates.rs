//! P4.10b: sensors capturing at their own rates in an assembled run.
//!
//! A sensor at its own rate captures at the first tick at or after each of
//! its deadlines and takes nothing in between. Every other sensor's
//! observations are exactly the every-tick run's: each (sensor, target, tick)
//! draws from its own stream, and each sensor's bias walk is its own.

use argusnet_sim::run_options::RunOptions;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::scheduler::parse_rate;
use argusnet_sim::snapshot::SimulationCheckpoint;

const RATED: &str = "drone-north";

fn bundle() -> ScenarioBundle {
    let path = format!(
        "{}/../../tests/fixtures/scenario_bundle_tt.json",
        env!("CARGO_MANIFEST_DIR")
    );
    ScenarioBundle::from_json(&std::fs::read_to_string(&path).expect("fixture"))
        .expect("bundle parses")
}

/// `RATED` at 2/3 Hz: every third 0.5 s tick.
fn rated_options() -> RunOptions {
    RunOptions {
        sensor_rates: [(RATED.to_string(), parse_rate("2/3").unwrap())]
            .into_iter()
            .collect(),
        ..RunOptions::default()
    }
}

fn rated_run(bundle: &ScenarioBundle) -> AssembledRun {
    let mut run = assemble(bundle).unwrap();
    rated_options().apply(&mut run, bundle).unwrap();
    run
}

/// Every tick's generated observations (stage 6, before fusion accepts or
/// rejects them), serialised, as `(node, observation)` pairs.
fn generated(mut run: AssembledRun) -> Vec<Vec<(String, String)>> {
    let mut ticks = Vec::new();
    loop {
        match run.step() {
            Ok(_) => ticks.push(
                run.observe()
                    .observations()
                    .iter()
                    .map(|o| (o.node_id.clone(), format!("{o:?}")))
                    .collect(),
            ),
            Err(argusnet_sim::scenario::RunError::Tick(
                argusnet_sim::simulation::TickError::Completed { .. },
            )) => return ticks,
            Err(error) => panic!("{error:?}"),
        }
    }
}

fn only(ticks: &[Vec<(String, String)>], keep: impl Fn(&str) -> bool) -> Vec<Vec<&str>> {
    ticks
        .iter()
        .map(|tick| {
            tick.iter()
                .filter(|(node, _)| keep(node))
                .map(|(_, o)| o.as_str())
                .collect()
        })
        .collect()
}

#[test]
fn a_rated_sensor_captures_only_when_due_and_the_others_are_unchanged() {
    let bundle = bundle();
    let legacy = generated(assemble(&bundle).unwrap());
    let rated = generated(rated_run(&bundle));
    assert_eq!(legacy.len(), rated.len());

    // Every other sensor generates exactly what it did: its own streams, its
    // own bias walk, and in this scenario the same platform motion.
    assert_eq!(
        only(&rated, |id| id != RATED),
        only(&legacy, |id| id != RATED)
    );

    // The rated sensor generates only at ticks 0, 3, 6, ... and does.
    let from_rated = only(&rated, |id| id == RATED);
    for (tick, observations) in from_rated.iter().enumerate() {
        if tick % 3 != 0 {
            assert!(observations.is_empty(), "tick {tick}");
        }
    }
    let reported: usize = from_rated.iter().map(Vec::len).sum();
    let legacy_reported: usize = only(&legacy, |id| id == RATED).iter().map(Vec::len).sum();
    assert!(reported > 0, "the rated sensor captured nothing");
    assert!(reported < legacy_reported);

    // Recorded, and the tracks say how old their evidence is.
    let run = rated_run(&bundle);
    assert!(run.tracks_lag_their_frames());
    let replay = run.run_to_replay().unwrap();
    let meta = serde_json::to_value(&replay.meta).unwrap();
    assert_eq!(meta["sensor_rates"][RATED], "2/3 Hz");
    assert!(replay
        .frames
        .iter()
        .flat_map(|frame| &frame.tracks)
        .all(|track| track.evidence_time_s.is_some()));
}

/// A run resumed from a checkpoint taken between captures goes on exactly as
/// the uninterrupted one; a run at other rates refuses that checkpoint.
#[test]
fn a_checkpoint_between_captures_resumes_exactly() {
    let bundle = bundle();
    let whole = rated_run(&bundle).run_to_replay().unwrap();

    // Tick 7 is between captures (6 and 9).
    let mut first = rated_run(&bundle);
    for _ in 0..7 {
        first.step().unwrap();
    }
    let checkpoint: SimulationCheckpoint =
        serde_json::from_str(&serde_json::to_string(&first.sim.checkpoint().unwrap()).unwrap())
            .unwrap();

    let mut resumed = rated_run(&bundle);
    resumed.restore(&checkpoint).unwrap();
    let rest = resumed.run_to_replay().unwrap();
    let tail: Vec<String> = whole.frames[7..]
        .iter()
        .map(|frame| serde_json::to_string(frame).unwrap())
        .collect();
    let resumed_frames: Vec<String> = rest
        .frames
        .iter()
        .map(|frame| serde_json::to_string(frame).unwrap())
        .collect();
    assert_eq!(resumed_frames, tail);

    let mut every_tick = assemble(&bundle).unwrap();
    let error = every_tick.restore(&checkpoint).unwrap_err().to_string();
    assert!(error.contains("capture rates"), "{error}");
}

#[test]
fn a_rate_for_no_sensor_or_faster_than_the_tick_is_refused() {
    let bundle = bundle();
    for (node, rate, expected) in [
        ("nobody", "1/1", "no sensor"),
        (RATED, "4/1", "faster than the tick"),
    ] {
        let mut run = assemble(&bundle).unwrap();
        let error = RunOptions {
            sensor_rates: [(node.to_string(), parse_rate(rate).unwrap())]
                .into_iter()
                .collect(),
            ..RunOptions::default()
        }
        .apply(&mut run, &bundle)
        .unwrap_err();
        assert!(error.contains(expected), "{error}");
    }
}
