//! Stage 7: the fusion update, in process.
//!
//! §4 names stage 7 *"fusion update, in-process rather than over gRPC"*, and
//! that is the whole change in shape. The reference does this:
//!
//! ```python
//! frame = _ingest_stream.ingest_frame(
//!     timestamp_s=timestamp_s,
//!     node_states=node_states,
//!     observations=observation_batch.observations,
//!     truths=truths,
//!     target_metadata=target_metadata,
//!     safety_events=_step_safety_events,
//! )
//! ```
//!
//! …where `_ingest_stream.ingest_frame` puts a request on a queue and blocks on
//! `next(self._responses)` — a synchronous gRPC round-trip on the sim thread,
//! every tick. Profiling measured it at **1.73 ms/frame, 5.9% of a
//! representative run**, so removing it is a minor throughput win; the reason to
//! remove it is that it is what stands between the world and an in-process
//! viewer, not the microseconds.
//!
//! Here the stage calls [`TrackingEngine::ingest_frame`] directly and writes the
//! resulting tracks back into the tick's world.
//!
//! # ⚠ Truth: fusion does not use it, but the tracker's metrics do
//!
//! `ingest_frame` takes a `truths` argument, and [`Stage::UpdateFusion`] is not
//! one of the two stages [`Stage::may_read_truth`] admits. That is not an
//! oversight in the capability list — reading `ingest_frame`, `truths` reaches
//! exactly two places:
//!
//! - `build_metrics`, which matches each track against the truth of the same id
//!   and computes a position error, and
//! - the `PlatformFrame` it returns, as a passthrough for the replay.
//!
//! **Nothing in the estimator sees it.** Fusion is measured against truth, not
//! informed by it, and `the_fused_tracks_are_identical_with_and_without_truth`
//! proves that rather than taking it on trust.
//!
//! So this stage passes an **empty** truth vector, and the truth-referenced
//! metrics are consequently not computed here. That leaves a real gap: under the
//! capability list as it stands, *no* stage can compute them — stage 11
//! (`CommitSnapshot`), where the replay is assembled, is not admitted either.
//!
//! The gap is left open rather than closed by widening the list. Handing fusion
//! a truth handle so it can compute an error metric would also let it fuse with
//! truth, and nothing structural would stop that — which is the one thing §3's
//! capability model exists to prevent. Separating measurement from estimation is
//! a design decision, and it is flagged rather than taken here.
//!
//! # The tracks are written back; the node states are not
//!
//! `ingest_frame` returns a whole `PlatformFrame`, including the node states it
//! was given. Only [`PlatformFrame::tracks`] goes back into the world.
//!
//! To be precise about why — an earlier version of this comment said writing
//! the node states back "can only lose", and that is an overclaim. The tracker
//! stores what it is handed and never modifies it, and this stage hands it the
//! world's own platforms, so the round trip is an **identity**: writing them
//! back would change nothing. It is skipped because it is redundant, not
//! because it is destructive, and `the_frame_round_trips_the_worlds_platforms`
//! pins the identity that makes that true. If the tracker ever starts editing
//! node states, that test fails and this decision needs revisiting.

use std::fmt;
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_core::{TrackerConfig, TrackingEngine};
use argusnet_types::{BearingObservation, PlatformFrame, SafetyEvent, TargetMetadata};

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

#[derive(Debug, Default)]
struct Inbox {
    observations: Vec<BearingObservation>,
    target_metadata: Vec<TargetMetadata>,
    safety_events: Vec<SafetyEvent>,
}

struct Shared {
    engine: TrackingEngine,
    /// What stage 6 produced, and what stages 4 and 10 raised. Taken on the
    /// next run, so a tick that produces nothing fuses nothing rather than
    /// re-fusing the last tick's batch.
    inbox: Inbox,
    frame: Option<Arc<PlatformFrame>>,
}

/// `TrackingEngine` is not `Debug`, and `StageHandler` requires it. Written by
/// hand rather than derived upstream: the engine's interior is the tracker's
/// business, and what a stage trace wants from it is how much is in flight.
impl fmt::Debug for Shared {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Shared")
            .field("pending_observations", &self.inbox.observations.len())
            .field("pending_safety_events", &self.inbox.safety_events.len())
            .field(
                "last_frame_tracks",
                &self.frame.as_ref().map(|frame| frame.tracks.len()),
            )
            .finish()
    }
}

/// A handle on the tracker, held before the stage is installed.
#[derive(Debug, Clone)]
pub struct FusionLink {
    inner: Arc<Mutex<Shared>>,
}

impl FusionLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// The observation batch for the next tick. Stage 6 fills this; until it
    /// lands, a caller or a test does.
    pub fn submit_observations(&self, observations: Vec<BearingObservation>) {
        self.lock().inbox.observations = observations;
    }

    pub fn submit_target_metadata(&self, metadata: Vec<TargetMetadata>) {
        self.lock().inbox.target_metadata = metadata;
    }

    /// Safety and deconfliction events raised earlier in this tick.
    pub fn submit_safety_events(&self, events: Vec<SafetyEvent>) {
        self.lock().inbox.safety_events = events;
    }

    /// The frame the last run produced.
    pub fn frame(&self) -> Option<Arc<PlatformFrame>> {
        self.lock().frame.clone()
    }

    pub fn config(&self) -> TrackerConfig {
        self.lock().engine.config().clone()
    }
}

/// [`Stage::UpdateFusion`] as a [`StageHandler`].
///
/// [`Stage::UpdateFusion`]: crate::stage::Stage::UpdateFusion
#[derive(Debug)]
pub struct FusionStage {
    link: FusionLink,
}

impl FusionStage {
    /// Refuses an invalid tracker config rather than carrying it to the first
    /// tick, where it would fault a run mid-flight.
    pub fn new(config: TrackerConfig) -> Result<Self, String> {
        let engine = TrackingEngine::new(config)?;
        Ok(Self {
            link: FusionLink {
                inner: Arc::new(Mutex::new(Shared {
                    engine,
                    inbox: Inbox::default(),
                    frame: None,
                })),
            },
        })
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> FusionLink {
        self.link.clone()
    }
}

impl StageHandler for FusionStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();

        // Taken, not copied: an inbox left full would re-fuse the same batch on
        // the next tick, which is exactly the shape of bug a stale-input guard
        // exists to prevent.
        let inbox = std::mem::take(&mut shared.inbox);

        // The reference calls `ingest_frame` unconditionally, every tick, even
        // with an empty batch -- the tracker's lifecycle ages tracks toward
        // `Coasting` and `Lost` on the strength of ticks passing, so skipping a
        // barren tick would keep a stale track alive forever.
        let node_states = context.world.platforms().to_vec();
        let frame = shared.engine.ingest_frame(
            context.timestamp_s,
            node_states,
            inbox.observations,
            // Empty by design. See the module docs: fusion is measured against
            // truth, not informed by it, and this stage holds no truth
            // capability.
            Vec::new(),
            inbox.target_metadata,
            inbox.safety_events,
        );

        context.world.set_tracks(frame.tracks.clone());
        let track_count = frame.tracks.len();
        let observation_count = frame.observations.len();
        let rejection_count = frame.rejected_observations.len();
        shared.frame = Some(frame);
        drop(shared);

        context.note(format!(
            "fused {observation_count} observation(s) into {track_count} track(s), \
             {rejection_count} rejected"
        ));
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_types::{NodeState, TruthState, Vector3};

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 11,
            dt_s: 1.0,
            steps: Some(2_000),
            run_id: "fusion".to_string(),
        })
        .expect("config is valid")
    }

    fn sensor(id: &str, x: f64, y: f64) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, y, 0.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: false,
            timestamp_s: 0.0,
            health: 1.0,
        }
    }

    /// A bearing from `(x, y, 0)` toward `target`, with no error.
    fn bearing(
        node_id: &str,
        target_id: &str,
        from: [f64; 3],
        toward: [f64; 3],
        timestamp_s: f64,
    ) -> BearingObservation {
        let origin = Vector3::new(from[0], from[1], from[2]);
        let direction = (Vector3::new(toward[0], toward[1], toward[2]) - origin).normalize();
        BearingObservation {
            node_id: node_id.to_string(),
            target_id: target_id.to_string(),
            origin,
            direction,
            bearing_std_rad: 0.001,
            timestamp_s,
            confidence: 1.0,
        }
    }

    /// Two sensors 400 m apart, both looking at a target between and above
    /// them — a wide enough intersection angle to clear the tracker's gate.
    fn crossing_pair(timestamp_s: f64) -> Vec<BearingObservation> {
        let target = [200.0, 300.0, 100.0];
        vec![
            bearing("s0", "t0", [0.0, 0.0, 0.0], target, timestamp_s),
            bearing("s1", "t0", [400.0, 0.0, 0.0], target, timestamp_s),
        ]
    }

    fn rig() -> (Simulation, FusionLink) {
        let mut sim = simulation();
        sim.load_world(
            vec![sensor("s0", 0.0, 0.0), sensor("s1", 400.0, 0.0)],
            Vec::new(),
        );
        let stage = FusionStage::new(TrackerConfig::default()).expect("config is valid");
        let link = stage.link();
        sim.set_stage_handler(Stage::UpdateFusion, Box::new(stage))
            .expect("stage 7 is not a transaction boundary");
        (sim, link)
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::UpdateFusion)
            .expect("stage 7 ran")
            .outcome
            .clone()
    }

    /// Step `count` ticks, feeding a fresh crossing pair each time.
    fn fly(sim: &mut Simulation, link: &FusionLink, count: usize) {
        for _ in 0..count {
            link.submit_observations(crossing_pair(sim.tick() as f64));
            sim.step().expect("tick runs");
        }
    }

    // -----------------------------------------------------------------
    // Truth
    // -----------------------------------------------------------------

    /// **The evidence behind this stage's central claim.** `ingest_frame` takes
    /// a `truths` argument, and this stage passes an empty vector because
    /// [`Stage::may_read_truth`] does not admit it. That is only defensible if
    /// fusion genuinely does not use truth — so rather than assert it from
    /// reading `build_metrics`, drive two identical engines, one given the
    /// truth of every target and one given none, and require the tracks to
    /// agree bit for bit.
    ///
    /// If a future change makes the estimator consult truth, this fails, and
    /// the capability question stops being a documentation matter.
    #[test]
    fn the_fused_tracks_are_identical_with_and_without_truth() {
        let truths = [TruthState {
            target_id: "t0".to_string(),
            position: Vector3::new(200.0, 300.0, 100.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            timestamp_s: 0.0,
        }];

        let mut without = TrackingEngine::new(TrackerConfig::default()).expect("valid");
        let mut with = TrackingEngine::new(TrackerConfig::default()).expect("valid");
        let nodes = vec![sensor("s0", 0.0, 0.0), sensor("s1", 400.0, 0.0)];

        let mut compared = 0;
        for tick in 0..8 {
            let timestamp_s = tick as f64;
            let blind = without.ingest_frame(
                timestamp_s,
                nodes.clone(),
                crossing_pair(timestamp_s),
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
            let sighted = with.ingest_frame(
                timestamp_s,
                nodes.clone(),
                crossing_pair(timestamp_s),
                truths
                    .iter()
                    .map(|truth| TruthState {
                        timestamp_s,
                        ..truth.clone()
                    })
                    .collect(),
                Vec::new(),
                Vec::new(),
            );

            assert_eq!(blind.tracks.len(), sighted.tracks.len(), "tick {tick}");
            for (a, b) in blind.tracks.iter().zip(sighted.tracks.iter()) {
                assert_eq!(a.track_id, b.track_id, "tick {tick}");
                assert_eq!(
                    a.position.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.position.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "tick {tick}: truth changed the fused position"
                );
                assert_eq!(
                    a.velocity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.velocity.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    "tick {tick}: truth changed the fused velocity"
                );
                compared += 1;
            }
        }

        // A run that produced no tracks would pass every assertion above
        // vacuously.
        assert!(
            compared > 0,
            "no tracks were compared, so nothing was proved"
        );
    }

    // -----------------------------------------------------------------
    // The stage
    // -----------------------------------------------------------------

    /// Two crossing bearings triangulate a target, and the resulting track goes
    /// into the tick's world rather than staying inside the tracker.
    #[test]
    fn crossing_bearings_produce_a_track_in_the_world() {
        let (mut sim, link) = rig();

        fly(&mut sim, &link, 4);

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        let tracks = sim.world().tracks();
        assert_eq!(tracks.len(), 1, "tracks: {tracks:?}");
        // Triangulated back to roughly where the bearings were aimed.
        assert!(
            (tracks[0].position[0] - 200.0).abs() < 25.0,
            "{:?}",
            tracks[0].position
        );
        assert!(
            (tracks[0].position[1] - 300.0).abs() < 25.0,
            "{:?}",
            tracks[0].position
        );
    }

    /// The inbox is *taken*, not copied. Leaving it full would re-fuse the same
    /// batch on the next tick — the tracker would see the same observation
    /// twice and treat the second as fresh evidence.
    #[test]
    fn the_observation_batch_is_consumed_rather_than_reused() {
        let (mut sim, link) = rig();
        link.submit_observations(crossing_pair(0.0));
        sim.step().expect("tick runs");

        // No new batch for this tick.
        sim.step().expect("tick runs");

        let frame = link.frame().expect("a frame was produced");
        assert_eq!(frame.observations.len(), 0);
        assert_eq!(frame.timestamp_s, 1.0);
    }

    /// The reference calls `ingest_frame` unconditionally, every tick, even
    /// with nothing to fuse — the track lifecycle ages toward `Coasting` and
    /// `Lost` on the strength of ticks passing, so a stage that skipped a barren
    /// tick would keep a stale track alive forever.
    #[test]
    fn a_tick_with_no_observations_still_runs_the_tracker() {
        let (mut sim, link) = rig();

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        let frame = link.frame().expect("a frame was produced");
        assert_eq!(frame.observations.len(), 0);
        assert_eq!(frame.timestamp_s, 0.0);
    }

    /// A track that stops being observed must age out. This is the behaviour
    /// the unconditional call above exists to preserve, so it is worth
    /// observing rather than asserting from the tracker's docs.
    #[test]
    fn an_unobserved_track_ages_out_over_barren_ticks() {
        let (mut sim, link) = rig();
        fly(&mut sim, &link, 4);
        assert_eq!(sim.world().tracks().len(), 1);

        // `max_stale_steps` defaults to 8; go well past it with nothing to see.
        for _ in 0..40 {
            sim.step().expect("tick runs");
        }

        assert!(
            sim.world().tracks().is_empty(),
            "the track survived 40 barren ticks: {:?}",
            sim.world().tracks()
        );
        assert!(link.frame().is_some());
    }

    /// Only the tracks come back into the world — and the reason the node
    /// states do not is that the write-back would be an **identity**, not that
    /// it would be harmful.
    ///
    /// The tracker stores what it is handed and never modifies it, and this
    /// stage hands it the world's own platforms, so `frame.nodes` comes back
    /// equal to what went in. That is what this pins, field by field. A mutant
    /// that adds the write-back is therefore equivalent, deliberately so — and
    /// if the tracker ever starts editing node states, this test fails and the
    /// decision to skip the write-back has to be made again on new evidence.
    #[test]
    fn the_frame_round_trips_the_worlds_platforms() {
        let (mut sim, link) = rig();
        sim.world_mut().platform_mut("s0").unwrap().health = 0.25;
        sim.world_mut().platform_mut("s0").unwrap().position = Vector3::new(7.0, 8.0, 9.0);

        fly(&mut sim, &link, 2);

        let frame = link.frame().expect("a frame was produced");
        let world = sim.world().platforms().to_vec();
        assert_eq!(frame.nodes.len(), world.len());
        for platform in &world {
            let echoed = frame
                .nodes
                .iter()
                .find(|node| node.node_id == platform.node_id)
                .expect("the tracker echoed every platform it was given");
            assert_eq!(echoed.position, platform.position);
            assert_eq!(echoed.velocity, platform.velocity);
            assert_eq!(echoed.health, platform.health);
            assert_eq!(echoed.is_mobile, platform.is_mobile);
        }
        // And the world still carries what the earlier stages left.
        assert_eq!(
            sim.world().platform("s0").expect("still there").health,
            0.25
        );
    }

    /// The truth vector this stage passes is empty, and a mutant that reaches
    /// for `context.truth` instead gets `None` — because §3 hands the
    /// capability out by stage and stage 7 is not on the list. The emptiness is
    /// therefore enforced by the capability model rather than by this stage
    /// remembering to pass `Vec::new()`, which is the stronger guarantee and
    /// the one worth pinning.
    #[test]
    fn stage_seven_cannot_reach_truth_at_all() {
        assert!(!Stage::UpdateFusion.may_read_truth());
        // The two that can, for contrast -- so a future widening of the list
        // has to come past this assertion.
        assert!(Stage::UpdateVioAndPose.may_read_truth());
        assert!(Stage::GenerateObservations.may_read_truth());
        assert!(!Stage::CommitSnapshot.may_read_truth());
    }

    /// Target metadata rides along to the frame, which is how per-target
    /// operator labels, priorities and observed/unobserved status reach the
    /// replay. Nothing else in this file would notice if it were dropped.
    #[test]
    fn target_metadata_rides_along_and_is_consumed() {
        let (mut sim, link) = rig();
        link.submit_target_metadata(vec![TargetMetadata {
            target_id: "t0".to_string(),
            operator_label: "Target Zero".to_string(),
            target_type: "simulated_mobile".to_string(),
            priority: 3,
            behavior: "orbit".to_string(),
            status: "observed".to_string(),
            first_seen_s: 0.0,
            last_seen_s: 0.0,
            confidence: 1.0,
            notes: String::new(),
        }]);

        sim.step().expect("tick runs");

        let frame = link.frame().expect("a frame was produced");
        assert_eq!(frame.target_metadata.len(), 1);
        assert_eq!(frame.target_metadata[0].operator_label, "Target Zero");

        sim.step().expect("tick runs");
        assert!(link.frame().expect("a frame").target_metadata.is_empty());
    }

    /// Each submission *replaces* the pending batch rather than appending to
    /// it. Appending would double-count a caller that submitted twice before a
    /// tick — the tracker would see the same bearing twice and treat the second
    /// as independent evidence, which is exactly how a triangulation gets
    /// falsely confident.
    #[test]
    fn submitting_twice_before_a_tick_replaces_rather_than_accumulates() {
        let (mut sim, link) = rig();

        link.submit_observations(crossing_pair(0.0));
        link.submit_observations(crossing_pair(0.0));
        sim.step().expect("tick runs");

        let frame = link.frame().expect("a frame was produced");
        assert_eq!(
            frame.observations.len() + frame.rejected_observations.len(),
            2,
            "the batch was accumulated, not replaced"
        );
    }

    /// Safety and deconfliction events raised earlier in the tick ride along on
    /// the frame, which is how they reach the replay.
    #[test]
    fn safety_events_ride_along_on_the_frame() {
        let (mut sim, link) = rig();
        link.submit_safety_events(vec![SafetyEvent {
            event_id: "e0".to_string(),
            subject_id: "s0".to_string(),
            timestamp_s: 0.0,
            state: "abort".to_string(),
            violations: vec!["max_speed".to_string()],
            command_clamped: true,
            blocked: true,
            reason: "over speed".to_string(),
        }]);

        sim.step().expect("tick runs");

        let frame = link.frame().expect("a frame was produced");
        assert_eq!(frame.safety_events.len(), 1);
        assert_eq!(frame.safety_events[0].event_id, "e0");

        // And they are consumed, not repeated.
        sim.step().expect("tick runs");
        assert!(link.frame().expect("a frame").safety_events.is_empty());
    }

    /// An invalid tracker config is refused when the stage is built, not
    /// carried to the first tick where it would fault a run mid-flight.
    #[test]
    fn an_invalid_config_is_refused_at_construction() {
        let bad = TrackerConfig {
            min_observations: 0,
            ..TrackerConfig::default()
        };

        assert!(FusionStage::new(bad).is_err());
    }

    /// Two runs with the same seed, config and observation stream fuse to the
    /// same tracks bit for bit.
    #[test]
    fn a_run_is_reproducible() {
        let mut runs = Vec::new();
        for _ in 0..2 {
            let (mut sim, link) = rig();
            fly(&mut sim, &link, 6);
            runs.push(sim.world().tracks().to_vec());
        }

        assert_eq!(runs[0].len(), runs[1].len());
        assert!(!runs[0].is_empty(), "no tracks, so nothing was compared");
        for (a, b) in runs[0].iter().zip(runs[1].iter()) {
            assert_eq!(
                a.position.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                b.position.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
            );
        }
    }
}
