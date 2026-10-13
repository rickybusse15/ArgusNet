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
use std::sync::Arc;

use argusnet_core::{TrackerConfig, TrackingEngine};
use argusnet_types::{BearingObservation, NodeState, PlatformFrame, SafetyEvent, TargetMetadata};

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

#[derive(Debug, Default)]
struct Inbox {
    observations: Vec<BearingObservation>,
    target_metadata: Vec<TargetMetadata>,
    safety_events: Vec<SafetyEvent>,
}

struct State {
    engine: TrackingEngine,
    /// What stage 6 produced, and what stages 4 and 10 raised. Taken on the
    /// next run, so a tick that produces nothing fuses nothing rather than
    /// re-fusing the last tick's batch.
    inbox: Inbox,
    frame: Option<Arc<PlatformFrame>>,
    /// Timestamped delivery between capture and fusion (B4). `None` is the
    /// legacy path: this tick's batch is fused this tick.
    delivery: Option<crate::delivery::DeliveryNetwork>,
    /// The boundaries the lane sat out since it last ran (B2.6), in order.
    /// Empty at the legacy rate.
    pending: Vec<PendingBoundary>,
}

/// One boundary stage 7 sat out (its lane at its own rate, B2.6): what it
/// would have fused then. Fused, in order, when the lane next runs, so the
/// tracker's state is what it would have been had it run every boundary.
/// Target metadata and safety events are not kept: the tracker only passes
/// them through to its frame, so the boundary publishes them itself.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
struct PendingBoundary {
    tick: u64,
    timestamp_s: f64,
    nodes: Vec<NodeState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    observations: Vec<crate::delivery::StoredObservation>,
}

/// `TrackingEngine` is not `Debug`, and `StageHandler` requires it. Written by
/// hand rather than derived upstream: the engine's interior is the tracker's
/// business, and what a stage trace wants from it is how much is in flight.
impl fmt::Debug for State {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FusionState")
            .field("pending_observations", &self.inbox.observations.len())
            .field("pending_safety_events", &self.inbox.safety_events.len())
            .field(
                "last_frame_tracks",
                &self.frame.as_ref().map(|frame| frame.tracks.len()),
            )
            .finish()
    }
}

/// The key a fusion checkpoint keeps its delivery network under.
const DELIVERY_KEY: &str = "delivery_network";
/// The key a fusion checkpoint keeps the boundaries it sat out under (B2.6).
const PENDING_KEY: &str = "pending";

/// A fusion checkpoint: the engine alone (legacy), or the engine beside the
/// delivery network and the boundaries sat out, each when there is one.
type SplitCheckpoint = (
    TrackingEngine,
    Option<crate::delivery::DeliveryNetwork>,
    Vec<PendingBoundary>,
);

fn split_checkpoint(
    state: &serde_json::Value,
) -> Result<SplitCheckpoint, crate::simulation::StageStateError> {
    if state.get(DELIVERY_KEY).is_none() && state.get(PENDING_KEY).is_none() {
        return Ok((
            crate::simulation::decode_stage_state(state)?,
            None,
            Vec::new(),
        ));
    }
    let engine =
        state
            .get("engine")
            .ok_or_else(|| crate::simulation::StageStateError::Malformed {
                detail: "a fusion checkpoint with a delivery network or pending boundaries has \
                         no engine"
                    .into(),
            })?;
    Ok((
        crate::simulation::decode_stage_state(engine)?,
        state
            .get(DELIVERY_KEY)
            .map(crate::simulation::decode_stage_state)
            .transpose()?,
        state
            .get(PENDING_KEY)
            .map(crate::simulation::decode_stage_state)
            .transpose()?
            .unwrap_or_default(),
    ))
}

/// [`Stage::UpdateFusion`] as a [`StageHandler`].
///
/// [`Stage::UpdateFusion`]: crate::stage::Stage::UpdateFusion
#[derive(Debug)]
pub struct FusionStage {
    state: State,
}

impl FusionStage {
    /// Refuses an invalid tracker config rather than carrying it to the first
    /// tick, where it would fault a run mid-flight.
    pub fn new(config: TrackerConfig) -> Result<Self, String> {
        let engine = TrackingEngine::new(config)?;
        Ok(Self {
            state: State {
                engine,
                inbox: Inbox::default(),
                frame: None,
                delivery: None,
                pending: Vec::new(),
            },
        })
    }

    /// Route observations through a delivery network (B4): fusion then
    /// ingests, at each tick, the evidence acquired the profile's fusion lag
    /// ago. `None` restores the legacy same-tick path.
    pub fn set_delivery(&mut self, delivery: Option<crate::delivery::DeliveryNetwork>) {
        self.state.delivery = delivery;
    }

    pub fn delivery(&self) -> Option<&crate::delivery::DeliveryNetwork> {
        self.state.delivery.as_ref()
    }

    pub fn delivery_mut(&mut self) -> Option<&mut crate::delivery::DeliveryNetwork> {
        self.state.delivery.as_mut()
    }

    /// The observation batch for the next run. Stage 6 fills it through
    /// [`Handoff::ObservationsToFusion`]; without stage 6, a caller or a test
    /// does.
    ///
    /// [`Handoff::ObservationsToFusion`]: crate::simulation::Handoff::ObservationsToFusion
    pub fn submit_observations(&mut self, observations: Vec<BearingObservation>) {
        self.state.inbox.observations = observations;
    }

    pub fn submit_target_metadata(&mut self, metadata: Vec<TargetMetadata>) {
        self.state.inbox.target_metadata = metadata;
    }

    /// Safety and deconfliction events raised earlier in this tick. Stage 4
    /// fills it through [`Handoff::SafetyEventsToFusion`].
    ///
    /// [`Handoff::SafetyEventsToFusion`]: crate::simulation::Handoff::SafetyEventsToFusion
    pub fn submit_safety_events(&mut self, events: Vec<SafetyEvent>) {
        self.state.inbox.safety_events = events;
    }

    /// The frame the last run produced.
    pub fn frame(&self) -> Option<Arc<PlatformFrame>> {
        self.state.frame.clone()
    }

    pub fn config(&self) -> &TrackerConfig {
        self.state.engine.config()
    }
}

impl StageHandler for FusionStage {
    /// A new engine with the same configuration: no tracks, no health.
    fn reset(&mut self) {
        let shared = &mut self.state;
        let config = shared.engine.config().clone();
        shared.engine = TrackingEngine::new(config).expect("the config was valid at construction");
        shared.inbox = Inbox::default();
        shared.frame = None;
        shared.pending.clear();
        if let Some(delivery) = shared.delivery.as_mut() {
            delivery.reset();
        }
    }

    /// Every track's filters, lifecycle and history, and each node's health.
    /// With a delivery network, the envelopes in transit too, under
    /// [`DELIVERY_KEY`]; with boundaries sat out (B2.6), those under
    /// [`PENDING_KEY`]. With neither the state is exactly the engine's, as it
    /// always was.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let engine = serde_json::to_value(&self.state.engine).expect("engine state serialises");
        if self.state.delivery.is_none() && self.state.pending.is_empty() {
            return Some(engine);
        }
        let mut state = serde_json::json!({ "engine": engine });
        if let Some(delivery) = &self.state.delivery {
            state[DELIVERY_KEY] =
                serde_json::to_value(delivery).expect("delivery state serialises");
        }
        if !self.state.pending.is_empty() {
            state[PENDING_KEY] =
                serde_json::to_value(&self.state.pending).expect("pending boundaries serialise");
        }
        Some(state)
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let (engine, delivery, _) = split_checkpoint(state)?;
        if delivery.is_some() != self.state.delivery.is_some() {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: "the checkpoint and this run disagree on whether delivery is modelled"
                    .into(),
            });
        }
        if let (Some(saved), Some(ours)) = (&delivery, &self.state.delivery) {
            if saved.profile() != ours.profile() {
                return Err(crate::simulation::StageStateError::Malformed {
                    detail: "the checkpoint's delivery profile differs from this run's".into(),
                });
            }
        }
        if engine.config() != self.state.engine.config() {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: "the checkpoint's tracker was configured differently from this run's"
                    .into(),
            });
        }
        Ok(())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let (engine, delivery, pending) = split_checkpoint(state)?;
        let shared = &mut self.state;
        shared.engine = engine;
        shared.inbox = Inbox::default();
        shared.frame = None;
        shared.pending = pending;
        if delivery.is_some() {
            shared.delivery = delivery;
        }
        Ok(())
    }

    /// The lane is not due (B2.6): keep this boundary's platforms and
    /// observations to fuse, in order, when it next runs, and publish a frame
    /// that fused nothing -- the tracks as they stand, this boundary's
    /// platforms, and the metadata and safety events the tracker would only
    /// have passed through.
    fn not_due(&mut self, context: &mut TickContext<'_>) -> Result<(), StageError> {
        let shared = &mut self.state;
        let inbox = std::mem::take(&mut shared.inbox);
        let nodes = context.world.platforms().to_vec();
        shared.pending.push(PendingBoundary {
            tick: context.tick,
            timestamp_s: context.timestamp_s,
            nodes: nodes.clone(),
            observations: inbox.observations.iter().map(Into::into).collect(),
        });
        let tracks = context.world.tracks().to_vec();
        let metrics = argusnet_types::PlatformMetrics {
            mean_error_m: None,
            max_error_m: None,
            active_track_count: tracks.len() as u32,
            observation_count: 0,
            accepted_observation_count: 0,
            rejected_observation_count: 0,
            mean_measurement_std_m: None,
            track_errors_m: Default::default(),
            rejection_counts: Default::default(),
            accepted_observations_by_target: Default::default(),
            rejected_observations_by_target: Default::default(),
        };
        shared.frame = Some(Arc::new(PlatformFrame {
            timestamp_s: context.timestamp_s,
            nodes,
            observations: Vec::new(),
            rejected_observations: Vec::new(),
            tracks,
            truths: Vec::new(),
            metrics,
            generation_rejections: Vec::new(),
            target_metadata: inbox.target_metadata,
            safety_events: inbox.safety_events,
        }));
        context.note(format!(
            "not due: {} boundary(ies) waiting to be fused",
            shared.pending.len()
        ));
        Ok(())
    }

    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let shared = &mut self.state;

        // Taken, not copied: an inbox left full would re-fuse the same batch on
        // the next tick, which is exactly the shape of bug a stale-input guard
        // exists to prevent.
        let inbox = std::mem::take(&mut shared.inbox);

        // The boundaries the lane sat out first, each as it was then (B2.6),
        // so the tracker ages and updates exactly as it would have every
        // boundary; at the legacy rate there are none. What they fused is
        // reported in this boundary's frame, before this boundary's own.
        let mut caught_up = Vec::new();
        let mut caught_up_rejections = Vec::new();
        for boundary in std::mem::take(&mut shared.pending) {
            let observations = boundary.observations.iter().map(Into::into).collect();
            let (fusion_time_s, observations, drops) = take_delivered(
                shared.delivery.as_mut(),
                boundary.tick,
                boundary.timestamp_s,
                observations,
            );
            let frame = shared.engine.ingest_frame(
                fusion_time_s,
                boundary.nodes,
                observations,
                Vec::new(),
                Vec::new(),
                Vec::new(),
            );
            caught_up.extend(frame.observations.iter().cloned());
            caught_up_rejections.extend(frame.rejected_observations.iter().cloned());
            caught_up_rejections.extend(drops.iter().map(|drop| drop.rejection(context.dt_s)));
        }

        // The reference calls `ingest_frame` unconditionally, every tick, even
        // with an empty batch -- the tracker's lifecycle ages tracks toward
        // `Coasting` and `Lost` on the strength of ticks passing, so skipping a
        // barren tick would keep a stale track alive forever.
        let node_states = context.world.platforms().to_vec();
        // With delivery modelled, fuse what was acquired at the horizon, at
        // the horizon's time. Before the first horizon nothing can have been
        // delivered, so an empty batch is fused at the current time; no track
        // can exist yet, so nothing is aged at the wrong time.
        let (fusion_time_s, observations, drops) = take_delivered(
            shared.delivery.as_mut(),
            context.tick,
            context.timestamp_s,
            inbox.observations,
        );
        let mut frame = shared.engine.ingest_frame(
            fusion_time_s,
            node_states,
            observations,
            // Empty by design. See the module docs: fusion is measured against
            // truth, not informed by it, and this stage holds no truth
            // capability.
            Vec::new(),
            inbox.target_metadata,
            inbox.safety_events,
        );

        if !caught_up.is_empty() || !caught_up_rejections.is_empty() {
            let frame = Arc::make_mut(&mut frame);
            caught_up.append(&mut frame.observations);
            frame.observations = caught_up;
            caught_up_rejections.append(&mut frame.rejected_observations);
            frame.rejected_observations = caught_up_rejections;
        }
        // Every delivery drop is recorded with its reason, beside the
        // tracker's own rejections.
        if !drops.is_empty() {
            Arc::make_mut(&mut frame)
                .rejected_observations
                .extend(drops.iter().map(|drop| drop.rejection(context.dt_s)));
        }
        context.world.set_tracks(frame.tracks.clone());
        let track_count = frame.tracks.len();
        let observation_count = frame.observations.len();
        let rejection_count = frame.rejected_observations.len();
        shared.frame = Some(frame);

        context.note(format!(
            "fused {observation_count} observation(s) into {track_count} track(s), \
             {rejection_count} rejected"
        ));
        Ok(StageOutcome::Ran)
    }
}

/// The batch to fuse at legacy tick `tick`: this tick's observations as they
/// are (legacy), or, with delivery modelled, those captured now go in transit
/// and what was acquired at the horizon comes out, at the horizon's time, with
/// whatever the network dropped.
fn take_delivered(
    delivery: Option<&mut crate::delivery::DeliveryNetwork>,
    tick: u64,
    timestamp_s: f64,
    observations: Vec<BearingObservation>,
) -> (f64, Vec<BearingObservation>, Vec<crate::delivery::Dropped>) {
    match delivery {
        None => (timestamp_s, observations, Vec::new()),
        Some(delivery) => {
            delivery.capture(tick, timestamp_s, &observations);
            let (time_s, batch) = delivery.deliver(tick).unwrap_or((timestamp_s, Vec::new()));
            (time_s, batch, delivery.take_drops())
        }
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
        Simulation::bare(SimulationConfig {
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
            ..Default::default()
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

    fn rig() -> Simulation {
        let mut sim = simulation();
        sim.load_world(
            vec![sensor("s0", 0.0, 0.0), sensor("s1", 400.0, 0.0)],
            Vec::new(),
        );
        let stage = FusionStage::new(TrackerConfig::default()).expect("config is valid");
        sim.set_stage_handler(Stage::UpdateFusion, Box::new(stage))
            .expect("stage 7 is not a transaction boundary");
        sim
    }

    fn fusion(sim: &mut Simulation) -> &mut FusionStage {
        sim.stage_mut(Stage::UpdateFusion)
            .expect("stage 7 is installed")
    }

    fn fused(sim: &Simulation) -> &FusionStage {
        sim.stage(Stage::UpdateFusion)
            .expect("stage 7 is installed")
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
    fn fly(sim: &mut Simulation, count: usize) {
        for _ in 0..count {
            let timestamp_s = sim.tick() as f64;
            fusion(sim).submit_observations(crossing_pair(timestamp_s));
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
        let mut sim = rig();

        fly(&mut sim, 4);

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
        let mut sim = rig();
        fusion(&mut sim).submit_observations(crossing_pair(0.0));
        sim.step().expect("tick runs");

        // No new batch for this tick.
        sim.step().expect("tick runs");

        let frame = fused(&sim).frame().expect("a frame was produced");
        assert_eq!(frame.observations.len(), 0);
        assert_eq!(frame.timestamp_s, 1.0);
    }

    /// The reference calls `ingest_frame` unconditionally, every tick, even
    /// with nothing to fuse — the track lifecycle ages toward `Coasting` and
    /// `Lost` on the strength of ticks passing, so a stage that skipped a barren
    /// tick would keep a stale track alive forever.
    #[test]
    fn a_tick_with_no_observations_still_runs_the_tracker() {
        let mut sim = rig();

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        let frame = fused(&sim).frame().expect("a frame was produced");
        assert_eq!(frame.observations.len(), 0);
        assert_eq!(frame.timestamp_s, 0.0);
    }

    /// A track that stops being observed must age out. This is the behaviour
    /// the unconditional call above exists to preserve, so it is worth
    /// observing rather than asserting from the tracker's docs.
    #[test]
    fn an_unobserved_track_ages_out_over_barren_ticks() {
        let mut sim = rig();
        fly(&mut sim, 4);
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
        assert!(fused(&sim).frame().is_some());
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
        let mut sim = rig();
        sim.world_mut().platform_mut("s0").unwrap().health = 0.25;
        sim.world_mut().platform_mut("s0").unwrap().position = Vector3::new(7.0, 8.0, 9.0);

        fly(&mut sim, 2);

        let frame = fused(&sim).frame().expect("a frame was produced");
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
        let mut sim = rig();
        fusion(&mut sim).submit_target_metadata(vec![TargetMetadata {
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

        let frame = fused(&sim).frame().expect("a frame was produced");
        assert_eq!(frame.target_metadata.len(), 1);
        assert_eq!(frame.target_metadata[0].operator_label, "Target Zero");

        sim.step().expect("tick runs");
        assert!(fused(&sim)
            .frame()
            .expect("a frame")
            .target_metadata
            .is_empty());
    }

    /// Each submission *replaces* the pending batch rather than appending to
    /// it. Appending would double-count a caller that submitted twice before a
    /// tick — the tracker would see the same bearing twice and treat the second
    /// as independent evidence, which is exactly how a triangulation gets
    /// falsely confident.
    #[test]
    fn submitting_twice_before_a_tick_replaces_rather_than_accumulates() {
        let mut sim = rig();

        fusion(&mut sim).submit_observations(crossing_pair(0.0));
        fusion(&mut sim).submit_observations(crossing_pair(0.0));
        sim.step().expect("tick runs");

        let frame = fused(&sim).frame().expect("a frame was produced");
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
        let mut sim = rig();
        fusion(&mut sim).submit_safety_events(vec![SafetyEvent {
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

        let frame = fused(&sim).frame().expect("a frame was produced");
        assert_eq!(frame.safety_events.len(), 1);
        assert_eq!(frame.safety_events[0].event_id, "e0");

        // And they are consumed, not repeated.
        sim.step().expect("tick runs");
        assert!(fused(&sim)
            .frame()
            .expect("a frame")
            .safety_events
            .is_empty());
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
            let mut sim = rig();
            fly(&mut sim, 6);
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

    fn fusion_state(sim: &Simulation) -> serde_json::Value {
        fused(sim).checkpoint_state().expect("stage 7 checkpoints")
    }

    fn executor(rate: Option<&str>, first: u64) -> crate::scheduler::PhasedLaneExecutor {
        let executor =
            crate::scheduler::PhasedLaneExecutor::new::<crate::simulation::TickError>(1.0, first)
                .unwrap();
        match rate {
            Some(rate) => executor
                .with_fusion_rate(crate::scheduler::parse_rate(rate).unwrap())
                .unwrap(),
            None => executor,
        }
    }

    /// Boundary `k`'s input: the crossing pair, observed at `k`.
    fn feed(sim: &mut Simulation, k: u64) {
        fusion(sim).submit_observations(crossing_pair(k as f64));
    }

    /// B2.6: a fusion lane at a third of the boundary rate keeps every
    /// boundary it sits out and fuses it when it runs: at each boundary it
    /// runs, the tracker is exactly the every-boundary run's. A boundary it
    /// sits out publishes the tracks as they stand and fuses nothing.
    #[test]
    fn a_fusion_lane_at_its_own_rate_catches_up_exactly() {
        let mut every = rig();
        let mut rated = rig();
        rated.declare_lane_rate(Stage::UpdateFusion, "1/3");
        let (mut every_lanes, mut rated_lanes) = (executor(None, 0), executor(Some("1/3"), 0));
        for k in 0..12 {
            feed(&mut every, k);
            feed(&mut rated, k);
            every_lanes.step(&mut every).expect("tick");
            rated_lanes.step(&mut rated).expect("tick");
            let frame = fused(&rated).frame().expect("a frame every boundary");
            if k % 3 == 0 {
                assert_eq!(fusion_state(&rated), fusion_state(&every), "boundary {k}");
                assert_eq!(
                    rated.world().tracks(),
                    every.world().tracks(),
                    "boundary {k}"
                );
                // The boundaries it caught up on are reported with this one.
                let fused_here = if k == 0 { 2 } else { 6 };
                assert_eq!(frame.observations.len(), fused_here, "boundary {k}");
            } else {
                assert_eq!(
                    fusion_state(&rated)["pending"].as_array().unwrap().len() as u64,
                    k % 3
                );
                assert!(frame.observations.is_empty(), "boundary {k}");
                assert_eq!(frame.timestamp_s, k as f64);
                assert_eq!(frame.tracks, rated.world().tracks());
            }
        }
        assert!(
            !every.world().tracks().is_empty(),
            "no tracks, so nothing was compared"
        );
        // The legacy rate checkpoints the engine alone, as it always did.
        assert!(fusion_state(&every).get("engine").is_none());
    }

    /// A checkpoint taken while boundaries are pending (ticks 4 and 5) holds
    /// them, and a run restored from it fuses exactly what an uninterrupted
    /// one does.
    #[test]
    fn pending_fusion_boundaries_survive_a_checkpoint() {
        let mut whole = rig();
        whole.declare_lane_rate(Stage::UpdateFusion, "1/3");
        let mut lanes = executor(Some("1/3"), 0);
        for k in 0..6 {
            feed(&mut whole, k);
            lanes.step(&mut whole).expect("tick");
        }
        let checkpoint = whole.checkpoint().expect("not faulted");
        assert_eq!(
            fusion_state(&whole)["pending"].as_array().unwrap().len(),
            2,
            "ticks 4 and 5 are pending"
        );

        let mut resumed = rig();
        resumed.declare_lane_rate(Stage::UpdateFusion, "1/3");
        resumed.restore(&checkpoint).expect("restores");
        let mut resumed_lanes = executor(Some("1/3"), 6);
        for k in 6..10 {
            feed(&mut whole, k);
            feed(&mut resumed, k);
            lanes.step(&mut whole).expect("tick");
            resumed_lanes.step(&mut resumed).expect("tick");
            assert_eq!(fusion_state(&resumed), fusion_state(&whole), "boundary {k}");
            assert_eq!(resumed.world().tracks(), whole.world().tracks());
        }
    }
}
