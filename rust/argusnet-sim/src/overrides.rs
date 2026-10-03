//! Stage 2's mission overrides: frontier, then POI, then egress.
//!
//! §4 names stage 2 *"frontier, then POI, then egress overrides, in that
//! precedence"*, and the reference is three near-identical blocks in a row:
//!
//! ```python
//! if _frontier_trajectories and _scan_phase in ("scanning", "localizing"):
//!     for ns in node_states:
//!         traj_fn = _frontier_trajectories.get(ns.node_id)
//!         if traj_fn is not None:
//!             pos_ov, vel_ov = traj_fn(timestamp_s)
//!             ns = replace(ns, position=list(pos_ov), velocity=list(vel_ov))
//! # ... then the same for _poi_trajectories, then for _egress_trajectories
//! ```
//!
//! Each block rewrites the list the next one reads, so a platform holding
//! entries in two tables is flown by the **last** one to touch it. Egress beats
//! POI beats frontier. That is a real decision — a drone recalled to base does
//! not keep inspecting — and it is the whole content of the stage.
//!
//! # This stage applies; it does not select
//!
//! The reference sets `_frontier_trajectories` some 350 lines *later* in the
//! same tick, next to the mission transitions. So a trajectory chosen on tick
//! `n` first moves a platform on tick `n + 1`, which is exactly §4's *"decisions
//! that take effect on the next tick"*. Selection is therefore stage 9's work
//! and is not here; this stage only samples what stage 9 already decided.
//!
//! Keeping the two apart is also what makes the precedence testable without a
//! planner: a test installs three trajectories by hand and asks which one flew
//! the platform.
//!
//! # The frontier gate reads the previous tick's phase
//!
//! Only the frontier block is gated, and on `scanning` or `localizing`. The
//! reference's own comment says why: *"Kept active through the localizing phase
//! so drones hold position rather than jumping back to their lawnmower
//! trajectories."* Stage 2 runs long before stage 9, so the phase it reads is
//! the one committed at the end of the previous tick — the same causal shape as
//! stage 8's localization gate.
//!
//! POI and egress are ungated in the reference, and are ungated here.

use std::collections::HashMap;

use argusnet_types::Vector3;

use crate::mission::{stage_nine_phase, MissionPhase};
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// Metres per second a redirected drone flies at. `_make_redirect_goto`'s only
/// caller passes `28.0` at both of its sites.
pub const REDIRECT_SPEED_MPS: f64 = 28.0;

/// The floor on a redirect's flight time, in seconds.
///
/// `max(dist / max(speed, 1e-9), 1.0e-3)`. A drone already standing on its
/// target still gets a millisecond of flight rather than a zero-length
/// interval, which is what keeps the `frac` division below finite.
pub const MIN_FLIGHT_TIME_S: f64 = 1.0e-3;

/// The guard on every interval division in the reference's trajectory.
const EPSILON_S: f64 = 1.0e-9;

/// A linear flight to a point, then a hover.
///
/// Ported from `_make_goto_hover`. Three of its properties are less obvious
/// than they look and each is pinned by a test:
///
/// - **Velocity is constant while in flight**, not scaled by progress. It is
///   the whole displacement divided by the whole interval, so it reads as the
///   average velocity rather than an instantaneous one.
/// - **On arrival velocity is exactly zero** and the position is exactly the
///   target — not the lerp evaluated at `frac == 1`, which would differ in the
///   last bits.
/// - **Before `t_start` the position clamps but the velocity does not.** The
///   reference clamps `frac` into `[0, 1]` and then computes velocity from the
///   interval alone, so a sample taken early reports the start position moving
///   at full speed. Nothing samples early today; it is reproduced because it is
///   what is written.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GotoHover {
    start: [f64; 3],
    target_xy: [f64; 2],
    target_z: f64,
    t_start: f64,
    t_arrive: f64,
}

impl GotoHover {
    /// A flight over an explicit interval — `_make_goto_hover(sp, txy, tz, ts, ta)`.
    pub fn over(
        start: [f64; 3],
        target_xy: [f64; 2],
        target_z: f64,
        t_start: f64,
        t_arrive: f64,
    ) -> Self {
        Self {
            start,
            target_xy,
            target_z,
            t_start,
            t_arrive,
        }
    }

    /// A flight at a speed — `_make_redirect_goto` without a route planner.
    ///
    /// The distance is the **horizontal** one: the reference measures
    /// `norm(txy - sp[:2])`, so a redirect that also climbs takes no longer for
    /// the climb. The arrival time is `t_start + max(dist / speed,
    /// MIN_FLIGHT_TIME_S)`.
    pub fn redirect(
        start: [f64; 3],
        target_xy: [f64; 2],
        target_z: f64,
        t_start: f64,
        speed_mps: f64,
    ) -> Self {
        // `np.linalg.norm`: the plain square root of the sum of squares, not
        // `hypot`, which rounds differently in the last bit (see
        // `follow_path::norm`). Found by the P4.8e.3 full-run comparison.
        let (dx, dy) = (target_xy[0] - start[0], target_xy[1] - start[1]);
        let dist_m = (dx * dx + dy * dy).sqrt();
        let flight_s = (dist_m / speed_mps.max(EPSILON_S)).max(MIN_FLIGHT_TIME_S);
        Self::over(start, target_xy, target_z, t_start, t_start + flight_s)
    }

    /// Position and velocity at `t`.
    pub fn sample(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        if t >= self.t_arrive {
            return (
                [self.target_xy[0], self.target_xy[1], self.target_z],
                [0.0, 0.0, 0.0],
            );
        }
        let span = (self.t_arrive - self.t_start).max(EPSILON_S);
        let frac = ((t - self.t_start) / span).clamp(0.0, 1.0);
        let delta = [
            self.target_xy[0] - self.start[0],
            self.target_xy[1] - self.start[1],
            self.target_z - self.start[2],
        ];
        let position = [
            self.start[0] + frac * delta[0],
            self.start[1] + frac * delta[1],
            self.start[2] + frac * delta[2],
        ];
        let dt_inv = 1.0 / span;
        let velocity = [delta[0] * dt_inv, delta[1] * dt_inv, delta[2] * dt_inv];
        (position, velocity)
    }

    /// Where this trajectory ends. Stage 9 stores it to decide when a platform
    /// has arrived.
    pub fn target(&self) -> [f64; 3] {
        [self.target_xy[0], self.target_xy[1], self.target_z]
    }

    pub fn t_arrive(&self) -> f64 {
        self.t_arrive
    }
}

/// Which table flew a platform on the last run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OverrideSource {
    Frontier,
    Poi,
    Egress,
}

impl OverrideSource {
    pub fn as_str(self) -> &'static str {
        match self {
            OverrideSource::Frontier => "frontier",
            OverrideSource::Poi => "poi",
            OverrideSource::Egress => "egress",
        }
    }
}

#[derive(Debug, Default)]
struct State {
    frontier: HashMap<String, GotoHover>,
    poi: HashMap<String, GotoHover>,
    egress: HashMap<String, GotoHover>,
    /// Gate the frontier table on stage 9's phase, read at stage 2 time, so
    /// it is the phase committed at the end of the previous tick.
    follows_mission: bool,
    applied: HashMap<String, OverrideSource>,
}

/// A write into stage 2's tables, queued by stage 9 and applied by the
/// session straight after it ([`Handoff::OverridesFromMission`]). Applied in
/// the order queued, which is the order stage 9 used to write them.
///
/// [`Handoff::OverridesFromMission`]: crate::simulation::Handoff::OverridesFromMission
#[derive(Debug, Clone, PartialEq)]
pub enum OverrideCommand {
    SetPoi(String, GotoHover),
    ClearPoi(String),
    SetEgress(String, GotoHover),
    /// `_egress_trajectories.clear()`.
    ClearAllEgress,
}

/// [`Stage::ApplyMissionOverrides`] as a [`StageHandler`].
///
/// [`Stage::ApplyMissionOverrides`]: crate::stage::Stage::ApplyMissionOverrides
#[derive(Debug, Default)]
pub struct MissionOverrideStage {
    state: State,
}

impl MissionOverrideStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Gate the frontier table on stage 9's phase, read through the views
    /// lent to stage 2 ([`stage_nine_phase`]).
    ///
    /// Left unfollowed, the frontier gate is **open**: a run with no mission
    /// machine has no phase to gate on, and the reference always has one. A
    /// test pins that, so a future run that acquires phases without following
    /// them fails rather than silently flying its drones through egress.
    pub fn follow_mission_phase(&mut self) {
        self.state.follows_mission = true;
    }

    pub fn set_frontier(&mut self, node_id: &str, trajectory: GotoHover) {
        self.state.frontier.insert(node_id.to_string(), trajectory);
    }

    pub fn clear_frontier(&mut self, node_id: &str) {
        self.state.frontier.remove(node_id);
    }

    pub fn set_poi(&mut self, node_id: &str, trajectory: GotoHover) {
        self.state.poi.insert(node_id.to_string(), trajectory);
    }

    pub fn clear_poi(&mut self, node_id: &str) {
        self.state.poi.remove(node_id);
    }

    pub fn set_egress(&mut self, node_id: &str, trajectory: GotoHover) {
        self.state.egress.insert(node_id.to_string(), trajectory);
    }

    pub fn clear_egress(&mut self, node_id: &str) {
        self.state.egress.remove(node_id);
    }

    /// `_egress_trajectories.clear()` — the reference drops every recall at
    /// once when the egress phase ends, rather than one platform at a time.
    pub fn clear_all_egress(&mut self) {
        self.state.egress.clear();
    }

    /// Apply one queued [`OverrideCommand`].
    pub fn apply(&mut self, command: OverrideCommand) {
        match command {
            OverrideCommand::SetPoi(node_id, trajectory) => self.set_poi(&node_id, trajectory),
            OverrideCommand::ClearPoi(node_id) => self.clear_poi(&node_id),
            OverrideCommand::SetEgress(node_id, trajectory) => {
                self.set_egress(&node_id, trajectory)
            }
            OverrideCommand::ClearAllEgress => self.clear_all_egress(),
        }
    }

    pub fn frontier_target(&self, node_id: &str) -> Option<[f64; 3]> {
        self.state.frontier.get(node_id).map(GotoHover::target)
    }

    /// Which table flew `node_id` on the last run, or `None` if nothing did.
    pub fn applied_source(&self, node_id: &str) -> Option<OverrideSource> {
        self.state.applied.get(node_id).copied()
    }

    /// How many platforms the last run overrode.
    pub fn applied_count(&self) -> usize {
        self.state.applied.len()
    }
}

/// What stage 2 checkpoints. Ordered maps, so a checkpoint's text does not
/// depend on hash order.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct OverrideCheckpoint {
    frontier: std::collections::BTreeMap<String, GotoHover>,
    poi: std::collections::BTreeMap<String, GotoHover>,
    egress: std::collections::BTreeMap<String, GotoHover>,
}

impl StageHandler for MissionOverrideStage {
    fn reset(&mut self) {
        let shared = &mut self.state;
        shared.frontier.clear();
        shared.poi.clear();
        shared.egress.clear();
        shared.applied.clear();
    }

    /// The redirects in force: each is a flight already under way, and a
    /// resumed run must finish it the same way.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        let state = OverrideCheckpoint {
            frontier: shared.frontier.clone().into_iter().collect(),
            poi: shared.poi.clone().into_iter().collect(),
            egress: shared.egress.clone().into_iter().collect(),
        };
        Some(serde_json::to_value(state).expect("override state serialises"))
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        crate::simulation::decode_stage_state::<OverrideCheckpoint>(state).map(|_| ())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: OverrideCheckpoint = crate::simulation::decode_stage_state(state)?;
        let shared = &mut self.state;
        shared.frontier = decoded.frontier.into_iter().collect();
        shared.poi = decoded.poi.into_iter().collect();
        shared.egress = decoded.egress.into_iter().collect();
        shared.applied.clear();
        Ok(())
    }

    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let shared = &mut self.state;
        shared.applied.clear();

        // The reference reads `_scan_phase` here, before stage 9 has advanced
        // it, so a stage with no phase to follow is the only case that needs
        // deciding.
        let followed = shared
            .follows_mission
            .then(|| stage_nine_phase(context.stages))
            .flatten();
        let frontier_open = match followed {
            Some(phase) => matches!(phase, MissionPhase::Scanning | MissionPhase::Localizing),
            None => true,
        };

        if shared.frontier.is_empty() && shared.poi.is_empty() && shared.egress.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let timestamp_s = context.timestamp_s;
        // The three passes are ordered, and later ones overwrite earlier ones,
        // because that is what three successive rebuilds of `node_states` do.
        let passes: [(OverrideSource, bool); 3] = [
            (OverrideSource::Frontier, frontier_open),
            (OverrideSource::Poi, true),
            (OverrideSource::Egress, true),
        ];

        for (source, open) in passes {
            if !open {
                continue;
            }
            let table = match source {
                OverrideSource::Frontier => &shared.frontier,
                OverrideSource::Poi => &shared.poi,
                OverrideSource::Egress => &shared.egress,
            };
            // Cloned out of the table so the borrow ends before `applied` is
            // written; the tables are one entry per platform, so this is small.
            let sampled: Vec<(String, [f64; 3], [f64; 3])> = table
                .iter()
                .map(|(node_id, trajectory)| {
                    let (position, velocity) = trajectory.sample(timestamp_s);
                    (node_id.clone(), position, velocity)
                })
                .collect();
            for (node_id, position, velocity) in sampled {
                let Some(platform) = context.world.platform_mut(&node_id) else {
                    // The reference iterates `node_states` and looks the
                    // trajectory up, so an entry for a platform that is not in
                    // the world is simply never read. Dropping it here is the
                    // same behaviour, not a swallowed error.
                    continue;
                };
                platform.position = Vector3::new(position[0], position[1], position[2]);
                platform.velocity = Vector3::new(velocity[0], velocity[1], velocity[2]);
                shared.applied.insert(node_id, source);
            }
        }

        if shared.applied.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let count = shared.applied.len();
        context.note(format!("overrode {count} platform(s)"));
        Ok(StageOutcome::Ran)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::mission::{MissionInputs, MissionStage};
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_types::NodeState;

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 11,
            dt_s: 1.0,
            steps: Some(2_000),
            run_id: "overrides".to_string(),
        })
        .expect("config is valid")
    }

    fn node(id: &str, x: f64, y: f64) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, y, 80.0),
            velocity: Vector3::new(1.0, 2.0, 3.0),
            is_mobile: true,
            timestamp_s: 0.0,
            health: 1.0,
            ..Default::default()
        }
    }

    fn rig(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        sim.set_stage_handler(
            Stage::ApplyMissionOverrides,
            Box::new(MissionOverrideStage::new()),
        )
        .expect("stage 2 is not a transaction boundary");
        sim
    }

    /// The installed stage, reached through the session that owns it.
    fn ov(sim: &mut Simulation) -> &mut MissionOverrideStage {
        sim.stage_mut(Stage::ApplyMissionOverrides)
            .expect("stage 2 is installed")
    }

    fn overridden(sim: &Simulation) -> &MissionOverrideStage {
        sim.stage(Stage::ApplyMissionOverrides)
            .expect("stage 2 is installed")
    }

    fn outcome(sim: &Simulation) -> StageOutcome {
        sim.trace()
            .records()
            .iter()
            .rev()
            .find(|r| r.stage == Stage::ApplyMissionOverrides)
            .expect("stage 2 ran")
            .outcome
            .clone()
    }

    fn position_of(sim: &Simulation, node_id: &str) -> Vector3<f64> {
        sim.world()
            .platform(node_id)
            .expect("platform is in the world")
            .position
    }

    fn velocity_of(sim: &Simulation, node_id: &str) -> Vector3<f64> {
        sim.world()
            .platform(node_id)
            .expect("platform is in the world")
            .velocity
    }

    // -----------------------------------------------------------------
    // The trajectory itself
    // -----------------------------------------------------------------

    /// `_make_goto_hover` lerps position but reports a *constant* velocity, so
    /// halfway through the flight the drone is halfway there and still moving
    /// at the average speed rather than at some eased fraction of it.
    #[test]
    fn velocity_is_constant_through_the_flight_while_position_lerps() {
        let traj = GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 50.0, 10.0, 20.0);

        let (quarter_pos, quarter_vel) = traj.sample(12.5);
        let (half_pos, half_vel) = traj.sample(15.0);

        assert_eq!(quarter_pos, [25.0, 0.0, 12.5]);
        assert_eq!(half_pos, [50.0, 0.0, 25.0]);
        // 100 m and 50 m over a 10 s interval, whatever the sample time.
        assert_eq!(quarter_vel, [10.0, 0.0, 5.0]);
        assert_eq!(half_vel, quarter_vel);
    }

    /// On arrival the reference returns the target and `zeros(3)` from a
    /// separate branch. Evaluating the lerp at `frac == 1` would land on the
    /// target too, but the velocity would still be the flight velocity.
    #[test]
    fn arrival_is_an_exact_stop_not_the_lerp_evaluated_at_one() {
        let traj = GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 50.0, 10.0, 20.0);

        let (at_arrival, vel_at_arrival) = traj.sample(20.0);
        let (long_after, vel_long_after) = traj.sample(1_000.0);

        assert_eq!(at_arrival, [100.0, 0.0, 50.0]);
        assert_eq!(vel_at_arrival, [0.0, 0.0, 0.0]);
        assert_eq!(long_after, at_arrival);
        assert_eq!(vel_long_after, [0.0, 0.0, 0.0]);
    }

    /// A quirk carried across deliberately: `frac` is clamped into `[0, 1]` but
    /// the velocity is computed from the interval alone, so a sample taken
    /// before the flight starts reports the start position at full speed.
    /// Nothing samples early today, and this pins that the clamp was not
    /// "fixed" into a zero velocity while porting.
    #[test]
    fn sampling_before_the_start_holds_position_but_not_velocity() {
        let traj = GotoHover::over([7.0, 8.0, 9.0], [100.0, 0.0], 50.0, 10.0, 20.0);

        let (position, velocity) = traj.sample(0.0);

        assert_eq!(position, [7.0, 8.0, 9.0]);
        assert_eq!(velocity, [9.3, -0.8, 4.100_000_000_000_000_5]);
    }

    /// `_make_redirect_goto` measures the distance in the XY plane only, so a
    /// redirect that also climbs 500 m takes exactly as long as one that does
    /// not climb at all.
    #[test]
    fn a_redirects_flight_time_ignores_the_climb() {
        let flat = GotoHover::redirect([0.0, 0.0, 80.0], [280.0, 0.0], 80.0, 0.0, 28.0);
        let climbing = GotoHover::redirect([0.0, 0.0, 80.0], [280.0, 0.0], 580.0, 0.0, 28.0);

        assert_eq!(flat.t_arrive(), 10.0);
        assert_eq!(climbing.t_arrive(), 10.0);
    }

    /// `max(dist / speed, 1.0e-3)`. A drone standing on its own target still
    /// gets a millisecond, which is what keeps the interval division finite.
    #[test]
    fn a_zero_length_redirect_still_gets_the_minimum_flight_time() {
        let traj = GotoHover::redirect([50.0, 50.0, 80.0], [50.0, 50.0], 120.0, 4.0, 28.0);

        assert_eq!(traj.t_arrive(), 4.0 + MIN_FLIGHT_TIME_S);
        // And it is finite, not a NaN from dividing by a zero-length interval.
        let (position, velocity) = traj.sample(4.0);
        assert!(position.iter().all(|v| v.is_finite()));
        assert!(velocity.iter().all(|v| v.is_finite()));
    }

    /// The flight time scales with the speed asked for. Every other test in
    /// this file happens to pass `REDIRECT_SPEED_MPS`, so a `redirect` that
    /// ignored its argument and used the constant would agree with all of them.
    #[test]
    fn a_redirects_flight_time_scales_with_the_speed_it_was_given() {
        let slow = GotoHover::redirect([0.0, 0.0, 80.0], [280.0, 0.0], 80.0, 0.0, 14.0);
        let nominal = GotoHover::redirect([0.0, 0.0, 80.0], [280.0, 0.0], 80.0, 0.0, 28.0);

        assert_eq!(slow.t_arrive(), 20.0);
        assert_eq!(nominal.t_arrive(), 10.0);
    }

    /// `max(ta - ts, 1e-9)` guards the one division that can see a zero-length
    /// interval: a trajectory whose start and arrival coincide, sampled
    /// *before* it starts — early enough that the arrival branch does not fire.
    /// Without the guard the velocity is an infinity, and a platform flown to
    /// an infinite position never comes back.
    #[test]
    fn a_zero_length_interval_sampled_early_stays_finite() {
        let traj = GotoHover::over([1.0, 2.0, 3.0], [100.0, 200.0], 300.0, 0.0, 0.0);

        let (position, velocity) = traj.sample(-1.0);

        assert_eq!(position, [1.0, 2.0, 3.0]);
        assert!(
            velocity.iter().all(|v| v.is_finite()),
            "velocity was {velocity:?}"
        );
    }

    // -----------------------------------------------------------------
    // Precedence
    // -----------------------------------------------------------------

    /// The reference rebuilds `node_states` three times in a row, so the last
    /// table to name a platform is the one that flies it. Egress beats POI
    /// beats frontier, and the transitive case matters: a drone in all three
    /// tables is flown by egress, not by POI.
    #[test]
    fn egress_beats_poi_beats_frontier() {
        let mut sim = rig(vec![node("all-three", 0.0, 0.0), node("two", 0.0, 0.0)]);
        // Each trajectory has already arrived at a distinct place, so the
        // resulting position names the winner without ambiguity.
        ov(&mut sim).set_frontier(
            "all-three",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );
        ov(&mut sim).set_poi(
            "all-three",
            GotoHover::over([0.0, 0.0, 0.0], [200.0, 0.0], 20.0, 0.0, 0.0),
        );
        ov(&mut sim).set_egress(
            "all-three",
            GotoHover::over([0.0, 0.0, 0.0], [300.0, 0.0], 30.0, 0.0, 0.0),
        );
        ov(&mut sim).set_frontier(
            "two",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );
        ov(&mut sim).set_poi(
            "two",
            GotoHover::over([0.0, 0.0, 0.0], [200.0, 0.0], 20.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");

        assert_eq!(position_of(&sim, "all-three").x, 300.0);
        assert_eq!(
            overridden(&sim).applied_source("all-three"),
            Some(OverrideSource::Egress)
        );
        assert_eq!(position_of(&sim, "two").x, 200.0);
        assert_eq!(
            overridden(&sim).applied_source("two"),
            Some(OverrideSource::Poi)
        );
    }

    /// An override replaces both position and velocity. Writing only the
    /// position would leave the platform reporting the controller's heading
    /// while flying the trajectory's path.
    #[test]
    fn an_override_replaces_the_velocity_too() {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_poi(
            "d0",
            GotoHover::over([0.0, 0.0, 80.0], [100.0, 0.0], 80.0, 0.0, 10.0),
        );

        assert_eq!(velocity_of(&sim, "d0"), Vector3::new(1.0, 2.0, 3.0));
        sim.step().expect("tick runs");

        assert_eq!(velocity_of(&sim, "d0"), Vector3::new(10.0, 0.0, 0.0));
    }

    /// A platform named by no table is not touched at all — not reset, not
    /// zeroed. The stage overrides; it does not own the platform.
    #[test]
    fn a_platform_in_no_table_is_left_alone() {
        let mut sim = rig(vec![node("flown", 0.0, 0.0), node("untouched", 5.0, 6.0)]);
        ov(&mut sim).set_poi(
            "flown",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");

        assert_eq!(position_of(&sim, "untouched"), Vector3::new(5.0, 6.0, 80.0));
        assert_eq!(velocity_of(&sim, "untouched"), Vector3::new(1.0, 2.0, 3.0));
        assert_eq!(overridden(&sim).applied_source("untouched"), None);
        assert_eq!(overridden(&sim).applied_count(), 1);
    }

    /// The reference looks the trajectory up per platform in `node_states`, so
    /// an entry for a platform that is not in the world is simply never read.
    /// It is not an error, and it must not stop the platforms that are.
    #[test]
    fn an_entry_for_an_absent_platform_is_ignored_rather_than_failing() {
        let mut sim = rig(vec![node("present", 0.0, 0.0)]);
        ov(&mut sim).set_poi(
            "ghost",
            GotoHover::over([0.0, 0.0, 0.0], [999.0, 0.0], 10.0, 0.0, 0.0),
        );
        ov(&mut sim).set_poi(
            "present",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::Ran);
        assert_eq!(position_of(&sim, "present").x, 100.0);
        assert_eq!(overridden(&sim).applied_count(), 1);
    }

    // -----------------------------------------------------------------
    // Outcomes
    // -----------------------------------------------------------------

    #[test]
    fn three_empty_tables_are_no_work() {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// Non-empty tables that reach no platform are still `NoWork`: the stage
    /// ran and changed nothing, which is a different fact from having nothing
    /// to consider.
    #[test]
    fn tables_that_reach_no_platform_are_no_work_too() {
        let mut sim = rig(vec![node("present", 0.0, 0.0)]);
        ov(&mut sim).set_poi(
            "ghost",
            GotoHover::over([0.0, 0.0, 0.0], [999.0, 0.0], 10.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");

        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// `_egress_trajectories.clear()` — the reference drops every recall at
    /// once when the egress phase ends, and the drone falls back to whatever
    /// lower-precedence table still names it.
    #[test]
    fn clearing_every_egress_route_falls_back_to_the_poi_table() {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_poi(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [200.0, 0.0], 20.0, 0.0, 0.0),
        );
        ov(&mut sim).set_egress(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [300.0, 0.0], 30.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, 300.0);

        ov(&mut sim).clear_all_egress();
        sim.step().expect("tick runs");

        assert_eq!(position_of(&sim, "d0").x, 200.0);
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Poi)
        );
    }

    /// Each clearer touches its own table and no other. Three tables behind
    /// six near-identical accessors is exactly the shape where a copy-paste
    /// reaches into the wrong `HashMap`, and nothing else here would notice.
    #[test]
    fn each_clearer_touches_only_its_own_table() {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);
        let frontier = GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0);
        let poi = GotoHover::over([0.0, 0.0, 0.0], [200.0, 0.0], 20.0, 0.0, 0.0);
        let egress = GotoHover::over([0.0, 0.0, 0.0], [300.0, 0.0], 30.0, 0.0, 0.0);
        ov(&mut sim).set_frontier("d0", frontier);
        ov(&mut sim).set_poi("d0", poi);
        ov(&mut sim).set_egress("d0", egress);

        ov(&mut sim).clear_egress("d0");
        sim.step().expect("tick runs");
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Poi)
        );

        ov(&mut sim).clear_poi("d0");
        sim.step().expect("tick runs");
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Frontier)
        );
        // The frontier entry survived both clears.
        assert_eq!(
            overridden(&sim).frontier_target("d0"),
            Some([100.0, 0.0, 10.0])
        );

        ov(&mut sim).clear_frontier("d0");
        sim.step().expect("tick runs");
        assert_eq!(overridden(&sim).applied_source("d0"), None);
        assert_eq!(overridden(&sim).frontier_target("d0"), None);
    }

    // -----------------------------------------------------------------
    // The frontier phase gate
    // -----------------------------------------------------------------

    /// Drives the mission machine through stage 9 so the phase stage 2 reads is
    /// one the machine actually committed, not one a test poked in.
    fn rig_with_mission(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());

        let mut overrides = MissionOverrideStage::new();
        let mission = MissionStage::new(Default::default());
        overrides.follow_mission_phase();

        sim.set_stage_handler(Stage::ApplyMissionOverrides, Box::new(overrides))
            .expect("stage 2 is not a transaction boundary");
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(mission))
            .expect("stage 9 is not a transaction boundary");
        sim
    }

    /// Stage 9's machine, reached through the session that owns it.
    fn machine(sim: &Simulation) -> &crate::mission::MissionStage {
        crate::mission::MissionStage::of(sim).expect("stage 9 is a mission machine")
    }

    fn machine_mut(sim: &mut Simulation) -> &mut crate::mission::MissionStage {
        crate::mission::MissionStage::of_mut(sim).expect("stage 9 is a mission machine")
    }

    /// The mission stage rejects a tick it was given no inputs for, and rejects
    /// inputs stamped for a different tick — so every step through the mission
    /// rig has to feed it, stamped now.
    fn step_with(sim: &mut Simulation, inputs: MissionInputs) {
        let timestamp_s = sim.tick() as f64;
        machine_mut(sim).set_inputs(MissionInputs {
            timestamp_s,
            ..inputs
        });
        sim.step().expect("tick runs");
    }

    /// A fleet that has done nothing: the machine keeps scanning.
    fn scanning_inputs() -> MissionInputs {
        MissionInputs {
            mobile_count: 1,
            localization_confidences: vec![0.10],
            ..MissionInputs::at(0.0)
        }
    }

    /// Everything the machine needs to leave `scanning` for `inspecting`:
    /// coverage past the gate, no gaps, and a converged fleet.
    fn converged_inputs() -> MissionInputs {
        MissionInputs {
            coverage_fraction: 0.95,
            gap_cell_fraction: Some(0.0),
            mobile_count: 1,
            localization_confidences: vec![0.99],
            ..MissionInputs::at(0.0)
        }
    }

    #[test]
    fn the_frontier_override_is_dropped_once_the_mission_leaves_localizing() {
        let mut sim = rig_with_mission(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_frontier(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );

        // Tick 1: still scanning, so the frontier override flies the drone.
        step_with(&mut sim, scanning_inputs());
        assert_eq!(machine(&sim).phase(), MissionPhase::Scanning);
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Frontier)
        );

        // Tick 2: stage 2 still sees `scanning` -- stage 9 has not run yet this
        // tick -- and only then does the machine move to `inspecting`.
        step_with(&mut sim, converged_inputs());
        assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Frontier)
        );

        // Tick 3 is the first one stage 2 sees `inspecting` on.
        step_with(&mut sim, converged_inputs());
        assert_eq!(overridden(&sim).applied_source("d0"), None);
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// The gate is on the frontier table alone. A POI override keeps flying
    /// through every phase, because the reference's POI block has no gate.
    #[test]
    fn the_gate_is_on_the_frontier_table_only() {
        let mut sim = rig_with_mission(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_poi(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [200.0, 0.0], 20.0, 0.0, 0.0),
        );

        step_with(&mut sim, converged_inputs());
        step_with(&mut sim, converged_inputs());

        assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);
        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Poi)
        );
    }

    /// `localizing` is inside the gate. The reference says why in its own
    /// comment: dropping the override there would snap the drone back to its
    /// lawnmower trajectory mid-localization.
    #[test]
    fn the_localizing_phase_keeps_the_frontier_override() {
        let mut sim = rig_with_mission(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_frontier(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );

        // Coverage and gaps pass, but the fleet has not converged, so
        // `scanning` goes to `localizing` rather than `inspecting`.
        let unconverged = MissionInputs {
            coverage_fraction: 0.95,
            gap_cell_fraction: Some(0.0),
            mobile_count: 1,
            localization_confidences: vec![0.10],
            ..MissionInputs::at(0.0)
        };
        step_with(&mut sim, unconverged.clone());
        assert_eq!(machine(&sim).phase(), MissionPhase::Localizing);

        step_with(&mut sim, unconverged);

        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Frontier)
        );
    }

    /// An unwatched link leaves the gate open, because a run with no mission
    /// machine has no phase to gate on. Pinned so that acquiring phases without
    /// acquiring a link fails here rather than silently flying drones through
    /// egress.
    #[test]
    fn an_unwatched_link_leaves_the_frontier_gate_open() {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);
        ov(&mut sim).set_frontier(
            "d0",
            GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0),
        );

        sim.step().expect("tick runs");

        assert_eq!(
            overridden(&sim).applied_source("d0"),
            Some(OverrideSource::Frontier)
        );
    }

    /// The gate suppresses the frontier pass; it does not empty the table. The
    /// same trajectory flies again when the phase comes back into range, which
    /// is what makes the gate a gate rather than a delete.
    #[test]
    fn a_gated_frontier_trajectory_survives_to_fly_again() {
        let mut sim = rig_with_mission(vec![node("d0", 0.0, 0.0)]);
        let trajectory = GotoHover::over([0.0, 0.0, 0.0], [100.0, 0.0], 10.0, 0.0, 0.0);
        ov(&mut sim).set_frontier("d0", trajectory);

        step_with(&mut sim, converged_inputs());
        step_with(&mut sim, converged_inputs());
        step_with(&mut sim, converged_inputs());
        assert_eq!(machine(&sim).phase(), MissionPhase::Inspecting);
        assert_eq!(overridden(&sim).applied_source("d0"), None);
        // Suppressed, not discarded.
        assert_eq!(
            overridden(&sim).frontier_target("d0"),
            Some([100.0, 0.0, 10.0])
        );
    }
}
