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

use std::collections::{BTreeMap, HashMap};

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
///
/// A routed flight (P3, `efficient_v1` only) also carries the corners of a
/// route around obstacles, flown at constant speed along the polyline. A
/// flight with no corners is exactly the reference's, and serialises exactly
/// as it did, so every existing checkpoint reads back unchanged. A routed
/// flight that climbs over a power line (R3) also carries the height at each
/// corner; one that does not serialises as it did.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct GotoHover {
    start: [f64; 3],
    target_xy: [f64; 2],
    target_z: f64,
    t_start: f64,
    t_arrive: f64,
    /// The route's corners between `start` and the target, in order. Empty
    /// for a straight flight.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    via_xy: Vec<[f64; 2]>,
    /// The height at each corner, when the flight climbs over a power line
    /// (R3): one per entry of `via_xy`. Empty when the height changes
    /// linearly over the whole flight.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    via_z: Vec<f64>,
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
            via_xy: Vec::new(),
            via_z: Vec::new(),
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

    /// A flight along a planned route at a speed (P3).
    ///
    /// `via_xy` are the route's corners between `start` and the target. The
    /// flight takes the polyline's length at `speed_mps`, floored like
    /// [`Self::redirect`], and the height changes linearly over the whole
    /// flight, as a straight redirect's does. With no corners this is
    /// exactly [`Self::redirect`].
    pub fn routed(
        start: [f64; 3],
        via_xy: Vec<[f64; 2]>,
        target_xy: [f64; 2],
        target_z: f64,
        t_start: f64,
        speed_mps: f64,
    ) -> Self {
        if via_xy.is_empty() {
            return Self::redirect(start, target_xy, target_z, t_start, speed_mps);
        }
        let mut flight = Self::over(start, target_xy, target_z, t_start, t_start);
        flight.via_xy = via_xy;
        let flight_s = (flight.path_length_m() / speed_mps.max(EPSILON_S)).max(MIN_FLIGHT_TIME_S);
        flight.t_arrive = t_start + flight_s;
        flight
    }

    /// [`Self::routed`], with the height at each corner (R3), so a route can
    /// climb over a power line and come down after it. The height changes
    /// linearly along each leg, from the start's to the first corner's and
    /// on to the target's; the flight time is still the horizontal length's,
    /// as every redirect's is. With no heights this is exactly
    /// [`Self::routed`].
    ///
    /// # Panics
    ///
    /// If `via_z` is neither empty nor one height per corner.
    pub fn routed_over(
        start: [f64; 3],
        via_xy: Vec<[f64; 2]>,
        via_z: Vec<f64>,
        target_xy: [f64; 2],
        target_z: f64,
        t_start: f64,
        speed_mps: f64,
    ) -> Self {
        assert!(
            via_z.is_empty() || via_z.len() == via_xy.len(),
            "one height per corner"
        );
        let mut flight = Self::routed(start, via_xy, target_xy, target_z, t_start, speed_mps);
        flight.via_z = via_z;
        flight
    }

    /// The horizontal points flown through: the start, any corners, the
    /// target.
    fn path_xy(&self) -> Vec<[f64; 2]> {
        let mut points = Vec::with_capacity(self.via_xy.len() + 2);
        points.push([self.start[0], self.start[1]]);
        points.extend(self.via_xy.iter().copied());
        points.push(self.target_xy);
        points
    }

    /// The horizontal length of the flight: the route's, or the straight
    /// line's when it has no corners.
    pub fn path_length_m(&self) -> f64 {
        self.path_xy()
            .windows(2)
            .map(|pair| {
                let (dx, dy) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
                (dx * dx + dy * dy).sqrt()
            })
            .sum()
    }

    /// The route's corners between the start and the target; empty for a
    /// straight flight.
    pub fn via_xy(&self) -> &[[f64; 2]] {
        &self.via_xy
    }

    /// The height at each corner; empty when the height changes linearly
    /// over the whole flight.
    pub fn via_z(&self) -> &[f64] {
        &self.via_z
    }

    /// Position and velocity at `t`.
    pub fn sample(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        if t >= self.t_arrive {
            return (
                [self.target_xy[0], self.target_xy[1], self.target_z],
                [0.0, 0.0, 0.0],
            );
        }
        if !self.via_xy.is_empty() {
            return self.sample_route(t);
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

    /// [`Self::sample`] for a routed flight, before arrival: the same
    /// fraction of the flight as a straight one, measured along the polyline,
    /// at the constant speed that covers it in the flight time. The velocity
    /// points along the leg being flown. Before `t_start` the position clamps
    /// to the start and the velocity is the first leg's, the straight
    /// flight's quirk carried over. With heights at the corners, the height
    /// and its rate are the leg's own.
    fn sample_route(&self, t: f64) -> ([f64; 3], [f64; 3]) {
        let span = (self.t_arrive - self.t_start).max(EPSILON_S);
        let frac = ((t - self.t_start) / span).clamp(0.0, 1.0);
        let points = self.path_xy();
        let total_m = self.path_length_m();
        let speed = total_m / span;
        let dz = self.target_z - self.start[2];
        let z = self.start[2] + frac * dz;
        let vz = dz / span;
        let mut left_m = frac * total_m;
        let last = points.len() - 2;
        for (index, pair) in points.windows(2).enumerate() {
            let (dx, dy) = (pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]);
            let leg_m = (dx * dx + dy * dy).sqrt();
            if leg_m <= 0.0 {
                continue;
            }
            // A corner belongs to the leg it starts, so a sample taken on
            // it flies the outgoing heading; only the target ends a leg.
            if left_m < leg_m || index == last {
                let along = (left_m / leg_m).min(1.0);
                let (ux, uy) = (dx / leg_m, dy / leg_m);
                if !self.via_z.is_empty() {
                    let height = |i: usize| match i {
                        0 => self.start[2],
                        i if i > self.via_z.len() => self.target_z,
                        i => self.via_z[i - 1],
                    };
                    let (z0, z1) = (height(index), height(index + 1));
                    return (
                        [
                            pair[0][0] + along * dx,
                            pair[0][1] + along * dy,
                            z0 + along * (z1 - z0),
                        ],
                        [ux * speed, uy * speed, (z1 - z0) / leg_m * speed],
                    );
                }
                return (
                    [pair[0][0] + along * dx, pair[0][1] + along * dy, z],
                    [ux * speed, uy * speed, vz],
                );
            }
            left_m -= leg_m;
        }
        // Only rounding past the end of the last leg with any length gets
        // here (the corners are never all on the start, since `routed` keeps
        // corners only from a route that turns).
        let z = if self.via_z.is_empty() {
            z
        } else {
            self.target_z
        };
        ([self.target_xy[0], self.target_xy[1], z], [0.0, 0.0, vz])
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
    /// Rejoin a platform's own path smoothly when its override ends, under a
    /// planning policy other than the reference (set at assembly).
    smooth_rejoin: bool,
    /// Where each platform an override flew last run was put, so that a
    /// platform whose override has ended starts its rejoin there.
    overridden_at: BTreeMap<String, [f64; 3]>,
    /// Platforms flying back to their own path, and where each is now.
    rejoining: BTreeMap<String, [f64; 3]>,
    /// Rejoins climbing over a structure in their way, and the height each
    /// holds until it is above its path again.
    climbing: BTreeMap<String, f64>,
    /// Where each rejoin's path was last tick, so how fast it is running
    /// away is measured from the path itself, not from a velocity stage 1
    /// may not have refreshed.
    rejoin_paths: BTreeMap<String, [f64; 3]>,
    /// The mission's redirect speed, which rejoins fly at too;
    /// [`REDIRECT_SPEED_MPS`] when none is set.
    rejoin_speed_mps: Option<f64>,
    /// Keeps each rejoin step out of the ground and out of structures: the
    /// way back to a path can cross what lies between.
    guard: Option<crate::belief_world::PhysicalGuard>,
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
    /// A gap-filling flight (P4, `efficient_v1` only): flown from the
    /// frontier table, so only while scanning or localizing, and beneath any
    /// POI or egress flight.
    SetFrontier(String, GotoHover),
    ClearFrontier(String),
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

    /// Under a planning policy other than the reference, a platform whose
    /// override ends (a POI released, a recall dropped, a gap filled) flies
    /// back to where its own path has it, at the redirect speed, instead
    /// of appearing there: the reference moves it the whole way in one tick.
    /// Off (the default) is the reference exactly. Before the first tick.
    pub fn set_smooth_rejoin(&mut self, smooth: bool) {
        self.state.smooth_rejoin = smooth;
        if !smooth {
            self.state.overridden_at.clear();
            self.state.rejoining.clear();
            self.state.climbing.clear();
            self.state.rejoin_paths.clear();
        }
    }

    /// The guard every rejoin step goes through (the one deconfliction uses);
    /// `None` leaves the steps unguarded.
    pub fn set_physical_guard(&mut self, guard: Option<crate::belief_world::PhysicalGuard>) {
        self.state.guard = guard;
    }

    /// Where `node_id` is on its way back to its own path, if it is.
    /// The speed rejoins fly at: the mission's redirect speed, so a drone
    /// flies back to its path as fast as its overrides flew it. Before the
    /// first tick.
    pub fn set_rejoin_speed(&mut self, speed_mps: f64) {
        self.state.rejoin_speed_mps = Some(speed_mps.max(f64::EPSILON));
    }

    pub fn rejoining_at(&self, node_id: &str) -> Option<[f64; 3]> {
        self.state.rejoining.get(node_id).copied()
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
            OverrideCommand::SetFrontier(node_id, trajectory) => {
                self.set_frontier(&node_id, trajectory)
            }
            OverrideCommand::ClearFrontier(node_id) => self.clear_frontier(&node_id),
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
    /// Smooth rejoins (a policy other than the reference): absent when
    /// empty, so every other checkpoint is unchanged.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    overridden_at: BTreeMap<String, [f64; 3]>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    rejoining: BTreeMap<String, [f64; 3]>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    climbing: BTreeMap<String, f64>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    rejoin_paths: BTreeMap<String, [f64; 3]>,
}

/// The least share of its speed a rejoin must close on its path by: a path
/// receding faster than the rest is not chased.
const REJOIN_MIN_CLOSING: f64 = 0.05;

/// How far short of a structure a step that would meet it stops, metres.
const REJOIN_STANDOFF_M: f64 = 0.5;

/// `from` toward `to`, kept out of structures and the ground. A building or
/// wall the step would meet, however thin, stops it [`REJOIN_STANDOFF_M`]
/// short (the segment is intersected with each, not sampled); so does the
/// first point within the guard's clearance of the ground, solved on the
/// terrain's cells. A step nothing stops ends at `to`.
fn guarded_step(
    guard: &crate::belief_world::PhysicalGuard,
    node_id: &str,
    timestamp_s: f64,
    from: [f64; 3],
    to: [f64; 3],
) -> [f64; 3] {
    let length = distance_3d(from, to);
    match guard.first_structure_hit(from, to) {
        Some(t) if length > 0.0 => {
            let f = ((t * length - REJOIN_STANDOFF_M) / length).max(0.0);
            let stop = [
                from[0] + (to[0] - from[0]) * f,
                from[1] + (to[1] - from[1]) * f,
                from[2] + (to[2] - from[2]) * f,
            ];
            // Stopped short of the structure, and the ground checked on the
            // way there.
            guard.record_structure_stop(node_id, timestamp_s, distance_3d(stop, to));
            guarded_to(guard, node_id, timestamp_s, from, stop)
        }
        _ => guarded_to(guard, node_id, timestamp_s, from, to),
    }
}

/// `from` toward `to`, stopped where it first comes within the guard's
/// clearance of the ground, then put through the guard.
fn guarded_to(
    guard: &crate::belief_world::PhysicalGuard,
    node_id: &str,
    timestamp_s: f64,
    from: [f64; 3],
    to: [f64; 3],
) -> [f64; 3] {
    let stop = match guard.first_ground_contact(from, to) {
        Some(t) => {
            let stop = [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
                from[2] + (to[2] - from[2]) * t,
            ];
            guard.record_ground_stop(node_id, timestamp_s, distance_3d(stop, to));
            stop
        }
        None => to,
    };
    guard.enforce(node_id, timestamp_s, stop)
}

fn distance_3d(a: [f64; 3], b: [f64; 3]) -> f64 {
    ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2) + (b[2] - a[2]).powi(2)).sqrt()
}

/// `step_m` from `at` toward `to`, held at or above `floor_z` when one is
/// set (the horizontal part of the step is what the floor leaves of it).
fn step_toward(at: [f64; 3], to: [f64; 3], step_m: f64, floor_z: Option<f64>) -> [f64; 3] {
    let Some(floor_z) = floor_z else {
        let distance = distance_3d(at, to);
        let unit = [
            (to[0] - at[0]) / distance,
            (to[1] - at[1]) / distance,
            (to[2] - at[2]) / distance,
        ];
        return [
            at[0] + unit[0] * step_m,
            at[1] + unit[1] * step_m,
            at[2] + unit[2] * step_m,
        ];
    };
    let z = to[2].max(floor_z);
    let rise = (z - at[2]).clamp(-step_m, step_m);
    let across = (step_m * step_m - rise * rise).sqrt();
    let horizontal = (to[0] - at[0]).hypot(to[1] - at[1]);
    let k = if horizontal > 0.0 {
        (across / horizontal).min(1.0)
    } else {
        0.0
    };
    [
        at[0] + (to[0] - at[0]) * k,
        at[1] + (to[1] - at[1]) * k,
        at[2] + rise,
    ]
}

/// Smooth rejoins, after this run's overrides are applied: a platform an
/// override flew last run and none flies now starts flying back from where it
/// was put, toward where its own path (sampled by stage 1 this tick) has it,
/// at the redirect speed. Within one step of it, it is on it again, and
/// the rejoin ends. A platform an override takes again stops rejoining.
/// Returns how many platforms are still rejoining.
fn rejoin(shared: &mut State, context: &mut TickContext<'_>) -> usize {
    let ended: Vec<(String, [f64; 3])> = shared
        .overridden_at
        .iter()
        .filter(|(node_id, _)| !shared.applied.contains_key(*node_id))
        .map(|(node_id, at)| (node_id.clone(), *at))
        .collect();
    for (node_id, at) in ended {
        shared.rejoining.entry(node_id).or_insert(at);
    }
    shared
        .rejoining
        .retain(|node_id, _| !shared.applied.contains_key(node_id));
    let rejoining = &shared.rejoining;
    shared
        .climbing
        .retain(|node_id, _| rejoining.contains_key(node_id));
    shared
        .rejoin_paths
        .retain(|node_id, _| rejoining.contains_key(node_id));
    let speed_mps = shared.rejoin_speed_mps.unwrap_or(REDIRECT_SPEED_MPS);
    let step_m = speed_mps * context.dt_s.max(0.0);
    let mut done = Vec::new();
    for (node_id, at) in shared.rejoining.iter_mut() {
        let Some(platform) = context.world.platform_mut(node_id) else {
            done.push(node_id.clone());
            continue;
        };
        let path = [
            platform.position.x,
            platform.position.y,
            platform.position.z,
        ];
        let distance = distance_3d(*at, path);
        let last_path = shared.rejoin_paths.insert(node_id.clone(), path);
        // A path running away from it at the rejoin speed or faster is never
        // caught: it takes its place on its path, as the reference has every
        // platform do, once nothing stands between. How fast the path runs
        // is measured from where it was last tick.
        if let (Some(last), true) = (
            last_path,
            distance > step_m && !shared.climbing.contains_key(node_id) && context.dt_s > 0.0,
        ) {
            let away = [
                (path[0] - at[0]) / distance,
                (path[1] - at[1]) / distance,
                (path[2] - at[2]) / distance,
            ];
            let receding = ((path[0] - last[0]) * away[0]
                + (path[1] - last[1]) * away[1]
                + (path[2] - last[2]) * away[2])
                / context.dt_s;
            let clear = shared.guard.as_ref().is_none_or(|guard| {
                guard.first_structure_hit(*at, path).is_none()
                    && guard.first_ground_contact(*at, path).is_none()
            });
            if receding >= (1.0 - REJOIN_MIN_CLOSING) * speed_mps && clear {
                done.push(node_id.clone());
                continue;
            }
        }
        // Within a step of its path: the last step is onto it. Unguarded,
        // stage 1's sample simply stands; guarded, that step is checked like
        // any other, and the rejoin ends only once it gets there.
        let last_step = distance <= step_m;
        if last_step && shared.guard.is_none() {
            done.push(node_id.clone());
            continue;
        }
        // Climbing over a structure: hold the height reached until above
        // the path, then come down to it.
        let horizontal = (path[0] - at[0]).hypot(path[1] - at[1]);
        if horizontal <= step_m {
            shared.climbing.remove(node_id);
        }
        let floor_z = shared.climbing.get(node_id).copied();
        let mut next = if last_step {
            path
        } else {
            step_toward(*at, path, step_m, floor_z)
        };
        if let Some(guard) = &shared.guard {
            let tentative = next;
            next = guarded_step(guard, node_id, context.timestamp_s, *at, tentative);
            // Pushed back out of a structure on the direct line, it would
            // aim through the same one next tick: climb over it instead. (A
            // path that merely runs ahead of it is not a stall: the guard
            // did not move it.)
            let pushed = distance_3d(next, tentative) > 1e-9;
            if last_step && !pushed {
                // On its path again: stage 1's sample stands.
                done.push(node_id.clone());
                continue;
            }
            if pushed && distance_3d(next, path) >= distance - 0.25 * step_m {
                let up = guarded_step(
                    guard,
                    node_id,
                    context.timestamp_s,
                    *at,
                    [at[0], at[1], at[2] + step_m],
                );
                if up[2] > at[2] + 0.5 * step_m {
                    shared.climbing.insert(node_id.clone(), up[2]);
                    next = up;
                } else {
                    // It cannot climb (under something overhead): it steps
                    // sideways along what blocks it, to its left first, and
                    // tries the direct line again from there next tick. With
                    // no way sideways either, it holds where it is, still
                    // rejoining; it never jumps through to its path.
                    shared.climbing.remove(node_id);
                    let (dx, dy) = (path[0] - at[0], path[1] - at[1]);
                    let across = dx.hypot(dy);
                    next = *at;
                    if across > 1e-9 {
                        for side in [1.0, -1.0] {
                            let aside = [
                                at[0] - side * dy / across * step_m,
                                at[1] + side * dx / across * step_m,
                                at[2],
                            ];
                            let stepped =
                                guarded_step(guard, node_id, context.timestamp_s, *at, aside);
                            if distance_3d(stepped, *at) > 0.5 * step_m {
                                next = stepped;
                                break;
                            }
                        }
                    }
                }
            }
        }
        let moved = distance_3d(*at, next).max(1e-9);
        platform.velocity = Vector3::new(
            (next[0] - at[0]) / moved * speed_mps,
            (next[1] - at[1]) / moved * speed_mps,
            (next[2] - at[2]) / moved * speed_mps,
        );
        *at = next;
        platform.position = Vector3::new(at[0], at[1], at[2]);
    }
    for node_id in done {
        shared.rejoining.remove(&node_id);
        shared.climbing.remove(&node_id);
        shared.rejoin_paths.remove(&node_id);
    }
    shared.overridden_at = shared
        .applied
        .keys()
        .filter_map(|node_id| {
            context
                .world
                .platform_mut(node_id)
                .map(|p| (node_id.clone(), [p.position.x, p.position.y, p.position.z]))
        })
        .collect();
    shared.rejoining.len()
}

impl StageHandler for MissionOverrideStage {
    fn reset(&mut self) {
        let shared = &mut self.state;
        shared.frontier.clear();
        shared.poi.clear();
        shared.egress.clear();
        shared.applied.clear();
        shared.overridden_at.clear();
        shared.rejoining.clear();
        shared.climbing.clear();
        shared.rejoin_paths.clear();
    }

    /// The redirects in force: each is a flight already under way, and a
    /// resumed run must finish it the same way.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        let state = OverrideCheckpoint {
            frontier: shared.frontier.clone().into_iter().collect(),
            poi: shared.poi.clone().into_iter().collect(),
            egress: shared.egress.clone().into_iter().collect(),
            overridden_at: shared.overridden_at.clone(),
            rejoining: shared.rejoining.clone(),
            climbing: shared.climbing.clone(),
            rejoin_paths: shared.rejoin_paths.clone(),
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
        shared.overridden_at = decoded.overridden_at;
        shared.rejoining = decoded.rejoining;
        shared.climbing = decoded.climbing;
        shared.rejoin_paths = decoded.rejoin_paths;
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

        if shared.frontier.is_empty()
            && shared.poi.is_empty()
            && shared.egress.is_empty()
            && shared.overridden_at.is_empty()
            && shared.rejoining.is_empty()
        {
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

        let rejoined = if shared.smooth_rejoin {
            rejoin(shared, context)
        } else {
            0
        };

        if shared.applied.is_empty() && rejoined == 0 {
            return Ok(StageOutcome::NoWork);
        }

        let count = shared.applied.len();
        context.note(format!("overrode {count} platform(s)"));
        if rejoined > 0 {
            context.note(format!("{rejoined} platform(s) rejoining their path"));
        }
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
        Simulation::bare(SimulationConfig {
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

    /// Stage 1 standing in for a controller: puts each platform on its own
    /// path, here a fixed point, every tick.
    #[derive(Debug)]
    struct OnPath(Vec<(String, [f64; 3])>);

    impl StageHandler for OnPath {
        fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
            for (node_id, at) in &self.0 {
                if let Some(platform) = context.world.platform_mut(node_id) {
                    platform.position = Vector3::new(at[0], at[1], at[2]);
                }
            }
            Ok(StageOutcome::Ran)
        }
    }

    /// Stage 1 for one platform whose path runs west at `speed_mps` from
    /// the origin, without touching its velocity (as a controller lane that
    /// did not sample this tick would leave it).
    #[derive(Debug)]
    struct Receding {
        speed_mps: f64,
        x_m: f64,
    }

    impl StageHandler for Receding {
        fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
            self.x_m -= self.speed_mps * context.dt_s;
            if let Some(platform) = context.world.platform_mut("d0") {
                platform.position = Vector3::new(self.x_m, 0.0, 80.0);
            }
            Ok(StageOutcome::Ran)
        }
    }

    /// A path running away faster than the rejoin speed is not chased for
    /// ever: the drone takes its place on it, as the reference does. One
    /// running away slower is caught.
    #[test]
    fn a_rejoin_does_not_chase_a_path_it_cannot_catch() {
        for (speed_mps, caught_by_flight) in [(40.0, false), (10.0, true)] {
            let mut sim = rejoin_rig(true);
            sim.set_stage_handler(
                Stage::SampleControllersAndBattery,
                Box::new(Receding {
                    speed_mps,
                    x_m: 0.0,
                }),
            )
            .expect("stage 1 takes a handler");
            let mut ticks = 0;
            let mut flown_ticks = 0;
            loop {
                sim.step().expect("tick runs");
                ticks += 1;
                assert!(ticks < 500, "still chasing at {speed_mps} m/s");
                if overridden(&sim).rejoining_at("d0").is_none() {
                    break;
                }
                flown_ticks += 1;
            }
            let path_x = -speed_mps * ticks as f64;
            assert_eq!(position_of(&sim, "d0").x, path_x, "on its path");
            if caught_by_flight {
                assert!(flown_ticks > 30, "{flown_ticks}");
            } else {
                assert!(flown_ticks <= 2, "{flown_ticks}");
            }
        }
    }

    fn rejoin_rig(smooth: bool) -> Simulation {
        rejoin_rig_at(smooth, 1000.0)
    }

    /// [`rejoin_rig`], with the POI flight holding the drone `east_m` east
    /// of its path.
    fn rejoin_rig_at(smooth: bool, east_m: f64) -> Simulation {
        let mut sim = rig(vec![node("d0", 0.0, 0.0)]);
        sim.set_stage_handler(
            Stage::SampleControllersAndBattery,
            Box::new(OnPath(vec![("d0".to_string(), [0.0, 0.0, 80.0])])),
        )
        .expect("stage 1 takes a handler");
        ov(&mut sim).set_smooth_rejoin(smooth);
        // A POI flight that holds the drone 1 km east of its path.
        ov(&mut sim).set_poi(
            "d0",
            GotoHover::over([east_m, 0.0, 80.0], [east_m, 0.0], 80.0, 0.0, 0.0),
        );
        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, east_m);
        ov(&mut sim).clear_poi("d0");
        sim
    }

    /// The reference puts a platform whose override has ended straight back
    /// on its path, however far away that is.
    #[test]
    fn the_reference_returns_a_released_platform_to_its_path_at_once() {
        let mut sim = rejoin_rig(false);
        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, 0.0);
        assert!(overridden(&sim)
            .checkpoint_state()
            .unwrap()
            .get("rejoining")
            .is_none());
    }

    /// With smooth rejoins it flies back at the redirect speed, is on its
    /// path again once within a step of it, and a checkpoint taken on the way
    /// resumes the same flight.
    #[test]
    fn a_smooth_rejoin_flies_back_to_the_path() {
        let mut sim = rejoin_rig(true);
        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, 1000.0 - REDIRECT_SPEED_MPS);
        assert_eq!(velocity_of(&sim, "d0").x, -REDIRECT_SPEED_MPS);
        assert_eq!(
            overridden(&sim).rejoining_at("d0").map(|p| p[0]),
            Some(972.0)
        );

        let state = overridden(&sim).checkpoint_state().unwrap();
        assert!(state.get("rejoining").is_some());
        let mut resumed = rejoin_rig(true);
        ov(&mut resumed).restore_state(&state).unwrap();

        let ticks = (1000.0 / REDIRECT_SPEED_MPS).ceil() as usize;
        for _ in 1..ticks {
            sim.step().expect("tick runs");
            resumed.step().expect("tick runs");
            assert_eq!(position_of(&sim, "d0"), position_of(&resumed, "d0"));
            assert!(position_of(&sim, "d0").x >= 0.0);
        }
        assert_eq!(position_of(&sim, "d0").x, 0.0, "on its path again");
        assert_eq!(overridden(&sim).rejoining_at("d0"), None);
        sim.step().expect("tick runs");
        assert_eq!(outcome(&sim), StageOutcome::NoWork);
    }

    /// A rejoin never flies through a structure between where the override
    /// left the platform and its path: every step goes through the guard.
    #[test]
    fn a_rejoin_never_enters_a_building_in_its_way() {
        let block = building(
            [[400.0, -50.0], [600.0, -50.0], [600.0, 50.0], [400.0, 50.0]],
            200.0,
        );
        let (mut sim, guard) = guarded_rejoin_rig(&block);
        let ticks = (1000.0 / REDIRECT_SPEED_MPS).ceil() as usize + 20;
        for _ in 0..ticks {
            sim.step().expect("tick runs");
            let p = position_of(&sim, "d0");
            assert!(
                !block.point_inside(p.x, p.y, p.z),
                "inside the building at ({}, {}, {})",
                p.x,
                p.y,
                p.z
            );
        }
        assert!(!guard.take_contacts().is_empty(), "the guard moved it");
    }

    /// A wall too wide to be pushed around is climbed over: the rejoin does
    /// not stall at its face, and it ends on the path.
    #[test]
    fn a_rejoin_climbs_over_a_wall_it_cannot_get_around() {
        let wall = building(
            [
                [400.0, -500.0],
                [600.0, -500.0],
                [600.0, 500.0],
                [400.0, 500.0],
            ],
            150.0,
        );
        let (mut sim, guard) = guarded_rejoin_rig(&wall);
        let mut highest = 0.0_f64;
        let mut ticks = 0;
        while overridden(&sim).rejoining_at("d0").is_some() || ticks == 0 {
            sim.step().expect("tick runs");
            ticks += 1;
            assert!(ticks < 200, "the rejoin stalled");
            let p = position_of(&sim, "d0");
            assert!(!wall.point_inside(p.x, p.y, p.z), "inside the wall");
            highest = highest.max(p.z);
        }
        assert!(highest > 150.0, "it went over the wall, at {highest} m");
        assert_eq!(position_of(&sim, "d0").x, 0.0);
        assert_eq!(position_of(&sim, "d0").z, 80.0);
        assert!(
            !guard.take_contacts().is_empty(),
            "the guard stopped it first"
        );
        assert!(overridden(&sim)
            .checkpoint_state()
            .unwrap()
            .get("climbing")
            .is_none());
    }

    /// A wall thinner than one step is not stepped through: every piece of
    /// each step is checked, so the drone is stopped at the wall's face and
    /// climbs it before it is ever past it.
    #[test]
    fn a_rejoin_does_not_step_through_a_thin_wall() {
        // Thinner than a step, and thinner than any sampling of one: the
        // step is intersected with the wall, not sampled. Centred half way
        // between whole metres, where no metre-spaced sample of these steps
        // falls.
        for half_m in [1.0, 0.1] {
            let wall = building(
                [
                    [500.5 - half_m, -500.0],
                    [500.5 + half_m, -500.0],
                    [500.5 + half_m, 500.0],
                    [500.5 - half_m, 500.0],
                ],
                150.0,
            );
            let (mut sim, _) = guarded_rejoin_rig(&wall);
            let mut highest = 0.0_f64;
            let mut ticks = 0;
            while overridden(&sim).rejoining_at("d0").is_some() || ticks == 0 {
                sim.step().expect("tick runs");
                ticks += 1;
                assert!(ticks < 200, "the rejoin stalled");
                let p = position_of(&sim, "d0");
                if p.x < 500.5 - half_m {
                    assert!(
                        highest > 150.0,
                        "past the {} m wall at {} m without going over it",
                        2.0 * half_m,
                        p.z
                    );
                }
                highest = highest.max(p.z);
            }
            assert_eq!(position_of(&sim, "d0").x, 0.0);
        }
    }

    /// The ground is met exactly too: a ridge 1 m wide, 150 m high, whose
    /// crest lies half-way between whole metres, where no metre-spaced
    /// sample of these steps falls. The drone goes over it, not through.
    #[test]
    fn a_rejoin_does_not_step_through_a_narrow_ridge() {
        use crate::belief_world::PhysicalGuard;
        use crate::runtime_world::{SharedObstacles, SharedTerrain};
        use argusnet_types::Bounds2d;
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::tiled::{Tile, TiledTerrain};
        use std::sync::Arc;
        let bounds = Bounds2d::new(-200.0, 1200.0, -600.0, 600.0);
        // One tile of 8 x 8 cells of 0.5 m from x = 500, y = 0: the ridge is
        // the column of nodes at x = 500.5.
        let heights_m = (0..9)
            .map(|_| {
                (0..9)
                    .map(|col| if col == 1 { 150.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let tile = Tile {
            tx: 175,
            ty: 150,
            lod: 0,
            x_min_m: 500.0,
            y_min_m: 0.0,
            cell_size_m: 0.5,
            heights_m,
            min_pyramid: Vec::new(),
            max_pyramid: Vec::new(),
        };
        let terrain = TiledTerrain::new(bounds, 8, 0.5, 0.0, vec![tile]).unwrap();
        assert_eq!(terrain.height_lod(500.5, 0.0, 0), 150.0);
        assert_eq!(terrain.height_lod(500.0, 0.0, 0), 0.0);
        let layer = ObstacleLayer::new(bounds, 200.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 0.5),
            SharedObstacles::new(Arc::new(layer)),
            2.0,
        );
        let mut sim = rejoin_rig_at(true, 1000.0);
        ov(&mut sim).set_physical_guard(Some(guard));
        let mut highest = 0.0_f64;
        let mut ticks = 0;
        while overridden(&sim).rejoining_at("d0").is_some() || ticks == 0 {
            sim.step().expect("tick runs");
            ticks += 1;
            assert!(ticks < 200, "the rejoin stalled");
            let p = position_of(&sim, "d0");
            if p.x < 500.5 {
                assert!(highest > 150.0, "through the ridge at {} m", p.z);
            }
            highest = highest.max(p.z);
        }
        assert_eq!(position_of(&sim, "d0").x, 0.0);
    }

    /// The last step onto the path is guarded too: released 20 m from its
    /// path, within one step of it, with a thin wall between, the drone does
    /// not appear on the far side; it goes over, then onto its path.
    #[test]
    fn the_last_step_onto_the_path_does_not_pass_through_a_wall() {
        let wall = building(
            [[10.4, -500.0], [10.6, -500.0], [10.6, 500.0], [10.4, 500.0]],
            150.0,
        );
        let mut sim = rejoin_rig_at(true, 20.0);
        ov(&mut sim).set_physical_guard(Some(guard_around(&wall)));
        let mut highest = 0.0_f64;
        let mut ticks = 0;
        while overridden(&sim).rejoining_at("d0").is_some() || ticks == 0 {
            sim.step().expect("tick runs");
            ticks += 1;
            assert!(ticks < 200, "the rejoin stalled");
            let p = position_of(&sim, "d0");
            if p.x < 10.4 {
                assert!(highest > 150.0, "through the wall at {} m", p.z);
            }
            highest = highest.max(p.z);
        }
        assert_eq!(position_of(&sim, "d0").x, 0.0);
        assert!(ticks > 1, "it did not simply appear on its path");
    }

    /// Rejoins fly at the mission's redirect speed when one is set.
    #[test]
    fn a_rejoin_flies_at_the_missions_redirect_speed() {
        let mut sim = rejoin_rig(true);
        ov(&mut sim).set_rejoin_speed(10.0);
        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, 990.0);
        assert_eq!(velocity_of(&sim, "d0").x, -10.0);
    }

    fn building(corners: [[f64; 2]; 4], height_m: f64) -> argusnet_world::obstacles::Obstacle {
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        Obstacle::polygon_prism(
            "block",
            BlockerType::Building,
            corners.to_vec(),
            0.0,
            height_m,
        )
        .unwrap()
    }

    /// Under an overhang it cannot climb past, with a wall between it and
    /// its path, a rejoin does not jump through to the path: it steps
    /// sideways along the wall (never more than a step a tick, never inside
    /// either) until the way is clear, then flies home.
    #[test]
    fn a_rejoin_under_an_overhang_goes_round_not_through() {
        use crate::belief_world::PhysicalGuard;
        use crate::runtime_world::{SharedObstacles, SharedTerrain};
        use argusnet_types::Bounds2d;
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::obstacles::{BlockerType, Obstacle};
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        // A slab over the drone (100 to 200 m up) and, under it, a wall
        // between it and its path from the ground to the slab's underside:
        // nothing above to climb into, no way through.
        let slab = Obstacle::polygon_prism(
            "slab",
            BlockerType::Building,
            vec![
                [800.0, -500.0],
                [1100.0, -500.0],
                [1100.0, 500.0],
                [800.0, 500.0],
            ],
            100.0,
            200.0,
        )
        .unwrap();
        let wall = building(
            [[850.0, -50.0], [870.0, -50.0], [870.0, 50.0], [850.0, 50.0]],
            100.0,
        );
        let bounds = Bounds2d::new(-200.0, 1200.0, -600.0, 600.0);
        let layer = ObstacleLayer::new(bounds, 200.0, vec![slab.clone(), wall.clone()]).unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        let guard = PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer)),
            2.0,
        );
        let mut sim = rejoin_rig(true);
        ov(&mut sim).set_physical_guard(Some(guard));
        let step_m = REDIRECT_SPEED_MPS;
        let mut last = position_of(&sim, "d0");
        let mut ticks = 0;
        while overridden(&sim).rejoining_at("d0").is_some() || ticks == 0 {
            sim.step().expect("tick runs");
            ticks += 1;
            assert!(ticks < 400, "the rejoin stalled");
            let p = position_of(&sim, "d0");
            let moved =
                ((p.x - last.x).powi(2) + (p.y - last.y).powi(2) + (p.z - last.z).powi(2)).sqrt();
            let on_path = overridden(&sim).rejoining_at("d0").is_none();
            // The path does not move, so even the last step onto it is a
            // step: a jump to the path is caught.
            assert!(moved <= step_m + 1e-6, "jumped {moved} m at tick {ticks}");
            if !on_path {
                for o in [&slab, &wall] {
                    assert!(!o.point_inside(p.x, p.y, p.z), "inside {} at {p:?}", o.id);
                }
            }
            last = p;
        }
        assert_eq!(position_of(&sim, "d0").x, 0.0);
    }

    fn guarded_rejoin_rig(
        block: &argusnet_world::obstacles::Obstacle,
    ) -> (Simulation, crate::belief_world::PhysicalGuard) {
        let guard = guard_around(block);
        let mut sim = rejoin_rig(true);
        ov(&mut sim).set_physical_guard(Some(guard.clone()));
        (sim, guard)
    }

    /// A guard over flat ground with `block` on it.
    fn guard_around(
        block: &argusnet_world::obstacles::Obstacle,
    ) -> crate::belief_world::PhysicalGuard {
        use crate::belief_world::PhysicalGuard;
        use crate::runtime_world::{SharedObstacles, SharedTerrain};
        use argusnet_types::Bounds2d;
        use argusnet_world::index::ObstacleLayer;
        use argusnet_world::tiled::TiledTerrain;
        use std::sync::Arc;
        let bounds = Bounds2d::new(-200.0, 1200.0, -600.0, 600.0);
        let layer = ObstacleLayer::new(bounds, 200.0, vec![block.clone()]).unwrap();
        let terrain = TiledTerrain::new(bounds, 8, 5.0, 0.0, Vec::new()).unwrap();
        PhysicalGuard::new(
            SharedTerrain::new(Arc::new(terrain), 5.0),
            SharedObstacles::new(Arc::new(layer)),
            2.0,
        )
    }

    /// An override that takes the platform again ends its rejoin.
    #[test]
    fn a_new_override_ends_a_rejoin() {
        let mut sim = rejoin_rig(true);
        sim.step().expect("tick runs");
        assert!(overridden(&sim).rejoining_at("d0").is_some());
        ov(&mut sim).set_poi(
            "d0",
            GotoHover::over([500.0, 0.0, 80.0], [500.0, 0.0], 80.0, 0.0, 0.0),
        );
        sim.step().expect("tick runs");
        assert_eq!(position_of(&sim, "d0").x, 500.0);
        assert_eq!(overridden(&sim).rejoining_at("d0"), None);
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

    /// A routed flight (P3) flies the polyline at constant speed: 300 m east,
    /// then 400 m north, 700 m at 28 m/s is 25 s. It starts where it starts,
    /// turns the corner on time, climbs linearly over the whole flight, and
    /// stops exactly on the target.
    #[test]
    fn a_routed_flight_turns_its_corners_at_constant_speed() {
        let traj = GotoHover::routed(
            [0.0, 0.0, 40.0],
            vec![[300.0, 0.0]],
            [300.0, 400.0],
            90.0,
            10.0,
            28.0,
        );
        assert_eq!(traj.t_arrive(), 35.0);
        assert_eq!(traj.path_length_m(), 700.0);
        assert_eq!(traj.target(), [300.0, 400.0, 90.0]);

        let (start, start_velocity) = traj.sample(10.0);
        assert_eq!(start, [0.0, 0.0, 40.0]);
        assert_eq!(start_velocity, [28.0, 0.0, 2.0]);

        // 5 s in: 140 m along the first leg.
        let (first_leg, first_velocity) = traj.sample(15.0);
        assert!((first_leg[0] - 140.0).abs() < 1e-9 && first_leg[1].abs() < 1e-9);
        assert!((first_leg[2] - 50.0).abs() < 1e-9);
        assert_eq!(first_velocity, [28.0, 0.0, 2.0]);

        // 15 s in: 420 m along, so 120 m past the corner, flying north.
        let (second_leg, second_velocity) = traj.sample(25.0);
        assert!((second_leg[0] - 300.0).abs() < 1e-9);
        assert!((second_leg[1] - 120.0).abs() < 1e-9);
        assert!((second_leg[2] - 70.0).abs() < 1e-9);
        assert!(second_velocity[0].abs() < 1e-9);
        assert!((second_velocity[1] - 28.0).abs() < 1e-9);

        let (arrived, stopped) = traj.sample(35.0);
        assert_eq!(arrived, [300.0, 400.0, 90.0]);
        assert_eq!(stopped, [0.0, 0.0, 0.0]);
    }

    /// With no corners a routed flight is the straight redirect, and a
    /// straight flight checkpoints exactly as it always has: no route field.
    /// A sample that lands exactly on a corner already flies the outgoing
    /// leg, so stage 3 never predicts it carrying on through the corner.
    #[test]
    fn a_sample_on_a_corner_flies_the_outgoing_leg() {
        // 100 m east then 100 m north at 10 m/s: the corner is at t = 10.
        let flight = GotoHover::routed(
            [0.0, 0.0, 50.0],
            vec![[100.0, 0.0]],
            [100.0, 100.0],
            50.0,
            0.0,
            10.0,
        );
        let (position, velocity) = flight.sample(10.0);
        assert_eq!([position[0], position[1]], [100.0, 0.0]);
        assert_eq!([velocity[0], velocity[1]], [0.0, 10.0]);
    }

    #[test]
    fn a_flight_with_no_corners_is_the_straight_redirect() {
        let routed = GotoHover::routed([0.0, 0.0, 80.0], Vec::new(), [280.0, 0.0], 80.0, 0.0, 28.0);
        let straight = GotoHover::redirect([0.0, 0.0, 80.0], [280.0, 0.0], 80.0, 0.0, 28.0);
        assert_eq!(routed, straight);
        let text = serde_json::to_value(&straight).unwrap();
        assert!(text.get("via_xy").is_none(), "{text}");
        let turned = GotoHover::routed(
            [0.0, 0.0, 80.0],
            vec![[100.0, 50.0]],
            [280.0, 0.0],
            80.0,
            0.0,
            28.0,
        );
        let back: GotoHover =
            serde_json::from_value(serde_json::to_value(&turned).unwrap()).unwrap();
        assert_eq!(back, turned, "a routed flight survives a checkpoint");
    }

    /// A flight with a height at each corner (R3) climbs and descends leg by
    /// leg: 100 m east at 40 m up to 60 m, 100 m level, 100 m down to 40 m,
    /// at 10 m/s. It survives a checkpoint, and without heights it is the
    /// routed flight exactly.
    #[test]
    fn a_flight_with_corner_heights_climbs_leg_by_leg() {
        let via = vec![[100.0, 0.0], [200.0, 0.0]];
        let flight = GotoHover::routed_over(
            [0.0, 0.0, 40.0],
            via.clone(),
            vec![60.0, 60.0],
            [300.0, 0.0],
            40.0,
            0.0,
            10.0,
        );
        assert_eq!(flight.t_arrive(), 30.0);
        let (climbing, climb_velocity) = flight.sample(5.0);
        assert!((climbing[0] - 50.0).abs() < 1e-9 && (climbing[2] - 50.0).abs() < 1e-9);
        assert!((climb_velocity[2] - 2.0).abs() < 1e-9);
        let (level, level_velocity) = flight.sample(15.0);
        assert!((level[2] - 60.0).abs() < 1e-9 && level_velocity[2].abs() < 1e-9);
        let (descending, descent_velocity) = flight.sample(25.0);
        assert!((descending[2] - 50.0).abs() < 1e-9);
        assert!((descent_velocity[2] + 2.0).abs() < 1e-9);
        assert_eq!(flight.sample(30.0).0, [300.0, 0.0, 40.0]);

        let back: GotoHover =
            serde_json::from_value(serde_json::to_value(&flight).unwrap()).unwrap();
        assert_eq!(back, flight, "the heights survive a checkpoint");
        let plain = GotoHover::routed_over(
            [0.0, 0.0, 40.0],
            via.clone(),
            Vec::new(),
            [300.0, 0.0],
            40.0,
            0.0,
            10.0,
        );
        assert_eq!(
            plain,
            GotoHover::routed([0.0, 0.0, 40.0], via, [300.0, 0.0], 40.0, 0.0, 10.0)
        );
        let text = serde_json::to_value(&plain).unwrap();
        assert!(text.get("via_z").is_none(), "{text}");
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
