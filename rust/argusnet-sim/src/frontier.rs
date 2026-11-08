//! Frontier exploration over the platforms' own map.
//!
//! In a run with no prior map, a search drone flying a lane planned before
//! the run maps wherever the lane happens to go. [`FrontierExplorer`] sends
//! it beyond the edge of the map instead:
//!
//! - **Goal.** The nearest block of cells (50 m at the default resolution)
//!   that is still at least [`UNEXPLORED_SHARE`] unmeasured
//!   ([`BeliefSnapshot::unexplored_centres`]), at least [`MIN_GOAL_DISTANCE_M`]
//!   away. A goal within [`CLAIM_RADIUS_M`] (about a scanner footprint) of
//!   another explorer's counts [`CLAIM_PENALTY_M`] further away, so the fleet
//!   spreads out. Ties go to the lowest block index, so the choice is
//!   deterministic.
//! - **Re-goal.** On arrival, or as soon as the goal's block is mostly
//!   measured (someone else mapped it on the way).
//! - **Nothing left.** The drone flies back toward where its original lane
//!   would have it, and follows the lane from there, so nothing jumps.
//! - **Flight.** The lane's speed and height above ground, over the believed
//!   terrain (unmeasured ground stands in high), with the lane's obstacle
//!   handling.
//!
//! Two earlier rules were measured and dropped. Aiming at the nearest
//! *frontier cell* (measured, with an unmeasured neighbour) stalled drones:
//! the scanner's rings leave gaps straight below a platform, so a frontier
//! cell under a hovering drone could stay a frontier forever. With a 100 m
//! minimum distance the drones kept moving, but hopping along the edge added
//! a thin strip each time and mapped less than the lanes did.
//!
//! The explorer's position, goal and last sample time are checkpointed. The
//! claims are rebuilt each tick from every explorer's goal, so they hold no
//! state of their own. The candidate list is built once a tick and shared.

use serde::{Deserialize, Serialize};

use crate::belief_world::BeliefSnapshot;
use crate::motion::{
    collision_aware_position, compose_terrain_state, ObstacleSampler, TerrainSampler,
};
use crate::waypoint_path::LoopedWaypointPath;

/// A goal within this of another explorer's goal counts as taken: about a
/// scanner's footprint, so explorers map different ground.
pub const CLAIM_RADIUS_M: f64 = 400.0;
/// A block at least this unmeasured is somewhere new to map.
pub const UNEXPLORED_SHARE: f64 = 0.5;
/// How much further away a taken goal counts.
pub const CLAIM_PENALTY_M: f64 = 1_000.0;
/// Goals nearer than this are not chosen: a goal this far away changes the
/// ground the scanner sees on the way, and a nearer one can sit in the gaps
/// the scanner's rings leave straight below a platform.
pub const MIN_GOAL_DISTANCE_M: f64 = 250.0;
/// A reached goal counts against candidates this near it, metres.
const SPENT_RADIUS_M: f64 = 150.0;
/// How many reached goals an explorer remembers.
const MAX_SPENT_GOALS: usize = 32;
/// How many seconds of flight ahead an explorer looks for a fitted wire to
/// climb over: long enough to be over it on arrival.
pub const WIRE_LOOKAHEAD_S: f64 = 4.0;

/// How fast an explorer changes height for a fitted wire, m/s. With
/// [`WIRE_LOOKAHEAD_S`] it reaches a wire's clearance before the wire.
const WIRE_CLIMB_MPS: f64 = 15.0;

/// How long an explorer the guard stopped flies back the way it came, s.
const ESCAPE_BACK_S: f64 = 2.0;

/// How long it holds its new height, s: past where it was stopped.
const ESCAPE_HOLD_S: f64 = 8.0;

/// The height over the ground an escape keeps, m.
pub const ESCAPE_MIN_AGL_M: f64 = 15.0;

/// How far it changes height, m: more than a wire's clearance.
const ESCAPE_HEIGHT_M: f64 = 30.0;

/// What an explorer carries between ticks.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExplorerState {
    pub xy_m: Option<[f64; 2]>,
    pub goal_m: Option<[f64; 2]>,
    /// True when the goal is the lane's point (no frontier was left).
    pub on_lane: bool,
    pub last_t_s: Option<f64>,
    /// The height flown last, with fitted wires about: whether it is under
    /// a wire or over one is a matter of where it is, not where the lane
    /// would put it. Absent without them, so such checkpoints are unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub flown_z_m: Option<f64>,
    /// Goals reached that still need work (W3): the rings of a scan leave a
    /// gap under the track, and the same line flown again leaves the same
    /// gap. Kept only for an explorer that rejoins its aircraft.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub spent_goals: Vec<[f64; 2]>,
    /// This explorer stays with its aircraft.
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub rejoins: bool,
    /// Getting away from where the guard stopped it. Absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub escape: Option<Escape>,
}

/// The way out after the guard stops an aircraft at something it had not
/// planned for: back the way it came, to a new height, and on from there.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Escape {
    /// The way back from the first stop, a unit vector.
    pub first_back: [f64; 2],
    /// The way being flown now, a unit vector.
    pub back: [f64; 2],
    /// Fly back until this time, seconds.
    pub back_until_s: f64,
    /// Hold this height until `until_s`, metres.
    pub z_m: f64,
    pub until_s: f64,
    /// How many stops this escape has met: up, down, then either side.
    pub stops: u32,
}

/// A search drone exploring the frontier of the map.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierExplorer {
    lane: LoopedWaypointPath,
    state: ExplorerState,
}

impl FrontierExplorer {
    pub fn new(lane: LoopedWaypointPath) -> Self {
        Self {
            lane,
            state: ExplorerState::default(),
        }
    }

    pub fn lane(&self) -> &LoopedWaypointPath {
        &self.lane
    }

    /// Explore, and fall back to the lane, no higher than `agl_m`.
    pub fn cap_agl(&mut self, agl_m: f64) {
        self.lane = self.lane.clone().with_agl_cap(agl_m);
    }

    /// Explore, and fall back to the lane, no faster than `speed_mps`.
    pub fn cap_speed(&mut self, speed_mps: f64) {
        self.lane = self.lane.clone().with_speed_cap(speed_mps);
    }

    pub fn state(&self) -> &ExplorerState {
        &self.state
    }

    pub fn restore(&mut self, state: ExplorerState) {
        self.state = state;
    }

    /// The guard moved this aircraft from `commanded_m`, where the explorer
    /// flew it, to `at_m`, so the explorer is no longer where it thinks. It
    /// goes on from where the aircraft is: flying on regardless would leave
    /// the aircraft held at the obstacle until the line to a position far
    /// ahead happened to clear it. It backs away, climbing; stopped again,
    /// descending; stopped again, to either side, as under a wire over a
    /// hilltop, where neither height is open. `from_m` is where the
    /// aircraft was: one further than a few steps from `commanded_m` was
    /// being flown by something else, so this says nothing about the path,
    /// and neither does a tick the explorer did not fly.
    pub fn stopped(
        &mut self,
        timestamp_s: f64,
        dt_s: f64,
        from_m: [f64; 3],
        commanded_m: [f64; 3],
        at_m: [f64; 3],
    ) {
        let reach = 3.0 * self.lane_speed_mps() * dt_s;
        let step = [
            commanded_m[0] - from_m[0],
            commanded_m[1] - from_m[1],
            commanded_m[2] - from_m[2],
        ];
        let flown = self.state.last_t_s == Some(timestamp_s);
        if !flown || (step[0] * step[0] + step[1] * step[1] + step[2] * step[2]).sqrt() > reach {
            return;
        }
        let state = &mut self.state;
        state.xy_m = Some([at_m[0], at_m[1]]);
        state.flown_z_m = Some(at_m[2]);
        // The first stop says which way is back. A stop on the way out says
        // nothing new: it was flying the escape, not the path.
        let (first, stops) = match &state.escape {
            Some(escape) => (escape.first_back, escape.stops + 1),
            None => {
                let (dx, dy) = (commanded_m[0] - at_m[0], commanded_m[1] - at_m[1]);
                let length = dx.hypot(dy).max(1e-9);
                ([-dx / length, -dy / length], 1)
            }
        };
        let [x, y] = first;
        let (back, z_m) = match stops % 4 {
            1 => (first, at_m[2] + ESCAPE_HEIGHT_M),
            2 => (first, at_m[2] - ESCAPE_HEIGHT_M),
            3 => ([-y, x], at_m[2]),
            _ => ([y, -x], at_m[2]),
        };
        state.escape = Some(Escape {
            first_back: first,
            back,
            back_until_s: timestamp_s + ESCAPE_BACK_S,
            z_m,
            until_s: timestamp_s + ESCAPE_HOLD_S,
            stops,
        });
    }

    /// Where the aircraft was, against where this explorer put it. Further
    /// apart than a few steps, something else is flying it (W3), and the
    /// explorer goes on from the aircraft. Flying on regardless would have
    /// the aircraft jump across the map when released, and a straight line
    /// from its pad that meets a hill is stopped by the guard for good.
    /// True when it rejoined, and the aircraft should not take this command.
    pub fn rejoin(&mut self, from_m: [f64; 3], commanded_m: [f64; 3], dt_s: f64) -> bool {
        self.state.rejoins = true;
        let reach = 3.0 * self.lane_speed_mps() * dt_s;
        let apart = (commanded_m[0] - from_m[0]).hypot(commanded_m[1] - from_m[1]) > reach;
        if apart {
            self.state.xy_m = Some([from_m[0], from_m[1]]);
            if self.state.flown_z_m.is_some() {
                self.state.flown_z_m = Some(from_m[2]);
            }
        }
        apart
    }

    /// The goal this explorer holds, for the other explorers' claims.
    pub fn goal(&self) -> Option<[f64; 2]> {
        (!self.state.on_lane).then_some(self.state.goal_m).flatten()
    }

    fn choose_goal(
        frontier: &[[f64; 2]],
        from: [f64; 2],
        claims: &[[f64; 2]],
        spent: &[[f64; 2]],
        belief: &BeliefSnapshot,
    ) -> Option<[f64; 2]> {
        let mut best: Option<(f64, [f64; 2])> = None;
        for &centre in frontier {
            let mut cost = (centre[0] - from[0]).hypot(centre[1] - from[1]);
            if cost < MIN_GOAL_DISTANCE_M {
                continue;
            }
            cost += belief.crossing_cost_m(from, centre);
            if spent
                .iter()
                .any(|s| (s[0] - centre[0]).hypot(s[1] - centre[1]) < SPENT_RADIUS_M)
            {
                cost += CLAIM_PENALTY_M;
            }
            if claims
                .iter()
                .any(|c| (c[0] - centre[0]).hypot(c[1] - centre[1]) < CLAIM_RADIUS_M)
            {
                cost += CLAIM_PENALTY_M;
            }
            // Strictly less: the first (lowest index) of equal costs wins.
            if best.is_none_or(|(b, _)| cost < b) {
                best = Some((cost, centre));
            }
        }
        best.map(|(_, centre)| centre)
    }

    /// Position and velocity at `timestamp_s`. `claims` holds the other
    /// explorers' goals; this explorer's new goal is added to it.
    /// `frontier` caches the tick's candidate goals: it is filled by the
    /// first explorer that needs a goal and shared by the rest (listing them
    /// scans the whole grid).
    pub fn sample(
        &mut self,
        timestamp_s: f64,
        terrain: &dyn TerrainSampler,
        obstacles: Option<&dyn ObstacleSampler>,
        belief: Option<&BeliefSnapshot>,
        frontier: &std::cell::OnceCell<Vec<[f64; 2]>>,
        claims: &mut Vec<[f64; 2]>,
    ) -> ([f64; 3], [f64; 3]) {
        let Some(belief) = belief else {
            return self.lane.sample(timestamp_s, terrain, obstacles);
        };
        // This explorer's own goal is among the claims gathered before
        // sampling; it is not competition, and it is re-added below.
        if let Some(own) = self.goal() {
            if let Some(at) = claims.iter().position(|claim| *claim == own) {
                claims.remove(at);
            }
        }
        let config = self.lane.config().clone();
        let speed = self.lane_speed_mps();
        let lane_xy = |t: f64| {
            let (p, _) = self.lane.sample(t, terrain, obstacles);
            [p[0], p[1]]
        };
        let state = &mut self.state;
        let xy = *state.xy_m.get_or_insert_with(|| lane_xy(timestamp_s));
        let dt = state
            .last_t_s
            .map_or(0.0, |last| (timestamp_s - last).max(0.0));
        state.last_t_s = Some(timestamp_s);

        let arrival_m = belief.resolution_m();
        // Back on the lane, a frontier is looked for again each tick.
        let needs_goal = match state.goal_m {
            None => true,
            Some(_) if state.on_lane => true,
            Some(goal) => {
                (goal[0] - xy[0]).hypot(goal[1] - xy[1]) < arrival_m
                    || !belief.goal_has_work(goal[0], goal[1], UNEXPLORED_SHARE)
            }
        };
        if needs_goal {
            if let (true, false, Some(goal)) = (state.rejoins, state.on_lane, state.goal_m) {
                if belief.goal_has_work(goal[0], goal[1], UNEXPLORED_SHARE) {
                    state.spent_goals.push(goal);
                    if state.spent_goals.len() > MAX_SPENT_GOALS {
                        state.spent_goals.remove(0);
                    }
                }
            }
            let frontier = frontier.get_or_init(|| belief.goal_centres(UNEXPLORED_SHARE));
            match Self::choose_goal(frontier, xy, claims, &state.spent_goals, belief) {
                Some(goal) => {
                    state.goal_m = Some(goal);
                    state.on_lane = false;
                }
                None => {
                    state.goal_m = Some(lane_xy(timestamp_s));
                    state.on_lane = true;
                }
            }
        }
        let goal = state.goal_m.expect("set above");
        if !state.on_lane {
            claims.push(goal);
        }

        if state
            .escape
            .as_ref()
            .is_some_and(|escape| timestamp_s >= escape.until_s)
        {
            state.escape = None;
        }
        let (dx, dy) = (goal[0] - xy[0], goal[1] - xy[1]);
        let distance = dx.hypot(dy);
        let step = (speed * dt).min(distance);
        let (direction, moving) = if distance > 1e-9 {
            ([dx / distance, dy / distance], step < distance || dt > 0.0)
        } else {
            ([0.0, 0.0], false)
        };
        let backing = state
            .escape
            .as_ref()
            .filter(|escape| timestamp_s < escape.back_until_s);
        let (direction, step, moving) = match backing {
            Some(escape) => (escape.back, speed * dt, true),
            None => (direction, step, moving),
        };
        // Out of a corridor it is running along, not on down it.
        let direction = match (moving, backing) {
            (true, None) => belief
                .out_of_corridor(xy, direction, (!state.on_lane).then_some(goal))
                .map_or(direction, |out| {
                    let turned = [direction[0] + out[0], direction[1] + out[1]];
                    let length = turned[0].hypot(turned[1]).max(1e-9);
                    [turned[0] / length, turned[1] / length]
                }),
            _ => direction,
        };
        let new_xy = [xy[0] + direction[0] * step, xy[1] + direction[1] * step];
        state.xy_m = Some(new_xy);
        let velocity_xy = if moving {
            [direction[0] * speed, direction[1] * speed]
        } else {
            [0.0, 0.0]
        };
        let (position, velocity) = compose_terrain_state(
            terrain,
            new_xy,
            velocity_xy,
            config.base_agl_m,
            0.0,
            config.min_agl_m,
            false,
        );
        let obstacles = if config.avoid_obstacles {
            obstacles
        } else {
            None
        };
        let resolved = collision_aware_position(terrain, obstacles, position, config.min_agl_m);
        let mut position = resolved.position_m;
        // Over a fitted wire ahead, not round it, and away from a stop: the
        // height is flown, not jumped, until it is back on its own.
        let fitted = belief.has_fitted_wires();
        if fitted || state.flown_z_m.is_some() || state.escape.is_some() {
            let flown_z = state.flown_z_m.unwrap_or(position[2]);
            // A survey height can stand above a wire. Held to it, an escape
            // from under the wire is commanded up through it whichever way
            // it turns, for good, so an escape keeps only a low floor.
            let floor_agl = if state.escape.is_some() {
                config.min_agl_m.min(ESCAPE_MIN_AGL_M)
            } else {
                config.min_agl_m
            };
            let lowest = terrain.height_at(position[0], position[1]) + floor_agl;
            if fitted {
                let bounds =
                    belief.wire_bounds_ahead(new_xy, direction, speed * WIRE_LOOKAHEAD_S, flown_z);
                if let Some(ceiling) = bounds.ceiling_m {
                    position[2] = position[2].min(ceiling.max(lowest));
                }
                if let Some(floor) = bounds.floor_m {
                    position[2] = position[2].max(floor);
                }
            }
            if let Some(escape) = &state.escape {
                position[2] = escape.z_m;
            }
            // A step through a wire is one the guard holds the aircraft on.
            let reach = WIRE_CLIMB_MPS * dt;
            let wanted = position[2].max(lowest);
            position[2] = wanted.clamp(flown_z - reach, flown_z + reach).max(lowest);
            let settled = !fitted && state.escape.is_none() && position[2] == wanted;
            state.flown_z_m = (!settled).then_some(position[2]);
        }
        (position, velocity)
    }

    fn lane_speed_mps(&self) -> f64 {
        self.lane.speed_mps()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_world::{BeliefGrid, ExplorationConfig};
    use crate::range_scan::{RangeReturn, RangeScan, ReturnClass};
    use crate::waypoint_path::WaypointPathConfig;
    use argusnet_types::Bounds2d;

    struct Flat;

    impl TerrainSampler for Flat {
        fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
            0.0
        }
        fn gradient_at(&self, _x_m: f64, _y_m: f64, _delta_m: Option<f64>) -> [f64; 2] {
            [0.0, 0.0]
        }
        fn clamp_altitude(&self, _x_m: f64, _y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
            z_m.max(min_agl_m.max(0.0))
        }
    }

    fn lane() -> LoopedWaypointPath {
        LoopedWaypointPath::new(WaypointPathConfig {
            waypoints_xy_m: vec![[100.0, 100.0], [100.0, 900.0]],
            speed_mps: 10.0,
            base_agl_m: 50.0,
            vertical_amplitude_m: 0.0,
            vertical_frequency_rad_s: 0.0,
            phase: 0.0,
            min_agl_m: 3.0,
            avoid_obstacles: false,
        })
        .unwrap()
    }

    /// A 1 km square at 10 m, measured where `x < measured_to_m`.
    fn belief(measured_to_m: f64) -> crate::belief_world::BeliefSnapshot {
        let mut grid = BeliefGrid::new(Bounds2d::new(0.0, 1000.0, 0.0, 1000.0), 10.0, 0.0).unwrap();
        let returns = (0..100)
            .flat_map(|j| (0..100).map(move |i| (i, j)))
            .map(|(i, j)| [i as f64 * 10.0 + 5.0, j as f64 * 10.0 + 5.0])
            .filter(|p| p[0] < measured_to_m)
            .map(|p| RangeReturn {
                point_m: [p[0], p[1], 0.0],
                class: ReturnClass::Ground,
                acquired_offset_s: None,
                acquired_from_m: None,
            })
            .collect();
        grid.integrate(
            &RangeScan {
                node_id: "d".into(),
                timestamp_s: 0.0,
                origin_m: [0.0, 0.0, 100.0],
                returns,
                misses: 0,
                wires: Vec::new(),
                sensor: Default::default(),
                pod: None,
                wire_points: Vec::new(),
                wire_lines: Vec::new(),
            },
            1,
        );
        grid.snapshot(&ExplorationConfig::default())
    }

    fn sample(
        explorer: &mut FrontierExplorer,
        t: f64,
        belief: Option<&crate::belief_world::BeliefSnapshot>,
        claims: &mut Vec<[f64; 2]>,
    ) -> ([f64; 3], [f64; 3]) {
        explorer.sample(t, &Flat, None, belief, &std::cell::OnceCell::new(), claims)
    }

    #[test]
    fn an_explorer_heads_for_the_nearest_unexplored_block_at_lane_speed() {
        let map = belief(500.0);
        let mut explorer = FrontierExplorer::new(lane());
        let mut claims = Vec::new();
        let (p0, _) = sample(&mut explorer, 0.0, Some(&map), &mut claims);
        assert_eq!([p0[0], p0[1]], [100.0, 100.0]);
        // Nearest half-unmeasured block centre from (100, 100): (525, 75).
        assert_eq!(explorer.goal(), Some([525.0, 75.0]));
        assert_eq!(claims, vec![[525.0, 75.0]]);
        let (p1, v1) = sample(&mut explorer, 1.0, Some(&map), &mut Vec::new());
        let moved = (p1[0] - 100.0).hypot(p1[1] - 100.0);
        assert!((moved - 10.0).abs() < 1e-9, "{moved}");
        assert!((v1[0].hypot(v1[1]) - 10.0).abs() < 1e-9);
        assert!((p1[2] - 50.0).abs() < 1e-9, "flies at the lane's height");
    }

    #[test]
    fn a_claimed_goal_sends_the_next_explorer_elsewhere() {
        let map = belief(500.0);
        let mut first = FrontierExplorer::new(lane());
        let mut second = FrontierExplorer::new(lane());
        let mut claims = Vec::new();
        sample(&mut first, 0.0, Some(&map), &mut claims);
        sample(&mut second, 0.0, Some(&map), &mut claims);
        let (a, b) = (first.goal().unwrap(), second.goal().unwrap());
        assert!(
            (a[0] - b[0]).hypot(a[1] - b[1]) >= CLAIM_RADIUS_M,
            "{a:?} {b:?}"
        );
        assert_eq!(b, [525.0, 475.0]);
    }

    #[test]
    fn with_nothing_left_to_map_it_flies_its_lane_and_without_a_map_it_is_the_lane() {
        let lane_now = lane().sample(0.0, &Flat, None);
        let mut explorer = FrontierExplorer::new(lane());
        let mut claims = Vec::new();
        let done = sample(&mut explorer, 0.0, Some(&belief(2000.0)), &mut claims);
        assert_eq!(done.0, lane_now.0);
        assert_eq!(explorer.goal(), None, "a lane point is not a claim");
        assert!(claims.is_empty());
        let mut plain = FrontierExplorer::new(lane());
        assert_eq!(
            sample(&mut plain, 7.0, None, &mut Vec::new()),
            lane().sample(7.0, &Flat, None)
        );
    }

    /// Stopped by the guard, an explorer goes on from where the aircraft is,
    /// not where it would have been: back the way it came, climbing, and on.
    #[test]
    fn a_stopped_explorer_backs_away_and_changes_height() {
        let map = belief(500.0);
        let mut explorer = FrontierExplorer::new(lane());
        let (p0, _) = sample(&mut explorer, 0.0, Some(&map), &mut Vec::new());
        let (p1, _) = sample(&mut explorer, 1.0, Some(&map), &mut Vec::new());
        let held = [p0[0] + 1.0, p0[1], p0[2]];
        explorer.stopped(1.0, 1.0, p0, p1, held);
        assert_eq!(explorer.state().xy_m, Some([held[0], held[1]]));
        let (p2, _) = sample(&mut explorer, 2.0, Some(&map), &mut Vec::new());
        assert!(p2[0] < held[0], "backs away: {p2:?} from {held:?}");
        assert!((p2[2] - (held[2] + WIRE_CLIMB_MPS)).abs() < 1e-9, "climbs");
        // Past the back-off it heads on, still high.
        let (p3, _) = sample(&mut explorer, 3.0, Some(&map), &mut Vec::new());
        let (p4, _) = sample(&mut explorer, 4.0, Some(&map), &mut Vec::new());
        assert!(p4[0] > p3[0], "heads on again");
        assert!(p4[2] >= held[2] + 2.0 * WIRE_CLIMB_MPS - 1e-9);
        // Stopped again, it tries below, on the same way back, then each
        // side of it, never reversing on itself.
        explorer.stopped(4.0, 1.0, p3, p4, p3);
        let escape = explorer.state().escape.clone().unwrap();
        assert!(escape.z_m < p3[2]);
        assert_eq!(escape.back, escape.first_back);
        let way = escape.first_back;
        sample(&mut explorer, 5.0, Some(&map), &mut Vec::new());
        let at = [100.0, 100.0, 60.0];
        let near = [101.0, 100.0, 60.0];
        explorer.stopped(5.0, 1.0, at, near, at);
        assert_eq!(
            explorer.state().escape.as_ref().unwrap().back,
            [-way[1], way[0]]
        );
        sample(&mut explorer, 6.0, Some(&map), &mut Vec::new());
        explorer.stopped(6.0, 1.0, at, near, at);
        assert_eq!(
            explorer.state().escape.as_ref().unwrap().back,
            [way[1], -way[0]]
        );
        // And in the end it is back on its own height, with nothing left to
        // record.
        for t in 5..40 {
            sample(&mut explorer, f64::from(t), Some(&map), &mut Vec::new());
        }
        assert!(explorer.state().escape.is_none());
        assert_eq!(explorer.state().flown_z_m, None);
    }

    /// A platform something else was flying says nothing about the path.
    #[test]
    fn a_stop_far_from_the_path_or_on_an_unflown_tick_changes_nothing() {
        let map = belief(500.0);
        let mut explorer = FrontierExplorer::new(lane());
        let (p0, _) = sample(&mut explorer, 0.0, Some(&map), &mut Vec::new());
        let before = explorer.state().clone();
        let far = [p0[0] - 500.0, p0[1], p0[2]];
        explorer.stopped(0.0, 1.0, far, p0, far);
        assert_eq!(explorer.state(), &before, "overridden");
        explorer.stopped(1.0, 1.0, p0, p0, p0);
        assert_eq!(explorer.state(), &before, "not flown this tick");
    }

    /// W3: an explorer stays with an aircraft something else flies, so it
    /// is released where it is, and is left alone while flying itself.
    #[test]
    fn an_explorer_rejoins_an_aircraft_flown_by_something_else() {
        let map = belief(500.0);
        let mut explorer = FrontierExplorer::new(lane());
        let (p0, _) = sample(&mut explorer, 0.0, Some(&map), &mut Vec::new());
        assert!(!explorer.rejoin(p0, p0, 1.0), "flying itself");
        assert_eq!(explorer.state().xy_m, Some([p0[0], p0[1]]));
        let far = [p0[0] - 500.0, p0[1] + 500.0, p0[2]];
        assert!(explorer.rejoin(far, p0, 1.0));
        assert_eq!(explorer.state().xy_m, Some([far[0], far[1]]));
    }

    #[test]
    fn a_restored_explorer_samples_as_the_original_would() {
        let map = belief(500.0);
        let mut original = FrontierExplorer::new(lane());
        for t in 0..5 {
            sample(&mut original, t as f64, Some(&map), &mut Vec::new());
        }
        let mut restored = FrontierExplorer::new(lane());
        restored.restore(original.state().clone());
        for t in 5..10 {
            let a = sample(&mut original, t as f64, Some(&map), &mut Vec::new());
            let b = sample(&mut restored, t as f64, Some(&map), &mut Vec::new());
            assert_eq!(a, b);
        }
    }
}
