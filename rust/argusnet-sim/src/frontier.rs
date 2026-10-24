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

/// What an explorer carries between ticks.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct ExplorerState {
    pub xy_m: Option<[f64; 2]>,
    pub goal_m: Option<[f64; 2]>,
    /// True when the goal is the lane's point (no frontier was left).
    pub on_lane: bool,
    pub last_t_s: Option<f64>,
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

    /// The goal this explorer holds, for the other explorers' claims.
    pub fn goal(&self) -> Option<[f64; 2]> {
        (!self.state.on_lane).then_some(self.state.goal_m).flatten()
    }

    fn choose_goal(frontier: &[[f64; 2]], from: [f64; 2], claims: &[[f64; 2]]) -> Option<[f64; 2]> {
        let mut best: Option<(f64, [f64; 2])> = None;
        for &centre in frontier {
            let mut cost = (centre[0] - from[0]).hypot(centre[1] - from[1]);
            if cost < MIN_GOAL_DISTANCE_M {
                continue;
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
                    || !belief.is_unexplored(goal[0], goal[1], UNEXPLORED_SHARE)
            }
        };
        if needs_goal {
            let frontier = frontier.get_or_init(|| belief.unexplored_centres(UNEXPLORED_SHARE));
            match Self::choose_goal(frontier, xy, claims) {
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

        let (dx, dy) = (goal[0] - xy[0], goal[1] - xy[1]);
        let distance = dx.hypot(dy);
        let step = (speed * dt).min(distance);
        let (direction, moving) = if distance > 1e-9 {
            ([dx / distance, dy / distance], step < distance || dt > 0.0)
        } else {
            ([0.0, 0.0], false)
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
        (resolved.position_m, velocity)
    }

    fn lane_speed_mps(&self) -> f64 {
        self.lane
            .config()
            .speed_mps
            .max(LoopedWaypointPath::MIN_SPEED_MPS)
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
