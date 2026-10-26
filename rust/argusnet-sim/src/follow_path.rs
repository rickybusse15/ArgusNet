//! `FollowPathController`: orbit a moving target, avoiding what is in the way.
//!
//! The last and largest of the three controllers `sim.py` line 136 names as
//! blocked from extraction, and the one the other two are built around.
//! `__call__` is ~370 lines in a single method and is **not divisible**: the
//! candidate scan, the replan decision, the walk along the route, the two
//! exclusion-zone passes and the altitude computation all read and write one
//! `_state` dict, in an order that matters. Porting a piece would ship dead
//! code, so the whole thing moves at once.
//!
//! # Capabilities, as everywhere else in this lane
//!
//! Five external reads, five narrow traits: terrain (from
//! [`crate::motion::TerrainSampler`]), obstacle containment, line of sight,
//! and the planner's two entry points. `argusnet-sim` gains no new crate
//! dependency, and the parity test replays every read from a recording that
//! panics on a miss — which matters more here than anywhere before it,
//! because this controller *chooses* using those reads. The call pattern is
//! the decision.
//!
//! # Three findings, preserved rather than corrected
//!
//! **An unseeded controller never flies.** Tick 0 has `delta_t == 0` and no
//! `last_xy`, so it takes the branch that jumps straight to
//! `route[1]` — the platform teleports to its first goal — and collapses the
//! route to `[chosen, chosen]`. Every later tick then walks a zero-length
//! route. The replan that would rebuild it never fires, because the
//! candidate score's second key is distance from where the platform *is*: a
//! frozen platform keeps choosing the candidate nearest itself, which never
//! drifts past the replan threshold. Verified over thirteen seconds: one
//! position, thirteen times, while the reported velocity kept changing.
//! [`FollowPathController::seed`] is what avoids this, and it is how a
//! launched drone hands over mid-flight.
//!
//! **`terrain_following_smoothing_s` does nothing.** The reference computes
//! an exponential moving average of the terrain height under the platform,
//! stores it, and never reads it again except to compute the next average.
//! The altitude on that branch is the flat `terrain_following_agl_m`. Ported
//! anyway, with a property test, because the state is real and a future fix
//! would wire it up — see [`FollowPathState::smoothed_terrain_z`].
//!
//! **The acceleration limit is applied before the orbit blend, and the blend
//! ignores it.** The limited velocity is stored as `last_velocity_xy` and
//! *then* the tangent blend rewrites the velocity without updating that
//! store, so the blend can move the speed past the limit and the next tick
//! limits against a value the platform never flew. Measured: a 5.75 m/s
//! change under a 1.5 m/s² limit.

use crate::motion::{
    advance_along_polyline, collision_aware_position, compose_air_state, BoundsXy, ObstacleSampler,
    TargetAltitudeHold, TerrainSampler,
};

/// Whether the target is visible from a candidate position.
///
/// Only `visible` is read by this controller; the reference's
/// `VisibilityResult` carries transmittance and noise multipliers that the
/// sensing path uses and this one ignores.
pub trait VisibilitySampler {
    fn is_visible(&self, observer_m: [f64; 3], target_m: [f64; 3]) -> bool;
}

/// A route the planner produced.
#[derive(Debug, Clone, PartialEq)]
pub struct PlannedRoute {
    pub points_xy_m: Vec<[f64; 2]>,
    pub length_m: f64,
}

/// The two planner entry points this controller uses.
///
/// Optional as a whole: with no planner the controller plans in straight
/// lines and scores candidates by euclidean distance, which is a different
/// code path rather than a degraded one.
pub trait RoutePlanner {
    /// A route between two points, or `None` when none exists.
    fn plan_route(&self, from_xy_m: [f64; 2], to_xy_m: [f64; 2]) -> Option<PlannedRoute>;

    /// The nearest point with enough clearance to sit at.
    fn nearest_free_point(&self, xy_m: [f64; 2]) -> [f64; 2];
}

/// A circular region the platform must stay out of, or a soft hint.
///
/// Only [`MissionZone::is_exclusion`] zones are enforced here. Surveillance,
/// patrol and objective zones are planner hints and this controller ignores
/// them entirely — which is worth stating because both enforcement passes
/// filter on the type and a port that dropped the filter would fence the
/// platform out of the areas it was sent to watch.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionZone {
    pub zone_id: String,
    pub zone_type: String,
    pub center_m: [f64; 3],
    pub radius_m: f64,
}

impl MissionZone {
    /// The reference spells this `"exclusion"` in one pass as a constant and
    /// in the other as a bare literal. Same value; one place here.
    pub fn is_exclusion(&self) -> bool {
        self.zone_type == "exclusion"
    }
}

/// Everything about a follow controller that does not change tick to tick.
#[derive(Debug, Clone, PartialEq)]
pub struct FollowPathConfig {
    pub map_bounds_m: BoundsXy,
    pub base_agl_m: f64,
    pub standoff_radius_m: f64,
    pub map_scale: f64,
    pub max_speed_mps: f64,
    /// Fixed angular offset, so drones on the same target do not stack.
    pub phase: f64,
    /// How far ahead of the target the orbit is centred.
    pub lead_s: f64,
    pub candidate_count: usize,
    /// Positive is counter-clockwise; the sign also picks the orbit tangent.
    pub rotation_rate_rad_s: f64,
    pub min_agl_m: f64,
    pub vertical_amplitude_m: f64,
    pub vertical_frequency_rad_s: f64,
    /// `Some` switches altitude to target-relative hold.
    pub target_altitude_offset_m: Option<f64>,
    pub max_accel_mps2: Option<f64>,
    pub terrain_following: bool,
    pub terrain_following_agl_m: f64,
    /// Inert. See the module comment.
    pub terrain_following_smoothing_s: f64,
    /// Which angular slot of `slot_count` this platform occupies.
    pub slot_index: usize,
    pub slot_count: usize,
}

impl Default for FollowPathConfig {
    fn default() -> Self {
        Self {
            map_bounds_m: BoundsXy {
                x_min_m: -1000.0,
                x_max_m: 1000.0,
                y_min_m: -1000.0,
                y_max_m: 1000.0,
            },
            base_agl_m: 120.0,
            standoff_radius_m: 180.0,
            map_scale: 1.0,
            max_speed_mps: 22.0,
            phase: 0.0,
            lead_s: 4.0,
            candidate_count: 12,
            rotation_rate_rad_s: 0.08,
            min_agl_m: 18.0,
            vertical_amplitude_m: 0.0,
            vertical_frequency_rad_s: 0.0,
            target_altitude_offset_m: None,
            max_accel_mps2: None,
            terrain_following: false,
            terrain_following_agl_m: 30.0,
            terrain_following_smoothing_s: 1.5,
            slot_index: 0,
            slot_count: 1,
        }
    }
}

/// Everything that survives a tick.
///
/// This is the reference's `_state` dict, typed. It is public because P4.1d
/// made handler state a checkpoint concern: a controller stage has to be
/// able to hand this over and take it back, or a resumed mission flies a
/// different path from an uninterrupted one.
#[derive(Debug, Clone, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct FollowPathState {
    /// The memo. A second call at the same timestamp returns the first
    /// call's answer rather than recomputing — and the reference compares
    /// with a `1e-9` tolerance, not equality.
    pub cached: Option<(f64, [f64; 3], [f64; 3])>,
    pub last_timestamp_s: Option<f64>,
    pub last_xy_m: Option<[f64; 2]>,
    pub last_direction_xy: Option<[f64; 2]>,
    pub route_points_xy_m: Option<Vec<[f64; 2]>>,
    pub route_goal_xy_m: Option<[f64; 2]>,
    /// Only tracked when `max_accel_mps2` is set: the reference writes this
    /// inside that branch, so with no limit configured it stays `None`.
    pub last_velocity_xy: Option<[f64; 2]>,
    /// Written every terrain-following tick and **never read** by anything
    /// that affects the output. See the module comment.
    pub smoothed_terrain_z: Option<f64>,
}

/// The external reads one tick needs.
pub struct FollowPathContext<'a> {
    pub terrain: &'a dyn TerrainSampler,
    pub obstacles: &'a dyn ObstacleSampler,
    pub visibility: &'a dyn VisibilitySampler,
    pub planner: Option<&'a dyn RoutePlanner>,
    pub zones: &'a [MissionZone],
}

/// Orbit a moving target at a standoff radius, avoiding obstacles and zones.
#[derive(Debug, Clone, PartialEq)]
pub struct FollowPathController {
    config: FollowPathConfig,
    state: FollowPathState,
}

/// The bounds margin every candidate is tested against, scaled by the map.
fn bounds_margin_m(config: &FollowPathConfig) -> f64 {
    15.0 * config.map_scale
}

/// Euclidean length, computed the way `np.linalg.norm` computes it.
///
/// **Not** `f64::hypot`. The two agree to within an ULP and disagree
/// sometimes, because `hypot` is a different algorithm — it avoids
/// intermediate overflow and is often correctly rounded, where numpy takes
/// the plain `sqrt` of the sum of squares. Over a sequence of ticks that
/// difference compounds: with `hypot` here, this controller matched the
/// reference exactly for five samples and diverged in the twelfth digit on
/// the sixth.
///
/// Parity with the reference is what this port owes; being marginally more
/// accurate than it is a change, not an improvement.
fn norm(v: [f64; 2]) -> f64 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

fn sub(a: [f64; 2], b: [f64; 2]) -> [f64; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

impl FollowPathController {
    pub fn new(config: FollowPathConfig) -> Self {
        Self {
            config,
            state: FollowPathState::default(),
        }
    }

    pub fn config(&self) -> &FollowPathConfig {
        &self.config
    }

    /// Orbit no faster than `speed_mps`.
    pub fn cap_speed(&mut self, speed_mps: f64) {
        self.config.max_speed_mps = self.config.max_speed_mps.min(speed_mps);
    }

    pub fn state(&self) -> &FollowPathState {
        &self.state
    }

    /// Take a checkpoint's state back.
    pub fn restore_state(&mut self, state: FollowPathState) {
        self.state = state;
    }

    /// Prime the controller mid-flight, without producing a sample.
    ///
    /// How a launched drone hands over: the platform already has a position
    /// and a heading, and the controller has to continue from them rather
    /// than from nothing. Note what it does *not* clear — the smoothed
    /// terrain height and the acceleration limiter's last velocity both
    /// survive a seed, matching the reference, which lists the keys it
    /// resets and does not list those two.
    pub fn seed(&mut self, position_m: [f64; 3], velocity_mps: [f64; 3], timestamp_s: f64) {
        self.state.cached = None;
        self.state.last_timestamp_s = Some(timestamp_s);
        self.state.last_xy_m = Some([position_m[0], position_m[1]]);
        self.state.last_direction_xy = Some([velocity_mps[0], velocity_mps[1]]);
        self.state.route_points_xy_m = None;
        self.state.route_goal_xy_m = None;
    }

    /// Return to a freshly-constructed state.
    ///
    /// Like [`FollowPathController::seed`], this leaves `smoothed_terrain_z`
    /// and `last_velocity_xy` alone — the reference's `reset_state` clears
    /// seven named keys and those are not among them. Preserved because a
    /// reset that cleared more would be a behaviour change dressed as
    /// tidying, and the difference is observable the first tick after.
    pub fn reset_state(&mut self) {
        self.state.cached = None;
        self.state.last_timestamp_s = None;
        self.state.last_xy_m = None;
        self.state.last_direction_xy = None;
        self.state.route_points_xy_m = None;
        self.state.route_goal_xy_m = None;
    }

    /// One tick.
    ///
    /// `target` supplies the target's position and velocity at a time; it is
    /// called twice, once at `timestamp_s` and once at `timestamp_s +
    /// lead_s`, and the orbit is centred on the *lead* sample while the
    /// follow floor at the end uses the *current* one.
    pub fn sample(
        &mut self,
        timestamp_s: f64,
        target: &mut dyn FnMut(f64) -> ([f64; 3], [f64; 3]),
        context: &FollowPathContext<'_>,
    ) -> ([f64; 3], [f64; 3]) {
        if let Some((cached_at, position, velocity)) = self.state.cached {
            if (timestamp_s - cached_at).abs() <= 1.0e-9 {
                return (position, velocity);
            }
        }

        // Time going backwards is a new run, not an interpolation. The
        // reference clears the route and the heading but deliberately keeps
        // the memo it just failed to match.
        if let Some(last) = self.state.last_timestamp_s {
            if timestamp_s < last - 1.0e-9 {
                self.state.last_timestamp_s = None;
                self.state.last_xy_m = None;
                self.state.last_direction_xy = None;
                self.state.route_points_xy_m = None;
                self.state.route_goal_xy_m = None;
            }
        }

        let (current_target_position, current_target_velocity) = target(timestamp_s);
        let (lead_position, lead_velocity) = target(timestamp_s + self.config.lead_s);

        let slot_offset = (self.config.slot_index as f64 * std::f64::consts::TAU)
            / (self.config.slot_count.max(1) as f64);
        let base_angle =
            self.config.phase + slot_offset + self.config.rotation_rate_rad_s * timestamp_s;

        let margin_m = bounds_margin_m(&self.config);
        let current_xy = match self.state.last_xy_m {
            Some(xy) => xy,
            None => {
                let clamped = self
                    .config
                    .map_bounds_m
                    .clamp([lead_position[0], lead_position[1]], margin_m);
                match context.planner {
                    Some(planner) => planner.nearest_free_point(clamped),
                    None => clamped,
                }
            }
        };
        let current_heading_xy = self
            .state
            .last_direction_xy
            .unwrap_or([lead_velocity[0], lead_velocity[1]]);

        let best = self.scan_candidates(
            timestamp_s,
            base_angle,
            TargetSample {
                position_m: lead_position,
                velocity_mps: lead_velocity,
            },
            current_xy,
            current_heading_xy,
            context,
        );

        let (mut route_points, route_goal) = self.resolve_route(current_xy, &best, context);

        let last_xy = self.state.last_xy_m;
        let delta_t_s = self
            .state
            .last_timestamp_s
            .map_or(0.0, |last| (timestamp_s - last).max(0.0));
        let travel_distance_m = self.config.max_speed_mps * delta_t_s;

        let (chosen_xy, route_direction_xy, route_goal) = if delta_t_s <= 1.0e-9 {
            if last_xy.is_none() && route_points.len() > 1 {
                // The teleport. See the module comment: this is what makes
                // an unseeded controller freeze.
                let chosen = route_points[1];
                let direction = sub(chosen, route_points[0]);
                route_points = vec![chosen, chosen];
                (chosen, direction, Some(chosen))
            } else {
                (route_points[0], current_heading_xy, route_goal)
            }
        } else {
            let advance = advance_along_polyline(&route_points, travel_distance_m)
                .expect("the route always has at least two points");
            route_points = advance.remaining_xy;
            (advance.cursor_xy, advance.direction, route_goal)
        };

        let mut xy_velocity = self.velocity_for(
            last_xy,
            chosen_xy,
            delta_t_s,
            route_direction_xy,
            lead_velocity,
        );
        self.limit_acceleration(&mut xy_velocity, delta_t_s);
        self.blend_orbit_tangent(&mut xy_velocity, last_xy, lead_position);

        let mut chosen_xy = chosen_xy;
        self.push_waypoint_out_of_zones(&mut chosen_xy, &mut xy_velocity, context.zones);

        let (agl_m, vertical_rate_mps) =
            self.altitude_for(timestamp_s, chosen_xy, delta_t_s, context);

        let vertical_phase =
            self.config.vertical_frequency_rad_s * timestamp_s + self.config.phase * 0.5;
        let hold = self
            .config
            .target_altitude_offset_m
            .map(|offset| TargetAltitudeHold {
                position_m: lead_position,
                velocity_mps: Some(lead_velocity),
                offset_m: offset + self.config.vertical_amplitude_m * vertical_phase.sin(),
                rate_offset_mps: Some(vertical_rate_mps),
            });

        let (position, velocity) = compose_air_state(
            context.terrain,
            chosen_xy,
            xy_velocity,
            self.config.min_agl_m,
            agl_m,
            vertical_rate_mps,
            hold,
        );
        let resolution = collision_aware_position(
            context.terrain,
            Some(context.obstacles),
            position,
            self.config.min_agl_m,
        );
        let mut position = resolution.position_m;
        let mut velocity = velocity;

        self.push_position_out_of_zones(&mut position, &mut velocity, context.zones);
        self.apply_follow_floor(
            &mut position,
            &mut velocity,
            current_target_position,
            current_target_velocity,
        );

        self.state.last_timestamp_s = Some(timestamp_s);
        self.state.last_xy_m = Some([position[0], position[1]]);
        self.state.last_direction_xy = Some([xy_velocity[0], xy_velocity[1]]);
        self.state.route_points_xy_m = Some(route_points);
        self.state.route_goal_xy_m = route_goal;
        self.state.cached = Some((timestamp_s, position, velocity));
        (position, velocity)
    }
}

/// A target's position and velocity at one instant.
///
/// The two always travel together — the orbit is centred on the position
/// and the altitude hold follows the velocity — so they are one argument.
#[derive(Debug, Clone, Copy)]
struct TargetSample {
    position_m: [f64; 3],
    velocity_mps: [f64; 3],
}

/// One scored orbit candidate.
struct Candidate {
    /// `0` visible, `1` not. The score's first key, so an invisible
    /// candidate loses to every visible one however much closer it is.
    visibility_rank: u8,
    route_cost_m: f64,
    heading_change_rad: f64,
    xy_m: [f64; 2],
    route_points_xy_m: Option<Vec<[f64; 2]>>,
}

/// What a scan produced: the winner, and the first candidate that was merely
/// inside the map. The latter is the fallback when nothing scored.
struct ScanResult {
    best: Option<Candidate>,
    first_in_bounds_xy: Option<[f64; 2]>,
}

impl FollowPathController {
    /// Score every candidate on the orbit and keep the best.
    ///
    /// The score is the lexicographic triple `(visible ? 0 : 1, route cost,
    /// heading change)`, and each key earns its place: seeing the target
    /// beats being close to it, and among equals the platform prefers not to
    /// turn. Candidates are rejected — not scored badly — for being outside
    /// the map, inside an obstacle, or unreachable by the planner.
    ///
    /// The order of those rejections is load-bearing for the recorded reads:
    /// bounds first (no reads at all), then the obstacle probe, then the
    /// planner, and only then line of sight. A port that tested visibility
    /// before the obstacle would ask about points the reference never
    /// mentioned.
    fn scan_candidates(
        &self,
        timestamp_s: f64,
        base_angle: f64,
        lead: TargetSample,
        current_xy: [f64; 2],
        current_heading_xy: [f64; 2],
        context: &FollowPathContext<'_>,
    ) -> ScanResult {
        let (lead_position, lead_velocity) = (lead.position_m, lead.velocity_mps);
        let margin_m = bounds_margin_m(&self.config);
        let mut best: Option<Candidate> = None;
        let mut first_in_bounds_xy: Option<[f64; 2]> = None;

        for index in 0..self.config.candidate_count {
            let angle = base_angle
                + index as f64 * (std::f64::consts::TAU / self.config.candidate_count as f64);
            let candidate_xy = [
                lead_position[0] + self.config.standoff_radius_m * angle.cos(),
                lead_position[1] + self.config.standoff_radius_m * angle.sin(),
            ];
            if !self.config.map_bounds_m.contains(candidate_xy, margin_m) {
                continue;
            }

            let vertical_phase =
                self.config.vertical_frequency_rad_s * timestamp_s + self.config.phase * 0.5;
            let agl_m =
                self.config.base_agl_m + self.config.vertical_amplitude_m * vertical_phase.sin();
            let hold = self
                .config
                .target_altitude_offset_m
                .map(|offset| TargetAltitudeHold {
                    position_m: lead_position,
                    velocity_mps: Some(lead_velocity),
                    offset_m: offset + self.config.vertical_amplitude_m * vertical_phase.sin(),
                    rate_offset_mps: Some(
                        self.config.vertical_amplitude_m
                            * self.config.vertical_frequency_rad_s
                            * vertical_phase.cos(),
                    ),
                });
            let (candidate_position, _) = compose_air_state(
                context.terrain,
                candidate_xy,
                [0.0, 0.0],
                self.config.min_agl_m,
                agl_m,
                0.0,
                hold,
            );

            if context.obstacles.collides(candidate_position) {
                continue;
            }

            let mut route_points_xy_m: Option<Vec<[f64; 2]>> = None;
            let mut route_cost_m = norm(sub(candidate_xy, current_xy));
            if let Some(planner) = context.planner {
                match planner.plan_route(current_xy, candidate_xy) {
                    Some(route) => {
                        route_cost_m = route.length_m;
                        route_points_xy_m = Some(route.points_xy_m);
                    }
                    // Unreachable, so not a candidate at any price.
                    None => continue,
                }
            }

            if first_in_bounds_xy.is_none() {
                first_in_bounds_xy = Some(candidate_xy);
            }

            let visible = context
                .visibility
                .is_visible(candidate_position, lead_position);

            // The turn is measured against the route's FIRST leg when the
            // planner gave one, not against the straight line to the goal --
            // a route that starts by doubling back costs a turn even if its
            // destination is straight ahead.
            let route_direction_xy = match &route_points_xy_m {
                Some(points) if points.len() > 1 => sub(points[1], points[0]),
                _ => sub(candidate_xy, current_xy),
            };
            let heading_norm = norm(current_heading_xy);
            let route_norm = norm(route_direction_xy);
            let heading_change_rad = if heading_norm > 1.0e-6 && route_norm > 1.0e-6 {
                let cosine = ((current_heading_xy[0] * route_direction_xy[0]
                    + current_heading_xy[1] * route_direction_xy[1])
                    / (heading_norm * route_norm))
                    .clamp(-1.0, 1.0);
                cosine.acos()
            } else {
                0.0
            };

            let candidate = Candidate {
                visibility_rank: u8::from(!visible),
                route_cost_m,
                heading_change_rad,
                xy_m: candidate_xy,
                route_points_xy_m,
            };
            let better = match &best {
                None => true,
                Some(current) => {
                    (
                        candidate.visibility_rank,
                        candidate.route_cost_m,
                        candidate.heading_change_rad,
                    ) < (
                        current.visibility_rank,
                        current.route_cost_m,
                        current.heading_change_rad,
                    )
                }
            };
            if better {
                best = Some(candidate);
            }
        }

        ScanResult {
            best,
            first_in_bounds_xy,
        }
    }

    /// Decide whether to keep the stored route or build a new one.
    ///
    /// Replanning every tick would throw away the route the platform is
    /// partway along, so the reference keeps it until one of three things is
    /// true: there is no usable route stored, the best candidate has drifted
    /// `max(30, standoff / 2)` from the goal the route was built for, or the
    /// planner can no longer reach that goal at all.
    ///
    /// That last check costs a planner call **every tick**, on the stored
    /// goal, separately from the per-candidate calls in the scan. It is the
    /// reason a port that skipped it replays fewer planner reads.
    fn resolve_route(
        &self,
        current_xy: [f64; 2],
        scan: &ScanResult,
        context: &FollowPathContext<'_>,
    ) -> (Vec<[f64; 2]>, Option<[f64; 2]>) {
        let stored_points = self.state.route_points_xy_m.clone();
        let stored_goal = self.state.route_goal_xy_m;

        let mut replan = match &stored_points {
            Some(points) => points.len() < 2,
            None => true,
        };
        match (stored_goal, &scan.best) {
            (Some(goal), Some(best)) => {
                let threshold_m = 30.0_f64.max(self.config.standoff_radius_m * 0.5);
                replan = replan || norm(sub(best.xy_m, goal)) >= threshold_m;
            }
            // No goal yet but a candidate to aim at: always plan.
            //
            // Unreachable as the state stands, and kept deliberately. The
            // goal and the route are written together at the end of every
            // tick and cleared together by `seed` and `reset_state`, so
            // "no goal" always means "no route", which the length test
            // above has already turned into a replan. A mutation run
            // confirms it: deleting this arm changes nothing.
            //
            // `the_goal_and_the_route_are_written_together` pins the
            // invariant that makes it dead, so if a future slice ever
            // clears one without the other, something says so rather than
            // this arm quietly becoming load-bearing.
            (None, Some(_)) => replan = true,
            _ => {}
        }
        if let (Some(planner), Some(goal)) = (context.planner, stored_goal) {
            replan = replan || planner.plan_route(current_xy, goal).is_none();
        }

        if !replan {
            if let Some(points) = stored_points {
                return (points, stored_goal);
            }
        }

        match &scan.best {
            Some(best) => {
                let goal = best.xy_m;
                let points = match &best.route_points_xy_m {
                    Some(points) => points.clone(),
                    None => vec![current_xy, goal],
                };
                (points, Some(goal))
            }
            None => match scan.first_in_bounds_xy {
                // Nothing scored, but something was at least on the map.
                Some(fallback) => (vec![current_xy, fallback], Some(fallback)),
                // Nowhere to go: a degenerate route that holds position.
                None => (vec![current_xy, current_xy], Some(current_xy)),
            },
        }
    }

    /// Horizontal velocity for this tick.
    ///
    /// Two regimes. With a previous position and real elapsed time it is the
    /// finite difference — where the platform actually went, divided by how
    /// long it took. Without either, there is nothing to difference, so the
    /// heading comes from the route and the *speed is the target's*, capped
    /// at the platform's own maximum: a follower matches its target rather
    /// than sprinting on the first tick.
    fn velocity_for(
        &self,
        last_xy: Option<[f64; 2]>,
        chosen_xy: [f64; 2],
        delta_t_s: f64,
        route_direction_xy: [f64; 2],
        lead_velocity: [f64; 3],
    ) -> [f64; 2] {
        let mut xy_velocity = match last_xy {
            Some(last) if delta_t_s > 1.0e-9 => {
                let dt = delta_t_s.max(1e-6);
                [(chosen_xy[0] - last[0]) / dt, (chosen_xy[1] - last[1]) / dt]
            }
            _ => {
                let direction_norm = norm(route_direction_xy);
                let lead_speed_mps = self
                    .config
                    .max_speed_mps
                    .min(norm([lead_velocity[0], lead_velocity[1]]));
                if direction_norm <= 1.0e-6 || lead_speed_mps <= 1.0e-6 {
                    [0.0, 0.0]
                } else {
                    // Divide THEN multiply, as the reference does. Folding
                    // it into one scale factor rounds differently and the
                    // error compounds across ticks.
                    [
                        route_direction_xy[0] / direction_norm * lead_speed_mps,
                        route_direction_xy[1] / direction_norm * lead_speed_mps,
                    ]
                }
            }
        };

        let speed_mps = norm(xy_velocity);
        if speed_mps > self.config.max_speed_mps && speed_mps > 1.0e-6 {
            // `v / speed * max`, not `v * (max / speed)`. See `norm`.
            xy_velocity = [
                xy_velocity[0] / speed_mps * self.config.max_speed_mps,
                xy_velocity[1] / speed_mps * self.config.max_speed_mps,
            ];
        }
        xy_velocity
    }

    /// Limit how fast the *speed* may change, leaving direction alone.
    ///
    /// The reference's comment says why: a platform on an orbit turns
    /// constantly, and an acceleration limit that counted turning would
    /// stop it tracking. So only the scalar changes, and the direction is
    /// free.
    ///
    /// Note this limits deceleration as well, which can make a limited
    /// platform *faster* than an unlimited one at the tick where a sudden
    /// stop is clamped — and that it runs before the orbit blend, which then
    /// rewrites the velocity without updating the stored value. See the
    /// module comment.
    fn limit_acceleration(&mut self, xy_velocity: &mut [f64; 2], delta_t_s: f64) {
        let Some(max_accel_mps2) = self.config.max_accel_mps2 else {
            // The store is only written inside this branch in the reference,
            // so with no limit configured it stays empty — and a limit
            // switched on mid-run therefore has no previous velocity to
            // limit against on its first tick.
            return;
        };
        if let Some(last_velocity) = self.state.last_velocity_xy {
            let last_speed = norm(last_velocity);
            let desired_speed = norm(*xy_velocity);
            let max_speed_change = max_accel_mps2 * delta_t_s.max(1e-6);
            let speed_diff = desired_speed - last_speed;
            if speed_diff.abs() > max_speed_change {
                let clamped_speed = (last_speed + max_speed_change.copysign(speed_diff)).max(0.0);
                if desired_speed > 1e-6 {
                    let scale = clamped_speed / desired_speed;
                    *xy_velocity = [xy_velocity[0] * scale, xy_velocity[1] * scale];
                }
            }
        }
        self.state.last_velocity_xy = Some(*xy_velocity);
    }

    /// Bend the velocity toward the orbit tangent near the standoff radius.
    ///
    /// Without it a circular orbit built from straight route legs looks like
    /// a polygon. The blend ramps in over a band of `standoff * 0.35` either
    /// side of the radius and is ignored below `0.05`, so far-away platforms
    /// fly straight in and only start curving as they arrive.
    ///
    /// The tangent's sense follows the sign of `rotation_rate_rad_s`, and a
    /// rate of exactly zero blends **counter-clockwise** rather than not at
    /// all — `copysign` would return `+1` for `0.0` anyway, but the
    /// reference spells the fallback out and so does this.
    fn blend_orbit_tangent(
        &self,
        xy_velocity: &mut [f64; 2],
        last_xy: Option<[f64; 2]>,
        lead_position: [f64; 3],
    ) {
        if self.config.standoff_radius_m <= 1.0 {
            return;
        }
        let Some(last_xy) = last_xy else {
            return;
        };
        let vec_from_target = sub(last_xy, [lead_position[0], lead_position[1]]);
        let distance_m = norm(vec_from_target);
        if distance_m <= 1e-3 {
            return;
        }
        let orbit_error_m = (distance_m - self.config.standoff_radius_m).abs();
        let blend =
            (1.0 - orbit_error_m / (self.config.standoff_radius_m * 0.35).max(1.0)).max(0.0);
        if blend <= 0.05 {
            return;
        }
        let sign = if self.config.rotation_rate_rad_s.abs() > 1e-9 {
            1.0_f64.copysign(self.config.rotation_rate_rad_s)
        } else {
            1.0
        };
        let tangent = [sign * -vec_from_target[1], sign * vec_from_target[0]];
        let tangent_norm = norm(tangent);
        if tangent_norm <= 1e-6 {
            return;
        }
        // A stationary platform still gets a tangent, at full speed: the
        // blend is what starts it orbiting.
        let mut speed_mps = norm(*xy_velocity);
        if speed_mps < 1e-6 {
            speed_mps = self.config.max_speed_mps;
        }
        let tangent_velocity = [
            tangent[0] / tangent_norm * speed_mps,
            tangent[1] / tangent_norm * speed_mps,
        ];
        *xy_velocity = [
            (1.0 - blend) * xy_velocity[0] + blend * tangent_velocity[0],
            (1.0 - blend) * xy_velocity[1] + blend * tangent_velocity[1],
        ];
        let blended_speed = norm(*xy_velocity);
        if blended_speed > self.config.max_speed_mps + 1e-6 {
            *xy_velocity = [
                xy_velocity[0] / blended_speed * self.config.max_speed_mps,
                xy_velocity[1] / blended_speed * self.config.max_speed_mps,
            ];
        }
    }

    /// Move a waypoint out of any exclusion zone, before the 3-D position is
    /// computed.
    ///
    /// The first of two passes, and they use **different margins**: this one
    /// pushes to `radius * 1.1` and the final one to `radius + 1.0`. On a
    /// 150 m zone that is 165 m versus 151 m — the early pass aims well
    /// clear so the route does not immediately re-enter, the late pass only
    /// has to be outside. Both are the reference's.
    fn push_waypoint_out_of_zones(
        &self,
        chosen_xy: &mut [f64; 2],
        xy_velocity: &mut [f64; 2],
        zones: &[MissionZone],
    ) {
        for zone in zones.iter().filter(|zone| zone.is_exclusion()) {
            let center = [zone.center_m[0], zone.center_m[1]];
            let offset = sub(*chosen_xy, center);
            let mut distance_m = norm(offset);
            if distance_m >= zone.radius_m {
                continue;
            }
            // Dead centre has no outward direction; the reference picks +x.
            let direction = if distance_m < 1e-6 {
                distance_m = 1.0;
                [1.0, 0.0]
            } else {
                [offset[0] / distance_m, offset[1] / distance_m]
            };
            *chosen_xy = [
                center[0] + direction[0] * (zone.radius_m * 1.1),
                center[1] + direction[1] * (zone.radius_m * 1.1),
            ];
            // Point the velocity along the reroute, keeping its magnitude.
            let speed_mps = norm(*xy_velocity);
            if speed_mps > 1e-6 {
                *xy_velocity = [direction[0] * speed_mps, direction[1] * speed_mps];
            }
        }
    }

    /// The hard boundary: move the final position out, and stop it heading
    /// back in.
    ///
    /// Unlike the waypoint pass this does not re-point the velocity — it
    /// removes only the component aimed at the zone centre, leaving the
    /// tangential part intact, so a platform skimming a zone keeps going
    /// around it rather than being turned away.
    fn push_position_out_of_zones(
        &self,
        position: &mut [f64; 3],
        velocity: &mut [f64; 3],
        zones: &[MissionZone],
    ) {
        for zone in zones.iter().filter(|zone| zone.is_exclusion()) {
            let center = [zone.center_m[0], zone.center_m[1]];
            let offset = sub([position[0], position[1]], center);
            let mut distance_m = norm(offset);
            if distance_m >= zone.radius_m {
                continue;
            }
            let direction = if distance_m < 1e-6 {
                distance_m = 1.0;
                [1.0, 0.0]
            } else {
                [offset[0] / distance_m, offset[1] / distance_m]
            };
            position[0] = center[0] + direction[0] * (zone.radius_m + 1.0);
            position[1] = center[1] + direction[1] * (zone.radius_m + 1.0);
            let inward_speed = velocity[0] * -direction[0] + velocity[1] * -direction[1];
            if inward_speed > 0.0 {
                velocity[0] += inward_speed * direction[0];
                velocity[1] += inward_speed * direction[1];
            }
        }
    }

    /// Height above ground for this tick, and the vertical rate.
    ///
    /// Two branches. Terrain following holds a flat `terrain_following_agl_m`
    /// — and computes an exponential moving average of the ground height
    /// that **nothing reads**, which is the finding in the module comment.
    /// Otherwise the height oscillates around `base_agl_m`.
    ///
    /// The vertical rate is the oscillation's derivative and is computed the
    /// same way on both branches, so a terrain-following platform with a
    /// non-zero amplitude reports a climb rate its altitude does not have.
    /// That is the reference's, and the fixture's terrain-following cases
    /// leave the amplitude at zero, so it never bites there.
    fn altitude_for(
        &mut self,
        timestamp_s: f64,
        chosen_xy: [f64; 2],
        delta_t_s: f64,
        context: &FollowPathContext<'_>,
    ) -> (f64, f64) {
        let vertical_phase =
            self.config.vertical_frequency_rad_s * timestamp_s + self.config.phase * 0.5;
        let vertical_rate_mps = self.config.vertical_amplitude_m
            * self.config.vertical_frequency_rad_s
            * vertical_phase.cos();

        let agl_m = if self.config.terrain_following {
            let raw_terrain_z = context.terrain.height_at(chosen_xy[0], chosen_xy[1]);
            let smoothed = match self.state.smoothed_terrain_z {
                Some(previous) if delta_t_s > 1e-9 => {
                    let alpha = 1.0
                        - (-delta_t_s / self.config.terrain_following_smoothing_s.max(0.01)).exp();
                    previous + alpha * (raw_terrain_z - previous)
                }
                _ => raw_terrain_z,
            };
            self.state.smoothed_terrain_z = Some(smoothed);
            self.config.terrain_following_agl_m
        } else {
            self.config.base_agl_m + self.config.vertical_amplitude_m * vertical_phase.sin()
        };

        (agl_m, vertical_rate_mps)
    }

    /// Never sink below a fraction of the way to the target's altitude.
    ///
    /// Only on the target-relative branch. The floor is `target_z + offset *
    /// 0.3` — thirty percent of the offset, not the whole of it, so a
    /// platform is allowed to dip well below its nominal hold but not to
    /// fall to the target's own height. When it fires, the vertical velocity
    /// is raised to at least the target's, so the platform stops descending
    /// relative to what it is following.
    fn apply_follow_floor(
        &self,
        position: &mut [f64; 3],
        velocity: &mut [f64; 3],
        target_position: [f64; 3],
        target_velocity: [f64; 3],
    ) {
        let Some(offset_m) = self.config.target_altitude_offset_m else {
            return;
        };
        let min_follow_z_m = target_position[2] + offset_m * 0.3;
        if position[2] < min_follow_z_m {
            position[2] = min_follow_z_m;
            velocity[2] = velocity[2].max(target_velocity[2]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FlatTerrain;

    impl TerrainSampler for FlatTerrain {
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

    struct NoObstacles;

    impl ObstacleSampler for NoObstacles {
        fn push_clear(&self, _point_m: [f64; 3], _margin_m: f64) -> Option<[f64; 2]> {
            None
        }
        fn collides(&self, _point_m: [f64; 3]) -> bool {
            false
        }
    }

    struct AlwaysVisible;

    impl VisibilitySampler for AlwaysVisible {
        fn is_visible(&self, _observer_m: [f64; 3], _target_m: [f64; 3]) -> bool {
            true
        }
    }

    fn context<'a>(zones: &'a [MissionZone]) -> FollowPathContext<'a> {
        FollowPathContext {
            terrain: &FlatTerrain,
            obstacles: &NoObstacles,
            visibility: &AlwaysVisible,
            planner: None,
            zones,
        }
    }

    fn target(timestamp_s: f64) -> ([f64; 3], [f64; 3]) {
        (
            [200.0 + 6.0 * timestamp_s, 100.0 + 2.0 * timestamp_s, 140.0],
            [6.0, 2.0, 0.0],
        )
    }

    /// The invariant that makes one arm of the replan decision dead code.
    ///
    /// `route_goal_xy_m` and `route_points_xy_m` are written together at the
    /// end of every tick and cleared together by `seed` and `reset_state`.
    /// So "a goal with no route" and "a route with no goal" are both
    /// unreachable, and the `(None, Some(_))` arm of the replan match can
    /// never fire — the length test has already forced a replan by then.
    ///
    /// Pinned because the arm is kept. If a future slice clears one without
    /// the other, this fails rather than that arm silently coming alive.
    #[test]
    fn the_goal_and_the_route_are_written_together() {
        let zones: Vec<MissionZone> = Vec::new();
        let mut controller = FollowPathController::new(FollowPathConfig::default());

        let paired = |state: &FollowPathState| {
            state.route_goal_xy_m.is_some() == state.route_points_xy_m.is_some()
        };
        assert!(paired(controller.state()), "a fresh controller is unpaired");

        for tick in 0..6 {
            controller.sample(f64::from(tick), &mut target, &context(&zones));
            assert!(
                paired(controller.state()),
                "tick {tick} left the goal and the route disagreeing"
            );
            assert!(
                controller.state().route_goal_xy_m.is_some(),
                "tick {tick} produced no goal"
            );
        }

        controller.seed([10.0, 20.0, 100.0], [1.0, 0.0, 0.0], 9.0);
        assert!(paired(controller.state()), "seed left them disagreeing");
        assert!(controller.state().route_goal_xy_m.is_none());

        controller.sample(10.0, &mut target, &context(&zones));
        controller.reset_state();
        assert!(paired(controller.state()), "reset left them disagreeing");
        assert!(controller.state().route_goal_xy_m.is_none());
    }

    /// `terrain_following_smoothing_s` is inert.
    ///
    /// The reference computes an exponential moving average of the ground
    /// height and never reads it. This sweeps the constant across four
    /// orders of magnitude and requires the output not to move — the same
    /// treatment the dead steepness floor got in `motion.rs`.
    ///
    /// The average itself IS computed and stored, and differs between
    /// these runs; that it does not reach the output is the point.
    #[test]
    fn the_terrain_following_filter_does_not_reach_the_output() {
        let zones: Vec<MissionZone> = Vec::new();
        let mut reference: Option<Vec<[f64; 3]>> = None;

        for smoothing_s in [0.01, 0.5, 1.5, 60.0, 600.0] {
            let config = FollowPathConfig {
                terrain_following: true,
                terrain_following_smoothing_s: smoothing_s,
                ..FollowPathConfig::default()
            };
            let mut controller = FollowPathController::new(config);
            controller.seed([500.0, 300.0, 200.0], [15.0, -5.0, 0.0], -1.0);
            let samples: Vec<[f64; 3]> = (0..5)
                .map(|tick| {
                    controller
                        .sample(f64::from(tick), &mut target, &context(&zones))
                        .0
                })
                .collect();

            assert!(
                controller.state().smoothed_terrain_z.is_some(),
                "the average was not even computed"
            );
            match &reference {
                None => reference = Some(samples),
                Some(expected) => assert_eq!(
                    &samples, expected,
                    "smoothing of {smoothing_s} s changed the output -- the EMA is no \
                     longer dead and the port must consume it"
                ),
            }
        }
    }

    /// A second call at the same timestamp costs nothing and answers the
    /// same, and the tolerance is `1e-9` rather than equality.
    #[test]
    fn the_memo_matches_within_a_nanosecond() {
        let zones: Vec<MissionZone> = Vec::new();
        let mut controller = FollowPathController::new(FollowPathConfig::default());
        let first = controller.sample(3.0, &mut target, &context(&zones));

        assert_eq!(controller.sample(3.0, &mut target, &context(&zones)), first);
        // Inside the tolerance: still the memo.
        assert_eq!(
            controller.sample(3.0 + 5e-10, &mut target, &context(&zones)),
            first
        );
        // Outside it: recomputed, and the state moves on.
        let later = controller.sample(3.1, &mut target, &context(&zones));
        assert_ne!(later, first);
    }
}
