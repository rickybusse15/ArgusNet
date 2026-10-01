//! The motion substrate stage 1's controllers are built on.
//!
//! §4 names stage 1 *"sample nominal controllers and account battery **using
//! the prior positions**"*. P4.8c delivered the battery half. This is not yet
//! the controllers either — it is the layer underneath them: the six
//! module-level helpers in `sim.py` that every controller calls.
//!
//! The reference says at its own line 136 why they have never been extracted
//! in Python:
//!
//! ```text
//! Extraction is deferred because the controllers directly call module-level
//! helpers (compose_air_state, collision_aware_position, advance_along_polyline,
//! clamp_to_bounds, within_bounds, _apply_weather_adjustments) that are
//! defined later in this file, which would create a circular import if
//! controllers.py tried to import them back from sim.py.
//! ```
//!
//! That is an import-cycle problem, and Rust does not have it. So the
//! substrate moves first and the controller bodies follow in the next slice
//! with something to stand on.
//!
//! # What is new here, and what is not
//!
//! Almost none of the *arithmetic* is new. The terrain sampling is
//! `argusnet-world`'s already (P4.3), and `push_outside_xy` has been a
//! `call("obstacle", …, method="push")` into Rust since Phase 3 — the Python
//! is a shim over this workspace. What is new is the **composition**: how the
//! altitude, the vertical rate and the collision loop are assembled from
//! those reads. That is what the fixture pins and what the tests below drive.
//!
//! # Capabilities, not a world dependency
//!
//! `argusnet-sim` does not depend on `argusnet-world`, and this module does
//! not add that dependency. The reference's `terrain: object` parameter is
//! duck-typed to exactly three methods — `height_at`, `gradient_at`,
//! `clamp_altitude` — so [`TerrainSampler`] *is* its real contract, written
//! down. [`ObstacleSampler`] is the same for the one obstacle read.
//!
//! This follows P4.8b's truth capability: a stage receives the narrowest
//! thing it can do its job with. It also makes the tests exact, because a
//! stub sampler fails loudly on a read nobody recorded — including a read at
//! the wrong gradient step.

/// The terrain reads the motion helpers make.
///
/// Three methods, because the reference uses three. `gradient_at` takes an
/// `Option<f64>` step rather than a plain `f64` so the default stays the
/// terrain's own business: there are **three** different gradient steps in
/// this codebase (`gradient_at` defaults to `base_resolution * 0.5`,
/// `slope_rad_at` uses 1.0, `slope_rad_at_many` uses
/// `max(base_resolution * 0.5, 0.25)`), and resolving the default here would
/// silently pick one of them.
pub trait TerrainSampler {
    /// Ground height in metres above datum at `(x_m, y_m)`.
    fn height_at(&self, x_m: f64, y_m: f64) -> f64;

    /// Height gradient `[dz/dx, dz/dy]`. `None` means the terrain's default
    /// finite-difference step, which is what the reference passes.
    fn gradient_at(&self, x_m: f64, y_m: f64, delta_m: Option<f64>) -> [f64; 2];

    /// `max(z_m, height_at(x, y) + max(min_agl_m, 0))`.
    fn clamp_altitude(&self, x_m: f64, y_m: f64, z_m: f64, min_agl_m: f64) -> f64;
}

/// The single obstacle read `collision_aware_position` makes.
///
/// Returning the pushed point directly, rather than a handle to push with,
/// keeps this to one call per probe and matches what the loop actually needs.
/// The margin is part of the signature because the reference hardcodes 1.0 at
/// the call site and a port that changed it would otherwise look correct.
pub trait ObstacleSampler {
    /// The point pushed clear of whatever it collides with, or `None` when it
    /// collides with nothing.
    ///
    /// **Not** a pure geometry test: the reference filters on
    /// `blocker_type not in {"building", "wall"}` before testing containment,
    /// so a point squarely inside a forest stand does not collide. See
    /// [`collision_aware_position`].
    fn push_clear(&self, point_m: [f64; 3], margin_m: f64) -> Option<[f64; 2]>;

    /// Whether the point is inside something, without pushing it out.
    ///
    /// Separate from [`ObstacleSampler::push_clear`] because callers differ
    /// in what they need and the difference is observable. A controller
    /// *rejecting* a candidate only asks the question; the reference calls
    /// `point_collides` and never `push_outside_xy` on that path, so a
    /// default implemented in terms of `push_clear` would compute a push
    /// nobody asked for — and against a recorded oracle, would fail on a
    /// read the reference never made.
    fn collides(&self, point_m: [f64; 3]) -> bool;
}

/// The margin `collision_aware_position` pushes with. Hardcoded at the call
/// site in the reference.
const PUSH_MARGIN_M: f64 = 1.0;

/// The iteration cap on the push loop. Eight in the reference, and reaching
/// it is a warning rather than an error — see [`collision_aware_position`].
const MAX_PUSH_ITERATIONS: usize = 8;

/// Compose a full 3-D state from a horizontal state and a height above ground.
///
/// # The vertical rate is coupled to the terrain, not just to the AGL rate
///
/// `vz = dot(gradient, xy_velocity) + agl_rate`. Flying at a constant height
/// above rising ground *is* climbing, and the dot product is that climb. A
/// port that returned `agl_rate` alone would be right on flat ground and
/// wrong everywhere else, which is why the fixture's terrain has relief.
///
/// # The descent guard fires on a conjunction
///
/// ```python
/// if z_m > nominal_z_m and vz_mps < 0.0:
///     vz_mps = 0.0
/// ```
///
/// Only when the clamp actually *raised* the altitude **and** the composed
/// rate is negative. A platform being held up off the ground must not also
/// report that it is descending — but one that is merely descending, with the
/// clamp inactive, keeps its rate. Both halves need to be tested or a port
/// that dropped either one passes; `clamped_and_descending`,
/// `clamped_and_ascending` and `downhill_descent_unclamped` are that trio.
///
/// # Speed adaptation scales the horizontal velocity before anything else
///
/// `max(0.7, 1 - min(|grad|, 1) * 0.3)` — note both saturations. The gradient
/// magnitude is capped at 1.0 **and** the factor is floored at 0.7, and those
/// are the same bound reached twice: with `min(|g|, 1) ∈ [0, 1]` the
/// expression already lies in `[0.7, 1.0]`, so the `max` can never bind.
///
/// The floor is therefore **dead code in the reference**, which a mutation
/// run proved rather than guessed — changing `0.7` to `0.0` survived every
/// fixture case, and no case could have caught it.
/// [`the_steepness_floor_is_unreachable`] pins the property instead, so that
/// if the `0.3` coefficient or the `min` cap ever changes and the floor
/// becomes load-bearing, something says so. It is preserved rather than
/// removed: deleting a line the reference has is a behaviour change if the
/// coefficient later moves, and this slice ports, it does not tidy.
///
/// Crucially the scaled velocity is what then enters the dot product, so
/// adaptation changes the vertical rate too — it is not only a speed cut.
pub fn compose_terrain_state(
    terrain: &dyn TerrainSampler,
    xy_m: [f64; 2],
    xy_velocity_mps: [f64; 2],
    agl_m: f64,
    agl_rate_mps: f64,
    min_agl_m: f64,
    terrain_speed_adaptation: bool,
) -> ([f64; 3], [f64; 3]) {
    let ground_m = terrain.height_at(xy_m[0], xy_m[1]);
    let ground_gradient = terrain.gradient_at(xy_m[0], xy_m[1], None);

    let mut xy_velocity = xy_velocity_mps;
    if terrain_speed_adaptation {
        // `sqrt` of the sum of squares, not `hypot`: the reference uses
        // `np.linalg.norm`, and the two differ by an ULP often enough to
        // compound over a sequence of ticks. See `follow_path::norm`.
        let gradient_magnitude = (ground_gradient[0] * ground_gradient[0]
            + ground_gradient[1] * ground_gradient[1])
            .sqrt();
        let steepness_factor = (1.0 - gradient_magnitude.min(1.0) * 0.3).max(0.7);
        xy_velocity = [
            xy_velocity[0] * steepness_factor,
            xy_velocity[1] * steepness_factor,
        ];
    }

    let nominal_z_m = ground_m + agl_m;
    let z_m = terrain.clamp_altitude(xy_m[0], xy_m[1], nominal_z_m, min_agl_m);

    let mut vz_mps =
        ground_gradient[0] * xy_velocity[0] + ground_gradient[1] * xy_velocity[1] + agl_rate_mps;
    if z_m > nominal_z_m && vz_mps < 0.0 {
        vz_mps = 0.0;
    }

    (
        [xy_m[0], xy_m[1], z_m],
        [xy_velocity[0], xy_velocity[1], vz_mps],
    )
}

/// What a target-following platform should be told to do about altitude.
///
/// Grouped into a struct rather than four more parameters because they are
/// meaningless apart: the reference's guard is
/// `target_position is None or target_altitude_offset_m is None`, an **or**
/// over two of them, so "a target position without an offset" behaves exactly
/// like no target at all. Making them one optional argument means that guard
/// cannot be got wrong — there is no state in which only half is supplied.
///
/// `velocity_mps` stays optional inside, because the reference does
/// distinguish a target whose velocity is unknown (vz contributes 0) from one
/// that is genuinely stationary. Those happen to agree numerically; they are
/// kept separate anyway because the next slice's controllers do not.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TargetAltitudeHold {
    /// The target's position. Only `z` is read.
    pub position_m: [f64; 3],
    /// The target's velocity, if known. Only `z` is read.
    pub velocity_mps: Option<[f64; 3]>,
    /// Metres to hold above the target.
    pub offset_m: f64,
    /// Vertical rate to add on top of the target's own. The reference spells
    /// its fallback `or 0.0`, so `None` and `Some(0.0)` agree.
    pub rate_offset_mps: Option<f64>,
}

/// Compose a 3-D state, optionally holding altitude relative to a target.
///
/// # Without a hold this delegates — and drops two things on the way
///
/// The delegating call passes `nominal_agl_m` as `agl_m` and **does not**
/// forward `terrain_speed_adaptation` at all, so it is always false on this
/// path. Both are the reference's, and both look like omissions until you
/// notice the signature has no adaptation parameter to forward.
///
/// # With a hold, the floor is a plain `max`, not the terrain clamp
///
/// ```python
/// nominal_z_m = max(ground_m + min_agl_m, target[2] + offset)
/// ```
///
/// `clamp_altitude` is never called on this path. It would compute the same
/// number here, and a port that used it would pass any test that only checked
/// the output — which is why the fixture records terrain reads by key and the
/// test asserts on which reads happened, not only on what came back.
///
/// The vertical rate on this path is the target's own `vz` plus the offset
/// rate. It has **no** terrain-gradient term: a platform holding station over
/// a moving target follows the target's climb, not the ground's.
pub fn compose_air_state(
    terrain: &dyn TerrainSampler,
    xy_m: [f64; 2],
    xy_velocity_mps: [f64; 2],
    min_agl_m: f64,
    nominal_agl_m: f64,
    agl_rate_mps: f64,
    hold: Option<TargetAltitudeHold>,
) -> ([f64; 3], [f64; 3]) {
    let Some(hold) = hold else {
        return compose_terrain_state(
            terrain,
            xy_m,
            xy_velocity_mps,
            nominal_agl_m,
            agl_rate_mps,
            min_agl_m,
            // Not forwarded by the reference. See the note above.
            false,
        );
    };

    let ground_m = terrain.height_at(xy_m[0], xy_m[1]);
    let target_vz = hold.velocity_mps.map_or(0.0, |velocity| velocity[2]);
    let nominal_z_m = (ground_m + min_agl_m).max(hold.position_m[2] + hold.offset_m);

    (
        [xy_m[0], xy_m[1], nominal_z_m],
        [
            xy_velocity_mps[0],
            xy_velocity_mps[1],
            target_vz + hold.rate_offset_mps.unwrap_or(0.0),
        ],
    )
}

/// The outcome of a [`collision_aware_position`] call.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CollisionResolution {
    /// The resolved position.
    pub position_m: [f64; 3],
    /// Whether the loop hit [`MAX_PUSH_ITERATIONS`] while still colliding.
    ///
    /// The reference logs a warning and returns the best effort it has. It is
    /// surfaced here instead of logged so the caller — and the safety monitor
    /// in stage 4 — can see that a position is *known* to be unresolved,
    /// rather than having that fact exist only in a log line.
    pub unresolved: bool,
}

/// Push a position clear of obstacles, re-clamping altitude after each push.
///
/// # The loop is the new code
///
/// The push itself is `argusnet-world`'s. What the reference adds is: probe,
/// push, **re-clamp the altitude at the new xy**, probe again, up to eight
/// times. The re-clamp matters because pushing sideways can move the platform
/// over higher ground, and a port that pushed in xy without revisiting z
/// would leave it underground — which CLAUDE.md lists as a hard invariant
/// ("Physical collision must never push entities below terrain").
///
/// # Not converging is not an error
///
/// After eight iterations the reference warns and returns the position it has.
/// It does not raise, and it does not return the original. A platform wedged
/// between two buildings keeps its best-effort position and the run continues.
/// That is preserved; what changes is that the caller is now *told*, via
/// [`CollisionResolution::unresolved`], instead of the fact living in a log.
///
/// # Velocity is returned untouched
///
/// A push moves the platform without changing where it is going. The reference
/// returns `velocity.copy()` on the colliding paths and the original binding
/// when there is no obstacle layer at all; that distinction is invisible to a
/// caller and is not reproduced. Nothing here mutates velocity, so there is
/// nothing to alias.
pub fn collision_aware_position(
    terrain: &dyn TerrainSampler,
    obstacles: Option<&dyn ObstacleSampler>,
    position_m: [f64; 3],
    min_agl_m: f64,
) -> CollisionResolution {
    let Some(obstacles) = obstacles else {
        // No layer: not even a probe. The reference returns before touching
        // the terrain, so a port that clamped altitude here would be adding a
        // terrain read on the commonest path in the simulation.
        return CollisionResolution {
            position_m,
            unresolved: false,
        };
    };

    let mut adjusted = position_m;
    for _ in 0..MAX_PUSH_ITERATIONS {
        let Some(pushed_xy) = obstacles.push_clear(adjusted, PUSH_MARGIN_M) else {
            return CollisionResolution {
                position_m: adjusted,
                unresolved: false,
            };
        };
        adjusted = [
            pushed_xy[0],
            pushed_xy[1],
            terrain.clamp_altitude(pushed_xy[0], pushed_xy[1], adjusted[2], min_agl_m),
        ];
    }

    CollisionResolution {
        position_m: adjusted,
        unresolved: true,
    }
}

/// Where a walk of `travel_distance_m` along `points_xy` ends up.
#[derive(Debug, Clone, PartialEq)]
pub struct PolylineAdvance {
    /// The route from the new cursor onward, with the cursor as its first
    /// vertex, ready to be fed straight back in on the next tick.
    pub remaining_xy: Vec<[f64; 2]>,
    /// The new cursor position.
    pub cursor_xy: [f64; 2],
    /// The unit heading of the segment the walk ended on.
    ///
    /// The **zero vector** when no segment was ever walked — see
    /// [`advance_along_polyline`].
    pub direction: [f64; 2],
}

/// Why a polyline could not be walked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolylineError {
    /// Fewer than two points. The reference raises `ValueError` here, and it
    /// matters that this is an error rather than a no-op: the walk's own
    /// "past the end" return is a two-point polyline precisely so that
    /// feeding it back in does not trip this.
    TooFewPoints(usize),
}

impl std::fmt::Display for PolylineError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolylineError::TooFewPoints(count) => {
                write!(formatter, "points_xy must have n >= 2, got {count}")
            }
        }
    }
}

impl std::error::Error for PolylineError {}

/// The length below which a segment is skipped rather than walked.
const DEGENERATE_SEGMENT_M: f64 = 1.0e-9;

/// Walk `travel_distance_m` along a polyline and report where that lands.
///
/// # Three shapes that look like defects and are not
///
/// **The route shortens as it is walked.** The returned polyline is
/// `[cursor, rest…]` — the cursor *becomes* the first vertex. The caller feeds
/// the result back next tick, so the route is consumed rather than re-walked
/// from the start each time.
///
/// **Running off the end returns the last point twice.** A one-point polyline
/// would fail this function's own validator on the next call, so the
/// degenerate return is deliberately two points. A port that returned one
/// would work until the drone reached the end of its route.
///
/// **The boundary is inclusive.** `remaining <= segment_length` resolves a
/// walk that lands exactly on a vertex to the segment *arriving* at it, and
/// reports that segment's heading. With `<` it would resolve to the next
/// segment and report the next heading — plausible, different, and only
/// visible at the exact vertex.
///
/// # The direction when nothing was walked
///
/// `last_direction` starts at zero and is only set by a segment long enough to
/// walk. A polyline of coincident points therefore returns a zero direction,
/// not a unit vector and not the previous heading. Callers treat zero as "no
/// heading"; a port that returned the final segment's direction would give a
/// heading to a platform that has nowhere to go.
pub fn advance_along_polyline(
    points_xy: &[[f64; 2]],
    travel_distance_m: f64,
) -> Result<PolylineAdvance, PolylineError> {
    if points_xy.len() < 2 {
        return Err(PolylineError::TooFewPoints(points_xy.len()));
    }

    // Negative distance is floored, not walked backwards.
    let mut remaining_distance = travel_distance_m.max(0.0);
    let mut cursor = points_xy[0];
    let mut last_direction = [0.0, 0.0];

    for index in 0..points_xy.len() - 1 {
        let endpoint = points_xy[index + 1];
        let segment = [endpoint[0] - cursor[0], endpoint[1] - cursor[1]];
        // `np.linalg.norm`, not `hypot`. See `follow_path::norm`.
        let segment_length = (segment[0] * segment[0] + segment[1] * segment[1]).sqrt();
        if segment_length <= DEGENERATE_SEGMENT_M {
            // Skipped without consuming distance and without setting the
            // direction — a coincident vertex is not a heading.
            cursor = endpoint;
            continue;
        }
        let direction = [segment[0] / segment_length, segment[1] / segment_length];
        last_direction = direction;
        if remaining_distance <= segment_length {
            let new_xy = [
                cursor[0] + direction[0] * remaining_distance,
                cursor[1] + direction[1] * remaining_distance,
            ];
            let mut remaining_xy = Vec::with_capacity(points_xy.len() - index);
            remaining_xy.push(new_xy);
            remaining_xy.extend_from_slice(&points_xy[index + 1..]);
            return Ok(PolylineAdvance {
                remaining_xy,
                cursor_xy: new_xy,
                direction,
            });
        }
        remaining_distance -= segment_length;
        cursor = endpoint;
    }

    let last = points_xy[points_xy.len() - 1];
    Ok(PolylineAdvance {
        remaining_xy: vec![last, last],
        cursor_xy: last,
        direction: last_direction,
    })
}

/// An axis-aligned horizontal extent, in metres.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundsXy {
    pub x_min_m: f64,
    pub x_max_m: f64,
    pub y_min_m: f64,
    pub y_max_m: f64,
}

impl BoundsXy {
    /// Whether `xy_m` lies inside, inset by `margin_m`.
    ///
    /// Inclusive on every edge. With a margin wider than half the span the
    /// inset range inverts and nothing is inside — which is the answer, not a
    /// bug: it is [`clamp_to_bounds`](BoundsXy::clamp) that has to do
    /// something sensible with an inverted range, not this.
    pub fn contains(&self, xy_m: [f64; 2], margin_m: f64) -> bool {
        xy_m[0] >= self.x_min_m + margin_m
            && xy_m[0] <= self.x_max_m - margin_m
            && xy_m[1] >= self.y_min_m + margin_m
            && xy_m[1] <= self.y_max_m - margin_m
    }

    /// Clamp `xy_m` into the bounds, inset by `margin_m`.
    ///
    /// # The collapse, and why it is per-axis
    ///
    /// When the margin exceeds half a span the inset range inverts, and the
    /// reference collapses that axis to the **midpoint of the original
    /// bounds** rather than letting a clamp see `min > max`. Its own comment
    /// says `np.clip` "handles [that] inconsistently", so the midpoint is the
    /// defined behaviour rather than a guard against a crash.
    ///
    /// The collapse is decided per axis. A margin can invert a narrow y span
    /// while a wide x span still clamps normally, and the fixture's bounds are
    /// deliberately asymmetric — with different midpoints — so that a port
    /// collapsing both axes together cannot produce the same numbers.
    pub fn clamp(&self, xy_m: [f64; 2], margin_m: f64) -> [f64; 2] {
        let (mut min_x, mut max_x) = (self.x_min_m + margin_m, self.x_max_m - margin_m);
        let (mut min_y, mut max_y) = (self.y_min_m + margin_m, self.y_max_m - margin_m);
        if min_x > max_x {
            let midpoint = (self.x_min_m + self.x_max_m) * 0.5;
            min_x = midpoint;
            max_x = midpoint;
        }
        if min_y > max_y {
            let midpoint = (self.y_min_m + self.y_max_m) * 0.5;
            min_y = midpoint;
            max_y = midpoint;
        }
        [xy_m[0].clamp(min_x, max_x), xy_m[1].clamp(min_y, max_y)]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A terrain with a chosen constant gradient and a flat ground plane.
    struct ConstantSlope {
        gradient: [f64; 2],
    }

    impl TerrainSampler for ConstantSlope {
        fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
            0.0
        }
        fn gradient_at(&self, _x_m: f64, _y_m: f64, _delta_m: Option<f64>) -> [f64; 2] {
            self.gradient
        }
        fn clamp_altitude(&self, _x_m: f64, _y_m: f64, z_m: f64, min_agl_m: f64) -> f64 {
            z_m.max(min_agl_m.max(0.0))
        }
    }

    /// The `max(0.7, …)` in the steepness factor can never bind.
    ///
    /// `min(|g|, 1) ∈ [0, 1]`, so `1 - min(|g|, 1) * 0.3 ∈ [0.7, 1.0]` before
    /// the floor is applied. This is why the `0.7 -> 0.0` mutant survives the
    /// parity fixture: the two are the same function.
    ///
    /// The sweep goes well past a gradient magnitude of 1 — a 45° slope is
    /// `|g| = 1`, and terrain steeper than that is what would reach the cap
    /// if anything did.
    #[test]
    fn the_steepness_floor_is_unreachable() {
        for step in 0..=200 {
            let magnitude = f64::from(step) * 0.05;
            let terrain = ConstantSlope {
                gradient: [magnitude, 0.0],
            };
            // Unit horizontal speed, so the returned speed IS the factor.
            let (_, with) =
                compose_terrain_state(&terrain, [0.0, 0.0], [1.0, 0.0], 100.0, 0.0, 0.0, true);
            let factor = with[0];
            assert!(
                (0.7..=1.0).contains(&factor),
                "|g| = {magnitude}: factor {factor} left [0.7, 1.0]"
            );
            // The floor is redundant: applying it changes nothing.
            let unfloored = 1.0 - magnitude.min(1.0) * 0.3;
            assert_eq!(
                factor, unfloored,
                "|g| = {magnitude}: the floor bound, so it is no longer dead code"
            );
        }
    }

    /// Adaptation off and on are the same function when the ground is flat.
    ///
    /// Guards the pair of fixture cases that isolate adaptation: if this ever
    /// fails, the flag is doing something beyond scaling by the gradient and
    /// those cases no longer isolate what they claim to.
    #[test]
    fn adaptation_is_identity_on_flat_ground() {
        let terrain = ConstantSlope {
            gradient: [0.0, 0.0],
        };
        let off = compose_terrain_state(&terrain, [0.0, 0.0], [9.0, -4.0], 50.0, 1.0, 10.0, false);
        let on = compose_terrain_state(&terrain, [0.0, 0.0], [9.0, -4.0], 50.0, 1.0, 10.0, true);
        assert_eq!(off, on);
    }

    /// A polyline of fewer than two points is an error, and the walk's own
    /// "past the end" return is two points so that feeding it back in does
    /// not trip that error. The pair is the contract; test them together.
    #[test]
    fn the_end_of_a_route_can_be_walked_again() {
        let route = [[0.0, 0.0], [10.0, 0.0]];
        let first = advance_along_polyline(&route, 999.0).expect("walkable");
        assert_eq!(first.remaining_xy, vec![[10.0, 0.0], [10.0, 0.0]]);
        // The returned route is still walkable -- this is the reason for the
        // duplicate point, and a one-point return would panic here.
        let again = advance_along_polyline(&first.remaining_xy, 5.0).expect("still walkable");
        assert_eq!(again.cursor_xy, [10.0, 0.0]);
        assert_eq!(
            again.direction,
            [0.0, 0.0],
            "a stalled route has no heading"
        );

        assert_eq!(
            advance_along_polyline(&[[0.0, 0.0]], 1.0),
            Err(PolylineError::TooFewPoints(1))
        );
        assert_eq!(
            advance_along_polyline(&[], 1.0),
            Err(PolylineError::TooFewPoints(0))
        );
    }

    /// A platform that cannot be pushed clear is reported, not silently
    /// returned as if resolved.
    ///
    /// The reference logs a warning and returns its best effort; the port
    /// returns the same position and sets `unresolved`. No fixture case
    /// wedges, so this is the only test of the exhaustion path.
    struct AlwaysColliding;

    impl ObstacleSampler for AlwaysColliding {
        fn push_clear(&self, point_m: [f64; 3], _margin_m: f64) -> Option<[f64; 2]> {
            // Pushes somewhere, and it is never enough.
            Some([point_m[0] + 1.0, point_m[1]])
        }
        fn collides(&self, _point_m: [f64; 3]) -> bool {
            true
        }
    }

    #[test]
    fn a_wedged_platform_is_reported_unresolved() {
        let terrain = ConstantSlope {
            gradient: [0.0, 0.0],
        };
        let resolution =
            collision_aware_position(&terrain, Some(&AlwaysColliding), [0.0, 0.0, 100.0], 10.0);
        assert!(resolution.unresolved, "exhaustion was not reported");
        // Eight pushes of 1 m each, and the position is kept rather than
        // reverted -- best effort, exactly as the reference returns it.
        assert_eq!(resolution.position_m[0], MAX_PUSH_ITERATIONS as f64);
    }

    /// The no-obstacle-layer path does not touch the terrain at all.
    ///
    /// This is the commonest path in the simulation, and a terrain read added
    /// here would cost one sample per platform per tick. A sampler that
    /// panics on any read is the cheapest way to assert "nothing was read".
    struct PanickingTerrain;

    impl TerrainSampler for PanickingTerrain {
        fn height_at(&self, _x_m: f64, _y_m: f64) -> f64 {
            panic!("the no-layer path read the terrain");
        }
        fn gradient_at(&self, _x_m: f64, _y_m: f64, _delta_m: Option<f64>) -> [f64; 2] {
            panic!("the no-layer path read the terrain");
        }
        fn clamp_altitude(&self, _x_m: f64, _y_m: f64, _z_m: f64, _min_agl_m: f64) -> f64 {
            panic!("the no-layer path read the terrain");
        }
    }

    #[test]
    fn no_obstacle_layer_reads_no_terrain() {
        let resolution = collision_aware_position(&PanickingTerrain, None, [1.0, 2.0, 3.0], 10.0);
        assert_eq!(resolution.position_m, [1.0, 2.0, 3.0]);
        assert!(!resolution.unresolved);
    }
}
