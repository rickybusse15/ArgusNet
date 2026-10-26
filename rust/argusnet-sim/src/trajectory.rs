//! The outermost two layers of a controller.
//!
//! P4.8d.1 ported the kinematic substrate; this is the pair of wrappers that
//! sit around whatever a controller computes. `_apply_weather_adjustments`
//! applies wind and weather to a finished sample, and
//! `LaunchableTrajectoryController` decides whether the platform has left the
//! ground at all. Both are named at `sim.py` line 136 as helpers that block
//! controller extraction, and neither depends on the controller bodies — so
//! they move first and `FollowPathController` lands on top of them.
//!
//! # Weather is a capability, not a port
//!
//! The same decision as [`crate::motion`]'s terrain: a narrow
//! [`WeatherSampler`] trait rather than a dependency on a ported weather
//! model. `argusnet-world` has no weather yet, and the full model
//! (atmosphere, precipitation, clouds, visibility) is a sensing concern whose
//! port is not this slice's. Motion needs exactly two numbers out of it, so
//! exactly two numbers are the contract, and the fixture records the reads as
//! an oracle.
//!
//! When the weather model does move, it implements this trait and nothing
//! here changes.

use crate::motion::{BoundsXy, TerrainSampler};

/// Something that can be asked where a platform is at a given time.
///
/// `&mut self` because controllers are stateful — `FollowPathController`
/// carries a cached timestamp and a rotation phase — even though the plain
/// analytic trajectories are not. The reference's `TrajectoryFn` is a
/// callable that may or may not have a `reset_state`, which is the same
/// thing said less precisely.
pub trait Trajectory {
    /// Position and velocity, both in metres and metres per second.
    fn sample(&mut self, timestamp_s: f64) -> ([f64; 3], [f64; 3]);

    /// Return to the state a freshly constructed trajectory would have.
    ///
    /// The reference reaches this through `hasattr(…, "reset_state")`, so a
    /// stateless trajectory simply has no such method. Here the default is
    /// that declaration.
    fn reset_state(&mut self) {}
}

/// The two weather reads the motion path makes.
///
/// Deliberately not the whole `WeatherModel`. Note the asymmetry between
/// them, which is the reference's and is easy to lose: `flight_speed_penalty`
/// internally evaluates the wind at **time zero**, so the gust phase affects
/// the drift and not the penalty. That is why the penalty takes no time
/// argument here — giving it one would invite a port to pass the current
/// timestamp and quietly couple the two.
pub trait WeatherSampler {
    /// Multiplicative speed factor in `(0, 1]`, `1` meaning no penalty.
    fn flight_speed_penalty(&self, altitude_m: f64) -> f64;

    /// Wind vector `[east, north, up]` in m/s. The vertical component is
    /// always zero in the reference's model, and is carried rather than
    /// dropped because nothing here is entitled to assume that stays true.
    fn wind_at(&self, altitude_m: f64, time_s: f64) -> [f64; 3];
}

/// Everything `apply_weather_adjustments` needs besides the sample itself.
///
/// Grouped because the reference's call has eight keyword arguments and
/// three of them are optional in ways that change the behaviour rather than
/// just the numbers.
pub struct WeatherContext<'a> {
    /// `None` makes the whole adjustment a passthrough.
    pub weather: Option<&'a dyn WeatherSampler>,
    /// `None` changes what "altitude" means — see
    /// [`apply_weather_adjustments`].
    pub terrain: Option<&'a dyn TerrainSampler>,
    /// `None` leaves the drift unclamped horizontally.
    pub bounds: Option<BoundsXy>,
    pub min_agl_m: f64,
    /// How much of one timestep's wind is applied as position drift.
    pub wind_scale: f64,
    /// The timestep the drift is integrated over.
    ///
    /// **Not** the simulation timestep in every caller:
    /// [`LaunchController`] never passes one, so it always gets the
    /// reference's default of `1.0` whatever `dt` the run uses. Preserved,
    /// not corrected — see [`LaunchController::sample`].
    pub dt_s: f64,
}

impl WeatherContext<'_> {
    /// The reference's defaults for the two scalars that have them.
    pub const DEFAULT_WIND_SCALE: f64 = 0.35;
    pub const DEFAULT_DT_S: f64 = 1.0;
}

/// The bounds margin `_apply_weather_adjustments` clamps with, hardcoded at
/// its call site in the reference.
const BOUNDS_MARGIN_M: f64 = 1.0;

/// Apply wind drift and a weather speed penalty to a finished sample.
///
/// # The terrain is read twice, and the order is the point
///
/// ```python
/// # Clamp position above terrain first so speed penalty uses the true AGL.
/// ```
///
/// The reference says why in a comment. The platform is lifted to
/// `ground + min_agl` *before* the altitude is computed, so a sample that
/// came back underground does not ask for the speed penalty at a negative
/// altitude. Then the wind moves it horizontally, over different ground, and
/// it is lifted again. A port that clamped once produces a plausible number
/// and can end up below the terrain — which CLAUDE.md lists as a hard
/// invariant.
///
/// # "Altitude" means two different things
///
/// With terrain, it is height above ground. Without, it is the raw `z`. That
/// is inconsistent, it is the reference's, and both weather reads get
/// whichever one applies. A port that always computed AGL would be *more*
/// correct and would disagree with every recorded value in the no-terrain
/// case.
///
/// # Only the horizontal moves
///
/// The speed penalty scales `vx`/`vy`; the drift moves `x`/`y`. `vz` is never
/// touched, and `z` only by the two clamps. A drone in a gale does not start
/// climbing.
///
/// # The drift is per-timestep, not per-clock
///
/// ```python
/// # Apply wind as a velocity-based drift over one timestep (not cumulative
/// # with absolute time — that was a bug causing drift to grow linearly
/// # forever).
/// ```
///
/// Also the reference's comment, also worth keeping: the drift is
/// `wind * dt * scale`, not `wind * timestamp * scale`. The timestamp reaches
/// `wind_at` to set the gust phase and goes no further.
pub fn apply_weather_adjustments(
    position_m: [f64; 3],
    velocity_mps: [f64; 3],
    timestamp_s: f64,
    context: &WeatherContext<'_>,
) -> ([f64; 3], [f64; 3]) {
    let Some(weather) = context.weather else {
        return (position_m, velocity_mps);
    };

    let mut position = position_m;
    let mut velocity = velocity_mps;

    // First clamp, and the altitude the weather is asked about.
    let altitude_m = match context.terrain {
        Some(terrain) => {
            let ground_m = terrain.height_at(position[0], position[1]);
            position[2] = position[2].max(ground_m + context.min_agl_m);
            position[2] - ground_m
        }
        None => position[2],
    };

    let penalty = weather.flight_speed_penalty(altitude_m);
    velocity[0] *= penalty;
    velocity[1] *= penalty;

    let wind = weather.wind_at(altitude_m, timestamp_s);
    position[0] += wind[0] * context.dt_s * context.wind_scale;
    position[1] += wind[1] * context.dt_s * context.wind_scale;

    if let Some(bounds) = context.bounds {
        let clamped = bounds.clamp([position[0], position[1]], BOUNDS_MARGIN_M);
        position[0] = clamped[0];
        position[1] = clamped[1];
    }

    // Second clamp, over whatever ground the drift moved it onto.
    if let Some(terrain) = context.terrain {
        let ground_m = terrain.height_at(position[0], position[1]);
        position[2] = position[2].max(ground_m + context.min_agl_m);
    }

    (position, velocity)
}

/// Holds a platform on the ground, climbs it, then delegates.
///
/// # Three phases and two boundaries
///
/// Parked until [`LaunchController::trigger_launch`], climbing for
/// `climb_duration_s`, then operational. Both boundaries are strict:
/// `elapsed < 0` parks (a sample from before the launch instant is not a
/// negative climb), and `elapsed < climb_duration_s` climbs — so a sample at
/// *exactly* the climb duration is the first operational one. A port using
/// `<=` delegates one tick late, which is invisible except at that instant.
///
/// # The wrapped trajectory is not consulted during the climb
///
/// Not an optimisation. The inner trajectory may be stateful, so asking it
/// for a sample the platform is not flying is a side effect, not just waste.
/// The fixture records the inner call timestamps for exactly this reason.
///
/// # The climb is a smooth step, and its rate is the analytic derivative
///
/// Height is `A * (3f² - 2f³)` and rate is `A * (6f - 6f²) / duration`, which
/// is **zero at both ends** — the climb starts and finishes at zero vertical
/// speed, and is symmetric about the midpoint. A linear ramp would be close
/// in position and wrong in velocity everywhere.
#[derive(Debug, Clone, PartialEq)]
pub struct LaunchController<T> {
    station_position_m: [f64; 3],
    operational: T,
    climb_duration_s: f64,
    operational_altitude_agl_m: f64,
    launched_at_s: Option<f64>,
    launch_target_id: Option<String>,
}

impl<T> LaunchController<T> {
    /// The reference's defaults.
    pub const DEFAULT_CLIMB_DURATION_S: f64 = 8.0;
    pub const DEFAULT_OPERATIONAL_ALTITUDE_AGL_M: f64 = 230.0;

    pub fn new(
        station_position_m: [f64; 3],
        operational: T,
        climb_duration_s: f64,
        operational_altitude_agl_m: f64,
    ) -> Self {
        Self {
            station_position_m,
            operational,
            climb_duration_s,
            operational_altitude_agl_m,
            launched_at_s: None,
            launch_target_id: None,
        }
    }

    /// Climb to no more than `agl_m` above the station.
    pub fn cap_operational_altitude(&mut self, agl_m: f64) {
        self.operational_altitude_agl_m = self.operational_altitude_agl_m.min(agl_m);
    }

    /// Launch, once.
    ///
    /// A second call is ignored — including its target id. The reference
    /// guards on `if not self.launched`, so a re-tasked platform keeps the
    /// launch time and the target it originally launched against.
    pub fn trigger_launch(&mut self, timestamp_s: f64, target_id: impl Into<String>) {
        if self.launched_at_s.is_none() {
            self.launched_at_s = Some(timestamp_s);
            self.launch_target_id = Some(target_id.into());
        }
    }

    pub fn is_launched(&self) -> bool {
        self.launched_at_s.is_some()
    }

    pub fn launch_target_id(&self) -> Option<&str> {
        self.launch_target_id.as_deref()
    }

    /// Sample, applying weather only on the operational phase.
    ///
    /// Note what `context.dt_s` will be: the reference calls
    /// `_apply_weather_adjustments` without a `dt_s`, so the launch path
    /// always integrates the drift over **one second** regardless of the
    /// simulation timestep. That is preserved rather than corrected —
    /// changing it would move every existing run, and it belongs in a
    /// decision of its own rather than arriving inside a port. The caller
    /// supplies the context, so the caller is where that decision lands.
    pub fn sample(&mut self, timestamp_s: f64, context: &WeatherContext<'_>) -> ([f64; 3], [f64; 3])
    where
        T: Trajectory,
    {
        self.sample_with(timestamp_s, context, |inner, t| inner.sample(t))
    }

    /// [`LaunchController::sample`] with the operational half sampled by the
    /// caller, for an inner controller that needs per-tick context the
    /// [`Trajectory`] trait cannot carry (the stage lends it the world).
    ///
    /// `operational` is called only in the operational phase -- never while
    /// parked or climbing -- which is the reference's behaviour and matters
    /// because the inner controller may be stateful.
    pub fn sample_with(
        &mut self,
        timestamp_s: f64,
        context: &WeatherContext<'_>,
        operational: impl FnOnce(&mut T, f64) -> ([f64; 3], [f64; 3]),
    ) -> ([f64; 3], [f64; 3]) {
        let Some(launched_at_s) = self.launched_at_s else {
            return (self.station_position_m, [0.0; 3]);
        };

        let elapsed_s = timestamp_s - launched_at_s;
        if elapsed_s < 0.0 {
            return (self.station_position_m, [0.0; 3]);
        }

        if elapsed_s < self.climb_duration_s {
            let fraction = elapsed_s / self.climb_duration_s;
            let smooth = 3.0 * fraction * fraction - 2.0 * fraction * fraction * fraction;
            let climb_rate_mps = self.operational_altitude_agl_m
                * (6.0 * fraction - 6.0 * fraction * fraction)
                / self.climb_duration_s;
            return (
                [
                    self.station_position_m[0],
                    self.station_position_m[1],
                    self.station_position_m[2] + self.operational_altitude_agl_m * smooth,
                ],
                [0.0, 0.0, climb_rate_mps],
            );
        }

        let (position, velocity) = operational(&mut self.operational, timestamp_s);
        apply_weather_adjustments(position, velocity, timestamp_s, context)
    }

    /// The wrapped trajectory, for callers that need to reach past this one.
    pub fn operational(&self) -> &T {
        &self.operational
    }

    pub fn operational_mut(&mut self) -> &mut T {
        &mut self.operational
    }

    /// When the launch happened, if it has.
    pub fn launched_at_s(&self) -> Option<f64> {
        self.launched_at_s
    }

    /// Take a checkpoint's launch state back. The operational half is
    /// restored by whoever owns its type.
    pub fn restore_launch(&mut self, launched_at_s: Option<f64>, target_id: Option<String>) {
        self.launched_at_s = launched_at_s;
        self.launch_target_id = target_id;
    }

    /// Back on the ground, leaving the operational half alone.
    pub fn reset_launch(&mut self) {
        self.launched_at_s = None;
        self.launch_target_id = None;
    }

    /// Return to the ground, and reset what is underneath.
    pub fn reset_state(&mut self)
    where
        T: Trajectory,
    {
        self.reset_launch();
        self.operational.reset_state();
    }
}

/// A trajectory with weather applied to whatever it produces.
///
/// The reference's `WeatherAdjustedTrajectory`, which differs from the launch
/// path in exactly one way worth naming: it carries its own `dt_s` field, so
/// a caller can integrate the drift over the real timestep. The launch path
/// cannot. Two defaults, one of them unreachable from the other route.
pub struct WeatherAdjusted<T: Trajectory> {
    inner: T,
}

impl<T: Trajectory> WeatherAdjusted<T> {
    pub fn new(inner: T) -> Self {
        Self { inner }
    }

    pub fn sample(
        &mut self,
        timestamp_s: f64,
        context: &WeatherContext<'_>,
    ) -> ([f64; 3], [f64; 3]) {
        let (position, velocity) = self.inner.sample(timestamp_s);
        apply_weather_adjustments(position, velocity, timestamp_s, context)
    }

    /// Resets pass through to the wrapped trajectory. The wrapper itself
    /// holds nothing.
    pub fn reset_state(&mut self) {
        self.inner.reset_state();
    }

    pub fn inner(&self) -> &T {
        &self.inner
    }
}
