//! The integer tick clock.
//!
//! [`ARCHITECTURE_AND_PROTOCOLS.md`](../../../docs/migration/ARCHITECTURE_AND_PROTOCOLS.md)
//! §4: *"Simulation time is derived from integer tick and a fixed, validated
//! timestep; timestamps at interchange boundaries are derived seconds."*
//!
//! The failure that rule guards against is accumulation — `t += dt` every tick,
//! where the error grows without bound and two runs that took different paths
//! to the same tick disagree about what time it is. Deriving `tick * dt`
//! afresh has no such state, and it is also exactly what `sim.py` does
//! (`timestamp_s = step * simulation_config.dt_s`), so the port introduces no
//! difference here at all.

use std::fmt;

/// A validated fixed timestep.
///
/// Validated once at construction so no stage has to re-check it, and so an
/// invalid configuration fails before any state is mutated rather than
/// producing `NaN` timestamps several stages in.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Timestep {
    seconds: f64,
}

/// Why a timestep was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimestepError {
    /// `NaN` or an infinity.
    NotFinite,
    /// Zero or negative: a tick that does not advance time never terminates,
    /// and a negative one runs the mission backwards through code that assumes
    /// otherwise.
    NotPositive,
    /// Above [`Timestep::MAX_SECONDS`].
    TooLarge,
}

impl fmt::Display for TimestepError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimestepError::NotFinite => write!(formatter, "timestep must be finite"),
            TimestepError::NotPositive => write!(formatter, "timestep must be greater than zero"),
            TimestepError::TooLarge => write!(
                formatter,
                "timestep must be at most {} s",
                Timestep::MAX_SECONDS
            ),
        }
    }
}

impl std::error::Error for TimestepError {}

impl Timestep {
    /// One hour. Not a physical claim — a guard against a timestep that is
    /// really a duration in disguise (someone passing `--duration-s` where
    /// `--dt-s` was meant), which otherwise runs one enormous tick and reports
    /// a finished mission.
    pub const MAX_SECONDS: f64 = 3600.0;

    pub fn new(seconds: f64) -> Result<Self, TimestepError> {
        if !seconds.is_finite() {
            return Err(TimestepError::NotFinite);
        }
        if seconds <= 0.0 {
            return Err(TimestepError::NotPositive);
        }
        if seconds > Self::MAX_SECONDS {
            return Err(TimestepError::TooLarge);
        }
        Ok(Self { seconds })
    }

    pub fn seconds(self) -> f64 {
        self.seconds
    }
}

/// Monotonic tick counter plus the timestep that gives ticks their meaning.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Clock {
    tick: u64,
    timestep: Timestep,
}

impl Clock {
    pub fn new(timestep: Timestep) -> Self {
        Self { tick: 0, timestep }
    }

    pub fn tick(self) -> u64 {
        self.tick
    }

    pub fn timestep(self) -> Timestep {
        self.timestep
    }

    /// Simulation time in seconds at the current tick.
    ///
    /// Derived, never accumulated — see the module comment. Matches
    /// `sim.py`'s `step * dt_s` bit-for-bit, both being an IEEE-754 double
    /// multiply.
    pub fn timestamp_s(self) -> f64 {
        self.tick as f64 * self.timestep.seconds
    }

    /// Advance one tick.
    pub fn advance(&mut self) {
        self.tick += 1;
    }

    /// Restore a validated checkpoint tick without replaying clock increments.
    pub(crate) fn restore_tick(&mut self, tick: u64) {
        self.tick = tick;
    }

    /// Return to tick zero, keeping the timestep.
    pub fn reset(&mut self) {
        self.tick = 0;
    }

    /// The millisecond value the seed derivation keys on.
    ///
    /// `build_observations` keys its per-`(node, target, tick)` stream on
    /// `round(timestamp_s * 1000.0)`, and ADR-006 keeps that scheme, so this
    /// has to agree with Python — including on the rounding rule. Python's
    /// `round()` is **half-to-even**; Rust's `f64::round()` is half-away-from-
    /// zero. They differ whenever `timestamp_s * 1000` lands exactly on a half,
    /// which a sub-millisecond timestep reaches immediately: at `dt = 0.0025`
    /// the first tick is `2.5 ms`, and the two rules disagree about whether
    /// that is 2 or 3.
    pub fn seed_time_key(self) -> i64 {
        round_half_to_even(self.timestamp_s() * 1000.0) as i64
    }
}

/// Python's rounding rule: ties go to the even integer.
///
/// Written out rather than reached for, because the only reason it is here is
/// to agree with a specific reference implementation, and a reader who does not
/// know that would reasonably "simplify" it to `f64::round`.
fn round_half_to_even(value: f64) -> f64 {
    let rounded = value.round(); // half away from zero
    if (value - value.trunc()).abs() == 0.5 && rounded % 2.0 != 0.0 {
        // A tie that landed on an odd integer: step one toward the even side.
        rounded - value.signum()
    } else {
        rounded
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_timestep_must_be_finite_positive_and_bounded() {
        assert_eq!(Timestep::new(f64::NAN), Err(TimestepError::NotFinite));
        assert_eq!(Timestep::new(f64::INFINITY), Err(TimestepError::NotFinite));
        assert_eq!(Timestep::new(0.0), Err(TimestepError::NotPositive));
        assert_eq!(Timestep::new(-0.1), Err(TimestepError::NotPositive));
        assert_eq!(
            Timestep::new(Timestep::MAX_SECONDS + 1.0),
            Err(TimestepError::TooLarge)
        );
        assert!(Timestep::new(0.1).is_ok());
        assert!(Timestep::new(Timestep::MAX_SECONDS).is_ok());
    }

    #[test]
    fn time_is_derived_from_the_tick_not_accumulated() {
        let mut clock = Clock::new(Timestep::new(0.1).expect("valid"));
        for _ in 0..1_000 {
            clock.advance();
        }
        // Accumulating 0.1 a thousand times drifts into the 1e-13s; deriving
        // does not, because there is only ever one multiply.
        assert_eq!(clock.timestamp_s(), 1_000f64 * 0.1);

        let mut accumulated = 0.0f64;
        for _ in 0..1_000 {
            accumulated += 0.1;
        }
        assert_ne!(
            accumulated,
            clock.timestamp_s(),
            "if these ever agree the test has stopped demonstrating anything"
        );
    }

    #[test]
    fn timestamp_matches_the_python_expression() {
        // sim.py: timestamp_s = step * simulation_config.dt_s
        for dt in [0.1f64, 0.05, 0.0025, 1.0 / 3.0] {
            let timestep = Timestep::new(dt).expect("valid");
            let mut clock = Clock::new(timestep);
            for step in 0..50u64 {
                assert_eq!(clock.timestamp_s(), step as f64 * dt, "dt={dt} step={step}");
                clock.advance();
            }
        }
    }

    #[test]
    fn seed_time_key_uses_pythons_rounding_rule() {
        // dt = 0.0025 puts every odd tick exactly on a half-millisecond.
        let timestep = Timestep::new(0.0025).expect("valid");
        let mut clock = Clock::new(timestep);
        // 0.0 -> 0, 2.5 -> 2 (even), 5.0 -> 5, 7.5 -> 8 (even), 10.0 -> 10
        let expected = [0i64, 2, 5, 8, 10];
        for want in expected {
            assert_eq!(clock.seed_time_key(), want, "at tick {}", clock.tick());
            clock.advance();
        }
    }

    #[test]
    fn half_to_even_differs_from_rusts_round_where_it_matters() {
        // Guards the reason the helper exists: if someone replaces it with
        // f64::round, these are the cases that change.
        assert_eq!(round_half_to_even(2.5), 2.0);
        assert_eq!(round_half_to_even(3.5), 4.0);
        assert_eq!(round_half_to_even(-2.5), -2.0);
        assert_eq!(round_half_to_even(-3.5), -4.0);
        assert_eq!(2.5f64.round(), 3.0, "f64::round is half-away-from-zero");

        // Everything that is not a tie must still round normally.
        assert_eq!(round_half_to_even(2.4), 2.0);
        assert_eq!(round_half_to_even(2.6), 3.0);
        assert_eq!(round_half_to_even(-2.6), -3.0);
        assert_eq!(round_half_to_even(0.0), 0.0);
    }

    #[test]
    fn reset_returns_to_tick_zero_and_keeps_the_timestep() {
        let timestep = Timestep::new(0.25).expect("valid");
        let mut clock = Clock::new(timestep);
        clock.advance();
        clock.advance();
        assert_eq!(clock.tick(), 2);
        clock.reset();
        assert_eq!(clock.tick(), 0);
        assert_eq!(clock.timestamp_s(), 0.0);
        assert_eq!(clock.timestep(), timestep);
    }
}
