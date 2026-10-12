//! Integer simulated time and drift-free schedules.

use std::fmt;
use std::num::NonZeroU64;

pub const NANOS_PER_SECOND: u64 = 1_000_000_000;

/// Simulated time: integer nanoseconds since the run's epoch.
///
/// Every operation is checked. Seconds are derived, for interchange only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SimTime(i64);

impl SimTime {
    pub const EPOCH: SimTime = SimTime(0);

    pub const fn from_nanos(nanos: i64) -> Self {
        Self(nanos)
    }

    pub const fn as_nanos(self) -> i64 {
        self.0
    }

    /// `self + nanos`, or [`TimeError::Overflow`].
    pub fn checked_add_nanos(self, nanos: u64) -> Result<Self, TimeError> {
        i64::try_from(nanos)
            .ok()
            .and_then(|nanos| self.0.checked_add(nanos))
            .map(Self)
            .ok_or(TimeError::Overflow)
    }

    /// Seconds since the epoch, for interchange. Never fed back into time.
    pub fn as_seconds_f64(self) -> f64 {
        self.0 as f64 / NANOS_PER_SECOND as f64
    }
}

impl fmt::Display for SimTime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} ns", self.0)
    }
}

/// Why a schedule or a deadline was refused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeError {
    /// The deadline is past the end of representable time.
    Overflow,
    /// A rational rate so high that consecutive deadlines, rounded up to the
    /// nanosecond, would coincide: `hz_num / hz_den` above 1 GHz.
    IndistinctDeadlines { hz_num: u64, hz_den: u64 },
    /// A one-shot schedule asked for an occurrence after its only one.
    NoSuchOccurrence { n: u64 },
    /// A lane that buffers the boundaries it sits out (mapping, B2.5) would
    /// sit out more than `limit` boundaries in a row at this rate: its
    /// backlog, kept in memory and in every checkpoint, would be unbounded in
    /// practice. `boundaries` is how many (`u64::MAX` for a one-shot rate,
    /// which never runs again).
    BacklogTooLong { boundaries: u64, limit: u64 },
}

impl fmt::Display for TimeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TimeError::Overflow => write!(formatter, "deadline overflows simulated time"),
            TimeError::IndistinctDeadlines { hz_num, hz_den } => write!(
                formatter,
                "rate {hz_num}/{hz_den} Hz has deadlines closer than 1 ns, which cannot be told apart"
            ),
            TimeError::NoSuchOccurrence { n } => {
                write!(formatter, "a one-shot schedule has no occurrence {n}")
            }
            TimeError::BacklogTooLong { boundaries, limit } => write!(
                formatter,
                "at this rate the lane would sit out up to {boundaries} ticks in a row and \
                 buffer every one; at most {limit} are allowed"
            ),
        }
    }
}

impl std::error::Error for TimeError {}

/// When something recurs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Schedule {
    /// Every `period` nanoseconds.
    Period(NonZeroU64),
    /// `hz_num / hz_den` times a second: a camera at 30 Hz is `30/1`, and
    /// need not divide a 50 Hz physics step.
    Rational {
        hz_num: NonZeroU64,
        hz_den: NonZeroU64,
    },
    /// Once, at a fixed time.
    Once(SimTime),
}

impl Schedule {
    /// A rational schedule, refused if its deadlines cannot be told apart.
    pub fn rational(hz_num: NonZeroU64, hz_den: NonZeroU64) -> Result<Self, TimeError> {
        let schedule = Schedule::Rational { hz_num, hz_den };
        schedule.validate()?;
        Ok(schedule)
    }

    /// Refuse a rate whose distinct deadlines are not representable. Done at
    /// configuration time, so a run never discovers it mid-flight.
    pub fn validate(&self) -> Result<(), TimeError> {
        match *self {
            Schedule::Rational { hz_num, hz_den } => {
                // Period = hz_den / hz_num seconds; at least 1 ns, else two
                // consecutive ceilings can land on the same nanosecond.
                if u128::from(hz_num.get())
                    > u128::from(hz_den.get()) * u128::from(NANOS_PER_SECOND)
                {
                    return Err(TimeError::IndistinctDeadlines {
                        hz_num: hz_num.get(),
                        hz_den: hz_den.get(),
                    });
                }
                Ok(())
            }
            Schedule::Period(_) | Schedule::Once(_) => Ok(()),
        }
    }

    /// The `n`th deadline (`n = 0` is the first), from `epoch`.
    ///
    /// Computed from `n` directly, in `u128`, never by accumulating a rounded
    /// period, so deadline `n` is the same however many came before it. A
    /// rational deadline is the exact time rounded **up** to the nanosecond:
    /// `epoch + ceil(n * hz_den * 1e9 / hz_num)`.
    pub fn nth_deadline(&self, epoch: SimTime, n: u64) -> Result<SimTime, TimeError> {
        self.validate()?;
        let offset: u128 = match *self {
            Schedule::Period(period) => u128::from(n) * u128::from(period.get()),
            Schedule::Rational { hz_num, hz_den } => {
                let numerator =
                    u128::from(n) * u128::from(hz_den.get()) * u128::from(NANOS_PER_SECOND);
                numerator.div_ceil(u128::from(hz_num.get()))
            }
            Schedule::Once(at) => {
                return if n == 0 {
                    Ok(at)
                } else {
                    Err(TimeError::NoSuchOccurrence { n })
                };
            }
        };
        let offset = u64::try_from(offset).map_err(|_| TimeError::Overflow)?;
        epoch.checked_add_nanos(offset)
    }

    /// The first occurrence index whose deadline is at or after `time`.
    pub fn first_at_or_after(&self, epoch: SimTime, time: SimTime) -> Result<u64, TimeError> {
        self.validate()?;
        let since = i128::from(time.as_nanos()) - i128::from(epoch.as_nanos());
        if since <= 0 {
            return Ok(0);
        }
        let since = since as u128;
        let n = match *self {
            Schedule::Period(period) => since.div_ceil(u128::from(period.get())),
            Schedule::Rational { hz_num, hz_den } => {
                // Smallest n with ceil(n * den * 1e9 / num) >= since, i.e.
                // n * den * 1e9 > (since - 1) * num.
                let per = u128::from(hz_den.get()) * u128::from(NANOS_PER_SECOND);
                ((since - 1) * u128::from(hz_num.get())) / per + 1
            }
            Schedule::Once(at) => return Ok(if at >= time { 0 } else { 1 }),
        };
        u64::try_from(n).map_err(|_| TimeError::Overflow)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nz(value: u64) -> NonZeroU64 {
        NonZeroU64::new(value).expect("non-zero")
    }

    /// 30 Hz never drifts: every deadline is the exact rational time rounded
    /// up, however far out, and every third one is exactly on a 10 Hz grid.
    #[test]
    fn a_rational_rate_never_drifts() {
        let camera = Schedule::rational(nz(30), nz(1)).unwrap();
        for n in [0u64, 1, 2, 3, 29, 30, 1_000_003, 3_000_000_000] {
            let deadline = camera.nth_deadline(SimTime::EPOCH, n).unwrap().as_nanos() as u128;
            let exact_times_30 = u128::from(n) * 1_000_000_000;
            assert!(deadline * 30 >= exact_times_30, "deadline {n} early");
            assert!((deadline - u128::from(deadline > 0)) * 30 < exact_times_30 || n == 0);
            if n % 3 == 0 {
                assert_eq!(
                    deadline * 30,
                    exact_times_30,
                    "deadline {n} off the 10 Hz grid"
                );
            }
        }
        // Accumulating a rounded period (33_333_334 ns) drifts by 2/3 ns a
        // frame: after 3e9 frames, 2 s. The direct computation does not.
        let n = 3_000_000_000u64;
        let direct = camera.nth_deadline(SimTime::EPOCH, n).unwrap().as_nanos();
        assert_eq!(direct, 100_000_000_000_000_000);
        assert_ne!(direct, 33_333_334 * n as i64);
    }

    /// A 30 Hz camera against a 50 Hz physics step: they coincide exactly every
    /// tenth of a second, and nowhere else.
    #[test]
    fn a_camera_need_not_divide_the_physics_rate() {
        let camera = Schedule::rational(nz(30), nz(1)).unwrap();
        let physics = Schedule::Period(nz(20_000_000));
        let mut shared = Vec::new();
        for n in 0..300 {
            let t = camera.nth_deadline(SimTime::EPOCH, n).unwrap();
            let k = physics.first_at_or_after(SimTime::EPOCH, t).unwrap();
            if physics.nth_deadline(SimTime::EPOCH, k).unwrap() == t {
                shared.push(t.as_nanos());
            }
        }
        let expected: Vec<i64> = (0..100).map(|k| k * 100_000_000).collect();
        assert_eq!(shared, expected);
    }

    #[test]
    fn first_at_or_after_inverts_nth_deadline() {
        let schedules = [
            Schedule::Period(nz(7)),
            Schedule::rational(nz(30), nz(1)).unwrap(),
            Schedule::rational(nz(200), nz(3)).unwrap(),
        ];
        for schedule in schedules {
            for n in 0..500 {
                let deadline = schedule.nth_deadline(SimTime::EPOCH, n).unwrap();
                assert_eq!(
                    schedule
                        .first_at_or_after(SimTime::EPOCH, deadline)
                        .unwrap(),
                    n
                );
                let after = SimTime::from_nanos(deadline.as_nanos() + 1);
                assert_eq!(
                    schedule.first_at_or_after(SimTime::EPOCH, after).unwrap(),
                    n + 1,
                    "{schedule:?} n={n}"
                );
            }
        }
    }

    #[test]
    fn deadlines_past_representable_time_are_refused() {
        let period = Schedule::Period(nz(u64::MAX));
        assert_eq!(
            period.nth_deadline(SimTime::EPOCH, 2),
            Err(TimeError::Overflow)
        );
        let slow = Schedule::rational(nz(1), nz(u64::MAX)).unwrap();
        assert_eq!(
            slow.nth_deadline(SimTime::EPOCH, 1),
            Err(TimeError::Overflow)
        );
        let late = SimTime::from_nanos(i64::MAX - 5);
        assert_eq!(
            Schedule::Period(nz(10)).nth_deadline(late, 1),
            Err(TimeError::Overflow)
        );
    }

    #[test]
    fn a_rate_above_a_gigahertz_is_refused_at_configuration() {
        assert!(Schedule::rational(nz(1_000_000_000), nz(1)).is_ok());
        assert_eq!(
            Schedule::rational(nz(1_000_000_001), nz(1)),
            Err(TimeError::IndistinctDeadlines {
                hz_num: 1_000_000_001,
                hz_den: 1
            })
        );
        // The bound is on the rate, not the numerator: just under 1 GHz with a
        // denominator is fine, just over is not.
        assert!(Schedule::rational(nz(2_999_999_999), nz(3)).is_ok());
        assert!(Schedule::rational(nz(3_000_000_001), nz(3)).is_err());
    }

    #[test]
    fn a_one_shot_has_one_occurrence() {
        let once = Schedule::Once(SimTime::from_nanos(42));
        assert_eq!(
            once.nth_deadline(SimTime::EPOCH, 0),
            Ok(SimTime::from_nanos(42))
        );
        assert_eq!(
            once.nth_deadline(SimTime::EPOCH, 1),
            Err(TimeError::NoSuchOccurrence { n: 1 })
        );
    }
}
