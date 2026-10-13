//! `scheduled_v1`'s foundations (P4.10a, B2): integer simulated time,
//! schedules whose deadlines never drift, and a totally ordered event queue.
//!
//! [`RUST_RUNTIME_ARCHITECTURE.md`] §3 is the contract:
//!
//! - internal time is checked integer nanoseconds ([`SimTime`]); floating
//!   seconds are derived interchange values only;
//! - a periodic schedule is an integer period or a rational frequency, and its
//!   `n`th deadline is computed from the epoch and `n` with wide checked
//!   arithmetic and one rounding rule -- ceiling to the next nanosecond --
//!   never by adding a rounded period ([`Schedule::nth_deadline`]);
//! - a rate whose distinct deadlines cannot be represented is rejected when the
//!   schedule is made, not when it misbehaves;
//! - equal-time work is ordered by phase, then stable source identity, then
//!   sequence ([`EventKey`]), and an event created for a phase already closed
//!   at its timestamp is deferred to the next eligible boundary, with its
//!   effective time recorded ([`EventQueue::push`]).
//!
//! [`LegacyLaneExecutor`] (B2.2) drives the legacy eleven-stage tick from this
//! queue as one compound lane (SCHEDULER_LEGACY_MAPPING.md); `legacy_stages_v1`
//! itself is unchanged. [`PhasedLaneExecutor`] (B2.3) takes fusion, mapping,
//! mission and publication out of that lane as their own phased tasks, in
//! their legacy order.
//!
//! [`RUST_RUNTIME_ARCHITECTURE.md`]: ../../../../docs/RUST_RUNTIME_ARCHITECTURE.md

mod executor;
mod lanes;
mod queue;
mod time;

pub use executor::{ExecutorError, LaneTask, LegacyLaneExecutor, LEGACY_LANE};
pub use lanes::{
    describe, parse_rate, rate_text, Lane, LanePart, LaneRates, PhasedLaneExecutor, PhasedTick,
    LANES_V1, RATED_LANES,
};
pub use queue::{EventKey, EventQueue, Phase, Scheduled, SourceId};
pub use time::{Schedule, SimTime, TimeError, NANOS_PER_SECOND};
