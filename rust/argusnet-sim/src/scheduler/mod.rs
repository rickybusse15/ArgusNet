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
//! Nothing here is wired into [`crate::simulation::Simulation`] yet; the
//! legacy eleven-stage tick (`legacy_stages_v1`) is unchanged. The serial
//! executor and the serial-equivalence gate are the next B2 slices.
//!
//! [`RUST_RUNTIME_ARCHITECTURE.md`]: ../../../../docs/RUST_RUNTIME_ARCHITECTURE.md

mod queue;
mod time;

pub use queue::{EventKey, EventQueue, Phase, Scheduled, SourceId};
pub use time::{Schedule, SimTime, TimeError, NANOS_PER_SECOND};
