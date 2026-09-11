//! The observation-generation rejection cascade.
//!
//! P4.5 asks for "the rejection cascade" and for an `ObservationBatch` that
//! "records attempted count, accepted observations, generation rejections and
//! counts by node/target/reason", keeping generation rejection distinct from
//! fusion rejection.
//!
//! Split the same way P4.2a and P4.5a were: the cascade is a pure decision over
//! already-computed geometry and visibility. Range, FOV angle, elevation, the
//! line-of-sight result and the combined detection probability arrive as
//! inputs; this module only decides. That keeps it checkable value-for-value
//! without dragging the whole world crate in, and it puts the two random draws
//! somewhere they can be seen and tested.
//!
//! # Gate order is part of the contract
//!
//! The reference orders the gates: range, FOV, **dropout**, elevation,
//! line-of-sight, detection. Dropout is a *random draw* sitting at position
//! three, ahead of two deterministic gates that cost nothing. That reads like
//! an accident, and reordering it would be an easy "cleanup" — but a pair that
//! would fail elevation still consumes its dropout draw today, so moving the
//! gate changes which pairs survive.
//!
//! `sim.py` notes that deferring RNG construction past the draw-free rejects is
//! "draw-order safe". That holds because each pair gets a stream keyed on
//! `(node, target, tick)` — but only when a seed is supplied. Without one the
//! reference falls back to a single shared generator, and then the number of
//! pairs reaching the dropout gate determines every later pair's draws.
//! `SimulationConfig.seed` is `int` with default 7 and never `None`, so the
//! production path is always keyed; the shared-stream path is reachable only
//! from direct callers. Latent, not live — and pinned below so a port cannot
//! quietly make it live.

use std::collections::BTreeMap;

/// Why a candidate observation was rejected during generation.
///
/// Generation rejection is deliberately distinct from fusion rejection, per
/// P4.5.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RejectReason {
    OutOfRange,
    OutsideFov,
    Dropout,
    LowElevation,
    TerrainOcclusion,
    BuildingOcclusion,
    WallOcclusion,
    VegetationOcclusion,
    OutOfCoverage,
}

impl RejectReason {
    /// The wire string the reference counts under.
    ///
    /// Note there is no `ObjectOcclusion` variant. Python defines
    /// `REJECT_OBJECT_OCCLUSION = REJECT_BUILDING_OCCLUSION` — ten constants,
    /// nine distinct strings — so an unrecognised blocker is counted as a
    /// building occlusion. Giving the fallback its own variant here would be
    /// the more natural Rust, and would silently change every rejection count
    /// that includes an unknown blocker. [`Blocker::Other`] maps here instead.
    pub fn as_str(self) -> &'static str {
        match self {
            RejectReason::OutOfRange => "out_of_range",
            RejectReason::OutsideFov => "outside_fov",
            RejectReason::Dropout => "dropout",
            RejectReason::LowElevation => "low_elevation",
            RejectReason::TerrainOcclusion => "terrain_occlusion",
            RejectReason::BuildingOcclusion => "building_occlusion",
            RejectReason::WallOcclusion => "wall_occlusion",
            RejectReason::VegetationOcclusion => "vegetation_occlusion",
            RejectReason::OutOfCoverage => "out_of_coverage",
        }
    }
}

/// What blocked the line of sight, as the visibility query reports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Blocker {
    Terrain,
    Building,
    Wall,
    Vegetation,
    OutOfCoverage,
    /// Anything the reference's dispatch does not name, which it folds onto
    /// `building_occlusion`. Kept as its own input variant so callers can say
    /// "unknown" honestly, while the count stays where the reference puts it.
    Other,
}

impl Blocker {
    fn reason(self) -> RejectReason {
        match self {
            Blocker::Terrain => RejectReason::TerrainOcclusion,
            Blocker::Building | Blocker::Other => RejectReason::BuildingOcclusion,
            Blocker::Wall => RejectReason::WallOcclusion,
            Blocker::Vegetation => RejectReason::VegetationOcclusion,
            Blocker::OutOfCoverage => RejectReason::OutOfCoverage,
        }
    }
}

/// Which loss dominates when probabilistic detection rejects a pair.
///
/// The reference reports a transmittance-dominated rejection as a vegetation
/// occlusion and everything else as a dropout, so the same gate produces two
/// different reasons.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DominantLoss {
    Transmittance,
    Other,
}

/// The line-of-sight outcome for one pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Visibility {
    Visible,
    Blocked(Blocker),
}

/// Everything the cascade needs about one (node, target) pair.
#[derive(Debug, Clone, Copy)]
pub struct PairInput {
    pub range_m: f64,
    pub max_range_m: f64,
    /// `None` for an omnidirectional sensor, or a directional one whose look
    /// direction could not be resolved -- the reference skips the FOV gate in
    /// both cases rather than treating it as a rejection.
    pub fov: Option<FovInput>,
    pub dropout_probability: f64,
    pub elevation_deg: f64,
    pub required_elevation_deg: f64,
    pub visibility: Visibility,
    pub combined_pd: f64,
    pub dominant_loss: DominantLoss,
}

#[derive(Debug, Clone, Copy)]
pub struct FovInput {
    pub angle_deg: f64,
    pub half_angle_deg: f64,
}

/// The outcome for one pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    /// Accepted, carrying how far toward the FOV edge the target sat. Zero for
    /// an omnidirectional sensor. The reference uses it to widen noise and
    /// shrink confidence near the edge.
    Accepted {
        fov_fraction: f64,
    },
    Rejected(RejectReason),
}

/// The two random draws the cascade makes, in the order it makes them.
///
/// A trait rather than a generator so the draw points are explicit: a test can
/// feed exact values and pin which gate consumed which draw, which is what
/// makes the ordering testable at all.
pub trait Draws {
    /// Uniform on [0, 1), matching `numpy.random.Generator.random`.
    fn next_uniform(&mut self) -> f64;
}

impl<F: FnMut() -> f64> Draws for F {
    fn next_uniform(&mut self) -> f64 {
        self()
    }
}

/// Run the cascade for one pair.
///
/// Gates are applied in the reference's order. The dropout draw is taken at
/// gate three even when a later deterministic gate would have rejected the
/// pair anyway -- see the module docs for why that is load-bearing.
pub fn evaluate_pair(input: &PairInput, draws: &mut impl Draws) -> Outcome {
    // 1. Range. The lower bound catches a target coincident with the node,
    //    where the line of sight has no direction to speak of.
    if input.range_m < 1e-6 || input.range_m > input.max_range_m {
        return Outcome::Rejected(RejectReason::OutOfRange);
    }

    // 2. Field of view, for directional sensors only.
    let mut fov_fraction = 0.0;
    if let Some(fov) = input.fov {
        if fov.angle_deg > fov.half_angle_deg {
            return Outcome::Rejected(RejectReason::OutsideFov);
        }
        // The reference floors the divisor at 1 degree, so a very narrow
        // sensor cannot produce a fraction far above one.
        fov_fraction = fov.angle_deg / fov.half_angle_deg.max(1.0);
    }

    // 3. Dropout -- the first draw, and strictly before the two gates below.
    if draws.next_uniform() < input.dropout_probability {
        return Outcome::Rejected(RejectReason::Dropout);
    }

    // 4. Elevation.
    if input.elevation_deg < input.required_elevation_deg {
        return Outcome::Rejected(RejectReason::LowElevation);
    }

    // 5. Line of sight.
    if let Visibility::Blocked(blocker) = input.visibility {
        return Outcome::Rejected(blocker.reason());
    }

    // 6. Probabilistic detection -- the second draw. Strictly greater-than, so
    //    a probability of exactly 1.0 always accepts and 0.0 always rejects.
    if draws.next_uniform() > input.combined_pd {
        return Outcome::Rejected(match input.dominant_loss {
            DominantLoss::Transmittance => RejectReason::VegetationOcclusion,
            DominantLoss::Other => RejectReason::Dropout,
        });
    }

    Outcome::Accepted { fov_fraction }
}

/// Generation-side tallies for one frame.
///
/// P4.5 requires attempted count, accepted observations, and counts by node,
/// target and reason. `BTreeMap` so iteration order is deterministic, which the
/// reference gets from sorting its counters on the way out.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct GenerationTally {
    pub attempted: u64,
    pub accepted: u64,
    pub rejection_counts: BTreeMap<&'static str, u64>,
    pub accepted_by_target: BTreeMap<String, u64>,
    pub rejected_by_target: BTreeMap<String, u64>,
    pub accepted_by_node_target: BTreeMap<(String, String), u64>,
}

impl GenerationTally {
    pub fn record(&mut self, node_id: &str, target_id: &str, outcome: Outcome) {
        self.attempted += 1;
        match outcome {
            Outcome::Accepted { .. } => {
                self.accepted += 1;
                *self
                    .accepted_by_target
                    .entry(target_id.to_string())
                    .or_default() += 1;
                *self
                    .accepted_by_node_target
                    .entry((node_id.to_string(), target_id.to_string()))
                    .or_default() += 1;
            }
            Outcome::Rejected(reason) => {
                *self.rejection_counts.entry(reason.as_str()).or_default() += 1;
                *self
                    .rejected_by_target
                    .entry(target_id.to_string())
                    .or_default() += 1;
            }
        }
    }

    /// Total rejections. Equals `attempted - accepted`; asserted in tests so a
    /// tally that loses an outcome cannot pass silently.
    pub fn rejected(&self) -> u64 {
        self.rejection_counts.values().sum()
    }
}
