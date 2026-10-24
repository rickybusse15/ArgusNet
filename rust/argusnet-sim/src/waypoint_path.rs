//! `looped_waypoint_path`: fly a closed polyline at constant speed, forever.
//!
//! The search half of every search-then-follow drone. `search_path` builds a
//! lawnmower pattern over the drone's sector, optionally re-routes it around
//! obstacles, and hands the resulting waypoints to `looped_waypoint_path`,
//! which is the part that runs every tick. The waypoint construction is
//! scenario construction and runs once; it belongs with the scenario builder
//! (P4.9), so this port takes the finished waypoints as input.
//!
//! # A trajectory, not a controller
//!
//! No state survives a sample: the position is a pure function of the
//! timestamp. That is why it has no checkpoint state, and why the stage can
//! resume it from nothing. It needs terrain and obstacles *per call* rather
//! than owning them, because the stage owns the world and lends it out.
//!
//! # Python's `%`, not Rust's
//!
//! Distance along the loop is `(speed * t + phase_offset) % length`, and
//! Python's float `%` takes the sign of the divisor where Rust's takes the
//! dividend's. For the non-negative operands a run produces they agree; the
//! port uses [`py_mod`] anyway so a negative timestamp (a seek before launch,
//! a test) lands where the reference would put it.

use serde::{Deserialize, Serialize};

use crate::motion::{
    collision_aware_position, compose_terrain_state, ObstacleSampler, TerrainSampler,
};

/// Python's float modulo: `fmod`, then shifted to take the divisor's sign.
pub fn py_mod(dividend: f64, divisor: f64) -> f64 {
    let remainder = dividend % divisor;
    if remainder != 0.0 {
        if (divisor < 0.0) != (remainder < 0.0) {
            remainder + divisor
        } else {
            remainder
        }
    } else {
        0.0_f64.copysign(divisor)
    }
}

/// Why a waypoint list cannot be flown. Both are the reference's `ValueError`s.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WaypointPathError {
    TooFewWaypoints { found: usize },
    ZeroLength,
}

impl std::fmt::Display for WaypointPathError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            WaypointPathError::TooFewWaypoints { found } => {
                write!(
                    formatter,
                    "a looped path needs at least 2 waypoints, got {found}"
                )
            }
            WaypointPathError::ZeroLength => {
                write!(
                    formatter,
                    "waypoints must contain at least one non-zero segment"
                )
            }
        }
    }
}

impl std::error::Error for WaypointPathError {}

/// The parameters `looped_waypoint_path` closes over.
///
/// Serializable because a controller built from it is part of a scenario
/// description, not because it is checkpoint state -- it has none.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaypointPathConfig {
    pub waypoints_xy_m: Vec<[f64; 2]>,
    pub speed_mps: f64,
    pub base_agl_m: f64,
    pub vertical_amplitude_m: f64,
    pub vertical_frequency_rad_s: f64,
    pub phase: f64,
    /// The reference's default is 3.0; `search_path` passes its own.
    pub min_agl_m: f64,
    /// `search_path` passes `environment.obstacles` whenever it has an
    /// environment. False means no collision probe at all -- which is a
    /// different read pattern, not merely a different answer.
    pub avoid_obstacles: bool,
}

/// A closed polyline flown at constant speed.
#[derive(Debug, Clone, PartialEq)]
pub struct LoopedWaypointPath {
    config: WaypointPathConfig,
    /// The waypoints with the first repeated at the end.
    wrapped: Vec<[f64; 2]>,
    segment_vectors: Vec<[f64; 2]>,
    segment_lengths: Vec<f64>,
    cumulative_lengths: Vec<f64>,
    path_length_m: f64,
    phase_offset_m: f64,
    /// `max(speed, 0.5)`: a lane flown slower than half a metre per second
    /// is flown at half a metre per second.
    speed_mps: f64,
}

impl LoopedWaypointPath {
    pub const MIN_SPEED_MPS: f64 = 0.5;

    pub fn new(config: WaypointPathConfig) -> Result<Self, WaypointPathError> {
        let points = &config.waypoints_xy_m;
        if points.len() < 2 {
            return Err(WaypointPathError::TooFewWaypoints {
                found: points.len(),
            });
        }
        let mut wrapped = points.clone();
        wrapped.push(points[0]);
        let segment_vectors: Vec<[f64; 2]> = wrapped
            .windows(2)
            .map(|pair| [pair[1][0] - pair[0][0], pair[1][1] - pair[0][1]])
            .collect();
        // `np.linalg.norm(axis=1)`: plain sqrt of the sum of squares, not
        // `hypot`. See `follow_path::norm` for why the difference matters.
        let segment_lengths: Vec<f64> = segment_vectors
            .iter()
            .map(|v| (v[0] * v[0] + v[1] * v[1]).sqrt())
            .collect();
        // `segment_lengths.sum()` is numpy's pairwise sum and `cumsum` is
        // sequential; for the zero-length test only the sign matters, and
        // every term is non-negative, so the two cannot disagree about it.
        if segment_lengths.iter().sum::<f64>() <= 1.0e-6 {
            return Err(WaypointPathError::ZeroLength);
        }
        let mut cumulative_lengths = Vec::with_capacity(segment_lengths.len());
        let mut running = 0.0;
        for length in &segment_lengths {
            running += length;
            cumulative_lengths.push(running);
        }
        let path_length_m = *cumulative_lengths.last().expect("at least one segment");
        let phase_offset_m =
            py_mod(config.phase, std::f64::consts::TAU) / std::f64::consts::TAU * path_length_m;
        let speed_mps = config.speed_mps.max(Self::MIN_SPEED_MPS);
        Ok(Self {
            config,
            wrapped,
            segment_vectors,
            segment_lengths,
            cumulative_lengths,
            path_length_m,
            phase_offset_m,
            speed_mps,
        })
    }

    pub fn config(&self) -> &WaypointPathConfig {
        &self.config
    }

    /// This lane flown no faster than `speed_mps`: the same lane, rebuilt
    /// at the lower speed, or itself unchanged when it is already slower. An
    /// explicit cap is a safety bound, so it overrides the half-metre floor
    /// an uncapped lane is flown at.
    pub fn with_speed_cap(self, speed_mps: f64) -> Self {
        if self.speed_mps <= speed_mps {
            return self;
        }
        let config = WaypointPathConfig {
            speed_mps,
            ..self.config.clone()
        };
        let mut lane = Self::new(config).expect("a valid lane stays valid at another speed");
        lane.speed_mps = speed_mps.max(0.0);
        lane
    }

    /// The speed the lane is flown at: the configured speed, at least
    /// [`Self::MIN_SPEED_MPS`] unless a cap holds it lower.
    pub fn speed_mps(&self) -> f64 {
        self.speed_mps
    }

    pub fn path_length_m(&self) -> f64 {
        self.path_length_m
    }

    /// Position and velocity at `timestamp_s`.
    ///
    /// `obstacles` is consulted only when the config asks for avoidance, so a
    /// caller can always pass the world's layer.
    pub fn sample(
        &self,
        timestamp_s: f64,
        terrain: &dyn TerrainSampler,
        obstacles: Option<&dyn ObstacleSampler>,
    ) -> ([f64; 3], [f64; 3]) {
        let distance_m = py_mod(
            self.speed_mps * timestamp_s + self.phase_offset_m,
            self.path_length_m,
        );
        // `np.searchsorted(cumulative, distance, side="right")`: the count of
        // entries <= distance. So a distance exactly on a vertex resolves to
        // the segment that *starts* there, and a zero-length segment is
        // stepped over.
        let index = self
            .cumulative_lengths
            .partition_point(|cumulative| *cumulative <= distance_m)
            .min(self.segment_vectors.len() - 1);
        let segment_start_m = if index == 0 {
            0.0
        } else {
            self.cumulative_lengths[index - 1]
        };
        let segment_distance_m = distance_m - segment_start_m;
        let segment_length_m = self.segment_lengths[index].max(1.0e-6);
        let vector = self.segment_vectors[index];
        let direction = [vector[0] / segment_length_m, vector[1] / segment_length_m];
        let start = self.wrapped[index];
        let xy_m = [
            start[0] + direction[0] * segment_distance_m,
            start[1] + direction[1] * segment_distance_m,
        ];
        let velocity_xy = [direction[0] * self.speed_mps, direction[1] * self.speed_mps];

        let vertical_phase =
            self.config.vertical_frequency_rad_s * timestamp_s + self.config.phase * 0.5;
        let agl_m =
            self.config.base_agl_m + self.config.vertical_amplitude_m * vertical_phase.sin();
        // Left to right, as numpy evaluates it: (amplitude * frequency) * cos.
        let agl_rate_mps = self.config.vertical_amplitude_m
            * self.config.vertical_frequency_rad_s
            * vertical_phase.cos();

        let (position, velocity) = compose_terrain_state(
            terrain,
            xy_m,
            velocity_xy,
            agl_m,
            agl_rate_mps,
            self.config.min_agl_m,
            false,
        );
        let obstacles = if self.config.avoid_obstacles {
            obstacles
        } else {
            None
        };
        let resolved =
            collision_aware_position(terrain, obstacles, position, self.config.min_agl_m);
        (resolved.position_m, velocity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn python_mod_takes_the_divisor_sign() {
        assert_eq!(py_mod(7.0, 3.0), 1.0);
        assert_eq!(py_mod(-1.0, 3.0), 2.0);
        assert_eq!(py_mod(1.0, -3.0), -2.0);
        assert!(py_mod(3.0, 3.0) == 0.0 && py_mod(3.0, 3.0).is_sign_positive());
        assert!(py_mod(3.0, -3.0).is_sign_negative());
    }

    /// A safety cap holds a lane below the half-metre floor an uncapped
    /// lane keeps; a cap above the lane's speed changes nothing.
    #[test]
    fn a_speed_cap_overrides_the_minimum_speed() {
        let lane = LoopedWaypointPath::new(config(vec![[0.0, 0.0], [100.0, 0.0]])).unwrap();
        assert_eq!(lane.clone().with_speed_cap(20.0), lane);
        let capped = lane.clone().with_speed_cap(0.062);
        assert_eq!(capped.speed_mps(), 0.062);
        let slow = LoopedWaypointPath::new(WaypointPathConfig {
            speed_mps: 0.1,
            ..config(vec![[0.0, 0.0], [100.0, 0.0]])
        })
        .unwrap();
        assert_eq!(slow.speed_mps(), LoopedWaypointPath::MIN_SPEED_MPS);
        assert_eq!(slow.with_speed_cap(0.2).speed_mps(), 0.2);
    }

    fn config(points: Vec<[f64; 2]>) -> WaypointPathConfig {
        WaypointPathConfig {
            waypoints_xy_m: points,
            speed_mps: 10.0,
            base_agl_m: 50.0,
            vertical_amplitude_m: 0.0,
            vertical_frequency_rad_s: 0.0,
            phase: 0.0,
            min_agl_m: 3.0,
            avoid_obstacles: false,
        }
    }

    #[test]
    fn the_reference_value_errors_are_refusals() {
        assert_eq!(
            LoopedWaypointPath::new(config(vec![[0.0, 0.0]])),
            Err(WaypointPathError::TooFewWaypoints { found: 1 })
        );
        assert_eq!(
            LoopedWaypointPath::new(config(vec![[1.0, 1.0], [1.0, 1.0]])),
            Err(WaypointPathError::ZeroLength)
        );
    }

    #[test]
    fn the_loop_closes_back_to_the_first_waypoint() {
        let path = LoopedWaypointPath::new(config(vec![[0.0, 0.0], [100.0, 0.0], [100.0, 100.0]]))
            .expect("valid");
        let closing = (100.0_f64 * 100.0 + 100.0 * 100.0).sqrt();
        assert_eq!(path.path_length_m(), 200.0 + closing);
    }
}
