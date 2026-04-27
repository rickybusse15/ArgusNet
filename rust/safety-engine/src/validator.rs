//! Constraint validation pipeline.
//!
//! Every candidate commanded state (position + velocity + acceleration) is
//! passed through [`validate_constraints`] before being applied.  The pipeline
//! is a sequence of independent checks; failure at any stage returns the
//! specific violated constraint via `Err(ConstraintViolation)`.
//!
//! Validation order (most-dangerous first, per SAFETY.md §2.1):
//! 1. Terrain clearance
//! 2. Altitude ceiling
//! 3. Speed limits
//! 4. Acceleration
//! 5. Climb / descent rate
//! 6. Turn radius
//! 7. Drone separation
//! 8. Energy reserve
//! 9. Communications
//! 10. Gimbal limits

use serde::{Deserialize, Serialize};
use terrain_engine::TerrainQuery;

use crate::limits::DronePhysicalLimits;

// ---------------------------------------------------------------------------
// Public data types
// ---------------------------------------------------------------------------

/// A constraint violation returned by [`validate_constraints`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ConstraintViolation {
    /// Commanded speed exceeds the platform maximum.
    SpeedAboveMaximum { commanded_mps: f64, limit_mps: f64 },

    /// Commanded speed is below the minimum controllable airspeed.
    SpeedBelowMinimum { commanded_mps: f64, limit_mps: f64 },

    /// Horizontal or vertical acceleration exceeds the platform limit.
    AccelerationExceeded {
        commanded_mps2: f64,
        limit_mps2: f64,
    },

    /// The commanded turn geometry would require a tighter radius than the
    /// platform can sustain at the current speed.
    TurnRadiusTooTight {
        commanded_radius_m: f64,
        min_radius_m: f64,
    },

    /// The commanded vertical velocity upward exceeds the climb rate limit.
    ClimbRateExceeded { commanded_mps: f64, limit_mps: f64 },

    /// The commanded vertical velocity downward exceeds the descent rate limit.
    DescentRateExceeded { commanded_mps: f64, limit_mps: f64 },

    /// The drone would be below the required terrain clearance floor.
    TerrainClearanceViolation { agl_m: f64, min_agl_m: f64 },

    /// The drone would exceed its operational altitude ceiling.
    AltitudeCeilingExceeded { agl_m: f64, max_agl_m: f64 },

    /// The battery reserve is below the mandatory return-to-home threshold.
    EnergyReserveLow {
        reserve_fraction: f64,
        min_fraction: f64,
    },

    /// The uplink RSSI is below the minimum commanded-flight threshold.
    CommsTooWeak { rssi_dbm: f64, threshold_dbm: f64 },

    /// The direct path to the GCS is obstructed by terrain (comms shadow).
    CommsShadowed,

    /// A peer drone is too close (horizontal or vertical separation violated).
    DroneSeparationViolation {
        drone_id: String,
        distance_m: f64,
        min_m: f64,
    },

    /// A gimbal axis angle is outside the platform's mechanical limits.
    GimbalOutOfRange {
        axis: String,
        angle_rad: f64,
        limit_rad: f64,
    },
}

/// Commanded state submitted to the validator.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DroneCommandedState {
    /// 3-D position in metres (x, y above reference; z = altitude above datum).
    pub position_m: [f64; 3],

    /// 3-D velocity in m/s.
    pub velocity_mps: [f64; 3],

    /// 3-D acceleration in m/s².
    pub acceleration_mps2: [f64; 3],

    /// Gimbal pitch angle (radians).  Positive = depressed below horizon.
    pub gimbal_pitch_rad: f64,

    /// Gimbal yaw offset from vehicle heading (radians).
    pub gimbal_yaw_offset_rad: f64,
}

/// A peer drone's position, used for separation checks.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DroneObservedState {
    pub drone_id: String,
    pub position_m: [f64; 3],
}

// ---------------------------------------------------------------------------
// Validation pipeline
// ---------------------------------------------------------------------------

/// Validate a candidate drone state against the physical limits.
///
/// Returns `Ok(())` if all constraints pass, or `Err(ConstraintViolation)`
/// for the first violation encountered.
///
/// # Validation order
/// Terrain clearance → altitude ceiling → speed → acceleration →
/// climb/descent → turn radius → separation → energy → comms → gimbal
pub fn validate_constraints(
    limits: &DronePhysicalLimits,
    candidate: &DroneCommandedState,
    terrain: &dyn TerrainQuery,
    peers: &[DroneObservedState],
    energy_fraction: f64,
    comms_rssi_dbm: f64,
) -> Result<(), ConstraintViolation> {
    let pos = candidate.position_m;
    let vel = candidate.velocity_mps;
    let acc = candidate.acceleration_mps2;

    // 1. Terrain clearance -------------------------------------------------
    let terrain_z = terrain.height_at(pos[0], pos[1]);
    let agl = pos[2] - terrain_z;
    if agl < limits.min_agl_m {
        return Err(ConstraintViolation::TerrainClearanceViolation {
            agl_m: agl,
            min_agl_m: limits.min_agl_m,
        });
    }

    // 2. Altitude ceiling --------------------------------------------------
    if agl > limits.max_agl_m {
        return Err(ConstraintViolation::AltitudeCeilingExceeded {
            agl_m: agl,
            max_agl_m: limits.max_agl_m,
        });
    }

    // 3. Speed limits ------------------------------------------------------
    let speed = (vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]).sqrt();
    if speed > limits.max_speed_mps {
        return Err(ConstraintViolation::SpeedAboveMaximum {
            commanded_mps: speed,
            limit_mps: limits.max_speed_mps,
        });
    }
    if speed < limits.min_speed_mps {
        return Err(ConstraintViolation::SpeedBelowMinimum {
            commanded_mps: speed,
            limit_mps: limits.min_speed_mps,
        });
    }

    // 4. Acceleration ------------------------------------------------------
    let h_accel = (acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
    if h_accel > limits.max_horizontal_accel_mps2 {
        return Err(ConstraintViolation::AccelerationExceeded {
            commanded_mps2: h_accel,
            limit_mps2: limits.max_horizontal_accel_mps2,
        });
    }
    let v_accel = acc[2].abs();
    if v_accel > limits.max_vertical_accel_mps2 {
        return Err(ConstraintViolation::AccelerationExceeded {
            commanded_mps2: v_accel,
            limit_mps2: limits.max_vertical_accel_mps2,
        });
    }

    // 5. Climb / descent rate ----------------------------------------------
    if vel[2] > limits.max_climb_rate_mps {
        return Err(ConstraintViolation::ClimbRateExceeded {
            commanded_mps: vel[2],
            limit_mps: limits.max_climb_rate_mps,
        });
    }
    if vel[2] < -limits.max_descent_rate_mps {
        return Err(ConstraintViolation::DescentRateExceeded {
            commanded_mps: vel[2].abs(),
            limit_mps: limits.max_descent_rate_mps,
        });
    }

    // 6. Turn radius -------------------------------------------------------
    // Estimate instantaneous turn radius from lateral acceleration.
    // lateral_accel = centripetal = v² / r  →  r = v² / lateral_accel
    let lateral_accel = (acc[0] * acc[0] + acc[1] * acc[1]).sqrt();
    if lateral_accel > 1e-6 {
        let h_speed = (vel[0] * vel[0] + vel[1] * vel[1]).sqrt();
        if h_speed > 1e-6 {
            let commanded_radius = h_speed * h_speed / lateral_accel;
            let min_radius = limits.min_turn_radius_at(h_speed);
            if commanded_radius < min_radius {
                return Err(ConstraintViolation::TurnRadiusTooTight {
                    commanded_radius_m: commanded_radius,
                    min_radius_m: min_radius,
                });
            }
        }
    }

    // 7. Drone separation --------------------------------------------------
    for peer in peers {
        let dp = [
            pos[0] - peer.position_m[0],
            pos[1] - peer.position_m[1],
            pos[2] - peer.position_m[2],
        ];
        let horiz_dist = (dp[0] * dp[0] + dp[1] * dp[1]).sqrt();
        let vert_dist = dp[2].abs();

        if horiz_dist < limits.min_drone_separation_m {
            return Err(ConstraintViolation::DroneSeparationViolation {
                drone_id: peer.drone_id.clone(),
                distance_m: horiz_dist,
                min_m: limits.min_drone_separation_m,
            });
        }
        if vert_dist < limits.min_drone_vertical_separation_m {
            return Err(ConstraintViolation::DroneSeparationViolation {
                drone_id: peer.drone_id.clone(),
                distance_m: vert_dist,
                min_m: limits.min_drone_vertical_separation_m,
            });
        }
    }

    // 8. Energy reserve ----------------------------------------------------
    if energy_fraction < limits.min_energy_reserve_fraction {
        return Err(ConstraintViolation::EnergyReserveLow {
            reserve_fraction: energy_fraction,
            min_fraction: limits.min_energy_reserve_fraction,
        });
    }

    // 9. Communications ----------------------------------------------------
    if comms_rssi_dbm < limits.min_comms_rssi_dbm {
        return Err(ConstraintViolation::CommsTooWeak {
            rssi_dbm: comms_rssi_dbm,
            threshold_dbm: limits.min_comms_rssi_dbm,
        });
    }

    // 10. Gimbal limits ----------------------------------------------------
    if candidate.gimbal_pitch_rad > limits.gimbal_max_depression_rad {
        return Err(ConstraintViolation::GimbalOutOfRange {
            axis: "pitch_down".to_string(),
            angle_rad: candidate.gimbal_pitch_rad,
            limit_rad: limits.gimbal_max_depression_rad,
        });
    }
    if candidate.gimbal_pitch_rad < -limits.gimbal_max_elevation_rad {
        return Err(ConstraintViolation::GimbalOutOfRange {
            axis: "pitch_up".to_string(),
            angle_rad: candidate.gimbal_pitch_rad,
            limit_rad: limits.gimbal_max_elevation_rad,
        });
    }
    if candidate.gimbal_yaw_offset_rad.abs() > limits.gimbal_max_yaw_offset_rad {
        return Err(ConstraintViolation::GimbalOutOfRange {
            axis: "yaw".to_string(),
            angle_rad: candidate.gimbal_yaw_offset_rad,
            limit_rad: limits.gimbal_max_yaw_offset_rad,
        });
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::DronePhysicalLimits;
    use terrain_engine::{FlatTerrain, TerrainBounds};

    fn make_flat() -> FlatTerrain {
        FlatTerrain::new(
            0.0,
            TerrainBounds::new(-10000.0, 10000.0, -10000.0, 10000.0, 0.0, 0.0),
        )
    }

    fn nominal_state() -> DroneCommandedState {
        DroneCommandedState {
            position_m: [0.0, 0.0, 100.0], // 100 m AGL on flat terrain
            velocity_mps: [10.0, 0.0, 0.0],
            acceleration_mps2: [0.0, 0.0, 0.0],
            gimbal_pitch_rad: 0.5, // within 80° limit
            gimbal_yaw_offset_rad: 0.0,
        }
    }

    #[test]
    fn valid_state_passes() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        let result = validate_constraints(&limits, &nominal_state(), &terrain, &[], 0.50, -60.0);
        assert!(result.is_ok(), "expected Ok but got {result:?}");
    }

    #[test]
    fn terrain_clearance_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // Place drone at 10 m AGL — below the 30 m minimum for interceptor.
        let mut state = nominal_state();
        state.position_m[2] = 10.0;
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::TerrainClearanceViolation { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn altitude_ceiling_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // Place drone at 600 m AGL — above the 500 m ceiling.
        let mut state = nominal_state();
        state.position_m[2] = 600.0;
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::AltitudeCeilingExceeded { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn speed_above_maximum_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        let mut state = nominal_state();
        // 50 m/s exceeds the 42 m/s limit.
        state.velocity_mps = [50.0, 0.0, 0.0];
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::SpeedAboveMaximum { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn energy_reserve_low_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // 10% reserve < 20% minimum.
        let err = validate_constraints(&limits, &nominal_state(), &terrain, &[], 0.10, -60.0)
            .unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::EnergyReserveLow { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn comms_too_weak_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // -95 dBm < -85 dBm threshold.
        let err = validate_constraints(&limits, &nominal_state(), &terrain, &[], 0.50, -95.0)
            .unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::CommsTooWeak { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn drone_separation_violation_horizontal() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // Place a peer 5 m away horizontally — below the 8 m minimum.
        let peers = vec![DroneObservedState {
            drone_id: "peer-1".to_string(),
            position_m: [5.0, 0.0, 100.0],
        }];
        let err = validate_constraints(&limits, &nominal_state(), &terrain, &peers, 0.50, -60.0)
            .unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::DroneSeparationViolation { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn gimbal_out_of_range_yaw() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // 2.0 rad yaw > 90° (1.571 rad) limit.
        let mut state = nominal_state();
        state.gimbal_yaw_offset_rad = 2.0;
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::GimbalOutOfRange { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn climb_rate_exceeded_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // 10 m/s climb > 8 m/s limit.
        let mut state = nominal_state();
        state.velocity_mps = [0.0, 0.0, 10.0];
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::ClimbRateExceeded { .. }),
            "wrong violation: {err:?}"
        );
    }

    #[test]
    fn descent_rate_exceeded_violation() {
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();
        // -7 m/s descent > 5 m/s descent limit.
        let mut state = nominal_state();
        state.velocity_mps = [0.0, 0.0, -7.0];
        let err = validate_constraints(&limits, &state, &terrain, &[], 0.50, -60.0).unwrap_err();
        assert!(
            matches!(err, ConstraintViolation::DescentRateExceeded { .. }),
            "wrong violation: {err:?}"
        );
    }
}
