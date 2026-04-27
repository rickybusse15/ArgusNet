//! Drone physical limits schema.
//!
//! `DronePhysicalLimits` is the single source of truth consumed by the safety
//! constraint validator, the flight dynamics simulator, the path planner, and
//! the safety monitor.  All values are in SI units (metres, seconds, radians)
//! unless otherwise noted.

use serde::{Deserialize, Serialize};

/// Complete physical envelope for one drone platform type.
///
/// All values are in SI units (metres, seconds, radians) unless noted.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DronePhysicalLimits {
    // ------------------------------------------------------------------
    // Speed envelope
    // ------------------------------------------------------------------
    /// Maximum airspeed (m/s). Hard ceiling; controller must not command above this.
    pub max_speed_mps: f64,

    /// Minimum airspeed for controlled flight (m/s). Below this the vehicle
    /// is stall-limited.  Hover-capable platforms may set this to 0.0.
    pub min_speed_mps: f64,

    // ------------------------------------------------------------------
    // Turn performance
    // ------------------------------------------------------------------
    /// Maximum sustained bank angle (radians). Used to derive min_turn_radius.
    pub max_bank_angle_rad: f64,

    /// Maximum instantaneous yaw rate (rad/s) at cruise speed.
    pub max_yaw_rate_rad_s: f64,

    // ------------------------------------------------------------------
    // Climb / descent
    // ------------------------------------------------------------------
    /// Maximum sustained climb rate (m/s, positive upward).
    pub max_climb_rate_mps: f64,

    /// Maximum sustained descent rate (m/s, positive magnitude).
    pub max_descent_rate_mps: f64,

    // ------------------------------------------------------------------
    // Acceleration limits
    // ------------------------------------------------------------------
    /// Maximum horizontal acceleration magnitude (m/s²).
    pub max_horizontal_accel_mps2: f64,

    /// Maximum vertical acceleration magnitude (m/s²).
    pub max_vertical_accel_mps2: f64,

    // ------------------------------------------------------------------
    // Altitude constraints
    // ------------------------------------------------------------------
    /// Minimum altitude above ground level (m). Hard floor.
    pub min_agl_m: f64,

    /// Maximum altitude above ground level (m). Operational ceiling.
    pub max_agl_m: f64,

    // ------------------------------------------------------------------
    // Energy and endurance
    // ------------------------------------------------------------------
    /// Minimum energy reserve before return-to-home is mandatory.
    /// Expressed as a fraction of full battery capacity [0.0, 1.0].
    pub min_energy_reserve_fraction: f64,

    /// Energy consumption rate at cruise speed (W).
    pub cruise_power_w: f64,

    // ------------------------------------------------------------------
    // Communications
    // ------------------------------------------------------------------
    /// Minimum acceptable uplink RSSI for commanded flight (dBm).
    pub min_comms_rssi_dbm: f64,

    /// Maximum communications range (m) at nominal RSSI.
    pub max_comms_range_m: f64,

    // ------------------------------------------------------------------
    // Sensor gimbal
    // ------------------------------------------------------------------
    /// Maximum gimbal pitch angle below the horizon (radians, positive = down).
    pub gimbal_max_depression_rad: f64,

    /// Maximum gimbal elevation above the horizon (radians, positive = up).
    pub gimbal_max_elevation_rad: f64,

    /// Maximum gimbal yaw offset from the vehicle heading (radians, symmetric).
    pub gimbal_max_yaw_offset_rad: f64,

    // ------------------------------------------------------------------
    // Separation from other drones
    // ------------------------------------------------------------------
    /// Minimum horizontal separation between any two drones (m).
    pub min_drone_separation_m: f64,

    /// Minimum vertical separation between any two drones (m).
    pub min_drone_vertical_separation_m: f64,

    // ------------------------------------------------------------------
    // Platform identity
    // ------------------------------------------------------------------
    /// Human-readable platform type label (e.g. `"interceptor"`, `"tracker"`).
    pub platform_type: String,
}

impl DronePhysicalLimits {
    /// Minimum turn radius (metres) at the given airspeed.
    ///
    /// r = v² / (g × tan(bank_angle))
    pub fn min_turn_radius_at(&self, speed_mps: f64) -> f64 {
        const G: f64 = 9.80665;
        // Clamp bank angle away from zero to avoid division-by-zero for
        // bank angles below 0.01° (near-zero bank configurations).
        let bank = self.max_bank_angle_rad.max(0.01_f64.to_radians());
        speed_mps * speed_mps / (G * bank.tan())
    }

    /// Factory: interceptor-role drone (fast, tight turn, moderate endurance).
    pub fn interceptor_default() -> Self {
        Self {
            max_speed_mps: 42.0,
            min_speed_mps: 0.0,        // multirotor: hover-capable
            max_bank_angle_rad: 0.524, // 30°
            max_yaw_rate_rad_s: 0.8,
            max_climb_rate_mps: 8.0,
            max_descent_rate_mps: 5.0,
            max_horizontal_accel_mps2: 8.0,
            max_vertical_accel_mps2: 4.0,
            min_agl_m: 30.0,
            max_agl_m: 500.0,
            min_energy_reserve_fraction: 0.20,
            cruise_power_w: 400.0,
            min_comms_rssi_dbm: -85.0,
            max_comms_range_m: 2000.0,
            gimbal_max_depression_rad: 1.396, // 80°
            gimbal_max_elevation_rad: 0.175,  // 10°
            gimbal_max_yaw_offset_rad: 1.571, // 90°
            min_drone_separation_m: 8.0,
            min_drone_vertical_separation_m: 5.0,
            platform_type: "interceptor".to_string(),
        }
    }

    /// Factory: argusnet-role drone (wider orbit, longer endurance, larger standoff).
    pub fn tracker_default() -> Self {
        Self {
            max_speed_mps: 35.0,
            min_speed_mps: 0.0,
            max_bank_angle_rad: 0.436, // 25°
            max_yaw_rate_rad_s: 0.5,
            max_climb_rate_mps: 6.0,
            max_descent_rate_mps: 4.0,
            max_horizontal_accel_mps2: 6.0,
            max_vertical_accel_mps2: 3.0,
            min_agl_m: 50.0,
            max_agl_m: 600.0,
            min_energy_reserve_fraction: 0.25,
            cruise_power_w: 350.0,
            min_comms_rssi_dbm: -85.0,
            max_comms_range_m: 3000.0,
            gimbal_max_depression_rad: 1.396, // 80°
            gimbal_max_elevation_rad: 0.175,  // 10°
            gimbal_max_yaw_offset_rad: 1.047, // 60°
            min_drone_separation_m: 12.0,
            min_drone_vertical_separation_m: 8.0,
            platform_type: "tracker".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interceptor_default_fields() {
        let l = DronePhysicalLimits::interceptor_default();
        assert_eq!(l.max_speed_mps, 42.0);
        assert_eq!(l.min_speed_mps, 0.0);
        assert_eq!(l.max_bank_angle_rad, 0.524);
        assert_eq!(l.min_agl_m, 30.0);
        assert_eq!(l.max_agl_m, 500.0);
        assert_eq!(l.min_energy_reserve_fraction, 0.20);
        assert_eq!(l.platform_type, "interceptor");
    }

    #[test]
    fn tracker_default_fields() {
        let l = DronePhysicalLimits::tracker_default();
        assert_eq!(l.max_speed_mps, 35.0);
        assert_eq!(l.min_agl_m, 50.0);
        assert_eq!(l.max_agl_m, 600.0);
        assert_eq!(l.min_energy_reserve_fraction, 0.25);
        assert_eq!(l.platform_type, "tracker");
    }

    #[test]
    fn min_turn_radius_at_known_speed() {
        let l = DronePhysicalLimits::interceptor_default();
        // At 30 m/s, bank 30°:  r = 30² / (9.80665 × tan(0.524))
        // tan(0.524) ≈ 0.5774 → r ≈ 900 / 5.664 ≈ 158.9 m
        let r = l.min_turn_radius_at(30.0);
        assert!(r > 150.0 && r < 170.0, "unexpected turn radius {r}");
    }

    #[test]
    fn min_turn_radius_zero_speed() {
        let l = DronePhysicalLimits::interceptor_default();
        assert_eq!(l.min_turn_radius_at(0.0), 0.0);
    }

    #[test]
    fn serde_roundtrip() {
        let original = DronePhysicalLimits::tracker_default();
        let json = serde_json::to_string(&original).expect("serialize");
        let decoded: DronePhysicalLimits = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(decoded.max_speed_mps, original.max_speed_mps);
        assert_eq!(decoded.platform_type, original.platform_type);
    }
}
