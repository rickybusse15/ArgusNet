//! Stateful safety monitor for per-drone constraint tracking.
//!
//! The `SafetyMonitor` aggregates `ConstraintViolation` events, maintains the
//! `DroneSafetyState` for every known drone, and produces a `SafetyDecision`
//! (including a clamped command) for each processed command.
//!
//! State escalation rules are defined in SAFETY.md §4.1.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};
use terrain_engine::TerrainQuery;

use crate::limits::DronePhysicalLimits;
use crate::validator::{
    validate_constraints, ConstraintViolation, DroneCommandedState, DroneObservedState,
};

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/// Per-drone safety state maintained by the monitor.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DroneSafetyState {
    /// All constraints satisfied; operating normally.
    Nominal,

    /// One or more non-critical constraints are within the caution margin.
    Caution,

    /// A constraint violation has occurred; command has been clamped.
    Warning,

    /// A critical constraint is violated, or multiple warnings have
    /// accumulated.  The drone should execute the abort behaviour.
    Abort,

    /// The communications link has failed or RSSI has been below threshold
    /// for longer than `lost_link_timeout_s`.
    LostLink,
}

/// Configuration for the safety monitor.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Consecutive seconds of low RSSI before `LostLink` is declared (default: 3.0).
    pub lost_link_timeout_s: f64,

    /// AGL altitude for the lost-link climb manoeuvre in metres (default: min_agl_m * 3).
    pub lost_link_climb_agl_m: f64,

    /// Orbit radius during lost-link search in metres (default: 30.0).
    pub lost_link_orbit_radius_m: f64,

    /// Duration of lost-link orbit before RTH in seconds (default: 15.0).
    pub lost_link_orbit_s: f64,

    /// Initial hover duration when LostLink is first detected in seconds (default: 5.0).
    pub lost_link_hover_s: f64,

    /// Seconds to hold in abort before initiating automatic RTH (default: 30.0).
    pub abort_hold_timeout_s: f64,

    /// Number of consecutive violation frames that escalate Warning → Abort (default: 3).
    pub abort_repeated_violation_frames: u32,

    /// Fraction of limit range that triggers `Caution` state (default: 0.10).
    pub caution_margin_fraction: f64,
}

impl Default for SafetyConfig {
    fn default() -> Self {
        Self {
            lost_link_timeout_s: 3.0,
            lost_link_climb_agl_m: 0.0, // computed from limits if left at 0
            lost_link_orbit_radius_m: 30.0,
            lost_link_orbit_s: 15.0,
            lost_link_hover_s: 5.0,
            abort_hold_timeout_s: 30.0,
            abort_repeated_violation_frames: 3,
            caution_margin_fraction: 0.10,
        }
    }
}

/// The output of a single `process_command` call.
#[derive(Clone, Debug)]
pub struct SafetyDecision {
    /// Updated safety state after processing this command.
    pub state: DroneSafetyState,

    /// The clamped command (identical to the input if no violation occurred).
    pub clamped_command: DroneCommandedState,

    /// All violations detected in this evaluation.
    pub violations: Vec<ConstraintViolation>,

    /// Human-readable summary for logging and telemetry.
    pub summary: String,
}

/// A telemetry frame from an active drone.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DroneTelemetry {
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub energy_fraction: f64,
    pub comms_rssi_dbm: f64,
    pub timestamp_s: f64,
}

// ---------------------------------------------------------------------------
// Internal per-drone tracking state
// ---------------------------------------------------------------------------

#[derive(Clone, Debug)]
struct DroneRecord {
    /// Current safety state.
    state: DroneSafetyState,

    /// Accumulated violations since the last `clear`.
    violation_history: Vec<ConstraintViolation>,

    /// Number of consecutive frames with an active violation (for escalation).
    consecutive_violation_frames: u32,

    /// Accumulated seconds of low-RSSI readings (for lost-link detection).
    low_rssi_accumulated_s: f64,

    /// Timestamp of the last telemetry update (reserved for future use).
    #[allow(dead_code)]
    last_telemetry_ts: Option<f64>,
}

impl Default for DroneRecord {
    fn default() -> Self {
        Self {
            state: DroneSafetyState::Nominal,
            violation_history: Vec::new(),
            consecutive_violation_frames: 0,
            low_rssi_accumulated_s: 0.0,
            last_telemetry_ts: None,
        }
    }
}

// ---------------------------------------------------------------------------
// SafetyMonitor
// ---------------------------------------------------------------------------

/// Stateful safety monitor that tracks per-drone constraint violations and
/// escalates `DroneSafetyState` according to the rules in SAFETY.md §4.1.
pub struct SafetyMonitor {
    config: SafetyConfig,
    drones: HashMap<String, DroneRecord>,
}

impl SafetyMonitor {
    /// Create a new monitor with the given configuration.
    pub fn new(config: SafetyConfig) -> Self {
        Self {
            config,
            drones: HashMap::new(),
        }
    }

    /// Return the current safety state for `drone_id`, or `Nominal` if the
    /// drone has not yet been seen.
    pub fn state(&self, drone_id: &str) -> DroneSafetyState {
        self.drones
            .get(drone_id)
            .map(|r| r.state.clone())
            .unwrap_or(DroneSafetyState::Nominal)
    }

    /// Return the accumulated violation history for `drone_id`.
    pub fn violations(&self, drone_id: &str) -> &[ConstraintViolation] {
        self.drones
            .get(drone_id)
            .map(|r| r.violation_history.as_slice())
            .unwrap_or(&[])
    }

    /// Clear the violation log and reset the drone state to `Nominal`.
    pub fn clear(&mut self, drone_id: &str) {
        if let Some(r) = self.drones.get_mut(drone_id) {
            r.state = DroneSafetyState::Nominal;
            r.violation_history.clear();
            r.consecutive_violation_frames = 0;
            r.low_rssi_accumulated_s = 0.0;
        }
    }

    /// Process a new commanded state for `drone_id` and return a
    /// `SafetyDecision` containing the updated state, any violations, and the
    /// (potentially clamped) command.
    pub fn process_command(
        &mut self,
        drone_id: &str,
        command: DroneCommandedState,
        limits: &DronePhysicalLimits,
        terrain: &dyn TerrainQuery,
        peers: &[DroneObservedState],
        energy_fraction: f64,
        comms_rssi_dbm: f64,
    ) -> SafetyDecision {
        // Ensure the drone record exists.
        self.drones.entry(drone_id.to_string()).or_default();

        // Collect all violations (run the validator repeatedly until clean).
        let mut violations: Vec<ConstraintViolation> = Vec::new();
        let mut clamped = command.clone();

        // Run the pipeline once and collect up to one violation per category.
        // For now we do a single pass (the monitor's job is escalation, not
        // exhaustive enumeration — callers that need all violations may do
        // multiple passes externally).
        match validate_constraints(limits, &clamped, terrain, peers, energy_fraction, comms_rssi_dbm) {
            Ok(()) => {
                // No violation — clamp is the identity.
            }
            Err(v) => {
                // Apply a best-effort clamp and record the violation.
                clamp_command(&mut clamped, &v, limits);
                violations.push(v);

                // Run a second pass on the clamped command so we can catch
                // secondary violations caused by the original (e.g. clamp
                // speed but still have terrain issue).
                if let Err(v2) =
                    validate_constraints(limits, &clamped, terrain, peers, energy_fraction, comms_rssi_dbm)
                {
                    violations.push(v2);
                }
            }
        }

        // Determine whether a terrain violation is present (critical — always Abort).
        let has_terrain_violation = violations.iter().any(|v| {
            matches!(v, ConstraintViolation::TerrainClearanceViolation { .. })
        });

        // Check caution margin before taking the mutable borrow needed for state updates.
        let near_limit = self.near_any_limit(limits, &clamped, energy_fraction, comms_rssi_dbm);
        // Read config values before mutable borrow.
        let abort_frames = self.config.abort_repeated_violation_frames;
        let lost_link_timeout = self.config.lost_link_timeout_s;

        // Now take the mutable reference to the record.
        let record = self.drones.get_mut(drone_id).expect("record was just inserted");

        // Update violation log.
        record.violation_history.extend(violations.clone());

        // State machine transitions (SAFETY.md §4.1).
        let new_state = if violations.is_empty() {
            record.consecutive_violation_frames = 0;
            if near_limit {
                DroneSafetyState::Caution
            } else {
                DroneSafetyState::Nominal
            }
        } else {
            record.consecutive_violation_frames += 1;

            let should_abort = has_terrain_violation
                || record.consecutive_violation_frames >= abort_frames;

            if should_abort {
                DroneSafetyState::Abort
            } else {
                DroneSafetyState::Warning
            }
        };

        // LostLink override: if RSSI is below threshold, we track accumulated
        // time via a simple frame-count heuristic.  In production, the caller
        // should pass a timestamp via `update_telemetry`; here we use a 1 Hz
        // default approximation.
        let final_state = if comms_rssi_dbm < limits.min_comms_rssi_dbm {
            record.low_rssi_accumulated_s += 1.0; // assume ~1 s per frame
            if record.low_rssi_accumulated_s >= lost_link_timeout {
                DroneSafetyState::LostLink
            } else {
                new_state
            }
        } else {
            // Hysteresis: only clear LostLink if RSSI is 3 dB above threshold.
            if record.state == DroneSafetyState::LostLink
                && comms_rssi_dbm > limits.min_comms_rssi_dbm + 3.0
            {
                record.low_rssi_accumulated_s = 0.0;
            } else if record.state != DroneSafetyState::LostLink {
                record.low_rssi_accumulated_s = 0.0;
            }
            new_state
        };

        record.state = final_state.clone();

        let summary = if violations.is_empty() {
            format!("{drone_id}: {final_state:?} — nominal")
        } else {
            format!(
                "{drone_id}: {final_state:?} — {} violation(s); first: {:?}",
                violations.len(),
                violations[0]
            )
        };

        SafetyDecision {
            state: final_state,
            clamped_command: clamped,
            violations,
            summary,
        }
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Return `true` when any parameter is within the caution margin of its limit.
    fn near_any_limit(
        &self,
        limits: &DronePhysicalLimits,
        state: &DroneCommandedState,
        energy_fraction: f64,
        comms_rssi_dbm: f64,
    ) -> bool {
        let m = self.config.caution_margin_fraction;

        let speed = {
            let v = state.velocity_mps;
            (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt()
        };

        // Speed near max.
        if speed > limits.max_speed_mps * (1.0 - m) {
            return true;
        }

        // Energy near minimum.
        if energy_fraction < limits.min_energy_reserve_fraction * (1.0 + m) {
            return true;
        }

        // RSSI near minimum (10% of 20 dB range = 2 dB).
        let rssi_margin = 20.0 * m;
        if comms_rssi_dbm < limits.min_comms_rssi_dbm + rssi_margin {
            return true;
        }

        false
    }
}

// ---------------------------------------------------------------------------
// Best-effort clamping
// ---------------------------------------------------------------------------

/// Apply a minimal best-effort clamp to `cmd` based on the violation type.
///
/// This is advisory — the safety monitor reports violations regardless, and
/// the operator / flight controller is responsible for applying the clamp.
fn clamp_command(cmd: &mut DroneCommandedState, violation: &ConstraintViolation, limits: &DronePhysicalLimits) {
    match violation {
        ConstraintViolation::SpeedAboveMaximum { .. } => {
            let v = cmd.velocity_mps;
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed > 0.0 {
                let scale = limits.max_speed_mps / speed;
                cmd.velocity_mps = [v[0] * scale, v[1] * scale, v[2] * scale];
            }
        }
        ConstraintViolation::SpeedBelowMinimum { .. } => {
            let v = cmd.velocity_mps;
            let speed = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            if speed > 0.0 {
                let scale = limits.min_speed_mps / speed;
                cmd.velocity_mps = [v[0] * scale, v[1] * scale, v[2] * scale];
            }
        }
        ConstraintViolation::ClimbRateExceeded { .. } => {
            cmd.velocity_mps[2] = limits.max_climb_rate_mps;
        }
        ConstraintViolation::DescentRateExceeded { .. } => {
            cmd.velocity_mps[2] = -limits.max_descent_rate_mps;
        }
        ConstraintViolation::TerrainClearanceViolation { min_agl_m, .. } => {
            // Push the altitude up to the minimum safe AGL.  The terrain
            // height at the commanded position would be needed for a full
            // clamp; here we add the deficit to the commanded altitude.
            // The safety monitor escalates to Abort regardless.
            cmd.position_m[2] += min_agl_m - (cmd.position_m[2]);
        }
        ConstraintViolation::GimbalOutOfRange { axis, limit_rad, .. } => {
            match axis.as_str() {
                "pitch_down" => cmd.gimbal_pitch_rad = *limit_rad,
                "pitch_up" => cmd.gimbal_pitch_rad = -limit_rad,
                "yaw" => {
                    cmd.gimbal_yaw_offset_rad =
                        cmd.gimbal_yaw_offset_rad.clamp(-limit_rad, *limit_rad)
                }
                _ => {}
            }
        }
        // Other violations do not have a simple pointwise clamp.
        _ => {}
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::limits::DronePhysicalLimits;
    use terrain_engine::{FlatTerrain, TerrainBounds};

    fn make_monitor() -> SafetyMonitor {
        SafetyMonitor::new(SafetyConfig::default())
    }

    fn make_flat() -> FlatTerrain {
        FlatTerrain::new(0.0, TerrainBounds::new(-10000.0, 10000.0, -10000.0, 10000.0, 0.0, 0.0))
    }

    fn nominal_command() -> DroneCommandedState {
        DroneCommandedState {
            position_m: [0.0, 0.0, 100.0],
            velocity_mps: [10.0, 0.0, 0.0],
            acceleration_mps2: [0.0, 0.0, 0.0],
            gimbal_pitch_rad: 0.5,
            gimbal_yaw_offset_rad: 0.0,
        }
    }

    #[test]
    fn nominal_state_on_clean_command() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        let decision = monitor.process_command(
            "drone-1",
            nominal_command(),
            &limits,
            &terrain,
            &[],
            0.50,
            -60.0,
        );
        assert!(
            matches!(
                decision.state,
                DroneSafetyState::Nominal | DroneSafetyState::Caution
            ),
            "expected Nominal or Caution, got {:?}",
            decision.state
        );
    }

    #[test]
    fn warning_on_first_violation() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // Speed violation (50 m/s > 42 m/s limit).
        let mut cmd = nominal_command();
        cmd.velocity_mps = [50.0, 0.0, 0.0];

        let decision = monitor.process_command(
            "drone-1",
            cmd,
            &limits,
            &terrain,
            &[],
            0.50,
            -60.0,
        );
        assert_eq!(decision.state, DroneSafetyState::Warning);
        assert!(!decision.violations.is_empty());
    }

    #[test]
    fn abort_on_terrain_violation() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // Place drone at 5 m AGL — below the 30 m minimum.
        let mut cmd = nominal_command();
        cmd.position_m[2] = 5.0;

        let decision = monitor.process_command(
            "drone-1",
            cmd,
            &limits,
            &terrain,
            &[],
            0.50,
            -60.0,
        );
        assert_eq!(
            decision.state,
            DroneSafetyState::Abort,
            "terrain violation should immediately escalate to Abort"
        );
    }

    #[test]
    fn state_escalation_warning_to_abort_after_repeated_violations() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // Speed violation repeated 3 times.
        let mut cmd = nominal_command();
        cmd.velocity_mps = [50.0, 0.0, 0.0];

        let d1 = monitor.process_command("drone-1", cmd.clone(), &limits, &terrain, &[], 0.50, -60.0);
        let d2 = monitor.process_command("drone-1", cmd.clone(), &limits, &terrain, &[], 0.50, -60.0);
        let d3 = monitor.process_command("drone-1", cmd.clone(), &limits, &terrain, &[], 0.50, -60.0);

        assert_eq!(d1.state, DroneSafetyState::Warning, "frame 1 should be Warning");
        assert_eq!(d2.state, DroneSafetyState::Warning, "frame 2 should be Warning");
        assert_eq!(d3.state, DroneSafetyState::Abort, "frame 3 should escalate to Abort");
    }

    #[test]
    fn lost_link_after_sustained_low_rssi() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // RSSI = -95 dBm (below -85 threshold); default timeout = 3 s → 3 frames.
        let cmd = nominal_command();

        for frame in 0..4 {
            let decision = monitor.process_command(
                "drone-1",
                cmd.clone(),
                &limits,
                &terrain,
                &[],
                0.50,
                -95.0, // below threshold
            );
            if frame < 2 {
                // First 2 frames: warning/caution/nominal (not yet lost link).
                assert_ne!(
                    decision.state,
                    DroneSafetyState::LostLink,
                    "frame {frame} should not yet be LostLink"
                );
            } else {
                // Frame 3 and beyond: LostLink should be declared.
                assert_eq!(
                    decision.state,
                    DroneSafetyState::LostLink,
                    "frame {frame} should be LostLink"
                );
            }
        }
    }

    #[test]
    fn separation_violation_detected() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // Peer drone is 5 m away — below the 8 m horizontal minimum.
        let peers = vec![DroneObservedState {
            drone_id: "peer-1".to_string(),
            position_m: [5.0, 0.0, 100.0],
        }];

        let decision = monitor.process_command(
            "drone-1",
            nominal_command(),
            &limits,
            &terrain,
            &peers,
            0.50,
            -60.0,
        );

        let has_sep = decision
            .violations
            .iter()
            .any(|v| matches!(v, ConstraintViolation::DroneSeparationViolation { .. }));
        assert!(has_sep, "expected DroneSeparationViolation in {:?}", decision.violations);
    }

    #[test]
    fn clear_resets_state() {
        let mut monitor = make_monitor();
        let limits = DronePhysicalLimits::interceptor_default();
        let terrain = make_flat();

        // Trigger abort.
        let mut cmd = nominal_command();
        cmd.position_m[2] = 5.0;
        monitor.process_command("drone-1", cmd, &limits, &terrain, &[], 0.50, -60.0);
        assert_eq!(monitor.state("drone-1"), DroneSafetyState::Abort);

        monitor.clear("drone-1");
        assert_eq!(monitor.state("drone-1"), DroneSafetyState::Nominal);
    }
}
