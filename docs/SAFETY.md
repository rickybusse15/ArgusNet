# Drone Physical Limits Schema and Safety Monitor Contract

Stage 0 interface definition. Formalises the drone physical limits schema,
constraint validation pipeline, safety monitor interface, and abort/lost-link
behaviours.

---

## 1. Drone Physical Limits Schema

### 1.1 Canonical Rust type

```rust
use serde::{Deserialize, Serialize};

/// Complete physical envelope for one drone platform type.
///
/// All values are in SI units (metres, seconds, radians) unless noted.
/// This struct is the single source of truth consumed by:
/// - The safety constraint validator (Section 3)
/// - The flight dynamics simulator (DynamicsConfig maps onto this)
/// - The path planner (turn radius, clearance)
/// - The safety monitor (Section 4)
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DronePhysicalLimits {
    // ------------------------------------------------------------------
    // Speed envelope
    // ------------------------------------------------------------------

    /// Maximum airspeed (m/s). Hard ceiling; controller must not command above this.
    pub max_speed_mps: f64,

    /// Minimum airspeed for controlled flight (m/s). Below this the vehicle
    /// is stall-limited. Hover-capable platforms may set this to 0.0.
    pub min_speed_mps: f64,

    // ------------------------------------------------------------------
    // Turn performance
    // ------------------------------------------------------------------

    /// Maximum sustained bank angle (radians). Used to derive min_turn_radius.
    /// Equivalent to `FlightEnvelope.max_bank_angle_deg` (converted to radians).
    pub max_bank_angle_rad: f64,

    /// Maximum instantaneous yaw rate (rad/s) at cruise speed.
    /// For fixed-wing platforms this is derived from bank angle and speed.
    /// For multirotors set directly.
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

    /// Minimum altitude above ground level (m). Hard floor; the vehicle
    /// must never descend below terrain + this margin.
    pub min_agl_m: f64,

    /// Maximum altitude above ground level (m). Operational ceiling.
    pub max_agl_m: f64,

    // ------------------------------------------------------------------
    // Energy and endurance
    // ------------------------------------------------------------------

    /// Minimum energy reserve before return-to-home is mandatory.
    /// Expressed as a fraction of full battery capacity [0.0, 1.0].
    /// Default: 0.20 (20% reserve).
    pub min_energy_reserve_fraction: f64,

    /// Energy consumption rate at cruise speed (W or normalised units/s).
    /// Used to estimate time-to-empty and plan return legs.
    pub cruise_power_w: f64,

    // ------------------------------------------------------------------
    // Communications
    // ------------------------------------------------------------------

    /// Minimum acceptable uplink signal strength for commanded flight,
    /// expressed as received signal strength indicator (RSSI) in dBm.
    /// Below this threshold the drone enters lost-link behaviour.
    /// Typical value: -85 dBm.
    pub min_comms_rssi_dbm: f64,

    /// Maximum communications range (m) at which nominal RSSI is maintained.
    /// Used by the terrain query comms_shadow check.
    pub max_comms_range_m: f64,

    // ------------------------------------------------------------------
    // Sensor gimbal
    // ------------------------------------------------------------------

    /// Maximum gimbal pitch angle below the horizon (radians, positive = down).
    /// 0.0 = level; pi/2 = straight down.
    pub gimbal_max_depression_rad: f64,

    /// Maximum gimbal elevation above the horizon (radians, positive = up).
    pub gimbal_max_elevation_rad: f64,

    /// Maximum gimbal yaw offset from the vehicle heading (radians).
    /// Symmetric: the gimbal can slew this far left or right.
    pub gimbal_max_yaw_offset_rad: f64,

    // ------------------------------------------------------------------
    // Separation from other drones
    // ------------------------------------------------------------------

    /// Minimum horizontal separation between any two drones (m).
    /// Collision avoidance must keep drones at least this far apart in XY.
    pub min_drone_separation_m: f64,

    /// Minimum vertical separation between any two drones (m).
    pub min_drone_vertical_separation_m: f64,

    // ------------------------------------------------------------------
    // Platform identity
    // ------------------------------------------------------------------

    /// Human-readable platform type label (e.g. "interceptor", "tracker").
    pub platform_type: String,
}

impl DronePhysicalLimits {
    /// Minimum turn radius (metres) at the current speed.
    ///
    /// r = v² / (g × tan(bank_angle))
    pub fn min_turn_radius_at(&self, speed_mps: f64) -> f64 {
        const G: f64 = 9.80665;
        let bank = self.max_bank_angle_rad.max(0.01_f64.to_radians());
        speed_mps * speed_mps / (G * bank.tan())
    }
}
```

### 1.2 Mapping to existing Python constants

The `DynamicsConfig` and `FlightEnvelope` in `config.py` and `behaviors.py`
contain partial overlapping data. The mapping:

| `DronePhysicalLimits` field | Python source | Current value |
|-----------------------------|---------------|---------------|
| `max_speed_mps` | `DynamicsConfig.drone_search_speed_base_mps` (approx) / `FlightEnvelope.max_speed_mps` | 50.0 / 28.0 |
| `min_speed_mps` | `FlightEnvelope.min_speed_mps` | 8.0 |
| `max_bank_angle_rad` | `FlightEnvelope.max_bank_angle_deg` (convert) | 30° → 0.524 rad |
| `max_climb_rate_mps` | `FlightEnvelope.max_climb_rate_mps` | 8.0 |
| `max_descent_rate_mps` | `FlightEnvelope.max_descent_rate_mps` | 5.0 |
| `max_horizontal_accel_mps2` | `DynamicsConfig.drone_max_accel_mps2` | 8.0 |
| `min_agl_m` | `DynamicsConfig.interceptor_follow_min_agl_m` / `terrain_following_agl_m` | 150.0 / 30.0 |
| `max_agl_m` | `FlightEnvelope.max_altitude_agl_m` | 500.0 |
| `min_energy_reserve_fraction` | Not present | Default 0.20 |
| `min_comms_rssi_dbm` | Not present | Default -85.0 |
| `max_comms_range_m` | `SensorConfig.drone_base_max_range_m` (proxy) | 650.0 |
| `gimbal_max_depression_rad` | `SensorConfig.drone_look_down_angle_deg` (proxy, convert) | 30° → 0.524 rad |
| `min_drone_separation_m` | `PlannerConfig.drone_clearance_m` (minimum) | 8.0 |

**Note:** `FlightEnvelope` lives in `behaviors.py` and is used by trajectory
generators. `DronePhysicalLimits` is the architecture-level schema that
subsumes `FlightEnvelope` and extends it with comms, energy, and gimbal fields.
`FlightEnvelope` should be made an alias or replaced by a view over
`DronePhysicalLimits` in a future migration.

### 1.3 Default platform profiles

```rust
impl DronePhysicalLimits {
    /// Interceptor-role drone: fast, tight turn, moderate endurance.
    pub fn interceptor_default() -> Self {
        Self {
            max_speed_mps: 42.0,
            min_speed_mps: 0.0,       // multirotor: hover-capable
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

    /// Tracker-role drone: wider orbit, longer endurance, larger standoff.
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
            gimbal_max_elevation_rad: 0.175,
            gimbal_max_yaw_offset_rad: 1.047, // 60°
            min_drone_separation_m: 12.0,
            min_drone_vertical_separation_m: 8.0,
            platform_type: "tracker".to_string(),
        }
    }
}
```

---

## 2. Constraint Validation Pipeline

Every candidate commanded state (position + velocity) is passed through a
validation pipeline before being applied. The pipeline is a sequence of
independent checks; failure at any stage returns the specific violated
constraint.

```rust
/// Result of a constraint validation check.
#[derive(Clone, Debug)]
pub enum ConstraintViolation {
    SpeedAboveMaximum { commanded_mps: f64, limit_mps: f64 },
    SpeedBelowMinimum { commanded_mps: f64, limit_mps: f64 },
    AccelerationExceeded { commanded_mps2: f64, limit_mps2: f64 },
    TurnRadiusTooTight { commanded_radius_m: f64, min_radius_m: f64 },
    ClimbRateExceeded { commanded_mps: f64, limit_mps: f64 },
    DescentRateExceeded { commanded_mps: f64, limit_mps: f64 },
    TerrainClearanceViolation { agl_m: f64, min_agl_m: f64 },
    AltitudeCeilingExceeded { agl_m: f64, max_agl_m: f64 },
    EnergyReserveLow { reserve_fraction: f64, min_fraction: f64 },
    CommsTooWeak { rssi_dbm: f64, threshold_dbm: f64 },
    CommsShadowed,
    DroneSeparationViolation { drone_id: String, distance_m: f64, min_m: f64 },
    GimbalOutOfRange { axis: String, angle_rad: f64, limit_rad: f64 },
}

/// Validate a candidate drone state against the physical limits.
///
/// Returns `Ok(())` if all constraints pass, or the first `Err(ConstraintViolation)`
/// encountered. Callers that need all violations may call in a loop after
/// clamping each violation in turn.
pub fn validate_constraints(
    limits: &DronePhysicalLimits,
    candidate: &DroneCommandedState,
    terrain: &dyn TerrainQuery,
    peers: &[DroneObservedState],
    energy_fraction: f64,
    comms_rssi_dbm: f64,
) -> Result<(), ConstraintViolation>;

/// Commanded state input to the validator.
#[derive(Clone, Debug)]
pub struct DroneCommandedState {
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub acceleration_mps2: [f64; 3],
    pub gimbal_pitch_rad: f64,
    pub gimbal_yaw_offset_rad: f64,
}

/// Peer drone observed state used for separation checks.
#[derive(Clone, Debug)]
pub struct DroneObservedState {
    pub drone_id: String,
    pub position_m: [f64; 3],
}
```

### 2.1 Validation sequence

The pipeline runs in this order to short-circuit on the most dangerous
violations first:

1. **Terrain clearance** — check AGL against `min_agl_m`.
2. **Altitude ceiling** — check AGL against `max_agl_m`.
3. **Speed limits** — check `|velocity|` against `[min_speed_mps, max_speed_mps]`.
4. **Acceleration** — check `|acceleration|` against `max_horizontal_accel_mps2`
   (horizontal) and `max_vertical_accel_mps2` (vertical).
5. **Climb/descent rate** — check `velocity.z` against rate limits.
6. **Turn radius** — compute instantaneous turn radius from lateral acceleration;
   compare to `min_turn_radius_at(|velocity|)`.
7. **Drone separation** — for each peer, check 3D distance against separation
   limits.
8. **Energy reserve** — check `energy_fraction` against `min_energy_reserve_fraction`.
9. **Communications** — check RSSI against `min_comms_rssi_dbm`. If RSSI is
   unavailable, call `terrain.comms_shadow(drone_xyz, gcs_xyz)`.
10. **Gimbal limits** — check commanded gimbal angles against the three axis limits.

### 2.2 Clamping vs rejection

The pipeline is **advisory by default**. A clamped version of the command is
produced alongside any violations. The safety monitor (Section 4) decides
whether to apply the clamp silently or escalate to an abort.

---

## 3. Python-side Constraint Enforcement (existing)

The Python simulation already enforces a subset of these constraints
implicitly:

| Constraint | Current Python mechanism |
|-----------|--------------------------|
| Terrain clearance | `TerrainModel.clamp_altitude` + `DynamicsConfig.*_min_agl_m` |
| Speed cap | `follow_speed_cap_mps` in `PlatformPresetProfile` |
| Collision avoidance | `planning.py` obstacle push + `DynamicsConfig.collision_push_margin_m` |
| Physical collision below terrain | Critical rule in CLAUDE.md: "Physical collision must never push entities below terrain" |
| Drone separation | Not formally enforced; implicit via orbit radius spacing |
| Energy / comms | Not modelled (simulation assumes perfect comms) |

The architecture update formalises these into the `DronePhysicalLimits` schema
and the `validate_constraints` function.

---

## 4. Safety Monitor Interface

The safety monitor is a stateful component that aggregates constraint
violation events, maintains per-drone safety state, and triggers escalated
responses.

```rust
/// Per-drone safety state maintained by the monitor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DroneSafetyState {
    /// Nominal: all constraints satisfied, operating normally.
    Nominal,

    /// Caution: one or more non-critical constraints are near their limits
    /// (within a configurable margin). No action taken; telemetry flagged.
    Caution,

    /// Warning: a constraint violation has occurred. Command is clamped.
    /// The drone continues to operate under the clamped command.
    Warning,

    /// Abort: a critical constraint is violated or multiple warnings
    /// have accumulated. The drone executes the abort behaviour (Section 5).
    Abort,

    /// LostLink: the communications link has failed or RSSI has dropped
    /// below threshold for longer than `lost_link_timeout_s`.
    LostLink,
}

/// Safety monitor trait.
pub trait SafetyMonitor: Send + Sync {
    /// Process a new commanded state for a drone.
    ///
    /// Returns the safety state after processing, along with the clamped
    /// command (which equals `command` if no violation occurred).
    fn process_command(
        &mut self,
        drone_id: &str,
        command: DroneCommandedState,
        context: &SafetyContext,
    ) -> SafetyDecision;

    /// Update the safety monitor with a new telemetry frame (position,
    /// energy, RSSI) without issuing a new command.
    fn update_telemetry(
        &mut self,
        drone_id: &str,
        telemetry: &DroneTelemetry,
        context: &SafetyContext,
    );

    /// Return the current safety state for a drone.
    fn state(&self, drone_id: &str) -> DroneSafetyState;

    /// Return the accumulated violation log for a drone since last cleared.
    fn violations(&self, drone_id: &str) -> &[ConstraintViolation];

    /// Clear the violation log and reset state to Nominal (use with caution).
    fn clear(&mut self, drone_id: &str);
}

/// Output of a safety monitor command evaluation.
#[derive(Clone, Debug)]
pub struct SafetyDecision {
    pub state: DroneSafetyState,
    pub clamped_command: DroneCommandedState,
    pub violations: Vec<ConstraintViolation>,
    /// Human-readable summary for logging/telemetry.
    pub summary: String,
}

/// Contextual data required by the safety monitor for each evaluation.
pub struct SafetyContext<'a> {
    pub limits: &'a DronePhysicalLimits,
    pub terrain: &'a dyn TerrainQuery,
    pub peers: &'a [DroneObservedState],
    pub energy_fraction: f64,
    pub comms_rssi_dbm: f64,
    /// Position of the ground control station (for comms shadow check).
    pub gcs_position_m: [f64; 3],
}

/// Telemetry frame from an active drone.
#[derive(Clone, Debug)]
pub struct DroneTelemetry {
    pub position_m: [f64; 3],
    pub velocity_mps: [f64; 3],
    pub energy_fraction: f64,
    pub comms_rssi_dbm: f64,
    pub timestamp_s: f64,
}
```

### 4.1 State escalation rules

| From | To | Trigger |
|------|----|---------|
| Nominal | Caution | Any constraint within 10% of its limit |
| Nominal / Caution | Warning | Any constraint violated; clamp applied |
| Warning | Abort | Same violation on 3 consecutive frames, or `TerrainClearanceViolation` on any frame |
| Any | LostLink | `rssi_dbm < min_comms_rssi_dbm` for `lost_link_timeout_s` consecutive seconds |
| Abort | Nominal | Violation cleared AND drone has returned to a safe state |
| LostLink | Nominal | Link restored AND `rssi_dbm > min_comms_rssi_dbm + 3 dB` (hysteresis) |

---

## 5. Abort and Lost-Link Behaviours

### 5.1 Abort behaviour

Triggered when `DroneSafetyState == Abort`.

```
1. Freeze the current commanded velocity (no new commands accepted).
2. Apply maximum deceleration to zero ground speed.
3. Climb to max(current_z + 20 m, terrain_height + min_agl_m + 20 m).
4. Hold position in a 10 m radius hover.
5. Transmit ABORT telemetry to GCS every 0.5 s.
6. Await operator RESUME command or timeout.
7. After abort_hold_timeout_s (default: 30 s) without RESUME, transition to
   return-to-home.
```

The abort hold altitude is chosen to clear the terrain clearance constraint
by a factor of 2 while remaining within comms range.

### 5.2 Low-energy return-to-home

Triggered when `energy_fraction < min_energy_reserve_fraction`.

```
1. Complete the current observation frame (if update_count > 0 this frame).
2. Compute direct RTH path using visibility-graph planner (planning.py).
3. Fly RTH path at max_speed_mps, maintaining min_agl_m.
4. If terrain comms_shadow covers the RTH path, climb to the minimum altitude
   that restores comms before proceeding.
5. Land at the home station.
```

Energy budget for RTH:
```
rtl_energy = (rtl_distance_m / max_speed_mps) * cruise_power_w / 3600.0  [Wh]
reserve_wh = battery_capacity_wh * min_energy_reserve_fraction
Trigger RTH when: remaining_wh <= reserve_wh + rtl_energy * 1.2 (20% margin)
```

### 5.3 Lost-link behaviour

Triggered when `DroneSafetyState == LostLink`.

```
1. Enter hover hold for lost_link_hover_s (default: 5 s).
   Purpose: allow transient interference to clear.
2. If link not restored:
   a. Climb to lost_link_climb_agl_m (default: min_agl_m * 3, capped at max_agl_m).
      Reason: higher altitude typically improves RF line-of-sight to GCS.
   b. Circle at lost_link_orbit_radius_m (default: 30 m) for lost_link_orbit_s (default: 15 s).
3. If link still not restored:
   a. Begin autonomous RTH on the last-known GCS bearing.
   b. Climb above terrain along the RTH path, maintaining comms shadow checks.
4. On RTH completion, land at home station.
5. If RTH path is entirely in comms shadow:
   a. Fly to the highest terrain point along the RTH corridor.
   b. Hold at that point until battery is at min_energy_reserve_fraction.
   c. Execute emergency landing at the current position.
```

### 5.4 Configuration

```rust
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SafetyConfig {
    /// Consecutive seconds of low RSSI before LostLink is declared.
    pub lost_link_timeout_s: f64,          // default: 3.0

    /// AGL altitude for the lost-link climb manoeuvre (m).
    pub lost_link_climb_agl_m: f64,        // default: min_agl_m * 3.0

    /// Orbit radius during lost-link search (m).
    pub lost_link_orbit_radius_m: f64,     // default: 30.0

    /// Duration of lost-link orbit before RTH (s).
    pub lost_link_orbit_s: f64,            // default: 15.0

    /// Initial hover duration when LostLink is first detected (s).
    pub lost_link_hover_s: f64,            // default: 5.0

    /// Seconds to hold in abort before initiating automatic RTH.
    pub abort_hold_timeout_s: f64,         // default: 30.0

    /// Fraction below which warning is escalated to abort on repeated violation.
    pub abort_repeated_violation_frames: u32, // default: 3

    /// Caution margin: fraction of limit range that triggers Caution state.
    pub caution_margin_fraction: f64,      // default: 0.10
}
```

---

## 6. Physical Collision Invariant

Per CLAUDE.md critical rules:

> Physical collision must never push entities below terrain.

The safety monitor enforces this as the **highest-priority** constraint in the
validation pipeline (step 1). Any clamped command that would result in a
position at or below `terrain.height_at(x, y) + min_agl_m` is escalated
directly to `Abort` without waiting for repeated violation.

This invariant must hold even when the safety monitor is operating in
simulation mode. Test coverage must include terrain-intersection edge cases
(steep ridgelines, river valleys, mountain-pass saddles).
