//! Battery accounting for a multirotor platform.
//!
//! Scoped by reachability, the way P4.4 taught: of `planning/battery.py`'s
//! surface, only `segment_cost_wh`, `return_home_cost_wh`,
//! `dynamic_reserve_wh` and `BatteryState::consume` are called from outside
//! the module (`travel_cost_wh` and `climb_cost_wh` transitively, through
//! `segment_cost_wh`). Not ported, because nothing calls them:
//! `route_cost_wh`, `usable_capacity_wh`, `compute_route_budget`,
//! `insert_return_waypoint`, `BudgetResult`.
//!
//! Also not ported: `BatteryState.fraction`. It is a property named for a
//! fraction that returns **watt-hours** -- its own docstring says "caller
//! normalises against capacity" -- and it is referenced nowhere in `src/` or
//! `tests/`. `sim.py` computes `remaining / capacity` itself. Carrying it
//! across would give a misleading name a second life in a typed language,
//! where `fraction()` returning an energy is worse than in Python.

/// How a tick's energy is charged (D-3, decided 2026-10-05).
///
/// `Legacy` is the reference's accounting: only motion costs energy, so a
/// stationary platform draws nothing. It is kept unchanged so every existing
/// run, golden file and acceptance result stays valid. `PhysicalV1` also
/// charges hover power for the part of each tick not spent moving; see
/// [`BatteryModel::tick_cost_wh`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EnergyProfile {
    #[default]
    Legacy,
    PhysicalV1,
}

impl EnergyProfile {
    pub fn is_legacy(&self) -> bool {
        *self == EnergyProfile::Legacy
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            EnergyProfile::Legacy => "legacy",
            EnergyProfile::PhysicalV1 => "physical_v1",
        }
    }
}

/// Energy model for a multirotor. Defaults match the reference's dataclass.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatteryModel {
    pub capacity_wh: f64,
    pub hover_power_w: f64,
    pub travel_power_w: f64,
    pub climb_power_w: f64,
    pub cruise_speed_m_per_s: f64,
    pub climb_speed_m_per_s: f64,
    pub reserve_fraction: f64,
}

impl Default for BatteryModel {
    fn default() -> Self {
        Self {
            capacity_wh: 100.0,
            hover_power_w: 300.0,
            travel_power_w: 350.0,
            climb_power_w: 500.0,
            cruise_speed_m_per_s: 10.0,
            climb_speed_m_per_s: 3.0,
            reserve_fraction: 0.20,
        }
    }
}

impl BatteryModel {
    /// Energy for a horizontal segment.
    ///
    /// The speed is floored at 0.01 m/s rather than guarded, so a zero cruise
    /// speed yields an enormous but finite cost instead of an infinity.
    pub fn travel_cost_wh(&self, distance_m: f64) -> f64 {
        let duration_s = distance_m / self.cruise_speed_m_per_s.max(0.01);
        self.travel_power_w * duration_s / 3600.0
    }

    /// Energy for a vertical segment.
    ///
    /// Asymmetric on purpose: a climb draws `climb_power_w`, a descent draws
    /// `hover_power_w * 0.8`. Note the branch is `delta_z_m > 0`, so a delta of
    /// exactly zero takes the descent power -- which costs nothing anyway,
    /// since the duration is zero, but the branch is worth reproducing rather
    /// than "tidying" to `>=`.
    pub fn climb_cost_wh(&self, delta_z_m: f64) -> f64 {
        let duration_s = delta_z_m.abs() / self.climb_speed_m_per_s.max(0.01);
        let power = if delta_z_m > 0.0 {
            self.climb_power_w
        } else {
            self.hover_power_w * 0.8
        };
        power * duration_s / 3600.0
    }

    /// Energy for one tick under [`EnergyProfile::PhysicalV1`] (D-3).
    ///
    /// The legacy segment cost for the motion, plus hover power for the part
    /// of the tick not spent in that modelled motion: a multirotor that is
    /// airborne draws power whether or not it moves. Continuous with the
    /// legacy model -- a platform moving at cruise for the whole tick costs
    /// exactly the segment cost -- and a stationary one draws `hover_power_w`
    /// for the full tick. With no previous position (the first tick) the
    /// platform hovers for the whole tick.
    pub fn tick_cost_wh(&self, start: Option<[f64; 3]>, end: [f64; 3], dt_s: f64) -> f64 {
        let Some(start) = start else {
            return self.hover_power_w * dt_s.max(0.0) / 3600.0;
        };
        let horizontal = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
        let delta_z = end[2] - start[2];
        let motion_s = horizontal / self.cruise_speed_m_per_s.max(0.01)
            + delta_z.abs() / self.climb_speed_m_per_s.max(0.01);
        let hover_s = (dt_s - motion_s).max(0.0);
        self.segment_cost_wh(start, end) + self.hover_power_w * hover_s / 3600.0
    }

    /// Energy to move between two points: horizontal plus vertical.
    pub fn segment_cost_wh(&self, start: [f64; 3], end: [f64; 3]) -> f64 {
        let horizontal = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
        self.travel_cost_wh(horizontal) + self.climb_cost_wh(end[2] - start[2])
    }

    /// Energy to fly home, accounting for wind along the homeward track.
    ///
    /// Airspeed is fixed at cruise; the wind component along the track changes
    /// the groundspeed, and the groundspeed is floored at 25% of cruise so a
    /// strong headwind cannot produce an unbounded estimate.
    ///
    /// The wind is only applied when the horizontal distance exceeds 1e-6: with
    /// no horizontal separation there is no track to resolve the wind onto.
    pub fn return_home_cost_wh(
        &self,
        position: [f64; 3],
        home: [f64; 3],
        wind_vector_mps: Option<[f64; 3]>,
    ) -> f64 {
        let delta_x = home[0] - position[0];
        let delta_y = home[1] - position[1];
        let distance_m = (delta_x * delta_x + delta_y * delta_y).sqrt();
        let mut ground_speed = self.cruise_speed_m_per_s.max(0.01);
        if let Some(wind) = wind_vector_mps {
            if distance_m > 1e-6 {
                let track = [delta_x / distance_m, delta_y / distance_m];
                let along_track = wind[0] * track[0] + wind[1] * track[1];
                ground_speed =
                    (self.cruise_speed_m_per_s + along_track).max(0.25 * self.cruise_speed_m_per_s);
            }
        }
        let travel_wh = self.travel_power_w * (distance_m / ground_speed) / 3600.0;
        travel_wh + self.climb_cost_wh(home[2] - position[2])
    }

    /// Reserve energy needed to get home safely.
    ///
    /// The static reserve is a **floor**, so the dynamic estimate can only make
    /// the reserve more conservative, never less. A port that took the dynamic
    /// value alone would let a platform far from home hold back less than the
    /// configured minimum.
    pub fn dynamic_reserve_wh(
        &self,
        position: [f64; 3],
        home: [f64; 3],
        wind_vector_mps: Option<[f64; 3]>,
        safety_margin: f64,
    ) -> f64 {
        let dynamic = self.return_home_cost_wh(position, home, wind_vector_mps) * safety_margin;
        dynamic.max(self.capacity_wh * self.reserve_fraction)
    }

    /// The reference's default `safety_margin`.
    pub const DEFAULT_SAFETY_MARGIN: f64 = 1.25;
}

/// Remaining energy at a point in the mission.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BatteryState {
    pub remaining_wh: f64,
    pub timestamp_s: f64,
}

impl BatteryState {
    /// Draw `cost_wh`, clamped at empty.
    ///
    /// The timestamp is carried through unchanged, exactly as the reference
    /// does. That looks like an oversight -- a state that consumes energy but
    /// never advances its own clock -- but nothing reads the field, so
    /// "fixing" it here would be an unverifiable behaviour change in a port.
    /// Recorded rather than silently corrected.
    pub fn consume(&self, cost_wh: f64) -> BatteryState {
        BatteryState {
            remaining_wh: (self.remaining_wh - cost_wh).max(0.0),
            timestamp_s: self.timestamp_s,
        }
    }

    /// Remaining energy as a fraction of the given capacity.
    ///
    /// Spelled out with the capacity as an argument, because the reference's
    /// `fraction` property takes none and returns watt-hours.
    pub fn fraction_of(&self, capacity_wh: f64) -> f64 {
        if capacity_wh <= 0.0 {
            return 0.0;
        }
        self.remaining_wh / capacity_wh
    }
}
