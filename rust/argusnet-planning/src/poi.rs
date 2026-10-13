//! Point-of-interest lifecycle for the inspection phase: assignment, handoff,
//! dwell accumulation and completion.
//!
//! # Construction order is part of the contract
//!
//! The reference keeps its POIs in a `dict`, which since Python 3.7 iterates
//! in insertion order, and three behaviours depend on it:
//!
//! * `assign_nearest` sorts by `(-priority, distance)` with `list.sort`, which
//!   is **stable**, so two POIs at the same priority and the same distance
//!   keep construction order.
//! * `assign_energy_aware` scans in construction order and keeps the **first**
//!   strictly-better score.
//! * `check_completions` walks in construction order, so the ids come back in
//!   that order rather than in the order they finished.
//!
//! POIs are therefore held in a `Vec` and looked up by index. A `HashMap`
//! would pass every individual method and still hand back a different POI, and
//! a different completion order, on the cases that matter.

use std::collections::HashMap;

/// Travel power the reference's cost estimate assumes, in watts. Not read
/// from the battery model: `poi.py` hardcodes its own copy, and the two would
/// have to be changed together.
pub const TRAVEL_POWER_W: f64 = 350.0;

/// A point to inspect.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InspectionPoi {
    pub poi_id: String,
    pub name: String,
    pub position: [f64; 3],
    pub priority: i64,
    pub required_dwell_s: f64,
}

/// Where a POI is in its lifecycle.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PoiState {
    Pending,
    Active,
    Complete,
}

impl PoiState {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Active => "active",
            Self::Complete => "complete",
        }
    }
}

/// Runtime status of one POI.
#[derive(Debug, Clone, PartialEq)]
pub struct PoiStatus {
    pub poi_id: String,
    pub state: PoiState,
    pub assigned_drone_id: Option<String>,
    pub arrival_time_s: Option<f64>,
    pub completion_time_s: Option<f64>,
    pub dwell_accumulated_s: f64,
}

/// Per-platform context for energy-aware assignment.
#[derive(Debug, Clone, PartialEq)]
pub struct PoiAssignmentContext {
    pub drone_id: String,
    pub drone_pos: [f64; 3],
    pub battery_remaining_wh: f64,
    pub battery_capacity_wh: f64,
    pub cruise_speed_m_per_s: f64,
    pub battery_reserve_fraction: f64,
}

impl PoiAssignmentContext {
    /// The reference's defaults for the two optional fields.
    pub fn new(
        drone_id: &str,
        drone_pos: [f64; 3],
        battery_remaining_wh: f64,
        battery_capacity_wh: f64,
    ) -> Self {
        Self {
            drone_id: drone_id.to_string(),
            drone_pos,
            battery_remaining_wh,
            battery_capacity_wh,
            cruise_speed_m_per_s: 28.0,
            battery_reserve_fraction: 0.20,
        }
    }
}

/// `np.linalg.norm` of the horizontal difference: the plain square root of the
/// sum of squares. Not `hypot`, which rounds differently in the last bit and
/// so can move a dwell exactly at the 30 m boundary or an energy tie.
fn distance_xy(a: [f64; 3], b: [f64; 3]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    (dx * dx + dy * dy).sqrt()
}

/// Tracks every POI's lifecycle.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PoiManager {
    /// Construction order. See the module docs — this is not incidental.
    pois: Vec<InspectionPoi>,
    states: Vec<PoiState>,
    assigned_drone: Vec<Option<String>>,
    arrival_time_s: Vec<Option<f64>>,
    completion_time_s: Vec<Option<f64>>,
    /// The running dwell. Deliberately separate from the status: the
    /// reference keeps it in its own `_dwell_acc` dict and only copies it into
    /// `POIStatus.dwell_accumulated_s` at completion, so a status read while a
    /// platform is hovering reports **zero** dwell however long it has been
    /// there. Surprising, and reproduced — a caller watching that field for
    /// progress is watching the wrong one, and merging the two here would hide
    /// that rather than fix it.
    dwell_s: Vec<f64>,
    /// What the status reports: zero until completion.
    reported_dwell_s: Vec<f64>,
    effective_priorities: Vec<i64>,
    /// drone id -> POI index. Order here is not observable, so a map is fine.
    drone_assignment: HashMap<String, usize>,
}

impl PoiManager {
    pub fn new(pois: Vec<InspectionPoi>) -> Self {
        let count = pois.len();
        let effective_priorities = pois.iter().map(|p| p.priority).collect();
        Self {
            pois,
            states: vec![PoiState::Pending; count],
            assigned_drone: vec![None; count],
            arrival_time_s: vec![None; count],
            completion_time_s: vec![None; count],
            dwell_s: vec![0.0; count],
            reported_dwell_s: vec![0.0; count],
            effective_priorities,
            drone_assignment: HashMap::new(),
        }
    }

    fn index_of(&self, poi_id: &str) -> Option<usize> {
        self.pois.iter().position(|p| p.poi_id == poi_id)
    }

    pub fn statuses(&self) -> Vec<PoiStatus> {
        (0..self.pois.len())
            .map(|index| PoiStatus {
                poi_id: self.pois[index].poi_id.clone(),
                state: self.states[index],
                assigned_drone_id: self.assigned_drone[index].clone(),
                arrival_time_s: self.arrival_time_s[index],
                completion_time_s: self.completion_time_s[index],
                dwell_accumulated_s: self.reported_dwell_s[index],
            })
            .collect()
    }

    pub fn completed_count(&self) -> usize {
        self.states
            .iter()
            .filter(|s| **s == PoiState::Complete)
            .count()
    }

    pub fn total_count(&self) -> usize {
        self.pois.len()
    }

    pub fn all_complete(&self) -> bool {
        self.completed_count() == self.total_count()
    }

    pub fn assignment_for(&self, drone_id: &str) -> Option<&str> {
        self.drone_assignment
            .get(drone_id)
            .map(|index| self.pois[*index].poi_id.as_str())
    }

    /// The POIs still waiting for a drone, in construction order: id,
    /// position, rescored priority and required dwell. For a planner that
    /// allocates them all at once (P1); the reference path does not read it.
    pub fn pending_pois(&self) -> Vec<(String, [f64; 3], i64, f64)> {
        (0..self.pois.len())
            .filter(|&index| self.states[index] == PoiState::Pending)
            .map(|index| {
                let poi = &self.pois[index];
                (
                    poi.poi_id.clone(),
                    poi.position,
                    self.effective_priorities[index],
                    poi.required_dwell_s,
                )
            })
            .collect()
    }

    /// Give `poi_id` to `drone_id`, as a fleet-wide planner chose (P1).
    /// Refused, and nothing changes, unless the POI is pending and the drone
    /// holds no POI that is not yet complete.
    pub fn assign(&mut self, drone_id: &str, poi_id: &str) -> bool {
        let Some(index) = self.index_of(poi_id) else {
            return false;
        };
        if self.states[index] != PoiState::Pending || self.existing_assignment(drone_id).is_some() {
            return false;
        }
        self.take_assignment(drone_id, index);
        true
    }

    fn take_assignment(&mut self, drone_id: &str, index: usize) {
        self.drone_assignment.insert(drone_id.to_string(), index);
        self.dwell_s[index] = 0.0;
        self.states[index] = PoiState::Active;
        self.assigned_drone[index] = Some(drone_id.to_string());
    }

    /// An assignment the platform already holds, if it has not completed.
    fn existing_assignment(&self, drone_id: &str) -> Option<usize> {
        let index = *self.drone_assignment.get(drone_id)?;
        (self.states[index] != PoiState::Complete).then_some(index)
    }

    /// Assign the highest-priority nearby pending POI.
    ///
    /// **Reads the RAW priority, not the rescored one.** `rescore_from_map`
    /// only affects `assign_energy_aware`; that asymmetry is in the reference
    /// and is reproduced rather than tidied, since changing it would silently
    /// redirect every platform on the non-energy path.
    ///
    /// A platform below 25% battery gets nothing. The comparison is `< 0.25`,
    /// so exactly a quarter is still allowed.
    pub fn assign_nearest(
        &mut self,
        drone_id: &str,
        drone_pos: [f64; 3],
        battery_fraction: f64,
    ) -> Option<&InspectionPoi> {
        if let Some(index) = self.existing_assignment(drone_id) {
            return Some(&self.pois[index]);
        }
        if battery_fraction < 0.25 {
            return None;
        }
        let mut pending: Vec<usize> = (0..self.pois.len())
            .filter(|index| self.states[*index] == PoiState::Pending)
            .collect();
        if pending.is_empty() {
            return None;
        }
        // `sort_by` is stable, matching `list.sort`: equal keys keep
        // construction order, which is what decides between two POIs at the
        // same priority and the same distance.
        pending.sort_by(|a, b| {
            let key = |index: &usize| {
                (
                    -self.pois[*index].priority,
                    distance_xy(self.pois[*index].position, drone_pos),
                )
            };
            let (pa, da) = key(a);
            let (pb, db) = key(b);
            pa.cmp(&pb).then(da.total_cmp(&db))
        });
        let chosen = pending[0];
        self.take_assignment(drone_id, chosen);
        Some(&self.pois[chosen])
    }

    /// Energy to fly there, in watt-hours.
    fn energy_cost_wh(&self, from: [f64; 3], to: [f64; 3], cruise_speed_m_per_s: f64) -> f64 {
        let distance = distance_xy(from, to);
        let seconds = distance / cruise_speed_m_per_s.max(0.1);
        TRAVEL_POWER_W * seconds / 3600.0
    }

    /// Assign a POI within the platform's energy budget.
    ///
    /// Scores as `effective_priority * 10 - cost / usable`, so priority tiers
    /// (ten apart) dominate and energy only separates POIs within a tier —
    /// the cost term cannot exceed 0.8 given the affordability gate below.
    ///
    /// A POI costing more than **80%** of the usable energy is skipped. The
    /// factor is not a reserve on top of `battery_reserve_fraction`; it is a
    /// second, unnamed margin applied to what is left after the first.
    pub fn assign_energy_aware(
        &mut self,
        context: &PoiAssignmentContext,
    ) -> Option<&InspectionPoi> {
        if let Some(index) = self.existing_assignment(&context.drone_id) {
            return Some(&self.pois[index]);
        }
        let usable_wh = context.battery_remaining_wh
            - context.battery_capacity_wh * context.battery_reserve_fraction;
        if usable_wh <= 0.0 {
            return None;
        }

        let mut best: Option<usize> = None;
        let mut best_score = f64::NEG_INFINITY;
        for index in 0..self.pois.len() {
            if self.states[index] != PoiState::Pending {
                continue;
            }
            let cost_wh = self.energy_cost_wh(
                context.drone_pos,
                self.pois[index].position,
                context.cruise_speed_m_per_s,
            );
            if cost_wh > usable_wh * 0.8 {
                continue;
            }
            let score =
                self.effective_priorities[index] as f64 * 10.0 - cost_wh / usable_wh.max(0.01);
            // Strictly greater: the first best in construction order wins.
            if score > best_score {
                best_score = score;
                best = Some(index);
            }
        }
        let chosen = best?;
        self.take_assignment(&context.drone_id, chosen);
        Some(&self.pois[chosen])
    }

    /// Move an active POI from one platform to another.
    ///
    /// Silently replaces any assignment the receiving platform already held —
    /// the reference does not check, and the old POI is left `Active` with a
    /// stale `assigned_drone_id`. Reproduced rather than fixed; a caller that
    /// hands off onto a busy platform has a problem this would only hide.
    pub fn trigger_handoff(&mut self, from_drone_id: &str, to_drone_id: &str) -> bool {
        let Some(index) = self.drone_assignment.get(from_drone_id).copied() else {
            return false;
        };
        if self.states[index] != PoiState::Active {
            return false;
        }
        self.drone_assignment.remove(from_drone_id);
        self.drone_assignment.insert(to_drone_id.to_string(), index);
        self.assigned_drone[index] = Some(to_drone_id.to_string());
        true
    }

    /// Add a second platform to an already-active POI, so both credit the same
    /// dwell and it completes in half the time.
    ///
    /// Note the status keeps only ONE `assigned_drone_id`: the second platform
    /// appears in the assignment map but not in the status, so a reader of the
    /// statuses alone cannot tell a POI is being worked by two.
    pub fn request_team_assign(&mut self, poi_id: &str, second_drone_id: &str) -> bool {
        let Some(index) = self.index_of(poi_id) else {
            return false;
        };
        if self.states[index] != PoiState::Active {
            return false;
        }
        if self.drone_assignment.contains_key(second_drone_id) {
            return false;
        }
        self.drone_assignment
            .insert(second_drone_id.to_string(), index);
        true
    }

    /// Raise the priority of POIs in poorly-covered regions, by up to +3.
    ///
    /// `density_at` returns local coverage in `[0, 1]`; the reference swallows
    /// any exception from its map and falls back to a density of 1.0, so this
    /// takes an `Option` and treats `None` the same way.
    ///
    /// **The boost rounds half to even.** `(1 - density) * 3` lands exactly on
    /// `0.5` at a density of 5/6, and Python's `round()` sends that to 0
    /// rather than 1. Rust's `f64::round()` would go the other way and boost
    /// the POI, changing which one is chosen — the same trap as the route
    /// cache key in P4.7d.
    ///
    /// Only pending POIs are rescored.
    pub fn rescore_from_map<F>(&mut self, mut density_at: F)
    where
        F: FnMut([f64; 2]) -> Option<f64>,
    {
        for index in 0..self.pois.len() {
            if self.states[index] != PoiState::Pending {
                continue;
            }
            let position = self.pois[index].position;
            let density = density_at([position[0], position[1]]).unwrap_or(1.0);
            let boost = ((1.0 - density) * 3.0).round_ties_even() as i64;
            self.effective_priorities[index] = self.pois[index].priority + boost;
        }
    }

    /// Credit dwell time when the platform is within 30 m of its POI.
    /// The comparison is `<= 30.0`, so exactly thirty metres counts.
    pub fn accumulate_dwell(&mut self, drone_id: &str, drone_pos: [f64; 3], dt_s: f64) {
        let Some(index) = self.drone_assignment.get(drone_id).copied() else {
            return;
        };
        if distance_xy(self.pois[index].position, drone_pos) <= 30.0 {
            self.dwell_s[index] += dt_s;
        }
    }

    /// The POIs that became complete on this call, in construction order.
    ///
    /// Not in the order they finished: the reference walks its POI dict, so a
    /// POI constructed first is reported first even if it reached its dwell
    /// last. Completing releases **every** platform assigned to the POI, which
    /// is what makes team assignment work.
    ///
    /// The reference writes `arrival_time_s or timestamp_s`, where a stored
    /// arrival of exactly `0.0` would be replaced by the current time because
    /// zero is falsy. That is latent rather than live — nothing ever sets
    /// `arrival_time_s` on an active POI, so it is always `None` here — and it
    /// is reproduced as "None or zero becomes the timestamp" so a future
    /// writer cannot change behaviour by accident.
    pub fn check_completions(&mut self, timestamp_s: f64) -> Vec<String> {
        let mut newly_done = Vec::new();
        for index in 0..self.pois.len() {
            if self.states[index] != PoiState::Active {
                continue;
            }
            let dwell = self.dwell_s[index];
            if dwell < self.pois[index].required_dwell_s {
                continue;
            }
            let arrival = match self.arrival_time_s[index] {
                Some(value) if value != 0.0 => value,
                _ => timestamp_s,
            };
            self.states[index] = PoiState::Complete;
            self.arrival_time_s[index] = Some(arrival);
            self.completion_time_s[index] = Some(timestamp_s);
            self.reported_dwell_s[index] = dwell;
            self.drone_assignment.retain(|_, held| *held != index);
            newly_done.push(self.pois[index].poi_id.clone());
        }
        newly_done
    }
}
