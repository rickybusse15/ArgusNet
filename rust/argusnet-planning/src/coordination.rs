//! Multi-drone coordination: coordinator election, claimed-cell updates under
//! simulated RF latency, and scanning-phase formation offsets.
//!
//! # Three orderings and one lookup are part of the contract
//!
//! * `elect_coordinator` keeps the **first** strictly-better fraction, so ties
//!   go to the earliest entry in the platform list.
//! * `flush_messages` applies a drone's queued messages in **arrival order**,
//!   writing each into `claimed_cells` in turn, so the **last** message
//!   delivered on a step is the one that survives.
//! * Queues are never removed once created -- `flush_messages` writes the
//!   remainder back, empty list included -- so the set of queue keys grows
//!   monotonically and keeps insertion order. `pending_messages` is therefore
//!   an insertion-ordered `Vec` of queues rather than a `HashMap`.
//! * The latency that `update_claimed` honours is read from the **shared
//!   state**, not from the policy. The policy carries a `message_latency_steps`
//!   of its own and nothing in the reference consults it on this path; a port
//!   that "tidied" the two into one would change behaviour whenever the sim
//!   seeds the shared state with a different value.
//!
//! # Two quirks preserved deliberately
//!
//! * A drone **missing** from `battery_states` scores a fraction of `1.0` --
//!   full -- rather than being skipped, so an unknown platform outranks every
//!   known one that has burned any energy at all.
//! * Election divides by `max(capacity_wh, 1.0)`, not by the capacity. Below
//!   1 Wh of capacity the fraction stops being a fraction. This differs from
//!   [`BatteryState::fraction_of`], which guards at `<= 0.0` instead, so the
//!   two must not be folded together.

use crate::battery::BatteryState;
use std::collections::HashMap;

/// Which formation the scanning phase flies.
///
/// `Unrecognized` is not defensive padding: the reference matches its
/// `formation_mode` string against two names and falls through to an `else`
/// that emits a **zero** offset, so an unknown mode returns a fully-populated
/// map of zeros rather than the empty map `"none"` returns. Those two results
/// are not interchangeable -- a caller distinguishing "no formation" from "the
/// formation put me on the lead's waypoint" sees the difference.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FormationMode {
    /// `"none"`: no formation. `formation_offsets` returns an empty map.
    Off,
    /// `"line_abreast"`: side-by-side, perpendicular to the lead heading.
    LineAbreast,
    /// `"v_formation"`: trailing behind the lead, alternating left and right.
    VFormation,
    /// Any other string, which the reference silently treats as zero offsets.
    Unrecognized(String),
}

impl FormationMode {
    /// Map a reference `formation_mode` string onto a variant.
    pub fn from_name(name: &str) -> Self {
        match name {
            "none" => Self::Off,
            "line_abreast" => Self::LineAbreast,
            "v_formation" => Self::VFormation,
            other => Self::Unrecognized(other.to_string()),
        }
    }

    /// The string the reference would carry.
    pub fn as_str(&self) -> &str {
        match self {
            Self::Off => "none",
            Self::LineAbreast => "line_abreast",
            Self::VFormation => "v_formation",
            Self::Unrecognized(name) => name,
        }
    }
}

/// Configuration for multi-drone coordination behaviour.
#[derive(Debug, Clone, PartialEq)]
pub struct CoordinationPolicy {
    /// Whether the sim loop asks for an election at all. The manager itself
    /// never reads this -- the reference does not either -- but it is part of
    /// the policy the sim carries, so it is kept rather than dropped.
    pub elect_coordinator: bool,
    /// Simulated RF latency in sim steps, as configured. Note that
    /// [`CoordinationManager::update_claimed`] honours the shared state's copy,
    /// not this one.
    pub message_latency_steps: i64,
    pub formation_mode: FormationMode,
    /// Lateral spacing between adjacent drones in the formation.
    pub formation_spacing_m: f64,
}

impl Default for CoordinationPolicy {
    fn default() -> Self {
        Self {
            elect_coordinator: true,
            message_latency_steps: 0,
            formation_mode: FormationMode::Off,
            formation_spacing_m: 80.0,
        }
    }
}

/// One queued claimed-cell update awaiting delivery.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PendingMessage {
    pub cell: (i64, i64),
    pub deliver_at_step: i64,
}

/// Mutable coordination state shared across all drone planners each step.
///
/// Owned by the sim loop and passed into the manager's methods, exactly as the
/// reference arranges it, so the manager holds nothing but its policy.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct SharedMissionState {
    /// `drone_id -> (ci, cj)` grid cell the drone is heading toward.
    pub claimed_cells: HashMap<String, (i64, i64)>,
    /// `drone_id -> poi_id` tentative assignment (informational).
    pub poi_assignments: HashMap<String, String>,
    /// The currently elected coordinator.
    pub coordinator_id: Option<String>,
    /// Queued updates, one queue per drone, in the order the drones first
    /// queued anything. See the module docs on why this is not a map.
    pending: Vec<(String, Vec<PendingMessage>)>,
    /// The latency actually honoured by `update_claimed`.
    pub message_latency_steps: i64,
}

impl SharedMissionState {
    /// An empty state with the given latency.
    pub fn with_latency(message_latency_steps: i64) -> Self {
        Self {
            message_latency_steps,
            ..Self::default()
        }
    }

    /// The queue for one drone, empty if it has never queued anything.
    pub fn pending_for(&self, drone_id: &str) -> &[PendingMessage] {
        self.pending
            .iter()
            .find(|(id, _)| id == drone_id)
            .map(|(_, queue)| queue.as_slice())
            .unwrap_or(&[])
    }

    /// Every queue, in the order the drones first queued anything. Queues that
    /// have drained are still present and still empty, matching the reference.
    pub fn pending_queues(&self) -> impl Iterator<Item = (&str, &[PendingMessage])> {
        self.pending
            .iter()
            .map(|(id, queue)| (id.as_str(), queue.as_slice()))
    }

    fn queue_mut(&mut self, drone_id: &str) -> &mut Vec<PendingMessage> {
        if let Some(index) = self.pending.iter().position(|(id, _)| id == drone_id) {
            return &mut self.pending[index].1;
        }
        self.pending.push((drone_id.to_string(), Vec::new()));
        let last = self.pending.len() - 1;
        &mut self.pending[last].1
    }
}

/// Implements the coordination primitives.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CoordinationManager {
    pub policy: CoordinationPolicy,
}

impl CoordinationManager {
    pub fn new(policy: CoordinationPolicy) -> Self {
        Self { policy }
    }

    // ------------------------------------------------------------------
    // Coordinator election
    // ------------------------------------------------------------------

    /// The drone with the highest remaining battery fraction, or `None` when
    /// `drone_ids` is empty.
    ///
    /// Two behaviours here look like bugs and are reproduced rather than
    /// corrected, because a sim seeded before this port would diverge if
    /// either were changed:
    ///
    /// 1. A drone absent from `battery_states` scores `1.0`.
    /// 2. The running best starts at `-1.0` while the running id starts at
    ///    `drone_ids[0]`. When *every* fraction is below `-1.0` -- reachable
    ///    only from a directly-constructed negative `remaining_wh` -- no
    ///    candidate ever clears the bar and the first id wins regardless of
    ///    which drone actually has the most energy.
    pub fn elect_coordinator(
        &self,
        drone_ids: &[String],
        battery_states: &HashMap<String, BatteryState>,
        battery_capacity_wh: f64,
    ) -> Option<String> {
        let first = drone_ids.first()?;

        let mut best_id = first.as_str();
        let mut best_fraction = -1.0f64;
        for drone_id in drone_ids {
            let fraction = match battery_states.get(drone_id) {
                Some(state) => state.remaining_wh / battery_capacity_wh.max(1.0),
                None => 1.0,
            };
            if fraction > best_fraction {
                best_fraction = fraction;
                best_id = drone_id.as_str();
            }
        }
        Some(best_id.to_string())
    }

    // ------------------------------------------------------------------
    // Claimed-cell updates with optional RF latency
    // ------------------------------------------------------------------

    /// Record a claimed-cell update, respecting the shared state's latency.
    ///
    /// A latency of exactly zero applies the update immediately. Anything else
    /// -- including a negative latency, which the reference does not guard
    /// against -- queues it for [`Self::flush_messages`]. A negative latency
    /// therefore does not deliver early; it delivers at the next flush, since
    /// no message is ever applied outside one.
    pub fn update_claimed(
        &self,
        drone_id: &str,
        cell: Option<(i64, i64)>,
        shared: &mut SharedMissionState,
        current_step: i64,
    ) {
        let Some(cell) = cell else {
            return;
        };

        let latency = shared.message_latency_steps;
        if latency == 0 {
            shared.claimed_cells.insert(drone_id.to_string(), cell);
        } else {
            let deliver_at_step = current_step + latency;
            shared.queue_mut(drone_id).push(PendingMessage {
                cell,
                deliver_at_step,
            });
        }
    }

    /// Apply every queued message whose delivery step has been reached.
    ///
    /// Within one drone's queue the messages are applied in arrival order, so
    /// when several come due on the same step the last one wins. Messages that
    /// are not yet due keep their relative order.
    pub fn flush_messages(&self, shared: &mut SharedMissionState, current_step: i64) {
        let mut delivered: Vec<(String, (i64, i64))> = Vec::new();
        for (drone_id, queue) in shared.pending.iter_mut() {
            let mut remaining = Vec::with_capacity(queue.len());
            for message in queue.iter() {
                if message.deliver_at_step <= current_step {
                    delivered.push((drone_id.clone(), message.cell));
                } else {
                    remaining.push(*message);
                }
            }
            *queue = remaining;
        }
        for (drone_id, cell) in delivered {
            shared.claimed_cells.insert(drone_id, cell);
        }
    }

    // ------------------------------------------------------------------
    // Formation offsets
    // ------------------------------------------------------------------

    /// Per-drone XY offset to add to each drone's target waypoint, in the
    /// order the drones were given.
    ///
    /// Empty when the mode is [`FormationMode::Off`] or fewer than two drones
    /// are present. Index 0 is the lead and always takes a zero offset.
    ///
    /// The scalar is formed in full *before* it scales the unit vector, which
    /// is what the reference's `perp * (sign * rank * s)` does. Folding the
    /// factors into the vector one at a time -- `((perp * sign) * rank) * s` --
    /// is not the same number: over random headings and spacings the two
    /// spellings disagree in the last bit about a third of the time, because
    /// floating-point multiplication is not associative. (The `* 0.5` in the
    /// V arm is the exception: halving is exact, so its grouping is free.)
    pub fn formation_offsets(
        &self,
        drone_ids: &[String],
        lead_heading_rad: f64,
    ) -> Vec<(String, [f64; 2])> {
        if self.policy.formation_mode == FormationMode::Off || drone_ids.len() < 2 {
            return Vec::new();
        }

        let spacing = self.policy.formation_spacing_m;
        let cos_h = lead_heading_rad.cos();
        let sin_h = lead_heading_rad.sin();

        // Perpendicular (left) and along-track (forward) unit vectors in XY.
        let perp = [-sin_h, cos_h];
        let fwd = [cos_h, sin_h];

        let mut offsets: Vec<(String, [f64; 2])> = Vec::with_capacity(drone_ids.len());
        for (index, drone_id) in drone_ids.iter().enumerate() {
            let offset = if index == 0 {
                [0.0, 0.0]
            } else {
                // Alternate left/right: +1, -1, +2, -2, ...
                let sign = if index % 2 == 1 { 1.0 } else { -1.0 };
                // The reference spells this `(idx + 1) // 2`; on a usize that
                // is exactly `div_ceil(2)`, and both are exact integer ops.
                let rank = index.div_ceil(2) as f64;
                match &self.policy.formation_mode {
                    FormationMode::LineAbreast => {
                        let lateral = sign * rank * spacing;
                        [perp[0] * lateral, perp[1] * lateral]
                    }
                    FormationMode::VFormation => {
                        let back = rank * spacing * 0.5;
                        let lateral = sign * rank * spacing * 0.5;
                        [
                            -fwd[0] * back + perp[0] * lateral,
                            -fwd[1] * back + perp[1] * lateral,
                        ]
                    }
                    // `Off` is filtered above; `Unrecognized` is the
                    // reference's silent zero arm.
                    FormationMode::Off | FormationMode::Unrecognized(_) => [0.0, 0.0],
                }
            };
            // A repeated id overwrites in place and keeps its first position,
            // the way assigning twice into a Python dict does.
            match offsets.iter_mut().find(|(id, _)| id == drone_id) {
                Some(slot) => slot.1 = offset,
                None => offsets.push((drone_id.clone(), offset)),
            }
        }
        offsets
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| n.to_string()).collect()
    }

    fn battery(remaining_wh: f64) -> BatteryState {
        BatteryState {
            remaining_wh,
            timestamp_s: 0.0,
        }
    }

    fn states(entries: &[(&str, f64)]) -> HashMap<String, BatteryState> {
        entries
            .iter()
            .map(|(id, wh)| (id.to_string(), battery(*wh)))
            .collect()
    }

    #[test]
    fn election_on_empty_list_is_none() {
        let manager = CoordinationManager::default();
        assert_eq!(manager.elect_coordinator(&[], &HashMap::new(), 500.0), None);
    }

    #[test]
    fn election_picks_the_highest_fraction() {
        let manager = CoordinationManager::default();
        let elected = manager.elect_coordinator(
            &ids(&["a", "b", "c"]),
            &states(&[("a", 100.0), ("b", 400.0), ("c", 250.0)]),
            500.0,
        );
        assert_eq!(elected.as_deref(), Some("b"));
    }

    #[test]
    fn election_ties_go_to_the_earlier_id() {
        let manager = CoordinationManager::default();
        let elected = manager.elect_coordinator(
            &ids(&["b", "a"]),
            &states(&[("a", 300.0), ("b", 300.0)]),
            500.0,
        );
        assert_eq!(elected.as_deref(), Some("b"));
    }

    #[test]
    fn a_drone_with_no_battery_state_scores_full() {
        let manager = CoordinationManager::default();
        let elected = manager.elect_coordinator(
            &ids(&["known", "unknown"]),
            &states(&[("known", 499.0)]),
            500.0,
        );
        assert_eq!(elected.as_deref(), Some("unknown"));
    }

    #[test]
    fn capacity_below_one_wh_divides_by_one() {
        let manager = CoordinationManager::default();
        // Under a true fraction both would be 1.0 and the first would win; the
        // `max(capacity, 1.0)` divisor makes them 0.25 and 0.5 instead.
        let elected = manager.elect_coordinator(
            &ids(&["small", "large"]),
            &states(&[("small", 0.25), ("large", 0.5)]),
            0.5,
        );
        assert_eq!(elected.as_deref(), Some("large"));
    }

    #[test]
    fn every_fraction_below_minus_one_keeps_the_first_id() {
        let manager = CoordinationManager::default();
        // "b" holds more energy, yet neither clears the -1.0 starting bar.
        let elected = manager.elect_coordinator(
            &ids(&["a", "b"]),
            &states(&[("a", -5000.0), ("b", -1000.0)]),
            500.0,
        );
        assert_eq!(elected.as_deref(), Some("a"));
    }

    #[test]
    fn zero_latency_applies_immediately() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(0);
        manager.update_claimed("a", Some((3, 4)), &mut shared, 10);
        assert_eq!(shared.claimed_cells.get("a"), Some(&(3, 4)));
        assert!(shared.pending_for("a").is_empty());
    }

    #[test]
    fn a_none_cell_is_ignored_at_every_latency() {
        let manager = CoordinationManager::default();
        for latency in [0, 3] {
            let mut shared = SharedMissionState::with_latency(latency);
            manager.update_claimed("a", None, &mut shared, 10);
            assert!(shared.claimed_cells.is_empty());
            assert!(shared.pending_for("a").is_empty());
        }
    }

    #[test]
    fn a_queued_message_lands_only_once_its_step_arrives() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(2);
        manager.update_claimed("a", Some((1, 1)), &mut shared, 5);

        manager.flush_messages(&mut shared, 6);
        assert_eq!(shared.claimed_cells.get("a"), None);
        assert_eq!(shared.pending_for("a").len(), 1);

        manager.flush_messages(&mut shared, 7);
        assert_eq!(shared.claimed_cells.get("a"), Some(&(1, 1)));
        assert!(shared.pending_for("a").is_empty());
    }

    #[test]
    fn the_last_message_due_on_a_step_wins() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(2);
        manager.update_claimed("a", Some((1, 1)), &mut shared, 5);
        manager.update_claimed("a", Some((2, 2)), &mut shared, 5);
        manager.flush_messages(&mut shared, 7);
        assert_eq!(shared.claimed_cells.get("a"), Some(&(2, 2)));
    }

    #[test]
    fn a_drained_queue_survives_as_an_empty_queue() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(1);
        manager.update_claimed("a", Some((1, 1)), &mut shared, 0);
        manager.flush_messages(&mut shared, 5);
        let queues: Vec<&str> = shared.pending_queues().map(|(id, _)| id).collect();
        assert_eq!(queues, vec!["a"]);
    }

    #[test]
    fn queues_keep_the_order_the_drones_first_queued_in() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(4);
        manager.update_claimed("c", Some((0, 0)), &mut shared, 0);
        manager.update_claimed("a", Some((0, 0)), &mut shared, 0);
        manager.update_claimed("c", Some((1, 0)), &mut shared, 1);
        let queues: Vec<&str> = shared.pending_queues().map(|(id, _)| id).collect();
        assert_eq!(queues, vec!["c", "a"]);
    }

    #[test]
    fn a_negative_latency_delivers_at_the_next_flush_not_before() {
        let manager = CoordinationManager::default();
        let mut shared = SharedMissionState::with_latency(-3);
        manager.update_claimed("a", Some((7, 7)), &mut shared, 10);
        // Queued rather than applied, despite the delivery step being in the
        // past, because nothing outside a flush ever applies a message.
        assert!(shared.claimed_cells.is_empty());
        assert_eq!(shared.pending_for("a")[0].deliver_at_step, 7);
        manager.flush_messages(&mut shared, 10);
        assert_eq!(shared.claimed_cells.get("a"), Some(&(7, 7)));
    }

    #[test]
    fn formation_off_returns_nothing() {
        let manager = CoordinationManager::default();
        assert!(manager.formation_offsets(&ids(&["a", "b"]), 0.0).is_empty());
    }

    #[test]
    fn a_single_drone_returns_nothing() {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::LineAbreast,
            ..CoordinationPolicy::default()
        });
        assert!(manager.formation_offsets(&ids(&["a"]), 0.0).is_empty());
    }

    #[test]
    fn an_unrecognized_mode_returns_zeros_rather_than_nothing() {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::from_name("echelon"),
            ..CoordinationPolicy::default()
        });
        let offsets = manager.formation_offsets(&ids(&["a", "b"]), 0.0);
        assert_eq!(offsets.len(), 2);
        assert!(offsets.iter().all(|(_, xy)| *xy == [0.0, 0.0]));
    }

    #[test]
    fn line_abreast_alternates_left_then_right() {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::LineAbreast,
            formation_spacing_m: 100.0,
            ..CoordinationPolicy::default()
        });
        // Heading +X, so the left perpendicular is +Y.
        let offsets = manager.formation_offsets(&ids(&["a", "b", "c", "d"]), 0.0);
        assert_eq!(offsets[0].1, [0.0, 0.0]);
        assert!((offsets[1].1[1] - 100.0).abs() < 1e-9);
        assert!((offsets[2].1[1] + 100.0).abs() < 1e-9);
        assert!((offsets[3].1[1] - 200.0).abs() < 1e-9);
    }

    #[test]
    fn v_formation_trails_behind_the_lead() {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::VFormation,
            formation_spacing_m: 100.0,
            ..CoordinationPolicy::default()
        });
        let offsets = manager.formation_offsets(&ids(&["a", "b"]), 0.0);
        // Half a spacing back along -X, half a spacing left along +Y.
        assert!((offsets[1].1[0] + 50.0).abs() < 1e-9);
        assert!((offsets[1].1[1] - 50.0).abs() < 1e-9);
    }

    #[test]
    fn a_repeated_id_keeps_its_first_slot_and_takes_the_later_offset() {
        let manager = CoordinationManager::new(CoordinationPolicy {
            formation_mode: FormationMode::LineAbreast,
            formation_spacing_m: 100.0,
            ..CoordinationPolicy::default()
        });
        let offsets = manager.formation_offsets(&ids(&["a", "b", "a"]), 0.0);
        assert_eq!(offsets.len(), 2);
        assert_eq!(offsets[0].0, "a");
        // Index 2 is right-hand rank 1, so "a" ends at -100 on Y rather than 0.
        assert!((offsets[0].1[1] + 100.0).abs() < 1e-9);
    }
}
