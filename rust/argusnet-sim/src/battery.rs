//! Stage 1's battery accounting.
//!
//! §4 names stage 1 *"sample nominal controllers and account battery **using
//! the prior positions**. The ordering relative to overrides is a
//! compatibility constraint, not an accident."* This is the battery half; the
//! controllers are a separate slice.
//!
//! The arithmetic is [`argusnet_planning::battery`], ported in P4.7a. What is
//! here is the **accounting loop** around it, which the reference keeps inline
//! in `run_simulation` and which carries four behaviours worth naming.
//!
//! # The first tick is free, and that is not a rounding error
//!
//! ```python
//! prev_pos = _battery_prev_positions[ns.node_id]
//! if prev_pos is not None:
//!     cost_wh = _battery_model.segment_cost_wh(prev_pos, ns.position)
//!     ...
//! _battery_prev_positions[ns.node_id] = np.array(ns.position, dtype=float)
//! ```
//!
//! Energy is charged for the **segment between the previous position and this
//! one**, so on the first tick there is no previous position and nothing is
//! drawn. A port that seeded `prev_pos` with the spawn position would be
//! indistinguishable on a moving drone and would silently charge a stationary
//! one for a segment of length zero — the same answer for the wrong reason,
//! until a scenario places a drone somewhere other than where it starts.
//!
//! Note also that the assignment is **outside** the `if`: the position is
//! recorded on every tick, including the one that charged nothing. Moving it
//! inside would make the first *two* ticks free.
//!
//! # Not every platform has a battery
//!
//! The loop skips a platform that is not mobile *or* not in the ledger, and
//! `battery_fraction` for anything unregistered is the sentinel **-1.0**, not
//! `0.0` and not `1.0`. A ground station is not a drone with a flat battery.
//!
//! # The reserve is dynamic, and the static fraction is a floor
//!
//! `dynamic_reserve_wh` estimates the energy to actually fly home from here,
//! into the wind, and takes `max` with `capacity * reserve_fraction` — so the
//! estimate can only ever make the reserve *more* conservative. With no
//! recorded home there is no estimate to make and the static floor is used
//! directly.
//!
//! # The low-battery warning latches per platform
//!
//! The reference guards on a `_battery_low_warned` set, so a drone that sits
//! below its reserve warns once rather than once per tick. Crossing back above
//! the reserve does **not** re-arm it, because nothing removes from that set.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use argusnet_planning::battery::{BatteryModel, BatteryState};
use serde::{Deserialize, Serialize};

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// `battery_fraction` for a platform with no battery in the ledger.
///
/// The reference's own comment: *"0.0 (empty) → 1.0 (full); -1.0 = not
/// applicable"*. A ground station is not a drone at 0%.
pub const NOT_APPLICABLE_FRACTION: f64 = -1.0;

/// Something the caller has to act on.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum BatteryEvent {
    /// A platform reached its return-home reserve, for the first time.
    ///
    /// Latched: emitted once per platform per run, matching the reference's
    /// `_battery_low_warned` set, which nothing ever removes from.
    ReserveReached {
        node_id: String,
        remaining_wh: f64,
        reserve_wh: f64,
        fraction: f64,
        at_s: f64,
    },
}

/// A handle onto the battery stage, held by whoever drives the tick.
///
/// Same shape as [`crate::mission::MissionLink`] and for the same reason: the
/// stage is owned by the `Simulation` once installed, and the caller still
/// needs to seed the fleet, read fractions and take events.
#[derive(Debug, Clone, Default)]
pub struct BatteryLink {
    inner: Arc<Mutex<Shared>>,
}

#[derive(Debug, Default)]
struct Shared {
    model: BatteryModel,
    states: BTreeMap<String, BatteryState>,
    /// `None` until the platform has been seen once. The distinction is the
    /// whole of "the first tick is free".
    previous_positions: BTreeMap<String, Option<[f64; 3]>>,
    homes: BTreeMap<String, [f64; 3]>,
    wind_mps: Option<[f64; 3]>,
    warned: BTreeSet<String>,
    events: Vec<BatteryEvent>,
}

impl BatteryLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Register a platform with a full battery and no previous position.
    pub fn register(&self, node_id: impl Into<String>, home: Option<[f64; 3]>) {
        let node_id = node_id.into();
        let mut shared = self.lock();
        let remaining_wh = shared.model.capacity_wh;
        shared.states.insert(
            node_id.clone(),
            BatteryState {
                remaining_wh,
                timestamp_s: 0.0,
            },
        );
        shared.previous_positions.insert(node_id.clone(), None);
        if let Some(home) = home {
            shared.homes.insert(node_id, home);
        }
    }

    pub fn set_model(&self, model: BatteryModel) {
        self.lock().model = model;
    }

    pub fn model(&self) -> BatteryModel {
        self.lock().model
    }

    /// Wind for the return-home estimate, for the ticks that follow.
    pub fn set_wind_mps(&self, wind_mps: Option<[f64; 3]>) {
        self.lock().wind_mps = wind_mps;
    }

    pub fn remaining_wh(&self, node_id: &str) -> Option<f64> {
        self.lock().states.get(node_id).map(|s| s.remaining_wh)
    }

    /// Remaining energy as a fraction of capacity, or
    /// [`NOT_APPLICABLE_FRACTION`] for a platform with no battery.
    pub fn fraction(&self, node_id: &str) -> f64 {
        let shared = self.lock();
        match shared.states.get(node_id) {
            Some(state) => state.fraction_of(shared.model.capacity_wh),
            None => NOT_APPLICABLE_FRACTION,
        }
    }

    /// Whether this platform has already warned. Latched for the run.
    pub fn has_warned(&self, node_id: &str) -> bool {
        self.lock().warned.contains(node_id)
    }

    /// Take the events the last run produced, leaving none behind.
    pub fn take_events(&self) -> Vec<BatteryEvent> {
        std::mem::take(&mut self.lock().events)
    }

    /// Drop every battery, so a reset does not carry the previous epoch's
    /// charge into a fresh run.
    pub fn clear(&self) {
        let mut shared = self.lock();
        shared.states.clear();
        shared.previous_positions.clear();
        shared.homes.clear();
        shared.warned.clear();
        shared.events.clear();
    }
}

/// [`Stage::SampleControllersAndBattery`]'s battery accounting.
///
/// [`Stage::SampleControllersAndBattery`]: crate::stage::Stage::SampleControllersAndBattery
#[derive(Debug, Default)]
pub struct BatteryStage {
    link: BatteryLink,
}

impl BatteryStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> BatteryLink {
        self.link.clone()
    }
}

impl StageHandler for BatteryStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();
        shared.events.clear();
        if shared.states.is_empty() {
            return Ok(StageOutcome::NoWork);
        }

        let model = shared.model;
        let wind = shared.wind_mps;
        let timestamp_s = context.timestamp_s;
        let mut drew_any = false;

        // Declaration order, which §4 asks for: several reference stages break
        // ties by position in this list.
        let sampled: Vec<(String, [f64; 3])> = context
            .world
            .platforms()
            .iter()
            .filter(|platform| platform.is_mobile)
            .map(|platform| {
                (
                    platform.node_id.clone(),
                    [
                        platform.position.x,
                        platform.position.y,
                        platform.position.z,
                    ],
                )
            })
            .collect();

        for (node_id, position) in sampled {
            // `if not ns.is_mobile or ns.node_id not in _battery_states`.
            if !shared.states.contains_key(&node_id) {
                continue;
            }
            let previous = shared
                .previous_positions
                .get(&node_id)
                .copied()
                .unwrap_or(None);

            if let Some(previous) = previous {
                let cost_wh = model.segment_cost_wh(previous, position);
                let drawn = shared
                    .states
                    .get(&node_id)
                    .map(|state| state.consume(cost_wh))
                    .ok_or_else(|| {
                        StageError::Invariant(format!("battery for {node_id} vanished mid-tick"))
                    })?;
                shared.states.insert(node_id.clone(), drawn);
                drew_any = true;

                let remaining_wh = drawn.remaining_wh;
                let fraction = drawn.fraction_of(model.capacity_wh);
                // The static fraction is a floor, never a replacement: with no
                // recorded home there is no estimate to make.
                let reserve_wh = match shared.homes.get(&node_id) {
                    Some(home) => model.dynamic_reserve_wh(
                        position,
                        *home,
                        wind,
                        BatteryModel::DEFAULT_SAFETY_MARGIN,
                    ),
                    None => model.capacity_wh * model.reserve_fraction,
                };
                if remaining_wh <= reserve_wh && !shared.warned.contains(&node_id) {
                    shared.warned.insert(node_id.clone());
                    shared.events.push(BatteryEvent::ReserveReached {
                        node_id: node_id.clone(),
                        remaining_wh,
                        reserve_wh,
                        fraction,
                        at_s: timestamp_s,
                    });
                    context.note(format!(
                        "battery {node_id} at {:.1}% reached return-home reserve {reserve_wh:.1} Wh",
                        fraction * 100.0
                    ));
                }
            }

            // Outside the `if`, exactly as the reference has it: the position
            // is recorded on every tick, including the one that charged
            // nothing. Moving this inside would make the first two ticks free.
            shared.previous_positions.insert(node_id, Some(position));
        }

        Ok(if drew_any {
            StageOutcome::Ran
        } else {
            // Every registered platform was on its first tick, so nothing was
            // drawn -- distinct from having no batteries at all, and both are
            // distinct from a stage that never ran.
            StageOutcome::NoWork
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use crate::simulation::Simulation;
    use crate::stage::Stage;
    use argusnet_types::{NodeState, Vector3};

    fn simulation() -> Simulation {
        Simulation::new(SimulationConfig {
            seed: 1,
            dt_s: 1.0,
            steps: Some(2_000),
            run_id: "battery".to_string(),
        })
        .expect("config is valid")
    }

    fn node(id: &str, mobile: bool, x: f64) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, 0.0, 50.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: mobile,
            timestamp_s: 0.0,
            health: 1.0,
        }
    }

    /// Install the stage and return the handle plus the simulation.
    fn rig(platforms: Vec<NodeState>) -> (Simulation, BatteryLink) {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        let stage = BatteryStage::new();
        let link = stage.link();
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(stage))
            .expect("stage 1 is not a transaction boundary");
        (sim, link)
    }

    /// Move a platform by `dx` metres along x, then step.
    fn fly(sim: &mut Simulation, node_id: &str, dx: f64) {
        sim.world_mut()
            .platform_mut(node_id)
            .expect("platform exists")
            .position
            .x += dx;
        sim.step().expect("no stage fails");
    }

    // -----------------------------------------------------------------
    // The first tick is free
    // -----------------------------------------------------------------

    #[test]
    fn the_first_tick_draws_nothing_because_there_is_no_previous_position() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        let full = link.remaining_wh("drone-1").unwrap();

        sim.step().unwrap();
        assert_eq!(
            link.remaining_wh("drone-1").unwrap(),
            full,
            "energy is charged for a segment, and the first tick has none"
        );

        // The second tick charges for the segment flown during it.
        fly(&mut sim, "drone-1", 100.0);
        assert!(link.remaining_wh("drone-1").unwrap() < full);
    }

    /// The position is recorded on the tick that charged nothing, so only the
    /// *first* tick is free. Recording it inside the `if` would make two free.
    #[test]
    fn only_the_first_tick_is_free() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        let full = link.remaining_wh("drone-1").unwrap();

        fly(&mut sim, "drone-1", 100.0); // tick 0: free, records x = 100
        assert_eq!(link.remaining_wh("drone-1").unwrap(), full);

        fly(&mut sim, "drone-1", 100.0); // tick 1: charged for 100 m
        let after_one = link.remaining_wh("drone-1").unwrap();
        assert!(after_one < full);

        fly(&mut sim, "drone-1", 100.0); // tick 2: charged for another 100 m
        let after_two = link.remaining_wh("drone-1").unwrap();
        let first_segment = full - after_one;
        let second_segment = after_one - after_two;
        assert!(
            (first_segment - second_segment).abs() < 1e-12,
            "equal segments must cost equally: {first_segment} vs {second_segment}"
        );
    }

    /// A hovering platform consumes **nothing**, and that is the reference's
    /// behaviour rather than an oversight in the port.
    ///
    /// `segment_cost_wh` is `travel_cost_wh(horizontal) +
    /// climb_cost_wh(dz)`, and both derive their energy from a *duration*
    /// computed from the distance. A zero-length segment has zero duration and
    /// therefore zero cost. There is no hover term anywhere in the per-tick
    /// accounting: `hover_power_w` is in the model, but it is used only as the
    /// **descent** power inside `climb_cost_wh`.
    ///
    /// So a drone that stops moving has unlimited endurance in this
    /// simulation. Pinned rather than corrected -- charging hover would be a
    /// behaviour change that every existing golden run would have to absorb,
    /// and it belongs in a decision of its own.
    #[test]
    fn a_hovering_platform_consumes_nothing_because_there_is_no_hover_term() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        let full = link.remaining_wh("drone-1").unwrap();
        for _ in 0..50 {
            sim.step().unwrap();
        }
        assert_eq!(
            link.remaining_wh("drone-1").unwrap(),
            full,
            "a zero-length segment has zero duration and so costs nothing"
        );
        assert!(
            link.model().hover_power_w > 0.0,
            "the model does carry a hover power -- it is simply never charged"
        );
    }

    /// Descending costs less than climbing the same distance: the branch in
    /// `climb_cost_wh` is `delta_z_m > 0`, so a descent draws
    /// `hover_power_w * 0.8` instead of `climb_power_w`.
    #[test]
    fn a_descent_costs_less_than_the_same_climb() {
        let cost_of = |dz: f64| {
            let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
            link.register("drone-1", None);
            sim.step().unwrap();
            let before = link.remaining_wh("drone-1").unwrap();
            sim.world_mut().platform_mut("drone-1").unwrap().position.z += dz;
            sim.step().unwrap();
            before - link.remaining_wh("drone-1").unwrap()
        };
        let climb = cost_of(100.0);
        let descent = cost_of(-100.0);
        assert!(
            climb > descent,
            "climb {climb} should exceed descent {descent}"
        );
        assert!(descent > 0.0);
    }

    // -----------------------------------------------------------------
    // Who has a battery
    // -----------------------------------------------------------------

    #[test]
    fn a_fixed_platform_is_never_charged_even_if_registered() {
        let (mut sim, link) = rig(vec![node("tower", false, 0.0)]);
        link.register("tower", None);
        let full = link.remaining_wh("tower").unwrap();
        // It has to MOVE, or the test passes for the wrong reason: a
        // zero-length segment costs nothing whether or not it was skipped.
        for _ in 0..5 {
            fly(&mut sim, "tower", 1_000.0);
        }
        assert_eq!(
            link.remaining_wh("tower").unwrap(),
            full,
            "`if not ns.is_mobile` skips it however far it moved"
        );
    }

    #[test]
    fn an_unregistered_platform_reads_the_not_applicable_sentinel() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0), node("tower", false, 0.0)]);
        link.register("drone-1", None);
        sim.step().unwrap();

        assert_eq!(link.fraction("drone-1"), 1.0);
        // Not 0.0, and not 1.0: a ground station is not a drone at 0%.
        assert_eq!(link.fraction("tower"), NOT_APPLICABLE_FRACTION);
        assert_eq!(link.fraction("nobody"), NOT_APPLICABLE_FRACTION);
        assert!(link.remaining_wh("tower").is_none());
    }

    #[test]
    fn a_stage_with_no_registered_batteries_reports_no_work() {
        let (mut sim, _link) = rig(vec![node("drone-1", true, 0.0)]);
        sim.step().unwrap();
        let record = sim
            .trace()
            .records()
            .iter()
            .find(|r| r.stage == Stage::SampleControllersAndBattery)
            .expect("stage 1 ran");
        assert_eq!(record.outcome, StageOutcome::NoWork);
    }

    /// A tick where every registered platform is on its first tick draws
    /// nothing, and says so -- distinct from having no batteries at all.
    #[test]
    fn a_first_tick_reports_no_work_and_the_second_reports_ran() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        sim.step().unwrap();
        sim.step().unwrap();
        let outcomes: Vec<_> = sim
            .trace()
            .records()
            .iter()
            .filter(|r| r.stage == Stage::SampleControllersAndBattery)
            .map(|r| r.outcome.clone())
            .collect();
        assert_eq!(outcomes, vec![StageOutcome::NoWork, StageOutcome::Ran]);
    }

    // -----------------------------------------------------------------
    // The reserve
    // -----------------------------------------------------------------

    /// With a recorded home the reserve is the estimated cost of flying there,
    /// which for a distant platform is larger than the static floor.
    #[test]
    fn a_distant_platform_holds_back_more_than_the_static_reserve() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", Some([0.0, 0.0, 0.0]));
        let model = link.model();
        let floor = model.capacity_wh * model.reserve_fraction;

        // Fly a long way out, then drain until the reserve trips.
        let mut reserve = None;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 500.0);
            if let Some(BatteryEvent::ReserveReached { reserve_wh, .. }) =
                link.take_events().first()
            {
                reserve = Some(*reserve_wh);
                break;
            }
        }
        let reserve = reserve.expect("the reserve should have tripped");
        assert!(link.has_warned("drone-1"));
        assert!(
            reserve > floor,
            "a platform far from home must hold back more than the static \
             floor: reserve {reserve} vs floor {floor}"
        );
    }

    #[test]
    fn with_no_recorded_home_the_reserve_is_the_static_floor() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        let model = link.model();
        let floor = model.capacity_wh * model.reserve_fraction;

        let mut tripped_at = None;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 200.0);
            for event in link.take_events() {
                let BatteryEvent::ReserveReached {
                    reserve_wh,
                    remaining_wh,
                    ..
                } = event;
                assert!(
                    (reserve_wh - floor).abs() < 1e-12,
                    "no home means the static floor, got {reserve_wh} want {floor}"
                );
                tripped_at = Some(remaining_wh);
            }
            if tripped_at.is_some() {
                break;
            }
        }
        assert!(tripped_at.is_some(), "the reserve should have tripped");
    }

    /// The warning latches: a platform below its reserve warns once, not once
    /// per tick, and climbing back above does not re-arm it.
    #[test]
    fn the_reserve_warning_is_latched_for_the_run() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);

        let mut warnings = 0;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            warnings += link.take_events().len();
        }
        assert_eq!(warnings, 1, "one warning for the whole run, not one a tick");
        assert!(link.has_warned("drone-1"));

        // A full recharge does not re-arm it, because nothing clears the set.
        link.register("drone-1", None);
        assert!(link.has_warned("drone-1"));
        for _ in 0..5 {
            fly(&mut sim, "drone-1", 300.0);
            assert!(link.take_events().is_empty());
        }
    }

    #[test]
    fn energy_never_goes_below_empty() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        for _ in 0..400 {
            fly(&mut sim, "drone-1", 5_000.0);
        }
        assert_eq!(link.remaining_wh("drone-1").unwrap(), 0.0);
        assert_eq!(link.fraction("drone-1"), 0.0);
    }

    // -----------------------------------------------------------------
    // Wind, and the ledger's lifecycle
    // -----------------------------------------------------------------

    /// A headwind makes flying home more expensive, so it raises the reserve
    /// and the platform warns sooner.
    #[test]
    fn a_headwind_raises_the_return_home_reserve() {
        let reserve_with = |wind: Option<[f64; 3]>| {
            let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
            link.register("drone-1", Some([0.0, 0.0, 50.0]));
            link.set_wind_mps(wind);
            for _ in 0..200 {
                fly(&mut sim, "drone-1", 200.0);
                if let Some(BatteryEvent::ReserveReached { reserve_wh, .. }) =
                    link.take_events().first()
                {
                    return *reserve_wh;
                }
            }
            panic!("the reserve never tripped");
        };
        // The platform flies out along +x, so home is along -x; a wind blowing
        // towards +x is a headwind on the way back.
        let calm = reserve_with(None);
        let headwind = reserve_with(Some([6.0, 0.0, 0.0]));
        assert!(
            headwind > calm,
            "headwind reserve {headwind} should exceed calm {calm}"
        );
    }

    #[test]
    fn clearing_the_ledger_drops_every_battery() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        fly(&mut sim, "drone-1", 100.0);
        fly(&mut sim, "drone-1", 100.0);
        assert!(link.remaining_wh("drone-1").is_some());

        link.clear();
        assert!(link.remaining_wh("drone-1").is_none());
        assert_eq!(link.fraction("drone-1"), NOT_APPLICABLE_FRACTION);
        assert!(!link.has_warned("drone-1"));
    }

    #[test]
    fn a_custom_model_changes_the_draw() {
        let drain = |capacity_wh: f64| {
            let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
            link.set_model(BatteryModel {
                capacity_wh,
                ..BatteryModel::default()
            });
            link.register("drone-1", None);
            fly(&mut sim, "drone-1", 1_000.0);
            fly(&mut sim, "drone-1", 1_000.0);
            link.fraction("drone-1")
        };
        // The same segment costs the same watt-hours, so a bigger tank loses a
        // smaller fraction of itself.
        assert!(drain(1_000.0) > drain(100.0));
    }

    #[test]
    fn events_are_taken_once() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        let mut seen = 0;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            let taken = link.take_events();
            seen += taken.len();
            assert!(link.take_events().is_empty(), "a second take returns none");
        }
        assert_eq!(seen, 1);
    }

    /// The reserve comparison is `remaining <= reserve`, not `<`.
    ///
    /// The two differ only when the remaining energy lands exactly on the
    /// reserve, which needs constructing: with `reserve_fraction = 0.0` and no
    /// recorded home the reserve is exactly `0.0`, and `consume` clamps a
    /// drained battery to exactly `0.0`. So an empty battery is at its reserve
    /// and must warn.
    #[test]
    fn a_battery_exactly_at_its_reserve_warns() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.set_model(BatteryModel {
            reserve_fraction: 0.0,
            ..BatteryModel::default()
        });
        link.register("drone-1", None);

        let mut events = Vec::new();
        for _ in 0..400 {
            fly(&mut sim, "drone-1", 5_000.0);
            events.extend(link.take_events());
            if !events.is_empty() {
                break;
            }
        }
        assert_eq!(link.remaining_wh("drone-1").unwrap(), 0.0);
        let Some(BatteryEvent::ReserveReached {
            remaining_wh,
            reserve_wh,
            ..
        }) = events.first()
        else {
            panic!("an empty battery sitting exactly on a zero reserve must warn");
        };
        assert_eq!(*remaining_wh, 0.0);
        assert_eq!(*reserve_wh, 0.0);
    }

    /// `clear` drops the warning latch too, so a cleared ledger can warn again.
    #[test]
    fn clearing_the_ledger_re_arms_the_warning() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if link.has_warned("drone-1") {
                break;
            }
        }
        assert!(link.has_warned("drone-1"), "it should have warned once");
        let _ = link.take_events();

        link.clear();
        assert!(
            !link.has_warned("drone-1"),
            "a cleared ledger must forget the latch, or a fresh epoch could \
             never warn"
        );

        // And it really does warn again on a fresh battery.
        link.register("drone-1", None);
        let mut warned_again = false;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if !link.take_events().is_empty() {
                warned_again = true;
                break;
            }
        }
        assert!(warned_again);
    }

    /// An event nobody took is dropped at the start of the next tick, rather
    /// than accumulating. A caller that polls late gets the current tick's
    /// events, never a backlog dated to an earlier one.
    #[test]
    fn an_untaken_event_does_not_survive_into_the_next_tick() {
        let (mut sim, link) = rig(vec![node("drone-1", true, 0.0)]);
        link.register("drone-1", None);

        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if link.has_warned("drone-1") {
                break;
            }
        }
        assert!(link.has_warned("drone-1"));

        // Deliberately do NOT take it, then step again.
        fly(&mut sim, "drone-1", 300.0);
        assert!(
            link.take_events().is_empty(),
            "the stage clears its events at the top of each run, so an \
             untaken event is gone rather than replayed a tick late"
        );
    }
}
