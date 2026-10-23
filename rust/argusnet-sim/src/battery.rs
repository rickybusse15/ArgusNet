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

use argusnet_planning::battery::{BatteryModel, BatteryState, EnergyProfile};
use serde::{Deserialize, Serialize};

use crate::controller::ControllerStage;
use crate::simulation::{
    decode_stage_state, Simulation, StageHandler, StageStateError, StageViews, TickContext,
};
use crate::stage::{Stage, StageError, StageOutcome};

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

#[derive(Debug, Default)]
struct State {
    model: BatteryModel,
    states: BTreeMap<String, BatteryState>,
    /// `None` until the platform has been seen once. The distinction is the
    /// whole of "the first tick is free".
    previous_positions: BTreeMap<String, Option<[f64; 3]>>,
    homes: BTreeMap<String, [f64; 3]>,
    /// How a tick's energy is charged (D-3). Legacy unless set.
    profile: EnergyProfile,
    wind_mps: Option<[f64; 3]>,
    warned: BTreeSet<String>,
    events: Vec<BatteryEvent>,
    /// Under `flown_v1`, where each mobile platform ended the last committed
    /// tick, recorded at commit ([`BatteryStage::record_flown`]) and charged
    /// by the next run. Checkpointed: a resumed world does not hold it.
    flown: Option<Vec<(String, [f64; 3])>>,
}

impl BatteryStage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a platform with a full battery and no previous position.
    pub fn register(&mut self, node_id: impl Into<String>, home: Option<[f64; 3]>) {
        let node_id = node_id.into();
        let shared = &mut self.state;
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

    pub fn set_model(&mut self, model: BatteryModel) {
        self.state.model = model;
    }

    pub fn model(&self) -> BatteryModel {
        self.state.model
    }

    /// How each tick's energy is charged (D-3). [`EnergyProfile::Legacy`]
    /// charges motion only; [`EnergyProfile::PhysicalV1`] also charges hover
    /// for the part of the tick not spent moving.
    pub fn set_energy_profile(&mut self, profile: EnergyProfile) {
        self.state.profile = profile;
    }

    pub fn energy_profile(&self) -> EnergyProfile {
        self.state.profile
    }

    /// Wind for the return-home estimate, for the ticks that follow.
    pub fn set_wind_mps(&mut self, wind_mps: Option<[f64; 3]>) {
        self.state.wind_mps = wind_mps;
    }

    /// The wind the reserve is estimated against, if any.
    pub fn wind_mps(&self) -> Option<[f64; 3]> {
        self.state.wind_mps
    }

    /// Set a registered platform's remaining energy -- a scenario that starts
    /// partly charged, or a test that needs a drone near empty. Returns
    /// whether the platform was registered.
    pub fn set_remaining_wh(&mut self, node_id: &str, remaining_wh: f64) -> bool {
        match self.state.states.get_mut(node_id) {
            Some(state) => {
                state.remaining_wh = remaining_wh;
                true
            }
            None => false,
        }
    }

    pub fn remaining_wh(&self, node_id: &str) -> Option<f64> {
        self.state.states.get(node_id).map(|s| s.remaining_wh)
    }

    /// Remaining energy as a fraction of capacity, or
    /// [`NOT_APPLICABLE_FRACTION`] for a platform with no battery.
    pub fn fraction(&self, node_id: &str) -> f64 {
        let shared = &self.state;
        match shared.states.get(node_id) {
            Some(state) => state.fraction_of(shared.model.capacity_wh),
            None => NOT_APPLICABLE_FRACTION,
        }
    }

    /// Whether this platform has already warned. Latched for the run.
    pub fn has_warned(&self, node_id: &str) -> bool {
        self.state.warned.contains(node_id)
    }

    /// Take the events the last run produced, leaving none behind.
    /// Under `flown_v1`, record where each mobile platform ended the tick
    /// just committed -- after overrides, deconfliction and safety -- for the
    /// next tick to charge. Called by the simulation at commit. Does nothing
    /// under another profile.
    pub fn record_flown(&mut self, world: &crate::world::TickWorld) {
        if self.state.profile != EnergyProfile::FlownV1 {
            return;
        }
        self.state.flown = Some(
            world
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
                .collect(),
        );
    }

    pub fn take_events(&mut self) -> Vec<BatteryEvent> {
        std::mem::take(&mut self.state.events)
    }

    /// Drop every battery, so a reset does not carry the previous epoch's
    /// charge into a fresh run.
    pub fn clear(&mut self) {
        let shared = &mut self.state;
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
    state: State,
}

/// Stage 1's battery ledger, read through the views lent to a later stage
/// (SESSION_OWNERSHIP.md §2a): the one chained into a [`ControllerStage`], or
/// a bare [`BatteryStage`] installed as stage 1. `None` if stage 1 has
/// neither.
pub fn stage_one_battery(stages: StageViews<'_>) -> Option<&BatteryStage> {
    stages
        .get::<ControllerStage>(Stage::SampleControllersAndBattery)
        .and_then(ControllerStage::battery)
        .or_else(|| stages.get::<BatteryStage>(Stage::SampleControllersAndBattery))
}

impl BatteryStage {
    /// [`stage_one_battery`], between ticks.
    pub fn of(sim: &Simulation) -> Option<&BatteryStage> {
        if let Some(controllers) = sim.stage::<ControllerStage>(Stage::SampleControllersAndBattery)
        {
            return controllers.battery();
        }
        sim.stage::<BatteryStage>(Stage::SampleControllersAndBattery)
    }

    /// [`Self::of`], mutably: for seeding the fleet and taking events.
    pub fn of_mut(sim: &mut Simulation) -> Option<&mut BatteryStage> {
        if sim
            .stage::<ControllerStage>(Stage::SampleControllersAndBattery)
            .is_some()
        {
            return sim
                .stage_mut::<ControllerStage>(Stage::SampleControllersAndBattery)
                .and_then(ControllerStage::battery_mut);
        }
        sim.stage_mut::<BatteryStage>(Stage::SampleControllersAndBattery)
    }
}

/// What the battery stage checkpoints: the ledger, not the configuration.
///
/// The model, home positions and wind are set by the run driver when it
/// builds the scenario and are identical in a resumed run. Energy drawn, the
/// previous position the next draw is measured from, and the one-shot
/// reserve latch are *history*, and each decides a future tick: without the
/// ledger a resumed platform is full again, without the previous position
/// its first resumed tick is free, and without the latch it warns twice.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
struct BatteryCheckpoint {
    /// `(remaining_wh, timestamp_s)` per platform. A local DTO rather than a
    /// serde derive on `argusnet_planning::BatteryState`, so the planning
    /// crate's public types do not grow a persistence contract for this.
    ledger: BTreeMap<String, (f64, f64)>,
    previous_positions: BTreeMap<String, Option<[f64; 3]>>,
    warned: BTreeSet<String>,
    /// Absent for legacy, so a legacy checkpoint is unchanged (D-3).
    #[serde(default, skip_serializing_if = "EnergyProfile::is_legacy")]
    energy_profile: EnergyProfile,
    /// Under `flown_v1`, where each platform ended the checkpointed tick.
    /// Absent otherwise.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    flown: Option<Vec<(String, [f64; 3])>>,
}

impl StageHandler for BatteryStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let shared = &mut self.state;
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
        // Under `flown_v1`, where the platforms ended the last tick. Before
        // any tick has been committed, where they are now: the first tick
        // draws hover alone, and the second is measured from here.
        let flown = shared.flown.take();
        let sampled: Vec<(String, [f64; 3])> = flown.unwrap_or_else(|| {
            context
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
                .collect()
        });

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

            // Legacy: the segment since the previous position, so the first
            // tick is free. Physical (D-3): motion plus hover for the rest of
            // the tick, first tick included -- the platform is airborne.
            let cost_wh = match (shared.profile, previous) {
                (EnergyProfile::Legacy, None) => None,
                (EnergyProfile::Legacy, Some(previous)) => {
                    Some(model.segment_cost_wh(previous, position))
                }
                (EnergyProfile::PhysicalV1 | EnergyProfile::FlownV1, previous) => {
                    Some(model.tick_cost_wh(previous, position, context.dt_s))
                }
            };
            if let Some(cost_wh) = cost_wh {
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

        // Publish the fraction onto the fleet, mirroring the reference's
        // `replace(ns, battery_fraction=...)` pass -- and note it runs over
        // *every* platform, not only the ones sampled above, because a
        // platform with no entry in the ledger has to be marked as having no
        // battery rather than left at whatever it held last tick.
        //
        // `None` is this crate's spelling of the reference's `-1.0`: a fixed
        // ground station has no battery, which is a different claim from a
        // drone at zero percent. Keeping it typed means a consumer cannot read
        // the sentinel as a reading, which is what the wire's `optional`
        // encoding preserves across the gRPC boundary.
        let capacity_wh = model.capacity_wh;
        let fractions: Vec<Option<f64>> = context
            .world
            .platforms()
            .iter()
            .map(|platform| {
                shared
                    .states
                    .get(&platform.node_id)
                    .map(|state| state.fraction_of(capacity_wh))
            })
            .collect();
        for (platform, fraction) in context.world.platforms_mut().iter_mut().zip(fractions) {
            platform.battery_fraction = fraction;
        }

        // `Ran`, not `drew_any`: the fractions above are a world mutation that
        // happens on every tick with a non-empty ledger, including the first,
        // where nothing is drawn because there is no previous position to
        // measure against. Reporting `NoWork` for a tick that moved the world
        // is exactly the conflation `StageOutcome`'s documentation warns about.
        // "Nothing was drawn" is still observable -- through the ledger, which
        // is where it belongs -- and the empty-ledger early return above still
        // reports the genuinely idle case.
        let _ = drew_any;
        Ok(StageOutcome::Ran)
    }

    /// Drop the ledger's history but keep the fleet registered, full.
    ///
    /// Not [`BatteryStage::clear`]: a reset is a fresh run of the *same*
    /// scenario, so every platform the driver registered is still there, with
    /// a full battery, no previous position and no warning.
    fn reset(&mut self) {
        let shared = &mut self.state;
        let capacity_wh = shared.model.capacity_wh;
        for state in shared.states.values_mut() {
            *state = BatteryState {
                remaining_wh: capacity_wh,
                timestamp_s: 0.0,
            };
        }
        for previous in shared.previous_positions.values_mut() {
            *previous = None;
        }
        shared.flown = None;
        shared.warned.clear();
        shared.events.clear();
    }

    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let shared = &self.state;
        let state = BatteryCheckpoint {
            ledger: shared
                .states
                .iter()
                .map(|(id, s)| (id.clone(), (s.remaining_wh, s.timestamp_s)))
                .collect(),
            previous_positions: shared.previous_positions.clone(),
            warned: shared.warned.clone(),
            energy_profile: shared.profile,
            flown: shared.flown.clone(),
        };
        Some(serde_json::to_value(state).expect("battery checkpoint serialises"))
    }

    fn check_state(&self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: BatteryCheckpoint = decode_stage_state(state)?;
        let shared = &self.state;
        if decoded.energy_profile != shared.profile {
            return Err(StageStateError::Malformed {
                detail: format!(
                    "the checkpoint charged energy as {}, this run as {}",
                    decoded.energy_profile.as_str(),
                    shared.profile.as_str()
                ),
            });
        }
        // The ledger must describe the fleet this run registered. A platform
        // present on one side only is a checkpoint from a different scenario.
        if decoded.ledger.keys().ne(shared.states.keys()) {
            return Err(StageStateError::Malformed {
                detail: "battery ledger names a different fleet from this run's".into(),
            });
        }
        if let Some((id, _)) = decoded
            .ledger
            .iter()
            .find(|(_, (remaining, _))| !remaining.is_finite() || *remaining < 0.0)
        {
            return Err(StageStateError::Malformed {
                detail: format!("battery {id} has an impossible remaining energy"),
            });
        }
        Ok(())
    }

    fn restore_state(&mut self, state: &serde_json::Value) -> Result<(), StageStateError> {
        let decoded: BatteryCheckpoint = decode_stage_state(state)?;
        let shared = &mut self.state;
        shared.states = decoded
            .ledger
            .into_iter()
            .map(|(id, (remaining_wh, timestamp_s))| {
                (
                    id,
                    BatteryState {
                        remaining_wh,
                        timestamp_s,
                    },
                )
            })
            .collect();
        shared.previous_positions = decoded.previous_positions;
        shared.warned = decoded.warned;
        shared.flown = decoded.flown;
        shared.events.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::SimulationConfig;
    use argusnet_types::{NodeState, Vector3};

    fn simulation() -> Simulation {
        Simulation::bare(SimulationConfig {
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
            ..Default::default()
        }
    }

    /// Install the stage as stage 1.
    fn rig(platforms: Vec<NodeState>) -> Simulation {
        let mut sim = simulation();
        sim.load_world(platforms, Vec::new());
        sim.set_stage_handler(
            Stage::SampleControllersAndBattery,
            Box::new(BatteryStage::new()),
        )
        .expect("stage 1 is not a transaction boundary");
        sim
    }

    fn bat(sim: &mut Simulation) -> &mut BatteryStage {
        BatteryStage::of_mut(sim).expect("stage 1 is a battery")
    }

    fn ledger(sim: &Simulation) -> &BatteryStage {
        BatteryStage::of(sim).expect("stage 1 is a battery")
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
    // Publishing the fraction onto the fleet
    // -----------------------------------------------------------------

    /// The ledger is the stage's private state; the fleet is what a snapshot,
    /// a replay and the viewer's battery readout all read. A stage that keeps
    /// a perfect ledger and never publishes it leaves that readout empty, so
    /// this asserts the world itself, not `BatteryStage::fraction`.
    #[test]
    fn the_fleet_carries_the_published_fraction_not_just_the_ledger() {
        let mut sim = rig(vec![
            node("drone-1", true, 0.0),
            node("tower", false, 500.0),
        ]);
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();

        assert_eq!(
            sim.world()
                .platform("drone-1")
                .expect("drone is in the fleet")
                .battery_fraction,
            Some(1.0)
        );
        // A ground station has no battery. `None`, not `Some(0.0)` -- the
        // whole reason the wire field is `optional`.
        assert_eq!(
            sim.world()
                .platform("tower")
                .expect("tower is in the fleet")
                .battery_fraction,
            None
        );
    }

    /// A registered platform publishes whether or not it is mobile.
    ///
    /// Consumption is gated on `is_mobile` -- a tower does not fly, so it is
    /// never charged for a segment -- and it is tempting to gate publishing on
    /// the same flag. That is wrong, and this is the test that says so: a
    /// registered fixed platform has a battery in the ledger, so its fraction
    /// is a real reading and has to reach the fleet. A mutant that published
    /// only to mobile platforms survived the first round because every other
    /// test's fixed platform was unregistered, and `None` is what both the
    /// correct and the broken version leave behind.
    #[test]
    fn a_registered_fixed_platform_publishes_even_though_it_is_never_charged() {
        let mut sim = rig(vec![
            node("drone-1", true, 0.0),
            node("tower", false, 500.0),
        ]);
        bat(&mut sim).register("drone-1", None);
        bat(&mut sim).register("tower", None);
        sim.step().unwrap();

        assert_eq!(
            sim.world().platform("tower").unwrap().battery_fraction,
            Some(1.0),
            "a registered platform publishes; being fixed only means it is never charged"
        );

        // And it stays published, at full, however far the drone flies.
        fly(&mut sim, "drone-1", 400.0);
        assert_eq!(
            sim.world().platform("tower").unwrap().battery_fraction,
            Some(1.0)
        );
        assert!(
            sim.world().platform("drone-1").unwrap().battery_fraction < Some(1.0),
            "the drone did fly, so it must have been charged"
        );
    }

    /// Pins the published value to the ledger exactly. An implementation that
    /// published a stale fraction, or recomputed it from the wrong capacity,
    /// would still leave the field "plausibly populated" -- a monotonic
    /// decrease is not enough to catch that.
    #[test]
    fn the_published_fraction_tracks_the_ledger_exactly_as_energy_is_drawn() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();

        let capacity_wh = ledger(&sim).model().capacity_wh;
        let mut previous = 1.0;
        for _ in 0..3 {
            fly(&mut sim, "drone-1", 250.0);
            let expected = ledger(&sim).remaining_wh("drone-1").expect("registered") / capacity_wh;
            let published = sim
                .world()
                .platform("drone-1")
                .expect("drone is in the fleet")
                .battery_fraction
                .expect("a registered drone publishes a fraction");
            assert_eq!(
                published, expected,
                "the published fraction must be the ledger's, to the bit"
            );
            assert!(
                published < previous,
                "flying 250 m must cost something: {published} !< {previous}"
            );
            previous = published;
        }
    }

    /// Registering a battery mid-run has to reach the fleet on the next tick.
    /// Publishing only on the tick that *registers* would leave the field
    /// stale for the rest of the run, which a single-step test cannot see.
    #[test]
    fn a_platform_registered_mid_run_starts_publishing_on_the_next_tick() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();
        assert_eq!(
            sim.world().platform("drone-1").unwrap().battery_fraction,
            Some(1.0)
        );

        // Clearing the ledger empties it, so the stage takes its early return
        // and publishes nothing -- the last published value stands.
        bat(&mut sim).clear();
        sim.step().unwrap();
        assert_eq!(
            sim.world().platform("drone-1").unwrap().battery_fraction,
            Some(1.0),
            "an empty ledger leaves the fleet untouched rather than blanking it"
        );

        // Re-register and the fleet picks it up again.
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();
        assert_eq!(
            sim.world().platform("drone-1").unwrap().battery_fraction,
            Some(1.0)
        );
    }

    // -----------------------------------------------------------------
    // The first tick is free
    // -----------------------------------------------------------------

    #[test]
    fn the_first_tick_draws_nothing_because_there_is_no_previous_position() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        let full = ledger(&sim).remaining_wh("drone-1").unwrap();

        sim.step().unwrap();
        assert_eq!(
            ledger(&sim).remaining_wh("drone-1").unwrap(),
            full,
            "energy is charged for a segment, and the first tick has none"
        );

        // The second tick charges for the segment flown during it.
        fly(&mut sim, "drone-1", 100.0);
        assert!(ledger(&sim).remaining_wh("drone-1").unwrap() < full);
    }

    /// The position is recorded on the tick that charged nothing, so only the
    /// *first* tick is free. Recording it inside the `if` would make two free.
    #[test]
    fn only_the_first_tick_is_free() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        let full = ledger(&sim).remaining_wh("drone-1").unwrap();

        fly(&mut sim, "drone-1", 100.0); // tick 0: free, records x = 100
        assert_eq!(ledger(&sim).remaining_wh("drone-1").unwrap(), full);

        fly(&mut sim, "drone-1", 100.0); // tick 1: charged for 100 m
        let after_one = ledger(&sim).remaining_wh("drone-1").unwrap();
        assert!(after_one < full);

        fly(&mut sim, "drone-1", 100.0); // tick 2: charged for another 100 m
        let after_two = ledger(&sim).remaining_wh("drone-1").unwrap();
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        let full = ledger(&sim).remaining_wh("drone-1").unwrap();
        for _ in 0..50 {
            sim.step().unwrap();
        }
        assert_eq!(
            ledger(&sim).remaining_wh("drone-1").unwrap(),
            full,
            "a zero-length segment has zero duration and so costs nothing"
        );
        assert!(
            ledger(&sim).model().hover_power_w > 0.0,
            "the model does carry a hover power -- it is simply never charged"
        );
    }

    /// D-3's physical profile charges hover for elapsed time: a stationary
    /// platform draws `hover_power_w` for every tick, the first included.
    #[test]
    fn under_the_physical_profile_hovering_costs_hover_power() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        bat(&mut sim).set_energy_profile(EnergyProfile::PhysicalV1);
        let full = ledger(&sim).remaining_wh("drone-1").unwrap();
        for _ in 0..50 {
            sim.step().unwrap();
        }
        let hover_w = ledger(&sim).model().hover_power_w;
        let drawn = full - ledger(&sim).remaining_wh("drone-1").unwrap();
        // dt is 1 s in this rig: 50 ticks of hover.
        assert!(
            (drawn - 50.0 * hover_w / 3600.0).abs() < 1e-9,
            "drew {drawn} Wh"
        );
    }

    /// Continuous with the legacy model: motion that fills the tick costs the
    /// segment cost and no more; motion that fills half of it adds hover for
    /// the other half.
    #[test]
    fn the_physical_tick_cost_adds_hover_only_for_time_not_moving() {
        let model = BatteryModel::default();
        let start = [0.0, 0.0, 50.0];
        let dt = 1.0;
        let full = [model.cruise_speed_m_per_s * dt, 0.0, 50.0];
        assert_eq!(
            model.tick_cost_wh(Some(start), full, dt),
            model.segment_cost_wh(start, full)
        );
        let half = [model.cruise_speed_m_per_s * dt * 0.5, 0.0, 50.0];
        let expected = model.segment_cost_wh(start, half) + model.hover_power_w * 0.5 / 3600.0;
        assert!((model.tick_cost_wh(Some(start), half, dt) - expected).abs() < 1e-12);
        assert_eq!(
            model.tick_cost_wh(None, start, dt),
            model.hover_power_w * dt / 3600.0,
            "the first tick hovers"
        );
        // Motion longer than the tick is never discounted below the segment.
        let far = [model.cruise_speed_m_per_s * dt * 3.0, 0.0, 50.0];
        assert_eq!(
            model.tick_cost_wh(Some(start), far, dt),
            model.segment_cost_wh(start, far)
        );
    }

    /// A checkpoint is only restored under the profile it was charged with.
    #[test]
    fn a_checkpoint_is_refused_under_another_energy_profile() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        bat(&mut sim).set_energy_profile(EnergyProfile::PhysicalV1);
        sim.step().unwrap();
        let state = ledger(&sim).checkpoint_state().unwrap();
        assert_eq!(state["energy_profile"], "physical_v1");
        let mut legacy = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut legacy).register("drone-1", None);
        let error = ledger(&legacy).check_state(&state).unwrap_err();
        assert!(format!("{error:?}").contains("physical_v1"), "{error:?}");
        // A legacy checkpoint carries no profile field at all.
        let legacy_state = ledger(&legacy).checkpoint_state().unwrap();
        assert!(legacy_state.get("energy_profile").is_none());
    }

    /// Descending costs less than climbing the same distance: the branch in
    /// `climb_cost_wh` is `delta_z_m > 0`, so a descent draws
    /// `hover_power_w * 0.8` instead of `climb_power_w`.
    #[test]
    fn a_descent_costs_less_than_the_same_climb() {
        let cost_of = |dz: f64| {
            let mut sim = rig(vec![node("drone-1", true, 0.0)]);
            bat(&mut sim).register("drone-1", None);
            sim.step().unwrap();
            let before = ledger(&sim).remaining_wh("drone-1").unwrap();
            sim.world_mut().platform_mut("drone-1").unwrap().position.z += dz;
            sim.step().unwrap();
            before - ledger(&sim).remaining_wh("drone-1").unwrap()
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
        let mut sim = rig(vec![node("tower", false, 0.0)]);
        bat(&mut sim).register("tower", None);
        let full = ledger(&sim).remaining_wh("tower").unwrap();
        // It has to MOVE, or the test passes for the wrong reason: a
        // zero-length segment costs nothing whether or not it was skipped.
        for _ in 0..5 {
            fly(&mut sim, "tower", 1_000.0);
        }
        assert_eq!(
            ledger(&sim).remaining_wh("tower").unwrap(),
            full,
            "`if not ns.is_mobile` skips it however far it moved"
        );
    }

    #[test]
    fn an_unregistered_platform_reads_the_not_applicable_sentinel() {
        let mut sim = rig(vec![node("drone-1", true, 0.0), node("tower", false, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();

        assert_eq!(ledger(&sim).fraction("drone-1"), 1.0);
        // Not 0.0, and not 1.0: a ground station is not a drone at 0%.
        assert_eq!(ledger(&sim).fraction("tower"), NOT_APPLICABLE_FRACTION);
        assert_eq!(ledger(&sim).fraction("nobody"), NOT_APPLICABLE_FRACTION);
        assert!(ledger(&sim).remaining_wh("tower").is_none());
    }

    #[test]
    fn a_stage_with_no_registered_batteries_reports_no_work() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        sim.step().unwrap();
        let record = sim
            .trace()
            .records()
            .iter()
            .find(|r| r.stage == Stage::SampleControllersAndBattery)
            .expect("stage 1 ran");
        assert_eq!(record.outcome, StageOutcome::NoWork);
    }

    /// The first tick draws nothing -- there is no previous position to
    /// measure a segment against -- but it does publish the fleet's fractions,
    /// so it is not a tick that did nothing. The "nothing was drawn"
    /// distinction lives in the ledger, which is where it is observable.
    #[test]
    fn a_first_tick_publishes_fractions_without_drawing_energy() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        sim.step().unwrap();

        let after_first = ledger(&sim)
            .remaining_wh("drone-1")
            .expect("drone has a battery");
        assert_eq!(
            after_first,
            ledger(&sim).model().capacity_wh,
            "the first tick has no segment to charge for"
        );

        let outcomes: Vec<_> = sim
            .trace()
            .records()
            .iter()
            .filter(|r| r.stage == Stage::SampleControllersAndBattery)
            .map(|r| r.outcome.clone())
            .collect();
        assert_eq!(outcomes, vec![StageOutcome::Ran]);
    }

    // -----------------------------------------------------------------
    // The reserve
    // -----------------------------------------------------------------

    /// With a recorded home the reserve is the estimated cost of flying there,
    /// which for a distant platform is larger than the static floor.
    #[test]
    fn a_distant_platform_holds_back_more_than_the_static_reserve() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", Some([0.0, 0.0, 0.0]));
        let model = ledger(&sim).model();
        let floor = model.capacity_wh * model.reserve_fraction;

        // Fly a long way out, then drain until the reserve trips.
        let mut reserve = None;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 500.0);
            if let Some(BatteryEvent::ReserveReached { reserve_wh, .. }) =
                bat(&mut sim).take_events().first()
            {
                reserve = Some(*reserve_wh);
                break;
            }
        }
        let reserve = reserve.expect("the reserve should have tripped");
        assert!(ledger(&sim).has_warned("drone-1"));
        assert!(
            reserve > floor,
            "a platform far from home must hold back more than the static \
             floor: reserve {reserve} vs floor {floor}"
        );
    }

    #[test]
    fn with_no_recorded_home_the_reserve_is_the_static_floor() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        let model = ledger(&sim).model();
        let floor = model.capacity_wh * model.reserve_fraction;

        let mut tripped_at = None;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 200.0);
            for event in bat(&mut sim).take_events() {
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);

        let mut warnings = 0;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            warnings += bat(&mut sim).take_events().len();
        }
        assert_eq!(warnings, 1, "one warning for the whole run, not one a tick");
        assert!(ledger(&sim).has_warned("drone-1"));

        // A full recharge does not re-arm it, because nothing clears the set.
        bat(&mut sim).register("drone-1", None);
        assert!(ledger(&sim).has_warned("drone-1"));
        for _ in 0..5 {
            fly(&mut sim, "drone-1", 300.0);
            assert!(bat(&mut sim).take_events().is_empty());
        }
    }

    #[test]
    fn energy_never_goes_below_empty() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        for _ in 0..400 {
            fly(&mut sim, "drone-1", 5_000.0);
        }
        assert_eq!(ledger(&sim).remaining_wh("drone-1").unwrap(), 0.0);
        assert_eq!(ledger(&sim).fraction("drone-1"), 0.0);
    }

    // -----------------------------------------------------------------
    // Wind, and the ledger's lifecycle
    // -----------------------------------------------------------------

    /// A headwind makes flying home more expensive, so it raises the reserve
    /// and the platform warns sooner.
    #[test]
    fn a_headwind_raises_the_return_home_reserve() {
        let reserve_with = |wind: Option<[f64; 3]>| {
            let mut sim = rig(vec![node("drone-1", true, 0.0)]);
            bat(&mut sim).register("drone-1", Some([0.0, 0.0, 50.0]));
            bat(&mut sim).set_wind_mps(wind);
            for _ in 0..200 {
                fly(&mut sim, "drone-1", 200.0);
                if let Some(BatteryEvent::ReserveReached { reserve_wh, .. }) =
                    bat(&mut sim).take_events().first()
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        fly(&mut sim, "drone-1", 100.0);
        fly(&mut sim, "drone-1", 100.0);
        assert!(ledger(&sim).remaining_wh("drone-1").is_some());

        bat(&mut sim).clear();
        assert!(ledger(&sim).remaining_wh("drone-1").is_none());
        assert_eq!(ledger(&sim).fraction("drone-1"), NOT_APPLICABLE_FRACTION);
        assert!(!ledger(&sim).has_warned("drone-1"));
    }

    #[test]
    fn a_custom_model_changes_the_draw() {
        let drain = |capacity_wh: f64| {
            let mut sim = rig(vec![node("drone-1", true, 0.0)]);
            bat(&mut sim).set_model(BatteryModel {
                capacity_wh,
                ..BatteryModel::default()
            });
            bat(&mut sim).register("drone-1", None);
            fly(&mut sim, "drone-1", 1_000.0);
            fly(&mut sim, "drone-1", 1_000.0);
            ledger(&sim).fraction("drone-1")
        };
        // The same segment costs the same watt-hours, so a bigger tank loses a
        // smaller fraction of itself.
        assert!(drain(1_000.0) > drain(100.0));
    }

    #[test]
    fn events_are_taken_once() {
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        let mut seen = 0;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            let taken = bat(&mut sim).take_events();
            seen += taken.len();
            assert!(
                bat(&mut sim).take_events().is_empty(),
                "a second take returns none"
            );
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).set_model(BatteryModel {
            reserve_fraction: 0.0,
            ..BatteryModel::default()
        });
        bat(&mut sim).register("drone-1", None);

        let mut events = Vec::new();
        for _ in 0..400 {
            fly(&mut sim, "drone-1", 5_000.0);
            events.extend(bat(&mut sim).take_events());
            if !events.is_empty() {
                break;
            }
        }
        assert_eq!(ledger(&sim).remaining_wh("drone-1").unwrap(), 0.0);
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if ledger(&sim).has_warned("drone-1") {
                break;
            }
        }
        assert!(
            ledger(&sim).has_warned("drone-1"),
            "it should have warned once"
        );
        let _ = bat(&mut sim).take_events();

        bat(&mut sim).clear();
        assert!(
            !ledger(&sim).has_warned("drone-1"),
            "a cleared ledger must forget the latch, or a fresh epoch could \
             never warn"
        );

        // And it really does warn again on a fresh battery.
        bat(&mut sim).register("drone-1", None);
        let mut warned_again = false;
        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if !bat(&mut sim).take_events().is_empty() {
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
        let mut sim = rig(vec![node("drone-1", true, 0.0)]);
        bat(&mut sim).register("drone-1", None);

        for _ in 0..200 {
            fly(&mut sim, "drone-1", 300.0);
            if ledger(&sim).has_warned("drone-1") {
                break;
            }
        }
        assert!(ledger(&sim).has_warned("drone-1"));

        // Deliberately do NOT take it, then step again.
        fly(&mut sim, "drone-1", 300.0);
        assert!(
            bat(&mut sim).take_events().is_empty(),
            "the stage clears its events at the top of each run, so an \
             untaken event is gone rather than replayed a tick late"
        );
    }
}
