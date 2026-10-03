//! Stage 9 in full: the mission machine, and the work its phases do.
//!
//! P4.8a separated *deciding* the phase ([`MissionState::advance`]) from the
//! side effects the reference performs inline -- rescoring POIs, assigning
//! them, redirecting drones, routing everyone home. The machine emits events
//! and leaves the work to "the caller". This is that caller.
//!
//! [`MissionState::advance`]: crate::mission::MissionState::advance
//!
//! # One tick, in the reference's order
//!
//! `run_simulation`'s `elif` chain tests the phase as it stood at the **top**
//! of the tick. In the inspecting branch it rescores (once per run), assigns,
//! redirects, hands off, credits dwell, team-assigns and completes, and only
//! *then* asks `all_complete` to decide on egress. So this stage:
//!
//! 1. reads the phase the machine settled on last tick;
//! 2. if that is `Inspecting`, does the whole inspecting body;
//! 3. builds the machine's inputs from **this** tick's stage 8 results and
//!    the POI manager as step 2 left it;
//! 4. advances the machine, and turns its egress events into overrides.
//!
//! The overrides are flown from the next tick's stage 2, which is where the
//! reference's `_poi_trajectories` / `_egress_trajectories` take effect.
//!
//! # The rescore is the driver's, not the event's
//!
//! The machine emits `RescorePois` on the tick it *enters* `Inspecting`. The
//! reference rescores at the top of the first tick *spent* inspecting --
//! after that tick's coverage update, which can change the density a POI
//! reads. So the driver keeps its own latch and rescores at step 2, and the
//! event is informational.

use std::collections::BTreeMap;

use argusnet_planning::frontier::{find_gap_cells, CoverageView};
use argusnet_planning::poi::{
    InspectionPoi, PoiAssignmentContext, PoiManager, PoiState, PoiStatus,
};

use crate::battery::{stage_one_battery, BatteryStage};
use crate::mapping::MappingStage;
use crate::mission::{
    MissionConfig, MissionEvent, MissionInputs, MissionLink, MissionPhase, MissionStage,
};
use crate::overrides::{GotoHover, OverrideCommand};
use crate::simulation::{StageHandler, TickContext};
use crate::stage::{Stage, StageError, StageOutcome};

/// Everything the driver takes from the scenario.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionDriverConfig {
    pub mission: MissionConfig,
    /// `_make_redirect_goto(..., 28.0)` for POI and egress redirects.
    pub redirect_speed_mps: f64,
    /// A drone below this battery fraction hands its POI off.
    pub handoff_low_fraction: f64,
    /// ...to a free drone above this one.
    pub handoff_fresh_fraction: f64,
    /// `coverage_in_region(poi, radius_m=50.0)` in `rescore_from_map`.
    pub rescore_radius_m: f64,
}

impl Default for MissionDriverConfig {
    fn default() -> Self {
        Self {
            mission: MissionConfig::default(),
            redirect_speed_mps: 28.0,
            handoff_low_fraction: 0.30,
            handoff_fresh_fraction: 0.50,
            rescore_radius_m: 50.0,
        }
    }
}

/// `WorldMap.coverage_in_region`: the fraction of grid cells within a disc
/// that have been visited at least once.
///
/// The disc is in **cell** units -- `r_cells = max(1, int(radius /
/// resolution))`, cells kept where `di^2 + dj^2 <= r_cells^2` -- centred on the
/// cell the point truncates into. `int()` truncates toward zero, as `as i64`
/// does. An empty window reports zero coverage.
pub fn coverage_in_region(mapping: &MappingStage, center_xy: [f64; 2], radius_m: f64) -> f64 {
    coverage_in_grid_region(
        mapping.bounds(),
        |i, j| mapping.coverage_count_at(i, j),
        center_xy,
        radius_m,
    )
}

/// [`coverage_in_region`] over any count grid, indexed `(i, j)`.
pub fn coverage_in_grid_region(
    bounds: argusnet_mapping::GridBounds,
    count_at: impl Fn(usize, usize) -> u32,
    center_xy: [f64; 2],
    radius_m: f64,
) -> f64 {
    let Ok((nx, ny)) = bounds.dimensions() else {
        return 0.0;
    };
    let ci = ((center_xy[0] - bounds.x_min_m) / bounds.resolution_m) as i64;
    let cj = ((center_xy[1] - bounds.y_min_m) / bounds.resolution_m) as i64;
    let r_cells = ((radius_m / bounds.resolution_m) as i64).max(1);
    let (i_lo, i_hi) = ((ci - r_cells).max(0), (ci + r_cells + 1).min(nx as i64));
    let (j_lo, j_hi) = ((cj - r_cells).max(0), (cj + r_cells + 1).min(ny as i64));
    if i_lo >= i_hi || j_lo >= j_hi {
        return 0.0;
    }
    let (mut total, mut covered) = (0u64, 0u64);
    for i in i_lo..i_hi {
        for j in j_lo..j_hi {
            if (i - ci).pow(2) + (j - cj).pow(2) <= r_cells.pow(2) {
                total += 1;
                if count_at(i as usize, j as usize) > 0 {
                    covered += 1;
                }
            }
        }
    }
    if total == 0 {
        0.0
    } else {
        covered as f64 / total as f64
    }
}

/// The fraction of cells that are enclosed interior holes -- the gap-fill
/// gate's input.
fn gap_cell_fraction(mapping: &MappingStage) -> f64 {
    let bounds = mapping.bounds();
    let Ok((nx, ny)) = bounds.dimensions() else {
        return 0.0;
    };
    let counts = mapping.coverage_counts();
    debug_assert_eq!(counts.len(), nx * ny, "coverage grid and bounds disagree");
    let view = CoverageView {
        counts: &counts,
        bounds: argusnet_planning::frontier::GridBounds {
            x_min_m: bounds.x_min_m,
            y_min_m: bounds.y_min_m,
            resolution_m: bounds.resolution_m,
            nx,
            ny,
        },
    };
    find_gap_cells(&view).len() as f64 / (nx * ny).max(1) as f64
}

/// One tick of recorded localization: what the machine would have read from
/// stage 8.
#[derive(Debug, Clone, PartialEq)]
pub struct RecordedLocalization {
    pub confidences: Vec<f64>,
    pub timed_out: bool,
}

/// A handle for replaying recorded localization into the mission machine.
///
/// Localization is the one input to the mission that no parity test can match
/// bit for bit: the Rust localizer draws from ADR-006 streams, not NumPy's. A
/// run handed the reference's recorded confidences instead is left with no
/// stochastic input at all, so everything the mission *does* -- transitions,
/// assignments, redirects, dwell, egress -- can be compared exactly for the
/// whole run. A replay seam, not a production path: an empty replay (the
/// default) reads stage 8 as normal.
#[derive(Debug, Clone, Default)]
pub struct LocalizationReplay {
    inner: std::sync::Arc<std::sync::Mutex<Vec<RecordedLocalization>>>,
}

impl LocalizationReplay {
    /// Replace stage 8's localization with `ticks`, indexed by tick number.
    pub fn load(&self, ticks: Vec<RecordedLocalization>) {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = ticks;
    }

    fn at(&self, tick: u64) -> Option<RecordedLocalization> {
        let ticks = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if ticks.is_empty() {
            return None;
        }
        ticks.get(tick as usize).cloned()
    }

    fn is_loaded(&self) -> bool {
        !self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .is_empty()
    }
}

/// What stage 9 checkpoints.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
struct DriverCheckpoint {
    machine: crate::mission::MissionState,
    pois: PoiManager,
    redirected: BTreeMap<String, String>,
    rescored: bool,
    #[serde(default)]
    coordinator: Option<String>,
}

/// What stage 9 reports for a replay: POI statuses with their positions, and
/// the elected coordinator. Refreshed at the end of every run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct DriverReport {
    /// Each POI's status beside its position, in construction order --
    /// `replace(s, position=_poi_manager._pois[s.poi_id].position)`.
    pub pois: Vec<(PoiStatus, [f64; 3])>,
    pub completed_count: usize,
    pub total_count: usize,
    /// `_shared_state.coordinator_id`.
    pub coordinator_drone_id: Option<String>,
}

/// A handle on the driver's [`DriverReport`].
#[derive(Debug, Clone, Default)]
pub struct DriverReportLink {
    inner: std::sync::Arc<std::sync::Mutex<DriverReport>>,
}

impl DriverReportLink {
    pub fn report(&self) -> DriverReport {
        self.inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }

    fn set(&self, report: DriverReport) {
        *self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = report;
    }
}

/// `CoordinationManager.elect_coordinator`: the highest battery fraction,
/// the first in fleet order on a tie, and a drone off the ledger counted as
/// full.
pub fn elect_coordinator(
    drone_ids: &[String],
    battery: &BatteryStage,
    capacity_wh: f64,
) -> Option<String> {
    let mut best: Option<(&String, f64)> = None;
    for id in drone_ids {
        let fraction = battery
            .remaining_wh(id)
            .map_or(1.0, |remaining| remaining / capacity_wh.max(1.0));
        // `best_frac = -1.0` and a strict `>`.
        if fraction > best.map_or(-1.0, |(_, f)| f) {
            best = Some((id, fraction));
        }
    }
    // `best_id = drone_ids[0]`, kept if no fraction ever beat -1.
    best.map(|(id, _)| id.clone())
        .or_else(|| drone_ids.first().cloned())
}

/// Stage 9: the mission machine and the work its phases do.
pub struct MissionDriverStage {
    machine: MissionStage,
    mission: MissionLink,
    /// Writes into stage 2's tables, queued this run and drained by
    /// [`Handoff::OverridesFromMission`] straight after it.
    ///
    /// [`Handoff::OverridesFromMission`]: crate::simulation::Handoff::OverridesFromMission
    override_commands: Vec<OverrideCommand>,
    pois: PoiManager,
    initial_pois: Vec<InspectionPoi>,
    /// `_drone_home_positions`: each launched drone's station.
    homes: BTreeMap<String, [f64; 3]>,
    /// `_drone_poi_assignments`: which POI each drone was last redirected to.
    redirected: BTreeMap<String, String>,
    /// `_poi_rescored`.
    rescored: bool,
    /// `_shared_state.coordinator_id`, elected once on the first scanning
    /// tick with a fleet (`_coordinator_elected`).
    coordinator: Option<String>,
    config: MissionDriverConfig,
    replay: LocalizationReplay,
    report: DriverReportLink,
}

impl std::fmt::Debug for MissionDriverStage {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MissionDriverStage")
            .field("phase", &self.mission.phase())
            .field("pois", &self.pois.total_count())
            .field("rescored", &self.rescored)
            .finish()
    }
}

impl MissionDriverStage {
    pub fn new(
        config: MissionDriverConfig,
        pois: Vec<InspectionPoi>,
        homes: BTreeMap<String, [f64; 3]>,
    ) -> Self {
        let machine = MissionStage::new(config.mission);
        let mission = machine.link();
        Self {
            machine,
            mission,
            override_commands: Vec::new(),
            pois: PoiManager::new(pois.clone()),
            initial_pois: pois,
            homes,
            redirected: BTreeMap::new(),
            rescored: false,
            coordinator: None,
            config,
            replay: LocalizationReplay::default(),
            report: DriverReportLink::default(),
        }
    }

    /// Take the override writes the last run queued, in order.
    pub fn take_override_commands(&mut self) -> Vec<OverrideCommand> {
        std::mem::take(&mut self.override_commands)
    }

    /// The handle a replay writer reads stage 9's report through.
    pub fn report_link(&self) -> DriverReportLink {
        self.report.clone()
    }

    fn publish_report(&self) {
        let position_of = |poi_id: &str| {
            self.initial_pois
                .iter()
                .find(|poi| poi.poi_id == poi_id)
                .map(|poi| poi.position)
                .expect("a status is for one of the manager's POIs")
        };
        self.report.set(DriverReport {
            pois: self
                .pois
                .statuses()
                .into_iter()
                .map(|status| {
                    let position = position_of(&status.poi_id);
                    (status, position)
                })
                .collect(),
            completed_count: self.pois.completed_count(),
            total_count: self.pois.total_count(),
            coordinator_drone_id: self.coordinator.clone(),
        });
    }

    /// The localization replay seam. See [`LocalizationReplay`].
    pub fn localization_replay(&self) -> LocalizationReplay {
        self.replay.clone()
    }

    pub fn config(&self) -> &MissionDriverConfig {
        &self.config
    }

    /// The POIs as constructed, before any rescore.
    pub fn pois(&self) -> &[InspectionPoi] {
        &self.initial_pois
    }

    /// The machine's handle: stage 2 and stage 8 watch it.
    pub fn mission_link(&self) -> MissionLink {
        self.mission.clone()
    }

    fn inspect(
        &mut self,
        timestamp_s: f64,
        dt_s: f64,
        mobile: &[(String, [f64; 3])],
        battery: &BatteryStage,
        mapping: &MappingStage,
    ) {
        if !self.rescored {
            let radius = self.config.rescore_radius_m;
            self.pois
                .rescore_from_map(|xy| Some(coverage_in_region(mapping, xy, radius)));
            self.rescored = true;
        }

        let capacity_wh = battery.model().capacity_wh;
        let fraction_of = |id: &str| {
            battery
                .remaining_wh(id)
                .map_or(1.0, |remaining| remaining / capacity_wh)
        };
        let mut unassigned: Vec<String> = Vec::new();
        for (id, position) in mobile {
            let remaining_wh = battery.remaining_wh(id).unwrap_or(capacity_wh);
            let context = PoiAssignmentContext::new(id, *position, remaining_wh, capacity_wh);
            let assigned = self.pois.assign_energy_aware(&context).is_some();
            if !assigned && self.pois.assignment_for(id).is_none() {
                unassigned.push(id.clone());
            }
            // Redirect on a NEW assignment only.
            if let Some(active) = self.pois.assignment_for(id).map(str::to_string) {
                if self.redirected.get(id) != Some(&active) {
                    self.redirected.insert(id.clone(), active.clone());
                    let poi = self
                        .initial_pois
                        .iter()
                        .find(|p| p.poi_id == active)
                        .expect("an assigned POI is one of the manager's");
                    self.override_commands.push(OverrideCommand::SetPoi(
                        id.clone(),
                        GotoHover::redirect(
                            *position,
                            [poi.position[0], poi.position[1]],
                            poi.position[2],
                            timestamp_s,
                            self.config.redirect_speed_mps,
                        ),
                    ));
                }
            }
            // Handoff: a low drone with a POI passes it to the first fresh,
            // free drone in fleet order.
            if fraction_of(id) < self.config.handoff_low_fraction
                && self.pois.assignment_for(id).is_some()
            {
                for (other, _) in mobile {
                    if other != id
                        && fraction_of(other) > self.config.handoff_fresh_fraction
                        && self.pois.assignment_for(other).is_none()
                    {
                        self.pois.trigger_handoff(id, other);
                        break;
                    }
                }
            }
            self.pois.accumulate_dwell(id, *position, dt_s);
        }
        // Team assignment: one POI left and more than one free drone.
        let pending: Vec<String> = self
            .pois
            .statuses()
            .into_iter()
            .filter(|s| s.state == PoiState::Pending)
            .map(|s| s.poi_id)
            .collect();
        if pending.len() == 1 && unassigned.len() > 1 {
            self.pois.request_team_assign(&pending[0], &unassigned[1]);
        }
        for done in self.pois.check_completions(timestamp_s) {
            let finished: Vec<String> = self
                .redirected
                .iter()
                .filter(|(_, poi)| **poi == done)
                .map(|(drone, _)| drone.clone())
                .collect();
            for drone in finished {
                self.redirected.remove(&drone);
                self.override_commands
                    .push(OverrideCommand::ClearPoi(drone.clone()));
            }
        }
    }
}

impl StageHandler for MissionDriverStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let timestamp_s = context.timestamp_s;
        // Stage 1's ledger, as it left it this tick, through the lent views
        // (SESSION_OWNERSHIP.md §2a). A run with none reads as an empty one:
        // every drone off the ledger, counted as full.
        // Whatever a run that no handoff drained left behind is dropped, so a
        // driver installed without stage 2 does not grow without bound.
        self.override_commands.clear();
        let empty = BatteryStage::new();
        let battery = stage_one_battery(context.stages).unwrap_or(&empty);
        // Stage 8's map as it left it this tick, through the lent views.
        let mapping = context
            .stages
            .get::<MappingStage>(Stage::UpdateMappingAndLocalization)
            .ok_or_else(|| {
                StageError::Invariant("stage 9 needs stage 8's map installed".to_string())
            })?;
        let phase = self.mission.phase();
        // `mobile_states`: flown positions, fleet order.
        let mobile: Vec<(String, [f64; 3])> = context
            .world
            .mobile_platforms()
            .map(|p| {
                (
                    p.node_id.clone(),
                    [p.position.x, p.position.y, p.position.z],
                )
            })
            .collect();

        // Coordinator election, once, on the first scanning tick that has a
        // fleet. The reference reads the batteries as stage 1 left them.
        if self.coordinator.is_none() && phase == MissionPhase::Scanning && !mobile.is_empty() {
            let ids: Vec<String> = mobile.iter().map(|(id, _)| id.clone()).collect();
            self.coordinator = elect_coordinator(&ids, battery, battery.model().capacity_wh);
        }

        if phase == MissionPhase::Inspecting {
            self.inspect(timestamp_s, context.dt_s, &mobile, battery, mapping);
        }

        let home_distances_m = mobile
            .iter()
            .map(|(id, position)| {
                self.homes.get(id).map(|home| {
                    let (dx, dy) = (home[0] - position[0], home[1] - position[1]);
                    (dx * dx + dy * dy).sqrt()
                })
            })
            .collect();
        let (localization_confidences, localizer_timed_out) = if self.replay.is_loaded() {
            let recorded = self.replay.at(context.tick).ok_or_else(|| {
                StageError::Invariant(format!(
                    "localization replay has no entry for tick {}",
                    context.tick
                ))
            })?;
            (recorded.confidences, recorded.timed_out)
        } else {
            (mapping.confidences(), mapping.localizer_timed_out())
        };
        // The gap search scans the whole grid. The reference runs it only on
        // the scanning gate -- while scanning, once coverage has reached the
        // threshold (`sim.py`) -- and that is the only place the machine
        // reads it, so it is computed there and nowhere else (P4.10c: it was
        // ~45 % of a scan scenario's tick when run every tick).
        let coverage_fraction = mapping.coverage_fraction();
        let gap_cell_fraction = (phase == MissionPhase::Scanning
            && coverage_fraction >= self.machine.config().scan_coverage_threshold)
            .then(|| gap_cell_fraction(mapping));
        self.mission.set_inputs(MissionInputs {
            timestamp_s,
            coverage_fraction,
            gap_cell_fraction,
            mobile_count: mobile.len(),
            localization_confidences,
            localizer_timed_out,
            all_pois_complete: self.pois.all_complete(),
            home_distances_m,
        });
        let outcome = self.machine.run(context)?;

        for event in self.mission.take_events() {
            match event {
                MissionEvent::EgressRoutesRequested { at_s } => {
                    for (id, _) in &mobile {
                        self.override_commands
                            .push(OverrideCommand::ClearPoi(id.clone()));
                    }
                    self.redirected.clear();
                    for (id, position) in &mobile {
                        if let Some(home) = self.homes.get(id) {
                            self.override_commands.push(OverrideCommand::SetEgress(
                                id.clone(),
                                GotoHover::redirect(
                                    *position,
                                    [home[0], home[1]],
                                    home[2],
                                    at_s,
                                    self.config.redirect_speed_mps,
                                ),
                            ));
                        }
                    }
                    context.note("egress routes set");
                }
                MissionEvent::EgressRoutesCleared { .. } => {
                    self.override_commands.push(OverrideCommand::ClearAllEgress)
                }
                MissionEvent::PhaseChanged { .. }
                | MissionEvent::RescorePois { .. }
                | MissionEvent::LocalizationTimedOut { .. } => {}
            }
        }
        self.publish_report();
        Ok(outcome)
    }

    /// The machine, the POI manager, which drone was last redirected where,
    /// and the rescore latch. The POIs themselves and the homes are scenario,
    /// rebuilt by assembly -- and checked against, below.
    fn checkpoint_state(&self) -> Option<serde_json::Value> {
        let state = DriverCheckpoint {
            machine: self.mission.state(),
            pois: self.pois.clone(),
            redirected: self.redirected.clone(),
            rescored: self.rescored,
            coordinator: self.coordinator.clone(),
        };
        Some(serde_json::to_value(state).expect("mission driver state serialises"))
    }

    fn check_state(
        &self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: DriverCheckpoint = crate::simulation::decode_stage_state(state)?;
        let ids = |statuses: Vec<argusnet_planning::poi::PoiStatus>| {
            statuses.into_iter().map(|s| s.poi_id).collect::<Vec<_>>()
        };
        if ids(decoded.pois.statuses()) != ids(self.pois.statuses()) {
            return Err(crate::simulation::StageStateError::Malformed {
                detail: "the checkpoint's POIs are not this run's".into(),
            });
        }
        Ok(())
    }

    fn restore_state(
        &mut self,
        state: &serde_json::Value,
    ) -> Result<(), crate::simulation::StageStateError> {
        let decoded: DriverCheckpoint = crate::simulation::decode_stage_state(state)?;
        self.mission.restore_state(decoded.machine);
        self.pois = decoded.pois;
        self.redirected = decoded.redirected;
        self.rescored = decoded.rescored;
        self.coordinator = decoded.coordinator;
        self.override_commands.clear();
        self.publish_report();
        Ok(())
    }

    fn reset(&mut self) {
        self.mission.reset();
        self.pois = PoiManager::new(self.initial_pois.clone());
        self.redirected.clear();
        self.rescored = false;
        self.coordinator = None;
        self.override_commands.clear();
        self.publish_report();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_mapping::GridBounds;

    /// A one-drone run with stage 8 and this driver installed, and
    /// localization replayed so only the gate under test decides the phase.
    fn gate_rig(replay: Vec<RecordedLocalization>) -> (crate::simulation::Simulation, MissionLink) {
        use crate::config::SimulationConfig;
        use crate::stage::Stage;
        use argusnet_types::{NodeState, Vector3};

        let mut sim = crate::simulation::Simulation::new(SimulationConfig {
            seed: 3,
            dt_s: 1.0,
            steps: Some(500),
            run_id: "gate".to_string(),
        })
        .expect("config");
        sim.load_world(
            vec![NodeState {
                node_id: "drone-1".into(),
                position: Vector3::new(25.0, 25.0, 80.0),
                velocity: Vector3::new(0.0, 0.0, 0.0),
                is_mobile: true,
                timestamp_s: 0.0,
                health: 1.0,
                ..Default::default()
            }],
            Vec::new(),
        );
        // 10 x 10 cells of 50 m.
        let bounds = GridBounds {
            x_min_m: 0.0,
            x_max_m: 500.0,
            y_min_m: 0.0,
            y_max_m: 500.0,
            resolution_m: 50.0,
        };
        let mut mapping = crate::mapping::MappingStage::new(bounds).expect("bounds");
        mapping.set_footprint_radius_m(40.0);
        let stage = MissionDriverStage::new(
            MissionDriverConfig {
                mission: MissionConfig {
                    scan_coverage_threshold: 0.3,
                    ..MissionConfig::default()
                },
                ..MissionDriverConfig::default()
            },
            Vec::new(),
            BTreeMap::new(),
        );
        stage.localization_replay().load(replay);
        let mission = stage.mission_link();
        mapping.watch_mission(mission.clone());
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
            .expect("stage 8");
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
            .expect("stage 9");
        (sim, mission)
    }

    fn converged(ticks: usize) -> Vec<RecordedLocalization> {
        vec![
            RecordedLocalization {
                confidences: vec![1.0],
                timed_out: false,
            };
            ticks
        ]
    }

    /// The gap-fill gate: coverage above the threshold is not enough while an
    /// enclosed hole of at least 2% of the grid remains. The 12 cells around a
    /// 2 x 2 hole (4%) are flown FIRST, so the hole is enclosed long before
    /// coverage reaches the 30% threshold -- an unenclosed hole is connected
    /// to the uncovered border and is legitimately not a gap. Then the rest.
    /// The mission must stay in `Scanning` until the hole is flown too.
    #[test]
    fn an_enclosed_hole_holds_the_mission_in_scanning() {
        let in_hole = |i: i32, j: i32| (4..=5).contains(&i) && (4..=5).contains(&j);
        let around_hole = |i: i32, j: i32| (3..=6).contains(&i) && (3..=6).contains(&j);
        let cells: Vec<(i32, i32)> = (0..10).flat_map(|i| (0..10).map(move |j| (i, j))).collect();
        let ring: Vec<(f64, f64)> = cells
            .iter()
            .filter(|(i, j)| around_hole(*i, *j) && !in_hole(*i, *j))
            .chain(cells.iter().filter(|(i, j)| !around_hole(*i, *j)))
            .map(|(i, j)| (25.0 + 50.0 * *i as f64, 25.0 + 50.0 * *j as f64))
            .collect();
        assert_eq!(ring.len(), 96);
        let hole = [
            (225.0, 225.0),
            (275.0, 225.0),
            (225.0, 275.0),
            (275.0, 275.0),
        ];
        let (mut sim, mission) = gate_rig(converged(ring.len() + hole.len()));
        for (x, y) in &ring {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = *x;
            drone.position.y = *y;
            sim.step().expect("tick");
            assert_eq!(
                mission.phase(),
                MissionPhase::Scanning,
                "left Scanning with the hole still open, at ({x}, {y})"
            );
        }
        // Fill the hole a cell at a time: 3%, 2%, then 1% remains. The gate
        // holds while the gap is >= 2% -- including exactly 2% -- and releases
        // on the third cell. (With no POIs, the fourth tick would go straight
        // on to egress, so the check stops at the release.)
        let expected = [
            MissionPhase::Scanning,
            MissionPhase::Scanning,
            MissionPhase::Inspecting,
        ];
        for ((x, y), phase) in hole.iter().zip(expected) {
            let drone = sim.world_mut().platform_mut("drone-1").expect("drone");
            drone.position.x = *x;
            drone.position.y = *y;
            sim.step().expect("tick");
            assert_eq!(mission.phase(), phase, "after flying hole cell ({x}, {y})");
        }
    }

    /// A timed-out localizer latches in the machine, from the replay too.
    #[test]
    fn a_replayed_localization_timeout_latches() {
        let mut replay = converged(3);
        replay[1].timed_out = true;
        let (mut sim, mission) = gate_rig(replay);
        sim.step().expect("tick 0");
        assert!(!mission.state().localization_timed_out());
        sim.step().expect("tick 1");
        assert!(
            mission.state().localization_timed_out(),
            "the timeout did not latch"
        );
    }

    fn driver(pois: Vec<InspectionPoi>) -> MissionDriverStage {
        MissionDriverStage::new(MissionDriverConfig::default(), pois, BTreeMap::new())
    }

    fn empty_map() -> crate::mapping::MappingStage {
        crate::mapping::MappingStage::new(GridBounds {
            x_min_m: 0.0,
            x_max_m: 1000.0,
            y_min_m: 0.0,
            y_max_m: 1000.0,
            resolution_m: 50.0,
        })
        .expect("bounds")
    }

    fn poi(id: &str, x: f64, priority: i64) -> InspectionPoi {
        InspectionPoi {
            poi_id: id.to_string(),
            name: id.to_string(),
            position: [x, 0.0, 40.0],
            priority,
            required_dwell_s: 15.0,
        }
    }

    /// The team branch is dead in the reference, and stays dead here.
    ///
    /// `run_simulation` calls `request_team_assign(pending_pois[0], ...)` --
    /// on a POI it has just filtered as **pending** -- and `request_team_assign`
    /// refuses anything that is not already **active**. So when one POI is
    /// left pending and two drones are free (here because both are too low to
    /// afford it), nothing is assigned. The driver reproduces the call; this
    /// pins that it changes nothing, so a "fix" that assigned the POI would be
    /// a behaviour change and fail here.
    #[test]
    fn the_reference_team_assignment_branch_assigns_nothing() {
        let mut battery = BatteryStage::new();
        battery.set_model(argusnet_planning::battery::BatteryModel {
            capacity_wh: 500.0,
            ..Default::default()
        });
        for id in ["drone-a", "drone-b", "drone-c"] {
            battery.register(id, None);
        }
        let mut stage = driver(vec![poi("poi-00", 100.0, 2), poi("poi-01", 900.0, 1)]);
        // Below the 20% reserve: no usable energy, so no assignment.
        for id in ["drone-b", "drone-c"] {
            assert!(battery.set_remaining_wh(id, 50.0));
        }
        let mobile = vec![
            ("drone-a".to_string(), [100.0, 0.0, 40.0]),
            ("drone-b".to_string(), [500.0, 0.0, 40.0]),
            ("drone-c".to_string(), [600.0, 0.0, 40.0]),
        ];
        stage.inspect(0.0, 1.0, &mobile, &battery, &empty_map());
        assert_eq!(stage.pois.assignment_for("drone-a"), Some("poi-00"));
        let pending = stage
            .pois
            .statuses()
            .into_iter()
            .filter(|s| s.state == PoiState::Pending)
            .count();
        assert_eq!(pending, 1, "the setup no longer reaches the team branch");
        assert_eq!(stage.pois.assignment_for("drone-b"), None);
        assert_eq!(stage.pois.assignment_for("drone-c"), None);
    }

    /// `elect_coordinator`: the highest battery fraction; the first in fleet
    /// order on a tie; a drone off the ledger counts as full.
    #[test]
    fn the_coordinator_is_the_fullest_battery_first_on_a_tie() {
        let mut battery = BatteryStage::new();
        battery.register("a", None);
        battery.register("b", None);
        let capacity_wh = battery.model().capacity_wh;
        let ids = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<_>>();
        assert_eq!(
            elect_coordinator(&ids(&["a", "b"]), &battery, capacity_wh),
            Some("a".into())
        );
        battery.set_remaining_wh("a", 0.2 * capacity_wh);
        assert_eq!(
            elect_coordinator(&ids(&["a", "b"]), &battery, capacity_wh),
            Some("b".into())
        );
        battery.set_remaining_wh("b", 0.1 * capacity_wh);
        assert_eq!(
            elect_coordinator(&ids(&["a", "b", "ghost"]), &battery, capacity_wh),
            Some("ghost".into()),
            "off the ledger is full"
        );
        assert_eq!(elect_coordinator(&[], &battery, capacity_wh), None);
    }

    /// Elected once (`_coordinator_elected`): a later scanning tick on which
    /// another drone's battery is fuller does not re-elect.
    #[test]
    fn the_coordinator_is_elected_once() {
        use crate::config::SimulationConfig;
        use crate::stage::Stage;
        use argusnet_types::{NodeState, Vector3};

        let mut sim = crate::simulation::Simulation::new(SimulationConfig {
            seed: 3,
            dt_s: 1.0,
            steps: Some(10),
            run_id: "coordinator".to_string(),
        })
        .expect("config");
        let drone = |id: &str, x: f64| NodeState {
            node_id: id.into(),
            position: Vector3::new(x, 25.0, 80.0),
            velocity: Vector3::new(0.0, 0.0, 0.0),
            is_mobile: true,
            timestamp_s: 0.0,
            health: 1.0,
            ..Default::default()
        };
        sim.load_world(
            vec![drone("drone-1", 25.0), drone("drone-2", 475.0)],
            Vec::new(),
        );
        let bounds = GridBounds {
            x_min_m: 0.0,
            x_max_m: 500.0,
            y_min_m: 0.0,
            y_max_m: 500.0,
            resolution_m: 50.0,
        };
        let mapping = crate::mapping::MappingStage::new(bounds).expect("bounds");
        let mut battery = BatteryStage::new();
        battery.register("drone-1", None);
        battery.register("drone-2", None);
        sim.set_stage_handler(Stage::SampleControllersAndBattery, Box::new(battery))
            .expect("stage 1");
        let stage =
            MissionDriverStage::new(MissionDriverConfig::default(), Vec::new(), BTreeMap::new());
        let report = stage.report_link();
        let mission = stage.mission_link();
        sim.set_stage_handler(Stage::UpdateMappingAndLocalization, Box::new(mapping))
            .expect("stage 8");
        sim.set_stage_handler(Stage::EvaluateMissionTransitions, Box::new(stage))
            .expect("stage 9");

        sim.step().expect("tick 0");
        assert_eq!(
            report.report().coordinator_drone_id.as_deref(),
            Some("drone-1")
        );
        assert!(BatteryStage::of_mut(&mut sim)
            .expect("stage 1 is a battery")
            .set_remaining_wh("drone-1", 10.0));
        sim.step().expect("tick 1");
        assert_eq!(mission.phase(), MissionPhase::Scanning);
        assert_eq!(
            report.report().coordinator_drone_id.as_deref(),
            Some("drone-1"),
            "re-elected"
        );
    }
}
