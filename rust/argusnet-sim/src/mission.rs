//! The scan / localize / inspect / egress mission state machine.
//!
//! P4.8 asks for "explicit mission state" and for the
//! scan/localize/inspect/egress transitions to be preserved. In the reference
//! they are not a machine at all: they are an `if/elif` chain over a
//! `_scan_phase` string, roughly 120 lines inside `run_simulation`, interleaved
//! with POI assignment, battery handoff and trajectory construction. The
//! transitions and the work they trigger are the same code.
//!
//! Here they are separated. [`MissionState::advance`] decides **only** what the
//! phase should be and emits [`MissionEvent`]s; every side effect the reference
//! performs inline — rescoring POIs, clearing redirects, building egress routes
//! — is the caller's, driven by those events. That is what makes the
//! transitions testable without a world.
//!
//! # One transition per tick, because the reference is an `elif` chain
//!
//! The chain tests `_scan_phase` as it stood at the **top** of the tick, so at
//! most one branch runs and a phase entered this tick does no work until the
//! next one. `ARCHITECTURE_AND_PROTOCOLS.md` §4 is explicit that this matters:
//!
//! > *"The first Rust port MUST preserve the Python causal order, including
//! > decisions that take effect on the next tick. It MUST NOT silently
//! > 'improve' that order."*
//!
//! A machine that cascaded — scanning straight through localizing into
//! inspecting in one call, when the inputs allow it — would finish missions in
//! fewer ticks and be wrong. [`MissionState::advance`] takes at most one step.
//!
//! # Three asymmetries, preserved and pinned
//!
//! Each was found by reading the two branches against each other, and each
//! would be easy to "fix" into a difference in behaviour.
//!
//! 1. **An empty fleet converges out of `Scanning` but is stuck in
//!    `Localizing`.** The scanning branch computes
//!    `loc_converged = (not mobile_states) or all(...)`, so with no drones it
//!    goes straight to `Inspecting`. The localizing branch guards with
//!    `if mobile_states and all(...)`, so with no drones it can never leave.
//!    The same input means "converged" in one branch and "not converged" in
//!    the other.
//!
//! 2. **A drone with no home counts as having arrived.** The egress branch
//!    does `if home is None: arrived += 1; continue`. So a fleet whose homes
//!    were never recorded completes egress immediately — but only if it is
//!    non-empty, because the outer guard is `if mobile_states and arrived >=
//!    len(mobile_states)`.
//!
//! 3. **The localization timeout latches and is never cleared**, and it is
//!    raised from the localization stage on *every* tick, not only while
//!    `Localizing`. A run that converges during scanning and skips
//!    `Localizing` entirely can still report `localization_timed_out`.
//!
//! # Thresholds, and which way each comparison points
//!
//! Getting one of these backwards changes when a mission advances and nothing
//! crashes, so they are named constants with the reference's own direction:
//!
//! | gate | comparison | default |
//! |---|---|---|
//! | scan coverage | `coverage_fraction >= threshold` | 0.70 |
//! | gap fill | `gap_fraction < gap_fill_min_fraction` | 0.02 |
//! | localization confidence | `confidence >= threshold` (all drones) | 0.65 |
//! | egress arrival | `distance < radius` | 15.0 m |
//!
//! Note that the gap gate is the only strict one.

use std::fmt;

use serde::{Deserialize, Serialize};

use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::simulation::{StageHandler, TickContext};
use crate::stage::{StageError, StageOutcome};

/// `scenario.options.scan_coverage_threshold` default.
pub const DEFAULT_SCAN_COVERAGE_THRESHOLD: f64 = 0.70;
/// `_loc_confidence_threshold`, a literal in the reference.
pub const DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD: f64 = 0.65;
/// `FrontierConfig.gap_fill_min_fraction`.
pub const DEFAULT_GAP_FILL_MIN_FRACTION: f64 = 0.02;
/// The egress branch's `dist_m < 15.0`.
pub const DEFAULT_EGRESS_ARRIVAL_RADIUS_M: f64 = 15.0;

/// Where the mission is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissionPhase {
    Scanning,
    Localizing,
    Inspecting,
    Egress,
    /// Terminal. The reference has no branch for it, so nothing leaves.
    Complete,
}

impl MissionPhase {
    /// The string the reference's `ScanMissionState.phase` carries, which is
    /// what reaches replay and the viewer.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Scanning => "scanning",
            Self::Localizing => "localizing",
            Self::Inspecting => "inspecting",
            Self::Egress => "egress",
            Self::Complete => "complete",
        }
    }
}

impl fmt::Display for MissionPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The gates. Defaults are the reference's.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionConfig {
    pub scan_coverage_threshold: f64,
    pub localization_confidence_threshold: f64,
    pub gap_fill_min_fraction: f64,
    pub egress_arrival_radius_m: f64,
}

impl Default for MissionConfig {
    fn default() -> Self {
        Self {
            scan_coverage_threshold: DEFAULT_SCAN_COVERAGE_THRESHOLD,
            localization_confidence_threshold: DEFAULT_LOCALIZATION_CONFIDENCE_THRESHOLD,
            gap_fill_min_fraction: DEFAULT_GAP_FILL_MIN_FRACTION,
            egress_arrival_radius_m: DEFAULT_EGRESS_ARRIVAL_RADIUS_M,
        }
    }
}

/// Everything the machine reads, gathered once per tick.
///
/// Deliberately values rather than borrowed world state: the machine cannot
/// reach into the simulation, so it cannot depend on anything the caller has
/// not decided to show it.
#[derive(Debug, Clone, PartialEq)]
pub struct MissionInputs {
    pub timestamp_s: f64,
    /// `cov.coverage_fraction`, in [0, 1].
    pub coverage_fraction: f64,
    /// Enclosed interior holes as a fraction of all cells.
    pub gap_cell_fraction: f64,
    /// Mobile platforms this tick — the reference's `len(mobile_states)`.
    ///
    /// Carried explicitly rather than inferred from either vector below,
    /// because the reference's guards are on `mobile_states` while the counts
    /// are over derived lists. Inferring it would make the two disagree
    /// silently if a platform ever went missing from one of them.
    pub mobile_count: usize,
    /// One confidence per mobile platform, in platform order.
    pub localization_confidences: Vec<f64>,
    /// `GridLocalizer.any_timed_out` this tick. Latched by the machine.
    pub localizer_timed_out: bool,
    /// `POIManager.all_complete`.
    pub all_pois_complete: bool,
    /// Distance to home per mobile platform, in platform order. `None` where
    /// no home was recorded — which the reference counts as arrived.
    pub home_distances_m: Vec<Option<f64>>,
}

impl MissionInputs {
    /// An input set for a fleet that has done nothing yet.
    pub fn at(timestamp_s: f64) -> Self {
        Self {
            timestamp_s,
            coverage_fraction: 0.0,
            gap_cell_fraction: 0.0,
            mobile_count: 0,
            localization_confidences: Vec::new(),
            localizer_timed_out: false,
            all_pois_complete: false,
            home_distances_m: Vec::new(),
        }
    }

    fn fleet_is_empty(&self) -> bool {
        self.mobile_count == 0
    }
}

/// Something the caller has to act on.
///
/// P4.8 says to "consume typed events once". Each event names work the
/// reference performs inline at the transition, so a caller that handles all of
/// them reproduces the reference's side effects without the machine reaching
/// into the world to do them itself.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MissionEvent {
    /// The phase changed. `at_s` is the new `phase_started_at_s`.
    PhaseChanged {
        from: MissionPhase,
        to: MissionPhase,
        at_s: f64,
    },
    /// First entry into `Inspecting`: rescore POIs against the scan map.
    /// Emitted **once per run**, matching the reference's `_poi_rescored`
    /// latch, which is never cleared.
    RescorePois { at_s: f64 },
    /// Entering `Egress`: drop POI redirects and route every drone home.
    EgressRoutesRequested { at_s: f64 },
    /// Leaving `Egress`: drop the egress redirects.
    EgressRoutesCleared { at_s: f64 },
    /// The localization timeout latched, this tick. Emitted once.
    LocalizationTimedOut { at_s: f64 },
}

/// The mission's phase, and the two latches that outlive it.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MissionState {
    phase: MissionPhase,
    phase_started_at_s: f64,
    localization_timed_out: bool,
    pois_rescored: bool,
}

impl Default for MissionState {
    fn default() -> Self {
        Self::new()
    }
}

impl MissionState {
    /// `_scan_phase = "scanning"`, `_phase_started_at_s = 0.0`.
    pub fn new() -> Self {
        Self {
            phase: MissionPhase::Scanning,
            phase_started_at_s: 0.0,
            localization_timed_out: false,
            pois_rescored: false,
        }
    }

    pub fn phase(&self) -> MissionPhase {
        self.phase
    }

    pub fn phase_started_at_s(&self) -> f64 {
        self.phase_started_at_s
    }

    /// Latched: once true it never clears, and it is raised from the
    /// localization stage regardless of phase.
    pub fn localization_timed_out(&self) -> bool {
        self.localization_timed_out
    }

    /// Whether the one-shot POI rescore has already happened.
    pub fn pois_rescored(&self) -> bool {
        self.pois_rescored
    }

    pub fn is_complete(&self) -> bool {
        self.phase == MissionPhase::Complete
    }

    /// Take at most one transition.
    ///
    /// Events come back in the order the caller should apply them: the timeout
    /// latch first (it is raised in an earlier stage than the phase chain),
    /// then the phase change, then the work that entering the new phase
    /// requires.
    pub fn advance(&mut self, config: &MissionConfig, inputs: &MissionInputs) -> Vec<MissionEvent> {
        let mut events = Vec::new();

        // Raised in the localization stage, which runs before the phase chain
        // and does not consult the phase at all.
        if inputs.localizer_timed_out && !self.localization_timed_out {
            self.localization_timed_out = true;
            events.push(MissionEvent::LocalizationTimedOut {
                at_s: inputs.timestamp_s,
            });
        }

        let next = match self.phase {
            MissionPhase::Scanning => self.scanning_transition(config, inputs),
            MissionPhase::Localizing => self.localizing_transition(config, inputs),
            MissionPhase::Inspecting => self.inspecting_transition(inputs),
            MissionPhase::Egress => self.egress_transition(config, inputs),
            // No branch in the reference. Terminal.
            MissionPhase::Complete => None,
        };

        let Some(to) = next else {
            return events;
        };

        let from = self.phase;
        self.phase = to;
        self.phase_started_at_s = inputs.timestamp_s;
        events.push(MissionEvent::PhaseChanged {
            from,
            to,
            at_s: inputs.timestamp_s,
        });

        if from == MissionPhase::Egress {
            events.push(MissionEvent::EgressRoutesCleared {
                at_s: inputs.timestamp_s,
            });
        }
        match to {
            MissionPhase::Inspecting => {
                // The reference rescores at the top of the *inspecting* branch,
                // guarded by a latch that is never cleared -- so it happens
                // once per run, not once per entry.
                if !self.pois_rescored {
                    self.pois_rescored = true;
                    events.push(MissionEvent::RescorePois {
                        at_s: inputs.timestamp_s,
                    });
                }
            }
            MissionPhase::Egress => events.push(MissionEvent::EgressRoutesRequested {
                at_s: inputs.timestamp_s,
            }),
            _ => {}
        }
        events
    }

    /// `scanning` -> `inspecting` when localization already converged, else
    /// `localizing`. Both gates must pass first.
    fn scanning_transition(
        &self,
        config: &MissionConfig,
        inputs: &MissionInputs,
    ) -> Option<MissionPhase> {
        if inputs.coverage_fraction < config.scan_coverage_threshold {
            return None;
        }
        // Strictly less: at exactly the threshold the mission keeps scanning.
        if inputs.gap_cell_fraction >= config.gap_fill_min_fraction {
            return None;
        }
        // `(not mobile_states) or all(...)`: an empty fleet is converged here.
        // The localizing branch reads the same emptiness the other way.
        let converged = inputs.fleet_is_empty() || self.confidences_converged(config, inputs);
        Some(if converged {
            MissionPhase::Inspecting
        } else {
            MissionPhase::Localizing
        })
    }

    /// `localizing` -> `inspecting` only with a **non-empty** fleet, all of
    /// whose estimates have converged.
    fn localizing_transition(
        &self,
        config: &MissionConfig,
        inputs: &MissionInputs,
    ) -> Option<MissionPhase> {
        if inputs.fleet_is_empty() {
            return None;
        }
        self.confidences_converged(config, inputs)
            .then_some(MissionPhase::Inspecting)
    }

    fn inspecting_transition(&self, inputs: &MissionInputs) -> Option<MissionPhase> {
        inputs.all_pois_complete.then_some(MissionPhase::Egress)
    }

    /// `egress` -> `complete` with a non-empty fleet, all of which have
    /// arrived. A drone with no recorded home counts as arrived.
    fn egress_transition(
        &self,
        config: &MissionConfig,
        inputs: &MissionInputs,
    ) -> Option<MissionPhase> {
        if inputs.fleet_is_empty() {
            return None;
        }
        // `arrived` counts over the per-platform list; the comparison is
        // against `len(mobile_states)`. They are the same number when the
        // caller supplies one entry per platform, and when they are not, this
        // refuses to complete rather than completing on a short list.
        let arrived = inputs
            .home_distances_m
            .iter()
            .filter(|distance| match distance {
                // `if home is None: arrived += 1; continue` -- a platform with
                // no recorded home is treated as already there.
                None => true,
                Some(d) => *d < config.egress_arrival_radius_m,
            })
            .count();
        (arrived >= inputs.mobile_count).then_some(MissionPhase::Complete)
    }

    fn confidences_converged(&self, config: &MissionConfig, inputs: &MissionInputs) -> bool {
        inputs
            .localization_confidences
            .iter()
            .all(|c| *c >= config.localization_confidence_threshold)
    }
}

// ---------------------------------------------------------------------------
// Stage 9
// ---------------------------------------------------------------------------

/// A handle onto the mission stage, held by whoever drives the tick.
///
/// [`Simulation::set_stage_handler`] takes ownership of a boxed
/// [`StageHandler`], so once the stage is installed the caller cannot reach it
/// again. The tick still needs to push this tick's inputs in and pull this
/// tick's events out, and a shared handle is how: the stage owns nothing the
/// caller needs and the caller owns nothing the stage needs.
///
/// Cloning it clones the handle, not the state.
///
/// [`Simulation::set_stage_handler`]: crate::simulation::Simulation::set_stage_handler
#[derive(Debug, Clone, Default)]
pub struct MissionLink {
    inner: Arc<Mutex<Shared>>,
}

#[derive(Debug, Default)]
struct Shared {
    state: MissionState,
    pending: Option<MissionInputs>,
    events: Vec<MissionEvent>,
}

impl MissionLink {
    fn lock(&self) -> MutexGuard<'_, Shared> {
        // A poisoned lock means something panicked mid-tick. The data behind
        // it is plain values with no invariant a panic could have broken
        // halfway, and refusing to read the mission phase because of an
        // unrelated panic elsewhere would turn one failure into two.
        self.inner.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Supply this tick's inputs. Must happen before the tick reaches stage 9,
    /// and the inputs must carry this tick's timestamp.
    pub fn set_inputs(&self, inputs: MissionInputs) {
        self.lock().pending = Some(inputs);
    }

    /// Take the events the last run produced, leaving none behind.
    ///
    /// P4.8 says to "consume typed events once"; a taking accessor is how that
    /// is enforced rather than asked for.
    pub fn take_events(&self) -> Vec<MissionEvent> {
        std::mem::take(&mut self.lock().events)
    }

    pub fn state(&self) -> MissionState {
        self.lock().state
    }

    /// Take a checkpoint's machine state back. Nothing pending survives.
    pub fn restore_state(&self, state: MissionState) {
        let mut shared = self.lock();
        shared.state = state;
        shared.pending = None;
        shared.events.clear();
    }

    /// Back to a fresh run's state: `Scanning`, no latches, nothing pending.
    pub fn reset(&self) {
        let mut shared = self.lock();
        shared.state = MissionState::default();
        shared.pending = None;
        shared.events.clear();
    }

    pub fn phase(&self) -> MissionPhase {
        self.lock().state.phase()
    }
}

/// [`Stage::EvaluateMissionTransitions`] as a [`StageHandler`].
///
/// The machine above is deliberately pure, so this is the only place it meets
/// the tick. Inputs are pushed in by whoever computed them -- stage 8 owns
/// coverage and localization -- rather than pulled, because a stage that could
/// reach into the world to fetch its own inputs could also reach into truth.
///
/// [`Stage::EvaluateMissionTransitions`]: crate::stage::Stage::EvaluateMissionTransitions
#[derive(Debug)]
pub struct MissionStage {
    config: MissionConfig,
    enabled: bool,
    link: MissionLink,
}

impl MissionStage {
    /// An enabled stage -- `mission_mode == "scan_map_inspect"`.
    pub fn new(config: MissionConfig) -> Self {
        Self {
            config,
            enabled: true,
            link: MissionLink::default(),
        }
    }

    /// A stage for a run with no mission machine, which the reference
    /// expresses by simply not entering the `if _mission_mode ==
    /// "scan_map_inspect"` block.
    ///
    /// Distinct from an enabled stage with no inputs: that is a wiring bug and
    /// is reported as one.
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            ..Self::new(MissionConfig::default())
        }
    }

    /// A handle to keep before installing the stage.
    pub fn link(&self) -> MissionLink {
        self.link.clone()
    }

    pub fn config(&self) -> &MissionConfig {
        &self.config
    }

    pub fn is_enabled(&self) -> bool {
        self.enabled
    }
}

impl StageHandler for MissionStage {
    fn run(&mut self, context: &mut TickContext<'_>) -> Result<StageOutcome, StageError> {
        let mut shared = self.link.lock();
        shared.events.clear();
        if !self.enabled {
            return Ok(StageOutcome::Skipped {
                reason: "mission machine disabled for this run".to_string(),
            });
        }
        let Some(inputs) = shared.pending.take() else {
            // Reporting NoWork here would let a run whose stage 8 was never
            // wired up look like a mission that simply had nothing to do.
            return Err(StageError::Invariant(
                "mission stage is enabled but received no inputs for this tick".to_string(),
            ));
        };
        if inputs.timestamp_s != context.timestamp_s {
            // `phase_started_at_s` comes from the inputs, so stale inputs would
            // date a transition to the wrong tick.
            return Err(StageError::Invariant(format!(
                "mission inputs are stamped {} but this tick is {}",
                inputs.timestamp_s, context.timestamp_s
            )));
        }

        let events = shared.state.advance(&self.config, &inputs);
        for event in &events {
            match event {
                MissionEvent::PhaseChanged { from, to, at_s } => {
                    context.note(format!("mission phase {from} -> {to} at {at_s}"));
                }
                MissionEvent::RescorePois { .. } => context.note("mission rescore pois"),
                MissionEvent::EgressRoutesRequested { .. } => {
                    context.note("mission egress routes requested");
                }
                MissionEvent::EgressRoutesCleared { .. } => {
                    context.note("mission egress routes cleared");
                }
                MissionEvent::LocalizationTimedOut { at_s } => {
                    context.note(format!("mission localization timed out at {at_s}"));
                }
            }
        }
        let outcome = if events.is_empty() {
            StageOutcome::NoWork
        } else {
            StageOutcome::Ran
        };
        shared.events = events;
        Ok(outcome)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fleet of `n` platforms, all at `confidence`, all far from home.
    fn inputs(t: f64, n: usize, confidence: f64) -> MissionInputs {
        MissionInputs {
            timestamp_s: t,
            coverage_fraction: 0.0,
            gap_cell_fraction: 0.0,
            mobile_count: n,
            localization_confidences: vec![confidence; n],
            localizer_timed_out: false,
            all_pois_complete: false,
            home_distances_m: vec![Some(1_000.0); n],
        }
    }

    /// Inputs that clear the scanning gates.
    fn scan_done(t: f64, n: usize, confidence: f64) -> MissionInputs {
        MissionInputs {
            coverage_fraction: 0.70,
            gap_cell_fraction: 0.0,
            ..inputs(t, n, confidence)
        }
    }

    fn phases(events: &[MissionEvent]) -> Vec<(MissionPhase, MissionPhase)> {
        events
            .iter()
            .filter_map(|e| match e {
                MissionEvent::PhaseChanged { from, to, .. } => Some((*from, *to)),
                _ => None,
            })
            .collect()
    }

    // -----------------------------------------------------------------
    // The happy path, and the one-transition-per-tick rule
    // -----------------------------------------------------------------

    #[test]
    fn a_run_walks_scanning_localizing_inspecting_egress_complete() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        assert_eq!(state.phase(), MissionPhase::Scanning);
        assert_eq!(state.phase_started_at_s(), 0.0);

        // Coverage reached, gaps clear, but localization has not converged.
        let events = state.advance(&cfg, &scan_done(10.0, 2, 0.30));
        assert_eq!(state.phase(), MissionPhase::Localizing);
        assert_eq!(state.phase_started_at_s(), 10.0);
        assert_eq!(
            phases(&events),
            vec![(MissionPhase::Scanning, MissionPhase::Localizing)]
        );

        let events = state.advance(&cfg, &inputs(20.0, 2, 0.65));
        assert_eq!(state.phase(), MissionPhase::Inspecting);
        assert!(events.contains(&MissionEvent::RescorePois { at_s: 20.0 }));

        let mut done = inputs(30.0, 2, 0.9);
        done.all_pois_complete = true;
        let events = state.advance(&cfg, &done);
        assert_eq!(state.phase(), MissionPhase::Egress);
        assert!(events.contains(&MissionEvent::EgressRoutesRequested { at_s: 30.0 }));

        let mut home = inputs(40.0, 2, 0.9);
        home.home_distances_m = vec![Some(1.0), Some(14.999)];
        let events = state.advance(&cfg, &home);
        assert_eq!(state.phase(), MissionPhase::Complete);
        assert!(state.is_complete());
        assert!(events.contains(&MissionEvent::EgressRoutesCleared { at_s: 40.0 }));
    }

    /// The `elif` chain tests the phase as it stood at the top of the tick, so
    /// a machine that cascaded would finish this mission three ticks early.
    #[test]
    fn at_most_one_transition_happens_per_tick() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();

        // Every gate in the whole mission is satisfied at once.
        let mut everything = scan_done(5.0, 1, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![Some(0.0)];

        let events = state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Inspecting);
        assert_eq!(phases(&events).len(), 1, "one step, not a cascade");

        state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Egress);
        state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Complete);
    }

    #[test]
    fn complete_is_terminal() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 1, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![Some(0.0)];
        for _ in 0..3 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Complete);
        let at = state.phase_started_at_s();
        // Keep feeding the inputs that *caused* completion, so a machine that
        // re-ran the egress test would transition again and be caught.
        for t in 0..10 {
            everything.timestamp_s = 100.0 + f64::from(t);
            assert!(state.advance(&cfg, &everything).is_empty());
            assert!(state.advance(&cfg, &scan_done(200.0, 1, 0.99)).is_empty());
        }
        assert_eq!(state.phase(), MissionPhase::Complete);
        assert_eq!(state.phase_started_at_s(), at, "the clock stops too");
    }

    /// Localization converging during scanning skips `Localizing` entirely.
    #[test]
    fn scanning_goes_straight_to_inspecting_when_localization_already_converged() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let events = state.advance(&cfg, &scan_done(7.0, 3, 0.65));
        assert_eq!(state.phase(), MissionPhase::Inspecting);
        assert_eq!(
            phases(&events),
            vec![(MissionPhase::Scanning, MissionPhase::Inspecting)]
        );
    }

    // -----------------------------------------------------------------
    // Gate directions
    // -----------------------------------------------------------------

    #[test]
    fn the_coverage_gate_is_inclusive_and_the_gap_gate_is_strict() {
        let cfg = MissionConfig::default();

        let mut just_under = MissionState::new();
        let mut i = scan_done(1.0, 1, 0.99);
        i.coverage_fraction = 0.699_999_999;
        just_under.advance(&cfg, &i);
        assert_eq!(just_under.phase(), MissionPhase::Scanning);

        let mut exactly = MissionState::new();
        i.coverage_fraction = 0.70;
        exactly.advance(&cfg, &i);
        assert_eq!(exactly.phase(), MissionPhase::Inspecting, "`>=`");

        // The gap gate is the only strict one: exactly at the fraction blocks.
        let mut at_gap = MissionState::new();
        i.gap_cell_fraction = cfg.gap_fill_min_fraction;
        at_gap.advance(&cfg, &i);
        assert_eq!(at_gap.phase(), MissionPhase::Scanning, "`<`");

        let mut under_gap = MissionState::new();
        i.gap_cell_fraction = cfg.gap_fill_min_fraction - 1e-12;
        under_gap.advance(&cfg, &i);
        assert_eq!(under_gap.phase(), MissionPhase::Inspecting);
    }

    #[test]
    fn every_platform_must_reach_the_confidence_threshold() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut i = scan_done(1.0, 3, 0.99);
        // One straggler, exactly one hair below.
        i.localization_confidences = vec![0.99, 0.649_999_999, 0.99];
        state.advance(&cfg, &i);
        assert_eq!(state.phase(), MissionPhase::Localizing);

        i.timestamp_s = 2.0;
        i.localization_confidences = vec![0.99, 0.65, 0.99];
        state.advance(&cfg, &i);
        assert_eq!(state.phase(), MissionPhase::Inspecting, "`>=`");
    }

    #[test]
    fn the_egress_arrival_radius_is_strict() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 1, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![Some(15.0)];
        for _ in 0..2 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Egress);

        state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Egress, "15.0 is not `< 15.0`");

        everything.home_distances_m = vec![Some(14.999_999)];
        everything.timestamp_s = 9.0;
        state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Complete);
    }

    #[test]
    fn one_platform_short_of_home_holds_the_whole_fleet_in_egress() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 3, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![Some(0.0), Some(0.0), Some(400.0)];
        for _ in 0..3 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Egress);
    }

    /// `Inspecting` holds until `POIManager.all_complete`. Nothing else leaves
    /// it, and there is no timeout.
    #[test]
    fn inspecting_holds_while_any_poi_is_pending() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        state.advance(&cfg, &scan_done(1.0, 2, 0.99));
        assert_eq!(state.phase(), MissionPhase::Inspecting);

        let mut pending = inputs(2.0, 2, 0.99);
        pending.all_pois_complete = false;
        pending.home_distances_m = vec![Some(0.0), Some(0.0)];
        for t in 0..40 {
            pending.timestamp_s = 2.0 + f64::from(t);
            assert!(state.advance(&cfg, &pending).is_empty());
        }
        assert_eq!(state.phase(), MissionPhase::Inspecting);

        pending.all_pois_complete = true;
        state.advance(&cfg, &pending);
        assert_eq!(state.phase(), MissionPhase::Egress);
    }

    /// The arrival count is compared against `mobile_count`, not against the
    /// length of the distance list. A short list means a platform is missing
    /// from it, and completing on that would report a fleet home when one of
    /// them was never looked at.
    #[test]
    fn a_short_home_distance_list_cannot_complete_egress() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 3, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![Some(0.0), Some(0.0), Some(0.0)];
        for _ in 0..2 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Egress);

        // Three platforms, two entries -- both arrived, still not enough.
        everything.home_distances_m = vec![Some(0.0), Some(0.0)];
        everything.timestamp_s = 20.0;
        assert!(state.advance(&cfg, &everything).is_empty());
        assert_eq!(state.phase(), MissionPhase::Egress);

        // The third turns up, and it completes.
        everything.home_distances_m = vec![Some(0.0), Some(0.0), Some(2.0)];
        everything.timestamp_s = 21.0;
        state.advance(&cfg, &everything);
        assert_eq!(state.phase(), MissionPhase::Complete);
    }

    // -----------------------------------------------------------------
    // The three asymmetries
    // -----------------------------------------------------------------

    /// Asymmetry 1. The same emptiness reads as "converged" in one branch and
    /// "not converged" in the other.
    #[test]
    fn an_empty_fleet_leaves_scanning_but_can_never_leave_localizing() {
        let cfg = MissionConfig::default();

        // From scanning: `(not mobile_states) or all(...)` -> converged.
        let mut from_scanning = MissionState::new();
        from_scanning.advance(&cfg, &scan_done(1.0, 0, 0.0));
        assert_eq!(from_scanning.phase(), MissionPhase::Inspecting);

        // Reach Localizing with a fleet, then lose it.
        let mut stuck = MissionState::new();
        stuck.advance(&cfg, &scan_done(1.0, 2, 0.10));
        assert_eq!(stuck.phase(), MissionPhase::Localizing);
        for t in 0..50 {
            // Empty fleet, and nothing left that could fail to converge.
            let events = stuck.advance(&cfg, &inputs(10.0 + f64::from(t), 0, 0.0));
            assert!(events.is_empty());
        }
        assert_eq!(
            stuck.phase(),
            MissionPhase::Localizing,
            "`if mobile_states and all(...)` is never satisfied by an empty fleet"
        );
    }

    /// Asymmetry 2. A platform with no recorded home is treated as home.
    #[test]
    fn a_platform_with_no_home_counts_as_arrived() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 2, 0.99);
        everything.all_pois_complete = true;
        everything.home_distances_m = vec![None, Some(3.0)];
        for _ in 0..3 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Complete);
    }

    /// ...but only for a non-empty fleet. An empty one cannot arrive.
    #[test]
    fn an_empty_fleet_never_completes_egress() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let mut everything = scan_done(1.0, 1, 0.99);
        everything.all_pois_complete = true;
        for _ in 0..2 {
            state.advance(&cfg, &everything);
        }
        assert_eq!(state.phase(), MissionPhase::Egress);

        let mut gone = inputs(50.0, 0, 0.0);
        gone.all_pois_complete = true;
        for _ in 0..20 {
            assert!(state.advance(&cfg, &gone).is_empty());
        }
        assert_eq!(state.phase(), MissionPhase::Egress);
    }

    /// Asymmetry 3. The timeout is raised from the localization stage, which
    /// runs in every phase, and it never clears.
    #[test]
    fn the_localization_timeout_latches_in_any_phase_and_never_clears() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        assert!(!state.localization_timed_out());

        let mut timed_out = inputs(3.0, 2, 0.99);
        timed_out.localizer_timed_out = true;
        let events = state.advance(&cfg, &timed_out);
        assert_eq!(state.phase(), MissionPhase::Scanning, "still scanning");
        assert!(state.localization_timed_out());
        assert_eq!(
            events,
            vec![MissionEvent::LocalizationTimedOut { at_s: 3.0 }]
        );

        // Emitted once, not every tick.
        assert!(state.advance(&cfg, &timed_out).is_empty());

        // And it does not clear when the localizer recovers.
        state.advance(&cfg, &inputs(4.0, 2, 0.99));
        assert!(state.localization_timed_out());

        // A run that skips Localizing entirely still reports it.
        state.advance(&cfg, &scan_done(5.0, 2, 0.99));
        assert_eq!(state.phase(), MissionPhase::Inspecting);
        assert!(state.localization_timed_out());
    }

    // -----------------------------------------------------------------
    // The rescore latch
    // -----------------------------------------------------------------

    /// `_poi_rescored` is set once and never cleared, so a second entry into
    /// `Inspecting` does not rescore. Nothing in the reference re-enters
    /// today, which is exactly why the latch has to be preserved rather than
    /// reasoned away.
    #[test]
    fn pois_are_rescored_once_per_run_not_once_per_entry() {
        let cfg = MissionConfig::default();
        let mut state = MissionState::new();
        let events = state.advance(&cfg, &scan_done(2.0, 1, 0.99));
        assert!(events.contains(&MissionEvent::RescorePois { at_s: 2.0 }));
        assert!(state.pois_rescored());

        // No phase path re-enters Inspecting today, so a second entry has to
        // be constructed. Doing it here rather than declaring the latch
        // untestable: if a future phase ever loops back, this is the
        // assertion that says what should happen.
        let mut re_entering = MissionState {
            phase: MissionPhase::Localizing,
            phase_started_at_s: 0.0,
            localization_timed_out: false,
            pois_rescored: true,
        };
        let events = re_entering.advance(&cfg, &inputs(9.0, 1, 0.99));
        assert_eq!(re_entering.phase(), MissionPhase::Inspecting);
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, MissionEvent::RescorePois { .. })),
            "the latch is never cleared, so a second entry does not rescore"
        );

        // And the latch outlives the phase it was set in.
        let mut done = inputs(3.0, 1, 0.99);
        done.all_pois_complete = true;
        state.advance(&cfg, &done);
        assert_eq!(state.phase(), MissionPhase::Egress);
        assert!(state.pois_rescored());
    }

    // -----------------------------------------------------------------
    // Serialization, since the phase string reaches replay
    // -----------------------------------------------------------------

    #[test]
    fn the_phase_strings_are_the_ones_replay_carries() {
        for (phase, name) in [
            (MissionPhase::Scanning, "scanning"),
            (MissionPhase::Localizing, "localizing"),
            (MissionPhase::Inspecting, "inspecting"),
            (MissionPhase::Egress, "egress"),
            (MissionPhase::Complete, "complete"),
        ] {
            assert_eq!(phase.as_str(), name);
            assert_eq!(phase.to_string(), name);
            assert_eq!(
                serde_json::to_string(&phase).unwrap(),
                format!("\"{name}\"")
            );
        }
    }

    #[test]
    fn a_custom_config_moves_the_gates() {
        let cfg = MissionConfig {
            scan_coverage_threshold: 0.95,
            localization_confidence_threshold: 0.10,
            gap_fill_min_fraction: 0.50,
            egress_arrival_radius_m: 1.0,
        };
        let mut state = MissionState::new();
        let mut i = scan_done(1.0, 1, 0.20);
        i.gap_cell_fraction = 0.40;
        state.advance(&cfg, &i);
        assert_eq!(state.phase(), MissionPhase::Scanning, "0.70 < 0.95");

        i.coverage_fraction = 0.95;
        i.timestamp_s = 2.0;
        state.advance(&cfg, &i);
        // 0.40 < 0.50 passes the gap gate, and 0.20 >= 0.10 converges.
        assert_eq!(state.phase(), MissionPhase::Inspecting);
    }
}
