//! The in-process simulation, as the viewer drives it (D1.2, P5.0).
//!
//! `argusnet-viewer --scene <scene> --run <config>` starts an
//! [`argusnet_sim::worker::SimWorker`]: the run lives on its own thread and the
//! render loop never ticks it. This module is the viewer's side of that:
//!
//! - **Frames.** Every committed tick arrives in order and is appended to the
//!   [`ReplayState`], exactly as a live stream's frames are, so the whole
//!   presentation (markers, trails, timeline, inspectors) works unchanged. The
//!   cursor follows the newest frame unless the operator is reviewing history.
//!   A reset is a new run, so it begins a new presentation epoch.
//! - **Commands.** Every control sends the existing `CommandEnvelope` and is
//!   shown as **pending** until the run answers that it took effect (Applied)
//!   or did not (Rejected or Cancelled, with the reason). The viewer never
//!   infers success from motion.
//! - **Controls offered are controls that work.** Rewind needs checkpoints the
//!   worker does not retain, so it is shown disabled with that reason rather
//!   than hidden or faked.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use argusnet_sim::builder::{build_bundle, ScenarioConfig};
use argusnet_sim::command::{AckStatus, CommandAck, CommandEnvelope, CommandKind, RejectReason};
use argusnet_sim::run_options::RunOptions;
use argusnet_sim::scenario::{assemble, AssembledRun, ScenarioBundle};
use argusnet_sim::worker::{Pacing, SimWorker, WorkerPhase, WorkerStatus};
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::replay::{ReplayDocument, ReplayFrame, ReplayState};
use crate::ui::SimRunState;

/// Speeds the control strip offers. Speed paces wall-clock time only; the
/// physics timestep never changes.
pub const SPEEDS: [f64; 5] = [0.5, 1.0, 2.0, 4.0, 10.0];

/// How long a refusal stays on the strip.
const NOTICE_FOR: Duration = Duration::from_secs(6);

/// Build a run from a scenario config (`argusnet.scenario-config`) or a
/// scenario bundle (`argusnet.scenario-bundle`), whichever the file is.
/// Assemble the run at `path` and install `options` on it -- through the same
/// [`RunOptions::apply`] as `argusnet-rs sim`, so the viewer's run is the CLI's.
pub fn load_run(path: &Path, options: &RunOptions) -> Result<AssembledRun, String> {
    let text =
        std::fs::read_to_string(path).map_err(|error| format!("{}: {error}", path.display()))?;
    let value: serde_json::Value =
        serde_json::from_str(&text).map_err(|error| format!("{}: {error}", path.display()))?;
    let bundle_text = match value.get("format").and_then(|f| f.as_str()) {
        Some("argusnet.scenario-config") => {
            let config = ScenarioConfig::from_json(&text).map_err(|error| error.to_string())?;
            build_bundle(&config)
                .map_err(|error| error.to_string())?
                .to_string()
        }
        Some("argusnet.scenario-bundle") => text,
        other => {
            return Err(format!(
                "{}: expected an argusnet.scenario-config or argusnet.scenario-bundle, found {}",
                path.display(),
                other.unwrap_or("no format")
            ))
        }
    };
    let bundle = ScenarioBundle::from_json(&bundle_text).map_err(|error| error.to_string())?;
    let mut run = assemble(&bundle).map_err(|error| error.to_string())?;
    options.apply(&mut run, &bundle)?;
    Ok(run)
}

/// A sim frame in the viewer's replay representation. Both sides speak the
/// replay schema; the round trip is the contract they share.
pub fn to_viewer_frame(frame: &argusnet_sim::replay::ReplayFrame) -> Result<ReplayFrame, String> {
    let value = serde_json::to_value(frame).map_err(|error| error.to_string())?;
    serde_json::from_value(value).map_err(|error| error.to_string())
}

/// What one control is waiting on.
#[derive(Debug, Clone)]
pub struct PendingCommand {
    pub label: &'static str,
    pub sent: Instant,
}

/// The viewer's handle on the in-process run.
#[derive(Resource)]
pub struct SimSession {
    worker: SimWorker,
    pub source: PathBuf,
    next_id: u64,
    pending: BTreeMap<u64, PendingCommand>,
    /// The last refusal, shown for a few seconds.
    notice: Option<(String, Instant)>,
    /// The worker epoch the replay's frames belong to.
    epoch: Option<u64>,
    /// The scene's replay metadata, carried into each run's document.
    meta: Option<ReplayDocument>,
    /// What the run's options record in its meta (its world knowledge and
    /// belief grid, its delivery profile, ...), merged over the scene's.
    run_meta: serde_json::Map<String, serde_json::Value>,
    pub frames_received: u64,
    pub conversion_errors: u64,
}

impl SimSession {
    pub fn start(source: PathBuf, options: RunOptions, pacing: Pacing) -> Self {
        let path = source.clone();
        let mission_rate = options.mission_rate;
        Self {
            worker: SimWorker::spawn_with_rate(
                move || load_run(&path, &options),
                pacing,
                mission_rate,
            ),
            source,
            next_id: 1,
            pending: BTreeMap::new(),
            notice: None,
            epoch: None,
            meta: None,
            run_meta: serde_json::Map::new(),
            frames_received: 0,
            conversion_errors: 0,
        }
    }

    /// Carry the run's own meta keys into the documents the session builds.
    pub fn with_run_meta(
        mut self,
        meta: impl IntoIterator<Item = (String, serde_json::Value)>,
    ) -> Self {
        self.run_meta = meta.into_iter().collect();
        self
    }

    pub fn status(&self) -> WorkerStatus {
        self.worker.status()
    }

    pub fn pending(&self) -> impl Iterator<Item = &PendingCommand> {
        self.pending.values()
    }

    pub fn is_pending(&self, label: &str) -> bool {
        self.pending.values().any(|p| p.label == label)
    }

    pub fn notice(&self) -> Option<&str> {
        self.notice
            .as_ref()
            .filter(|(_, at)| at.elapsed() < NOTICE_FOR)
            .map(|(text, _)| text.as_str())
    }

    /// Send a control. Returns its id, or `None` when it was refused at once
    /// (the reason goes to the notice).
    pub fn send(&mut self, label: &'static str, payload: CommandKind) -> Option<u64> {
        let status = self.worker.status();
        let Some(run_id) = status.run_id else {
            let why = match status.phase {
                WorkerPhase::Failed(_) => "The world failed to load; nothing to control.",
                _ => "The run is still loading.",
            };
            self.notice = Some((why.into(), Instant::now()));
            return None;
        };
        let id = self.next_id;
        self.next_id += 1;
        match self
            .worker
            .submit(CommandEnvelope::new(id, run_id, status.epoch, payload))
        {
            Ok(()) => {
                self.pending.insert(
                    id,
                    PendingCommand {
                        label,
                        sent: Instant::now(),
                    },
                );
                Some(id)
            }
            Err(ack) => {
                self.refused(label, &ack);
                None
            }
        }
    }

    fn refused(&mut self, label: &str, ack: &CommandAck) {
        let reason = match &ack.status {
            AckStatus::Rejected(reason) | AckStatus::Cancelled(reason) => describe(reason),
            other => format!("{other:?}"),
        };
        self.notice = Some((format!("{label}: {reason}"), Instant::now()));
    }

    /// Take in acknowledgements: a command stops being pending when it took
    /// effect or was refused, and a refusal is shown with its reason.
    pub fn take_acks(&mut self) {
        let acks = self.worker.acks();
        for ack in acks {
            let Some(pending) = self.pending.get(&ack.command_id) else {
                continue;
            };
            match &ack.status {
                AckStatus::Accepted => {}
                AckStatus::Applied | AckStatus::Duplicate => {
                    self.pending.remove(&ack.command_id);
                }
                AckStatus::Rejected(_) | AckStatus::Cancelled(_) => {
                    let label = pending.label;
                    self.pending.remove(&ack.command_id);
                    self.refused(label, &ack);
                }
            }
        }
    }

    /// Append every new frame to `replay`. Returns how many were appended.
    pub fn take_frames(&mut self, replay: &mut ReplayState) -> u64 {
        if self.meta.is_none() {
            // The scene's metadata (ids, zones, terrain summary) describes the
            // world the run happens in; its recorded frames do not.
            let mut meta = replay
                .document
                .as_ref()
                .and_then(|d| d.meta.clone())
                .unwrap_or_else(|| serde_json::Value::Object(serde_json::Map::new()));
            if let Some(object) = meta.as_object_mut() {
                for (key, value) in &self.run_meta {
                    object.insert(key.clone(), value.clone());
                }
            }
            self.meta = Some(ReplayDocument {
                meta: Some(meta),
                summary: None,
                frames: Vec::new(),
            });
            replay.begin_epoch(self.meta.clone());
            replay.live = true;
            replay.playing = false;
        }
        let mut appended = 0_u64;
        let frames = self.worker.frames();
        for worker_frame in frames {
            if self.epoch != Some(worker_frame.epoch) {
                if self.epoch.is_some() {
                    replay.begin_epoch(self.meta.clone());
                    replay.live = true;
                }
                self.epoch = Some(worker_frame.epoch);
            }
            match to_viewer_frame(&worker_frame.frame) {
                Ok(frame) => {
                    replay.push_frame(frame);
                    appended += 1;
                }
                Err(_) => self.conversion_errors += 1,
            }
        }
        // A reset with no frame yet: show the empty new run now, not after
        // its first tick.
        let status = self.worker.status();
        if self.epoch.is_some_and(|epoch| epoch != status.epoch) && status.run_id.is_some() {
            replay.begin_epoch(self.meta.clone());
            replay.live = true;
            self.epoch = Some(status.epoch);
        }
        if appended > 0 {
            self.frames_received += appended;
            let frame_count = replay.frame_count();
            let was_at_tail = frame_count as u64 <= appended
                || replay.frame_index + (appended as usize) + 1 >= frame_count;
            if was_at_tail {
                replay.step_to(frame_count.saturating_sub(1));
            }
        }
        appended
    }

    /// The source indicator's state.
    pub fn run_state(&self) -> SimRunState {
        match self.worker.status().phase {
            WorkerPhase::Running => SimRunState::Running,
            WorkerPhase::Faulted(_) | WorkerPhase::Failed(_) => SimRunState::Faulted,
            _ => SimRunState::Paused,
        }
    }
}

/// A refusal in the operator's words.
pub fn describe(reason: &RejectReason) -> String {
    match reason {
        RejectReason::Busy { .. } => "the run is busy; try again".into(),
        RejectReason::StaleEpoch { .. } | RejectReason::StaleRevision { .. } => {
            "the run was reset before this took effect".into()
        }
        RejectReason::StaleRun { .. } => "addressed to a different run".into(),
        RejectReason::Completed => "the run has finished; reset to run it again".into(),
        RejectReason::Faulted => "the run faulted; reset it".into(),
        RejectReason::InvalidSpeed => "not a valid speed".into(),
        RejectReason::RewindUnavailable { .. } => "rewind needs retained checkpoints".into(),
        RejectReason::Unsupported { name } => format!("not supported ({name})"),
        other => format!("{other:?}"),
    }
}

/// What the strip's main button does in each phase.
pub fn primary_action(phase: &WorkerPhase) -> Option<(&'static str, CommandKind)> {
    match phase {
        WorkerPhase::Paused => Some(("Run", CommandKind::Resume)),
        WorkerPhase::Running => Some(("Pause", CommandKind::Pause)),
        _ => None,
    }
}

/// The source indicator's input, kept current for the shell's header.
#[derive(Resource, Default)]
pub struct SimSourceState(pub Option<SimRunState>);

pub fn ingest_sim_frames_system(
    mut session: ResMut<SimSession>,
    mut replay: ResMut<ReplayState>,
    mut source: ResMut<SimSourceState>,
) {
    session.take_frames(&mut replay);
    session.take_acks();
    source.0 = Some(session.run_state());
}

/// The run's control strip, over the top-left of the world canvas.
pub fn sim_controls_ui_system(
    mut contexts: EguiContexts,
    mut session: ResMut<SimSession>,
    ui_state: Res<crate::state::ViewerUiState>,
) {
    let Ok(ctx) = contexts.ctx_mut() else {
        return;
    };
    // The theme's faces bind a pass after they are installed; drawing with
    // them before that panics in epaint.
    if !crate::theme::fonts_ready(ctx) {
        return;
    }
    let status = session.status();
    egui::Area::new(egui::Id::new("argusnet_sim_controls"))
        .fixed_pos(egui::pos2(
            ui_state.canvas_origin()[0] + 12.0,
            ui_state.canvas_origin()[1] + 44.0,
        ))
        .show(ctx, |ui| {
            egui::Frame::popup(ui.style()).show(ui, |ui| {
                ui.horizontal(|ui| {
                    let (glyph, color, text) = phase_badge(&status.phase);
                    ui.label(
                        crate::theme::TypeRole::Utility
                            .text(format!("{glyph} SIM / {text}"))
                            .color(color)
                            .strong(),
                    );
                    ui.separator();
                    ui.monospace(format!(
                        "tick {} · {:.1} s · epoch {}",
                        status.tick,
                        status.tick as f64 * status.dt_s,
                        status.epoch
                    ));
                });
                ui.horizontal(|ui| {
                    let ready = status.run_id.is_some();
                    match primary_action(&status.phase) {
                        Some((label, payload)) => {
                            let pending = session.is_pending("Run") || session.is_pending("Pause");
                            if ui
                                .add_enabled(!pending, egui::Button::new(label))
                                .on_disabled_hover_text("Waiting for the run to acknowledge")
                                .clicked()
                            {
                                session.send(label, payload);
                            }
                        }
                        None => {
                            ui.add_enabled(false, egui::Button::new("Run"))
                                .on_disabled_hover_text(match &status.phase {
                                    WorkerPhase::Loading => "The world is still loading",
                                    WorkerPhase::Completed => {
                                        "The run has finished; reset to run it again"
                                    }
                                    _ => "The run cannot continue; reset it",
                                });
                        }
                    }
                    let can_step = status.phase == WorkerPhase::Paused;
                    if ui
                        .add_enabled(
                            can_step && !session.is_pending("Step"),
                            egui::Button::new("Step"),
                        )
                        .on_hover_text("Run exactly one tick, then pause")
                        .on_disabled_hover_text("Pause the run to step it")
                        .clicked()
                    {
                        session.send("Step", CommandKind::SingleStep);
                    }
                    ui.separator();
                    ui.label("Speed");
                    for speed in SPEEDS {
                        let selected = (status.speed - speed).abs() < 1e-9;
                        if ui
                            .add_enabled(
                                ready && !session.is_pending("Speed"),
                                egui::Button::new(format!("{speed}×")).selected(selected),
                            )
                            .clicked()
                            && !selected
                        {
                            session.send("Speed", CommandKind::SetSpeed { multiplier: speed });
                        }
                    }
                    ui.separator();
                    if ui
                        .add_enabled(
                            ready && !session.is_pending("Reset"),
                            egui::Button::new("Reset"),
                        )
                        .on_hover_text("Start the run again from tick 0, paused")
                        .clicked()
                    {
                        session.send("Reset", CommandKind::Reset);
                    }
                    ui.add_enabled(false, egui::Button::new("Rewind"))
                        .on_disabled_hover_text(
                            "Rewinding the run needs retained checkpoints, which this build does \
                             not keep. Drag the timeline to review history instead.",
                        );
                });
                let pending: Vec<_> = session.pending().map(|p| p.label).collect();
                if !pending.is_empty() {
                    ui.colored_label(
                        egui::Color32::from_rgb(230, 180, 70),
                        format!("PENDING ACK: {}", pending.join(", ")),
                    );
                }
                if let WorkerPhase::Failed(reason) | WorkerPhase::Faulted(reason) = &status.phase {
                    ui.colored_label(egui::Color32::from_rgb(235, 90, 80), reason);
                }
                if let Some(notice) = session.notice() {
                    ui.colored_label(egui::Color32::from_rgb(235, 150, 70), notice);
                }
            });
        });
}

fn phase_badge(phase: &WorkerPhase) -> (char, egui::Color32, &'static str) {
    use crate::theme::Status;
    let (status, text) = match phase {
        WorkerPhase::Loading => (Status::Pending, "LOADING"),
        WorkerPhase::Paused => (Status::Pending, "PAUSED"),
        WorkerPhase::Running => (Status::Nominal, "RUNNING"),
        WorkerPhase::Completed => (Status::Stale, "COMPLETED"),
        WorkerPhase::Faulted(_) => (Status::Critical, "FAULTED"),
        WorkerPhase::Failed(_) => (Status::Critical, "FAILED TO LOAD"),
        WorkerPhase::Stopped => (Status::Unavailable, "STOPPED"),
    };
    (status.glyph(), status.color(), text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load_run_legacy(path: &Path) -> Result<AssembledRun, String> {
        load_run(path, &RunOptions::default())
    }

    fn fixture(name: &str) -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures")
            .join(name)
    }

    fn wait(
        session: &mut SimSession,
        replay: &mut ReplayState,
        what: impl Fn(&SimSession, &ReplayState) -> bool,
    ) {
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            session.take_frames(replay);
            session.take_acks();
            if what(session, replay) {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "timed out: {:?}",
                session.status()
            );
            std::thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn every_sim_frame_converts_to_the_viewer_representation() {
        let mut run = load_run_legacy(&fixture("scenario_bundle_tt.json")).expect("loads");
        for _ in 0..5 {
            run.step().expect("ticks");
            let frame = run.last_frame().expect("a frame");
            let converted = to_viewer_frame(frame).expect("converts");
            assert_eq!(converted.nodes.len(), frame.nodes.len());
            assert_eq!(converted.tracks.len(), frame.tracks.len());
        }
    }

    #[test]
    fn a_scenario_config_loads_too() {
        let run = load_run_legacy(&fixture("scenario_config_scan_16_drones.json")).expect("loads");
        assert!(!run.run_id().is_empty());
        let Err(error) =
            load_run_legacy(&fixture("scenario_bundle_tt.json").with_extension("missing"))
        else {
            panic!("a missing file is an error");
        };
        assert!(error.contains("scenario_bundle_tt.missing"));
    }

    /// The session's frames are the run's, in order, and a control stays
    /// pending until the run says it took effect.
    #[test]
    fn controls_are_pending_until_applied_and_frames_follow_the_tail() {
        let mut session = SimSession::start(
            fixture("scenario_bundle_tt.json"),
            RunOptions::default(),
            Pacing::Unpaced,
        );
        let mut replay = ReplayState::new(None);
        wait(&mut session, &mut replay, |s, _| {
            s.status().phase == WorkerPhase::Paused
        });
        let id = session.send("Step", CommandKind::SingleStep).expect("sent");
        assert!(session.is_pending("Step"));
        wait(&mut session, &mut replay, |s, _| !s.is_pending("Step"));
        assert!(session.notice().is_none());
        assert_eq!(replay.frame_count(), 1);
        assert!(replay.live);
        assert!(id > 0);

        session.send("Run", CommandKind::Resume).expect("sent");
        wait(&mut session, &mut replay, |s, _| {
            s.status().phase == WorkerPhase::Completed
        });
        session.take_frames(&mut replay);
        assert_eq!(
            replay.frame_count(),
            80,
            "every tick of the 80-step fixture"
        );
        assert_eq!(
            replay.frame_index, 79,
            "the cursor followed the newest frame"
        );
        assert_eq!(session.conversion_errors, 0);

        // Finished: Run is refused with the reason, not silently ignored.
        session.send("Run", CommandKind::Resume).expect("sent");
        wait(&mut session, &mut replay, |s, _| !s.is_pending("Run"));
        assert!(
            session.notice().is_some_and(|n| n.contains("finished")),
            "{:?}",
            session.notice()
        );

        // Reset: a new presentation epoch, empty, at tick 0.
        let epoch = replay.epoch();
        session.send("Reset", CommandKind::Reset).expect("sent");
        wait(&mut session, &mut replay, |s, r| {
            !s.is_pending("Reset") && r.epoch() > epoch
        });
        assert_eq!(replay.frame_count(), 0);
        assert_eq!(session.status().tick, 0);
        assert_eq!(session.run_state(), SimRunState::Paused);
    }

    #[test]
    fn a_run_that_cannot_load_is_reported_as_faulted() {
        let mut session = SimSession::start(
            fixture("no_such_config.json"),
            RunOptions::default(),
            Pacing::Unpaced,
        );
        let mut replay = ReplayState::new(None);
        wait(&mut session, &mut replay, |s, _| {
            matches!(s.status().phase, WorkerPhase::Failed(_))
        });
        assert_eq!(session.run_state(), SimRunState::Faulted);
        assert!(session.send("Run", CommandKind::Resume).is_none());
        assert!(session
            .notice()
            .is_some_and(|n| n.contains("failed to load")));
    }

    fn wave3_options() -> RunOptions {
        let mut delivery = argusnet_sim::run_options::DeliverySeconds::with_lag(1.5);
        delivery.delay_s = 0.5;
        delivery.jitter_s = 0.5;
        RunOptions {
            energy_profile: argusnet_sim::run_options::EnergyProfile::PhysicalV1,
            delivery: Some(delivery),
            faults: Some(
                RunOptions::read_faults(&fixture("fault_manifest_example.json")).expect("manifest"),
            ),
            ..RunOptions::default()
        }
    }

    fn frames_of(run: AssembledRun) -> Vec<String> {
        run.run_to_replay()
            .expect("runs")
            .frames
            .iter()
            .map(|frame| serde_json::to_string(frame).unwrap())
            .collect()
    }

    /// `--run` with options is the CLI's run with the same options: both go
    /// through `RunOptions::apply`, and the replay records what was applied.
    #[test]
    fn run_options_reach_the_viewers_run() {
        let options = wave3_options();
        let path = fixture("scenario_bundle_tt.json");
        let viewer = load_run(&path, &options).expect("loads with options");
        let replay = viewer.run_to_replay().expect("runs");
        let meta = serde_json::to_value(&replay.meta).unwrap();
        assert_eq!(meta["energy_profile"], "physical_v1");
        assert_eq!(meta["delivery_profile"]["fusion_lag_ticks"], 3);
        assert!(meta["fault_manifest"].is_object());
        assert!(meta["delivery_totals"]["captured"].as_u64().unwrap() > 0);

        let bundle = ScenarioBundle::from_json(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let mut direct = assemble(&bundle).unwrap();
        options.apply(&mut direct, &bundle).expect("applies");
        let viewer_frames: Vec<String> = replay
            .frames
            .iter()
            .map(|frame| serde_json::to_string(frame).unwrap())
            .collect();
        assert_eq!(viewer_frames, frames_of(direct));
        assert_ne!(
            viewer_frames,
            frames_of(load_run_legacy(&path).unwrap()),
            "the options changed the run"
        );
    }

    /// `--mission-rate` reaches the viewer's run as the CLI's: declared on
    /// the run and recorded in its replay (the session's worker steps it at
    /// that rate; `worker::tests` compares the frames with the CLI's).
    #[test]
    fn a_mission_rate_is_declared_and_recorded() {
        let options = RunOptions {
            mission_rate: Some(argusnet_sim::scheduler::parse_rate("1/7").unwrap()),
            ..RunOptions::default()
        };
        assert!(!options.is_legacy());
        let run = load_run(&fixture("scenario_bundle_scan.json"), &options).expect("loads");
        assert_eq!(
            run.sim
                .lane_rates()
                .get(&argusnet_sim::stage::Stage::EvaluateMissionTransitions)
                .map(String::as_str),
            Some("1/7")
        );
        let meta = run.replay_extra_meta();
        assert_eq!(meta["execution_model"], "scheduled_v1");
        assert_eq!(meta["lane_rates"]["mission"], "1/7 Hz");
    }

    #[test]
    fn an_unsupported_option_is_refused_at_load() {
        let mut options = RunOptions::default();
        let mut delivery = argusnet_sim::run_options::DeliverySeconds::with_lag(0.5);
        delivery.delay_s = 2.0;
        options.delivery = Some(delivery);
        let Err(error) = load_run(&fixture("scenario_bundle_tt.json"), &options) else {
            panic!("a delay beyond the fusion lag is refused");
        };
        assert!(error.contains("unsupported delivery profile"), "{error}");
    }

    /// Reset keeps the options: the run after a reset is the first run again,
    /// delivery, energy profile and faults included.
    #[test]
    fn a_reset_run_keeps_its_options() {
        let mut session = SimSession::start(
            fixture("scenario_bundle_tt.json"),
            wave3_options(),
            Pacing::Unpaced,
        );
        let mut replay = ReplayState::new(None);
        let run_once = |session: &mut SimSession, replay: &mut ReplayState| {
            wait(session, replay, |s, _| {
                s.status().phase == WorkerPhase::Paused
            });
            session.send("Run", CommandKind::Resume).expect("sent");
            wait(session, replay, |s, _| {
                s.status().phase == WorkerPhase::Completed
            });
            session.take_frames(replay);
            replay
                .document
                .as_ref()
                .map(|document| {
                    document
                        .frames
                        .iter()
                        .map(|frame| format!("{:?}", frame.tracks))
                        .collect::<Vec<_>>()
                })
                .unwrap_or_default()
        };
        let first = run_once(&mut session, &mut replay);
        let epoch = replay.epoch();
        session.send("Reset", CommandKind::Reset).expect("sent");
        wait(&mut session, &mut replay, |s, r| {
            !s.is_pending("Reset") && r.epoch() > epoch
        });
        let second = run_once(&mut session, &mut replay);
        assert_eq!(first.len(), 80);
        assert_eq!(first, second);
    }
}
