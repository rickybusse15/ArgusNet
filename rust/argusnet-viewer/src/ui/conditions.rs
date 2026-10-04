//! Failure, loading and degraded states (C7.1, VIEWER_UI §12). Each state is
//! a typed [`Condition`] with one presentation: how severe it is, what is
//! stale or preserved, and a bounded next action. They show on the canvas,
//! most severe first, over the last coherent view: a condition never empties
//! the world, zeroes the terrain or the zones, or reads as confident state.
//!
//! [`detect`] finds the conditions the viewer can observe today (no source,
//! a live link reconnecting or dropping frames, belief unavailable under its
//! overlay, a failed simulation, a replay schema this viewer does not
//! support). The rest (loading and verifying a world, commands, recording,
//! simulation pace, transport security) have their presentations now and
//! their inputs when the runtime integration (D1–D4) delivers them.

use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::state::{RuntimeOverlayVisibility, ViewerMode, ViewerUiState};
use crate::theme::Status;

/// What a command is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CommandState {
    Pending,
    Busy,
    Rejected(String),
    Stale,
}

/// What a recording is doing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecordingState {
    Backpressure,
    Paused,
    Incomplete,
    DiskFailure(String),
}

/// Which transport check failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityCheck {
    Tls,
    Authentication,
    Capability,
}

/// A failure, loading or degraded state (§12's list, in its order).
#[derive(Debug, Clone, PartialEq)]
pub enum Condition {
    NoSource,
    LoadingManifest,
    VerifyingWorld {
        revision: String,
    },
    WorldUnavailable {
        reason: String,
    },
    WorldIncompatible {
        reason: String,
    },
    BeliefUnavailable {
        reason: String,
    },
    BeliefStale {
        age_s: f32,
    },
    BeliefAwaitingBase {
        revision: u64,
    },
    Reconnecting {
        endpoint: String,
        last_sequence: u64,
    },
    LiveDrops {
        dropped: u64,
    },
    SimPaused,
    SimSlow {
        ratio: f32,
    },
    Command {
        command: String,
        state: CommandState,
    },
    Recording(RecordingState),
    SimFault {
        reason: String,
    },
    Security {
        check: SecurityCheck,
        reason: String,
    },
    UnsupportedVersion {
        what: &'static str,
        found: String,
        supported: &'static str,
    },
}

/// A next action the banner can take itself. Every other action is text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionAction {
    HideBeliefOverlay,
    OpenScenarioPanel,
}

/// How a condition is shown.
#[derive(Debug, Clone, PartialEq)]
pub struct Presentation {
    pub status: Status,
    pub title: String,
    /// What is stale, or what the view still shows and why it can be trusted.
    pub preserved: String,
    /// The bounded next action, in words.
    pub next: String,
    /// The same action as a button, where the viewer can take it.
    pub action: Option<ConditionAction>,
}

/// Most severe first.
fn rank(status: Status) -> u8 {
    match status {
        Status::Critical => 0,
        Status::Degraded => 1,
        Status::Stale => 2,
        Status::Unavailable => 3,
        Status::Pending => 4,
        Status::Unknown => 5,
        Status::Nominal => 6,
    }
}

impl Condition {
    pub fn present(&self) -> Presentation {
        let p = |status, title: String, preserved: &str, next: String| Presentation {
            status,
            title,
            preserved: preserved.to_string(),
            next,
            action: None,
        };
        match self {
            Condition::NoSource => p(
                Status::Unavailable,
                "No source".into(),
                "Nothing is presented: no replay, live link or simulation is open.",
                "Open a scene package with a replay, or run a simulation from Tools → Scenario."
                    .into(),
            ),
            Condition::LoadingManifest => p(
                Status::Pending,
                "Loading scene manifest".into(),
                "The current scene stays shown until the new one is complete.",
                "Wait for the manifest to load.".into(),
            ),
            Condition::VerifyingWorld { revision } => p(
                Status::Pending,
                format!("Verifying world revision {revision}"),
                "The current world stays shown until the new revision verifies.",
                "Wait for verification.".into(),
            ),
            Condition::WorldUnavailable { reason } => p(
                Status::Degraded,
                format!("World unavailable: {reason}"),
                "No terrain is invented: heights fall back to each entity's own altitude.",
                "Open a package that carries its world revision.".into(),
            ),
            Condition::WorldIncompatible { reason } => p(
                Status::Critical,
                format!("World incompatible: {reason}"),
                "The incompatible world is not drawn; the last compatible one stays shown.",
                "Open a package built for this viewer.".into(),
            ),
            Condition::BeliefUnavailable { reason } => Presentation {
                action: Some(ConditionAction::HideBeliefOverlay),
                ..p(
                    Status::Unavailable,
                    format!("Belief unavailable: {reason}"),
                    "The overlay draws everywhere as unknown (hatched), never as a guess.",
                    "Hide the belief overlay, or open a source that carries belief.".into(),
                )
            },
            Condition::BeliefStale { age_s } => p(
                Status::Stale,
                format!("Belief stale: last revision {age_s:.0} s ago"),
                "Cells observed before then are drawn faded toward grey.",
                "Wait for the next revision, or check the mapping source.".into(),
            ),
            Condition::BeliefAwaitingBase { revision } => p(
                Status::Pending,
                format!("Belief awaiting base revision {revision}"),
                "Patches are held until their base arrives; none is drawn out of order.",
                "Wait for the base revision.".into(),
            ),
            Condition::Reconnecting {
                endpoint,
                last_sequence,
            } => p(
                Status::Degraded,
                format!("Reconnecting to {endpoint}"),
                &format!(
                    "Showing the last frame received (sequence {last_sequence}); nothing newer."
                ),
                "Retrying automatically.".into(),
            ),
            Condition::LiveDrops { dropped } => p(
                Status::Degraded,
                format!("{dropped} live frames dropped"),
                "Dropped frames are not interpolated: the timeline shows the gaps.",
                "Check the link; the viewer keeps presenting the newest frame.".into(),
            ),
            Condition::SimPaused => p(
                Status::Pending,
                "Simulation paused".into(),
                "The world is frozen at the last committed tick.",
                "Resume with Space.".into(),
            ),
            Condition::SimSlow { ratio } => p(
                Status::Degraded,
                format!("Simulation at {:.0}% of real time", ratio * 100.0),
                "Every tick is still presented; time runs slower than the clock.",
                "Lower the quality tier or the scenario's size.".into(),
            ),
            Condition::Command { command, state } => {
                let (status, what, next) = match state {
                    CommandState::Pending => (
                        Status::Pending,
                        "pending".to_string(),
                        "Wait for acknowledgement.".to_string(),
                    ),
                    CommandState::Busy => (
                        Status::Pending,
                        "busy".to_string(),
                        "Wait; the target is executing an earlier command.".to_string(),
                    ),
                    CommandState::Rejected(reason) => (
                        Status::Critical,
                        format!("rejected: {reason}"),
                        "Review the reason and issue a corrected command.".to_string(),
                    ),
                    CommandState::Stale => (
                        Status::Stale,
                        "stale".to_string(),
                        "Re-issue against the current state.".to_string(),
                    ),
                };
                p(
                    status,
                    format!("Command {command} {what}"),
                    "The world shows committed state only, not the command's intent.",
                    next,
                )
            }
            Condition::Recording(state) => {
                let (status, title, next) = match state {
                    RecordingState::Backpressure => (
                        Status::Degraded,
                        "Recording under backpressure".to_string(),
                        "Reduce load; frames are queued, not dropped.".to_string(),
                    ),
                    RecordingState::Paused => (
                        Status::Pending,
                        "Recording paused".to_string(),
                        "Resume recording to capture this period.".to_string(),
                    ),
                    RecordingState::Incomplete => (
                        Status::Stale,
                        "Recording incomplete".to_string(),
                        "Recover the run directory before relying on it.".to_string(),
                    ),
                    RecordingState::DiskFailure(reason) => (
                        Status::Critical,
                        format!("Recording failed: {reason}"),
                        "Free disk space or choose another directory.".to_string(),
                    ),
                };
                p(
                    status,
                    title,
                    "Presentation continues; only the record is affected.",
                    next,
                )
            }
            Condition::SimFault { reason } => Presentation {
                action: Some(ConditionAction::OpenScenarioPanel),
                ..p(
                    Status::Critical,
                    format!("Simulation failed: {reason}"),
                    "The last completed scene stays shown.",
                    "Open Scenario to retry, or reset.".into(),
                )
            },
            Condition::Security { check, reason } => {
                let what = match check {
                    SecurityCheck::Tls => "TLS",
                    SecurityCheck::Authentication => "Authentication",
                    SecurityCheck::Capability => "Capability",
                };
                p(
                    Status::Critical,
                    format!("{what} failure: {reason}"),
                    "No data from this source is presented until it is trusted.",
                    "Check the certificates and credentials, then reconnect.".into(),
                )
            }
            Condition::UnsupportedVersion {
                what,
                found,
                supported,
            } => p(
                Status::Critical,
                format!("Unsupported {what} version {found}"),
                "Fields this viewer does not know may be missing or misread.",
                format!(
                    "Open it with a viewer that supports {found}, or re-export as {supported}."
                ),
            ),
        }
    }

    /// One of every condition, for fixtures.
    pub fn fixtures() -> Vec<Condition> {
        vec![
            Condition::NoSource,
            Condition::LoadingManifest,
            Condition::VerifyingWorld {
                revision: "w-7".into(),
            },
            Condition::WorldUnavailable {
                reason: "no terrain in this replay".into(),
            },
            Condition::WorldIncompatible {
                reason: "datum mismatch".into(),
            },
            Condition::BeliefUnavailable {
                reason: "no belief data".into(),
            },
            Condition::BeliefStale { age_s: 75.0 },
            Condition::BeliefAwaitingBase { revision: 12 },
            Condition::Reconnecting {
                endpoint: "https://127.0.0.1:50051".into(),
                last_sequence: 418,
            },
            Condition::LiveDrops { dropped: 3 },
            Condition::SimPaused,
            Condition::SimSlow { ratio: 0.6 },
            Condition::Command {
                command: "goto drone-1".into(),
                state: CommandState::Pending,
            },
            Condition::Command {
                command: "goto drone-1".into(),
                state: CommandState::Busy,
            },
            Condition::Command {
                command: "goto drone-1".into(),
                state: CommandState::Rejected("inside a keep-out zone".into()),
            },
            Condition::Command {
                command: "goto drone-1".into(),
                state: CommandState::Stale,
            },
            Condition::Recording(RecordingState::Backpressure),
            Condition::Recording(RecordingState::Paused),
            Condition::Recording(RecordingState::Incomplete),
            Condition::Recording(RecordingState::DiskFailure("no space left".into())),
            Condition::SimFault {
                reason: "scene build failed".into(),
            },
            Condition::Security {
                check: SecurityCheck::Tls,
                reason: "unknown CA".into(),
            },
            Condition::Security {
                check: SecurityCheck::Authentication,
                reason: "certificate expired".into(),
            },
            Condition::Security {
                check: SecurityCheck::Capability,
                reason: "truth not granted".into(),
            },
            Condition::UnsupportedVersion {
                what: "replay schema",
                found: "3.0".into(),
                supported: "2.x",
            },
        ]
    }
}

/// The replay schema major version this viewer reads.
pub const SUPPORTED_REPLAY_MAJOR: &str = "2";

/// The replay's declared schema version, judged. An undeclared version is
/// accepted (older replays and fixtures carry none); a declared one must be
/// 2.x.
pub fn replay_schema_condition(declared: Option<&str>) -> Option<Condition> {
    let found = declared?;
    let major = found.split('.').next().unwrap_or_default();
    (major != SUPPORTED_REPLAY_MAJOR).then(|| Condition::UnsupportedVersion {
        what: "replay schema",
        found: found.to_string(),
        supported: "2.x",
    })
}

/// What the live link reports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveInputs {
    pub endpoint: String,
    pub reconnecting: bool,
    pub dropped: u64,
    pub last_sequence: u64,
}

/// Everything detection reads.
#[derive(Debug, Clone, Default)]
pub struct ConditionInputs {
    pub has_replay: bool,
    pub live: Option<LiveInputs>,
    /// Why belief is unavailable, when it is.
    pub belief_unavailable: Option<String>,
    pub belief_overlay_shown: bool,
    pub sim_error: Option<String>,
    pub replay_schema: Option<String>,
}

/// The conditions the inputs show, most severe first. Belief unavailable is
/// raised only while its overlay is shown: hidden, it misleads nobody.
pub fn detect(inputs: &ConditionInputs) -> Vec<Condition> {
    let mut found = Vec::new();
    if !inputs.has_replay && inputs.live.is_none() {
        found.push(Condition::NoSource);
    }
    if let Some(live) = &inputs.live {
        if live.reconnecting {
            found.push(Condition::Reconnecting {
                endpoint: live.endpoint.clone(),
                last_sequence: live.last_sequence,
            });
        }
        if live.dropped > 0 {
            found.push(Condition::LiveDrops {
                dropped: live.dropped,
            });
        }
    }
    if inputs.belief_overlay_shown {
        if let Some(reason) = &inputs.belief_unavailable {
            found.push(Condition::BeliefUnavailable {
                reason: reason.clone(),
            });
        }
    }
    if let Some(reason) = &inputs.sim_error {
        found.push(Condition::SimFault {
            reason: reason.clone(),
        });
    }
    found.extend(replay_schema_condition(inputs.replay_schema.as_deref()));
    found.sort_by_key(|condition| rank(condition.present().status));
    found
}

/// The conditions in force, refreshed each frame.
#[derive(Resource, Debug, Default, Clone, PartialEq)]
pub struct ActiveConditions(pub Vec<Condition>);

/// Gather the inputs and detect.
#[allow(clippy::too_many_arguments)]
pub fn detect_conditions_system(
    replay: Res<crate::replay::ReplayState>,
    belief: Res<crate::belief_render::BeliefLayer>,
    visibility: Res<RuntimeOverlayVisibility>,
    sim: Res<crate::state::SimulationRunner>,
    #[cfg(feature = "live-stream")] live: Option<Res<crate::live::LiveStream>>,
    mut active: ResMut<ActiveConditions>,
    mut announced: Local<Vec<String>>,
) {
    #[cfg(feature = "live-stream")]
    let live = live.map(|live| LiveInputs {
        endpoint: live.endpoint.clone(),
        reconnecting: live.status() == crate::live::LiveConnectionStatus::Reconnecting,
        dropped: live.dropped_frame_count,
        last_sequence: live.latest_sequence,
    });
    #[cfg(not(feature = "live-stream"))]
    let live = None;
    let inputs = ConditionInputs {
        has_replay: replay.document.is_some(),
        live,
        belief_unavailable: match belief.as_ref() {
            crate::belief_render::BeliefLayer::Unavailable(reason) => Some(reason.to_string()),
            crate::belief_render::BeliefLayer::Ready(_) => None,
        },
        belief_overlay_shown: visibility.belief,
        sim_error: (sim.phase == crate::state::SimPhase::Error)
            .then(|| sim.error.clone().unwrap_or_else(|| "unknown error".into())),
        replay_schema: replay
            .document
            .as_ref()
            .and_then(|document| document.meta.as_ref())
            .and_then(|meta| meta.get("schema_version"))
            .and_then(|version| version.as_str())
            .map(str::to_owned),
    };
    let found = detect(&inputs);
    // Announced once, when it appears, not on every frame (§13).
    let titles: Vec<String> = found.iter().map(|c| c.present().title).collect();
    for title in &titles {
        if !announced.contains(title) {
            warn!("viewer condition: {title}");
        }
    }
    *announced = titles;
    active.set_if_neq(ActiveConditions(found));
}

/// Draw one condition: status glyph and word, title, what is preserved, and
/// the next action. Returns the action if its button was pressed.
pub fn condition_card(ui: &mut egui::Ui, presentation: &Presentation) -> Option<ConditionAction> {
    let mut pressed = None;
    ui.horizontal(|ui| {
        ui.label(presentation.status.rich_text());
        ui.strong(&presentation.title);
    });
    ui.label(egui::RichText::new(&presentation.preserved).small());
    ui.horizontal(|ui| {
        ui.label(egui::RichText::new(&presentation.next).small().italics());
        if let Some(action) = presentation.action {
            let label = match action {
                ConditionAction::HideBeliefOverlay => "Hide overlay",
                ConditionAction::OpenScenarioPanel => "Open Scenario",
            };
            if ui.small_button(label).clicked() {
                pressed = Some(action);
            }
        }
    });
    pressed
}

/// The canvas banner: every active condition, most severe first, at the
/// canvas's top centre over the last coherent view.
pub fn conditions_ui_system(
    mut contexts: EguiContexts,
    active: Res<ActiveConditions>,
    mut ui_state: ResMut<ViewerUiState>,
    mut visibility: ResMut<RuntimeOverlayVisibility>,
) {
    if active.0.is_empty() {
        return;
    }
    let Ok(context) = contexts.ctx_mut() else {
        return;
    };
    if !crate::theme::fonts_ready(context) {
        return;
    }
    let [min_x, min_y, max_x, _] = ui_state.canvas_rect.unwrap_or([0.0, 0.0, 800.0, 600.0]);
    let mut pressed = None;
    egui::Area::new(egui::Id::new("viewer_conditions"))
        .pivot(egui::Align2::CENTER_TOP)
        .fixed_pos(egui::pos2((min_x + max_x) * 0.5, min_y + 8.0))
        .order(egui::Order::Foreground)
        .show(context, |ui| {
            ui.set_max_width(((max_x - min_x) * 0.5).max(280.0));
            for condition in &active.0 {
                egui::Frame::popup(ui.style()).show(ui, |ui| {
                    if let Some(action) = condition_card(ui, &condition.present()) {
                        pressed = Some(action);
                    }
                });
            }
        });
    match pressed {
        Some(ConditionAction::HideBeliefOverlay) => visibility.belief = false,
        Some(ConditionAction::OpenScenarioPanel) => {
            ui_state.mode = ViewerMode::Scenario;
            ui_state.drawer_open = true;
        }
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_state_names_what_is_stale_and_a_next_action() {
        let fixtures = Condition::fixtures();
        // §12's list: every kind is represented.
        assert_eq!(fixtures.len(), 25);
        for condition in &fixtures {
            let p = condition.present();
            assert!(!p.title.is_empty(), "{condition:?}");
            assert!(p.preserved.len() > 20, "{condition:?} says what is stale");
            assert!(p.next.ends_with('.'), "{condition:?} offers a next action");
            assert_ne!(p.status, Status::Nominal, "{condition:?} is not nominal");
        }
    }

    #[test]
    fn every_state_draws_with_text_not_only_colour() {
        let context = egui::Context::default();
        crate::theme::install(&context);
        for condition in Condition::fixtures() {
            let presentation = condition.present();
            let mut text = String::new();
            // The bundled faces bind on the second pass.
            for _ in 0..2 {
                let mut output = context.run_ui(egui::RawInput::default(), |ui| {
                    condition_card(ui, &presentation);
                });
                output.textures_delta.clear();
                text = output
                    .shapes
                    .iter()
                    .filter_map(|shape| match &shape.shape {
                        egui::epaint::Shape::Text(text) => Some(text.galley.text().to_string()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
            }
            assert!(text.contains(presentation.status.label()), "{text}");
            assert!(text.contains(&presentation.title), "{text}");
            assert!(text.contains(&presentation.next), "{text}");
        }
    }

    #[test]
    fn detection_reports_what_it_can_observe_most_severe_first() {
        let inputs = ConditionInputs {
            has_replay: true,
            live: Some(LiveInputs {
                endpoint: "https://host:50051".into(),
                reconnecting: true,
                dropped: 2,
                last_sequence: 9,
            }),
            belief_unavailable: Some("no belief data".into()),
            belief_overlay_shown: true,
            sim_error: Some("scene build failed".into()),
            replay_schema: Some("3.1".into()),
        };
        let found = detect(&inputs);
        let statuses: Vec<Status> = found.iter().map(|c| c.present().status).collect();
        assert_eq!(statuses[0], Status::Critical);
        assert!(
            statuses.windows(2).all(|w| rank(w[0]) <= rank(w[1])),
            "{statuses:?}"
        );
        assert_eq!(found.len(), 5, "{found:?}");
        assert!(!found.contains(&Condition::NoSource), "live is a source");
    }

    #[test]
    fn a_quiet_replay_raises_nothing() {
        let inputs = ConditionInputs {
            has_replay: true,
            belief_unavailable: Some("no belief data".into()),
            belief_overlay_shown: false,
            replay_schema: Some("2.0".into()),
            ..Default::default()
        };
        assert!(detect(&inputs).is_empty(), "hidden belief misleads nobody");
        assert_eq!(
            detect(&ConditionInputs::default()),
            [Condition::NoSource],
            "nothing open"
        );
    }

    #[test]
    fn replay_schema_versions_are_judged_by_major() {
        assert_eq!(
            replay_schema_condition(None),
            None,
            "undeclared is accepted"
        );
        assert_eq!(replay_schema_condition(Some("2.0")), None);
        assert_eq!(replay_schema_condition(Some("2.7")), None);
        for bad in ["3.0", "1.4", "garbage", ""] {
            assert!(
                matches!(
                    replay_schema_condition(Some(bad)),
                    Some(Condition::UnsupportedVersion { .. })
                ),
                "{bad}"
            );
        }
    }
}
