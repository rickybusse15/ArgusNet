//! The persistent source indicator (VIEWER_UI §3, C2.2): which source the
//! window shows, its execution environment and its temporal context, always
//! visible in the header.
//!
//! The context is resolved from what the viewer knows, as a pure function, so
//! the strings are exactly §3's and testable without rendering. The in-process
//! simulation worker (D1) and SIL/HIL adapters are not wired yet; their states
//! are modelled here so that wiring them is a matter of supplying the input.

use bevy_egui::egui;

use crate::theme::{Status, TypeRole};

/// The in-process simulation's run state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SimRunState {
    Running,
    Paused,
    Faulted,
}

/// A qualified flight-software or hardware adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InLoop {
    Software,
    Hardware,
}

/// §3's visible context state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceContext {
    /// Following the newest coherent live snapshot.
    Live,
    /// Inspecting retained live history while live data keeps arriving.
    LiveHistory,
    Sim(SimRunState),
    InLoop {
        kind: InLoop,
        history: bool,
    },
    Replay,
    /// Simulation-derived data with the truth capability on.
    EvaluationTruth,
    /// Nothing loaded.
    NoSource,
}

impl SourceContext {
    /// The indicator's text: §3's strings, verbatim.
    pub fn label(self) -> &'static str {
        match self {
            SourceContext::Live => "LIVE",
            SourceContext::LiveHistory => "LIVE / HISTORY",
            SourceContext::Sim(SimRunState::Running) => "SIM / RUNNING",
            SourceContext::Sim(SimRunState::Paused) => "SIM / PAUSED",
            SourceContext::Sim(SimRunState::Faulted) => "SIM / FAULTED",
            SourceContext::InLoop {
                kind: InLoop::Software,
                history: false,
            } => "SIL",
            SourceContext::InLoop {
                kind: InLoop::Software,
                history: true,
            } => "SIL / HISTORY",
            SourceContext::InLoop {
                kind: InLoop::Hardware,
                history: false,
            } => "HIL",
            SourceContext::InLoop {
                kind: InLoop::Hardware,
                history: true,
            } => "HIL / HISTORY",
            SourceContext::Replay => "REPLAY",
            SourceContext::EvaluationTruth => "EVALUATION / TRUTH ENABLED",
            SourceContext::NoSource => "NO SOURCE",
        }
    }

    /// The status vocabulary's shape and colour for the indicator, so the
    /// context is never told by colour alone (§5.3). Truth enabled is
    /// flagged with the accent: it is the one context an operator must never
    /// mistake for another.
    pub fn status(self) -> Status {
        match self {
            SourceContext::Live
            | SourceContext::Sim(SimRunState::Running)
            | SourceContext::InLoop { history: false, .. }
            | SourceContext::Replay => Status::Nominal,
            SourceContext::LiveHistory | SourceContext::InLoop { history: true, .. } => {
                Status::Stale
            }
            SourceContext::Sim(SimRunState::Paused) => Status::Pending,
            SourceContext::Sim(SimRunState::Faulted) => Status::Critical,
            SourceContext::EvaluationTruth => Status::Degraded,
            SourceContext::NoSource => Status::Unavailable,
        }
    }

    /// The indicator as drawn: glyph and label, in the utility role.
    pub fn rich_text(self) -> egui::RichText {
        let status = self.status();
        TypeRole::Utility
            .text(format!("{} {}", status.glyph(), self.label()))
            .color(status.color())
            .strong()
    }

    /// One line on what the context permits (§3's table), for the tooltip.
    pub fn explanation(self) -> &'static str {
        match self {
            SourceContext::Live => {
                "Following the live stream. Read-only; history never rewinds the external world."
            }
            SourceContext::LiveHistory => {
                "Reviewing retained live history; live data is still arriving. Read-only."
            }
            SourceContext::Sim(_) => "In-process simulation: local, acknowledged commands.",
            SourceContext::InLoop { .. } => "Flight-software or hardware in the loop. Read-only supervision.",
            SourceContext::Replay => "A recorded replay. Read-only.",
            SourceContext::EvaluationTruth => {
                "A simulation replay with ground truth shown. Truth is never available in live operation."
            }
            SourceContext::NoSource => "No replay, stream or simulation is loaded.",
        }
    }
}

/// What the viewer knows about its source this frame.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SourceInputs {
    /// Frames are loaded.
    pub has_frames: bool,
    /// The frames are a live stream's history.
    pub live: bool,
    /// The cursor is on the newest frame.
    pub at_newest: bool,
    /// An in-process simulation drives the frames (D1).
    pub sim: Option<SimRunState>,
    /// A SIL/HIL adapter drives the frames.
    pub in_loop: Option<InLoop>,
    /// The truth capability is on.
    pub truth_shown: bool,
}

impl SourceInputs {
    /// The context to show. Live outranks everything: truth is never shown
    /// as available in live operation, and history is labelled as history.
    pub fn context(self) -> SourceContext {
        if self.live {
            return if self.at_newest || !self.has_frames {
                SourceContext::Live
            } else {
                SourceContext::LiveHistory
            };
        }
        if let Some(kind) = self.in_loop {
            return SourceContext::InLoop {
                kind,
                history: self.has_frames && !self.at_newest,
            };
        }
        if let Some(state) = self.sim {
            return SourceContext::Sim(state);
        }
        if !self.has_frames {
            return SourceContext::NoSource;
        }
        if self.truth_shown {
            SourceContext::EvaluationTruth
        } else {
            SourceContext::Replay
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn replay() -> SourceInputs {
        SourceInputs {
            has_frames: true,
            at_newest: false,
            ..SourceInputs::default()
        }
    }

    #[test]
    fn the_labels_are_exactly_the_specified_strings() {
        let all = [
            (SourceContext::Live, "LIVE"),
            (SourceContext::LiveHistory, "LIVE / HISTORY"),
            (SourceContext::Sim(SimRunState::Running), "SIM / RUNNING"),
            (SourceContext::Sim(SimRunState::Paused), "SIM / PAUSED"),
            (SourceContext::Sim(SimRunState::Faulted), "SIM / FAULTED"),
            (
                SourceContext::InLoop {
                    kind: InLoop::Software,
                    history: false,
                },
                "SIL",
            ),
            (
                SourceContext::InLoop {
                    kind: InLoop::Hardware,
                    history: false,
                },
                "HIL",
            ),
            (SourceContext::Replay, "REPLAY"),
            (SourceContext::EvaluationTruth, "EVALUATION / TRUTH ENABLED"),
        ];
        for (context, label) in all {
            assert_eq!(context.label(), label);
            assert!(!context.explanation().is_empty());
        }
        // A fault reads as critical; truth is flagged.
        assert_eq!(
            SourceContext::Sim(SimRunState::Faulted).status(),
            Status::Critical
        );
        assert_eq!(SourceContext::EvaluationTruth.status(), Status::Degraded);
    }

    #[test]
    fn a_replay_is_replay_until_truth_is_enabled() {
        assert_eq!(replay().context(), SourceContext::Replay);
        let evaluation = SourceInputs {
            truth_shown: true,
            ..replay()
        };
        assert_eq!(evaluation.context(), SourceContext::EvaluationTruth);
    }

    #[test]
    fn live_follows_the_newest_frame_or_says_it_is_history() {
        let following = SourceInputs {
            live: true,
            at_newest: true,
            ..replay()
        };
        assert_eq!(following.context(), SourceContext::Live);
        let reviewing = SourceInputs {
            at_newest: false,
            ..following
        };
        assert_eq!(reviewing.context(), SourceContext::LiveHistory);
        // Connected, nothing received yet: still live, not history.
        let waiting = SourceInputs {
            live: true,
            ..SourceInputs::default()
        };
        assert_eq!(waiting.context(), SourceContext::Live);
    }

    #[test]
    fn live_never_claims_truth() {
        let live_with_truth_toggle = SourceInputs {
            live: true,
            at_newest: true,
            truth_shown: true,
            ..replay()
        };
        assert_eq!(live_with_truth_toggle.context(), SourceContext::Live);
    }

    #[test]
    fn a_simulation_reports_its_run_state() {
        for state in [
            SimRunState::Running,
            SimRunState::Paused,
            SimRunState::Faulted,
        ] {
            let inputs = SourceInputs {
                sim: Some(state),
                ..replay()
            };
            assert_eq!(inputs.context(), SourceContext::Sim(state));
        }
    }

    #[test]
    fn in_the_loop_sources_are_qualified_by_history() {
        let current = SourceInputs {
            in_loop: Some(InLoop::Hardware),
            at_newest: true,
            ..replay()
        };
        assert_eq!(current.context().label(), "HIL");
        let history = SourceInputs {
            at_newest: false,
            ..current
        };
        assert_eq!(history.context().label(), "HIL / HISTORY");
    }

    #[test]
    fn nothing_loaded_says_so() {
        assert_eq!(SourceInputs::default().context(), SourceContext::NoSource);
    }
}
