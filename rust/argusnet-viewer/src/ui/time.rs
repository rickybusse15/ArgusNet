//! The timeline's model (C4, VIEWER_UI §8): the one set of time commands that
//! the timeline's buttons and the keyboard both issue, and the event marks it
//! draws, grouped by kind. Pure functions over the replay, so tested without
//! rendering.

use crate::replay::{ReplayDocument, ReplayState};

/// A time control. The timeline's buttons and the keyboard bindings issue
/// these and nothing else, so the two can never disagree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlaybackCommand {
    PlayPause,
    StepBack,
    StepForward,
    First,
    Last,
    PreviousEvent,
    NextEvent,
}

/// Apply `command` to the replay cursor. Returns whether the cursor moved or
/// playback toggled. Stepping and jumping pause playback.
pub fn apply_playback(replay: &mut ReplayState, command: PlaybackCommand) -> bool {
    let count = replay.frame_count();
    if count == 0 {
        return false;
    }
    let before = (replay.frame_index, replay.playing);
    match command {
        PlaybackCommand::PlayPause => replay.playing = !replay.playing,
        PlaybackCommand::StepBack => {
            replay.playing = false;
            replay.step_to(replay.frame_index.saturating_sub(1));
        }
        PlaybackCommand::StepForward => {
            replay.playing = false;
            replay.step_to(replay.frame_index + 1);
        }
        PlaybackCommand::First => {
            replay.playing = false;
            replay.step_to(0);
        }
        PlaybackCommand::Last => {
            replay.playing = false;
            replay.step_to(count - 1);
        }
        PlaybackCommand::PreviousEvent => {
            replay.jump_to_event(false);
        }
        PlaybackCommand::NextEvent => {
            replay.jump_to_event(true);
        }
    }
    before != (replay.frame_index, replay.playing)
}

/// The kinds of event mark (C4: mission / safety / command / evidence /
/// fault), one lane each, in this order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MarkGroup {
    Mission,
    Safety,
    Command,
    Evidence,
    Fault,
}

impl MarkGroup {
    pub const ALL: [MarkGroup; 5] = [
        MarkGroup::Mission,
        MarkGroup::Safety,
        MarkGroup::Command,
        MarkGroup::Evidence,
        MarkGroup::Fault,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MarkGroup::Mission => "Mission",
            MarkGroup::Safety => "Safety",
            MarkGroup::Command => "Command",
            MarkGroup::Evidence => "Evidence",
            MarkGroup::Fault => "Fault",
        }
    }

    /// Why a lane can be empty for a reason other than "nothing happened".
    pub fn unavailable_in_replays(self) -> Option<&'static str> {
        match self {
            MarkGroup::Command => Some("commands are not recorded in replays"),
            _ => None,
        }
    }
}

/// One event mark.
#[derive(Debug, Clone, PartialEq)]
pub struct Mark {
    pub time_s: f32,
    pub frame_index: usize,
    pub group: MarkGroup,
    pub label: String,
}

/// Node health below which a node's first crossing is marked as a fault;
/// the same threshold the alerts use.
pub const FAULT_HEALTH: f32 = super::HEALTH_ALERT_THRESHOLD;

/// Every mark in the replay, in time order then lane order:
/// - **mission**: scan-mission phase changes;
/// - **safety**: safety-monitor decisions and deconfliction resolutions;
/// - **evidence**: inspection events;
/// - **fault**: a node's health first falling below [`FAULT_HEALTH`];
/// - **command**: none, since replays record no commands.
pub fn timeline_marks(document: &ReplayDocument) -> Vec<Mark> {
    let mut marks = Vec::new();
    let mut last_phase: Option<&str> = None;
    let mut faulted: Vec<&str> = Vec::new();
    for (frame_index, frame) in document.frames.iter().enumerate() {
        let t = frame.timestamp_s;
        let mut mark = |group, label: String| {
            marks.push(Mark {
                time_s: t,
                frame_index,
                group,
                label,
            })
        };
        if let Some(mission) = &frame.scan_mission_state {
            if last_phase.is_some_and(|phase| phase != mission.phase) {
                mark(MarkGroup::Mission, format!("phase {}", mission.phase));
            }
            last_phase = Some(mission.phase.as_str());
        }
        for event in &frame.safety_events {
            let decision = if event.blocked {
                "blocked"
            } else if event.command_clamped {
                "clamped"
            } else {
                "passed"
            };
            mark(
                MarkGroup::Safety,
                format!("{} {decision}", event.subject_id),
            );
        }
        for event in &frame.deconfliction_events {
            mark(
                MarkGroup::Safety,
                format!(
                    "{} yields to {} ({})",
                    event.yielding_drone_id, event.conflicting_drone_id, event.resolution
                ),
            );
        }
        for event in &frame.inspection_events {
            mark(
                MarkGroup::Evidence,
                format!("{} {} {}", event.node_id, event.event_type, event.zone_id),
            );
        }
        for node in &frame.nodes {
            let low = node.health < FAULT_HEALTH;
            let was = faulted.contains(&node.node_id.as_str());
            if low && !was {
                faulted.push(node.node_id.as_str());
                mark(
                    MarkGroup::Fault,
                    format!("{} health {:.0}%", node.node_id, node.health * 100.0),
                );
            } else if !low && was {
                faulted.retain(|id| *id != node.node_id.as_str());
            }
        }
    }
    marks.sort_by(|a, b| a.time_s.total_cmp(&b.time_s).then(a.group.cmp(&b.group)));
    marks
}

/// The marks committed in frame `frame_index`: the selected event summary.
pub fn marks_in_frame(marks: &[Mark], frame_index: usize) -> Vec<&Mark> {
    marks
        .iter()
        .filter(|mark| mark.frame_index == frame_index)
        .collect()
}

/// Where the cursor is, as the timeline states it: source time, mission
/// elapsed since the first retained frame, frame, epoch, and what part of
/// the run cannot be seen.
pub fn time_readout(replay: &ReplayState) -> Option<String> {
    let (first, last) = replay.time_range()?;
    let now = replay.time_s();
    let mut text = format!(
        "epoch {} · t {now:.2} s · elapsed {:.2} of {:.2} s · frame {} / {}",
        replay.epoch(),
        now - first,
        last - first,
        replay.frame_index + 1,
        replay.frame_count()
    );
    if let Some(trimmed) = replay.trimmed_before_s() {
        text.push_str(&format!(" · history from {trimmed:.1} s"));
    }
    Some(text)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> ReplayDocument {
        let node = |health: f32| {
            serde_json::json!({"node_id": "n1", "position": [0.0, 0.0, 0.0],
                               "velocity": [0.0, 0.0, 0.0], "is_mobile": true,
                               "health": health})
        };
        let mission = |phase: &str| {
            serde_json::json!({"phase": phase, "scan_coverage_fraction": 0.0,
                               "scan_coverage_threshold": 0.9, "localization_estimates": [],
                               "poi_statuses": [], "completed_poi_count": 0,
                               "total_poi_count": 0})
        };
        serde_json::from_value(serde_json::json!({
            "frames": [
                {"timestamp_s": 0.0, "nodes": [node(0.9)], "scan_mission_state": mission("scanning")},
                {"timestamp_s": 1.0, "nodes": [node(0.2)], "scan_mission_state": mission("scanning"),
                 "safety_events": [{"event_id": "s1", "subject_id": "n1", "timestamp_s": 1.0,
                                    "blocked": true, "violations": []}]},
                {"timestamp_s": 2.0, "nodes": [node(0.1)], "scan_mission_state": mission("inspecting"),
                 "inspection_events": [{"zone_id": "z1", "node_id": "n1",
                                        "event_type": "entered", "timestamp_s": 2.0}]},
                {"timestamp_s": 3.0, "nodes": [node(0.9)],
                 "deconfliction_events": [{"yielding_drone_id": "n1", "conflicting_drone_id": "n2",
                                           "predicted_separation_m": 4.0, "resolution": "hold",
                                           "timestamp_s": 3.0}]},
                {"timestamp_s": 4.0, "nodes": [node(0.25)]}
            ]
        }))
        .unwrap()
    }

    #[test]
    fn marks_fall_in_their_lanes_in_time_order() {
        let marks = timeline_marks(&document());
        let summary: Vec<(f32, MarkGroup)> =
            marks.iter().map(|mark| (mark.time_s, mark.group)).collect();
        assert_eq!(
            summary,
            [
                (1.0, MarkGroup::Safety),
                (1.0, MarkGroup::Fault),
                (2.0, MarkGroup::Mission),
                (2.0, MarkGroup::Evidence),
                (3.0, MarkGroup::Safety),
                (4.0, MarkGroup::Fault),
            ]
        );
        assert_eq!(marks[0].label, "n1 blocked");
        assert_eq!(marks[2].label, "phase inspecting");
    }

    #[test]
    fn a_fault_is_marked_once_per_crossing() {
        let faults: Vec<f32> = timeline_marks(&document())
            .into_iter()
            .filter(|mark| mark.group == MarkGroup::Fault)
            .map(|mark| mark.time_s)
            .collect();
        // Low at 1 s and 2 s is one fault; recovered at 3 s; low again at 4 s.
        assert_eq!(faults, [1.0, 4.0]);
    }

    #[test]
    fn the_command_lane_says_why_it_is_empty() {
        assert!(timeline_marks(&document())
            .iter()
            .all(|mark| mark.group != MarkGroup::Command));
        assert!(MarkGroup::Command.unavailable_in_replays().is_some());
    }

    #[test]
    fn the_selected_frame_lists_its_marks() {
        let marks = timeline_marks(&document());
        let in_frame: Vec<&str> = marks_in_frame(&marks, 2)
            .iter()
            .map(|mark| mark.label.as_str())
            .collect();
        assert_eq!(in_frame, ["phase inspecting", "n1 entered z1"]);
        assert!(marks_in_frame(&marks, 0).is_empty());
    }

    #[test]
    fn keys_and_buttons_share_one_command_set() {
        let mut replay = ReplayState::new(Some(document()));
        assert!(apply_playback(&mut replay, PlaybackCommand::Last));
        assert_eq!(replay.frame_index, 4);
        assert!(apply_playback(&mut replay, PlaybackCommand::StepBack));
        assert_eq!(replay.frame_index, 3);
        assert!(apply_playback(&mut replay, PlaybackCommand::PlayPause));
        assert!(replay.playing);
        assert!(apply_playback(&mut replay, PlaybackCommand::First));
        assert!(!replay.playing, "stepping pauses");
        assert_eq!(replay.frame_index, 0);
        assert!(
            !apply_playback(&mut replay, PlaybackCommand::StepBack),
            "already first"
        );
        let mut empty = ReplayState::new(None);
        assert!(!apply_playback(&mut empty, PlaybackCommand::PlayPause));
    }

    #[test]
    fn the_readout_states_epoch_elapsed_and_frame() {
        let mut replay = ReplayState::new(Some(document()));
        apply_playback(&mut replay, PlaybackCommand::StepForward);
        let text = time_readout(&replay).unwrap();
        assert!(text.contains("frame 2 / 5"), "{text}");
        assert!(text.contains("elapsed 1.00 of 4.00 s"), "{text}");
        assert!(text.starts_with("epoch "), "{text}");
        assert!(time_readout(&ReplayState::new(None)).is_none());
    }
}
