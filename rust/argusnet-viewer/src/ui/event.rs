//! The safety-event inspector (VIEWER_UI §7.4, C3.3): what a safety event
//! decided, on what, and when it took effect, with the committed snapshots on
//! either side. Built from the replay as a pure function, like the platform
//! inspector; what a replay does not record reads as unavailable.

use crate::replay::ReplayDocument;
use crate::theme::Field;

use super::platform::Row;

/// A safety event, by where it was committed and its own id. The epoch
/// voids it: frame indices name different frames in another run.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EventRef {
    pub epoch: u64,
    pub frame_index: usize,
    pub event_id: String,
}

/// What the inspector shows.
#[derive(Debug, Clone, PartialEq)]
pub struct EventInspection {
    pub title: String,
    /// The platform the event is about.
    pub subject: String,
    pub rows: Vec<Row>,
    /// The committed snapshots before and after the event's own.
    pub preceding: Option<usize>,
    pub following: Option<usize>,
}

const NOT_IN_REPLAY: &str = "not in this replay";

/// The event `event` names in `document`, or `None` if it does not resolve
/// (another epoch, or a frame or id this replay does not have).
pub fn inspect_safety_event(
    document: &ReplayDocument,
    epoch: u64,
    event: &EventRef,
) -> Option<EventInspection> {
    if event.epoch != epoch {
        return None;
    }
    let frame = document.frames.get(event.frame_index)?;
    let found = frame
        .safety_events
        .iter()
        .find(|candidate| candidate.event_id == event.event_id)?;
    let decision = if found.blocked {
        "blocked"
    } else if found.command_clamped {
        "clamped"
    } else {
        "passed"
    };
    let state = if found.state.is_empty() {
        found.reason.as_str()
    } else {
        found.state.as_str()
    };
    let rows = vec![
        Row::new("Decision", Field::Value(decision.into())),
        Row::new(
            "State",
            if state.is_empty() {
                Field::Unavailable("not reported")
            } else {
                Field::Value(state.into())
            },
        ),
        Row::new(
            "Constraints violated",
            if found.violations.is_empty() {
                Field::Value("none".into())
            } else {
                Field::Value(found.violations.join(", "))
            },
        ),
        Row::new(
            "Modification",
            Field::Value(
                if found.command_clamped {
                    "command clamped"
                } else if found.blocked {
                    "command refused"
                } else {
                    "none"
                }
                .into(),
            ),
        ),
        Row::unavailable("Proposed command", "commands are not recorded in replays"),
        Row::unavailable("Evaluated geometry", NOT_IN_REPLAY),
        Row::unavailable("Constraint values", NOT_IN_REPLAY),
        Row::unavailable("Fallback", NOT_IN_REPLAY),
        Row::unavailable("Acknowledgement", "commands are not recorded in replays"),
        Row::new(
            "Effective",
            Field::Value(format!(
                "frame {} · t = {:.2} s",
                event.frame_index + 1,
                found.timestamp_s
            )),
        ),
    ];
    Some(EventInspection {
        title: format!("Safety event {}", found.event_id),
        subject: found.subject_id.clone(),
        rows,
        preceding: event.frame_index.checked_sub(1),
        following: (event.frame_index + 1 < document.frames.len()).then_some(event.frame_index + 1),
    })
}

/// The safety events committed in frame `frame_index`, as references.
pub fn safety_events_in(
    document: &ReplayDocument,
    epoch: u64,
    frame_index: usize,
) -> Vec<(EventRef, String)> {
    document
        .frames
        .get(frame_index)
        .map(|frame| {
            frame
                .safety_events
                .iter()
                .map(|event| {
                    let decision = if event.blocked {
                        "blocked"
                    } else if event.command_clamped {
                        "clamped"
                    } else {
                        "passed"
                    };
                    (
                        EventRef {
                            epoch,
                            frame_index,
                            event_id: event.event_id.clone(),
                        },
                        format!("{}: {decision}", event.subject_id),
                    )
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn document() -> ReplayDocument {
        serde_json::from_value(serde_json::json!({
            "frames": [
                {"timestamp_s": 0.0},
                {"timestamp_s": 0.5, "safety_events": [
                    {"event_id": "se-1", "subject_id": "drone-1", "timestamp_s": 0.5,
                     "state": "geofence", "violations": ["altitude_max"],
                     "command_clamped": true, "blocked": false, "reason": ""},
                    {"event_id": "se-2", "subject_id": "drone-2", "timestamp_s": 0.5,
                     "accepted": false, "violations": [], "reason": "keep-out zone"}
                ]},
                {"timestamp_s": 1.0}
            ]
        }))
        .unwrap()
    }

    fn reference(frame_index: usize, id: &str) -> EventRef {
        EventRef {
            epoch: 3,
            frame_index,
            event_id: id.into(),
        }
    }

    fn row<'a>(inspection: &'a EventInspection, label: &str) -> &'a Field<String> {
        &inspection
            .rows
            .iter()
            .find(|row| row.label == label)
            .unwrap_or_else(|| panic!("no row {label}"))
            .value
    }

    #[test]
    fn an_event_shows_its_decision_constraints_and_effective_tick() {
        let inspection = inspect_safety_event(&document(), 3, &reference(1, "se-1")).unwrap();
        assert_eq!(inspection.subject, "drone-1");
        assert_eq!(
            row(&inspection, "Decision"),
            &Field::Value("clamped".into())
        );
        assert_eq!(
            row(&inspection, "Constraints violated"),
            &Field::Value("altitude_max".into())
        );
        assert_eq!(
            row(&inspection, "Modification"),
            &Field::Value("command clamped".into())
        );
        assert_eq!(
            row(&inspection, "Effective"),
            &Field::Value("frame 2 · t = 0.50 s".into())
        );
    }

    #[test]
    fn a_refused_command_reads_as_blocked_with_its_reason() {
        let inspection = inspect_safety_event(&document(), 3, &reference(1, "se-2")).unwrap();
        assert_eq!(
            row(&inspection, "Decision"),
            &Field::Value("blocked".into())
        );
        assert_eq!(
            row(&inspection, "State"),
            &Field::Value("keep-out zone".into())
        );
        assert_eq!(
            row(&inspection, "Constraints violated"),
            &Field::Value("none".into())
        );
    }

    #[test]
    fn what_replays_do_not_record_is_unavailable() {
        let inspection = inspect_safety_event(&document(), 3, &reference(1, "se-1")).unwrap();
        for label in [
            "Proposed command",
            "Evaluated geometry",
            "Constraint values",
            "Fallback",
            "Acknowledgement",
        ] {
            assert!(
                matches!(row(&inspection, label), Field::Unavailable(_)),
                "{label}"
            );
        }
    }

    #[test]
    fn the_neighbouring_snapshots_are_the_frames_either_side() {
        let doc = document();
        let inspection = inspect_safety_event(&doc, 3, &reference(1, "se-1")).unwrap();
        assert_eq!(
            (inspection.preceding, inspection.following),
            (Some(0), Some(2))
        );
        let mut last = doc.clone();
        last.frames.truncate(2);
        let inspection = inspect_safety_event(&last, 3, &reference(1, "se-1")).unwrap();
        assert_eq!(inspection.following, None, "no snapshot after the last");
    }

    #[test]
    fn an_event_from_another_epoch_or_missing_does_not_resolve() {
        let doc = document();
        assert!(inspect_safety_event(&doc, 4, &reference(1, "se-1")).is_none());
        assert!(inspect_safety_event(&doc, 3, &reference(1, "se-9")).is_none());
        assert!(inspect_safety_event(&doc, 3, &reference(7, "se-1")).is_none());
    }

    #[test]
    fn a_frames_events_are_listed_as_references() {
        let events = safety_events_in(&document(), 3, 1);
        let labels: Vec<&str> = events.iter().map(|(_, label)| label.as_str()).collect();
        assert_eq!(labels, ["drone-1: clamped", "drone-2: blocked"]);
        assert_eq!(events[0].0, reference(1, "se-1"));
        assert!(safety_events_in(&document(), 3, 0).is_empty());
    }
}
