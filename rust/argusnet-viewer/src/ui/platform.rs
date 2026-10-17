//! The platform inspector (VIEWER_UI §7.2, C3.1): §7.2's eleven sections in
//! order, built from a replay frame as a pure function so what is shown, and
//! what is shown as unavailable, is testable without rendering.
//!
//! Every value is a [`Field`]: what the replay does not carry reads as
//! unavailable, with the reason, never as zero. Legacy estimator outputs --
//! confidence, the localization timeout and the dropout-derived health -- are
//! marked legacy and are not presented as physical integrity (§7.2).

use crate::replay::ReplayFrame;
use crate::theme::Field;

/// One readout.
#[derive(Debug, Clone, PartialEq)]
pub struct Row {
    pub label: &'static str,
    pub value: Field<String>,
    /// A legacy estimator output: shown, labelled, never mapped to integrity.
    pub legacy: bool,
}

impl Row {
    pub(super) fn new(label: &'static str, value: Field<String>) -> Self {
        Self {
            label,
            value,
            legacy: false,
        }
    }

    fn legacy(label: &'static str, value: Field<String>) -> Self {
        Self {
            label,
            value,
            legacy: true,
        }
    }

    pub(super) fn unavailable(label: &'static str, why: &'static str) -> Self {
        Self::new(label, Field::Unavailable(why))
    }
}

/// One of §7.2's sections.
#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub title: &'static str,
    /// In §4.4 rule 3's reduced inspector.
    pub essential: bool,
    pub rows: Vec<Row>,
}

const NOT_IN_REPLAY: &str = "not in this replay";

/// §7.2's sections for `node_id` in `frame`; `None` if the frame has no such
/// platform (the selection no longer resolves). With the replay's
/// `meta.assignment_history` (P0b), the task and its last change come from
/// the recorded history at the frame's time instead of being guessed from POI
/// statuses, which miss a drone helping on a team, sweeping a gap or flying
/// home.
pub fn inspect_platform(
    frame: &ReplayFrame,
    node_id: &str,
    history: Option<&serde_json::Value>,
) -> Option<Vec<Section>> {
    let node = frame.nodes.iter().find(|node| node.node_id == node_id)?;
    let mission = frame.scan_mission_state.as_ref();
    let estimate = mission.and_then(|state| {
        state
            .localization_estimates
            .iter()
            .find(|estimate| estimate.drone_id == node_id)
    });
    let [vx, vy, vz] = node.velocity;
    let ground_speed = (vx * vx + vy * vy).sqrt();

    let identity = Section {
        title: "1  Identity",
        essential: false,
        rows: vec![
            Row::new("ID", Field::Value(node.node_id.clone())),
            Row::new(
                "Class",
                Field::Value(if node.is_mobile { "mobile" } else { "fixed" }.into()),
            ),
            Row::new(
                "Role",
                match mission {
                    Some(state) if state.coordinator_drone_id.as_deref() == Some(node_id) => {
                        Field::Value("mission coordinator".into())
                    }
                    Some(_) => Field::Value("member".into()),
                    None => Field::Unavailable("no mission state in this replay"),
                },
            ),
            Row::new("Source", Field::Value("replay".into())),
        ],
    };

    let actual = Section {
        title: "2  Actual state",
        essential: true,
        rows: vec![
            Row::new(
                "Position",
                Field::Value(format!(
                    "{:.1}, {:.1}, {:.1} m",
                    node.position[0], node.position[1], node.position[2]
                )),
            ),
            Row::new(
                "As of",
                Field::Value(format!("t = {:.2} s", frame.timestamp_s)),
            ),
        ],
    };

    let commanded = Section {
        title: "3  Commanded state",
        essential: true,
        rows: vec![
            Row::unavailable("Command", "commands are not recorded in replays"),
            Row::unavailable("Acknowledgement", "commands are not recorded in replays"),
        ],
    };

    let task = match mission {
        None => Field::Unavailable("no mission state in this replay"),
        Some(state) => {
            if state.phase == "egress" {
                Field::Value("return home".into())
            } else if let Some(poi) = state.poi_statuses.iter().find(|poi| {
                poi.assigned_drone_id.as_deref() == Some(node_id) && poi.status != "complete"
            }) {
                Field::Value(format!("inspect {} ({})", poi.poi_id, poi.status))
            } else {
                Field::Value(format!("{} (no assignment)", state.phase))
            }
        }
    };
    let recorded = history.map(|history| recorded_task(history, node_id, frame.timestamp_s));
    let mut tasking_rows = match recorded {
        Some(recorded) => vec![
            Row::new("Current task", recorded.task),
            Row::new("Last change", recorded.last_change),
            Row::new("Task changes", recorded.changes),
        ],
        None => vec![
            Row::new("Current task", task),
            Row::unavailable("Last change", "no assignment history in this replay"),
        ],
    };
    tasking_rows.push(Row::unavailable("Route / trajectory", NOT_IN_REPLAY));
    tasking_rows.push(Row::unavailable("Next transition", NOT_IN_REPLAY));
    let tasking = Section {
        title: "4  Task and route",
        essential: false,
        rows: tasking_rows,
    };

    let mut localization_rows = vec![
        Row::unavailable("Integrity", "no validated integrity adapter"),
        Row::new(
            "Position σ",
            estimate.map_or(Field::Unavailable("no estimate for this platform"), |e| {
                Field::Value(format!("{:.2} m", e.position_std_m))
            }),
        ),
        Row::legacy(
            "Confidence",
            estimate.map_or(Field::Unavailable("no estimate for this platform"), |e| {
                Field::Value(format!("{:.2}", e.confidence))
            }),
        ),
        Row::legacy(
            "Localization timeout",
            mission.map_or(
                Field::Unavailable("no mission state in this replay"),
                |state| {
                    Field::Value(
                        if state.localization_timed_out {
                            "timed out"
                        } else {
                            "no"
                        }
                        .into(),
                    )
                },
            ),
        ),
    ];
    localization_rows.push(Row::unavailable("Anchor age / source", NOT_IN_REPLAY));
    localization_rows.push(Row::unavailable("Reference frame", NOT_IN_REPLAY));
    let localization = Section {
        title: "5  Localization",
        essential: true,
        rows: localization_rows,
    };

    let dynamics = Section {
        title: "6  Dynamics",
        essential: false,
        rows: vec![
            Row::new(
                "Altitude",
                Field::Value(format!("{:.1} m", node.position[2])),
            ),
            Row::new(
                "Ground speed",
                Field::Value(format!("{ground_speed:.1} m/s")),
            ),
            Row::new("Vertical speed", Field::Value(format!("{vz:+.1} m/s"))),
            Row::new(
                "Heading",
                if ground_speed > 0.1 {
                    Field::Value(format!("{:.0}°", vy.atan2(vx).to_degrees()))
                } else {
                    Field::Unavailable("stationary: no heading from velocity")
                },
            ),
        ],
    };

    let egress = mission.and_then(|state| {
        state
            .egress_progress
            .iter()
            .find(|progress| progress.drone_id == node_id)
    });
    let energy = Section {
        title: "7  Battery and return",
        essential: true,
        rows: vec![
            Row::new(
                "Battery",
                node.battery_fraction
                    .map_or(Field::Unavailable(NOT_IN_REPLAY), |fraction| {
                        Field::Value(format!("{:.0}%", fraction * 100.0))
                    }),
            ),
            Row::unavailable("Reserve", NOT_IN_REPLAY),
            Row::new(
                "Return home",
                egress.map_or(Field::Unavailable("not returning"), |progress| {
                    Field::Value(format!("{:.0} m to home", progress.distance_to_home_m))
                }),
            ),
        ],
    };

    let link = Section {
        title: "8  Link",
        essential: true,
        rows: vec![
            // Health is 1 - dropout probability / 2: a legacy sensing
            // quantity, not a measured link.
            Row::legacy(
                "Health",
                Field::Value(format!("{:.0}%", node.health * 100.0)),
            ),
            Row::unavailable("Latency / loss", NOT_IN_REPLAY),
            Row::unavailable("Lost-link policy", NOT_IN_REPLAY),
        ],
    };

    let captures = frame
        .observations
        .iter()
        .filter(|observation| observation.node_id == node_id)
        .count();
    let sensors = Section {
        title: "9  Sensors",
        essential: false,
        rows: vec![
            Row::new("Type", Field::Value(node.sensor_type.clone())),
            Row::new(
                "Field of view",
                node.fov_half_angle_deg
                    .map_or(Field::Unavailable(NOT_IN_REPLAY), |half| {
                        Field::Value(format!("{:.0}°", half * 2.0))
                    }),
            ),
            Row::new(
                "Range",
                node.max_range_m
                    .map_or(Field::Unavailable(NOT_IN_REPLAY), |range| {
                        Field::Value(format!("{range:.0} m"))
                    }),
            ),
            Row::new("Captures this frame", Field::Value(captures.to_string())),
        ],
    };

    let events: Vec<String> = frame
        .safety_events
        .iter()
        .filter(|event| event.subject_id == node_id)
        .map(|event| {
            let state = if event.state.is_empty() {
                &event.reason
            } else {
                &event.state
            };
            let action = if event.blocked {
                "blocked"
            } else if event.command_clamped {
                "clamped"
            } else {
                "passed"
            };
            format!("{state}: {action}")
        })
        .collect();
    let safety = Section {
        title: "10  Safety",
        essential: true,
        rows: vec![
            Row::unavailable("Envelope", NOT_IN_REPLAY),
            Row::new(
                "Events this frame",
                Field::Value(if events.is_empty() {
                    "none".into()
                } else {
                    events.join("; ")
                }),
            ),
        ],
    };

    let executive = Section {
        title: "11  Onboard executive",
        essential: false,
        rows: vec![
            Row::unavailable("Supervisor state", "no onboard executive in simulation"),
            Row::unavailable("Active contingency", "no onboard executive in simulation"),
            Row::unavailable(
                "Pending operator request",
                "commands are not recorded in replays",
            ),
        ],
    };

    Some(vec![
        identity,
        actual,
        commanded,
        tasking,
        localization,
        dynamics,
        energy,
        link,
        sensors,
        safety,
        executive,
    ])
}

/// A drone's task at a time, from the recorded assignment history.
struct RecordedTask {
    task: Field<String>,
    last_change: Field<String>,
    changes: Field<String>,
}

/// The last event at or before `at_s` in `node_id`'s history: its task, when
/// and why it began, and how many changes came before. A drone the history
/// does not list, or one not yet assigned at `at_s`, reads as unavailable.
fn recorded_task(history: &serde_json::Value, node_id: &str, at_s: f32) -> RecordedTask {
    let events: Vec<&serde_json::Value> = history
        .get(node_id)
        .and_then(|events| events.as_array())
        .map(|events| {
            events
                .iter()
                .filter(|event| {
                    event
                        .get("timestamp_s")
                        .and_then(|t| t.as_f64())
                        .is_some_and(|t| t <= f64::from(at_s) + 1e-6)
                })
                .collect()
        })
        .unwrap_or_default();
    let Some(last) = events.last() else {
        let why = if history.get(node_id).is_some() {
            "no task assigned yet"
        } else {
            "this platform has no assignment history"
        };
        return RecordedTask {
            task: Field::Unavailable(why),
            last_change: Field::Unavailable(why),
            changes: Field::Unavailable(why),
        };
    };
    let text = |key: &str| last.get(key).and_then(|v| v.as_str()).unwrap_or("");
    let at = last
        .get("timestamp_s")
        .and_then(|t| t.as_f64())
        .unwrap_or(0.0);
    let mut change = format!("t = {at:.2} s, {}", text("kind"));
    if !text("reason").is_empty() {
        change.push_str(&format!(" ({})", text("reason")));
    }
    if !text("previous").is_empty() {
        change.push_str(&format!(", from {}", text("previous")));
    }
    if !text("to").is_empty() {
        change.push_str(&format!(", to {}", text("to")));
    }
    RecordedTask {
        task: Field::Value(text("task").to_string()),
        last_change: Field::Value(change),
        // The first event is the drone's first task, not a change.
        changes: Field::Value((events.len() - 1).to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(json: serde_json::Value) -> ReplayFrame {
        serde_json::from_value(json).expect("a replay frame")
    }

    fn minimal() -> ReplayFrame {
        frame(serde_json::json!({
            "timestamp_s": 4.0,
            "nodes": [{"node_id": "drone-1", "position": [1.0, 2.0, 30.0],
                       "velocity": [0.0, 0.0, 0.0], "is_mobile": true}]
        }))
    }

    fn row<'a>(sections: &'a [Section], label: &str) -> &'a Row {
        sections
            .iter()
            .flat_map(|section| &section.rows)
            .find(|row| row.label == label)
            .unwrap_or_else(|| panic!("no row {label}"))
    }

    #[test]
    fn the_eleven_sections_come_in_the_specified_order() {
        let sections = inspect_platform(&minimal(), "drone-1", None).unwrap();
        let titles: Vec<&str> = sections.iter().map(|s| s.title).collect();
        assert_eq!(titles.len(), 11);
        for (index, title) in titles.iter().enumerate() {
            assert!(title.starts_with(&format!("{} ", index + 1)), "{title}");
        }
    }

    #[test]
    fn what_a_replay_lacks_reads_as_unavailable_never_zero() {
        let sections = inspect_platform(&minimal(), "drone-1", None).unwrap();
        for label in [
            "Battery",
            "Field of view",
            "Range",
            "Role",
            "Current task",
            "Position σ",
            "Confidence",
            "Command",
            "Integrity",
        ] {
            let value = &row(&sections, label).value;
            assert!(
                matches!(value, Field::Unavailable(_)),
                "{label} should be unavailable, was {value:?}"
            );
            assert!(!value.display(|v| v.clone()).starts_with('0'), "{label}");
        }
        assert!(matches!(
            row(&sections, "Heading").value,
            Field::Unavailable(_)
        ));
    }

    #[test]
    fn present_fields_are_shown() {
        let frame = frame(serde_json::json!({
            "timestamp_s": 4.0,
            "nodes": [{"node_id": "drone-1", "position": [1.0, 2.0, 30.0],
                       "velocity": [3.0, 4.0, 0.0], "is_mobile": true,
                       "battery_fraction": 0.42, "fov_half_angle_deg": 30.0,
                       "max_range_m": 250.0}],
            "scan_mission_state": {
                "phase": "localizing", "scan_coverage_fraction": 0.5,
                "scan_coverage_threshold": 0.9, "completed_poi_count": 0,
                "total_poi_count": 0, "poi_statuses": [],
                "coordinator_drone_id": "drone-1",
                "localization_estimates": [{"drone_id": "drone-1", "timestamp_s": 4.0,
                    "position_estimate": [1.0, 2.0, 30.0], "heading_rad": 0.0,
                    "position_std_m": 1.25, "confidence": 0.8}]
            }
        }));
        let sections = inspect_platform(&frame, "drone-1", None).unwrap();
        assert_eq!(row(&sections, "Battery").value, Field::Value("42%".into()));
        assert_eq!(
            row(&sections, "Ground speed").value,
            Field::Value("5.0 m/s".into())
        );
        assert_eq!(
            row(&sections, "Role").value,
            Field::Value("mission coordinator".into())
        );
        assert_eq!(
            row(&sections, "Position σ").value,
            Field::Value("1.25 m".into())
        );
        assert_eq!(
            row(&sections, "Field of view").value,
            Field::Value("60°".into())
        );
    }

    #[test]
    fn legacy_estimator_outputs_are_labelled_and_integrity_is_not_inferred() {
        let sections = inspect_platform(&minimal(), "drone-1", None).unwrap();
        for label in ["Confidence", "Localization timeout", "Health"] {
            assert!(row(&sections, label).legacy, "{label} is legacy");
        }
        let integrity = row(&sections, "Integrity");
        assert!(!integrity.legacy);
        assert_eq!(
            integrity.value,
            Field::Unavailable("no validated integrity adapter")
        );
    }

    #[test]
    fn the_essential_sections_are_the_ones_section_4_4_names() {
        let sections = inspect_platform(&minimal(), "drone-1", None).unwrap();
        let essential: Vec<&str> = sections
            .iter()
            .filter(|section| section.essential)
            .map(|section| section.title)
            .collect();
        assert_eq!(
            essential,
            [
                "2  Actual state",
                "3  Commanded state",
                "5  Localization",
                "7  Battery and return",
                "8  Link",
                "10  Safety"
            ]
        );
    }

    #[test]
    fn a_platform_not_in_the_frame_does_not_resolve() {
        assert!(inspect_platform(&minimal(), "drone-2", None).is_none());
    }

    fn history() -> serde_json::Value {
        serde_json::json!({
            "drone-1": [
                {"timestamp_s": 0.0, "kind": "assign", "task": "search", "reason": "initial"},
                {"timestamp_s": 3.0, "kind": "reassign", "task": "poi:poi-02",
                 "previous": "search", "reason": "poi_assigned"},
                {"timestamp_s": 9.0, "kind": "handoff", "task": "search",
                 "previous": "poi:poi-02", "reason": "safe_handoff", "to": "drone-2"}
            ]
        })
    }

    /// P0b: with a recorded history, the task and its last change come from
    /// it at the frame's time, not from the POI statuses.
    #[test]
    fn the_task_comes_from_the_recorded_history_at_the_frame_time() {
        let history = history();
        let sections = inspect_platform(&minimal(), "drone-1", Some(&history)).unwrap();
        assert_eq!(
            row(&sections, "Current task").value,
            Field::Value("poi:poi-02".into())
        );
        assert_eq!(
            row(&sections, "Last change").value,
            Field::Value("t = 3.00 s, reassign (poi_assigned), from search".into())
        );
        assert_eq!(
            row(&sections, "Task changes").value,
            Field::Value("1".into())
        );

        let mut later = minimal();
        later.timestamp_s = 12.0;
        let sections = inspect_platform(&later, "drone-1", Some(&history)).unwrap();
        assert_eq!(
            row(&sections, "Current task").value,
            Field::Value("search".into())
        );
        assert_eq!(
            row(&sections, "Last change").value,
            Field::Value("t = 9.00 s, handoff (safe_handoff), from poi:poi-02, to drone-2".into())
        );
        assert_eq!(
            row(&sections, "Task changes").value,
            Field::Value("2".into())
        );
    }

    /// A platform the history does not list, a time before its first task,
    /// and a replay with no history all read as unavailable, never as a task.
    #[test]
    fn a_missing_history_reads_as_unavailable() {
        let history = history();
        let other = frame(serde_json::json!({
            "timestamp_s": 4.0,
            "nodes": [{"node_id": "ground-1", "position": [0.0, 0.0, 0.0],
                       "velocity": [0.0, 0.0, 0.0], "is_mobile": false}]
        }));
        let sections = inspect_platform(&other, "ground-1", Some(&history)).unwrap();
        assert!(matches!(
            row(&sections, "Current task").value,
            Field::Unavailable(_)
        ));

        let mut early = minimal();
        early.timestamp_s = -1.0;
        let sections = inspect_platform(&early, "drone-1", Some(&history)).unwrap();
        assert!(matches!(
            row(&sections, "Current task").value,
            Field::Unavailable(_)
        ));

        let sections = inspect_platform(&minimal(), "drone-1", None).unwrap();
        assert!(matches!(
            row(&sections, "Last change").value,
            Field::Unavailable(_)
        ));
    }
}
