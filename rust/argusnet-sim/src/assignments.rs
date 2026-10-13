//! Who was doing what, and when: the per-drone assignment history
//! (PLANNING_EFFICIENCY_PLAN P0).
//!
//! A drone's *task* is what it flies on a tick. Stage 2's recall tables win,
//! in their own order (egress, then a POI, then a frontier goal), and
//! otherwise the platform's own controller decides: a search lane, an
//! explorer, a target it follows. A search-then-follow drone's task is
//! `search` until its sensor engages the target, and `follow:<target>` after.
//! Every change is an event, so a drone helping on a POI is never mistaken
//! for an idle one, whatever the POI statuses say.
//!
//! The history is a record of the run, not run state: it is rebuilt from the
//! tick's committed state after each tick, cleared by a reset, and like the
//! replay's frames a resumed run's history starts at its checkpoint.

use std::collections::BTreeMap;

use serde::Serialize;

/// One change in a drone's task.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AssignmentEvent {
    pub timestamp_s: f64,
    /// `assign` (the first task the run sees), `reassign` (a different task),
    /// `handoff` (a POI passed to another drone) or `team_join` (a second
    /// drone added to a POI).
    pub kind: &'static str,
    pub task: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub previous: Option<String>,
    /// A stable reason code.
    pub reason: &'static str,
    /// The drone a `handoff` passed its POI to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub to: Option<String>,
}

/// A handoff or team join stage 9 applied on the tick, which no task change
/// shows: the reference leaves a drone that hands its POI on flying to it.
#[derive(Debug, Clone, PartialEq)]
pub struct PoiTransfer {
    pub kind: &'static str,
    pub drone_id: String,
    pub poi_id: String,
    pub to: Option<String>,
    pub reason: &'static str,
}

/// Every drone's events, in the order they happened.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct AssignmentHistory {
    current: BTreeMap<String, String>,
    events: BTreeMap<String, Vec<AssignmentEvent>>,
}

impl AssignmentHistory {
    /// Record the tick: each drone's task (in fleet order), then the
    /// transfers stage 9 applied.
    pub fn observe(
        &mut self,
        timestamp_s: f64,
        tasks: &[(String, String)],
        transfers: &[PoiTransfer],
    ) {
        for (drone_id, task) in tasks {
            let previous = self.current.get(drone_id);
            if previous == Some(task) {
                continue;
            }
            let event = AssignmentEvent {
                timestamp_s,
                kind: if previous.is_some() {
                    "reassign"
                } else {
                    "assign"
                },
                task: task.clone(),
                previous: previous.cloned(),
                reason: reason(previous.map(String::as_str), task),
                to: None,
            };
            self.current.insert(drone_id.clone(), task.clone());
            self.events.entry(drone_id.clone()).or_default().push(event);
        }
        for transfer in transfers {
            self.events
                .entry(transfer.drone_id.clone())
                .or_default()
                .push(AssignmentEvent {
                    timestamp_s,
                    kind: transfer.kind,
                    task: format!("poi:{}", transfer.poi_id),
                    previous: None,
                    reason: transfer.reason,
                    to: transfer.to.clone(),
                });
        }
    }

    /// The task `drone_id` was last recorded on.
    pub fn current(&self, drone_id: &str) -> Option<&str> {
        self.current.get(drone_id).map(String::as_str)
    }

    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    pub fn clear(&mut self) {
        self.current.clear();
        self.events.clear();
    }

    pub fn events(&self) -> &BTreeMap<String, Vec<AssignmentEvent>> {
        &self.events
    }

    /// `meta.assignment_history`: drone id to its events.
    pub fn document(&self) -> serde_json::Value {
        serde_json::to_value(&self.events).expect("events serialise")
    }
}

/// Why a drone went from `previous` to `task`, read from the transition.
fn reason(previous: Option<&str>, task: &str) -> &'static str {
    let Some(previous) = previous else {
        return "initial";
    };
    if task == "egress" {
        "egress"
    } else if task.starts_with("poi:") {
        "poi_assigned"
    } else if task == "frontier" {
        "frontier_goal"
    } else if previous == "egress" {
        "egress_cleared"
    } else if previous.starts_with("poi:") {
        "poi_released"
    } else if previous == "frontier" {
        "frontier_cleared"
    } else if task.starts_with("follow:") {
        "target_engaged"
    } else {
        "controller"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tasks(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(drone, task)| (drone.to_string(), task.to_string()))
            .collect()
    }

    #[test]
    fn a_task_change_is_one_event_with_its_reason() {
        let mut history = AssignmentHistory::default();
        history.observe(0.0, &tasks(&[("a", "search"), ("b", "follow:t1")]), &[]);
        history.observe(1.0, &tasks(&[("a", "search"), ("b", "follow:t1")]), &[]);
        history.observe(2.0, &tasks(&[("a", "poi:p1"), ("b", "follow:t1")]), &[]);
        history.observe(9.0, &tasks(&[("a", "search"), ("b", "egress")]), &[]);
        let a = &history.events()["a"];
        assert_eq!(a.len(), 3);
        assert_eq!((a[0].kind, a[0].reason), ("assign", "initial"));
        assert_eq!(
            (a[1].kind, a[1].reason, a[1].timestamp_s),
            ("reassign", "poi_assigned", 2.0)
        );
        assert_eq!(a[1].previous.as_deref(), Some("search"));
        assert_eq!(a[2].reason, "poi_released");
        let b = &history.events()["b"];
        assert_eq!(b.last().unwrap().reason, "egress");
    }

    #[test]
    fn a_handoff_is_recorded_on_the_drone_that_gave_it() {
        let mut history = AssignmentHistory::default();
        history.observe(0.0, &tasks(&[("a", "poi:p1"), ("b", "search")]), &[]);
        history.observe(
            5.0,
            &tasks(&[("a", "poi:p1"), ("b", "search")]),
            &[PoiTransfer {
                kind: "handoff",
                drone_id: "a".into(),
                poi_id: "p1".into(),
                to: Some("b".into()),
                reason: "low_battery",
            }],
        );
        let last = history.events()["a"].last().unwrap().clone();
        assert_eq!(last.kind, "handoff");
        assert_eq!(last.to.as_deref(), Some("b"));
        let document = history.document();
        assert_eq!(document["a"][1]["to"], "b");
        assert!(document["a"][0].get("to").is_none());
    }

    #[test]
    fn engaging_a_target_and_clearing_egress_have_their_reasons() {
        assert_eq!(reason(Some("search"), "follow:t1"), "target_engaged");
        assert_eq!(reason(Some("egress"), "search"), "egress_cleared");
        assert_eq!(reason(Some("frontier"), "explore"), "frontier_cleared");
        assert_eq!(reason(Some("explore"), "frontier"), "frontier_goal");
    }
}
