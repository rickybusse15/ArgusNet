//! Allocation lines (console parity V4, planning-efficiency P1): a line
//! from each drone to the POI it holds, so the fleet's assignment reads at a
//! glance on the canvas. Drawn from the frame's own mission state -- each
//! POI's assigned drone and position -- and nothing else: a frame without
//! mission state, or a POI without a recorded position, draws no line.

use bevy::prelude::*;

use crate::replay::{ReplayFrame, ReplayState};
use crate::state::MissionOverlaySettings;

/// One drone-to-POI line.
#[derive(Debug, Clone, PartialEq)]
pub struct AllocationLine {
    pub drone_id: String,
    pub poi_id: String,
    pub from: Vec3,
    pub to: Vec3,
    /// The drone is dwelling on it (`active`), not yet there (`pending`).
    pub active: bool,
}

/// The lines for `frame`: every POI not yet complete with an assigned drone
/// in the frame and a recorded position, in the frame's POI order.
pub fn allocation_lines(frame: &ReplayFrame) -> Vec<AllocationLine> {
    let Some(mission) = frame.scan_mission_state.as_ref() else {
        return Vec::new();
    };
    mission
        .poi_statuses
        .iter()
        .filter(|poi| poi.status != "complete")
        .filter_map(|poi| {
            let drone_id = poi.assigned_drone_id.as_ref()?;
            let position = poi.position?;
            let node = frame.nodes.iter().find(|node| &node.node_id == drone_id)?;
            Some(AllocationLine {
                drone_id: drone_id.clone(),
                poi_id: poi.poi_id.clone(),
                from: Vec3::from_array(node.position),
                to: Vec3::from_array(position),
                active: poi.status == "active",
            })
        })
        .collect()
}

/// Allocation lines: violet, a hue no other overlay uses (range rays are
/// amber, trails green, mapped wires cyan), brighter for a POI being
/// inspected than for one the drone is still flying to.
pub const ALLOCATION_COLOR: [f32; 3] = [0.78, 0.45, 1.0];

pub fn allocation_lines_enabled(overlay: Res<MissionOverlaySettings>) -> bool {
    overlay.show_allocation_lines
}

/// Draw the cursor frame's allocation lines, with a ring at each POI.
pub fn draw_allocation_lines_system(mut gizmos: Gizmos, replay_state: Res<ReplayState>) {
    let Some(frame) = replay_state.current_frame() else {
        return;
    };
    let [r, g, b] = ALLOCATION_COLOR;
    for line in allocation_lines(frame) {
        let alpha = if line.active { 0.95 } else { 0.5 };
        let color = Color::srgba(r, g, b, alpha);
        gizmos.line(line.from, line.to, color);
        let ring = bevy::math::Isometry3d::new(line.to + Vec3::Z, Quat::IDENTITY);
        gizmos.circle(ring, 6.0, color);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(json: serde_json::Value) -> ReplayFrame {
        serde_json::from_value(json).expect("frame parses")
    }

    /// A line for each held POI not yet complete, from its drone to it;
    /// none for a complete POI, an unassigned one, a POI without a
    /// position, or a drone not in the frame.
    #[test]
    fn each_held_poi_gets_a_line_from_its_drone() {
        let frame = frame(serde_json::json!({
            "timestamp_s": 10.0,
            "nodes": [
                {"node_id": "d1", "position": [0.0, 0.0, 50.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": true},
                {"node_id": "d2", "position": [10.0, 0.0, 50.0], "velocity": [0.0, 0.0, 0.0], "is_mobile": true}
            ],
            "scan_mission_state": {
                "phase": "inspecting",
                "scan_coverage_fraction": 0.9,
                "scan_coverage_threshold": 0.8,
                "localization_estimates": [],
                "completed_poi_count": 1,
                "total_poi_count": 6,
                "poi_statuses": [
                    {"poi_id": "p1", "status": "active", "assigned_drone_id": "d1", "position": [100.0, 0.0, 0.0]},
                    {"poi_id": "p2", "status": "pending", "assigned_drone_id": "d2", "position": [0.0, 100.0, 0.0]},
                    {"poi_id": "p3", "status": "complete", "assigned_drone_id": "d1", "position": [5.0, 5.0, 0.0]},
                    {"poi_id": "p4", "status": "pending", "position": [7.0, 7.0, 0.0]},
                    {"poi_id": "p5", "status": "pending", "assigned_drone_id": "d2"},
                    {"poi_id": "p6", "status": "pending", "assigned_drone_id": "gone", "position": [1.0, 1.0, 0.0]}
                ]
            }
        }));
        let lines = allocation_lines(&frame);
        assert_eq!(
            lines
                .iter()
                .map(|l| (l.drone_id.as_str(), l.poi_id.as_str(), l.active))
                .collect::<Vec<_>>(),
            [("d1", "p1", true), ("d2", "p2", false)]
        );
        assert_eq!(lines[0].from, Vec3::new(0.0, 0.0, 50.0));
        assert_eq!(lines[0].to, Vec3::new(100.0, 0.0, 0.0));
    }

    /// No mission state, no lines.
    #[test]
    fn a_frame_without_mission_state_draws_none() {
        let frame = frame(serde_json::json!({"timestamp_s": 0.0}));
        assert!(allocation_lines(&frame).is_empty());
    }
}
