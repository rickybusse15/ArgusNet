use std::collections::HashMap;

use bevy::prelude::Resource;
use bevy::prelude::Vec3;
use serde::de::{self, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::Value;

/// Custom deserializer that accepts either a flat `[f64, ...]` array
/// or a nested `[[f64, ...], ...]` matrix (flattening the latter).
fn deserialize_flat_f64_vec<'de, D>(deserializer: D) -> Result<Option<Vec<f64>>, D::Error>
where
    D: Deserializer<'de>,
{
    struct FlatF64VecVisitor;

    impl<'de> Visitor<'de> for FlatF64VecVisitor {
        type Value = Option<Vec<f64>>;

        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("null, a flat array of f64, or a nested 2D array of f64")
        }

        fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
            Ok(None)
        }

        fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
            let mut result = Vec::new();
            // Peek at the serde_json::Value representation to handle both forms.
            while let Some(element) = seq.next_element::<serde_json::Value>()? {
                match element {
                    serde_json::Value::Number(n) => {
                        result.push(n.as_f64().unwrap_or(0.0));
                    }
                    serde_json::Value::Array(inner) => {
                        for item in inner {
                            if let serde_json::Value::Number(n) = item {
                                result.push(n.as_f64().unwrap_or(0.0));
                            }
                        }
                    }
                    _ => {}
                }
            }
            if result.is_empty() {
                Ok(None)
            } else {
                Ok(Some(result))
            }
        }
    }

    deserializer.deserialize_option(OptionVisitor(FlatF64VecVisitor))
}

/// Wrapper to handle the `Option` layer before delegating to the inner visitor.
struct OptionVisitor<V>(V);

impl<'de, V: Visitor<'de, Value = Option<Vec<f64>>>> Visitor<'de> for OptionVisitor<V> {
    type Value = Option<Vec<f64>>;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        self.0.expecting(formatter)
    }

    fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
        Ok(None)
    }

    fn visit_some<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
        deserializer.deserialize_seq(self.0)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, seq: A) -> Result<Self::Value, A::Error> {
        self.0.visit_seq(seq)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MissionZone {
    pub zone_id: String,
    pub zone_type: String,
    pub center: [f32; 3],
    pub radius_m: f32,
    #[serde(default = "default_priority")]
    pub priority: u32,
    #[serde(default)]
    pub label: String,
}

fn default_priority() -> u32 {
    1
}

#[derive(Debug, Clone, Deserialize)]
pub struct TerrainViewerMesh {
    pub x_min_m: f32,
    pub x_max_m: f32,
    pub y_min_m: f32,
    pub y_max_m: f32,
    pub rows: usize,
    pub cols: usize,
    pub heights_m: Vec<Vec<f32>>,
}

impl TerrainViewerMesh {
    pub fn sample_height(&self, x_m: f32, y_m: f32) -> Option<f32> {
        if self.rows < 2 || self.cols < 2 || self.heights_m.len() < 2 {
            return None;
        }
        if x_m < self.x_min_m || x_m > self.x_max_m || y_m < self.y_min_m || y_m > self.y_max_m {
            return None;
        }
        let span_x = (self.x_max_m - self.x_min_m).max(1.0);
        let span_y = (self.y_max_m - self.y_min_m).max(1.0);
        let tx =
            ((x_m - self.x_min_m) / span_x).clamp(0.0, 1.0) * (self.cols.saturating_sub(1) as f32);
        let ty =
            ((y_m - self.y_min_m) / span_y).clamp(0.0, 1.0) * (self.rows.saturating_sub(1) as f32);
        let col = tx.floor().min((self.cols.saturating_sub(2)) as f32) as usize;
        let row = ty.floor().min((self.rows.saturating_sub(2)) as f32) as usize;
        let ax = tx - (col as f32);
        let ay = ty - (row as f32);
        let z00 = *self.heights_m.get(row)?.get(col)?;
        let z10 = *self.heights_m.get(row)?.get(col + 1)?;
        let z01 = *self.heights_m.get(row + 1)?.get(col)?;
        let z11 = *self.heights_m.get(row + 1)?.get(col + 1)?;
        let z0 = z00 + ((z10 - z00) * ax);
        let z1 = z01 + ((z11 - z01) * ax);
        Some(z0 + ((z1 - z0) * ay))
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReplayDocument {
    pub meta: Option<Value>,
    pub summary: Option<Value>,
    pub frames: Vec<ReplayFrame>,
}

impl ReplayDocument {
    pub fn zones(&self) -> Vec<MissionZone> {
        self.meta
            .as_ref()
            .and_then(|m| m.get("zones"))
            .and_then(|v| serde_json::from_value::<Vec<MissionZone>>(v.clone()).ok())
            .unwrap_or_default()
    }

    pub fn terrain_viewer_mesh(&self) -> Option<TerrainViewerMesh> {
        self.meta
            .as_ref()
            .and_then(|m| m.get("terrain"))
            .and_then(|terrain| terrain.get("viewer_mesh"))
            .and_then(|mesh| serde_json::from_value::<TerrainViewerMesh>(mesh.clone()).ok())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct ReplayFrame {
    pub timestamp_s: f32,
    #[serde(default)]
    pub tracks: Vec<TrackState>,
    #[serde(default)]
    pub truths: Vec<TruthState>,
    #[serde(default)]
    pub nodes: Vec<NodeState>,
    #[serde(default)]
    pub observations: Vec<ObservationState>,
    #[serde(default)]
    pub rejected_observations: Vec<RejectedObservation>,
    #[serde(default)]
    pub generation_rejections: Vec<RejectedObservation>,
    #[serde(default)]
    pub metrics: FrameMetrics,
    #[serde(default)]
    pub mapping_state: Option<MappingState>,
    #[serde(default)]
    pub localization_state: Option<LocalizationState>,
    #[serde(default)]
    pub inspection_events: Vec<InspectionEvent>,
    #[serde(default)]
    pub deconfliction_events: Vec<DeconflictionEvent>,
    #[serde(default)]
    pub scan_mission_state: Option<ScanMissionState>,
    #[serde(default)]
    pub tracking_mission_state: Option<TrackingMissionState>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct ScanMissionState {
    pub phase: String,
    pub scan_coverage_fraction: f32,
    pub scan_coverage_threshold: f32,
    pub localization_estimates: Vec<LocalizationEstimate>,
    pub poi_statuses: Vec<PoiStatus>,
    pub completed_poi_count: usize,
    pub total_poi_count: usize,
    #[serde(default)]
    pub phase_started_at_s: f32,
    /// Flat array of (x, y, terrain_height) triples from Python.
    /// Python's to_jsonable() flattens tuple-of-tuples into a single Vec<f32>.
    #[serde(default)]
    pub newly_scanned_cells: Vec<f32>,
    /// True when the localizing phase advanced via timeout rather than convergence.
    #[serde(default)]
    pub localization_timed_out: bool,
    /// The drone_id elected as mission coordinator, if any.
    #[serde(default)]
    pub coordinator_drone_id: Option<String>,
    /// Per-drone return-to-home progress. Non-empty only during the egress phase.
    #[serde(default)]
    pub egress_progress: Vec<EgressProgress>,
    /// Safety-gate rejections recorded since the previous frame.
    #[serde(default)]
    pub safety_events: Vec<SafetyEventRecord>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TrackingMissionState {
    /// Safety-gate rejections recorded since the previous frame.
    #[serde(default)]
    pub safety_events: Vec<SafetyEventRecord>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct SafetyEventRecord {
    pub timestamp_s: f32,
    pub drone_id: String,
    pub task_type: String,
    /// Replay JSON serialises tuples as arrays; accept either fixed-shape
    /// `[x, y]` or a flat 2-element vec.
    #[serde(default)]
    pub target_xy_m: [f32; 2],
    #[serde(default)]
    pub target_z_m: f32,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub violations: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LocalizationEstimate {
    pub drone_id: String,
    pub timestamp_s: f32,
    pub position_estimate: [f32; 3],
    pub heading_rad: f32,
    pub position_std_m: f32,
    pub confidence: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct PoiStatus {
    pub poi_id: String,
    pub status: String,
    #[serde(default)]
    pub assigned_drone_id: Option<String>,
    #[serde(default)]
    pub dwell_accumulated_s: f32,
    #[serde(default)]
    pub position: Option<[f32; 3]>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct EgressProgress {
    pub drone_id: String,
    pub distance_to_home_m: f32,
    pub home_position: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct TrackState {
    pub track_id: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    #[serde(default)]
    pub measurement_std_m: f32,
    #[serde(default)]
    pub update_count: u32,
    #[serde(default)]
    pub stale_steps: u32,
    #[serde(default, deserialize_with = "deserialize_flat_f64_vec")]
    pub covariance: Option<Vec<f64>>,
    /// IMM CV-model weight in [0, 1]; None when not present in replay.
    #[serde(default)]
    pub mode_probability_cv: Option<f32>,
    /// Node IDs that contributed to the most recent update.
    #[serde(default)]
    pub contributing_node_ids: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TruthState {
    pub target_id: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
}

#[derive(Debug, Clone, Deserialize)]
pub struct NodeState {
    pub node_id: String,
    pub position: [f32; 3],
    pub velocity: [f32; 3],
    pub is_mobile: bool,
    #[serde(default = "default_health")]
    pub health: f32,
    #[serde(default = "default_sensor_type")]
    pub sensor_type: String,
    #[serde(default)]
    pub fov_half_angle_deg: Option<f32>,
    #[serde(default)]
    pub max_range_m: Option<f32>,
}

fn default_health() -> f32 {
    1.0
}

fn default_sensor_type() -> String {
    "optical".into()
}

#[derive(Debug, Clone, Deserialize)]
pub struct ObservationState {
    pub node_id: String,
    pub target_id: String,
    pub origin: [f32; 3],
    pub direction: [f32; 3],
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    #[serde(default)]
    pub timestamp_s: f32,
}

fn default_confidence() -> f32 {
    1.0
}

#[derive(Debug, Clone, Deserialize)]
pub struct RejectedObservation {
    pub node_id: String,
    pub target_id: String,
    #[serde(default)]
    pub reason: String,
    #[serde(default)]
    pub detail: String,
    #[serde(default)]
    pub timestamp_s: f32,
    #[serde(default)]
    pub origin: Option<[f32; 3]>,
    #[serde(default)]
    pub attempted_point: Option<[f32; 3]>,
    #[serde(default)]
    pub closest_point: Option<[f32; 3]>,
    #[serde(default)]
    pub blocker_type: String,
    #[serde(default)]
    pub first_hit_range_m: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct FrameMetrics {
    #[serde(default)]
    pub mean_error_m: Option<f32>,
    #[serde(default)]
    pub max_error_m: Option<f32>,
    #[serde(default)]
    pub active_track_count: u32,
    #[serde(default)]
    pub observation_count: u32,
    #[serde(default)]
    pub accepted_observation_count: u32,
    #[serde(default)]
    pub rejected_observation_count: u32,
    #[serde(default)]
    pub mean_measurement_std_m: Option<f32>,
    #[serde(default)]
    pub rejection_counts: HashMap<String, u32>,
    #[serde(default)]
    pub track_errors_m: HashMap<String, f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct MappingState {
    #[serde(default)]
    pub coverage_fraction: f32,
    #[serde(default)]
    pub covered_cells: u32,
    #[serde(default)]
    pub total_cells: u32,
    #[serde(default)]
    pub mean_revisits: f32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct LocalizationState {
    #[serde(default)]
    pub active_localizations: u32,
    #[serde(default)]
    pub mean_position_std_m: f32,
    #[serde(default)]
    pub mean_observation_confidence: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct InspectionEvent {
    pub zone_id: String,
    pub node_id: String,
    pub event_type: String,
    pub timestamp_s: f32,
    #[serde(default)]
    pub zone_coverage_fraction: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DeconflictionEvent {
    pub yielding_drone_id: String,
    pub conflicting_drone_id: String,
    pub predicted_separation_m: f32,
    pub resolution: String,
    pub timestamp_s: f32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    Track,
    Truth,
    Node,
}

#[derive(Debug, Clone)]
pub struct RuntimeMarker {
    pub kind: MarkerKind,
    pub label: String,
    pub position: Vec3,
    pub velocity: Vec3,
    /// For `Node` markers: whether this node is a mobile platform (drone).
    /// `None` for tracks and truths.
    pub is_mobile: Option<bool>,
}

#[derive(Debug, Clone, Resource)]
pub struct ReplayState {
    pub document: Option<ReplayDocument>,
    pub playing: bool,
    pub playback_speed: f32,
    pub frame_index: usize,
    pub accumulator_s: f32,
}

impl Default for ReplayState {
    fn default() -> Self {
        Self {
            document: None,
            playing: false,
            playback_speed: 1.0,
            frame_index: 0,
            accumulator_s: 0.0,
        }
    }
}

impl ReplayState {
    pub fn new(document: Option<ReplayDocument>) -> Self {
        Self {
            document,
            ..Self::default()
        }
    }

    pub fn frame_count(&self) -> usize {
        self.document
            .as_ref()
            .map(|document| document.frames.len())
            .unwrap_or(0)
    }

    pub fn current_frame(&self) -> Option<&ReplayFrame> {
        let document = self.document.as_ref()?;
        document.frames.get(
            self.frame_index
                .min(document.frames.len().saturating_sub(1)),
        )
    }

    pub fn current_timestamp_s(&self) -> f32 {
        self.current_frame()
            .map(|frame| frame.timestamp_s)
            .unwrap_or(0.0)
    }

    pub fn step_to(&mut self, frame_index: usize) {
        let frame_count = self.frame_count();
        self.frame_index = frame_index.min(frame_count.saturating_sub(1));
        self.accumulator_s = 0.0;
    }

    pub fn advance(&mut self, delta_s: f32) {
        if !self.playing || self.frame_count() <= 1 {
            return;
        }

        self.accumulator_s += delta_s.max(0.0) * self.playback_speed.max(0.1);
        let frame_step = self.frame_step_s();
        while self.accumulator_s >= frame_step {
            self.accumulator_s -= frame_step;
            if self.frame_index + 1 >= self.frame_count() {
                self.frame_index = self.frame_count().saturating_sub(1);
                self.playing = false;
                self.accumulator_s = 0.0;
                break;
            }
            self.frame_index += 1;
            if self.frame_index + 1 >= self.frame_count() {
                self.playing = false;
                self.accumulator_s = 0.0;
                break;
            }
        }
    }

    pub fn current_markers(&self) -> Vec<RuntimeMarker> {
        let Some(frame) = self.current_frame() else {
            return Vec::new();
        };

        let mut markers =
            Vec::with_capacity(frame.tracks.len() + frame.truths.len() + frame.nodes.len());
        for track in &frame.tracks {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Track,
                label: track.track_id.clone(),
                position: Vec3::from_array(track.position),
                velocity: Vec3::from_array(track.velocity),
                is_mobile: None,
            });
        }
        for truth in &frame.truths {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Truth,
                label: truth.target_id.clone(),
                position: Vec3::from_array(truth.position),
                velocity: Vec3::from_array(truth.velocity),
                is_mobile: None,
            });
        }
        for node in &frame.nodes {
            markers.push(RuntimeMarker {
                kind: MarkerKind::Node,
                label: node.node_id.clone(),
                position: Vec3::from_array(node.position),
                velocity: Vec3::from_array(node.velocity),
                is_mobile: Some(node.is_mobile),
            });
        }
        markers
    }

    pub fn trail_points(&self, marker: &RuntimeMarker, max_samples: usize) -> Vec<Vec3> {
        let Some(document) = self.document.as_ref() else {
            return Vec::new();
        };
        let start = self
            .frame_index
            .saturating_sub(max_samples.saturating_sub(1));
        let mut points = Vec::new();
        for frame in &document.frames[start
            ..=self
                .frame_index
                .min(document.frames.len().saturating_sub(1))]
        {
            match marker.kind {
                MarkerKind::Track => {
                    if let Some(track) = frame
                        .tracks
                        .iter()
                        .find(|item| item.track_id == marker.label)
                    {
                        points.push(Vec3::from_array(track.position));
                    }
                }
                MarkerKind::Truth => {
                    if let Some(truth) = frame
                        .truths
                        .iter()
                        .find(|item| item.target_id == marker.label)
                    {
                        points.push(Vec3::from_array(truth.position));
                    }
                }
                MarkerKind::Node => {
                    if let Some(node) = frame.nodes.iter().find(|item| item.node_id == marker.label)
                    {
                        points.push(Vec3::from_array(node.position));
                    }
                }
            }
        }
        points
    }

    pub fn frame_step_s(&self) -> f32 {
        let Some(document) = self.document.as_ref() else {
            return 0.25;
        };
        if let Some(meta) = &document.meta {
            if let Some(value) = meta.get("dt_s").and_then(|value| value.as_f64()) {
                return value.max(0.01) as f32;
            }
        }
        if document.frames.len() >= 2 {
            return (document.frames[1].timestamp_s - document.frames[0].timestamp_s)
                .abs()
                .max(0.01);
        }
        0.25
    }
}

impl TryFrom<Value> for ReplayDocument {
    type Error = serde_json::Error;

    fn try_from(value: Value) -> Result<Self, Self::Error> {
        serde_json::from_value(value)
    }
}

// ---------------------------------------------------------------------------
// Live-stream conversions (feature = "live-stream")
//
// These From impls allow the viewer to accept frames directly from the gRPC
// server without a JSON round-trip.  They are compiled only when the
// `live-stream` feature is enabled to avoid pulling tonic/prost into default
// builds.
// ---------------------------------------------------------------------------

#[cfg(feature = "live-stream")]
mod live_stream {
    use argusnet_proto::pb;

    use super::{NodeState, TrackState, TruthState};

    impl From<pb::TrackState> for TrackState {
        fn from(t: pb::TrackState) -> Self {
            let pos = t.position.unwrap_or_default();
            let vel = t.velocity.unwrap_or_default();
            let covariance = if t.covariance_row_major.is_empty() {
                None
            } else {
                Some(t.covariance_row_major)
            };
            let mode_probability_cv = if t.mode_probability_cv > 0.0 {
                Some(t.mode_probability_cv as f32)
            } else {
                None
            };
            TrackState {
                track_id: t.track_id,
                position: [pos.x_m as f32, pos.y_m as f32, pos.z_m as f32],
                velocity: [vel.x_m as f32, vel.y_m as f32, vel.z_m as f32],
                measurement_std_m: t.measurement_std_m,
                update_count: t.update_count,
                stale_steps: t.stale_steps,
                covariance,
                mode_probability_cv,
                contributing_node_ids: t.contributing_nodes,
            }
        }
    }

    impl From<pb::TruthState> for TruthState {
        fn from(t: pb::TruthState) -> Self {
            let pos = t.position.unwrap_or_default();
            let vel = t.velocity.unwrap_or_default();
            TruthState {
                target_id: t.target_id,
                position: [pos.x_m as f32, pos.y_m as f32, pos.z_m as f32],
                velocity: [vel.x_m as f32, vel.y_m as f32, vel.z_m as f32],
            }
        }
    }

    impl From<pb::NodeState> for NodeState {
        fn from(n: pb::NodeState) -> Self {
            let pos = n.position.unwrap_or_default();
            let vel = n.velocity.unwrap_or_default();
            NodeState {
                node_id: n.node_id,
                position: [pos.x_m as f32, pos.y_m as f32, pos.z_m as f32],
                velocity: [vel.x_m as f32, vel.y_m as f32, vel.z_m as f32],
                is_mobile: n.is_mobile,
                health: n.health,
                sensor_type: n.sensor_type,
                fov_half_angle_deg: if n.fov_half_angle_deg == 0.0 {
                    None
                } else {
                    Some(n.fov_half_angle_deg)
                },
                max_range_m: if n.max_range_m == 0.0 {
                    None
                } else {
                    Some(n.max_range_m)
                },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::{MarkerKind, ReplayDocument, ReplayState};

    #[test]
    fn replay_state_advances_and_stops_at_last_frame() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.5},
            "frames": [
                {"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []},
                {"timestamp_s": 0.5, "tracks": [], "truths": [], "nodes": []},
                {"timestamp_s": 1.0, "tracks": [], "truths": [], "nodes": []}
            ]
        }))
        .unwrap();
        let mut replay = ReplayState::new(Some(document));
        replay.playing = true;
        replay.advance(1.1);
        assert_eq!(replay.frame_index, 2);
        assert!(!replay.playing);
    }

    #[test]
    fn covariance_flat_array_deserializes() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "t1", "position": [1.0,2.0,3.0], "velocity": [0.0,0.0,0.0],
                             "covariance": [1.0,0.0,0.0,0.0, 0.0,1.0,0.0,0.0, 0.0,0.0,1.0,0.0, 0.0,0.0,0.0,1.0]}],
                "truths": [], "nodes": []
            }]
        })).unwrap();
        let cov = document.frames[0].tracks[0].covariance.as_ref().unwrap();
        assert_eq!(cov.len(), 16);
        assert!((cov[0] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_nested_array_deserializes() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "t1", "position": [1.0,2.0,3.0], "velocity": [0.0,0.0,0.0],
                             "covariance": [[1.0,0.0],[0.0,2.0]]}],
                "truths": [], "nodes": []
            }]
        }))
        .unwrap();
        let cov = document.frames[0].tracks[0].covariance.as_ref().unwrap();
        assert_eq!(cov.len(), 4);
        assert!((cov[3] - 2.0).abs() < 1e-9);
    }

    #[test]
    fn covariance_null_deserializes() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "t1", "position": [1.0,2.0,3.0], "velocity": [0.0,0.0,0.0]}],
                "truths": [], "nodes": []
            }]
        })).unwrap();
        assert!(document.frames[0].tracks[0].covariance.is_none());
    }

    #[test]
    fn mapping_localization_inspection_deserializes() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{"timestamp_s": 1.0, "tracks": [], "truths": [], "nodes": [],
                "mapping_state": {"coverage_fraction": 0.42, "covered_cells": 10,
                                  "total_cells": 24, "mean_revisits": 1.5},
                "localization_state": {"active_localizations": 2,
                                       "mean_position_std_m": 12.3,
                                       "mean_observation_confidence": 0.85},
                "inspection_events": [{"zone_id": "z0", "node_id": "d0",
                                       "event_type": "entered", "timestamp_s": 1.0,
                                       "zone_coverage_fraction": 0.05}]
            }]
        }))
        .unwrap();
        let f = &document.frames[0];
        assert!((f.mapping_state.as_ref().unwrap().coverage_fraction - 0.42).abs() < 1e-4);
        assert_eq!(f.inspection_events[0].event_type, "entered");
        assert_eq!(
            f.localization_state.as_ref().unwrap().active_localizations,
            2
        );
    }

    #[test]
    fn node_sensor_fields_deserialize() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0, "tracks": [], "truths": [],
                "nodes": [{"node_id": "g1", "position": [0.0,0.0,3.0], "velocity": [0.0,0.0,0.0],
                            "is_mobile": false, "sensor_type": "radar", "fov_half_angle_deg": 180.0,
                            "max_range_m": 900.0}]
            }]
        }))
        .unwrap();
        let node = &document.frames[0].nodes[0];
        assert_eq!(node.sensor_type, "radar");
        assert!((node.fov_half_angle_deg.unwrap() - 180.0).abs() < 1e-3);
        assert!((node.max_range_m.unwrap() - 900.0).abs() < 1e-3);
    }

    #[test]
    fn replay_state_builds_runtime_markers() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [{"track_id": "t1", "position": [1.0, 2.0, 3.0], "velocity": [0.0, 0.0, 0.0]}],
                "truths": [{"target_id": "truth-1", "position": [4.0, 5.0, 6.0], "velocity": [1.0, 0.0, 0.0]}],
                "nodes": [{"node_id": "node-1", "position": [7.0, 8.0, 9.0], "velocity": [0.0, 1.0, 0.0], "is_mobile": true}]
            }]
        }))
        .unwrap();
        let replay = ReplayState::new(Some(document));
        let markers = replay.current_markers();
        assert_eq!(markers.len(), 3);
        assert_eq!(markers[0].kind, MarkerKind::Track);
        assert_eq!(markers[1].kind, MarkerKind::Truth);
        assert_eq!(markers[2].kind, MarkerKind::Node);
    }

    #[test]
    fn terrain_viewer_mesh_deserializes_and_samples_height() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {
                "terrain": {
                    "viewer_mesh": {
                        "x_min_m": -10.0,
                        "x_max_m": 10.0,
                        "y_min_m": -10.0,
                        "y_max_m": 10.0,
                        "rows": 2,
                        "cols": 2,
                        "heights_m": [
                            [10.0, 14.0],
                            [18.0, 22.0]
                        ]
                    }
                }
            },
            "frames": [{
                "timestamp_s": 0.0,
                "tracks": [],
                "truths": [],
                "nodes": []
            }]
        }))
        .unwrap();

        let mesh = document.terrain_viewer_mesh().expect("terrain mesh");
        let sampled = mesh.sample_height(0.0, 0.0).expect("sampled height");
        assert!((sampled - 16.0).abs() < 1.0e-6);
        assert_eq!(mesh.sample_height(20.1, 0.0), None);
    }
}
