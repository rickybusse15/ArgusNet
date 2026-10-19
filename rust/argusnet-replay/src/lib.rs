//! Replay-document DTOs: the shape of the JSON contract between the Python
//! simulation and everything that reads a replay.
//!
//! CLAUDE.md calls replay JSON "the interface between simulation and viewer",
//! and until P4.0 these types lived inside `argusnet-viewer`, which meant that
//! reading a replay meant linking a renderer. They are pure deserialization
//! with no rendering concern, so P4.0 moves them here, per "keep runtime types
//! free of Bevy/prost".
//!
//! **This crate must stay free of Bevy.** It turned out to be free of it
//! already: the DTOs carry positions as `[f32; 3]`, and the only `Vec3` in the
//! original module belonged to `RuntimeMarker` and `ReplayState`, which are
//! playback and presentation rather than document shape and so stayed in the
//! viewer. That also makes the `glam`/`bevy_math` version coupling this
//! extraction was expected to need unnecessary -- there is no shared vector
//! type across the boundary to keep in step.
//!
//! `prost` enters only under the optional `live-stream` feature, which carries
//! the conversions from `argusnet-proto`'s wire messages into these DTOs so the
//! viewer can accept live frames without a JSON round trip.

use std::collections::HashMap;

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

/// Accept `null` where the replay schema permits it but the viewer wants a value.
///
/// Python dataclasses serialize an unset optional as JSON `null`, and
/// `docs/replay-schema.json` explicitly allows it (for example
/// `SafetyValidationResult.reason` is typed `["string", "null"]`).
/// `#[serde(default)]` alone does **not** cover this: it handles a *missing*
/// key, not a present-but-null one, so a single null field aborted parsing of
/// the entire replay.
fn null_as_default<'de, D, T>(deserializer: D) -> Result<T, D::Error>
where
    D: Deserializer<'de>,
    T: Deserialize<'de> + Default,
{
    Ok(Option::<T>::deserialize(deserializer)?.unwrap_or_default())
}

/// Read an optional range where `0.0` carries the proto's "unset" meaning.
///
/// `world_model.proto` documents `double max_range_m = 9;  // 0 = unset`, and the
/// live path honours it (`frame_from_pb` maps `0.0` to `None`). The replay path
/// deserialized it straight into `Some(0.0)`, which the viewer would draw as a
/// zero-radius radar ring and a zero-length FOV cone instead of falling back to
/// its default range.
fn optional_positive_range<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: Deserializer<'de>,
{
    let value = Option::<f32>::deserialize(deserializer)?;
    Ok(value.filter(|range| *range > 0.0))
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

pub use argusnet_world::raster::TerrainRaster as TerrainViewerMesh;

/// Fallback stale-step threshold, used only when a replay carries no
/// `meta.platform.max_stale_steps`.
///
/// Mirrors `TrackerConfig::default().max_stale_steps` in `argusnet-core`. The
/// viewer cannot read that constant directly because `argusnet-core` is only in
/// the dependency graph under the `live-stream` feature; the duplication goes
/// away when the shared value types move into their own crate.
pub const DEFAULT_MAX_STALE_STEPS: u32 = 8;

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

    /// The run's power lines (`meta.power_lines`, R3), each checked by the
    /// model as it is read. Empty when the replay carries none; an error, not
    /// an empty list, when it carries one the model refuses, so a bad line is
    /// reported instead of silently not drawn.
    pub fn power_lines(&self) -> Result<Vec<argusnet_world::powerline::PowerLine>, String> {
        match self.meta.as_ref().and_then(|m| m.get("power_lines")) {
            None | Some(Value::Null) => Ok(Vec::new()),
            Some(lines) => serde_json::from_value(lines.clone())
                .map_err(|error| format!("meta.power_lines: {error}")),
        }
    }

    /// Stale-step threshold the tracker was actually configured with for this run.
    ///
    /// Replays record it under `meta.platform.max_stale_steps`, so the viewer reads
    /// the run's own value instead of mirroring a constant by hand. The previous
    /// hardcoded `5` had already drifted from `TrackerConfig`'s default of `8` and
    /// from the `6` that seeded replays carry, so tracks were flagged stale in the
    /// alert panel well before the tracker itself considered them stale.
    pub fn max_stale_steps(&self) -> Option<u32> {
        self.meta
            .as_ref()
            .and_then(|m| m.get("platform"))
            .and_then(|platform| platform.get("max_stale_steps"))
            .and_then(|value| value.as_u64())
            .map(|value| value as u32)
    }

    /// The self-built map's layout, in a run that started from no world
    /// knowledge; `None` in every other run.
    pub fn belief_grid(&self) -> Option<BeliefGridLayout> {
        self.meta
            .as_ref()
            .and_then(|m| m.get("world_knowledge"))
            .filter(|wk| wk.get("knowledge").and_then(Value::as_str) == Some("none"))
            .and_then(|wk| wk.get("belief_grid"))
            .and_then(|grid| serde_json::from_value(grid.clone()).ok())
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
    pub target_metadata: Vec<TargetMetadata>,
    #[serde(default)]
    pub safety_events: Vec<SafetyEvent>,
    /// A run that maps the world for itself (`world_knowledge: none`): the
    /// belief cells this frame changed. Applying the frames' rows in order
    /// rebuilds the map; see [`ReplayDocument::belief_grid`] for its layout.
    #[serde(default)]
    pub belief_updates: Vec<BeliefCellRow>,
    /// A sample of each mobile platform's range returns this frame.
    #[serde(default)]
    pub range_scans: Vec<RangeScanSample>,
    /// Platforms physics put back out of the true ground or a structure.
    #[serde(default)]
    pub world_contacts: Vec<WorldContact>,
}

/// `[i, j, ground_cm, structure_top_cm, occupancy_pct, confidence]`, as the
/// schema's `belief_updates` row.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct BeliefCellRow(
    pub u32,
    pub u32,
    pub Option<i32>,
    pub Option<i32>,
    pub u8,
    pub u8,
);

impl BeliefCellRow {
    pub fn ij(&self) -> (u32, u32) {
        (self.0, self.1)
    }
    /// Believed ground height, metres; `None` until a ground return lands.
    pub fn ground_m(&self) -> Option<f32> {
        self.2.map(|cm| cm as f32 / 100.0)
    }
    /// Highest structure return, metres, when the cell is believed occupied.
    pub fn structure_top_m(&self) -> Option<f32> {
        self.3.map(|cm| cm as f32 / 100.0)
    }
    pub fn occupancy(&self) -> f32 {
        self.4 as f32 / 100.0
    }
    /// 0 (no returns) to 4 (eight or more).
    pub fn confidence_level(&self) -> u8 {
        self.5
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct RangeScanSample {
    pub node_id: String,
    pub origin_m: [f32; 3],
    #[serde(default)]
    pub ground: Vec<[f32; 3]>,
    #[serde(default)]
    pub structure: Vec<[f32; 3]>,
    #[serde(default)]
    pub misses: u32,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct WorldContact {
    pub node_id: String,
    pub timestamp_s: f32,
    pub kind: String,
    pub correction_m: f32,
}

/// Where a run's belief cells are: `meta.world_knowledge.belief_grid`.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
pub struct BeliefGridLayout {
    pub origin_m: [f32; 2],
    pub resolution_m: f32,
    pub dimensions: [u32; 2],
}

impl BeliefGridLayout {
    /// The centre of cell `(i, j)`, metres.
    pub fn centre(&self, i: u32, j: u32) -> [f32; 2] {
        [
            self.origin_m[0] + (i as f32 + 0.5) * self.resolution_m,
            self.origin_m[1] + (j as f32 + 0.5) * self.resolution_m,
        ]
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct BeliefStats {
    #[serde(default)]
    pub known_cells: u64,
    #[serde(default)]
    pub total_cells: u64,
    #[serde(default)]
    pub known_fraction: f32,
    #[serde(default)]
    pub ground_rmse_m: Option<f32>,
    #[serde(default)]
    pub ground_max_abs_error_m: Option<f32>,
    #[serde(default)]
    pub structure_cells: u64,
    #[serde(default)]
    pub structure_precision: Option<f32>,
    #[serde(default)]
    pub structure_recall: Option<f32>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TargetMetadata {
    pub target_id: String,
    pub operator_label: String,
    pub target_type: String,
    pub priority: i32,
    pub behavior: String,
    pub status: String,
    pub first_seen_s: f32,
    pub last_seen_s: f32,
    pub confidence: f32,
    pub notes: String,
}

/// Replay JSON serializes safety events via the Python `SafetyValidationResult`
/// schema (`accepted`, `clamped`), while the live gRPC wire uses `SafetyEvent`
/// (`blocked`, `command_clamped`). `accepted` is the exact logical inverse of
/// `blocked`, not a rename, so it cannot be handled with a plain `#[serde(alias)]`.
#[derive(Debug, Clone, Deserialize, Default)]
struct SafetyEventRaw {
    #[serde(alias = "validation_id")]
    event_id: String,
    #[serde(alias = "subject_id")]
    subject_id: String,
    timestamp_s: f32,
    #[serde(default, deserialize_with = "null_as_default")]
    state: String,
    #[serde(default, deserialize_with = "null_as_default")]
    violations: Vec<String>,
    #[serde(default, alias = "clamped")]
    command_clamped: bool,
    #[serde(default)]
    blocked: bool,
    #[serde(default)]
    accepted: Option<bool>,
    #[serde(default, deserialize_with = "null_as_default")]
    reason: String,
}

impl From<SafetyEventRaw> for SafetyEvent {
    fn from(raw: SafetyEventRaw) -> Self {
        let blocked = raw.accepted.map_or(raw.blocked, |accepted| !accepted);
        SafetyEvent {
            event_id: raw.event_id,
            subject_id: raw.subject_id,
            timestamp_s: raw.timestamp_s,
            state: raw.state,
            violations: raw.violations,
            command_clamped: raw.command_clamped,
            blocked,
            reason: raw.reason,
        }
    }
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(from = "SafetyEventRaw")]
pub struct SafetyEvent {
    pub event_id: String,
    pub subject_id: String,
    pub timestamp_s: f32,
    pub state: String,
    pub violations: Vec<String>,
    pub command_clamped: bool,
    pub blocked: bool,
    pub reason: String,
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
    /// Fused team position from cooperative co-localization. None outside
    /// cooperative search or before enough drones have converged.
    #[serde(default)]
    pub team_localization: Option<TeamLocalization>,
}

#[derive(Debug, Clone, Deserialize, Default)]
pub struct TeamLocalization {
    pub position: [f32; 3],
    pub position_std_m: f32,
    pub confidence: f32,
    #[serde(default)]
    pub contributing_node_ids: Vec<String>,
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
    /// When the track last took a measurement (P4.10b); only timed runs
    /// (delivery modelled, a fusion rate or a sensor rate) write it.
    #[serde(default)]
    pub evidence_time_s: Option<f64>,
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
    #[serde(default, deserialize_with = "optional_positive_range")]
    pub max_range_m: Option<f32>,
    /// Remaining battery, 0.0-1.0. Written by the simulation on every node;
    /// the viewer previously had no field for it, so the drone battery readout
    /// promised by `docs/VIEWER_UI.md` could never populate.
    #[serde(default)]
    pub battery_fraction: Option<f32>,
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
    /// The self-built map's progress, in a run without a prior map.
    #[serde(default)]
    pub belief: Option<BeliefStats>,
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

    use super::{
        FrameMetrics, NodeState, ObservationState, RejectedObservation, ReplayFrame, SafetyEvent,
        TargetMetadata, TrackState, TruthState,
    };

    fn vec3_opt(value: Option<pb::Vector3>) -> [f32; 3] {
        let v = value.unwrap_or_default();
        [v.x_m as f32, v.y_m as f32, v.z_m as f32]
    }

    impl From<pb::BearingObservation> for ObservationState {
        fn from(o: pb::BearingObservation) -> Self {
            ObservationState {
                origin: vec3_opt(o.origin),
                direction: vec3_opt(o.direction),
                node_id: o.node_id,
                target_id: o.target_id,
                confidence: o.confidence as f32,
                timestamp_s: o.timestamp_s as f32,
            }
        }
    }

    impl From<pb::ObservationRejection> for RejectedObservation {
        fn from(r: pb::ObservationRejection) -> Self {
            RejectedObservation {
                origin: r.origin.map(|v| vec3_opt(Some(v))),
                attempted_point: r.attempted_point.map(|v| vec3_opt(Some(v))),
                closest_point: r.closest_point.map(|v| vec3_opt(Some(v))),
                node_id: r.node_id,
                target_id: r.target_id,
                reason: r.reason,
                detail: r.detail,
                timestamp_s: r.timestamp_s as f32,
                blocker_type: r.blocker_type,
                first_hit_range_m: r.first_hit_range_m.map(|v| v as f32),
            }
        }
    }

    impl From<pb::PlatformMetrics> for FrameMetrics {
        fn from(m: pb::PlatformMetrics) -> Self {
            FrameMetrics {
                mean_error_m: m.mean_error_m.map(|v| v as f32),
                max_error_m: m.max_error_m.map(|v| v as f32),
                active_track_count: m.active_track_count,
                observation_count: m.observation_count,
                accepted_observation_count: m.accepted_observation_count,
                rejected_observation_count: m.rejected_observation_count,
                mean_measurement_std_m: m.mean_measurement_std_m.map(|v| v as f32),
                rejection_counts: m.rejection_counts.into_iter().collect(),
                track_errors_m: m
                    .track_errors_m
                    .into_iter()
                    .map(|(key, value)| (key, value as f32))
                    .collect(),
            }
        }
    }

    /// Convert a streamed frame into the viewer's replay representation.
    ///
    /// Fields that only exist in replay JSON (mapping/localization/inspection
    /// state, events) are absent on the tracker wire and default to empty.
    pub fn frame_from_pb(frame: pb::PlatformFrame) -> ReplayFrame {
        ReplayFrame {
            timestamp_s: frame.timestamp_s as f32,
            tracks: frame.tracks.into_iter().map(TrackState::from).collect(),
            truths: frame.truths.into_iter().map(TruthState::from).collect(),
            nodes: frame.nodes.into_iter().map(NodeState::from).collect(),
            observations: frame
                .observations
                .into_iter()
                .map(ObservationState::from)
                .collect(),
            rejected_observations: frame
                .rejected_observations
                .into_iter()
                .map(RejectedObservation::from)
                .collect(),
            generation_rejections: frame
                .generation_rejections
                .into_iter()
                .map(RejectedObservation::from)
                .collect(),
            metrics: frame.metrics.map(FrameMetrics::from).unwrap_or_default(),
            mapping_state: None,
            localization_state: None,
            inspection_events: Vec::new(),
            deconfliction_events: Vec::new(),
            scan_mission_state: None,
            target_metadata: frame
                .target_metadata
                .into_iter()
                .map(|value| TargetMetadata {
                    target_id: value.target_id,
                    operator_label: value.operator_label,
                    target_type: value.target_type,
                    priority: value.priority,
                    behavior: value.behavior,
                    status: value.status,
                    first_seen_s: value.first_seen_s as f32,
                    last_seen_s: value.last_seen_s as f32,
                    confidence: value.confidence as f32,
                    notes: value.notes,
                })
                .collect(),
            safety_events: frame
                .safety_events
                .into_iter()
                .map(|value| SafetyEvent {
                    event_id: value.event_id,
                    subject_id: value.subject_id,
                    timestamp_s: value.timestamp_s as f32,
                    state: value.state,
                    violations: value.violations,
                    command_clamped: value.command_clamped,
                    blocked: value.blocked,
                    reason: value.reason,
                })
                .collect(),
            // Replay-only: the tracker wire carries no belief.
            belief_updates: Vec::new(),
            range_scans: Vec::new(),
            world_contacts: Vec::new(),
        }
    }

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
                measurement_std_m: t.measurement_std_m as f32,
                update_count: t.update_count,
                stale_steps: t.stale_steps,
                covariance,
                mode_probability_cv,
                contributing_node_ids: t.contributing_nodes,
                // The wire does not carry it.
                evidence_time_s: None,
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
                health: n.health as f32,
                sensor_type: n.sensor_type,
                fov_half_angle_deg: if n.fov_half_angle_deg == 0.0 {
                    None
                } else {
                    Some(n.fov_half_angle_deg as f32)
                },
                max_range_m: if n.max_range_m == 0.0 {
                    None
                } else {
                    Some(n.max_range_m as f32)
                },
                // `pb::NodeState` has no battery field, so live frames cannot
                // carry it. Adding one to `world_model.proto` is an additive
                // change; until then the battery readout is replay-only.
                battery_fraction: None,
            }
        }
    }
}

#[cfg(feature = "live-stream")]
pub use live_stream::frame_from_pb;
#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::ReplayDocument;

    /// `meta.power_lines` is read through the model's own check: absent is no
    /// lines, a valid line is read, and one the model refuses is an error
    /// naming why, never an empty list.
    #[test]
    fn power_lines_are_read_checked() {
        let replay = |lines: serde_json::Value| -> ReplayDocument {
            serde_json::from_value(json!({"meta": {"power_lines": lines}, "frames": []})).unwrap()
        };
        let none: ReplayDocument =
            serde_json::from_value(json!({"meta": {}, "frames": []})).unwrap();
        assert_eq!(none.power_lines(), Ok(Vec::new()));
        let line = json!({
            "id": "line-a", "conductor_offsets_m": [0.0], "sag_ratio": 0.03,
            "conductor_radius_m": 0.01, "clearance_m": 5.0,
            "supports": [
                {"xy_m": [0.0, 0.0], "base_z_m": 0.0, "height_m": 20.0},
                {"xy_m": [100.0, 0.0], "base_z_m": 0.0, "height_m": 20.0}
            ]
        });
        let read = replay(json!([line.clone()])).power_lines().unwrap();
        assert_eq!(read.len(), 1);
        assert_eq!(read[0].supports[1].xy_m, [100.0, 0.0]);
        let mut bare = line;
        bare["conductor_offsets_m"] = json!([]);
        let refused = replay(json!([bare])).power_lines().unwrap_err();
        assert!(refused.contains("line-a"), "{refused}");
    }

    /// Regression: `SafetyValidationResult.reason` is `str | None` in Python and
    /// serializes to JSON `null`, which `docs/replay-schema.json` explicitly
    /// permits. `#[serde(default)]` covers a *missing* key, not a null one, so a
    /// single null field aborted deserialization of the entire replay.
    #[test]
    fn a_null_safety_reason_does_not_abort_the_whole_replay() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25},
            "frames": [{
                "timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": [],
                "safety_events": [{
                    "validation_id": "s-1", "subject_id": "drone-0", "timestamp_s": 0.0,
                    "accepted": true, "violations": null, "clamped": false,
                    "reason": null, "state": null
                }]
            }]
        }))
        .expect("a null reason must not fail the replay");
        let event = &document.frames[0].safety_events[0];
        assert_eq!(event.reason, "");
        assert!(event.violations.is_empty());
    }

    /// Regression: `world_model.proto` documents `max_range_m` as "0 = unset" and
    /// the live path honours it, but the replay path produced `Some(0.0)` -- drawn
    /// as a zero-radius radar ring and a zero-length FOV cone.
    #[test]
    fn zero_max_range_means_unset_on_the_replay_path_too() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25},
            "frames": [{
                "timestamp_s": 0.0, "tracks": [], "truths": [],
                "nodes": [
                    {"node_id": "unset", "position": [0.0, 0.0, 0.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": true, "max_range_m": 0.0},
                    {"node_id": "set", "position": [0.0, 0.0, 0.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": true, "max_range_m": 650.0}
                ]
            }]
        }))
        .unwrap();
        assert_eq!(document.frames[0].nodes[0].max_range_m, None);
        assert_eq!(document.frames[0].nodes[1].max_range_m, Some(650.0));
    }

    /// Regression: the simulation writes `battery_fraction` on every node, but the
    /// viewer had no field for it, so the drone battery readout documented in
    /// `docs/VIEWER_UI.md` could never populate.
    #[test]
    fn battery_fraction_survives_the_replay_boundary() {
        let document: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25},
            "frames": [{
                "timestamp_s": 0.0, "tracks": [], "truths": [],
                "nodes": [
                    {"node_id": "drone-0", "position": [0.0, 0.0, 0.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": true, "battery_fraction": 0.62},
                    {"node_id": "ground-0", "position": [0.0, 0.0, 0.0],
                     "velocity": [0.0, 0.0, 0.0], "is_mobile": false}
                ]
            }]
        }))
        .unwrap();
        assert_eq!(document.frames[0].nodes[0].battery_fraction, Some(0.62));
        assert_eq!(document.frames[0].nodes[1].battery_fraction, None);
    }

    /// `ReplayState::max_stale_steps` in the viewer wraps this with the
    /// `DEFAULT_MAX_STALE_STEPS` fallback; the document's own accessor reports
    /// only what the run recorded, and `None` when it recorded nothing.
    #[test]
    fn max_stale_steps_reports_only_what_the_replay_recorded() {
        let with_platform: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25, "platform": {"min_observations": 2, "max_stale_steps": 6}},
            "frames": [{"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []}]
        }))
        .unwrap();
        assert_eq!(with_platform.max_stale_steps(), Some(6));

        let without_platform: ReplayDocument = serde_json::from_value(json!({
            "meta": {"dt_s": 0.25},
            "frames": [{"timestamp_s": 0.0, "tracks": [], "truths": [], "nodes": []}]
        }))
        .unwrap();
        assert_eq!(without_platform.max_stale_steps(), None);
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
        let sampled = mesh.height_at(0.0, 0.0).expect("sampled height");
        assert!((sampled - 16.0).abs() < 1.0e-6);
        assert_eq!(mesh.height_at(20.1, 0.0), None);
    }
}

/// Reader half of the cross-language replay contract test.
///
/// Deserializes the same fixture `tests/test_replay_contract.py` validates on the
/// writer side, so a change to either the Python writer or this reader that
/// breaks the other fails a test. See `tests/fixtures/README-replay-contract.md`.
#[cfg(test)]
mod contract_fixture_tests {
    use super::*;

    fn fixture() -> ReplayDocument {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../tests/fixtures/replay_contract_fixture.json"
        );
        let raw = std::fs::read_to_string(path)
            .unwrap_or_else(|error| panic!("failed to read {path}: {error}"));
        serde_json::from_str(&raw)
            .unwrap_or_else(|error| panic!("viewer cannot deserialize the replay fixture: {error}"))
    }

    #[test]
    fn the_viewer_can_read_what_the_simulation_writes() {
        let document = fixture();
        assert!(!document.frames.is_empty(), "fixture must carry frames");
        assert!(document.meta.is_some(), "fixture must carry meta");
    }

    #[test]
    fn nodes_populate_the_fields_the_operator_panels_display() {
        let document = fixture();
        let nodes: Vec<&NodeState> = document
            .frames
            .iter()
            .flat_map(|f| f.nodes.iter())
            .collect();
        assert!(!nodes.is_empty(), "fixture must carry nodes");

        for node in &nodes {
            assert!(!node.node_id.is_empty());
            assert!(!node.sensor_type.is_empty());
            assert!(node.health.is_finite());
        }

        // Battery is the field the viewer had no home for, so the readout
        // documented in docs/VIEWER_UI.md could never populate.
        let mobile: Vec<&&NodeState> = nodes.iter().filter(|n| n.is_mobile).collect();
        assert!(!mobile.is_empty(), "fixture must carry mobile nodes");
        for node in mobile {
            let fraction = node
                .battery_fraction
                .unwrap_or_else(|| panic!("node {} lost its battery_fraction", node.node_id));
            assert!(
                (0.0..=1.0).contains(&fraction),
                "implausible battery {fraction}"
            );
        }

        // `0 = unset` must never survive as `Some(0.0)`.
        for node in &nodes {
            assert_ne!(node.max_range_m, Some(0.0));
        }
    }

    #[test]
    fn tracks_populate_the_fields_the_track_tables_display() {
        let document = fixture();
        let tracks: Vec<&TrackState> = document
            .frames
            .iter()
            .flat_map(|f| f.tracks.iter())
            .collect();
        assert!(!tracks.is_empty(), "fixture must exercise tracks");
        for track in &tracks {
            assert!(!track.track_id.is_empty());
            assert!(track.measurement_std_m.is_finite());
        }

        // The IMM readout was blank in replay mode because `_track_from_proto`
        // dropped this on the way out of the gRPC response, even though the
        // viewer declares the field and the proto carries it.
        assert!(
            tracks.iter().any(|t| t.mode_probability_cv.is_some()),
            "no track carries mode_probability_cv; the viewer's IMM readout is dead again"
        );
        for track in &tracks {
            if let Some(weight) = track.mode_probability_cv {
                assert!(
                    (0.0..=1.0).contains(&weight),
                    "implausible IMM weight {weight}"
                );
            }
        }
    }

    /// `covariance` is written flat (36 numbers) by `to_jsonable`, which flattens
    /// any `ndim > 1` array. The reader accepts flat or nested; this pins that the
    /// writer's actual shape stays readable.
    #[test]
    fn track_covariance_arrives_in_a_shape_the_viewer_accepts() {
        let document = fixture();
        let covariances: Vec<&Vec<f64>> = document
            .frames
            .iter()
            .flat_map(|f| f.tracks.iter())
            .filter_map(|t| t.covariance.as_ref())
            .collect();
        assert!(!covariances.is_empty(), "fixture must carry covariances");
        for covariance in covariances {
            assert_eq!(covariance.len(), 36, "expected a flattened 6x6 covariance");
            assert!(covariance.iter().all(|value| value.is_finite()));
        }
    }
}
