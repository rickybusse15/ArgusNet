//! The versioned replay writer.
//!
//! [`DELIVERY_PLAN.md`](../../../docs/migration/DELIVERY_PLAN.md) §P4.8: *"Add
//! complete checkpoints and a versioned Rust replay writer. Python evaluation
//! reads its output through strict schema validation."* This is that writer.
//!
//! # A replay is not a snapshot and not a checkpoint
//!
//! Three types describe the run and they are deliberately distinct — see
//! [`crate::snapshot`] for the first two:
//!
//! | type | answers | persisted |
//! |---|---|---|
//! | [`RuntimeSnapshot`](crate::snapshot::RuntimeSnapshot) | "what should the operator see right now?" | no, in-process |
//! | [`SimulationCheckpoint`](crate::snapshot::SimulationCheckpoint) | "what does resuming this run need?" | yes, private |
//! | [`ReplayDocument`] | "what happened, for someone else to read?" | yes, public |
//!
//! A replay is lossy by construction and safe to hand out. A checkpoint is
//! causally complete and is not, because it necessarily contains the truth
//! state a published replay may have filtered.
//!
//! # The schema is the contract, and it is enforced elsewhere
//!
//! `docs/replay-schema.json` is the interface between simulation and viewer —
//! CLAUDE.md's "replay-as-contract" rule. Rust does not validate against it at
//! runtime: pulling a JSON-Schema validator into the simulation crate to check
//! a document this module just built would test serde against itself. The
//! enforcement that matters is cross-language, and it lives in
//! `tests/test_rust_replay_contract.py`, which loads this writer's committed
//! fixture and validates it with `jsonschema` through the same
//! `validate_replay_with_schema` the Python writer answers to.
//!
//! That fixture is regenerated and byte-compared by
//! [`tests::the_committed_fixture_is_what_this_writer_emits`], the same
//! arrangement `scripts/regenerate_proto.py --check` uses for the protobuf
//! bindings: drift fails in Rust CI, and the contract is checked in Python CI,
//! neither able to silently rebaseline the other.
//!
//! # Numbers are `f64` all the way out
//!
//! The replay DTOs in `argusnet-replay` narrow to `f32` for the renderer's
//! benefit. This writer does not: it is the *producer*, and narrowing here
//! would bake a presentation decision into the archive. `serde_json` emits the
//! shortest representation that round-trips, so a value that came from `f64`
//! arithmetic is recoverable from the text.

use std::collections::BTreeMap;

use argusnet_types::{
    BearingObservation, NodeState, ObservationRejection, PlatformMetrics, TrackState, TruthState,
};
use serde::{Deserialize, Serialize};

/// The version this writer emits, matching `sim.py`'s `"2.0"`.
///
/// One string, two producers: a Rust-written replay and a Python-written one
/// describe the same schema, so they must claim the same version. Bump this
/// **and** `sim.py` together, or a consumer that branches on the version will
/// take different paths for documents that are in fact identical in shape.
pub const SCHEMA_VERSION: &str = "2.0";

/// Serialize a `Vector3` as the schema's `[x, y, z]`, not as `{x, y, z}`.
///
/// `nalgebra`'s own `Serialize` emits a struct-like form. The schema's
/// `vector3` is an array with `minItems`/`maxItems` of 3, so the default would
/// produce a document that fails validation — silently, since nothing in Rust
/// checks the schema.
fn vector3(value: argusnet_types::Vector3<f64>) -> [f64; 3] {
    [value.x, value.y, value.z]
}

/// One node's state as the replay records it.
///
/// Field-for-field with `docs/replay-schema.json`'s `node_state`, including
/// the four added by P4.8d-i. `battery_fraction` carries the `-1.0`
/// not-applicable sentinel rather than being omitted, because that is the
/// representation the schema documents and the Python writer already emits;
/// the `Option` lives in the runtime type and the wire, not here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayNode {
    pub node_id: String,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub is_mobile: bool,
    pub timestamp_s: f64,
    pub health: f64,
    pub sensor_type: String,
    pub fov_half_angle_deg: f64,
    pub max_range_m: f64,
    pub battery_fraction: f64,
}

/// `battery_fraction` for a platform with no battery.
///
/// Mirrors [`crate::battery::NOT_APPLICABLE_FRACTION`] and Python's
/// `BATTERY_NOT_APPLICABLE`. The schema admits exactly this value or a reading
/// in `[0.0, 1.0]`, with the gap between them excluded.
pub const BATTERY_NOT_APPLICABLE: f64 = -1.0;

/// The schema's documented "unset" for the two sensor fields that have no
/// presence on the wire. Distinct from the battery sentinel above, and
/// deliberately so: a zero-radius sensor is not a reading anyone means, but a
/// flat battery is.
const UNSET_SENSOR_FIELD: f64 = 0.0;

impl From<&NodeState> for ReplayNode {
    fn from(node: &NodeState) -> Self {
        Self {
            node_id: node.node_id.clone(),
            position: vector3(node.position),
            velocity: vector3(node.velocity),
            is_mobile: node.is_mobile,
            timestamp_s: node.timestamp_s,
            health: node.health,
            sensor_type: node.sensor_type.clone(),
            fov_half_angle_deg: node.fov_half_angle_deg.unwrap_or(UNSET_SENSOR_FIELD),
            max_range_m: node.max_range_m.unwrap_or(UNSET_SENSOR_FIELD),
            battery_fraction: node.battery_fraction.unwrap_or(BATTERY_NOT_APPLICABLE),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTruth {
    pub target_id: String,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    pub timestamp_s: f64,
}

impl From<&TruthState> for ReplayTruth {
    fn from(truth: &TruthState) -> Self {
        Self {
            target_id: truth.target_id.clone(),
            position: vector3(truth.position),
            velocity: vector3(truth.velocity),
            timestamp_s: truth.timestamp_s,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayObservation {
    pub node_id: String,
    pub target_id: String,
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub bearing_std_rad: f64,
    pub timestamp_s: f64,
    pub confidence: f64,
}

impl From<&BearingObservation> for ReplayObservation {
    fn from(observation: &BearingObservation) -> Self {
        Self {
            node_id: observation.node_id.clone(),
            target_id: observation.target_id.clone(),
            origin: vector3(observation.origin),
            direction: vector3(observation.direction),
            bearing_std_rad: observation.bearing_std_rad,
            timestamp_s: observation.timestamp_s,
            confidence: observation.confidence,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRejection {
    pub node_id: String,
    pub target_id: String,
    pub timestamp_s: f64,
    pub reason: String,
    pub detail: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub origin: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attempted_point: Option<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub closest_point: Option<[f64; 3]>,
    pub blocker_type: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub first_hit_range_m: Option<f64>,
}

impl From<&ObservationRejection> for ReplayRejection {
    fn from(rejection: &ObservationRejection) -> Self {
        Self {
            node_id: rejection.node_id.clone(),
            target_id: rejection.target_id.clone(),
            timestamp_s: rejection.timestamp_s,
            reason: rejection.reason.clone(),
            detail: rejection.detail.clone(),
            origin: rejection.origin.map(vector3),
            attempted_point: rejection.attempted_point.map(vector3),
            closest_point: rejection.closest_point.map(vector3),
            blocker_type: rejection.blocker_type.clone(),
            first_hit_range_m: rejection.first_hit_range_m,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTrack {
    pub track_id: String,
    pub timestamp_s: f64,
    pub position: [f64; 3],
    pub velocity: [f64; 3],
    /// Row-major, 36 elements. The schema's `covariance_flat` does not pin the
    /// length, but `evaluation/replay.py` reshapes to `(6, 6)` unconditionally,
    /// so anything else crashes the Python reader.
    pub covariance: Vec<f64>,
    pub measurement_std_m: f64,
    pub update_count: u32,
    pub stale_steps: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_probability_cv: Option<f64>,
    pub contributing_node_ids: Vec<String>,
    /// `tentative`/`confirmed`/`coasting`, as the reference writer emits it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lifecycle_state: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub quality_score: Option<f64>,
    /// When the track last took a measurement (P4.10b). Written only by a
    /// run whose tracks can be older than their frame (delivery modelled, a
    /// fusion rate or a sensor rate), so a legacy replay is byte for byte what it was; see
    /// `AssembledRun::record_frame`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub evidence_time_s: Option<f64>,
}

impl From<&TrackState> for ReplayTrack {
    fn from(track: &TrackState) -> Self {
        Self {
            track_id: track.track_id.clone(),
            timestamp_s: track.timestamp_s,
            position: vector3(track.position),
            velocity: vector3(track.velocity),
            // `Matrix6` is column-major in memory; the schema says row-major.
            // Iterating rows explicitly rather than calling `as_slice()`, which
            // would transpose the matrix into the document without complaint.
            covariance: (0..6)
                .flat_map(|row| (0..6).map(move |column| (row, column)))
                .map(|(row, column)| track.covariance[(row, column)])
                .collect(),
            measurement_std_m: track.measurement_std_m,
            update_count: track.update_count,
            stale_steps: track.stale_steps,
            mode_probability_cv: track.mode_probability_cv,
            contributing_node_ids: track.contributing_node_ids.as_ref().clone(),
            lifecycle_state: track.lifecycle_state.clone(),
            quality_score: track.quality_score,
            evidence_time_s: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayMetrics {
    pub mean_error_m: Option<f64>,
    pub max_error_m: Option<f64>,
    pub active_track_count: u32,
    pub observation_count: u32,
    pub accepted_observation_count: u32,
    pub rejected_observation_count: u32,
    pub mean_measurement_std_m: Option<f64>,
    pub track_errors_m: BTreeMap<String, f64>,
    pub rejection_counts: BTreeMap<String, u32>,
    pub accepted_observations_by_target: BTreeMap<String, u32>,
    pub rejected_observations_by_target: BTreeMap<String, u32>,
}

impl From<&PlatformMetrics> for ReplayMetrics {
    fn from(metrics: &PlatformMetrics) -> Self {
        Self {
            mean_error_m: metrics.mean_error_m,
            max_error_m: metrics.max_error_m,
            active_track_count: metrics.active_track_count,
            observation_count: metrics.observation_count,
            accepted_observation_count: metrics.accepted_observation_count,
            rejected_observation_count: metrics.rejected_observation_count,
            mean_measurement_std_m: metrics.mean_measurement_std_m,
            track_errors_m: metrics.track_errors_m.clone(),
            rejection_counts: metrics.rejection_counts.clone(),
            accepted_observations_by_target: metrics.accepted_observations_by_target.clone(),
            rejected_observations_by_target: metrics.rejected_observations_by_target.clone(),
        }
    }
}

/// `mapping_state`: the coverage map's four aggregates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayMappingState {
    pub coverage_fraction: f64,
    pub covered_cells: u64,
    pub total_cells: u64,
    pub mean_revisits: f64,
    /// The self-built map's progress, in a run without a prior map.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub belief: Option<ReplayBeliefStats>,
}

/// How much of the world the platforms have mapped, and how well. The error
/// figures compare the belief with the true terrain. They are evaluation,
/// computed by the writer, never read by the run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayBeliefStats {
    pub known_cells: u64,
    pub total_cells: u64,
    pub known_fraction: f64,
    /// Over known cells: belief ground minus true ground at the cell centre.
    pub ground_rmse_m: Option<f64>,
    pub ground_max_abs_error_m: Option<f64>,
    /// Cells believed occupied by a structure.
    pub structure_cells: u64,
    /// Of the believed structure cells, the fraction with a true building or
    /// wall within one cell. `None` with none believed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_precision: Option<f64>,
    /// Of the cells a true building or wall covers (at its centre), the
    /// fraction with a believed structure within one cell. `None` in a world
    /// without any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub structure_recall: Option<f64>,
    /// Cells with a canopy return (R2). Written only when there are some.
    #[serde(default, skip_serializing_if = "is_zero")]
    pub canopy_cells: u64,
}

fn is_zero(value: &u64) -> bool {
    *value == 0
}

/// A sample of one platform's range returns this tick, rounded to the
/// decimetre.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayRangeScan {
    pub node_id: String,
    pub origin_m: [f64; 3],
    pub ground: Vec<[f64; 3]>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub structure: Vec<[f64; 3]>,
    /// Returns from tree crowns (R2), written only when there are some.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub canopy: Vec<[f64; 3]>,
    pub misses: u32,
    /// The nearest point of each power-line span the scan put points on
    /// (R3). Absent when none.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub wire: Vec<[f64; 3]>,
}

/// Most returns per scan a replay frame carries. The frame's returns are a
/// picture of the sensing; the map itself travels in `belief_updates`.
pub const REPLAY_RETURNS_PER_SCAN: usize = 32;

impl From<&crate::range_scan::RangeScan> for ReplayRangeScan {
    /// An evenly strided sample of at most [`REPLAY_RETURNS_PER_SCAN`]
    /// returns, in ray order.
    fn from(scan: &crate::range_scan::RangeScan) -> Self {
        let dm = |p: [f64; 3]| p.map(|v| (v * 10.0).round() / 10.0);
        let stride = scan.returns.len().div_ceil(REPLAY_RETURNS_PER_SCAN).max(1);
        let of = |class| {
            scan.returns
                .iter()
                .step_by(stride)
                .filter(|hit| hit.class == class)
                .map(|hit| dm(hit.point_m))
                .collect()
        };
        Self {
            node_id: scan.node_id.clone(),
            origin_m: dm(scan.origin_m),
            ground: of(crate::range_scan::ReturnClass::Ground),
            structure: of(crate::range_scan::ReturnClass::Structure),
            canopy: of(crate::range_scan::ReturnClass::Canopy),
            misses: scan.misses,
            wire: scan.wires.iter().map(|wire| dm(wire.point_m)).collect(),
        }
    }
}

/// `localization_state`: track quality, derived from the frame's tracks and
/// the tick's generated observations.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayLocalizationState {
    pub active_localizations: u32,
    pub mean_position_std_m: f64,
    pub mean_observation_confidence: f64,
    /// Always empty from both writers; kept so the object matches Python's.
    #[serde(default)]
    pub status_counts: BTreeMap<String, u32>,
    #[serde(default = "map_frame_id")]
    pub frame_id: String,
}

fn map_frame_id() -> String {
    "map".to_string()
}

/// One zone entry, exit, coverage update or violation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayInspectionEvent {
    pub zone_id: String,
    pub node_id: String,
    pub event_type: String,
    pub timestamp_s: f64,
    pub zone_coverage_fraction: f64,
}

/// One stage-3 resolution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayDeconflictionEvent {
    pub yielding_drone_id: String,
    pub conflicting_drone_id: String,
    pub predicted_separation_m: f64,
    pub resolution: String,
    pub timestamp_s: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayLocalizationEstimate {
    pub drone_id: String,
    pub timestamp_s: f64,
    pub position_estimate: [f64; 3],
    pub heading_rad: f64,
    pub position_std_m: f64,
    pub confidence: f64,
}

/// A POI's status, with the nulls Python writes for fields not yet reached.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayPoiStatus {
    pub poi_id: String,
    pub status: String,
    pub assigned_drone_id: Option<String>,
    pub arrival_time_s: Option<f64>,
    pub completion_time_s: Option<f64>,
    pub dwell_accumulated_s: f64,
    pub position: Option<[f64; 3]>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayEgressProgress {
    pub drone_id: String,
    pub distance_to_home_m: f64,
    pub home_position: [f64; 3],
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTeamLocalization {
    pub position: [f64; 3],
    pub position_std_m: f64,
    pub confidence: f64,
    pub contributing_node_ids: Vec<String>,
}

/// `scan_mission_state`, present on every frame of a `scan_map_inspect` run.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayScanMissionState {
    pub phase: String,
    pub scan_coverage_fraction: f64,
    pub scan_coverage_threshold: f64,
    pub localization_estimates: Vec<ReplayLocalizationEstimate>,
    pub poi_statuses: Vec<ReplayPoiStatus>,
    pub completed_poi_count: usize,
    pub total_poi_count: usize,
    pub phase_started_at_s: f64,
    /// `(x, y, terrain height)` triples, flattened as Python's writer
    /// flattens its tuple of tuples.
    pub newly_scanned_cells: Vec<f64>,
    pub localization_timed_out: bool,
    pub coordinator_drone_id: Option<String>,
    pub egress_progress: Vec<ReplayEgressProgress>,
    pub team_localization: Option<ReplayTeamLocalization>,
}

/// Operator-facing identity and status for one target.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTargetMetadata {
    pub target_id: String,
    pub operator_label: String,
    pub target_type: String,
    pub priority: u32,
    pub behavior: String,
    pub status: String,
    pub first_seen_s: f64,
    pub last_seen_s: f64,
    pub confidence: f64,
    pub notes: String,
}

/// One frame: the seven keys the schema marks required, and the optional
/// payloads this writer produces.
///
/// An optional payload is omitted rather than emitted empty. An absent key
/// says "not produced here" -- `belief_cells`, `pose_estimates` and
/// `mission_events` are never produced, and a run with no mission machine has
/// no `scan_mission_state` -- and a consumer reads absence and emptiness the
/// same way. The skip keeps documents from producers that fill none of them,
/// such as the committed fixture, unchanged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayFrame {
    pub timestamp_s: f64,
    pub nodes: Vec<ReplayNode>,
    pub observations: Vec<ReplayObservation>,
    pub rejected_observations: Vec<ReplayRejection>,
    pub tracks: Vec<ReplayTrack>,
    pub truths: Vec<ReplayTruth>,
    pub metrics: ReplayMetrics,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub generation_rejections: Vec<ReplayRejection>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mapping_state: Option<ReplayMappingState>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub localization_state: Option<ReplayLocalizationState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub inspection_events: Vec<ReplayInspectionEvent>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub deconfliction_events: Vec<ReplayDeconflictionEvent>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scan_mission_state: Option<ReplayScanMissionState>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub target_metadata: Vec<ReplayTargetMetadata>,
    /// Stage 4's validations, as the fused frame echoes them (P4.9d).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub safety_events: Vec<ReplaySafetyEvent>,
    /// The belief cells this tick changed, in a run that maps the world for
    /// itself (`world_knowledge: none`). Each row replaces the cell's last,
    /// so applying the frames in order rebuilds the map.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub belief_updates: Vec<crate::belief_world::BeliefCellUpdate>,
    /// The belief's canopy cells this tick changed (R2): soft structure
    /// seen from tree crowns. Each row replaces the cell's last.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub canopy_updates: Vec<crate::belief_world::CanopyCellUpdate>,
    /// This tick's range returns, per mobile platform.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub range_scans: Vec<ReplayRangeScan>,
    /// Platforms physics put back out of the ground or a structure this tick:
    /// the belief had them somewhere they could not be.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub world_contacts: Vec<crate::belief_world::WorldContact>,
    /// Platforms that touched a tree this tick (R2): each trunk or crown the
    /// last tick's flight crossed. Recorded, never pushed out of: a crown is
    /// soft, and a trunk the physics has already pushed clear of.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub tree_contacts: Vec<ReplayTreeContact>,
    /// The power-line spans the platforms' own map gained this tick, in a
    /// run that maps the world for itself (R3): whole spans, each once, the
    /// wire-hazard prior's on the first tick. Applying the frames in order
    /// gives the wires the map holds.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub mapped_wires: Vec<argusnet_world::powerline::PowerLine>,
}

/// One platform touching one tree in one tick (R2).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayTreeContact {
    pub node_id: String,
    pub tree_id: String,
    pub part: argusnet_world::trees::TreePart,
}

/// The trees each mobile platform touched in the last tick: the segment it
/// flew, from where its velocity says it was a tick ago to where it is,
/// against every trunk (swept: a trunk passed between two ticks counts) and
/// every crown it crossed, in the order it met them. Stateless, so a
/// resumed run records what the uninterrupted one would.
pub fn tree_contacts(
    trees: &argusnet_world::trees::TreeLayer,
    nodes: &[argusnet_types::NodeState],
    dt_s: f64,
) -> Vec<ReplayTreeContact> {
    use argusnet_world::trees::TreePart;
    let mut contacts = Vec::new();
    for node in nodes.iter().filter(|node| node.is_mobile) {
        let to: [f64; 3] = std::array::from_fn(|i| node.position[i]);
        let from: [f64; 3] = std::array::from_fn(|i| to[i] - node.velocity[i] * dt_s);
        let mut met: Vec<(f64, TreePart, &str)> = trees
            .crowns_along(from, to)
            .into_iter()
            .map(|(t0, _, tree)| (t0, TreePart::Crown, tree.id.as_str()))
            .collect();
        if let Some(hit) = trees.first_trunk_hit(from, to) {
            met.push((hit.t, TreePart::Trunk, hit.tree.id.as_str()));
        }
        met.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)).then(a.2.cmp(b.2)));
        contacts.extend(met.into_iter().map(|(_, part, id)| ReplayTreeContact {
            node_id: node.node_id.clone(),
            tree_id: id.to_string(),
            part,
        }));
    }
    contacts
}

/// A `SafetyValidationResult` as the reference writes it: the fused frame's
/// `SafetyEvent`, read back by `_safety_event_from_proto`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySafetyEvent {
    pub validation_id: String,
    pub subject_id: String,
    pub accepted: bool,
    pub timestamp_s: f64,
    pub validator: String,
    pub violations: Vec<String>,
    pub clamped: bool,
    pub reason: Option<String>,
}

impl From<&argusnet_types::SafetyEvent> for ReplaySafetyEvent {
    fn from(event: &argusnet_types::SafetyEvent) -> Self {
        let nonempty = |text: &str| (!text.is_empty()).then(|| text.to_string());
        Self {
            validation_id: event.event_id.clone(),
            subject_id: event.subject_id.clone(),
            accepted: !event.blocked,
            timestamp_s: event.timestamp_s,
            // `_safety_event_from_proto` names the validator itself.
            validator: "blocking_monitor".to_string(),
            violations: event.violations.clone(),
            clamped: event.command_clamped,
            // `value.reason or value.state or None`.
            reason: nonempty(&event.reason).or_else(|| nonempty(&event.state)),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayMeta {
    pub schema_version: String,
    pub scenario_name: String,
    pub frame_count: usize,
    pub dt_s: f64,
    pub seed: u64,
    pub node_ids: Vec<String>,
    pub track_ids: Vec<String>,
    /// The producing runtime. Python's writer has no equivalent field, which
    /// is the point: a document that came out of Rust should say so, so an
    /// investigation into a parity difference starts from fact rather than
    /// from the filename someone chose.
    pub producer: String,
    /// Absent rather than a wall-clock string when the caller has not supplied
    /// one. The fixture this crate commits must be byte-reproducible, and a
    /// timestamp taken at write time is the one field that cannot be.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub generated_at_utc: Option<String>,
    /// Everything else the reference's `meta` carries about the scenario and
    /// the run -- terrain, obstacles, land cover, zones, assignments, the
    /// tracker's platform settings, generation totals -- flattened into
    /// `meta` beside the fields above. Empty from a writer given none.
    #[serde(flatten)]
    pub scenario: BTreeMap<String, serde_json::Value>,
}

/// The name this writer puts in [`ReplayMeta::producer`].
pub const PRODUCER: &str = "argusnet-sim";

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplaySummary {
    pub mean_error_m: Option<f64>,
    pub peak_error_m: Option<f64>,
    pub mean_observations_per_frame: f64,
    pub mean_active_tracks: f64,
    pub mean_accepted_observations_per_frame: f64,
    pub mean_rejected_observations_per_frame: f64,
    pub total_accepted_observations: u64,
    pub total_rejected_observations: u64,
    pub observation_rejection_rate: f64,
    pub rejection_counts: BTreeMap<String, u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReplayDocument {
    pub meta: ReplayMeta,
    pub summary: ReplaySummary,
    pub frames: Vec<ReplayFrame>,
}

/// Accumulates frames and freezes them into a [`ReplayDocument`].
///
/// Frames are pushed in tick order and never reordered: the schema's `frames`
/// is an ordered sequence, and a consumer scrubbing a timeline reads the index
/// as time.
#[derive(Debug, Clone)]
pub struct ReplayWriter {
    scenario_name: String,
    dt_s: f64,
    seed: u64,
    generated_at_utc: Option<String>,
    meta: BTreeMap<String, serde_json::Value>,
    frames: Vec<ReplayFrame>,
}

impl ReplayWriter {
    pub fn new(scenario_name: impl Into<String>, dt_s: f64, seed: u64) -> Self {
        Self {
            scenario_name: scenario_name.into(),
            dt_s,
            seed,
            generated_at_utc: None,
            meta: BTreeMap::new(),
            frames: Vec::new(),
        }
    }

    /// The last frame pushed.
    pub fn last_frame(&self) -> Option<&ReplayFrame> {
        self.frames.last()
    }

    /// Drop every frame pushed so far, keeping `meta`: a reset run starts its
    /// replay again.
    pub fn clear_frames(&mut self) {
        self.frames.clear();
    }

    /// Add keys to `meta`. A key the writer computes itself (`frame_count`,
    /// `node_ids` and the rest of [`ReplayMeta`]'s fields) is not
    /// overridable: it is dropped here rather than emitted twice.
    pub fn extend_meta(&mut self, entries: impl IntoIterator<Item = (String, serde_json::Value)>) {
        const OWN: [&str; 9] = [
            "schema_version",
            "scenario_name",
            "frame_count",
            "dt_s",
            "seed",
            "node_ids",
            "track_ids",
            "producer",
            "generated_at_utc",
        ];
        for (key, value) in entries {
            if !OWN.contains(&key.as_str()) {
                self.meta.insert(key, value);
            }
        }
    }

    /// The keys added with [`ReplayWriter::extend_meta`].
    pub fn extra_meta(&self) -> &BTreeMap<String, serde_json::Value> {
        &self.meta
    }

    /// Stamp the document with a generation time.
    ///
    /// Taken from the caller rather than read from the clock here, so that a
    /// test can produce a byte-identical document twice. A writer that called
    /// `SystemTime::now()` internally could not be fixture-compared at all.
    pub fn generated_at_utc(mut self, timestamp: impl Into<String>) -> Self {
        self.generated_at_utc = Some(timestamp.into());
        self
    }

    pub fn push_frame(&mut self, frame: ReplayFrame) {
        self.frames.push(frame);
    }

    /// Build a frame from runtime values.
    ///
    /// Takes slices rather than a `RuntimeSnapshot` because a snapshot has
    /// truth filtered out of it by design, and a replay's `truths` array is
    /// what every accuracy metric in `evaluation/` is computed against.
    #[allow(clippy::too_many_arguments)]
    pub fn frame(
        timestamp_s: f64,
        nodes: &[NodeState],
        observations: &[BearingObservation],
        rejected: &[ObservationRejection],
        tracks: &[TrackState],
        truths: &[TruthState],
        metrics: &PlatformMetrics,
        generation_rejections: &[ObservationRejection],
    ) -> ReplayFrame {
        ReplayFrame {
            timestamp_s,
            nodes: nodes.iter().map(ReplayNode::from).collect(),
            observations: observations.iter().map(ReplayObservation::from).collect(),
            rejected_observations: rejected.iter().map(ReplayRejection::from).collect(),
            tracks: tracks.iter().map(ReplayTrack::from).collect(),
            truths: truths.iter().map(ReplayTruth::from).collect(),
            metrics: ReplayMetrics::from(metrics),
            generation_rejections: generation_rejections
                .iter()
                .map(ReplayRejection::from)
                .collect(),
            mapping_state: None,
            localization_state: None,
            inspection_events: Vec::new(),
            deconfliction_events: Vec::new(),
            scan_mission_state: None,
            target_metadata: Vec::new(),
            safety_events: Vec::new(),
            belief_updates: Vec::new(),
            canopy_updates: Vec::new(),
            range_scans: Vec::new(),
            world_contacts: Vec::new(),
            tree_contacts: Vec::new(),
            mapped_wires: Vec::new(),
        }
    }

    pub fn frame_count(&self) -> usize {
        self.frames.len()
    }

    /// Freeze into a document, computing `meta` and `summary` from the frames.
    ///
    /// `node_ids` and `track_ids` are sorted and deduplicated across the whole
    /// run, matching `build_replay_document`. Sorted rather than
    /// first-appearance ordered so two runs that discovered the same platforms
    /// in a different order produce comparable documents.
    pub fn finish(self) -> ReplayDocument {
        let mut node_ids: Vec<String> = self
            .frames
            .iter()
            .flat_map(|frame| frame.nodes.iter().map(|node| node.node_id.clone()))
            .collect();
        node_ids.sort();
        node_ids.dedup();

        let mut track_ids: Vec<String> = self
            .frames
            .iter()
            .flat_map(|frame| frame.tracks.iter().map(|track| track.track_id.clone()))
            .collect();
        track_ids.sort();
        track_ids.dedup();

        let summary = self.summarize();
        ReplayDocument {
            meta: ReplayMeta {
                schema_version: SCHEMA_VERSION.to_string(),
                scenario_name: self.scenario_name,
                frame_count: self.frames.len(),
                dt_s: self.dt_s,
                seed: self.seed,
                node_ids,
                track_ids,
                producer: PRODUCER.to_string(),
                generated_at_utc: self.generated_at_utc,
                scenario: self.meta,
            },
            summary,
            frames: self.frames,
        }
    }

    fn summarize(&self) -> ReplaySummary {
        let errors: Vec<f64> = self
            .frames
            .iter()
            .filter_map(|frame| frame.metrics.mean_error_m)
            .collect();
        let frame_count = self.frames.len() as f64;

        // `mean` over an empty run is 0.0, not NaN -- matching the reference,
        // which guards every `np.mean` with an emptiness check. `mean_error_m`
        // and `peak_error_m` are the exception: they are nullable in the
        // schema, and `None` is the honest answer when no frame had truth to
        // measure against. Reporting 0.0 there would claim perfect accuracy.
        let mean = |values: &[f64]| -> f64 {
            if values.is_empty() {
                0.0
            } else {
                values.iter().sum::<f64>() / values.len() as f64
            }
        };

        let accepted: u64 = self
            .frames
            .iter()
            .map(|frame| u64::from(frame.metrics.accepted_observation_count))
            .sum();
        let rejected: u64 = self
            .frames
            .iter()
            .map(|frame| u64::from(frame.metrics.rejected_observation_count))
            .sum();

        let mut rejection_counts: BTreeMap<String, u32> = BTreeMap::new();
        for frame in &self.frames {
            for (reason, count) in &frame.metrics.rejection_counts {
                *rejection_counts.entry(reason.clone()).or_insert(0) += count;
            }
        }

        let per_frame = |total: u64| -> f64 {
            if self.frames.is_empty() {
                0.0
            } else {
                total as f64 / frame_count
            }
        };

        ReplaySummary {
            mean_error_m: if errors.is_empty() {
                None
            } else {
                Some(mean(&errors))
            },
            peak_error_m: errors.iter().copied().fold(None, |best: Option<f64>, e| {
                Some(best.map_or(e, |b| b.max(e)))
            }),
            mean_observations_per_frame: per_frame(
                self.frames
                    .iter()
                    .map(|f| u64::from(f.metrics.observation_count))
                    .sum(),
            ),
            mean_active_tracks: per_frame(
                self.frames
                    .iter()
                    .map(|f| u64::from(f.metrics.active_track_count))
                    .sum(),
            ),
            mean_accepted_observations_per_frame: per_frame(accepted),
            mean_rejected_observations_per_frame: per_frame(rejected),
            total_accepted_observations: accepted,
            total_rejected_observations: rejected,
            // `max(..., 1)` mirrors the reference's guard against a run with
            // no observations at all, where the rate is 0.0 rather than 0/0.
            observation_rejection_rate: rejected as f64 / (accepted + rejected).max(1) as f64,
            rejection_counts,
        }
    }
}

/// Serialize a document the way the fixture and the writer both expect.
///
/// Pretty-printed with a trailing newline: the fixture is reviewed in diffs,
/// and a single-line 40 KB JSON blob is not reviewable. `serde_json`'s pretty
/// printer is stable across versions for a given input, which is what makes
/// the byte comparison in the tests meaningful.
pub fn to_json(document: &ReplayDocument) -> Result<String, serde_json::Error> {
    let mut text = serde_json::to_string_pretty(document)?;
    text.push('\n');
    Ok(text)
}

#[cfg(test)]
mod tests {
    use super::*;
    use argusnet_types::{Matrix6, Vector3};
    use std::path::PathBuf;
    use std::sync::Arc;

    /// Where the committed fixture lives, relative to the workspace root.
    const FIXTURE: &str = "tests/fixtures/rust_replay_fixture.json";

    fn fixture_path() -> PathBuf {
        // CARGO_MANIFEST_DIR is rust/argusnet-sim; the fixtures live at the
        // repository root, which is two levels up.
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(FIXTURE)
    }

    fn node(id: &str, x: f64, battery: Option<f64>) -> NodeState {
        NodeState {
            node_id: id.to_string(),
            position: Vector3::new(x, 10.0, 50.0),
            velocity: Vector3::new(1.5, 0.0, -0.25),
            is_mobile: battery.is_some(),
            timestamp_s: 0.5,
            health: 0.875,
            // Any of `argusnet_types::SENSOR_TYPES` is valid in the schema
            // since decision D-14; the fixture keeps the radar it had.
            sensor_type: "radar".to_string(),
            fov_half_angle_deg: Some(37.5),
            max_range_m: Some(1200.0),
            battery_fraction: battery,
        }
    }

    fn track(id: &str) -> TrackState {
        TrackState {
            track_id: id.to_string(),
            timestamp_s: 0.5,
            position: Vector3::new(100.0, 200.0, 30.0),
            velocity: Vector3::new(-2.0, 1.0, 0.0),
            // Asymmetric on purpose: a symmetric matrix cannot tell a
            // row-major serialization from a column-major one.
            covariance: Matrix6::from_fn(|row, column| (row * 6 + column) as f64),
            measurement_std_m: 2.25,
            update_count: 7,
            stale_steps: 1,
            lifecycle_state: Some("confirmed".to_string()),
            quality_score: Some(0.8),
            mode_probability_cv: Some(0.6),
            contributing_node_ids: Arc::new(vec!["node-a".to_string()]),
            evidence_time_s: None,
        }
    }

    fn metrics() -> PlatformMetrics {
        PlatformMetrics {
            mean_error_m: Some(3.0),
            max_error_m: Some(4.5),
            active_track_count: 1,
            observation_count: 4,
            accepted_observation_count: 3,
            rejected_observation_count: 1,
            mean_measurement_std_m: Some(2.25),
            track_errors_m: BTreeMap::from([("track-1".to_string(), 3.0)]),
            rejection_counts: BTreeMap::from([("out_of_range".to_string(), 1)]),
            accepted_observations_by_target: BTreeMap::from([("target-1".to_string(), 3)]),
            rejected_observations_by_target: BTreeMap::from([("target-1".to_string(), 1)]),
        }
    }

    fn truth() -> TruthState {
        TruthState {
            target_id: "target-1".to_string(),
            position: Vector3::new(105.0, 198.0, 29.0),
            velocity: Vector3::new(-2.0, 1.0, 0.0),
            timestamp_s: 0.5,
        }
    }

    fn observation() -> BearingObservation {
        BearingObservation {
            node_id: "node-a".to_string(),
            target_id: "target-1".to_string(),
            origin: Vector3::new(0.0, 10.0, 50.0),
            direction: Vector3::new(0.6, 0.8, 0.0),
            bearing_std_rad: 0.015,
            timestamp_s: 0.5,
            confidence: 0.9,
        }
    }

    /// The document the committed fixture holds. Two frames, so a consumer
    /// that only ever reads `frames[0]` cannot pass by accident.
    fn fixture_document() -> ReplayDocument {
        let mut writer = ReplayWriter::new("rust-replay-contract", 0.5, 42)
            .generated_at_utc("2026-01-01T00:00:00+00:00");
        for tick in 0..2u32 {
            let offset = f64::from(tick) * 10.0;
            writer.push_frame(ReplayWriter::frame(
                0.5 + f64::from(tick) * 0.5,
                &[
                    node("node-a", offset, Some(1.0 - f64::from(tick) * 0.25)),
                    // A second drone, so the fixture's node count (3) differs
                    // from its frame count (2). With two of each, a mutant
                    // that reported `node_ids.len()` as `frame_count`
                    // survived: both numbers were 2 and every assertion held.
                    node("node-b", offset + 25.0, Some(0.5)),
                    // A fixed platform: no battery, which must serialize as the
                    // -1.0 sentinel and not as 0.0.
                    node("tower", 500.0, None),
                ],
                &[observation()],
                &[],
                &[track("track-1")],
                &[truth()],
                &metrics(),
                &[],
            ));
        }
        writer.finish()
    }

    // -----------------------------------------------------------------
    // The fixture contract
    // -----------------------------------------------------------------

    /// The committed fixture is exactly what this writer emits.
    ///
    /// Same arrangement as `scripts/regenerate_proto.py --check`: the artifact
    /// Python validates against is byte-compared here, so neither language can
    /// silently rebaseline the other. Regenerate with
    /// `ARGUSNET_BLESS_REPLAY_FIXTURE=1 cargo test -p argusnet-sim`.
    #[test]
    fn the_committed_fixture_is_what_this_writer_emits() {
        let expected = to_json(&fixture_document()).expect("document serializes");
        let path = fixture_path();

        if std::env::var_os("ARGUSNET_BLESS_REPLAY_FIXTURE").is_some() {
            std::fs::create_dir_all(path.parent().expect("fixture has a parent"))
                .expect("fixture directory");
            std::fs::write(&path, &expected).expect("fixture written");
            return;
        }

        let actual = std::fs::read_to_string(&path).unwrap_or_else(|error| {
            panic!(
                "{}: {error}. Regenerate with \
                 ARGUSNET_BLESS_REPLAY_FIXTURE=1 cargo test -p argusnet-sim",
                path.display()
            )
        });
        assert_eq!(
            actual, expected,
            "the committed replay fixture is stale. Python validates it against \
             docs/replay-schema.json, so regenerate it deliberately -- with \
             ARGUSNET_BLESS_REPLAY_FIXTURE=1 -- and re-read the schema before \
             assuming the new shape is still in contract."
        );
    }

    // -----------------------------------------------------------------
    // Shapes the schema pins that serde would otherwise get wrong
    // -----------------------------------------------------------------

    /// `nalgebra`'s own `Serialize` emits a struct, not an array.
    ///
    /// The schema's `vector3` is `minItems: 3, maxItems: 3`, so deriving
    /// `Serialize` straight onto `Vector3` would produce a document that fails
    /// validation -- and nothing on the Rust side checks the schema, so it
    /// would fail in Python CI with a shape error rather than here.
    #[test]
    fn vectors_serialize_as_three_element_arrays() {
        let value = serde_json::to_value(ReplayNode::from(&node("n", 1.0, Some(0.5))))
            .expect("node serializes");
        assert_eq!(value["position"], serde_json::json!([1.0, 10.0, 50.0]));
        assert_eq!(value["velocity"], serde_json::json!([1.5, 0.0, -0.25]));
    }

    /// The schema says row-major; `Matrix6` is column-major in memory.
    ///
    /// `covariance.as_slice()` would compile, produce 36 numbers, and quietly
    /// transpose the matrix. With an asymmetric fixture the two orders differ,
    /// which is the only reason this test can see the difference.
    #[test]
    fn the_covariance_is_flattened_row_major_not_as_slice() {
        let replay = ReplayTrack::from(&track("track-1"));
        let expected: Vec<f64> = (0..36).map(|i| i as f64).collect();
        assert_eq!(replay.covariance, expected);

        // And the transpose is genuinely different, so the assertion above is
        // not satisfied by both orders.
        let column_major: Vec<f64> = track("track-1").covariance.as_slice().to_vec();
        assert_ne!(
            column_major, expected,
            "fixture is symmetric -- this test cannot detect a transpose"
        );
    }

    /// `_safety_event_from_proto`: `reason or state or None`, the validator
    /// named by the reader, and `accepted` as the inverse of `blocked`.
    #[test]
    fn a_safety_event_reads_back_as_the_reference_reads_it() {
        let event = |state: &str, reason: &str| argusnet_types::SafetyEvent {
            event_id: "safety-3-drone-1".into(),
            subject_id: "drone-1".into(),
            timestamp_s: 1.5,
            state: state.into(),
            violations: vec!["max_speed".into()],
            command_clamped: true,
            blocked: true,
            reason: reason.into(),
        };
        let read = ReplaySafetyEvent::from(&event("abort", "overspeed"));
        assert_eq!(read.reason.as_deref(), Some("overspeed"));
        assert_eq!(read.validator, "blocking_monitor");
        assert!(!read.accepted && read.clamped);
        assert_eq!(read.validation_id, "safety-3-drone-1");
        assert_eq!(
            ReplaySafetyEvent::from(&event("abort", ""))
                .reason
                .as_deref(),
            Some("abort")
        );
        assert_eq!(ReplaySafetyEvent::from(&event("", "")).reason, None);
    }

    /// The reference writer emits the lifecycle and quality of every track
    /// (P4.9c found them missing here).
    #[test]
    fn a_track_carries_its_lifecycle_and_quality() {
        let value = serde_json::to_value(ReplayTrack::from(&track("t"))).expect("serializes");
        assert_eq!(value["lifecycle_state"], "confirmed");
        assert_eq!(value["quality_score"], 0.8);
    }

    /// 36 elements, because `evaluation/replay.py` reshapes to (6, 6)
    /// unconditionally and anything else crashes the Python reader.
    #[test]
    fn the_covariance_is_thirty_six_elements() {
        assert_eq!(ReplayTrack::from(&track("t")).covariance.len(), 36);
    }

    // -----------------------------------------------------------------
    // The battery sentinel, end to end
    // -----------------------------------------------------------------

    /// A platform with no battery writes `-1.0`; a flat one writes `0.0`.
    ///
    /// The runtime type distinguishes these with `Option`, the wire with an
    /// `optional` field, and the replay with this sentinel. This is the third
    /// representation of the same distinction, and the one a viewer reads.
    #[test]
    fn a_platform_with_no_battery_writes_the_sentinel_not_zero() {
        assert_eq!(
            ReplayNode::from(&node("tower", 0.0, None)).battery_fraction,
            BATTERY_NOT_APPLICABLE
        );
        assert_eq!(
            ReplayNode::from(&node("drone", 0.0, Some(0.0))).battery_fraction,
            0.0
        );
        assert_ne!(BATTERY_NOT_APPLICABLE, 0.0);
    }

    /// The sentinel here and the one in the battery stage are the same value.
    /// Two constants that happen to agree today are a future divergence.
    #[test]
    fn the_replay_sentinel_matches_the_battery_stages() {
        assert_eq!(
            BATTERY_NOT_APPLICABLE,
            crate::battery::NOT_APPLICABLE_FRACTION
        );
    }

    /// An unset sensor field writes the schema's `0`, which is its documented
    /// "unset" -- unlike battery, where zero is a real reading.
    #[test]
    fn unset_sensor_fields_write_zero() {
        let bare = NodeState {
            node_id: "bare".to_string(),
            ..Default::default()
        };
        let replay = ReplayNode::from(&bare);
        assert_eq!(replay.fov_half_angle_deg, 0.0);
        assert_eq!(replay.max_range_m, 0.0);
        assert_eq!(replay.sensor_type, argusnet_types::DEFAULT_SENSOR_TYPE);
    }

    // -----------------------------------------------------------------
    // meta and summary
    // -----------------------------------------------------------------

    #[test]
    fn the_document_declares_its_schema_version_and_producer() {
        let document = fixture_document();
        assert_eq!(document.meta.schema_version, SCHEMA_VERSION);
        assert_eq!(document.meta.schema_version, "2.0");
        assert_eq!(document.meta.producer, PRODUCER);
        assert_eq!(document.meta.frame_count, document.frames.len());
        assert_eq!(document.meta.frame_count, 2);
        // Pinned against the id count too. These were both 2 in the original
        // fixture, so `frame_count: node_ids.len()` was indistinguishable from
        // the correct expression -- a mutant survived on that coincidence.
        assert_eq!(document.meta.node_ids.len(), 3);
        assert_ne!(
            document.meta.frame_count,
            document.meta.node_ids.len(),
            "the fixture must not let a frame count and an id count coincide"
        );
    }

    /// Sorted and deduplicated across the whole run, not per frame.
    #[test]
    fn node_and_track_ids_are_sorted_and_deduplicated_across_frames() {
        let document = fixture_document();
        assert_eq!(document.meta.node_ids, vec!["node-a", "node-b", "tower"]);
        assert_eq!(document.meta.track_ids, vec!["track-1"]);
    }

    /// `frame_count` counts frames, and nothing else that happens to be the
    /// same length.
    ///
    /// Stated over several shapes so no single coincidence can satisfy it:
    /// one frame with four platforms, four frames with one, and so on.
    #[test]
    fn frame_count_counts_frames_not_platforms() {
        for (frames, platforms) in [(1usize, 4usize), (4, 1), (3, 7), (5, 2)] {
            let mut writer = ReplayWriter::new("counts", 0.1, 1);
            let fleet: Vec<NodeState> = (0..platforms)
                .map(|i| node(&format!("node-{i}"), i as f64, Some(1.0)))
                .collect();
            for tick in 0..frames {
                writer.push_frame(ReplayWriter::frame(
                    tick as f64 * 0.1,
                    &fleet,
                    &[],
                    &[],
                    &[],
                    &[],
                    &metrics(),
                    &[],
                ));
            }
            let document = writer.finish();
            assert_eq!(
                document.meta.frame_count, frames,
                "{frames} frames of {platforms} platforms"
            );
            assert_eq!(document.meta.node_ids.len(), platforms);
        }
    }

    #[test]
    fn the_summary_totals_every_frame_not_just_the_last() {
        let document = fixture_document();
        // Two frames, metrics() each: 3 accepted, 1 rejected per frame.
        assert_eq!(document.summary.total_accepted_observations, 6);
        assert_eq!(document.summary.total_rejected_observations, 2);
        assert_eq!(document.summary.mean_accepted_observations_per_frame, 3.0);
        assert_eq!(document.summary.observation_rejection_rate, 0.25);
        assert_eq!(
            document.summary.rejection_counts,
            BTreeMap::from([("out_of_range".to_string(), 2)]),
            "rejection reasons accumulate across frames"
        );
    }

    /// An empty run reports `None` for error, not `0.0`.
    ///
    /// Zero would claim perfect accuracy for a run that measured nothing. The
    /// schema makes both error fields nullable precisely so this case can be
    /// stated rather than faked.
    #[test]
    fn a_run_with_no_measured_error_reports_null_not_zero() {
        let mut writer = ReplayWriter::new("empty", 0.1, 1);
        let mut no_error = metrics();
        no_error.mean_error_m = None;
        no_error.max_error_m = None;
        writer.push_frame(ReplayWriter::frame(
            0.0,
            &[],
            &[],
            &[],
            &[],
            &[],
            &no_error,
            &[],
        ));
        let document = writer.finish();
        assert_eq!(document.summary.mean_error_m, None);
        assert_eq!(document.summary.peak_error_m, None);
        // But the counting statistics are still real numbers.
        assert_eq!(document.summary.mean_active_tracks, 1.0);
    }

    /// A writer with no frames at all must not divide by zero.
    #[test]
    fn a_run_with_no_frames_produces_zeros_rather_than_nan() {
        let document = ReplayWriter::new("nothing", 0.1, 1).finish();
        assert_eq!(document.meta.frame_count, 0);
        assert_eq!(document.summary.mean_observations_per_frame, 0.0);
        assert_eq!(document.summary.observation_rejection_rate, 0.0);
        assert!(document.summary.mean_active_tracks.is_finite());
    }

    #[test]
    fn peak_error_is_the_maximum_across_frames_not_the_last() {
        let mut writer = ReplayWriter::new("peaks", 0.1, 1);
        for error in [1.0, 9.0, 4.0] {
            let mut m = metrics();
            m.mean_error_m = Some(error);
            writer.push_frame(ReplayWriter::frame(0.0, &[], &[], &[], &[], &[], &m, &[]));
        }
        let document = writer.finish();
        assert_eq!(document.summary.peak_error_m, Some(9.0));
        assert_eq!(document.summary.mean_error_m, Some((1.0 + 9.0 + 4.0) / 3.0));
    }

    /// Frames keep tick order. The schema's `frames` is an ordered sequence
    /// and a scrubbing consumer reads the index as time.
    #[test]
    fn frames_are_kept_in_push_order() {
        let mut writer = ReplayWriter::new("order", 0.5, 1);
        for stamp in [0.0, 0.5, 1.0] {
            writer.push_frame(ReplayWriter::frame(
                stamp,
                &[],
                &[],
                &[],
                &[],
                &[],
                &metrics(),
                &[],
            ));
        }
        let stamps: Vec<f64> = writer
            .finish()
            .frames
            .iter()
            .map(|frame| frame.timestamp_s)
            .collect();
        assert_eq!(stamps, vec![0.0, 0.5, 1.0]);
    }

    /// The document round-trips through its own JSON.
    #[test]
    fn the_document_round_trips_through_json() {
        let original = fixture_document();
        let text = to_json(&original).expect("serializes");
        let restored: ReplayDocument = serde_json::from_str(&text).expect("deserializes");
        assert_eq!(restored, original);
    }

    /// Every key the schema marks required is present on every frame.
    ///
    /// Checked structurally here so a `skip_serializing_if` added to the wrong
    /// field fails in this crate rather than in Python CI.
    #[test]
    fn every_frame_carries_the_seven_required_keys() {
        let value = serde_json::to_value(fixture_document()).expect("serializes");
        for frame in value["frames"].as_array().expect("frames is an array") {
            for key in [
                "timestamp_s",
                "nodes",
                "observations",
                "rejected_observations",
                "tracks",
                "truths",
                "metrics",
            ] {
                assert!(
                    frame.get(key).is_some(),
                    "frame is missing the required key {key}"
                );
            }
        }
        assert!(value["meta"].get("dt_s").is_some(), "meta requires dt_s");
    }

    /// Generation time is caller-supplied, so a document is reproducible.
    ///
    /// A writer that read the clock internally could not be fixture-compared
    /// at all -- the byte comparison above would fail on every run.
    #[test]
    fn two_writers_with_the_same_inputs_produce_identical_bytes() {
        let first = to_json(&fixture_document()).expect("serializes");
        let second = to_json(&fixture_document()).expect("serializes");
        assert_eq!(first, second);
    }

    /// Without a stamp the key is absent rather than null or empty.
    #[test]
    fn an_unstamped_document_omits_generated_at_utc() {
        let document = ReplayWriter::new("unstamped", 0.1, 1).finish();
        let value = serde_json::to_value(&document).expect("serializes");
        assert!(value["meta"].get("generated_at_utc").is_none());
    }
}

#[cfg(test)]
mod tree_contact_tests {
    use super::*;
    use argusnet_world::trees::{Species, Tree, TreeLayer, TreePart};

    fn node(
        id: &str,
        position: [f64; 3],
        velocity: [f64; 3],
        mobile: bool,
    ) -> argusnet_types::NodeState {
        let mut state: argusnet_types::NodeState = serde_json::from_value(serde_json::json!({
            "node_id": id,
            "position": position,
            "velocity": velocity,
            "is_mobile": mobile,
            "timestamp_s": 0.0,
            "health": 1.0,
            "sensor_type": "optical",
            "fov_half_angle_deg": 30.0,
            "max_range_m": 100.0,
        }))
        .expect("node");
        state.is_mobile = mobile;
        state
    }

    /// The last tick's flight, from its velocity, against each crown and
    /// trunk: swept, so a crown or trunk passed between ticks counts, in
    /// the order met; a fixed node never touches one.
    #[test]
    fn contacts_are_the_trees_the_last_ticks_flight_crossed() {
        let tree = |id: &str, x: f64| Tree {
            id: id.into(),
            xy_m: [x, 0.0],
            base_z_m: 0.0,
            height_m: 20.0,
            crown_base_m: 8.0,
            crown_radius_m: 3.0,
            trunk_radius_m: 0.3,
            species: Species::Broadleaf,
            leaf_density: 0.8,
        };
        let layer = TreeLayer::new(vec![tree("a", 10.0), tree("b", 20.0)]).unwrap();
        // At 14 m, from x = 0 to x = 28 in a 0.25 s tick: both crowns
        // (and, inside each, its trunk, which reaches the crown's middle).
        let flying = node("d", [28.0, 0.0, 14.0], [112.0, 0.0, 0.0], true);
        let contacts = tree_contacts(&layer, &[flying], 0.25);
        let seen: Vec<_> = contacts
            .iter()
            .map(|c| (c.tree_id.as_str(), c.part))
            .collect();
        assert_eq!(
            seen,
            [
                ("a", TreePart::Crown),
                ("a", TreePart::Trunk),
                ("b", TreePart::Crown)
            ]
        );
        // Over the trees, or a ground station inside a crown: nothing.
        let high = node("d", [28.0, 0.0, 30.0], [112.0, 0.0, 0.0], true);
        let fixed = node("g", [10.0, 0.0, 14.0], [0.0; 3], false);
        assert!(tree_contacts(&layer, &[high, fixed], 0.25).is_empty());
        // Hovering inside a crown is a contact every tick.
        let hovering = node("d", [11.0, 0.0, 14.0], [0.0; 3], true);
        assert_eq!(tree_contacts(&layer, &[hovering], 0.25).len(), 1);
    }
}
