//! G0 (E1.2): every string the Rust runtime writes into a replay's closed
//! vocabularies is one `docs/replay-schema.json` accepts. Each enum's variants
//! are listed through an exhaustive `match`, so a new variant fails to compile
//! here until it is checked; the table of vocabularies and what their readers
//! do with an unknown value is `docs/migration/CONTRACT_VOCABULARIES.md`.

use std::collections::BTreeSet;

use argusnet_core::LifecycleState;
use argusnet_mapping::BeliefCellStatus;
use argusnet_planning::deconfliction::Resolution;
use argusnet_sensing::cascade::RejectReason;
use argusnet_sim::delivery::DropReason;
use argusnet_sim::mapping::INSPECTION_EVENT_TYPES;
use argusnet_sim::mission::MissionPhase;

fn schema_enum(pointer: &str) -> BTreeSet<String> {
    let path = format!(
        "{}/../../docs/replay-schema.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let schema: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&path).expect("schema")).expect("json");
    schema
        .pointer(pointer)
        .and_then(|value| value.as_array())
        .unwrap_or_else(|| panic!("no enum at {pointer}"))
        .iter()
        .filter_map(|value| value.as_str().map(str::to_owned))
        .collect()
}

fn assert_within(what: &str, written: &[&str], pointer: &str) {
    let allowed = schema_enum(pointer);
    let outside: Vec<&&str> = written.iter().filter(|w| !allowed.contains(**w)).collect();
    assert!(
        outside.is_empty(),
        "{what}: the runtime writes {outside:?}, which the replay schema ({pointer}) refuses"
    );
}

#[test]
fn scan_mission_phases() {
    fn every(phase: MissionPhase) -> MissionPhase {
        match phase {
            MissionPhase::Scanning
            | MissionPhase::Localizing
            | MissionPhase::Inspecting
            | MissionPhase::Egress
            | MissionPhase::Complete => phase,
        }
    }
    let written: Vec<&str> = [
        MissionPhase::Scanning,
        MissionPhase::Localizing,
        MissionPhase::Inspecting,
        MissionPhase::Egress,
        MissionPhase::Complete,
    ]
    .map(|phase| every(phase).as_str())
    .to_vec();
    assert_within(
        "scan mission phase",
        &written,
        "/definitions/scan_mission_state/properties/phase/enum",
    );
}

#[test]
fn observation_rejection_reasons() {
    fn every(reason: RejectReason) -> &'static str {
        match reason {
            RejectReason::OutOfRange
            | RejectReason::OutsideFov
            | RejectReason::Dropout
            | RejectReason::LowElevation
            | RejectReason::TerrainOcclusion
            | RejectReason::BuildingOcclusion
            | RejectReason::WallOcclusion
            | RejectReason::VegetationOcclusion
            | RejectReason::OutOfCoverage => reason.as_str(),
        }
    }
    let mut written: Vec<&str> = [
        RejectReason::OutOfRange,
        RejectReason::OutsideFov,
        RejectReason::Dropout,
        RejectReason::LowElevation,
        RejectReason::TerrainOcclusion,
        RejectReason::BuildingOcclusion,
        RejectReason::WallOcclusion,
        RejectReason::VegetationOcclusion,
        RejectReason::OutOfCoverage,
    ]
    .map(every)
    .to_vec();
    // The fusion stage's own rejections.
    written.extend([
        argusnet_core::REJECT_UNKNOWN_NODE,
        argusnet_core::REJECT_INVALID_TARGET,
        argusnet_core::REJECT_INVALID_DIRECTION,
        argusnet_core::REJECT_LOW_CONFIDENCE,
        argusnet_core::REJECT_INVALID_BEARING_STD,
        argusnet_core::REJECT_EXCESS_BEARING_STD,
        argusnet_core::REJECT_TIMESTAMP_SKEW,
        argusnet_core::REJECT_DUPLICATE_NODE,
        argusnet_core::REJECT_INSUFFICIENT_CLUSTER,
        argusnet_core::REJECT_WEAK_GEOMETRY,
        argusnet_core::REJECT_FUSION_FAILURE,
    ]);
    // Timestamped delivery's drops (B4), each with its reason.
    fn delivered(reason: DropReason) -> &'static str {
        match reason {
            DropReason::Late { .. }
            | DropReason::Duplicate
            | DropReason::StaleEpoch { .. }
            | DropReason::Overflow { .. } => reason.code(),
        }
    }
    written.extend(
        [
            DropReason::Late { horizon_tick: 0 },
            DropReason::Duplicate,
            DropReason::StaleEpoch { current_epoch: 0 },
            DropReason::Overflow { capacity: 0 },
        ]
        .map(delivered),
    );
    assert_within(
        "observation rejection reason",
        &written,
        "/definitions/observation_rejection/properties/reason/enum",
    );
}

#[test]
fn track_lifecycle_states() {
    fn every(state: LifecycleState) -> &'static str {
        match state {
            LifecycleState::Tentative
            | LifecycleState::Confirmed
            | LifecycleState::Coasting
            | LifecycleState::Deleted => state.as_str(),
        }
    }
    let written: Vec<&str> = LifecycleState::ALL.map(every).to_vec();
    assert_within(
        "track lifecycle state",
        &written,
        "/definitions/track_state/properties/lifecycle_state/enum",
    );
}

#[test]
fn deconfliction_resolutions() {
    fn every(resolution: Resolution) -> &'static str {
        match resolution {
            Resolution::CorridorHold | Resolution::LateralOffset | Resolution::VerticalOffset => {
                resolution.as_str()
            }
        }
    }
    let written: Vec<&str> = [
        Resolution::CorridorHold,
        Resolution::LateralOffset,
        Resolution::VerticalOffset,
    ]
    .map(every)
    .to_vec();
    assert_within(
        "deconfliction resolution",
        &written,
        "/definitions/deconfliction_event/properties/resolution/enum",
    );
}

#[test]
fn belief_cell_statuses() {
    fn every(status: BeliefCellStatus) -> &'static str {
        match status {
            BeliefCellStatus::OutsideGeofence
            | BeliefCellStatus::Unknown
            | BeliefCellStatus::KnownObstacle
            | BeliefCellStatus::Uncertain
            | BeliefCellStatus::KnownSafe => status.as_str(),
        }
    }
    let written: Vec<&str> = [
        BeliefCellStatus::OutsideGeofence,
        BeliefCellStatus::Unknown,
        BeliefCellStatus::KnownObstacle,
        BeliefCellStatus::Uncertain,
        BeliefCellStatus::KnownSafe,
    ]
    .map(every)
    .to_vec();
    assert_within(
        "belief cell status",
        &written,
        "/definitions/belief_cell/properties/status/enum",
    );
}

#[test]
fn inspection_event_types() {
    assert_within(
        "inspection event type",
        &INSPECTION_EVENT_TYPES,
        "/definitions/inspection_event/properties/event_type/enum",
    );
}

/// D-14: the sensor vocabulary is one list, and the schema is exactly it.
#[test]
fn sensor_types_are_exactly_the_schemas() {
    let schema = schema_enum("/definitions/node_state/properties/sensor_type/enum");
    let declared: BTreeSet<String> = argusnet_types::SENSOR_TYPES
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(declared, schema);
}

#[test]
fn world_contact_kinds() {
    assert_within(
        "world contact kind",
        &argusnet_sim::belief_world::WORLD_CONTACT_KINDS,
        "/definitions/frame/properties/world_contacts/items/properties/kind/enum",
    );
}
