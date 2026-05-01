from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any

import numpy as np

Vector3 = np.ndarray


def vec3(x_m: float, y_m: float, z_m: float) -> Vector3:
    return np.array([x_m, y_m, z_m], dtype=float)


@dataclass(frozen=True)
class NodeState:
    node_id: str
    position: Vector3
    velocity: Vector3
    is_mobile: bool
    timestamp_s: float
    health: float = 1.0
    sensor_type: str = "optical"
    fov_half_angle_deg: float = 180.0
    max_range_m: float = 0.0
    battery_fraction: float = 1.0  # 0.0 (empty) → 1.0 (full); -1.0 = not applicable


@dataclass(frozen=True)
class BearingObservation:
    node_id: str
    target_id: str
    origin: Vector3
    direction: Vector3
    bearing_std_rad: float
    timestamp_s: float
    confidence: float = 1.0


@dataclass(frozen=True)
class ObservationRejection:
    node_id: str
    target_id: str
    timestamp_s: float
    reason: str
    detail: str = ""
    origin: Vector3 | None = None
    attempted_point: Vector3 | None = None
    closest_point: Vector3 | None = None
    blocker_type: str = ""
    first_hit_range_m: float | None = None


@dataclass(frozen=True)
class TruthState:
    target_id: str
    position: Vector3
    velocity: Vector3
    timestamp_s: float


@dataclass(frozen=True)
class TrackState:
    track_id: str
    timestamp_s: float
    position: Vector3
    velocity: Vector3
    covariance: np.ndarray
    measurement_std_m: float
    update_count: int
    stale_steps: int
    lifecycle_state: str | None = None
    quality_score: float | None = None


@dataclass(frozen=True)
class PlatformMetrics:
    mean_error_m: float | None
    max_error_m: float | None
    active_track_count: int
    observation_count: int
    accepted_observation_count: int
    rejected_observation_count: int
    mean_measurement_std_m: float | None
    track_errors_m: dict[str, float] = field(default_factory=dict)
    rejection_counts: dict[str, int] = field(default_factory=dict)
    accepted_observations_by_target: dict[str, int] = field(default_factory=dict)
    rejected_observations_by_target: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MappingState:
    coverage_fraction: float  # 0.0–1.0 fraction of grid cells observed
    covered_cells: int
    total_cells: int
    mean_revisits: float


@dataclass(frozen=True)
class LocalizationState:
    active_localizations: int
    mean_position_std_m: float
    mean_observation_confidence: float


@dataclass(frozen=True)
class InspectionEvent:
    zone_id: str
    node_id: str
    event_type: str  # "entered" | "coverage_updated" | "exited"
    timestamp_s: float
    zone_coverage_fraction: float


@dataclass(frozen=True)
class DeconflictionEvent:
    yielding_drone_id: str
    conflicting_drone_id: str
    predicted_separation_m: float
    resolution: str  # "lateral_offset" | "vertical_offset" | "corridor_hold"
    timestamp_s: float


@dataclass(frozen=True)
class PlatformFrame:
    timestamp_s: float
    nodes: list[NodeState]
    observations: list[BearingObservation]
    rejected_observations: list[ObservationRejection]
    tracks: list[TrackState]
    truths: list[TruthState]
    metrics: PlatformMetrics
    generation_rejections: list[ObservationRejection] = field(default_factory=list)
    mapping_state: MappingState | None = None
    localization_state: LocalizationState | None = None
    inspection_events: list[InspectionEvent] = field(default_factory=list)
    deconfliction_events: list[DeconflictionEvent] = field(default_factory=list)
    scan_mission_state: ScanMissionState | None = None
    tracking_mission_state: TrackingMissionState | None = None


@dataclass(frozen=True)
class NodeHealthMetrics:
    node_id: str
    last_seen_s: float
    observation_rate_hz: float
    mean_latency_s: float
    accepted_count: int
    rejected_count: int
    health_score: float


@dataclass(frozen=True)
class HealthReport:
    status: str
    started_at_utc: str
    processed_frame_count: int
    node_health: list[NodeHealthMetrics] = field(default_factory=list)
    mean_frame_rate_hz: float = 0.0
    mean_ingest_latency_s: float = 0.0
    active_node_count: int = 0
    stale_node_count: int = 0


ZONE_TYPE_SURVEILLANCE = "surveillance"
ZONE_TYPE_EXCLUSION = "exclusion"
ZONE_TYPE_PATROL = "patrol"
ZONE_TYPE_OBJECTIVE = "objective"
ZONE_TYPES = frozenset(
    {ZONE_TYPE_SURVEILLANCE, ZONE_TYPE_EXCLUSION, ZONE_TYPE_PATROL, ZONE_TYPE_OBJECTIVE}
)


@dataclass(frozen=True)
class MissionZone:
    zone_id: str
    zone_type: str
    center: Vector3
    radius_m: float
    priority: int = 1
    label: str = ""

    def __post_init__(self) -> None:
        if self.zone_type not in ZONE_TYPES:
            raise ValueError(
                f"zone_type must be one of {sorted(ZONE_TYPES)}, got {self.zone_type!r}"
            )
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive.")

    def contains_point(self, xy: Iterable[float]) -> bool:
        point_xy = np.asarray(tuple(xy), dtype=float)[:2]
        center_xy = np.asarray(self.center[:2], dtype=float)
        delta = point_xy - center_xy
        return float(np.dot(delta, delta)) <= self.radius_m * self.radius_m


# --- Scan / Map / Localize / Inspect mission types ---


@dataclass(frozen=True)
class MapFeature:
    """A detected feature in the world map (terrain peak, building edge, etc.)."""

    feature_id: str
    position: Vector3
    feature_type: str  # "terrain_peak" | "building_edge" | "elevation_keypoint"
    height_m: float = 0.0
    confidence: float = 1.0


@dataclass(frozen=True)
class InspectionPOI:
    """A Point of Interest to be inspected after the scanning phase."""

    poi_id: str
    name: str
    position: Vector3
    priority: int = 1  # higher = more urgent; used by POIManager for ordering
    required_dwell_s: float = 15.0
    sensor_modality: str = "optical"  # "optical" | "thermal" | "any"


@dataclass(frozen=True)
class POIStatus:
    """Runtime status of one InspectionPOI."""

    poi_id: str
    status: str  # "pending" | "active" | "complete"
    assigned_drone_id: str | None = None
    arrival_time_s: float | None = None
    completion_time_s: float | None = None
    dwell_accumulated_s: float = 0.0
    position: Vector3 | None = None


@dataclass(frozen=True)
class EgressDroneProgress:
    """Per-drone return-to-home progress emitted during the egress phase."""

    drone_id: str
    distance_to_home_m: float
    home_position: Vector3


@dataclass(frozen=True)
class SafetyEventRecord:
    """Replay-facing record of a safety-gate rejection.

    Emitted whenever the mission executor's safety gate rejects a proposed
    drone command. Mirrors mission.execution.SafetyEvent but lives in core
    types so the replay schema does not depend on the mission package.
    """

    timestamp_s: float
    drone_id: str
    task_type: str
    target_xy_m: tuple[float, float]
    target_z_m: float
    reason: str
    violations: tuple[str, ...] = ()


@dataclass(frozen=True)
class LocalizationEstimate:
    """Drone self-localization estimate derived from map matching."""

    drone_id: str
    timestamp_s: float
    position_estimate: Vector3
    heading_rad: float
    position_std_m: float  # 1-sigma position uncertainty
    confidence: float  # 0.0 (no match) -> 1.0 (perfect match)


@dataclass(frozen=True)
class ScanMissionState:
    """Top-level mission state serialised into every replay frame."""

    phase: str  # "scanning" | "localizing" | "inspecting" | "egress" | "complete"
    scan_coverage_fraction: float  # 0->1, fraction of map area covered
    scan_coverage_threshold: float  # target fraction to trigger phase transition
    localization_estimates: list[LocalizationEstimate]
    poi_statuses: list[POIStatus]
    completed_poi_count: int
    total_poi_count: int
    phase_started_at_s: float = 0.0  # sim timestamp when current phase began
    # Per-frame delta: newly covered cell centres as (x_m, y_m, terrain_height_m).
    # The viewer accumulates these into a persistent LiDAR-style point cloud as
    # the replay plays back, producing a real-time map reconstruction.
    newly_scanned_cells: tuple = ()  # Tuple[Tuple[float,float,float], ...]
    # Team-level flag: True if *any* drone advanced via timeout rather than convergence.
    # Does not identify which specific drone(s) timed out vs. genuinely converged.
    localization_timed_out: bool = False
    coordinator_drone_id: str | None = None  # elected by highest battery fraction (one-shot)
    egress_progress: tuple = ()  # Tuple[EgressDroneProgress, ...]; non-empty during egress
    # Safety-gate rejections recorded since the previous frame.
    # Empty in nominal flights; populated when the executor blocks an unsafe waypoint.
    safety_events: tuple = ()  # Tuple[SafetyEventRecord, ...]


@dataclass(frozen=True)
class TrackingMissionState:
    """Top-level mission state for target_tracking missions.

    Parallel to ``ScanMissionState`` but minimal: target_tracking does not
    have a phase machine, so today this carries the per-frame safety-event
    delta produced by the mission executor's dispatch loop.
    """

    safety_events: tuple = ()  # Tuple[SafetyEventRecord, ...]


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {
            field_info.name: to_jsonable(getattr(value, field_info.name))
            for field_info in fields(value)
        }
    if isinstance(value, np.ndarray):
        if value.ndim > 1:
            return value.flatten().tolist()
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): to_jsonable(nested) for key, nested in value.items()}
    if isinstance(value, (list, tuple)):
        # Flatten nested numeric lists (e.g. covariance matrices already
        # converted from numpy to list-of-lists before reaching to_jsonable).
        if (
            value
            and isinstance(value[0], (list, tuple))
            and value[0]
            and isinstance(value[0][0], (int, float, np.floating, np.integer))
        ):
            return [float(elem) for row in value for elem in row]
        return [to_jsonable(item) for item in value]
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
