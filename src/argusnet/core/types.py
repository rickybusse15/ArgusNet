from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional

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
    origin: Optional[Vector3] = None
    attempted_point: Optional[Vector3] = None
    closest_point: Optional[Vector3] = None
    blocker_type: str = ""
    first_hit_range_m: Optional[float] = None


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
    lifecycle_state: Optional[str] = None
    quality_score: Optional[float] = None


@dataclass(frozen=True)
class PlatformMetrics:
    mean_error_m: Optional[float]
    max_error_m: Optional[float]
    active_track_count: int
    observation_count: int
    accepted_observation_count: int
    rejected_observation_count: int
    mean_measurement_std_m: Optional[float]
    track_errors_m: Dict[str, float] = field(default_factory=dict)
    rejection_counts: Dict[str, int] = field(default_factory=dict)
    accepted_observations_by_target: Dict[str, int] = field(default_factory=dict)
    rejected_observations_by_target: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MappingState:
    coverage_fraction: float       # 0.0–1.0 fraction of grid cells observed
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
    event_type: str                # "entered" | "coverage_updated" | "exited"
    timestamp_s: float
    zone_coverage_fraction: float


@dataclass(frozen=True)
class PlatformFrame:
    timestamp_s: float
    nodes: List[NodeState]
    observations: List[BearingObservation]
    rejected_observations: List[ObservationRejection]
    tracks: List[TrackState]
    truths: List[TruthState]
    metrics: PlatformMetrics
    generation_rejections: List[ObservationRejection] = field(default_factory=list)
    mapping_state: Optional[MappingState] = None
    localization_state: Optional[LocalizationState] = None
    inspection_events: List[InspectionEvent] = field(default_factory=list)


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
    node_health: List[NodeHealthMetrics] = field(default_factory=list)
    mean_frame_rate_hz: float = 0.0
    mean_ingest_latency_s: float = 0.0
    active_node_count: int = 0
    stale_node_count: int = 0


ZONE_TYPE_SURVEILLANCE = "surveillance"
ZONE_TYPE_EXCLUSION = "exclusion"
ZONE_TYPE_PATROL = "patrol"
ZONE_TYPE_OBJECTIVE = "objective"
ZONE_TYPES = frozenset({ZONE_TYPE_SURVEILLANCE, ZONE_TYPE_EXCLUSION, ZONE_TYPE_PATROL, ZONE_TYPE_OBJECTIVE})


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
            raise ValueError(f"zone_type must be one of {sorted(ZONE_TYPES)}, got {self.zone_type!r}")
        if self.radius_m <= 0:
            raise ValueError("radius_m must be positive.")


def to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {field_info.name: to_jsonable(getattr(value, field_info.name)) for field_info in fields(value)}
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
