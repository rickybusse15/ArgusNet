from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .behaviors import BEHAVIOR_PRESETS, build_target_trajectory as build_behavior_trajectory
from .environment import (
    Bounds2D,
    BuildingPrism,
    CylinderObstacle,
    EnvironmentCRS,
    EnvironmentModel,
    ForestStand,
    LandCoverClass,
    LandCoverLayer,
    ObstacleLayer,
    OrientedBox,
    SensorVisibilityModel,
    WallSegment,
)
from .models import (
    BearingObservation,
    LaunchEvent,
    MissionZone,
    NodeState,
    ObservationRejection,
    PlatformFrame,
    TruthState,
    ZONE_TYPE_EXCLUSION,
    ZONE_TYPE_OBJECTIVE,
    ZONE_TYPE_PATROL,
    ZONE_TYPE_SURVEILLANCE,
    vec3,
)
from .config import (
    DEFAULT_MAP_PRESET_SCALES,
    DEFAULT_PLATFORM_PRESETS,
    DynamicsConfig,
    GroundStationLayoutConfig,
    PlatformPresetProfile,
    SensorConfig,
    SimulationConstants,
    TargetTrajectoryConfig,
)
from .planning import PathPlanner2D, PlannerConfig
from .replay import ReplayDocument, build_replay_document, write_replay_document
from .sensor_models import SensorErrorConfig, SensorModel, detection_probability as sensor_detection_probability
from .sensor_models import atmospheric_attenuation as sensor_atmospheric_attenuation
from .service import TrackerConfig, TrackingService
from .terrain import KNOWN_TERRAIN_PRESETS, OccludingObject, TerrainModel, terrain_model_from_preset, xy_bounds
from .weather import KNOWN_WEATHER_PRESETS, WeatherModel, weather_from_preset

# Default constants instance — used when no explicit config is supplied.
_DEFAULT_CONSTANTS = SimulationConstants.default()

TrajectoryFn = Callable[[float], Tuple[np.ndarray, np.ndarray]]

REJECT_OUT_OF_RANGE = "out_of_range"
REJECT_DROPOUT = "dropout"
REJECT_LOW_ELEVATION = "low_elevation"
REJECT_TERRAIN_OCCLUSION = "terrain_occlusion"
REJECT_BUILDING_OCCLUSION = "building_occlusion"
REJECT_WALL_OCCLUSION = "wall_occlusion"
REJECT_VEGETATION_OCCLUSION = "vegetation_occlusion"
REJECT_OUT_OF_COVERAGE = "out_of_coverage"
REJECT_OUTSIDE_FOV = "outside_fov"
REJECT_OBJECT_OCCLUSION = REJECT_BUILDING_OCCLUSION

# Backward-compatible module-level aliases — these read from the default
# SimulationConstants so existing imports keep working.
MAP_PRESET_SCALES = dict(DEFAULT_MAP_PRESET_SCALES)
TARGET_MOTION_PRESETS = set(BEHAVIOR_PRESETS)
DRONE_MODE_PRESETS = {"follow", "search", "mixed"}
TERRAIN_PRESET_CHOICES = frozenset({"default", *KNOWN_TERRAIN_PRESETS})
PLATFORM_PRESET_CHOICES = frozenset({"baseline", "wide_area"})

DEFAULT_SIM_DURATION_S = _DEFAULT_CONSTANTS.dynamics.default_duration_s
DEFAULT_SIM_DT_S = _DEFAULT_CONSTANTS.dynamics.default_dt_s
DEFAULT_SIM_STEPS = int(math.ceil(DEFAULT_SIM_DURATION_S / DEFAULT_SIM_DT_S)) + 1
AERIAL_TARGET_MIN_AGL_M = _DEFAULT_CONSTANTS.dynamics.aerial_target_min_agl_m
INTERCEPTOR_SEARCH_MIN_AGL_M = _DEFAULT_CONSTANTS.dynamics.interceptor_search_min_agl_m
INTERCEPTOR_FOLLOW_MIN_AGL_M = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_min_agl_m
INTERCEPTOR_FOLLOW_RADIUS_M = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_radius_m
INTERCEPTOR_FOLLOW_ALTITUDE_OFFSET_M = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_altitude_offset_m
INTERCEPTOR_FOLLOW_LEAD_S = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_lead_s
INTERCEPTOR_FOLLOW_CANDIDATE_COUNT = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_candidate_count
INTERCEPTOR_FOLLOW_ROTATION_RATE_RAD_S = _DEFAULT_CONSTANTS.dynamics.interceptor_follow_rotation_rate_rad_s
GROUND_CONTACT_TOP_PAD_M = _DEFAULT_CONSTANTS.dynamics.ground_contact_top_pad_m
METRICS_CSV_FIELDS = [
    "time_s",
    "track_id",
    "true_x_m",
    "true_y_m",
    "true_z_m",
    "track_x_m",
    "track_y_m",
    "track_z_m",
    "error_m",
    "measurement_std_m",
    "stale_steps",
    "observations",
    "sim_rejected_observations",
]


# PlatformPresetProfile and PLATFORM_PRESETS have been moved to config.py.
# The names are re-exported via the import above for backward compatibility.
PLATFORM_PRESETS: Mapping[str, PlatformPresetProfile] = DEFAULT_PLATFORM_PRESETS


@dataclass(frozen=True)
class SimNode:
    node_id: str
    is_mobile: bool
    bearing_std_rad: float
    dropout_probability: float
    max_range_m: float
    trajectory: TrajectoryFn
    sensor_type: str = "optical"
    fov_half_angle_deg: float = 180.0
    sensor_direction_fn: Optional[Callable[[float], np.ndarray]] = None

    def state(self, timestamp_s: float) -> NodeState:
        position, velocity = self.trajectory(timestamp_s)
        return NodeState(
            node_id=self.node_id,
            position=position,
            velocity=velocity,
            is_mobile=self.is_mobile,
            timestamp_s=timestamp_s,
            health=1.0 - self.dropout_probability * 0.5,
            sensor_type=self.sensor_type,
            fov_half_angle_deg=self.fov_half_angle_deg,
            max_range_m=self.max_range_m,
        )


@dataclass(frozen=True)
class SimTarget:
    target_id: str
    trajectory: TrajectoryFn

    def truth(self, timestamp_s: float) -> TruthState:
        position, velocity = self.trajectory(timestamp_s)
        return TruthState(
            target_id=self.target_id,
            position=position,
            velocity=velocity,
            timestamp_s=timestamp_s,
        )


@dataclass(frozen=True)
class ObservationBatch:
    observations: List[BearingObservation]
    attempted_count: int
    rejection_counts: Dict[str, int]
    accepted_by_target: Dict[str, int]
    rejected_by_target: Dict[str, int]
    accepted_by_node_target: Dict[Tuple[str, str], int]
    generation_rejections: List["ObservationRejection"] = field(default_factory=list)


@dataclass(frozen=True)
class ScenarioOptions:
    map_preset: str = "regional"
    target_motion_preset: str = "mixed"
    drone_mode_preset: str = "mixed"
    terrain_preset: str = "alpine"
    weather_preset: str = "clear"
    clean_terrain: bool = False
    platform_preset: str = "baseline"
    ground_station_count: int = 7
    target_count: int = 2
    drone_count: int = 2

    def __post_init__(self) -> None:
        if self.map_preset not in MAP_PRESET_SCALES:
            raise ValueError(f"map_preset must be one of {sorted(MAP_PRESET_SCALES)}.")
        if self.target_motion_preset not in TARGET_MOTION_PRESETS:
            raise ValueError(f"target_motion_preset must be one of {sorted(TARGET_MOTION_PRESETS)}.")
        if self.drone_mode_preset not in DRONE_MODE_PRESETS:
            raise ValueError(f"drone_mode_preset must be one of {sorted(DRONE_MODE_PRESETS)}.")
        if self.terrain_preset not in TERRAIN_PRESET_CHOICES:
            raise ValueError(f"terrain_preset must be one of {sorted(TERRAIN_PRESET_CHOICES)}.")
        if self.weather_preset not in KNOWN_WEATHER_PRESETS:
            raise ValueError(f"weather_preset must be one of {sorted(KNOWN_WEATHER_PRESETS)}.")
        if self.platform_preset not in PLATFORM_PRESET_CHOICES:
            raise ValueError(f"platform_preset must be one of {sorted(PLATFORM_PRESET_CHOICES)}.")
        if self.ground_station_count <= 0:
            raise ValueError("ground_station_count must be greater than 0.")
        if self.target_count <= 0:
            raise ValueError("target_count must be greater than 0.")
        if self.drone_count <= 0:
            raise ValueError("drone_count must be greater than 0.")


@dataclass(frozen=True)
class ScenarioDefinition:
    scenario_name: str
    nodes: Tuple[SimNode, ...]
    targets: Tuple[SimTarget, ...]
    terrain: Optional[object] = None
    occluding_objects: Tuple[object, ...] = ()
    environment: Optional[EnvironmentModel] = None
    weather: Optional[WeatherModel] = None
    constants: SimulationConstants = _DEFAULT_CONSTANTS
    options: ScenarioOptions = field(default_factory=ScenarioOptions)
    map_bounds_m: Mapping[str, float] = field(default_factory=dict)
    target_motion_assignments: Mapping[str, str] = field(default_factory=dict)
    drone_planner_modes: Mapping[str, str] = field(default_factory=dict)
    drone_target_assignments: Mapping[str, str] = field(default_factory=dict)
    drone_roles: Mapping[str, str] = field(default_factory=dict)
    adaptive_drone_controllers: Mapping[str, object] = field(default_factory=dict)
    launchable_controllers: Mapping[str, object] = field(default_factory=dict)
    mission_zones: Tuple[MissionZone, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "nodes", tuple(self.nodes))
        object.__setattr__(self, "targets", tuple(self.targets))
        object.__setattr__(self, "occluding_objects", tuple(self.occluding_objects))
        object.__setattr__(
            self,
            "constants",
            self.constants if isinstance(self.constants, SimulationConstants) else _DEFAULT_CONSTANTS,
        )
        object.__setattr__(self, "options", self.options if isinstance(self.options, ScenarioOptions) else ScenarioOptions())
        object.__setattr__(self, "map_bounds_m", MappingProxyType(dict(self.map_bounds_m)))
        object.__setattr__(self, "target_motion_assignments", MappingProxyType(dict(self.target_motion_assignments)))
        object.__setattr__(self, "drone_planner_modes", MappingProxyType(dict(self.drone_planner_modes)))
        object.__setattr__(self, "drone_target_assignments", MappingProxyType(dict(self.drone_target_assignments)))
        object.__setattr__(self, "drone_roles", MappingProxyType(dict(self.drone_roles)))
        object.__setattr__(self, "adaptive_drone_controllers", MappingProxyType(dict(self.adaptive_drone_controllers)))
        object.__setattr__(self, "launchable_controllers", MappingProxyType(dict(self.launchable_controllers)))
        object.__setattr__(self, "mission_zones", tuple(self.mission_zones))
        if not self.scenario_name.strip():
            raise ValueError("scenario_name must be non-empty.")
        if not self.nodes:
            raise ValueError("ScenarioDefinition requires at least one node.")
        if not self.targets:
            raise ValueError("ScenarioDefinition requires at least one target.")
        if self.map_bounds_m:
            required_keys = ("x_min_m", "x_max_m", "y_min_m", "y_max_m")
            for key in required_keys:
                value = self.map_bounds_m.get(key)
                if not isinstance(value, (int, float)) or not np.isfinite(value):
                    raise ValueError(f"map_bounds_m must contain a finite {key}.")
        active_environment = self.environment
        if active_environment is None:
            active_terrain = self.terrain
            if active_terrain is None:
                raise ValueError("ScenarioDefinition requires either environment or terrain.")
            active_bounds = (
                Bounds2D.from_mapping(self.map_bounds_m)
                if self.map_bounds_m
                else Bounds2D.from_mapping(
                    xy_bounds(
                        (
                            point
                            for node in self.nodes
                            for point in [node.state(0.0).position]
                        ),
                        padding_m=150.0,
                    )
                )
            )
            active_environment = EnvironmentModel.from_legacy(
                environment_id=self.scenario_name.strip() or "scenario",
                bounds_xy_m=active_bounds,
                terrain_model=active_terrain,
                occluding_objects=self.occluding_objects,
            )
        object.__setattr__(self, "environment", active_environment)
        object.__setattr__(self, "terrain", active_environment.terrain)
        if not self.occluding_objects:
            object.__setattr__(self, "occluding_objects", tuple(active_environment.obstacles.primitives))

    def node_states(self, timestamp_s: float) -> List[NodeState]:
        return [node.state(timestamp_s) for node in self.nodes]

    def truths(self, timestamp_s: float) -> List[TruthState]:
        return [target.truth(timestamp_s) for target in self.targets]

    def reset_runtime_state(self) -> None:
        for node in self.nodes:
            trajectory = getattr(node, "trajectory", None)
            if trajectory is not None and hasattr(trajectory, "reset_state"):
                trajectory.reset_state()
        for controller in self.adaptive_drone_controllers.values():
            if hasattr(controller, "reset_state"):
                controller.reset_state()


@dataclass(frozen=True)
class SimulationConfig:
    steps: int = DEFAULT_SIM_STEPS
    dt_s: float = DEFAULT_SIM_DT_S
    seed: int = 7
    requested_duration_s: Optional[float] = None

    def __post_init__(self) -> None:
        if self.steps <= 0:
            raise ValueError("steps must be greater than 0.")
        if not np.isfinite(self.dt_s) or self.dt_s <= 0.0:
            raise ValueError("dt_s must be finite and greater than 0.")
        if not isinstance(self.seed, int):
            raise ValueError("seed must be an integer.")
        if self.requested_duration_s is not None:
            if not np.isfinite(self.requested_duration_s) or self.requested_duration_s <= 0.0:
                raise ValueError("requested_duration_s must be finite and greater than 0.")

    @classmethod
    def from_duration(
        cls,
        duration_s: float,
        dt_s: float = DEFAULT_SIM_DT_S,
        seed: int = 7,
    ) -> "SimulationConfig":
        if not np.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and greater than 0.")
        if not np.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and greater than 0.")
        steps = int(math.ceil(duration_s / dt_s)) + 1
        return cls(steps=steps, dt_s=dt_s, seed=seed, requested_duration_s=float(duration_s))

    @property
    def actual_duration_s(self) -> float:
        return max(0.0, (self.steps - 1) * self.dt_s)


@dataclass
class SimulationResult:
    scenario_name: str
    simulation_config: SimulationConfig
    tracker_config: TrackerConfig
    frames: List[PlatformFrame]
    metrics_rows: List[dict]
    summary: Dict[str, object]
    replay_metadata: Dict[str, object] = field(default_factory=dict)


@dataclass
class FollowPathController:
    target_trajectory: TrajectoryFn
    environment: EnvironmentModel
    map_bounds_m: Mapping[str, float]
    base_agl_m: float
    standoff_radius_m: float
    map_scale: float
    max_speed_mps: float
    phase: float
    lead_s: float = 4.0
    candidate_count: int = 12
    rotation_rate_rad_s: float = 0.08
    planner: Optional[PathPlanner2D] = None
    min_agl_m: float = 18.0
    vertical_amplitude_m: float = 0.0
    vertical_frequency_rad_s: float = 0.0
    target_altitude_offset_m: Optional[float] = None
    max_accel_mps2: Optional[float] = None
    terrain_following: bool = False
    terrain_following_agl_m: float = 30.0
    terrain_following_smoothing_s: float = 1.5
    # Cooperative orbit parameters — multiple drones on same target spread angularly
    # so they form an optimal triangulation / intercept geometry.
    # slot_index selects which angular segment this drone occupies;
    # slot_count is the total number of drones sharing this orbit.
    drone_role: str = "interceptor"   # "tracker" | "interceptor"
    slot_index: int = 0               # which angular slot (0-based)
    slot_count: int = 1               # total slots / drones on this target
    _state: Dict[str, object] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        self._state = {
            "cached_timestamp": None,
            "cached_result": None,
            "last_timestamp": None,
            "last_xy": None,
            "last_direction_xy": None,
            "route_points": None,
            "route_goal_xy": None,
            "last_velocity_xy": None,
            "smoothed_terrain_z": None,
        }

    def seed(self, *, position: np.ndarray, velocity: np.ndarray, timestamp_s: float) -> None:
        self._state["cached_timestamp"] = None
        self._state["cached_result"] = None
        self._state["last_timestamp"] = float(timestamp_s)
        self._state["last_xy"] = np.asarray(position[:2], dtype=float).copy()
        self._state["last_direction_xy"] = np.asarray(velocity[:2], dtype=float).copy()
        self._state["route_points"] = None
        self._state["route_goal_xy"] = None

    def reset_state(self) -> None:
        self._state["cached_timestamp"] = None
        self._state["cached_result"] = None
        self._state["last_timestamp"] = None
        self._state["last_xy"] = None
        self._state["last_direction_xy"] = None
        self._state["route_points"] = None
        self._state["route_goal_xy"] = None

    def __call__(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        terrain = self.environment.terrain
        sensor_profile = SensorVisibilityModel.optical_default()
        state = self._state
        cached_timestamp = state["cached_timestamp"]
        if isinstance(cached_timestamp, (int, float)) and abs(timestamp_s - cached_timestamp) <= 1e-9:
            cached_position, cached_velocity = state["cached_result"]
            return cached_position.copy(), cached_velocity.copy()

        last_timestamp = state["last_timestamp"]
        if isinstance(last_timestamp, (int, float)) and timestamp_s < last_timestamp - 1e-9:
            state["last_timestamp"] = None
            state["last_xy"] = None
            state["last_direction_xy"] = None
            state["route_points"] = None
            state["route_goal_xy"] = None

        lead_position, lead_velocity = self.target_trajectory(timestamp_s + self.lead_s)
        # Slot offset spreads N cooperative drones evenly around the orbit circle.
        # Trackers: maximise triangulation baseline; interceptors: pincer approach.
        slot_offset = (self.slot_index * math.tau) / max(self.slot_count, 1)
        base_angle = self.phase + slot_offset + self.rotation_rate_rad_s * timestamp_s
        first_in_bounds_xy: Optional[np.ndarray] = None
        route_goal_xy = state["route_goal_xy"]
        current_xy = state["last_xy"]
        if not isinstance(current_xy, np.ndarray):
            current_xy = clamp_to_bounds(lead_position[:2], self.map_bounds_m, margin_m=15.0 * self.map_scale)
            if self.planner is not None:
                current_xy = self.planner.nearest_free_point(current_xy, self.planner.config.drone_clearance_m)
        last_direction_xy = state["last_direction_xy"]
        if isinstance(last_direction_xy, np.ndarray):
            current_heading_xy = last_direction_xy
        else:
            current_heading_xy = np.asarray(lead_velocity[:2], dtype=float)
        best_option: Optional[Tuple[int, float, float, np.ndarray, Optional[np.ndarray]]] = None

        for index in range(self.candidate_count):
            angle = base_angle + index * (math.tau / self.candidate_count)
            candidate_xy = lead_position[:2] + self.standoff_radius_m * np.array(
                [math.cos(angle), math.sin(angle)],
                dtype=float,
            )
            if not within_bounds(candidate_xy, self.map_bounds_m, margin_m=15.0 * self.map_scale):
                continue
            vertical_phase = self.vertical_frequency_rad_s * timestamp_s + self.phase * 0.5
            agl_m = self.base_agl_m + self.vertical_amplitude_m * math.sin(vertical_phase)
            altitude_offset_m = (
                None
                if self.target_altitude_offset_m is None
                else self.target_altitude_offset_m + self.vertical_amplitude_m * math.sin(vertical_phase)
            )
            candidate_position, _ = compose_air_state(
                xy_m=candidate_xy,
                xy_velocity_mps=[0.0, 0.0],
                terrain=terrain,
                min_agl_m=self.min_agl_m,
                nominal_agl_m=agl_m,
                agl_rate_mps=0.0,
                target_position=lead_position,
                target_velocity=lead_velocity,
                target_altitude_offset_m=altitude_offset_m,
                target_altitude_rate_offset_mps=(
                    None
                    if self.target_altitude_offset_m is None
                    else self.vertical_amplitude_m * self.vertical_frequency_rad_s * math.cos(vertical_phase)
                ),
            )
            if self.environment.obstacles.point_collides(
                float(candidate_position[0]),
                float(candidate_position[1]),
                float(candidate_position[2]),
            ):
                continue
            candidate_route_points: Optional[np.ndarray] = None
            candidate_route_cost = float(np.linalg.norm(candidate_xy - current_xy))
            if self.planner is not None:
                candidate_route = self.planner.plan_route(
                    current_xy,
                    candidate_xy,
                    clearance_m=self.planner.config.drone_clearance_m,
                )
                if candidate_route is None:
                    continue
                candidate_route_points = candidate_route.points_xy_m
                candidate_route_cost = candidate_route.length_m
            if first_in_bounds_xy is None:
                first_in_bounds_xy = candidate_xy
            visibility = self.environment.query.los(
                candidate_position,
                lead_position,
                sensor_profile=sensor_profile,
            )
            heading_change = 0.0
            heading_norm = float(np.linalg.norm(current_heading_xy))
            route_direction_xy = (
                candidate_route_points[1] - candidate_route_points[0]
                if isinstance(candidate_route_points, np.ndarray) and len(candidate_route_points) > 1
                else (candidate_xy - current_xy)
            )
            route_norm = float(np.linalg.norm(route_direction_xy))
            if heading_norm > 1.0e-6 and route_norm > 1.0e-6:
                cosine = float(
                    np.clip(
                        np.dot(current_heading_xy, route_direction_xy) / (heading_norm * route_norm),
                        -1.0,
                        1.0,
                    )
                )
                heading_change = math.acos(cosine)
            candidate_score = (
                0 if visibility.visible else 1,
                candidate_route_cost,
                heading_change,
                candidate_xy,
                candidate_route_points,
            )
            if best_option is None or candidate_score[:3] < best_option[:3]:
                best_option = candidate_score

        planned_route_points = state["route_points"]
        replan_required = not isinstance(planned_route_points, np.ndarray) or len(planned_route_points) < 2
        if isinstance(route_goal_xy, np.ndarray) and best_option is not None:
            replan_threshold_m = max(30.0, self.standoff_radius_m * 0.5)
            replan_required = replan_required or (float(np.linalg.norm(best_option[3] - route_goal_xy)) >= replan_threshold_m)
        elif best_option is not None:
            replan_required = True

        if self.planner is not None and isinstance(route_goal_xy, np.ndarray):
            replan_required = replan_required or (
                self.planner.plan_route(
                    current_xy,
                    route_goal_xy,
                    clearance_m=self.planner.config.drone_clearance_m,
                )
                is None
            )

        if replan_required:
            if best_option is not None:
                route_goal_xy = np.asarray(best_option[3], dtype=float)
                if isinstance(best_option[4], np.ndarray):
                    planned_route_points = best_option[4]
                else:
                    planned_route_points = np.vstack([current_xy, route_goal_xy])
            elif first_in_bounds_xy is not None:
                planned_route_points = np.vstack([current_xy, first_in_bounds_xy])
                route_goal_xy = first_in_bounds_xy
            else:
                planned_route_points = np.vstack([current_xy, current_xy])
                route_goal_xy = current_xy
        elif isinstance(planned_route_points, np.ndarray):
            planned_route_points = planned_route_points.copy()

        last_xy = state["last_xy"]
        last_timestamp = state["last_timestamp"]
        delta_t_s = 0.0 if not isinstance(last_timestamp, (int, float)) else max(timestamp_s - last_timestamp, 0.0)
        max_speed_mps = self.max_speed_mps
        travel_distance_m = max_speed_mps * delta_t_s
        active_route_points = (
            planned_route_points if isinstance(planned_route_points, np.ndarray) else np.vstack([current_xy, current_xy])
        )
        if delta_t_s <= 1.0e-9:
            if not isinstance(last_xy, np.ndarray) and len(active_route_points) > 1:
                chosen_xy = np.asarray(active_route_points[1], dtype=float)
                route_direction_xy = chosen_xy - np.asarray(active_route_points[0], dtype=float)
                active_route_points = np.vstack([chosen_xy, chosen_xy])
                route_goal_xy = chosen_xy
            else:
                chosen_xy = active_route_points[0]
                route_direction_xy = current_heading_xy
        else:
            active_route_points, chosen_xy, route_direction_xy = advance_along_polyline(active_route_points, travel_distance_m)

        if not isinstance(last_xy, np.ndarray) or delta_t_s <= 1.0e-9:
            direction_xy = np.asarray(route_direction_xy, dtype=float)
            direction_norm = float(np.linalg.norm(direction_xy))
            lead_speed_mps = min(max_speed_mps, float(np.linalg.norm(lead_velocity[:2])))
            xy_velocity = (
                np.zeros(2, dtype=float)
                if direction_norm <= 1.0e-6 or lead_speed_mps <= 1.0e-6
                else (direction_xy / direction_norm) * lead_speed_mps
            )
        else:
            xy_velocity = (chosen_xy - last_xy) / max(delta_t_s, 1e-6)
        speed_mps = float(np.linalg.norm(xy_velocity))
        if speed_mps > max_speed_mps and speed_mps > 1.0e-6:
            xy_velocity = xy_velocity / speed_mps * max_speed_mps

        # Acceleration limiting — limit speed (scalar) changes only so that
        # direction changes (needed for orbit tracking) are not penalised.
        if self.max_accel_mps2 is not None:
            last_vel = state.get("last_velocity_xy")
            if isinstance(last_vel, np.ndarray):
                last_speed = float(np.linalg.norm(last_vel))
                desired_speed = float(np.linalg.norm(xy_velocity))
                max_speed_change = self.max_accel_mps2 * max(delta_t_s, 1e-6)
                speed_diff = desired_speed - last_speed
                if abs(speed_diff) > max_speed_change:
                    clamped_speed = last_speed + math.copysign(max_speed_change, speed_diff)
                    clamped_speed = max(clamped_speed, 0.0)
                    if desired_speed > 1e-6:
                        xy_velocity = xy_velocity * (clamped_speed / desired_speed)
            state["last_velocity_xy"] = xy_velocity.copy()

        # Smooth orbit blend — when the drone is close to the standoff radius,
        # blend the path velocity toward the circle tangent so orbits appear as
        # smooth arcs rather than straight-segment polygons.
        if self.standoff_radius_m > 1.0 and isinstance(last_xy, np.ndarray):
            vec_from_target = last_xy - lead_position[:2]
            dist_from_target = float(np.linalg.norm(vec_from_target))
            if dist_from_target > 1e-3:
                orbit_error = abs(dist_from_target - self.standoff_radius_m)
                orbit_blend = max(0.0, 1.0 - orbit_error / max(self.standoff_radius_m * 0.35, 1.0))
                if orbit_blend > 0.05:
                    # Tangent direction: rotate radius vector 90° in the direction
                    # of rotation_rate_rad_s (positive = counter-clockwise).
                    sign = math.copysign(1.0, self.rotation_rate_rad_s) if abs(self.rotation_rate_rad_s) > 1e-9 else 1.0
                    tangent_xy = sign * np.array([-vec_from_target[1], vec_from_target[0]], dtype=float)
                    tangent_norm = float(np.linalg.norm(tangent_xy))
                    if tangent_norm > 1e-6:
                        cur_spd = float(np.linalg.norm(xy_velocity))
                        if cur_spd < 1e-6:
                            cur_spd = max_speed_mps
                        tangent_vel = (tangent_xy / tangent_norm) * cur_spd
                        xy_velocity = (1.0 - orbit_blend) * xy_velocity + orbit_blend * tangent_vel
                        blended_speed = float(np.linalg.norm(xy_velocity))
                        if blended_speed > max_speed_mps + 1e-6:
                            xy_velocity = xy_velocity / blended_speed * max_speed_mps

        # Altitude computation — terrain-following or standard oscillation.
        if self.terrain_following:
            raw_terrain_z = terrain.height_at(float(chosen_xy[0]), float(chosen_xy[1]))
            prev_smoothed = state.get("smoothed_terrain_z")
            if prev_smoothed is None or delta_t_s <= 1e-9:
                smoothed_z = raw_terrain_z
            else:
                alpha = 1.0 - math.exp(-delta_t_s / max(self.terrain_following_smoothing_s, 0.01))
                smoothed_z = prev_smoothed + alpha * (raw_terrain_z - prev_smoothed)
            state["smoothed_terrain_z"] = smoothed_z
            agl_m = self.terrain_following_agl_m
        else:
            agl_m = self.base_agl_m + self.vertical_amplitude_m * math.sin(
                self.vertical_frequency_rad_s * timestamp_s + self.phase * 0.5
            )

        vertical_phase = self.vertical_frequency_rad_s * timestamp_s + self.phase * 0.5
        vertical_rate_mps = self.vertical_amplitude_m * self.vertical_frequency_rad_s * math.cos(vertical_phase)
        position, velocity = compose_air_state(
            xy_m=chosen_xy,
            xy_velocity_mps=xy_velocity,
            terrain=terrain,
            min_agl_m=self.min_agl_m,
            nominal_agl_m=agl_m,
            agl_rate_mps=vertical_rate_mps,
            target_position=lead_position,
            target_velocity=lead_velocity,
            target_altitude_offset_m=(
                None
                if self.target_altitude_offset_m is None
                else self.target_altitude_offset_m + self.vertical_amplitude_m * math.sin(vertical_phase)
            ),
            target_altitude_rate_offset_mps=(
                None if self.target_altitude_offset_m is None else vertical_rate_mps
            ),
        )
        position, velocity = collision_aware_position(
            position,
            velocity,
            terrain=terrain,
            min_agl_m=self.min_agl_m,
            obstacle_layer=self.environment.obstacles,
        )
        state["last_timestamp"] = float(timestamp_s)
        state["last_xy"] = position[:2].copy()
        state["last_direction_xy"] = np.asarray(xy_velocity[:2], dtype=float).copy()
        state["route_points"] = active_route_points
        state["route_goal_xy"] = None if route_goal_xy is None else np.asarray(route_goal_xy, dtype=float).copy()
        state["cached_timestamp"] = float(timestamp_s)
        state["cached_result"] = (position.copy(), velocity.copy())
        return position, velocity


@dataclass
class ObservationTriggeredFollowController:
    node_id: str
    search_trajectory: TrajectoryFn
    follow_trajectory: FollowPathController
    preferred_target_id: str
    engaged: bool = field(default=False, init=False)
    _last_timestamp_s: Optional[float] = field(default=None, init=False, repr=False)
    _last_result: Optional[Tuple[np.ndarray, np.ndarray]] = field(default=None, init=False, repr=False)

    def __call__(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        active_trajectory = self.follow_trajectory if self.engaged else self.search_trajectory
        position, velocity = active_trajectory(timestamp_s)
        self._last_timestamp_s = float(timestamp_s)
        self._last_result = (position.copy(), velocity.copy())
        return position, velocity

    def update_from_frame(self, frame: PlatformFrame, observation_batch: ObservationBatch) -> None:
        if self.engaged:
            return
        has_observation = observation_batch.accepted_by_node_target.get((self.node_id, self.preferred_target_id), 0) > 0
        if not has_observation:
            return
        if self._last_result is not None and self._last_timestamp_s is not None:
            last_position, last_velocity = self._last_result
            self.follow_trajectory.seed(
                position=last_position,
                velocity=last_velocity,
                timestamp_s=self._last_timestamp_s,
            )
        self.engaged = True

    def reset_state(self) -> None:
        self.engaged = False
        self._last_timestamp_s = None
        self._last_result = None
        if hasattr(self.follow_trajectory, "reset_state"):
            self.follow_trajectory.reset_state()


@dataclass
class LaunchableTrajectoryController:
    """Wraps an operational trajectory with a ground-start and climb phase.

    Before ``trigger_launch`` is called the drone sits at *station_position*.
    After launch it climbs to *operational_altitude_agl* over *climb_duration_s*
    using smooth-step interpolation, then delegates to *operational_trajectory*.
    """

    station_position: np.ndarray
    operational_trajectory: TrajectoryFn
    climb_duration_s: float = 8.0
    operational_altitude_agl: float = 230.0
    weather: Optional[WeatherModel] = None
    terrain: Optional[object] = None
    map_bounds_m: Optional[Mapping[str, float]] = None
    min_operational_agl_m: float = 18.0
    launched: bool = field(default=False, init=False)
    launch_time_s: Optional[float] = field(default=None, init=False)
    launch_target_id: Optional[str] = field(default=None, init=False)
    assigned_station_id: str = ""

    def __call__(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        if not self.launched or self.launch_time_s is None:
            return self.station_position.copy(), np.zeros(3, dtype=float)

        elapsed = timestamp_s - self.launch_time_s
        if elapsed < 0:
            return self.station_position.copy(), np.zeros(3, dtype=float)

        if elapsed < self.climb_duration_s:
            fraction = elapsed / self.climb_duration_s
            # Smooth-step easing for natural climb profile
            smooth = 3.0 * fraction ** 2 - 2.0 * fraction ** 3
            target_z = self.station_position[2] + self.operational_altitude_agl * smooth
            climb_rate = self.operational_altitude_agl * (6.0 * fraction - 6.0 * fraction ** 2) / self.climb_duration_s
            position = np.array(
                [self.station_position[0], self.station_position[1], target_z],
                dtype=float,
            )
            velocity = np.array([0.0, 0.0, climb_rate], dtype=float)
            return position, velocity

        # Operational phase — delegate to the wrapped trajectory and then apply
        # weather-induced drift and speed penalties.
        position, velocity = self.operational_trajectory(timestamp_s)
        return _apply_weather_adjustments(
            position,
            velocity,
            timestamp_s=timestamp_s,
            weather=self.weather,
            terrain=self.terrain,
            map_bounds_m=self.map_bounds_m,
            min_agl_m=self.min_operational_agl_m,
        )

    def trigger_launch(self, timestamp_s: float, target_id: str) -> None:
        if not self.launched:
            self.launched = True
            self.launch_time_s = float(timestamp_s)
            self.launch_target_id = target_id

    def reset_state(self) -> None:
        self.launched = False
        self.launch_time_s = None
        self.launch_target_id = None
        if hasattr(self.operational_trajectory, "reset_state"):
            self.operational_trajectory.reset_state()


DEFAULT_SIMULATION_TRACKER_CONFIG = TrackerConfig(
    max_stale_steps=_DEFAULT_CONSTANTS.dynamics.default_max_stale_steps,
)


def compose_terrain_state(
    xy_m: Sequence[float],
    xy_velocity_mps: Sequence[float],
    agl_m: float,
    agl_rate_mps: float,
    terrain: object,
    min_agl_m: float,
    terrain_speed_adaptation: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(xy_m, dtype=float)
    xy_velocity = np.asarray(xy_velocity_mps, dtype=float)
    ground_m = terrain.height_at(float(xy[0]), float(xy[1]))
    ground_gradient = terrain.gradient_at(float(xy[0]), float(xy[1]))
    if terrain_speed_adaptation:
        gradient_mag = float(np.linalg.norm(ground_gradient))
        steepness_factor = max(0.7, 1.0 - min(gradient_mag, 1.0) * 0.3)
        xy_velocity = xy_velocity * steepness_factor
    nominal_z_m = ground_m + agl_m
    z_m = terrain.clamp_altitude(xy, nominal_z_m, min_agl_m=min_agl_m)
    vz_mps = float(np.dot(ground_gradient, xy_velocity) + agl_rate_mps)
    if z_m > nominal_z_m and vz_mps < 0.0:
        vz_mps = 0.0
    position = np.array([xy[0], xy[1], z_m], dtype=float)
    velocity = np.array([xy_velocity[0], xy_velocity[1], vz_mps], dtype=float)
    return position, velocity


def compose_air_state(
    *,
    xy_m: Sequence[float],
    xy_velocity_mps: Sequence[float],
    terrain: object,
    min_agl_m: float,
    nominal_agl_m: float,
    agl_rate_mps: float,
    target_position: Optional[np.ndarray] = None,
    target_velocity: Optional[np.ndarray] = None,
    target_altitude_offset_m: Optional[float] = None,
    target_altitude_rate_offset_mps: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if target_position is None or target_altitude_offset_m is None:
        return compose_terrain_state(
            xy_m=xy_m,
            xy_velocity_mps=xy_velocity_mps,
            agl_m=nominal_agl_m,
            agl_rate_mps=agl_rate_mps,
            terrain=terrain,
            min_agl_m=min_agl_m,
        )

    xy = np.asarray(xy_m, dtype=float)
    xy_velocity = np.asarray(xy_velocity_mps, dtype=float)
    ground_m = terrain.height_at(float(xy[0]), float(xy[1]))
    target = np.asarray(target_position, dtype=float)
    target_velocity_value = np.zeros(3, dtype=float) if target_velocity is None else np.asarray(target_velocity, dtype=float)
    nominal_z_m = max(
        float(ground_m + min_agl_m),
        float(target[2] + target_altitude_offset_m),
    )
    position = np.array([xy[0], xy[1], nominal_z_m], dtype=float)
    velocity = np.array(
        [
            xy_velocity[0],
            xy_velocity[1],
            float(target_velocity_value[2] + (target_altitude_rate_offset_mps or 0.0)),
        ],
        dtype=float,
    )
    return position, velocity


def collision_aware_position(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    terrain: object,
    min_agl_m: float,
    obstacle_layer: Optional[ObstacleLayer] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if obstacle_layer is None:
        return position, velocity
    adjusted_position = np.asarray(position, dtype=float).copy()
    for _ in range(8):
        collision = obstacle_layer.point_collides(
            float(adjusted_position[0]),
            float(adjusted_position[1]),
            float(adjusted_position[2]),
        )
        if collision is None:
            return adjusted_position, velocity.copy()
        pushed_xy = collision.push_outside_xy(float(adjusted_position[0]), float(adjusted_position[1]), margin_m=1.0)
        adjusted_position = np.array(
            [
                float(pushed_xy[0]),
                float(pushed_xy[1]),
                terrain.clamp_altitude(pushed_xy, float(adjusted_position[2]), min_agl_m=min_agl_m),
            ],
            dtype=float,
        )
    return adjusted_position, velocity.copy()


def push_waypoint_outside_obstacles(
    point_xy: Sequence[float],
    *,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer],
    agl_m: float,
) -> np.ndarray:
    candidate_xy = np.asarray(point_xy, dtype=float).reshape(2)
    if obstacle_layer is None:
        return candidate_xy
    for _ in range(8):
        candidate_z = terrain.height_at(float(candidate_xy[0]), float(candidate_xy[1])) + agl_m
        collision = obstacle_layer.point_collides(float(candidate_xy[0]), float(candidate_xy[1]), float(candidate_z))
        if collision is None:
            return candidate_xy
        candidate_xy = collision.push_outside_xy(float(candidate_xy[0]), float(candidate_xy[1]), margin_m=1.0)
    return candidate_xy


def adjust_waypoints_for_obstacles(
    waypoints_xy: np.ndarray,
    *,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer],
    agl_m: float,
) -> np.ndarray:
    return np.vstack(
        [
            push_waypoint_outside_obstacles(
                point_xy,
                terrain=terrain,
                obstacle_layer=obstacle_layer,
                agl_m=agl_m,
            )
            for point_xy in np.asarray(waypoints_xy, dtype=float)
        ]
    )


def advance_along_polyline(points_xy: np.ndarray, travel_distance_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    polyline = np.asarray(points_xy, dtype=float)
    if polyline.ndim != 2 or polyline.shape[0] < 2 or polyline.shape[1] != 2:
        raise ValueError("points_xy must have shape (n, 2) with n >= 2.")

    remaining_distance = max(float(travel_distance_m), 0.0)
    cursor = polyline[0].copy()
    last_direction = np.zeros(2, dtype=float)
    for index in range(len(polyline) - 1):
        endpoint = polyline[index + 1]
        segment = endpoint - cursor
        segment_length = float(np.linalg.norm(segment))
        if segment_length <= 1.0e-9:
            cursor = endpoint.copy()
            continue
        direction = segment / segment_length
        last_direction = direction
        if remaining_distance <= segment_length:
            new_xy = cursor + direction * remaining_distance
            remaining_polyline = np.vstack([new_xy, polyline[index + 1 :]])
            return remaining_polyline, new_xy, direction
        remaining_distance -= segment_length
        cursor = endpoint.copy()

    return np.vstack([polyline[-1], polyline[-1]]), polyline[-1].copy(), last_direction


def perimeter_waypoints(bounds: Mapping[str, float], inset_fraction: float = 0.18) -> np.ndarray:
    span_x_m = bounds["x_max_m"] - bounds["x_min_m"]
    span_y_m = bounds["y_max_m"] - bounds["y_min_m"]
    inset_x_m = span_x_m * inset_fraction
    inset_y_m = span_y_m * inset_fraction
    return np.array(
        [
            [bounds["x_min_m"] + inset_x_m, bounds["y_min_m"] + inset_y_m],
            [bounds["x_max_m"] - inset_x_m, bounds["y_min_m"] + inset_y_m],
            [bounds["x_max_m"] - inset_x_m, bounds["y_max_m"] - inset_y_m],
            [bounds["x_min_m"] + inset_x_m, bounds["y_max_m"] - inset_y_m],
        ],
        dtype=float,
    )


def build_orbital_anchor_points(center_xy: np.ndarray, radii_xy: np.ndarray, sample_count: int = 16) -> np.ndarray:
    angles = np.linspace(0.0, math.tau, num=max(sample_count, 8), endpoint=False)
    return np.column_stack(
        [
            center_xy[0] + radii_xy[0] * np.cos(angles),
            center_xy[1] + radii_xy[1] * np.sin(angles),
        ]
    )


def build_racetrack_anchor_points(
    center_xy: np.ndarray,
    straight_length_m: float,
    turn_radius_m: float,
    sample_count: int = 20,
) -> np.ndarray:
    straight_length_m = max(float(straight_length_m), 10.0)
    turn_radius_m = max(float(turn_radius_m), 5.0)
    track_length_m = (2.0 * straight_length_m) + (2.0 * math.pi * turn_radius_m)
    distances = np.linspace(0.0, track_length_m, num=max(sample_count, 8), endpoint=False)
    points: List[np.ndarray] = []
    for distance_m in distances:
        if distance_m < straight_length_m:
            x_m = center_xy[0] - (straight_length_m * 0.5) + distance_m
            y_m = center_xy[1] - turn_radius_m
            points.append(np.array([x_m, y_m], dtype=float))
        elif distance_m < straight_length_m + (math.pi * turn_radius_m):
            arc_distance_m = distance_m - straight_length_m
            angle = (-math.pi / 2.0) + (arc_distance_m / turn_radius_m)
            points.append(
                np.array(
                    [
                        center_xy[0] + (straight_length_m * 0.5) + turn_radius_m * math.cos(angle),
                        center_xy[1] + turn_radius_m * math.sin(angle),
                    ],
                    dtype=float,
                )
            )
        elif distance_m < (2.0 * straight_length_m) + (math.pi * turn_radius_m):
            top_distance_m = distance_m - straight_length_m - (math.pi * turn_radius_m)
            x_m = center_xy[0] + (straight_length_m * 0.5) - top_distance_m
            y_m = center_xy[1] + turn_radius_m
            points.append(np.array([x_m, y_m], dtype=float))
        else:
            arc_distance_m = distance_m - (2.0 * straight_length_m) - (math.pi * turn_radius_m)
            angle = (math.pi / 2.0) + (arc_distance_m / turn_radius_m)
            points.append(
                np.array(
                    [
                        center_xy[0] - (straight_length_m * 0.5) + turn_radius_m * math.cos(angle),
                        center_xy[1] + turn_radius_m * math.sin(angle),
                    ],
                    dtype=float,
                )
            )
    return np.vstack(points)


def static_ground_point(x_m: float, y_m: float, mast_agl_m: float, terrain: TerrainModel) -> TrajectoryFn:
    zero = np.zeros(3, dtype=float)

    def trajectory(_: float) -> Tuple[np.ndarray, np.ndarray]:
        ground_m = terrain.height_at(x_m, y_m)
        position = np.array([x_m, y_m, ground_m + mast_agl_m], dtype=float)
        return position, zero.copy()

    return trajectory


def ground_station_name(index: int) -> str:
    phonetic_names = [
        "alpha",
        "bravo",
        "charlie",
        "delta",
        "echo",
        "foxtrot",
        "golf",
        "hotel",
        "india",
        "juliet",
        "kilo",
        "lima",
    ]
    if index < len(phonetic_names):
        return f"ground-{phonetic_names[index]}"
    return f"ground-{index + 1:02d}"


def build_ground_station_specs(
    *,
    terrain: object,
    obstacle_layer: ObstacleLayer,
    map_bounds_m: Mapping[str, float],
    count: int,
    constants: SimulationConstants = _DEFAULT_CONSTANTS,
) -> List[Dict[str, object]]:
    sensor_cfg = constants.sensor
    layout_cfg = constants.ground_station_layout
    span_x_m = map_bounds_m["x_max_m"] - map_bounds_m["x_min_m"]
    span_y_m = map_bounds_m["y_max_m"] - map_bounds_m["y_min_m"]
    center_xy = np.array(
        [
            (map_bounds_m["x_min_m"] + map_bounds_m["x_max_m"]) * 0.5,
            (map_bounds_m["y_min_m"] + map_bounds_m["y_max_m"]) * 0.5,
        ],
        dtype=float,
    )
    normalized_offsets = list(layout_cfg.normalized_offsets)
    mast_agls = list(sensor_cfg.ground_station_mast_agls_m)
    bearing_stds = list(sensor_cfg.ground_station_bearing_stds_rad)
    dropout_probabilities = list(sensor_cfg.ground_station_dropout_probabilities)
    max_range_factors = list(sensor_cfg.ground_station_max_range_factors_m)

    specs: List[Dict[str, object]] = []
    for index in range(count):
        if index < len(normalized_offsets):
            offset_x, offset_y = normalized_offsets[index]
            nominal_xy = np.array(
                [
                    center_xy[0] + offset_x * (span_x_m * 0.5),
                    center_xy[1] + offset_y * (span_y_m * 0.5),
                ],
                dtype=float,
            )
        else:
            angle = (index - len(normalized_offsets)) * (math.tau / max(count - len(normalized_offsets), 4)) + layout_cfg.overflow_angle_offset_rad
            nominal_xy = center_xy + np.array(
                [
                    math.cos(angle) * span_x_m * layout_cfg.overflow_radius_fraction,
                    math.sin(angle) * span_y_m * layout_cfg.overflow_radius_fraction,
                ],
                dtype=float,
            )
        mast_agl_m = mast_agls[index % len(mast_agls)]
        adjusted_xy = clamp_to_bounds(nominal_xy, map_bounds_m, margin_m=min(span_x_m, span_y_m) * layout_cfg.bounds_margin_fraction)
        adjusted_xy = push_waypoint_outside_obstacles(
            adjusted_xy,
            terrain=terrain,
            obstacle_layer=obstacle_layer,
            agl_m=mast_agl_m,
        )
        specs.append(
            {
                "node_id": ground_station_name(index),
                "xy_m": (float(adjusted_xy[0]), float(adjusted_xy[1])),
                "mast_agl_m": mast_agl_m,
                "bearing_std_rad": bearing_stds[index % len(bearing_stds)],
                "dropout_probability": dropout_probabilities[index % len(dropout_probabilities)],
                "max_range_m": max_range_factors[index % len(max_range_factors)],
            }
        )
    return specs


def orbital_path(
    center_xy: np.ndarray,
    radii_xy: np.ndarray,
    base_agl_m: float,
    vertical_amplitude_m: float,
    omega: float,
    phase: float,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer] = None,
    planner: Optional[PathPlanner2D] = None,
    clearance_m: Optional[float] = None,
    min_agl_m: float = 18.0,
) -> TrajectoryFn:
    radius_x, radius_y = float(radii_xy[0]), float(radii_xy[1])
    if planner is not None:
        loop_route = planner.route_waypoints(
            build_orbital_anchor_points(np.asarray(center_xy, dtype=float), np.asarray(radii_xy, dtype=float)),
            clearance_m=planner.config.target_clearance_m if clearance_m is None else clearance_m,
            closed=True,
        )
        if loop_route is None:
            shifted_center_xy = planner.nearest_free_point(
                center_xy,
                planner.config.target_clearance_m if clearance_m is None else clearance_m,
            )
            loop_route = planner.route_waypoints(
                build_orbital_anchor_points(np.asarray(shifted_center_xy, dtype=float), np.asarray(radii_xy, dtype=float)),
                clearance_m=planner.config.target_clearance_m if clearance_m is None else clearance_m,
                closed=True,
            )
        if loop_route is not None:
            return looped_waypoint_path(
                waypoints_xy=loop_route.points_xy_m,
                speed_mps=max(0.5, max(radius_x, radius_y) * max(abs(omega), 0.05)),
                base_agl_m=base_agl_m,
                vertical_amplitude=vertical_amplitude_m,
                vertical_frequency=max(abs(omega) * 0.5, 0.01),
                phase=phase,
                terrain=terrain,
                obstacle_layer=obstacle_layer,
                min_agl_m=min_agl_m,
            )

    def trajectory(timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        angle = omega * timestamp_s + phase
        x_m = center_xy[0] + radius_x * np.cos(angle)
        y_m = center_xy[1] + radius_y * np.sin(angle)
        vx_mps = -radius_x * omega * np.sin(angle)
        vy_mps = radius_y * omega * np.cos(angle)

        agl_m = base_agl_m + vertical_amplitude_m * np.sin(angle * 0.5)
        agl_rate_mps = vertical_amplitude_m * 0.5 * omega * np.cos(angle * 0.5)
        position, velocity = compose_terrain_state(
            xy_m=[x_m, y_m],
            xy_velocity_mps=[vx_mps, vy_mps],
            agl_m=agl_m,
            agl_rate_mps=agl_rate_mps,
            terrain=terrain,
            min_agl_m=min_agl_m,
        )
        return collision_aware_position(
            position,
            velocity,
            terrain=terrain,
            min_agl_m=min_agl_m,
            obstacle_layer=obstacle_layer,
        )

    return trajectory


def sinusoid_target(
    start_xy: np.ndarray,
    velocity_xy: np.ndarray,
    lateral_axis: int,
    lateral_amplitude: float,
    lateral_frequency: float,
    base_agl_m: float,
    vertical_amplitude: float,
    vertical_frequency: float,
    phase: float,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer] = None,
    min_agl_m: float = 3.0,
) -> TrajectoryFn:
    if lateral_axis not in (0, 1):
        raise ValueError("lateral_axis must be 0 (x) or 1 (y).")

    def trajectory(timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        xy = start_xy + velocity_xy * timestamp_s
        derived_velocity_xy = velocity_xy.copy()

        lateral_phase = lateral_frequency * timestamp_s + phase
        xy[lateral_axis] += lateral_amplitude * np.sin(lateral_phase)
        derived_velocity_xy[lateral_axis] += lateral_amplitude * lateral_frequency * np.cos(lateral_phase)

        vertical_phase = vertical_frequency * timestamp_s + phase * 0.5
        agl_m = base_agl_m + vertical_amplitude * np.sin(vertical_phase)
        agl_rate_mps = vertical_amplitude * vertical_frequency * np.cos(vertical_phase)
        position, velocity = compose_terrain_state(
            xy_m=xy,
            xy_velocity_mps=derived_velocity_xy,
            agl_m=agl_m,
            agl_rate_mps=agl_rate_mps,
            terrain=terrain,
            min_agl_m=min_agl_m,
            terrain_speed_adaptation=False,
        )
        return collision_aware_position(
            position,
            velocity,
            terrain=terrain,
            min_agl_m=min_agl_m,
            obstacle_layer=obstacle_layer,
        )

    return trajectory


def racetrack_target(
    center_xy: np.ndarray,
    straight_length_m: float,
    turn_radius_m: float,
    speed_mps: float,
    base_agl_m: float,
    vertical_amplitude: float,
    vertical_frequency: float,
    phase: float,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer] = None,
    planner: Optional[PathPlanner2D] = None,
    clearance_m: Optional[float] = None,
    min_agl_m: float = 3.0,
) -> TrajectoryFn:
    straight_length_m = max(float(straight_length_m), 10.0)
    turn_radius_m = max(float(turn_radius_m), 5.0)
    speed_mps = max(float(speed_mps), 0.5)
    track_length_m = (2.0 * straight_length_m) + (2.0 * math.pi * turn_radius_m)
    phase_offset_m = (phase % math.tau) / math.tau * track_length_m
    if planner is not None:
        active_clearance_m = planner.config.target_clearance_m if clearance_m is None else clearance_m
        loop_route = planner.route_waypoints(
            build_racetrack_anchor_points(np.asarray(center_xy, dtype=float), straight_length_m, turn_radius_m),
            clearance_m=active_clearance_m,
            closed=True,
        )
        if loop_route is None:
            shifted_center_xy = planner.nearest_free_point(center_xy, active_clearance_m)
            loop_route = planner.route_waypoints(
                build_racetrack_anchor_points(np.asarray(shifted_center_xy, dtype=float), straight_length_m, turn_radius_m),
                clearance_m=active_clearance_m,
                closed=True,
            )
        if loop_route is not None:
            return looped_waypoint_path(
                waypoints_xy=loop_route.points_xy_m,
                speed_mps=speed_mps,
                base_agl_m=base_agl_m,
                vertical_amplitude=vertical_amplitude,
                vertical_frequency=vertical_frequency,
                phase=phase,
                terrain=terrain,
                obstacle_layer=obstacle_layer,
                min_agl_m=min_agl_m,
            )

    def trajectory(timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        distance_m = (speed_mps * timestamp_s + phase_offset_m) % track_length_m
        if distance_m < straight_length_m:
            x_m = center_xy[0] - (straight_length_m * 0.5) + distance_m
            y_m = center_xy[1] - turn_radius_m
            xy = np.array([x_m, y_m], dtype=float)
            velocity_xy = np.array([speed_mps, 0.0], dtype=float)
        elif distance_m < straight_length_m + (math.pi * turn_radius_m):
            arc_distance_m = distance_m - straight_length_m
            angle = (-math.pi / 2.0) + (arc_distance_m / turn_radius_m)
            xy = np.array(
                [
                    center_xy[0] + (straight_length_m * 0.5) + turn_radius_m * math.cos(angle),
                    center_xy[1] + turn_radius_m * math.sin(angle),
                ],
                dtype=float,
            )
            velocity_xy = np.array(
                [
                    -speed_mps * math.sin(angle),
                    speed_mps * math.cos(angle),
                ],
                dtype=float,
            )
        elif distance_m < (2.0 * straight_length_m) + (math.pi * turn_radius_m):
            top_distance_m = distance_m - straight_length_m - (math.pi * turn_radius_m)
            x_m = center_xy[0] + (straight_length_m * 0.5) - top_distance_m
            y_m = center_xy[1] + turn_radius_m
            xy = np.array([x_m, y_m], dtype=float)
            velocity_xy = np.array([-speed_mps, 0.0], dtype=float)
        else:
            arc_distance_m = distance_m - (2.0 * straight_length_m) - (math.pi * turn_radius_m)
            angle = (math.pi / 2.0) + (arc_distance_m / turn_radius_m)
            xy = np.array(
                [
                    center_xy[0] - (straight_length_m * 0.5) + turn_radius_m * math.cos(angle),
                    center_xy[1] + turn_radius_m * math.sin(angle),
                ],
                dtype=float,
            )
            velocity_xy = np.array(
                [
                    -speed_mps * math.sin(angle),
                    speed_mps * math.cos(angle),
                ],
                dtype=float,
            )

        vertical_phase = vertical_frequency * timestamp_s + phase * 0.5
        agl_m = base_agl_m + vertical_amplitude * np.sin(vertical_phase)
        agl_rate_mps = vertical_amplitude * vertical_frequency * np.cos(vertical_phase)
        position, velocity = compose_terrain_state(
            xy_m=xy,
            xy_velocity_mps=velocity_xy,
            agl_m=agl_m,
            agl_rate_mps=agl_rate_mps,
            terrain=terrain,
            min_agl_m=min_agl_m,
            terrain_speed_adaptation=False,
        )
        return collision_aware_position(
            position,
            velocity,
            terrain=terrain,
            min_agl_m=min_agl_m,
            obstacle_layer=obstacle_layer,
        )

    return trajectory


def looped_waypoint_path(
    waypoints_xy: np.ndarray,
    speed_mps: float,
    base_agl_m: float,
    vertical_amplitude: float,
    vertical_frequency: float,
    phase: float,
    terrain: object,
    obstacle_layer: Optional[ObstacleLayer] = None,
    min_agl_m: float = 3.0,
) -> TrajectoryFn:
    points = np.asarray(waypoints_xy, dtype=float)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] != 2:
        raise ValueError("waypoints_xy must be an array of shape (n, 2) with n >= 2.")

    wrapped_points = np.vstack([points, points[0]])
    segment_vectors = wrapped_points[1:] - wrapped_points[:-1]
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    if float(segment_lengths.sum()) <= 1e-6:
        raise ValueError("waypoints_xy must contain at least one non-zero segment.")
    cumulative_lengths = np.cumsum(segment_lengths)
    path_length_m = float(cumulative_lengths[-1])
    phase_offset_m = (phase % math.tau) / math.tau * path_length_m
    speed_mps = max(float(speed_mps), 0.5)

    def trajectory(timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        distance_m = (speed_mps * timestamp_s + phase_offset_m) % path_length_m
        segment_index = int(np.searchsorted(cumulative_lengths, distance_m, side="right"))
        segment_index = min(segment_index, len(segment_vectors) - 1)
        segment_start_distance = 0.0 if segment_index == 0 else float(cumulative_lengths[segment_index - 1])
        segment_distance = distance_m - segment_start_distance
        segment_length = max(float(segment_lengths[segment_index]), 1e-6)
        direction_xy = segment_vectors[segment_index] / segment_length
        xy = wrapped_points[segment_index] + direction_xy * segment_distance
        velocity_xy = direction_xy * speed_mps

        vertical_phase = vertical_frequency * timestamp_s + phase * 0.5
        agl_m = base_agl_m + vertical_amplitude * np.sin(vertical_phase)
        agl_rate_mps = vertical_amplitude * vertical_frequency * np.cos(vertical_phase)
        position, velocity = compose_terrain_state(
            xy_m=xy,
            xy_velocity_mps=velocity_xy,
            agl_m=agl_m,
            agl_rate_mps=agl_rate_mps,
            terrain=terrain,
            min_agl_m=min_agl_m,
            terrain_speed_adaptation=False,
        )
        return collision_aware_position(
            position,
            velocity,
            terrain=terrain,
            min_agl_m=min_agl_m,
            obstacle_layer=obstacle_layer,
        )

    return trajectory


def noisy_unit_vector(rng: np.random.Generator, direction: np.ndarray, noise_std_rad: float) -> np.ndarray:
    direction = direction / np.linalg.norm(direction)
    perturbation = rng.normal(0.0, noise_std_rad, size=3)
    candidate = direction + perturbation
    norm = np.linalg.norm(candidate)
    if norm < 1e-12:
        return direction
    return candidate / norm


def line_of_sight_clear(
    terrain: TerrainModel,
    origin: np.ndarray,
    target: np.ndarray,
    sample_step_m: float = 18.0,
    terrain_clearance_m: float = 1.0,
) -> bool:
    segment = target - origin
    distance_m = float(np.linalg.norm(segment))
    if distance_m < sample_step_m:
        return True

    sample_count = max(2, int(np.ceil(distance_m / sample_step_m)))
    for index in range(1, sample_count):
        alpha = index / sample_count
        point = origin + segment * alpha
        terrain_height_m = terrain.height_at(float(point[0]), float(point[1])) + terrain_clearance_m
        if point[2] <= terrain_height_m:
            return False
    return True


def occluding_object_for_segment(
    terrain: TerrainModel,
    origin: np.ndarray,
    target: np.ndarray,
    occluding_objects: Sequence[OccludingObject],
) -> Optional[OccludingObject]:
    for occluding_object in occluding_objects:
        if occluding_object.segment_intersects(origin=origin, target=target, terrain=terrain):
            return occluding_object
    return None


def elevation_angle_deg(line_of_sight: np.ndarray) -> float:
    range_m = float(np.linalg.norm(line_of_sight))
    if range_m < 1e-6:
        return 90.0
    return float(np.degrees(np.arcsin(np.clip(line_of_sight[2] / range_m, -1.0, 1.0))))


def scale_xy(point_xy: Sequence[float], scale: float) -> np.ndarray:
    return np.asarray(point_xy, dtype=float) * float(scale)


def map_scale_for_preset(map_preset: str) -> float:
    try:
        return MAP_PRESET_SCALES[map_preset]
    except KeyError as error:
        raise ValueError(f"Unknown map preset {map_preset!r}.") from error


def platform_profile_for_preset(platform_preset: str) -> PlatformPresetProfile:
    try:
        return PLATFORM_PRESETS[platform_preset]
    except KeyError as error:
        raise ValueError(f"Unknown platform preset {platform_preset!r}.") from error


def map_bounds_for_scale(scale: float, constants: SimulationConstants = _DEFAULT_CONSTANTS) -> Dict[str, float]:
    dyn = constants.dynamics
    return {
        "x_min_m": -dyn.map_bounds_x_extent_m * scale,
        "x_max_m": dyn.map_bounds_x_extent_m * scale,
        "y_min_m": dyn.map_bounds_y_min_m * scale,
        "y_max_m": dyn.map_bounds_y_max_m * scale,
    }


def scaled_terrain_model(scale: float) -> TerrainModel:
    return TerrainModel.default().scaled(scale)


def map_bounds_object(bounds: Mapping[str, float]) -> Bounds2D:
    return Bounds2D(
        x_min_m=float(bounds["x_min_m"]),
        x_max_m=float(bounds["x_max_m"]),
        y_min_m=float(bounds["y_min_m"]),
        y_max_m=float(bounds["y_max_m"]),
    )


def terrain_resolution_for_scale(scale: float) -> float:
    if scale <= 2.5:
        return 5.0
    if scale <= 4.0:
        return 10.0
    return 15.0


def _point_in_polygon(point_xy: np.ndarray, polygon_xy: np.ndarray) -> bool:
    inside = False
    for index in range(len(polygon_xy)):
        a = polygon_xy[index]
        b = polygon_xy[(index + 1) % len(polygon_xy)]
        if ((a[1] > point_xy[1]) != (b[1] > point_xy[1])):
            x_cross = (b[0] - a[0]) * (point_xy[1] - a[1]) / max((b[1] - a[1]), 1.0e-12) + a[0]
            if point_xy[0] < x_cross:
                inside = not inside
    return inside


@dataclass(frozen=True)
class LandCoverPatch:
    polygon_xy_m: np.ndarray
    land_cover_class: LandCoverClass
    density: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "polygon_xy_m", np.asarray(self.polygon_xy_m, dtype=float))


def procedural_land_cover_layer(
    bounds_xy_m: Bounds2D,
    terrain: object,
    obstacles: Sequence[object],
    *,
    patches: Sequence[LandCoverPatch] = (),
    resolution_m: float,
) -> LandCoverLayer:
    cols = max(1, int(math.ceil(bounds_xy_m.width_m / resolution_m)))
    rows = max(1, int(math.ceil(bounds_xy_m.height_m / resolution_m)))
    classes = np.full((rows, cols), int(LandCoverClass.OPEN), dtype=np.uint8)
    density = np.zeros((rows, cols), dtype=np.uint8)

    x_values = bounds_xy_m.x_min_m + ((np.arange(cols, dtype=float) + 0.5) * resolution_m)
    y_values = bounds_xy_m.y_min_m + ((np.arange(rows, dtype=float) + 0.5) * resolution_m)

    for row, y_m in enumerate(y_values):
        for col, x_m in enumerate(x_values):
            point_xy = np.array([x_m, y_m], dtype=float)
            point_z = float(terrain.height_at(float(x_m), float(y_m)))
            if point_z <= terrain.ground_plane_m + 1.0:
                classes[row, col] = int(LandCoverClass.WATER)

            for obstacle in obstacles:
                if isinstance(obstacle, ForestStand):
                    if _point_in_polygon(point_xy, obstacle.footprint_xy_m):
                        classes[row, col] = int(LandCoverClass.FOREST)
                        density[row, col] = max(density[row, col], int(np.clip(obstacle.density, 0.0, 1.0) * 255))
                elif hasattr(obstacle, "footprint_xy_m"):
                    footprint_source = obstacle.footprint_xy_m
                    if callable(footprint_source):
                        footprint = np.asarray(footprint_source(), dtype=float)
                    elif isinstance(footprint_source, np.ndarray):
                        footprint = footprint_source
                    else:
                        footprint = np.asarray(footprint_source, dtype=float)
                    if _point_in_polygon(point_xy, footprint):
                        classes[row, col] = int(LandCoverClass.URBAN)
                        density[row, col] = max(density[row, col], 180)
            for patch in patches:
                if _point_in_polygon(point_xy, patch.polygon_xy_m):
                    classes[row, col] = int(patch.land_cover_class)
                    density[row, col] = max(density[row, col], int(np.clip(patch.density, 0, 255)))

    return LandCoverLayer.from_rasters(
        bounds_xy_m=bounds_xy_m,
        classes=classes,
        density=density,
        resolution_m=resolution_m,
        tile_size_cells=256,
    )


def build_obstacle_cluster(
    cluster_id: str,
    center_xy: np.ndarray,
    scale: float,
    terrain: object,
) -> Tuple[List[object], List[object]]:
    center_xy = np.asarray(center_xy, dtype=float)
    hangar_center = np.array([float(center_xy[0] - 28.0 * scale), float(center_xy[1] - 18.0 * scale)], dtype=float)
    hangar_footprint = OrientedBox(
        primitive_id=f"{cluster_id}-hangar-shape",
        blocker_type="building",
        center_x_m=float(hangar_center[0]),
        center_y_m=float(hangar_center[1]),
        length_m=56.0 * scale,
        width_m=32.0 * scale,
        yaw_rad=0.18,
        base_z_m=0.0,
        top_z_m=1.0,
    ).footprint_xy_m()
    hangar_min_z, hangar_max_z = _terrain_height_span_at_points(terrain, hangar_footprint)
    hangar_base_z, hangar_top_z = _solid_obstacle_elevations(hangar_min_z, hangar_max_z, 32.0)

    warehouse_footprint = np.array(
        [
            [float(center_xy[0] + 12.0 * scale), float(center_xy[1] - 12.0 * scale)],
            [float(center_xy[0] + 62.0 * scale), float(center_xy[1] - 12.0 * scale)],
            [float(center_xy[0] + 62.0 * scale), float(center_xy[1] + 28.0 * scale)],
            [float(center_xy[0] + 12.0 * scale), float(center_xy[1] + 28.0 * scale)],
        ],
        dtype=float,
    )
    warehouse_min_z, warehouse_max_z = _terrain_height_span_at_points(terrain, warehouse_footprint)
    warehouse_base_z, warehouse_top_z = _solid_obstacle_elevations(warehouse_min_z, warehouse_max_z, 28.0)

    tower_center = np.array([float(center_xy[0] - 6.0 * scale), float(center_xy[1] + 38.0 * scale)], dtype=float)
    tower_radius_m = 10.0 * scale
    tower_min_z, tower_max_z = _terrain_height_span_for_cylinder(
        terrain,
        float(tower_center[0]),
        float(tower_center[1]),
        tower_radius_m,
    )
    tower_base_z, tower_top_z = _solid_obstacle_elevations(tower_min_z, tower_max_z, 54.0)

    hard_obstacles: List[object] = [
        OrientedBox(
            primitive_id=f"{cluster_id}-hangar",
            blocker_type="building",
            center_x_m=float(hangar_center[0]),
            center_y_m=float(hangar_center[1]),
            length_m=56.0 * scale,
            width_m=32.0 * scale,
            yaw_rad=0.18,
            base_z_m=hangar_base_z,
            top_z_m=hangar_top_z,
        ),
        BuildingPrism(
            primitive_id=f"{cluster_id}-warehouse",
            footprint_xy_m=warehouse_footprint,
            base_z_m=warehouse_base_z,
            top_z_m=warehouse_top_z,
        ),
        CylinderObstacle(
            primitive_id=f"{cluster_id}-tower",
            blocker_type="building",
            center_x_m=float(tower_center[0]),
            center_y_m=float(tower_center[1]),
            radius_m=tower_radius_m,
            base_z_m=tower_base_z,
            top_z_m=tower_top_z,
        ),
    ]
    forest_footprint = [
        [float(center_xy[0] - 84.0 * scale), float(center_xy[1] + 26.0 * scale)],
        [float(center_xy[0] - 24.0 * scale), float(center_xy[1] + 18.0 * scale)],
        [float(center_xy[0] + 8.0 * scale), float(center_xy[1] + 74.0 * scale)],
        [float(center_xy[0] - 62.0 * scale), float(center_xy[1] + 92.0 * scale)],
        [float(center_xy[0] - 102.0 * scale), float(center_xy[1] + 58.0 * scale)],
    ]
    soft_obstacles: List[object] = [
        _forest_stand_for_terrain(
            primitive_id=f"{cluster_id}-forest",
            footprint_xy_m=forest_footprint,
            terrain=terrain,
            reference_xy_m=(float(center_xy[0] - 48.0 * scale), float(center_xy[1] + 48.0 * scale)),
            canopy_top_agl_m=26.0,
            density=0.52,
        )
    ]
    return hard_obstacles, soft_obstacles


def _rectangle_patch(center_xy: np.ndarray, length_m: float, width_m: float, land_cover_class: LandCoverClass, density: int = 0) -> LandCoverPatch:
    center_xy = np.asarray(center_xy, dtype=float)
    half_length = length_m * 0.5
    half_width = width_m * 0.5
    return LandCoverPatch(
        polygon_xy_m=np.array(
            [
                [center_xy[0] - half_length, center_xy[1] - half_width],
                [center_xy[0] + half_length, center_xy[1] - half_width],
                [center_xy[0] + half_length, center_xy[1] + half_width],
                [center_xy[0] - half_length, center_xy[1] + half_width],
            ],
            dtype=float,
        ),
        land_cover_class=land_cover_class,
        density=density,
    )


def _ellipse_patch(
    center_xy: np.ndarray,
    radius_x_m: float,
    radius_y_m: float,
    land_cover_class: LandCoverClass,
    density: int = 0,
    segment_count: int = 20,
) -> LandCoverPatch:
    angles = np.linspace(0.0, math.tau, num=max(segment_count, 8), endpoint=False, dtype=float)
    center_xy = np.asarray(center_xy, dtype=float)
    polygon_xy_m = np.column_stack(
        [
            center_xy[0] + np.cos(angles) * radius_x_m,
            center_xy[1] + np.sin(angles) * radius_y_m,
        ]
    )
    return LandCoverPatch(
        polygon_xy_m=polygon_xy_m,
        land_cover_class=land_cover_class,
        density=density,
    )


def _river_corridor_patch(scale: float) -> LandCoverPatch:
    return LandCoverPatch(
        polygon_xy_m=np.array(
            [
                [-520.0, -42.0],
                [-260.0, -28.0],
                [-20.0, -38.0],
                [220.0, -18.0],
                [520.0, 2.0],
                [520.0, 68.0],
                [220.0, 48.0],
                [-20.0, 72.0],
                [-260.0, 60.0],
                [-520.0, 36.0],
            ],
            dtype=float,
        )
        * scale,
        land_cover_class=LandCoverClass.WATER,
        density=0,
    )


def _preset_environment_profile(
    *,
    terrain_preset: str,
    scale: float,
    map_bounds_m: Mapping[str, float],
    terrain: object,
    hard_obstacles: Sequence[object],
    soft_obstacles: Sequence[object],
) -> Tuple[List[object], List[LandCoverPatch], List[LandCoverPatch]]:
    bounds_center = np.array(
        [
            (map_bounds_m["x_min_m"] + map_bounds_m["x_max_m"]) * 0.5,
            (map_bounds_m["y_min_m"] + map_bounds_m["y_max_m"]) * 0.5,
        ],
        dtype=float,
    )
    river_patch = _river_corridor_patch(scale)
    coastal_patch = _rectangle_patch(
        center_xy=np.array([bounds_center[0], map_bounds_m["y_min_m"] + 95.0 * scale], dtype=float),
        length_m=(map_bounds_m["x_max_m"] - map_bounds_m["x_min_m"]) * 0.92,
        width_m=(map_bounds_m["y_max_m"] - map_bounds_m["y_min_m"]) * 0.28,
        land_cover_class=LandCoverClass.WATER,
    )
    lake_patches = [
        _ellipse_patch(np.array([-220.0, 165.0], dtype=float) * scale, 115.0 * scale, 78.0 * scale, LandCoverClass.WATER),
        _ellipse_patch(np.array([245.0, -175.0], dtype=float) * scale, 95.0 * scale, 62.0 * scale, LandCoverClass.WATER),
    ]
    urban_patches = [
        _rectangle_patch(bounds_center, 280.0 * scale, 220.0 * scale, LandCoverClass.URBAN, density=190),
        _rectangle_patch(np.array([310.0, 205.0], dtype=float) * scale, 150.0 * scale, 120.0 * scale, LandCoverClass.URBAN, density=180),
        _rectangle_patch(np.array([-260.0, -180.0], dtype=float) * scale, 160.0 * scale, 120.0 * scale, LandCoverClass.URBAN, density=180),
    ]
    terrain_only_patches: List[LandCoverPatch] = []

    if terrain_preset == "jungle_canopy":
        extra_canopy = [
            _forest_stand_for_terrain(
                primitive_id="jungle-canopy-central",
                footprint_xy_m=[
                    [-360.0 * scale, -120.0 * scale],
                    [240.0 * scale, -180.0 * scale],
                    [340.0 * scale, 160.0 * scale],
                    [-280.0 * scale, 220.0 * scale],
                ],
                terrain=terrain,
                reference_xy_m=(0.0, 0.0),
                canopy_top_agl_m=30.0,
                density=0.82,
            ),
            _forest_stand_for_terrain(
                primitive_id="jungle-canopy-east",
                footprint_xy_m=[
                    [160.0 * scale, -250.0 * scale],
                    [460.0 * scale, -180.0 * scale],
                    [420.0 * scale, 120.0 * scale],
                    [180.0 * scale, 80.0 * scale],
                ],
                terrain=terrain,
                reference_xy_m=(300.0 * scale, -40.0 * scale),
                canopy_top_agl_m=32.0,
                density=0.78,
            ),
        ]
        return ([*soft_obstacles, *extra_canopy], [], terrain_only_patches)

    if terrain_preset == "arctic_tundra":
        sparse_manmade = [obstacle for obstacle in hard_obstacles if getattr(obstacle, "blocker_type", "") in {"building", "wall"}][:2]
        return (sparse_manmade, [], terrain_only_patches)

    if terrain_preset == "military_compound":
        return (list(hard_obstacles), urban_patches, terrain_only_patches)

    if terrain_preset == "river_valley":
        riparian_stands = [
            _forest_stand_for_terrain(
                primitive_id="riparian-west",
                footprint_xy_m=[
                    [-420.0 * scale, 60.0 * scale],
                    [-90.0 * scale, 84.0 * scale],
                    [-70.0 * scale, 132.0 * scale],
                    [-380.0 * scale, 112.0 * scale],
                ],
                terrain=terrain,
                reference_xy_m=(-220.0 * scale, 96.0 * scale),
                canopy_top_agl_m=24.0,
                density=0.55,
            ),
            _forest_stand_for_terrain(
                primitive_id="riparian-east",
                footprint_xy_m=[
                    [40.0 * scale, -82.0 * scale],
                    [360.0 * scale, -58.0 * scale],
                    [380.0 * scale, -8.0 * scale],
                    [60.0 * scale, -24.0 * scale],
                ],
                terrain=terrain,
                reference_xy_m=(220.0 * scale, -42.0 * scale),
                canopy_top_agl_m=24.0,
                density=0.52,
            ),
        ]
        return (riparian_stands, [river_patch], [river_patch])

    if terrain_preset in {"urban_flat"}:
        return (list(hard_obstacles[:3]), urban_patches[:2], terrain_only_patches)

    if terrain_preset in {"coastal"}:
        return (list(soft_obstacles[:1]), [coastal_patch], [coastal_patch])

    if terrain_preset in {"lake_district"}:
        return (list(soft_obstacles[:2]), lake_patches, lake_patches)

    if terrain_preset in {"rolling_highlands"}:
        return (list(soft_obstacles), [], terrain_only_patches)

    if terrain_preset in {"desert_canyon"}:
        return (list(hard_obstacles[:1]), [], terrain_only_patches)

    if terrain_preset in {"alpine"}:
        return ([*soft_obstacles[:1], *hard_obstacles[:2]], [], terrain_only_patches)

    return ([], [], terrain_only_patches)


def build_default_environment(
    scale: float,
    terrain_model: TerrainModel,
    map_bounds_m: Mapping[str, float],
    *,
    map_preset: str = "medium",
    terrain_preset: str = "default",
    clean_terrain: bool = False,
) -> EnvironmentModel:
    environment_id = f"procedural-{scale:.2f}"
    bounds = map_bounds_object(map_bounds_m)
    terrain = EnvironmentModel.from_legacy(
        environment_id=environment_id,
        bounds_xy_m=bounds,
        terrain_model=terrain_model,
        terrain_resolution_m=terrain_resolution_for_scale(scale),
    ).terrain

    hangar_center = np.array([20.0 * scale, -35.0 * scale], dtype=float)
    hangar_footprint = OrientedBox(
        primitive_id="hangar-south-shape",
        blocker_type="building",
        center_x_m=float(hangar_center[0]),
        center_y_m=float(hangar_center[1]),
        length_m=72.0 * scale,
        width_m=42.0 * scale,
        yaw_rad=0.12,
        base_z_m=0.0,
        top_z_m=1.0,
    ).footprint_xy_m()
    hangar_min_z, hangar_max_z = _terrain_height_span_at_points(terrain, hangar_footprint)
    hangar_base_z, hangar_top_z = _solid_obstacle_elevations(hangar_min_z, hangar_max_z, 42.0)

    tower_center = np.array([145.0 * scale, 20.0 * scale], dtype=float)
    tower_radius_m = 14.0 * scale
    tower_min_z, tower_max_z = _terrain_height_span_for_cylinder(
        terrain,
        float(tower_center[0]),
        float(tower_center[1]),
        tower_radius_m,
    )
    tower_base_z, tower_top_z = _solid_obstacle_elevations(tower_min_z, tower_max_z, 68.0)

    warehouse_footprint = np.array(
        [
            [-145.0 * scale, 135.0 * scale],
            [-80.0 * scale, 135.0 * scale],
            [-80.0 * scale, 190.0 * scale],
            [-145.0 * scale, 190.0 * scale],
        ],
        dtype=float,
    )
    warehouse_min_z, warehouse_max_z = _terrain_height_span_at_points(terrain, warehouse_footprint)
    warehouse_base_z, warehouse_top_z = _solid_obstacle_elevations(warehouse_min_z, warehouse_max_z, 34.0)

    east_fence_start = np.array([175.0 * scale, -120.0 * scale], dtype=float)
    east_fence_end = np.array([230.0 * scale, 110.0 * scale], dtype=float)
    east_fence_footprint = WallSegment(
        primitive_id="east-fence-shape",
        blocker_type="wall",
        start_xy_m=east_fence_start,
        end_xy_m=east_fence_end,
        thickness_m=6.0,
        base_z_m=0.0,
        top_z_m=1.0,
    ).footprint_xy_m()
    east_fence_min_z, east_fence_max_z = _terrain_height_span_at_points(terrain, east_fence_footprint)
    east_fence_base_z, east_fence_top_z = _solid_obstacle_elevations(east_fence_min_z, east_fence_max_z, 18.0)

    hard_obstacles: List[object] = [
        OrientedBox(
            primitive_id="hangar-south",
            blocker_type="building",
            center_x_m=float(hangar_center[0]),
            center_y_m=float(hangar_center[1]),
            length_m=72.0 * scale,
            width_m=42.0 * scale,
            yaw_rad=0.12,
            base_z_m=hangar_base_z,
            top_z_m=hangar_top_z,
        ),
        CylinderObstacle(
            primitive_id="tower-east",
            blocker_type="building",
            center_x_m=float(tower_center[0]),
            center_y_m=float(tower_center[1]),
            radius_m=tower_radius_m,
            base_z_m=tower_base_z,
            top_z_m=tower_top_z,
        ),
        BuildingPrism(
            primitive_id="warehouse-north",
            footprint_xy_m=warehouse_footprint,
            base_z_m=warehouse_base_z,
            top_z_m=warehouse_top_z,
        ),
        WallSegment(
            primitive_id="east-fence",
            blocker_type="wall",
            start_xy_m=east_fence_start,
            end_xy_m=east_fence_end,
            thickness_m=6.0,
            base_z_m=east_fence_base_z,
            top_z_m=east_fence_top_z,
        ),
    ]
    soft_obstacles: List[object] = [
        _forest_stand_for_terrain(
            primitive_id="pine-stand-west",
            footprint_xy_m=[
                [-250.0 * scale, 35.0 * scale],
                [-150.0 * scale, 30.0 * scale],
                [-120.0 * scale, 135.0 * scale],
                [-205.0 * scale, 185.0 * scale],
                [-275.0 * scale, 125.0 * scale],
            ],
            terrain=terrain,
            reference_xy_m=(-200.0 * scale, 105.0 * scale),
            canopy_top_agl_m=30.0,
            density=0.58,
        ),
    ]
    cluster_centers_by_preset = {
        "large": [
            np.array([-230.0, -175.0], dtype=float) * scale,
        ],
        "xlarge": [
            np.array([-320.0, -210.0], dtype=float) * scale,
            np.array([285.0, -230.0], dtype=float) * scale,
            np.array([-260.0, 235.0], dtype=float) * scale,
        ],
        "regional": [
            np.array([-360.0, -240.0], dtype=float) * scale,
            np.array([320.0, -260.0], dtype=float) * scale,
            np.array([-340.0, 260.0], dtype=float) * scale,
            np.array([310.0, 245.0], dtype=float) * scale,
            np.array([10.0, 20.0], dtype=float) * scale,
        ],
    }
    for cluster_index, center_xy in enumerate(cluster_centers_by_preset.get(map_preset, ())):
        extra_hard, extra_soft = build_obstacle_cluster(f"{map_preset}-cluster-{cluster_index}", center_xy, scale, terrain)
        hard_obstacles.extend(extra_hard)
        soft_obstacles.extend(extra_soft)
    if map_preset == "regional":
        regional_east_start = np.array([420.0 * scale, -320.0 * scale], dtype=float)
        regional_east_end = np.array([470.0 * scale, 320.0 * scale], dtype=float)
        regional_east_footprint = WallSegment(
            primitive_id="regional-corridor-east-shape",
            blocker_type="wall",
            start_xy_m=regional_east_start,
            end_xy_m=regional_east_end,
            thickness_m=8.0,
            base_z_m=0.0,
            top_z_m=1.0,
        ).footprint_xy_m()
        regional_east_min_z, regional_east_max_z = _terrain_height_span_at_points(terrain, regional_east_footprint)
        regional_east_base_z, regional_east_top_z = _solid_obstacle_elevations(regional_east_min_z, regional_east_max_z, 20.0)
        regional_west_start = np.array([-470.0 * scale, -290.0 * scale], dtype=float)
        regional_west_end = np.array([-430.0 * scale, 300.0 * scale], dtype=float)
        regional_west_footprint = WallSegment(
            primitive_id="regional-corridor-west-shape",
            blocker_type="wall",
            start_xy_m=regional_west_start,
            end_xy_m=regional_west_end,
            thickness_m=8.0,
            base_z_m=0.0,
            top_z_m=1.0,
        ).footprint_xy_m()
        regional_west_min_z, regional_west_max_z = _terrain_height_span_at_points(terrain, regional_west_footprint)
        regional_west_base_z, regional_west_top_z = _solid_obstacle_elevations(regional_west_min_z, regional_west_max_z, 20.0)
        hard_obstacles.extend(
            [
                WallSegment(
                    primitive_id="regional-corridor-east",
                    blocker_type="wall",
                    start_xy_m=regional_east_start,
                    end_xy_m=regional_east_end,
                    thickness_m=8.0,
                    base_z_m=regional_east_base_z,
                    top_z_m=regional_east_top_z,
                ),
                WallSegment(
                    primitive_id="regional-corridor-west",
                    blocker_type="wall",
                    start_xy_m=regional_west_start,
                    end_xy_m=regional_west_end,
                    thickness_m=8.0,
                    base_z_m=regional_west_base_z,
                    top_z_m=regional_west_top_z,
                ),
            ]
        )
    preset_obstacles, land_cover_patches, terrain_only_patches = _preset_environment_profile(
        terrain_preset=terrain_preset,
        scale=scale,
        map_bounds_m=map_bounds_m,
        terrain=terrain,
        hard_obstacles=hard_obstacles,
        soft_obstacles=soft_obstacles,
    )
    all_obstacles = [] if clean_terrain else list(preset_obstacles)
    active_patches = list(terrain_only_patches if clean_terrain else land_cover_patches)
    land_cover = procedural_land_cover_layer(
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=all_obstacles,
        patches=active_patches,
        resolution_m=max(10.0, terrain.base_resolution_m * 2.0),
    )
    return EnvironmentModel(
        environment_id=environment_id,
        crs=EnvironmentCRS(runtime_crs_id="local-enu"),
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=ObstacleLayer(
            bounds_xy_m=bounds,
            tile_size_m=terrain.tile_size_cells * terrain.base_resolution_m,
            primitives=tuple(all_obstacles),
        ),
        land_cover=land_cover,
    )


def within_bounds(xy_m: Sequence[float], bounds: Mapping[str, float], margin_m: float = 0.0) -> bool:
    x_m = float(xy_m[0])
    y_m = float(xy_m[1])
    return (
        x_m >= bounds["x_min_m"] + margin_m
        and x_m <= bounds["x_max_m"] - margin_m
        and y_m >= bounds["y_min_m"] + margin_m
        and y_m <= bounds["y_max_m"] - margin_m
    )


def clamp_to_bounds(xy_m: Sequence[float], bounds: Mapping[str, float], margin_m: float = 0.0) -> np.ndarray:
    min_x = bounds["x_min_m"] + margin_m
    max_x = bounds["x_max_m"] - margin_m
    min_y = bounds["y_min_m"] + margin_m
    max_y = bounds["y_max_m"] - margin_m
    return np.array(
        [
            np.clip(float(xy_m[0]), min_x, max_x),
            np.clip(float(xy_m[1]), min_y, max_y),
        ],
        dtype=float,
    )


def _obstacle_footprint_xy(obstacle: object) -> Optional[np.ndarray]:
    footprint_source = getattr(obstacle, "footprint_xy_m", None)
    if footprint_source is None:
        return None
    footprint = footprint_source() if callable(footprint_source) else footprint_source
    footprint_array = np.asarray(footprint, dtype=float)
    if footprint_array.ndim != 2 or footprint_array.shape[1] != 2 or len(footprint_array) < 2:
        return None
    return footprint_array


def _terrain_height_span_at_points(
    terrain: object,
    points_xy_m: Sequence[Sequence[float]],
) -> Tuple[float, float]:
    heights = [float(terrain.height_at(float(x_m), float(y_m))) for x_m, y_m in points_xy_m]
    return min(heights), max(heights)


def _terrain_max_at_points(terrain: object, points_xy_m: Sequence[Sequence[float]]) -> float:
    return _terrain_height_span_at_points(terrain, points_xy_m)[1]


def _terrain_height_span_for_cylinder(
    terrain: object,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    *,
    sample_count: int = 8,
) -> Tuple[float, float]:
    angles = np.linspace(0.0, math.tau, num=sample_count, endpoint=False, dtype=float)
    sample_points = [
        (center_x_m + radius_m * math.cos(float(angle)), center_y_m + radius_m * math.sin(float(angle)))
        for angle in angles
    ]
    sample_points.append((center_x_m, center_y_m))
    return _terrain_height_span_at_points(terrain, sample_points)


def _terrain_max_for_cylinder(
    terrain: object,
    center_x_m: float,
    center_y_m: float,
    radius_m: float,
    *,
    sample_count: int = 8,
) -> float:
    return _terrain_height_span_for_cylinder(
        terrain,
        center_x_m,
        center_y_m,
        radius_m,
        sample_count=sample_count,
    )[1]


def _solid_obstacle_elevations(
    min_terrain_z: float,
    max_terrain_z: float,
    nominal_height_m: float,
    *,
    top_pad_m: float = GROUND_CONTACT_TOP_PAD_M,
) -> Tuple[float, float]:
    base_z = float(min_terrain_z)
    top_z = float(max_terrain_z + max(float(nominal_height_m), 0.0) + top_pad_m)
    return base_z, top_z


def _forest_stand_for_terrain(
    *,
    primitive_id: str,
    footprint_xy_m: Sequence[Sequence[float]],
    terrain: object,
    reference_xy_m: Sequence[float],
    canopy_top_agl_m: float,
    density: float,
) -> ForestStand:
    footprint = np.asarray(footprint_xy_m, dtype=float)
    min_terrain_z, max_terrain_z = _terrain_height_span_at_points(terrain, footprint)
    reference_z = float(terrain.height_at(float(reference_xy_m[0]), float(reference_xy_m[1])))
    return ForestStand(
        primitive_id=primitive_id,
        footprint_xy_m=footprint,
        canopy_base_z_m=min_terrain_z,
        canopy_top_z_m=max(reference_z + float(canopy_top_agl_m), max_terrain_z + GROUND_CONTACT_TOP_PAD_M),
        density=density,
    )


def split_bounds_along_x(bounds: Mapping[str, float], count: int) -> List[Dict[str, float]]:
    if count <= 1:
        return [dict(bounds)]
    sectors: List[Dict[str, float]] = []
    width_m = (bounds["x_max_m"] - bounds["x_min_m"]) / count
    for index in range(count):
        sectors.append(
            {
                "x_min_m": bounds["x_min_m"] + index * width_m,
                "x_max_m": bounds["x_min_m"] + (index + 1) * width_m,
                "y_min_m": bounds["y_min_m"],
                "y_max_m": bounds["y_max_m"],
            }
        )
    return sectors


def build_patrol_waypoints(
    rng: np.random.Generator,
    center_xy: np.ndarray,
    width_m: float,
    height_m: float,
    bounds: Mapping[str, float],
) -> np.ndarray:
    base_offsets = np.array(
        [
            [-0.48, -0.34],
            [0.15, -0.46],
            [0.44, -0.08],
            [0.28, 0.38],
            [-0.16, 0.47],
            [-0.46, 0.18],
        ],
        dtype=float,
    )
    jitter = rng.uniform(-0.08, 0.08, size=base_offsets.shape)
    extents = np.array([width_m, height_m], dtype=float)
    waypoints = center_xy + (base_offsets + jitter) * extents
    margin_m = min(width_m, height_m) * 0.15
    return np.vstack([clamp_to_bounds(point, bounds, margin_m=margin_m) for point in waypoints])


def build_lawnmower_waypoints(bounds: Mapping[str, float], lane_spacing_m: float) -> np.ndarray:
    span_x_m = bounds["x_max_m"] - bounds["x_min_m"]
    span_y_m = bounds["y_max_m"] - bounds["y_min_m"]
    x_min_m = bounds["x_min_m"] + span_x_m * 0.2
    x_max_m = bounds["x_max_m"] - span_x_m * 0.2
    y_min_m = bounds["y_min_m"] + span_y_m * 0.2
    y_max_m = bounds["y_max_m"] - span_y_m * 0.2
    if x_max_m <= x_min_m or y_max_m <= y_min_m:
        center_x_m = (bounds["x_min_m"] + bounds["x_max_m"]) * 0.5
        center_y_m = (bounds["y_min_m"] + bounds["y_max_m"]) * 0.5
        return np.array(
            [
                [center_x_m - span_x_m * 0.15, center_y_m],
                [center_x_m + span_x_m * 0.15, center_y_m],
            ],
            dtype=float,
        )

    lane_spacing_m = max(float(lane_spacing_m), 20.0)
    y_values = list(np.arange(y_min_m, y_max_m + lane_spacing_m * 0.5, lane_spacing_m))
    if not y_values:
        y_values = [y_min_m, y_max_m]
    elif y_values[-1] < y_max_m:
        y_values.append(y_max_m)

    waypoints: List[List[float]] = []
    sweep_to_max = True
    for y_m in y_values:
        if sweep_to_max:
            waypoints.append([x_min_m, float(y_m)])
            waypoints.append([x_max_m, float(y_m)])
        else:
            waypoints.append([x_max_m, float(y_m)])
            waypoints.append([x_min_m, float(y_m)])
        sweep_to_max = not sweep_to_max
    return np.asarray(waypoints, dtype=float)


def search_path(
    sector_bounds: Mapping[str, float],
    terrain: object,
    base_agl_m: float,
    vertical_amplitude_m: float,
    vertical_frequency_rad_s: float,
    lane_spacing_m: float,
    speed_mps: float,
    phase: float,
    environment: Optional[EnvironmentModel] = None,
    planner: Optional[PathPlanner2D] = None,
    min_agl_m: float = 18.0,
) -> TrajectoryFn:
    waypoints = build_lawnmower_waypoints(sector_bounds, lane_spacing_m=lane_spacing_m)
    if planner is not None:
        route = planner.route_waypoints(
            waypoints,
            clearance_m=planner.config.drone_clearance_m,
            closed=True,
        )
        if route is None:
            route = planner.route_waypoints(
                perimeter_waypoints(sector_bounds),
                clearance_m=planner.config.drone_clearance_m,
                closed=True,
            )
        if route is not None:
            waypoints = route.points_xy_m
        elif environment is not None:
            waypoints = adjust_waypoints_for_obstacles(
                waypoints,
                terrain=terrain,
                obstacle_layer=environment.obstacles,
                agl_m=base_agl_m,
            )
    elif environment is not None:
        waypoints = adjust_waypoints_for_obstacles(
            waypoints,
            terrain=terrain,
            obstacle_layer=environment.obstacles,
            agl_m=base_agl_m,
        )
    return looped_waypoint_path(
        waypoints_xy=waypoints,
        speed_mps=speed_mps,
        base_agl_m=base_agl_m,
        vertical_amplitude=vertical_amplitude_m,
        vertical_frequency=vertical_frequency_rad_s,
        phase=phase,
        terrain=terrain,
        obstacle_layer=None if environment is None else environment.obstacles,
        min_agl_m=min_agl_m,
    )


def follow_path(
    target_trajectory: TrajectoryFn,
    environment: EnvironmentModel,
    map_bounds_m: Mapping[str, float],
    base_agl_m: float,
    vertical_amplitude_m: float,
    vertical_frequency_rad_s: float,
    standoff_radius_m: float,
    map_scale: float,
    max_speed_mps: float,
    phase: float,
    lead_s: float = 4.0,
    candidate_count: int = 12,
    rotation_rate_rad_s: float = 0.08,
    planner: Optional[PathPlanner2D] = None,
    min_agl_m: float = 18.0,
    target_altitude_offset_m: Optional[float] = None,
    max_accel_mps2: Optional[float] = None,
    terrain_following: bool = False,
    terrain_following_agl_m: float = 30.0,
    terrain_following_smoothing_s: float = 1.5,
    drone_role: str = "interceptor",
    slot_index: int = 0,
    slot_count: int = 1,
) -> TrajectoryFn:
    return FollowPathController(
        target_trajectory=target_trajectory,
        environment=environment,
        map_bounds_m=map_bounds_m,
        base_agl_m=base_agl_m,
        standoff_radius_m=standoff_radius_m,
        map_scale=map_scale,
        max_speed_mps=max_speed_mps,
        phase=phase,
        lead_s=lead_s,
        candidate_count=candidate_count,
        rotation_rate_rad_s=rotation_rate_rad_s,
        planner=planner,
        min_agl_m=min_agl_m,
        vertical_amplitude_m=vertical_amplitude_m,
        vertical_frequency_rad_s=vertical_frequency_rad_s,
        target_altitude_offset_m=target_altitude_offset_m,
        max_accel_mps2=max_accel_mps2,
        terrain_following=terrain_following,
        terrain_following_agl_m=terrain_following_agl_m,
        terrain_following_smoothing_s=terrain_following_smoothing_s,
        drone_role=drone_role,
        slot_index=slot_index,
        slot_count=slot_count,
    )


def target_motion_sequence(target_motion_preset: str, count: int) -> List[str]:
    return [target_motion_preset] * count


def drone_mode_sequence(drone_mode_preset: str, count: int) -> List[str]:
    if drone_mode_preset == "mixed":
        return ["follow" if index % 2 == 0 else "search" for index in range(count)]
    return [drone_mode_preset] * count


def _stable_seed(base_seed: int, *parts: object) -> int:
    seed = int(base_seed) & 0xFFFFFFFF
    for part in parts:
        for char in str(part):
            seed = ((seed * 16777619) ^ ord(char)) & 0xFFFFFFFF
    return seed


def _apply_weather_adjustments(
    position: np.ndarray,
    velocity: np.ndarray,
    *,
    timestamp_s: float,
    weather: Optional[WeatherModel],
    terrain: Optional[object],
    map_bounds_m: Optional[Mapping[str, float]],
    min_agl_m: float,
    wind_scale: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray]:
    adjusted_position = np.asarray(position, dtype=float).copy()
    adjusted_velocity = np.asarray(velocity, dtype=float).copy()
    if weather is None:
        return adjusted_position, adjusted_velocity

    # Clamp position above terrain first so speed penalty uses the true AGL.
    if terrain is not None:
        terrain_height = float(terrain.height_at(float(adjusted_position[0]), float(adjusted_position[1])))
        adjusted_position[2] = max(adjusted_position[2], terrain_height + min_agl_m)
        altitude_m = float(adjusted_position[2]) - terrain_height
    else:
        terrain_height = None
        altitude_m = float(adjusted_position[2])
    speed_penalty = weather.flight_speed_penalty(altitude_m)
    adjusted_velocity[:2] = adjusted_velocity[:2] * speed_penalty

    wind = weather.wind.wind_at(altitude_m, timestamp_s)
    adjusted_position[:2] += wind[:2] * max(timestamp_s, 0.0) * 0.08 * wind_scale

    if map_bounds_m is not None:
        adjusted_position[:2] = clamp_to_bounds(adjusted_position[:2], map_bounds_m, margin_m=1.0)
    if terrain is not None:
        terrain_height = float(terrain.height_at(float(adjusted_position[0]), float(adjusted_position[1])))
        adjusted_position[2] = max(adjusted_position[2], terrain_height + min_agl_m)
    return adjusted_position, adjusted_velocity


@dataclass
class WeatherAdjustedTrajectory:
    trajectory: TrajectoryFn
    weather: Optional[WeatherModel]
    terrain: Optional[object]
    map_bounds_m: Optional[Mapping[str, float]]
    min_agl_m: float
    wind_scale: float = 0.35

    def __call__(self, timestamp_s: float) -> Tuple[np.ndarray, np.ndarray]:
        position, velocity = self.trajectory(timestamp_s)
        return _apply_weather_adjustments(
            position,
            velocity,
            timestamp_s=timestamp_s,
            weather=self.weather,
            terrain=self.terrain,
            map_bounds_m=self.map_bounds_m,
            min_agl_m=self.min_agl_m,
            wind_scale=self.wind_scale,
        )

    def reset_state(self) -> None:
        if hasattr(self.trajectory, "reset_state"):
            self.trajectory.reset_state()

    def __getattr__(self, name: str) -> object:
        return getattr(self.trajectory, name)


def _target_motion_profile(
    motion_name: str,
    index: int,
    platform_profile: PlatformPresetProfile,
    terrain: object,
    map_bounds_m: Mapping[str, float],
    constants: SimulationConstants,
) -> Tuple[float, float]:
    center_x = 0.5 * (float(map_bounds_m["x_min_m"]) + float(map_bounds_m["x_max_m"]))
    center_y = 0.5 * (float(map_bounds_m["y_min_m"]) + float(map_bounds_m["y_max_m"]))
    terrain_height = float(terrain.height_at(center_x, center_y))
    base_agl = constants.dynamics.aerial_target_min_agl_m + 20.0 * (index % 4)
    base_altitude_m = terrain_height + base_agl
    speed_lookup = {
        "sinusoid": 7.0,
        "racetrack": 7.2,
        "waypoint_patrol": 6.8,
        "loiter": 18.0,
        "transit": 24.0,
        "evasive": 22.0,
        "search_pattern": 16.0,
        "mixed": 20.0,
    }
    speed_mps = speed_lookup.get(motion_name, 18.0) * platform_profile.target_speed_scale
    return base_altitude_m, speed_mps


def _sensor_model_for_node(node: SimNode, *, seed: int) -> SensorModel:
    clutter_rate = 0.01 if node.sensor_type == "radar" else 0.004
    if node.is_mobile:
        clutter_rate *= 0.5
    config = SensorErrorConfig(
        base_bearing_std_rad=node.bearing_std_rad,
        range_noise_exponent=1.5,
        range_noise_reference_m=max(node.max_range_m * 0.45, 50.0),
        max_bearing_std_rad=max(node.bearing_std_rad * 6.0, node.bearing_std_rad + 0.05),
        min_detection_probability=0.08 if node.is_mobile else 0.12,
        detection_range_knee_m=max(node.max_range_m * 0.55, 50.0),
        detection_range_falloff_m=max(node.max_range_m * 0.20, 25.0),
        false_alarm_rate_per_scan=clutter_rate,
        clutter_bearing_std_rad=min(max(node.bearing_std_rad * 8.0, 0.08), 0.25),
        bias_drift_rate_rad_per_s=2.0e-5 if node.is_mobile else 1.0e-5,
        bias_drift_max_rad=max(node.bearing_std_rad * 2.0, 0.006),
    )
    model = SensorModel(config=config)
    model.initialize(seed=seed)
    return model


def _build_sensor_models(nodes: Sequence[SimNode], *, seed: int) -> Dict[str, SensorModel]:
    return {
        node.node_id: _sensor_model_for_node(node, seed=_stable_seed(seed, "sensor", node.node_id))
        for node in nodes
    }


def build_target_trajectory(
    target_id: str,
    motion_name: str,
    index: int,
    scale: float,
    platform_profile: PlatformPresetProfile,
    rng: np.random.Generator,
    terrain: object,
    map_bounds_m: Mapping[str, float],
    obstacle_layer: Optional[ObstacleLayer] = None,
    planner: Optional[PathPlanner2D] = None,
) -> TrajectoryFn:
    phase = float(rng.uniform(0.0, math.tau))

    if motion_name == "sinusoid":
        start_positions = [
            np.array([-40.0, 40.0], dtype=float),
            np.array([210.0, -160.0], dtype=float),
        ]
        velocities = [
            np.array([4.1, 1.7], dtype=float),
            np.array([-3.5, 2.6], dtype=float),
        ]
        lateral_axes = [1, 0]
        lateral_amplitudes = [34.0, 26.0]
        lateral_frequencies = [0.13, 0.11]
        base_agls = [135.0, 165.0]
        vertical_amplitudes = [18.0, 24.0]
        vertical_frequencies = [0.09, 0.08]
        i = index % len(start_positions)
        start_xy = scale_xy(start_positions[i], scale) + rng.uniform(-30.0, 30.0, size=2) * scale * (index // len(start_positions))
        return sinusoid_target(
            start_xy=clamp_to_bounds(start_xy, map_bounds_m, margin_m=40.0 * scale),
            velocity_xy=scale_xy(velocities[i], platform_profile.target_speed_scale),
            lateral_axis=lateral_axes[i],
            lateral_amplitude=lateral_amplitudes[i] * scale,
            lateral_frequency=lateral_frequencies[i] * (1.0 + 0.1 * (index // len(start_positions))),
            base_agl_m=base_agls[i] + 20.0 * (index // len(start_positions)),
            vertical_amplitude=vertical_amplitudes[i],
            vertical_frequency=vertical_frequencies[i],
            phase=phase,
            terrain=terrain,
            obstacle_layer=obstacle_layer,
            min_agl_m=AERIAL_TARGET_MIN_AGL_M,
        )

    if motion_name == "racetrack":
        centers = [
            np.array([-85.0, 70.0], dtype=float),
            np.array([180.0, -145.0], dtype=float),
        ]
        straight_lengths = [230.0, 270.0]
        turn_radii = [54.0, 60.0]
        speeds = [7.2, 6.6]
        base_agls = [145.0, 175.0]
        vertical_amplitudes = [16.0, 22.0]
        vertical_frequencies = [0.07, 0.06]
        i = index % len(centers)
        center_xy = clamp_to_bounds(
            scale_xy(centers[i], scale) + rng.uniform(-18.0, 18.0, size=2) * scale
            + rng.uniform(-60.0, 60.0, size=2) * scale * (index // len(centers)),
            map_bounds_m,
            margin_m=80.0 * scale,
        )
        return racetrack_target(
            center_xy=center_xy,
            straight_length_m=straight_lengths[i] * scale,
            turn_radius_m=turn_radii[i] * scale,
            speed_mps=speeds[i] * platform_profile.target_speed_scale,
            base_agl_m=base_agls[i] + 15.0 * (index // len(centers)),
            vertical_amplitude=vertical_amplitudes[i],
            vertical_frequency=vertical_frequencies[i],
            phase=phase,
            terrain=terrain,
            obstacle_layer=obstacle_layer,
            planner=None,
            clearance_m=None,
            min_agl_m=AERIAL_TARGET_MIN_AGL_M,
        )

    if motion_name == "waypoint_patrol":
        centers = [
            np.array([-110.0, 145.0], dtype=float),
            np.array([180.0, -125.0], dtype=float),
        ]
        widths = [220.0, 190.0]
        heights = [165.0, 150.0]
        speeds = [6.8, 6.2]
        base_agls = [155.0, 185.0]
        vertical_amplitudes = [18.0, 24.0]
        vertical_frequencies = [0.06, 0.05]
        i = index % len(centers)
        center_xy = clamp_to_bounds(
            scale_xy(centers[i], scale) + rng.uniform(-20.0, 20.0, size=2) * scale
            + rng.uniform(-50.0, 50.0, size=2) * scale * (index // len(centers)),
            map_bounds_m,
            margin_m=90.0 * scale,
        )
        waypoints = build_patrol_waypoints(
            rng=rng,
            center_xy=center_xy,
            width_m=widths[i] * scale,
            height_m=heights[i] * scale,
            bounds=map_bounds_m,
        )
        waypoints = adjust_waypoints_for_obstacles(
            waypoints,
            terrain=terrain,
            obstacle_layer=obstacle_layer,
            agl_m=base_agls[i],
        )
        return looped_waypoint_path(
            waypoints_xy=waypoints,
            speed_mps=speeds[i] * platform_profile.target_speed_scale,
            base_agl_m=base_agls[i] + 15.0 * (index // len(centers)),
            vertical_amplitude=vertical_amplitudes[i],
            vertical_frequency=vertical_frequencies[i],
            phase=phase,
            terrain=terrain,
            obstacle_layer=obstacle_layer,
            min_agl_m=AERIAL_TARGET_MIN_AGL_M,
        )

    raise ValueError(f"Unknown target motion {motion_name!r} for {target_id}.")


def generate_mission_zones(
    map_bounds_m: Mapping[str, float],
    scale: float,
    terrain: object,
    rng: np.random.Generator,
    count: Optional[int] = None,
    obstacles: Sequence[object] = (),
) -> List[MissionZone]:
    x_min = float(map_bounds_m["x_min_m"])
    x_max = float(map_bounds_m["x_max_m"])
    y_min = float(map_bounds_m["y_min_m"])
    y_max = float(map_bounds_m["y_max_m"])
    x_span = x_max - x_min
    y_span = y_max - y_min
    min_span = min(x_span, y_span)
    edge_buffer_m = 0.05 * min_span

    if count is None:
        count = max(3, int(scale * 1.5))

    zone_types = [ZONE_TYPE_SURVEILLANCE, ZONE_TYPE_OBJECTIVE, ZONE_TYPE_PATROL, ZONE_TYPE_EXCLUSION]
    base_radius = min_span * 0.06
    zone_specs = []
    for i in range(count):
        zone_type = zone_types[i % len(zone_types)]
        zone_specs.append(
            {
                "index": i,
                "zone_type": zone_type,
                "radius_m": float(rng.uniform(base_radius * 0.6, base_radius * 1.4)),
                "priority": 1 if zone_type == ZONE_TYPE_OBJECTIVE else (2 if zone_type == ZONE_TYPE_SURVEILLANCE else 3),
                "label": f"{zone_type}-{i}",
            }
        )

    building_positions: List[Tuple[float, float]] = []
    building_extents_m: List[float] = []
    for obstacle in obstacles:
        blocker_type = getattr(obstacle, "blocker_type", "")
        if blocker_type and blocker_type != "building":
            continue
        footprint = _obstacle_footprint_xy(obstacle)
        if hasattr(obstacle, "center_x_m") and hasattr(obstacle, "center_y_m"):
            center_xy = np.array([float(obstacle.center_x_m), float(obstacle.center_y_m)], dtype=float)
        elif footprint is not None:
            center_xy = footprint.mean(axis=0)
        else:
            continue

        if hasattr(obstacle, "radius_m"):
            extent_m = float(obstacle.radius_m)
        elif hasattr(obstacle, "length_m") and hasattr(obstacle, "width_m"):
            extent_m = 0.5 * max(float(obstacle.length_m), float(obstacle.width_m))
        elif footprint is not None:
            extent_m = float(np.max(np.linalg.norm(footprint - center_xy, axis=1)))
        else:
            extent_m = base_radius

        building_positions.append((float(center_xy[0]), float(center_xy[1])))
        building_extents_m.append(max(extent_m, 1.0))

    sample_x = np.linspace(x_min + edge_buffer_m, x_max - edge_buffer_m, num=16, dtype=float)
    sample_y = np.linspace(y_min + edge_buffer_m, y_max - edge_buffer_m, num=16, dtype=float)
    sampled_heights = np.zeros((len(sample_y), len(sample_x)), dtype=float)
    for row, y_m in enumerate(sample_y):
        for col, x_m in enumerate(sample_x):
            sampled_heights[row, col] = float(terrain.height_at(float(x_m), float(y_m)))
    gradient_y, gradient_x = np.gradient(sampled_heights, sample_y, sample_x, edge_order=1)
    gradient_magnitude = np.sqrt((gradient_x * gradient_x) + (gradient_y * gradient_y))

    terrain_samples = [
        {
            "xy": np.array([float(sample_x[col]), float(sample_y[row])], dtype=float),
            "height": float(sampled_heights[row, col]),
            "gradient": float(gradient_magnitude[row, col]),
        }
        for row in range(len(sample_y))
        for col in range(len(sample_x))
    ]
    candidate_limit = max(8, count * 4)
    peak_candidates = [sample["xy"] for sample in sorted(terrain_samples, key=lambda sample: sample["height"], reverse=True)[:candidate_limit]]
    flat_candidates = [sample["xy"] for sample in sorted(terrain_samples, key=lambda sample: (sample["gradient"], -sample["height"]))[:candidate_limit]]

    placed_zones: List[MissionZone] = []
    objective_centers: List[np.ndarray] = []
    zones_by_index: Dict[int, MissionZone] = {}

    def _zone_margin(radius_m: float) -> float:
        return radius_m + edge_buffer_m

    def _clamped_center(candidate_xy: np.ndarray, radius_m: float) -> np.ndarray:
        return clamp_to_bounds(candidate_xy, map_bounds_m, margin_m=_zone_margin(radius_m))

    def _random_center(radius_m: float) -> np.ndarray:
        cx = float(rng.uniform(x_min + 0.20 * x_span, x_max - 0.20 * x_span))
        cy = float(rng.uniform(y_min + 0.20 * y_span, y_max - 0.20 * y_span))
        return _clamped_center(np.array([cx, cy], dtype=float), radius_m)

    def _separated(candidate_xy: np.ndarray, radius_m: float) -> bool:
        for zone in placed_zones:
            min_spacing_m = 0.3 * (radius_m + float(zone.radius_m))
            if float(np.linalg.norm(candidate_xy - zone.center[:2])) < min_spacing_m:
                return False
        return True

    def _relative_candidates(candidates: Sequence[np.ndarray], radius_m: float) -> Sequence[np.ndarray]:
        if not objective_centers:
            return candidates
        related = []
        for candidate_xy in candidates:
            nearest_objective_m = min(float(np.linalg.norm(candidate_xy - objective_xy)) for objective_xy in objective_centers)
            if 2.0 * radius_m <= nearest_objective_m <= 3.0 * radius_m:
                related.append(candidate_xy)
        return related or candidates

    def _zone_from_center(spec: Mapping[str, object], center_xy: np.ndarray) -> MissionZone:
        cx, cy = _clamped_center(center_xy, float(spec["radius_m"]))
        cz = float(terrain.height_at(float(cx), float(cy)))
        return MissionZone(
            zone_id=f"zone-{spec['index']}",
            zone_type=str(spec["zone_type"]),
            center=vec3(float(cx), float(cy), cz),
            radius_m=float(spec["radius_m"]),
            priority=int(spec["priority"]),
            label=str(spec["label"]),
        )

    def _try_place(spec: Mapping[str, object], sampler) -> MissionZone:
        radius_m = float(spec["radius_m"])
        for _ in range(20):
            candidate_xy = sampler(radius_m)
            if candidate_xy is None:
                break
            clamped_xy = _clamped_center(np.asarray(candidate_xy, dtype=float), radius_m)
            if _separated(clamped_xy, radius_m):
                return _zone_from_center(spec, clamped_xy)
        for _ in range(20):
            random_xy = _random_center(radius_m)
            if _separated(random_xy, radius_m):
                return _zone_from_center(spec, random_xy)
        return _zone_from_center(spec, _random_center(radius_m))

    def _sample_objective(radius_m: float) -> Optional[np.ndarray]:
        if not building_positions:
            return None
        anchor_index = int(rng.integers(0, len(building_positions)))
        anchor_xy = np.asarray(building_positions[anchor_index], dtype=float)
        anchor_extent_m = building_extents_m[anchor_index]
        angle = float(rng.uniform(0.0, math.tau))
        offset_m = 1.5 * anchor_extent_m + radius_m + float(rng.uniform(0.0, max(radius_m, 0.35 * anchor_extent_m)))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        return anchor_xy + direction * offset_m

    def _sample_exclusion(radius_m: float) -> Optional[np.ndarray]:
        if not building_positions:
            return None
        anchor_index = int(rng.integers(0, len(building_positions)))
        anchor_xy = np.asarray(building_positions[anchor_index], dtype=float)
        anchor_extent_m = building_extents_m[anchor_index]
        angle = float(rng.uniform(0.0, math.tau))
        offset_m = anchor_extent_m + radius_m + float(rng.uniform(0.0, radius_m))
        direction = np.array([math.cos(angle), math.sin(angle)], dtype=float)
        return anchor_xy + direction * offset_m

    def _sample_surveillance(radius_m: float) -> Optional[np.ndarray]:
        candidates = _relative_candidates(peak_candidates, radius_m)
        if not candidates:
            return None
        top_band = min(len(candidates), 6)
        anchor_xy = np.asarray(candidates[int(rng.integers(0, top_band))], dtype=float)
        return anchor_xy + rng.uniform(-0.25, 0.25, size=2) * radius_m

    def _sample_patrol(radius_m: float) -> Optional[np.ndarray]:
        candidates = _relative_candidates(flat_candidates, radius_m)
        if not candidates:
            return None
        top_band = min(len(candidates), 8)
        anchor_xy = np.asarray(candidates[int(rng.integers(0, top_band))], dtype=float)
        return anchor_xy + rng.uniform(-0.35, 0.35, size=2) * radius_m

    for pass_types, sampler_by_type in (
        (
            {ZONE_TYPE_OBJECTIVE, ZONE_TYPE_EXCLUSION},
            {
                ZONE_TYPE_OBJECTIVE: _sample_objective,
                ZONE_TYPE_EXCLUSION: _sample_exclusion,
            },
        ),
        (
            {ZONE_TYPE_SURVEILLANCE, ZONE_TYPE_PATROL},
            {
                ZONE_TYPE_SURVEILLANCE: _sample_surveillance,
                ZONE_TYPE_PATROL: _sample_patrol,
            },
        ),
    ):
        for spec in zone_specs:
            zone_type = str(spec["zone_type"])
            if zone_type not in pass_types:
                continue
            zone = _try_place(spec, sampler_by_type[zone_type])
            zones_by_index[int(spec["index"])] = zone
            placed_zones.append(zone)
            if zone.zone_type == ZONE_TYPE_OBJECTIVE:
                objective_centers.append(zone.center[:2].copy())

    return [zones_by_index[index] for index in range(count)]


def build_default_scenario(
    options: Optional[ScenarioOptions] = None,
    seed: int = 7,
    constants: SimulationConstants = _DEFAULT_CONSTANTS,
) -> ScenarioDefinition:
    active_options = options or ScenarioOptions()
    active_weather = weather_from_preset(active_options.weather_preset)
    scale = map_scale_for_preset(active_options.map_preset)
    platform_profile = platform_profile_for_preset(active_options.platform_preset)
    terrain = (
        scaled_terrain_model(scale)
        if active_options.terrain_preset == "default"
        else terrain_model_from_preset(active_options.terrain_preset, scale)
    )
    map_bounds_m = map_bounds_for_scale(scale, constants=constants)
    environment = build_default_environment(
        scale,
        terrain,
        map_bounds_m,
        map_preset=active_options.map_preset,
        terrain_preset=active_options.terrain_preset,
        clean_terrain=active_options.clean_terrain,
    )
    terrain_layer = environment.terrain
    planner = PathPlanner2D(
        bounds_xy_m=environment.bounds_xy_m,
        obstacle_layer=environment.obstacles,
        config=PlannerConfig(),
    )
    rng = np.random.default_rng(seed)

    _default_target_names = ["lynx", "orca", "fox", "hawk", "viper", "cobra", "eagle", "raven",
                              "wolf", "bear", "puma", "falcon", "shark", "stag", "crane", "pike"]
    target_count = active_options.target_count
    target_ids = [
        f"asset-{_default_target_names[i % len(_default_target_names)]}"
        if i < len(_default_target_names) else f"asset-{i}"
        for i in range(target_count)
    ]
    target_assignments = {}
    targets: List[SimTarget] = []
    for index, target_id in enumerate(target_ids):
        motion_name = target_motion_sequence(active_options.target_motion_preset, len(target_ids))[index]
        altitude_m, speed_mps = _target_motion_profile(
            motion_name,
            index,
            platform_profile,
            terrain_layer,
            map_bounds_m,
            constants,
        )
        base_trajectory = build_behavior_trajectory(
            motion_name,
            map_bounds_m,
            altitude_m=altitude_m,
            speed_mps=speed_mps,
            seed=_stable_seed(seed, "target", target_id, motion_name, index),
            terrain_height_fn=lambda x_m, y_m, terrain_layer=terrain_layer: float(
                terrain_layer.height_at(float(x_m), float(y_m))
            ),
        )
        trajectory = WeatherAdjustedTrajectory(
            trajectory=base_trajectory,
            weather=active_weather,
            terrain=terrain_layer,
            map_bounds_m=map_bounds_m,
            min_agl_m=constants.dynamics.aerial_target_min_agl_m,
        )
        target_assignments[target_id] = motion_name
        targets.append(SimTarget(target_id=target_id, trajectory=trajectory))
    occluding_objects = tuple(environment.obstacles.primitives)

    sensor_cfg = constants.sensor
    dynamics_cfg = constants.dynamics
    ground_specs = build_ground_station_specs(
        terrain=terrain_layer,
        obstacle_layer=environment.obstacles,
        map_bounds_m=map_bounds_m,
        count=active_options.ground_station_count,
        constants=constants,
    )
    nodes = [
        SimNode(
            node_id=spec["node_id"],
            is_mobile=False,
            bearing_std_rad=spec["bearing_std_rad"] * sensor_cfg.ground_station_bearing_std_rad_scale,
            dropout_probability=spec["dropout_probability"] * sensor_cfg.ground_station_dropout_probability_scale,
            max_range_m=float(spec["max_range_m"]) * platform_profile.ground_station_range_scale,
            trajectory=static_ground_point(
                float(spec["xy_m"][0]),
                float(spec["xy_m"][1]),
                mast_agl_m=float(spec["mast_agl_m"]),
                terrain=terrain_layer,
            ),
            sensor_type=sensor_cfg.ground_station_sensor_type,
            fov_half_angle_deg=sensor_cfg.ground_station_fov_half_angle_deg,
        )
        for spec in ground_specs
    ]

    _compass_names = ["east", "north", "west", "south", "ne", "nw", "se", "sw"]
    drone_count = active_options.drone_count
    drone_specs = []
    for di in range(drone_count):
        drone_name = _compass_names[di % len(_compass_names)] if di < len(_compass_names) else str(di)
        drone_specs.append({
            "node_id": f"drone-{drone_name}",
            "bearing_std_rad": sensor_cfg.drone_base_bearing_std_rad + sensor_cfg.drone_bearing_std_rad_increment * (di % 4),
            "dropout_probability": sensor_cfg.drone_base_dropout_probability + sensor_cfg.drone_dropout_probability_increment * (di % 3),
            "max_range_m": sensor_cfg.drone_base_max_range_m * platform_profile.drone_max_range_scale,
            "base_agl_m": dynamics_cfg.drone_base_agl_m + dynamics_cfg.drone_agl_increment_m * (di % 4),
            "vertical_amplitude_m": dynamics_cfg.drone_vertical_amplitude_base_m + dynamics_cfg.drone_vertical_amplitude_increment_m * (di % 3),
            "vertical_frequency_rad_s": dynamics_cfg.drone_vertical_frequency_base_rad_s - dynamics_cfg.drone_vertical_frequency_decrement_rad_s * (di % 3),
            "search_speed_mps": (dynamics_cfg.drone_search_speed_base_mps - dynamics_cfg.drone_search_speed_decrement_mps * (di % 3)) * platform_profile.drone_search_speed_scale,
        })

    drone_modes = drone_mode_sequence(active_options.drone_mode_preset, len(drone_specs))
    search_drone_indices = [index for index, mode in enumerate(drone_modes) if mode == "search"]
    search_sectors = split_bounds_along_x(map_bounds_m, len(search_drone_indices))
    search_sector_by_index = {
        drone_index: search_sectors[search_index]
        for search_index, drone_index in enumerate(search_drone_indices)
    }

    # Pre-compute ground station positions at t=0 for drone-to-station assignment
    _station_positions = {}
    for gs_spec in ground_specs:
        gs_traj = static_ground_point(
            float(gs_spec["xy_m"][0]),
            float(gs_spec["xy_m"][1]),
            mast_agl_m=float(gs_spec["mast_agl_m"]),
            terrain=terrain_layer,
        )
        _station_positions[gs_spec["node_id"]] = gs_traj(0.0)[0]

    target_by_id = {target.target_id: target for target in targets}
    drone_assignments: Dict[str, str] = {}
    drone_target_assignments: Dict[str, str] = {}
    drone_roles_map: Dict[str, str] = {}
    adaptive_drone_controllers: Dict[str, ObservationTriggeredFollowController] = {}
    launchable_controllers: Dict[str, LaunchableTrajectoryController] = {}

    # -----------------------------------------------------------------------
    # Cooperative slot assignment
    # Role is determined by drone_mode:
    #   "follow"  → interceptor  (directly dispatched to engage a known target)
    #   "search"  → tracker      (searches first, then orbits wide to localise)
    # Within each role-group, drones targeting the same asset spread
    # evenly around the standoff circle so sensor baselines are maximised.
    # -----------------------------------------------------------------------
    _n_drones = len(drone_specs)
    _drone_role_list: List[str] = [
        "interceptor" if drone_modes[i] == "follow" else "tracker"
        for i in range(_n_drones)
    ]

    # First pass: count how many drones of each role target each asset
    _tracker_total: Dict[str, int] = {}
    _interceptor_total: Dict[str, int] = {}
    for index in range(_n_drones):
        _tid = target_ids[index % len(target_ids)]
        if _drone_role_list[index] == "tracker":
            _tracker_total[_tid] = _tracker_total.get(_tid, 0) + 1
        else:
            _interceptor_total[_tid] = _interceptor_total.get(_tid, 0) + 1

    # Running slot counters (incremented as we assign)
    _tracker_slots: Dict[str, int] = {}
    _interceptor_slots: Dict[str, int] = {}

    for index, spec in enumerate(drone_specs):
        node_id = spec["node_id"]
        drone_mode = drone_modes[index]
        target_id = target_ids[index % len(target_ids)]
        role = _drone_role_list[index]
        drone_assignments[node_id] = drone_mode
        drone_target_assignments[node_id] = target_id
        drone_roles_map[node_id] = role

        # Assign cooperative slot for this drone's role on this target
        if role == "tracker":
            slot_index = _tracker_slots.get(target_id, 0)
            _tracker_slots[target_id] = slot_index + 1
            slot_count = _tracker_total.get(target_id, 1)
            standoff_r = dynamics_cfg.tracker_standoff_radius_m
            alt_offset = dynamics_cfg.tracker_altitude_offset_m
            min_agl = dynamics_cfg.tracker_follow_min_agl_m
            rot_rate = dynamics_cfg.tracker_rotation_rate_rad_s
            lead_s = dynamics_cfg.tracker_lead_s
            cand_count = dynamics_cfg.tracker_candidate_count
        else:  # interceptor
            slot_index = _interceptor_slots.get(target_id, 0)
            _interceptor_slots[target_id] = slot_index + 1
            slot_count = _interceptor_total.get(target_id, 1)
            standoff_r = dynamics_cfg.interceptor_follow_radius_m
            alt_offset = dynamics_cfg.interceptor_follow_altitude_offset_m
            min_agl = dynamics_cfg.interceptor_follow_min_agl_m
            rot_rate = dynamics_cfg.interceptor_follow_rotation_rate_rad_s
            lead_s = dynamics_cfg.interceptor_follow_lead_s
            cand_count = dynamics_cfg.interceptor_follow_candidate_count

        # Assign this drone to a ground station (round-robin)
        assigned_station_idx = index % len(ground_specs)
        assigned_station_id = ground_specs[assigned_station_idx]["node_id"]
        station_pos = _station_positions[assigned_station_id]

        if drone_mode == "follow":
            raw_trajectory = follow_path(
                target_trajectory=target_by_id[target_id].trajectory,
                environment=environment,
                map_bounds_m=map_bounds_m,
                base_agl_m=spec["base_agl_m"],
                vertical_amplitude_m=spec["vertical_amplitude_m"],
                vertical_frequency_rad_s=spec["vertical_frequency_rad_s"],
                standoff_radius_m=standoff_r,
                map_scale=scale,
                max_speed_mps=platform_profile.follow_speed_cap_mps,
                phase=float(rng.uniform(0.0, math.tau)),
                lead_s=lead_s,
                candidate_count=cand_count,
                rotation_rate_rad_s=rot_rate,
                planner=None,
                min_agl_m=min_agl,
                target_altitude_offset_m=alt_offset,
                max_accel_mps2=dynamics_cfg.drone_max_accel_mps2,
                drone_role=role,
                slot_index=slot_index,
                slot_count=slot_count,
            )
        else:
            search_trajectory = search_path(
                sector_bounds=search_sector_by_index[index],
                terrain=terrain_layer,
                base_agl_m=spec["base_agl_m"],
                vertical_amplitude_m=spec["vertical_amplitude_m"],
                vertical_frequency_rad_s=spec["vertical_frequency_rad_s"],
                lane_spacing_m=dynamics_cfg.drone_search_lane_spacing_scale * scale,
                speed_mps=spec["search_speed_mps"],
                phase=float(rng.uniform(0.0, math.tau)),
                environment=environment,
                planner=planner,
                min_agl_m=dynamics_cfg.interceptor_search_min_agl_m,
            )
            follow_trajectory = follow_path(
                target_trajectory=target_by_id[target_id].trajectory,
                environment=environment,
                map_bounds_m=map_bounds_m,
                base_agl_m=spec["base_agl_m"],
                vertical_amplitude_m=spec["vertical_amplitude_m"],
                vertical_frequency_rad_s=spec["vertical_frequency_rad_s"],
                standoff_radius_m=standoff_r,
                map_scale=scale,
                max_speed_mps=platform_profile.follow_speed_cap_mps,
                phase=float(rng.uniform(0.0, math.tau)),
                lead_s=lead_s,
                candidate_count=cand_count,
                rotation_rate_rad_s=rot_rate,
                planner=None,
                min_agl_m=min_agl,
                target_altitude_offset_m=alt_offset,
                max_accel_mps2=dynamics_cfg.drone_max_accel_mps2,
                drone_role=role,
                slot_index=slot_index,
                slot_count=slot_count,
            )
            obs_controller = ObservationTriggeredFollowController(
                node_id=node_id,
                search_trajectory=search_trajectory,
                follow_trajectory=follow_trajectory,
                preferred_target_id=target_id,
            )
            adaptive_drone_controllers[node_id] = obs_controller
            raw_trajectory = obs_controller

        # Wrap in launch controller — drones start grounded at their station
        launch_ctrl = LaunchableTrajectoryController(
            station_position=station_pos.copy(),
            operational_trajectory=raw_trajectory,
            climb_duration_s=dynamics_cfg.launch_climb_duration_s,
            operational_altitude_agl=spec["base_agl_m"],
            weather=active_weather,
            terrain=terrain_layer,
            map_bounds_m=map_bounds_m,
            min_operational_agl_m=min(
                dynamics_cfg.interceptor_search_min_agl_m,
                dynamics_cfg.interceptor_follow_min_agl_m,
            ),
            assigned_station_id=assigned_station_id,
        )
        launchable_controllers[node_id] = launch_ctrl

        nodes.append(
            SimNode(
                node_id=node_id,
                is_mobile=True,
                bearing_std_rad=spec["bearing_std_rad"],
                dropout_probability=spec["dropout_probability"],
                max_range_m=spec["max_range_m"],
                trajectory=launch_ctrl,
                sensor_type=sensor_cfg.drone_sensor_type,
                fov_half_angle_deg=sensor_cfg.drone_fov_half_angle_deg,
            )
        )

    mission_zones = generate_mission_zones(
        map_bounds_m,
        scale,
        terrain_layer,
        rng,
        obstacles=environment.obstacles.primitives,
    )

    scenario_name = (
        f"civilian-observation-{active_options.map_preset}-"
        f"{active_options.target_motion_preset}-{active_options.drone_mode_preset}"
    )
    return ScenarioDefinition(
        scenario_name=scenario_name,
        nodes=tuple(nodes),
        targets=tuple(targets),
        terrain=terrain_layer,
        occluding_objects=tuple(occluding_objects),
        environment=environment,
        weather=active_weather,
        constants=constants,
        options=active_options,
        map_bounds_m=map_bounds_m,
        target_motion_assignments=target_assignments,
        drone_planner_modes=drone_assignments,
        drone_target_assignments=drone_target_assignments,
        drone_roles=drone_roles_map,
        adaptive_drone_controllers=adaptive_drone_controllers,
        launchable_controllers=launchable_controllers,
        mission_zones=tuple(mission_zones),
    )


def _generation_rejection(
    *,
    node_state: NodeState,
    truth: TruthState,
    timestamp_s: float,
    reason: str,
    detail: str,
    closest_point: Optional[np.ndarray] = None,
    blocker_type: str = "",
    first_hit_range_m: Optional[float] = None,
) -> ObservationRejection:
    return ObservationRejection(
        node_id=node_state.node_id,
        target_id=truth.target_id,
        timestamp_s=timestamp_s,
        reason=reason,
        detail=detail,
        origin=np.asarray(node_state.position, dtype=float).copy(),
        attempted_point=np.asarray(truth.position, dtype=float).copy(),
        closest_point=None if closest_point is None else np.asarray(closest_point, dtype=float).copy(),
        blocker_type=blocker_type,
        first_hit_range_m=first_hit_range_m,
    )


def build_observations(
    rng: np.random.Generator,
    nodes: Sequence[SimNode],
    truths: Sequence[TruthState],
    timestamp_s: float,
    terrain: object,
    occluding_objects: Sequence[object] = (),
    min_elevation_deg: float = 0.5,
    environment: Optional[EnvironmentModel] = None,
    sensor_profile: Optional[SensorVisibilityModel] = None,
    weather: Optional[WeatherModel] = None,
    sensor_models: Optional[Mapping[str, SensorModel]] = None,
    constants: SimulationConstants = _DEFAULT_CONSTANTS,
    seed: Optional[int] = None,
) -> ObservationBatch:
    active_environment = environment
    if active_environment is None:
        bounds = Bounds2D.from_mapping(
            xy_bounds(
                [*(node.state(timestamp_s).position for node in nodes), *(truth.position for truth in truths)],
                padding_m=120.0,
            )
        )
        active_environment = EnvironmentModel.from_legacy(
            environment_id="observation-frame",
            bounds_xy_m=bounds,
            terrain_model=terrain,
            occluding_objects=occluding_objects,
        )
    active_profile = sensor_profile or SensorVisibilityModel.optical_default()
    active_sensor_models = dict(sensor_models) if sensor_models is not None else _build_sensor_models(
        nodes,
        seed=0 if seed is None else seed,
    )
    sensor_cfg = constants.sensor
    observations = []
    rejection_counts: Counter[str] = Counter()
    accepted_by_target: Counter[str] = Counter()
    rejected_by_target: Counter[str] = Counter()
    accepted_by_node_target: Counter[Tuple[str, str]] = Counter()
    generation_rejections: List[ObservationRejection] = []
    attempted_count = 0

    for node in nodes:
        node_state = node.state(timestamp_s)
        sensor_model = active_sensor_models.get(node.node_id)
        if sensor_model is None:
            sensor_model = _sensor_model_for_node(
                node,
                seed=_stable_seed(0 if seed is None else seed, "sensor", node.node_id),
            )
        for truth in truths:
            attempted_count += 1
            pair_rng = (
                rng
                if seed is None
                else np.random.default_rng(
                    _stable_seed(seed, node.node_id, truth.target_id, round(timestamp_s * 1000.0)),
                )
            )
            line_of_sight = truth.position - node_state.position
            range_m = float(np.linalg.norm(line_of_sight))
            if range_m < 1e-6 or range_m > node.max_range_m:
                rejection_counts[REJECT_OUT_OF_RANGE] += 1
                rejected_by_target[truth.target_id] += 1
                generation_rejections.append(
                    _generation_rejection(
                        node_state=node_state,
                        truth=truth,
                        timestamp_s=timestamp_s,
                        reason=REJECT_OUT_OF_RANGE,
                        detail=(
                            "Target is coincident with the node origin."
                            if range_m < 1e-6
                            else f"Range {range_m:.1f} m exceeds max sensor range {node.max_range_m:.1f} m."
                        ),
                    )
                )
                continue
            # --- FOV check for directional sensors ---
            fov_fraction = 0.0
            if node.fov_half_angle_deg < 179.0:
                sensor_dir = None
                if node.sensor_direction_fn is not None:
                    sensor_dir = node.sensor_direction_fn(timestamp_s)
                if sensor_dir is None:
                    _, vel = node.trajectory(timestamp_s)
                    speed_xy = float(np.linalg.norm(vel[:2]))
                    if speed_xy > 0.5:
                        look = vel.copy()
                        look[2] = -np.tan(np.radians(sensor_cfg.drone_look_down_angle_deg)) * speed_xy
                        sensor_dir = look / np.linalg.norm(look)
                if sensor_dir is not None:
                    los_unit = line_of_sight / range_m
                    cos_angle = float(np.dot(los_unit, sensor_dir))
                    angle_deg = float(np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))))
                    if angle_deg > node.fov_half_angle_deg:
                        rejection_counts[REJECT_OUTSIDE_FOV] += 1
                        rejected_by_target[truth.target_id] += 1
                        generation_rejections.append(
                            _generation_rejection(
                                node_state=node_state,
                                truth=truth,
                                timestamp_s=timestamp_s,
                                reason=REJECT_OUTSIDE_FOV,
                                detail=(
                                    f"Target angle {angle_deg:.1f} deg exceeds FOV half-angle "
                                    f"{node.fov_half_angle_deg:.1f} deg."
                                ),
                            )
                        )
                        continue
                    fov_fraction = angle_deg / max(node.fov_half_angle_deg, 1.0)
            if pair_rng.random() < node.dropout_probability:
                rejection_counts[REJECT_DROPOUT] += 1
                rejected_by_target[truth.target_id] += 1
                generation_rejections.append(
                    _generation_rejection(
                        node_state=node_state,
                        truth=truth,
                        timestamp_s=timestamp_s,
                        reason=REJECT_DROPOUT,
                        detail=f"Observation dropped with p={node.dropout_probability:.3f}.",
                    )
                )
                continue
            # Airborne nodes must be able to look down toward lower-altitude targets.
            required_elevation_deg = (
                sensor_cfg.mobile_min_elevation_deg
                if node.is_mobile
                else min_elevation_deg
            )
            elevation_deg = elevation_angle_deg(line_of_sight)
            if elevation_deg < required_elevation_deg:
                rejection_counts[REJECT_LOW_ELEVATION] += 1
                rejected_by_target[truth.target_id] += 1
                generation_rejections.append(
                    _generation_rejection(
                        node_state=node_state,
                        truth=truth,
                        timestamp_s=timestamp_s,
                        reason=REJECT_LOW_ELEVATION,
                        detail=(
                            f"Elevation angle {elevation_deg:.1f} deg is below required "
                            f"{required_elevation_deg:.1f} deg."
                        ),
                    )
                )
                continue
            detection = active_environment.query.compute_detection_probability(
                node_state.position,
                truth.position,
                sensor_profile=active_profile,
                weather=weather,
                max_range_m=node.max_range_m,
                terrain_clearance_m=sensor_cfg.los_terrain_clearance_m,
            )
            visibility = detection.vis_result
            if not visibility.visible:
                blocker_type = visibility.blocker_type
                if blocker_type == "terrain":
                    rejection_counts[REJECT_TERRAIN_OCCLUSION] += 1
                elif blocker_type == "building":
                    rejection_counts[REJECT_BUILDING_OCCLUSION] += 1
                elif blocker_type == "wall":
                    rejection_counts[REJECT_WALL_OCCLUSION] += 1
                elif blocker_type == "vegetation":
                    rejection_counts[REJECT_VEGETATION_OCCLUSION] += 1
                elif blocker_type == "out_of_coverage":
                    rejection_counts[REJECT_OUT_OF_COVERAGE] += 1
                else:
                    rejection_counts[REJECT_OBJECT_OCCLUSION] += 1
                generation_reason = {
                    "terrain": REJECT_TERRAIN_OCCLUSION,
                    "building": REJECT_BUILDING_OCCLUSION,
                    "wall": REJECT_WALL_OCCLUSION,
                    "vegetation": REJECT_VEGETATION_OCCLUSION,
                    "out_of_coverage": REJECT_OUT_OF_COVERAGE,
                }.get(blocker_type, REJECT_OBJECT_OCCLUSION)
                rejected_by_target[truth.target_id] += 1
                generation_rejections.append(
                    _generation_rejection(
                        node_state=node_state,
                        truth=truth,
                        timestamp_s=timestamp_s,
                        reason=generation_reason,
                        detail=(
                            f"LOS blocked by {blocker_type or 'obstacle'}."
                            if visibility.first_hit_range_m is None
                            else (
                                f"LOS blocked by {blocker_type or 'obstacle'} at "
                                f"{visibility.first_hit_range_m:.1f} m."
                            )
                        ),
                        closest_point=visibility.closest_point,
                        blocker_type=blocker_type,
                        first_hit_range_m=visibility.first_hit_range_m,
                    )
                )
                continue
            sensor_transmittance = sensor_atmospheric_attenuation(range_m, sensor_model.config)
            sensor_pd = sensor_detection_probability(
                range_m,
                sensor_model.config,
                atmospheric_transmittance=sensor_transmittance,
            )
            combined_pd = max(0.0, min(1.0, detection.p_d * sensor_pd))
            if pair_rng.random() > combined_pd:
                reason = (
                    REJECT_VEGETATION_OCCLUSION
                    if detection.dominant_loss_factor == "transmittance"
                    else REJECT_DROPOUT
                )
                rejection_counts[reason] += 1
                rejected_by_target[truth.target_id] += 1
                generation_rejections.append(
                    _generation_rejection(
                        node_state=node_state,
                        truth=truth,
                        timestamp_s=timestamp_s,
                        reason=reason,
                        detail=(
                            f"Probabilistic detection rejected the observation with "
                            f"P_d={combined_pd:.3f} ({detection.dominant_loss_factor})."
                        ),
                        closest_point=visibility.closest_point,
                        blocker_type=visibility.blocker_type,
                        first_hit_range_m=visibility.first_hit_range_m,
                    )
                )
                continue

            # FOV edge quality degradation: noise increases and confidence
            # decreases as the target approaches the FOV boundary.
            fov_noise_multiplier = 1.0 + fov_fraction * sensor_cfg.fov_noise_multiplier_slope
            fov_confidence_factor = 1.0 - fov_fraction * sensor_cfg.fov_confidence_slope
            # Altitude-based quality/range falloff for airborne sensors:
            # higher altitude degrades resolution but extends coverage.
            altitude_noise_multiplier = 1.0
            altitude_confidence_factor = 1.0
            if node.is_mobile:
                agl_m = max(float(node_state.position[2] - terrain.height_at(
                    float(node_state.position[0]), float(node_state.position[1]))), 0.0)
                # Quality degrades above 200 m AGL (atmospheric + resolution loss)
                if agl_m > sensor_cfg.altitude_quality_threshold_m:
                    excess = (
                        agl_m - sensor_cfg.altitude_quality_threshold_m
                    ) / max(sensor_cfg.altitude_quality_range_m, 1.0)
                    normalized_excess = min(excess, 1.0)
                    altitude_noise_multiplier = (
                        1.0 + normalized_excess * sensor_cfg.altitude_noise_multiplier_max
                    )
                    altitude_confidence_factor = max(
                        sensor_cfg.altitude_confidence_floor,
                        1.0 - normalized_excess * sensor_cfg.altitude_confidence_slope,
                    )
            effective_bearing_std = (
                sensor_model.effective_bearing_std(node.bearing_std_rad, range_m)
                * detection.effective_noise_multiplier
                * fov_noise_multiplier
                * altitude_noise_multiplier
            )
            observed_direction = noisy_unit_vector(
                pair_rng,
                line_of_sight,
                effective_bearing_std,
            )
            observed_direction = sensor_model.apply_bias(
                observed_direction,
                timestamp_s,
                pair_rng,
            )
            confidence = max(
                sensor_cfg.confidence_floor,
                (1.0 - (range_m / node.max_range_m) * sensor_cfg.confidence_range_slope)
                * combined_pd
                * fov_confidence_factor
                * altitude_confidence_factor,
            )
            observations.append(
                BearingObservation(
                    node_id=node.node_id,
                    target_id=truth.target_id,
                    origin=node_state.position,
                    direction=observed_direction,
                    bearing_std_rad=effective_bearing_std,
                    timestamp_s=timestamp_s,
                    confidence=confidence,
                )
            )
            accepted_by_target[truth.target_id] += 1
            accepted_by_node_target[(node.node_id, truth.target_id)] += 1

        clutter_rng = (
            rng
            if seed is None
            else np.random.default_rng(
                _stable_seed(seed, node.node_id, "clutter", round(timestamp_s * 1000.0)),
            )
        )
        for clutter in sensor_model.generate_clutter(
            clutter_rng,
            node_state.position,
            timestamp_s,
            node.node_id,
        ):
            observations.append(
                replace(
                    clutter,
                    confidence=min(clutter.confidence, sensor_cfg.confidence_floor * 0.9),
                )
            )

    return ObservationBatch(
        observations=observations,
        attempted_count=attempted_count,
        rejection_counts=dict(rejection_counts),
        accepted_by_target=dict(accepted_by_target),
        rejected_by_target=dict(rejected_by_target),
        accepted_by_node_target=dict(accepted_by_node_target),
        generation_rejections=generation_rejections,
    )


def write_metrics_csv(path: str, rows: Sequence[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        fieldnames = list(rows[0].keys()) if rows else METRICS_CSV_FIELDS
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        if rows:
            writer.writerows(rows)


def _build_metrics_rows(
    frame: PlatformFrame,
    truths: Sequence[TruthState],
    observation_batch: ObservationBatch,
) -> Tuple[List[dict], List[float]]:
    truth_by_id = {truth.target_id: truth for truth in truths}
    observation_counts: Dict[str, int] = {}
    for observation in frame.observations:
        observation_counts[observation.target_id] = observation_counts.get(observation.target_id, 0) + 1

    rows: List[dict] = []
    errors: List[float] = []
    for track in frame.tracks:
        truth = truth_by_id.get(track.track_id)
        if truth is None:
            continue
        error_m = float(np.linalg.norm(track.position - truth.position))
        errors.append(error_m)
        rows.append(
            {
                "time_s": round(frame.timestamp_s, 3),
                "track_id": track.track_id,
                "true_x_m": round(float(truth.position[0]), 3),
                "true_y_m": round(float(truth.position[1]), 3),
                "true_z_m": round(float(truth.position[2]), 3),
                "track_x_m": round(float(track.position[0]), 3),
                "track_y_m": round(float(track.position[1]), 3),
                "track_z_m": round(float(track.position[2]), 3),
                "error_m": round(error_m, 3),
                "measurement_std_m": round(track.measurement_std_m, 3),
                "stale_steps": track.stale_steps,
                "observations": observation_counts.get(track.track_id, 0),
                "sim_rejected_observations": int(sum(observation_batch.rejection_counts.values())),
            }
        )
    return rows, errors


def _build_simulation_summary(
    frames: Sequence[PlatformFrame],
    per_track_errors: Sequence[float],
    generation_attempted_count: int,
    generation_rejection_counts: Counter[str],
    generation_accepted_by_target: Counter[str],
    generation_rejected_by_target: Counter[str],
) -> Dict[str, object]:
    accepted_count = int(sum(generation_accepted_by_target.values()))
    rejected_count = int(sum(generation_rejection_counts.values()))
    return {
        "frame_count": len(frames),
        "track_rmse_m": float(np.sqrt(np.mean(np.square(per_track_errors)))) if per_track_errors else None,
        "track_update_count": len(per_track_errors),
        "generation_attempted_count": int(generation_attempted_count),
        "generation_accepted_count": accepted_count,
        "generation_rejected_count": rejected_count,
        "generation_acceptance_rate": (accepted_count / generation_attempted_count) if generation_attempted_count else 0.0,
        "generation_rejection_counts": dict(sorted(generation_rejection_counts.items())),
        "generation_accepted_by_target": dict(sorted(generation_accepted_by_target.items())),
        "generation_rejected_by_target": dict(sorted(generation_rejected_by_target.items())),
    }


def _build_replay_metadata(
    frames: Sequence[PlatformFrame],
    scenario: ScenarioDefinition,
    simulation_config: SimulationConfig,
    tracker_config: TrackerConfig,
    summary: Dict[str, object],
) -> Dict[str, object]:
    environment = scenario.environment
    if environment is None:
        raise ValueError("ScenarioDefinition.environment must be populated before replay export.")
    terrain_metadata = environment.terrain.to_metadata()
    requested_duration_s = (
        simulation_config.requested_duration_s
        if simulation_config.requested_duration_s is not None
        else simulation_config.actual_duration_s
    )
    return {
        "schema_version": "2.0",
        "terrain": terrain_metadata,
        "occluding_objects": environment.obstacles.to_metadata(),
        "environment_id": environment.environment_id,
        "crs_id": environment.crs.runtime_crs_id,
        "environment_bounds_m": environment.bounds_xy_m.to_metadata(),
        "terrain_summary": environment.terrain.terrain_summary(),
        "land_cover": environment.land_cover.to_metadata(),
        "land_cover_legend": LandCoverClass.legend(),
        "platform": {
            "min_observations": tracker_config.min_observations,
            "max_stale_steps": tracker_config.max_stale_steps,
        },
        "scenario_options": {
            "map_preset": scenario.options.map_preset,
            "target_motion_preset": scenario.options.target_motion_preset,
            "drone_mode_preset": scenario.options.drone_mode_preset,
            "terrain_preset": scenario.options.terrain_preset,
            "weather_preset": scenario.options.weather_preset,
            "clean_terrain": scenario.options.clean_terrain,
            "platform_preset": scenario.options.platform_preset,
            "ground_station_count": scenario.options.ground_station_count,
            "target_count": scenario.options.target_count,
            "drone_count": scenario.options.drone_count,
        },
        "zones": [
            {
                "zone_id": z.zone_id,
                "zone_type": z.zone_type,
                "center": z.center.tolist(),
                "radius_m": z.radius_m,
                "priority": z.priority,
                "label": z.label,
            }
            for z in scenario.mission_zones
        ],
        "requested_duration_s": float(requested_duration_s),
        "actual_duration_s": float(simulation_config.actual_duration_s),
        "target_motion_assignments": dict(sorted(scenario.target_motion_assignments.items())),
        "drone_planner_modes": dict(sorted(scenario.drone_planner_modes.items())),
        "drone_target_assignments": dict(sorted(scenario.drone_target_assignments.items())),
        "drone_roles": dict(sorted(scenario.drone_roles.items())),
        "observation_generation": {
            "attempted_count": summary["generation_attempted_count"],
            "accepted_count": summary["generation_accepted_count"],
            "rejected_count": summary["generation_rejected_count"],
            "acceptance_rate": summary["generation_acceptance_rate"],
            "rejection_counts": summary["generation_rejection_counts"],
            "accepted_by_target": summary["generation_accepted_by_target"],
            "rejected_by_target": summary["generation_rejected_by_target"],
        },
    }


def run_simulation(
    scenario: ScenarioDefinition,
    simulation_config: SimulationConfig,
    tracker_config: Optional[TrackerConfig] = None,
) -> SimulationResult:
    active_tracker_config = tracker_config or TrackerConfig()
    scenario.reset_runtime_state()
    rng = np.random.default_rng(simulation_config.seed)
    sensor_models = _build_sensor_models(scenario.nodes, seed=simulation_config.seed)
    service = TrackingService(config=active_tracker_config, retain_history=True)

    frames: List[PlatformFrame] = []
    metrics_rows: List[dict] = []
    per_track_errors: List[float] = []
    generation_rejection_counts: Counter[str] = Counter()
    generation_accepted_by_target: Counter[str] = Counter()
    generation_rejected_by_target: Counter[str] = Counter()
    generation_attempted_count = 0
    adaptive_drone_controllers = list(scenario.adaptive_drone_controllers.values())
    _launch_controllers = dict(scenario.launchable_controllers)
    # Index: station_id -> list of launchable drone controllers assigned to it
    _station_to_drones: Dict[str, List[LaunchableTrajectoryController]] = {}
    for ctrl in _launch_controllers.values():
        sid = ctrl.assigned_station_id
        _station_to_drones.setdefault(sid, []).append(ctrl)

    for step in range(simulation_config.steps):
        timestamp_s = step * simulation_config.dt_s
        node_states = scenario.node_states(timestamp_s)
        truths = scenario.truths(timestamp_s)
        observation_batch = build_observations(
            rng=rng,
            nodes=scenario.nodes,
            truths=truths,
            timestamp_s=timestamp_s,
            terrain=scenario.terrain,
            occluding_objects=scenario.occluding_objects,
            environment=scenario.environment,
            weather=scenario.weather,
            sensor_models=sensor_models,
            constants=scenario.constants,
            seed=simulation_config.seed,
        )
        generation_attempted_count += observation_batch.attempted_count
        generation_rejection_counts.update(observation_batch.rejection_counts)
        generation_accepted_by_target.update(observation_batch.accepted_by_target)
        generation_rejected_by_target.update(observation_batch.rejected_by_target)

        # --- Drone launch trigger ---
        # Ground stations that detect a target within 80% of their max_range
        # trigger an unlaunched drone assigned to that station.
        step_launch_events: List[LaunchEvent] = []
        for node in scenario.nodes:
            if node.sensor_type != "radar" or node.is_mobile:
                continue
            station_state = node.state(timestamp_s)
            drones_at_station = _station_to_drones.get(node.node_id, [])
            for truth in truths:
                range_m = float(np.linalg.norm(truth.position - station_state.position))
                if range_m < node.max_range_m * scenario.constants.sensor.launch_detection_range_fraction:
                    for ctrl in drones_at_station:
                        if not ctrl.launched:
                            ctrl.trigger_launch(timestamp_s, truth.target_id)
                            # Find drone_id by reverse lookup
                            drone_id = next(
                                (did for did, c in _launch_controllers.items() if c is ctrl),
                                "unknown",
                            )
                            step_launch_events.append(LaunchEvent(
                                drone_id=drone_id,
                                station_id=node.node_id,
                                target_id=truth.target_id,
                                launch_time_s=timestamp_s,
                            ))
                            break  # one launch per detection per step

        frame = service.ingest_frame(
            timestamp_s=timestamp_s,
            node_states=node_states,
            observations=observation_batch.observations,
            truths=truths,
        )
        for controller in adaptive_drone_controllers:
            if hasattr(controller, "update_from_frame"):
                controller.update_from_frame(frame, observation_batch)
        frame = PlatformFrame(
            timestamp_s=frame.timestamp_s,
            # Preserve the scenario-authored sensor metadata in replay frames.
            nodes=node_states,
            observations=frame.observations,
            rejected_observations=frame.rejected_observations,
            tracks=frame.tracks,
            truths=frame.truths,
            metrics=frame.metrics,
            generation_rejections=observation_batch.generation_rejections,
            launch_events=step_launch_events,
        )
        frames.append(frame)
        rows, errors = _build_metrics_rows(frame, truths, observation_batch)
        metrics_rows.extend(rows)
        per_track_errors.extend(errors)

    summary = _build_simulation_summary(
        frames=frames,
        per_track_errors=per_track_errors,
        generation_attempted_count=generation_attempted_count,
        generation_rejection_counts=generation_rejection_counts,
        generation_accepted_by_target=generation_accepted_by_target,
        generation_rejected_by_target=generation_rejected_by_target,
    )
    replay_metadata = _build_replay_metadata(
        frames=frames,
        scenario=scenario,
        simulation_config=simulation_config,
        tracker_config=active_tracker_config,
        summary=summary,
    )
    return SimulationResult(
        scenario_name=scenario.scenario_name,
        simulation_config=simulation_config,
        tracker_config=active_tracker_config,
        frames=frames,
        metrics_rows=metrics_rows,
        summary=summary,
        replay_metadata=replay_metadata,
    )


def build_simulation_report_lines(result: SimulationResult) -> List[str]:
    lines = ["time_s  active_tracks  observations  mean_error_m  max_error_m"]
    stride = max(1, int(round(1.0 / result.simulation_config.dt_s)))
    for frame in result.frames[::stride]:
        mean_error = "n/a" if frame.metrics.mean_error_m is None else f"{frame.metrics.mean_error_m:6.2f}"
        max_error = "n/a" if frame.metrics.max_error_m is None else f"{frame.metrics.max_error_m:6.2f}"
        lines.append(
            f"{frame.timestamp_s:>6.1f}  "
            f"{frame.metrics.active_track_count:>13}  "
            f"{frame.metrics.observation_count:>12}  "
            f"{mean_error:>12}  "
            f"{max_error:>11}"
        )

    rmse = result.summary["track_rmse_m"]
    if rmse is None:
        lines.append("")
        lines.append("No fused tracks were produced.")
    else:
        lines.append("")
        lines.append(f"Track RMSE: {rmse:.2f} m across {result.summary['track_update_count']} track updates")
    return lines


def build_replay_document_from_result(result: SimulationResult) -> ReplayDocument:
    return build_replay_document(
        result.frames,
        scenario_name=result.scenario_name,
        dt_s=result.simulation_config.dt_s,
        seed=result.simulation_config.seed,
        extra_meta=result.replay_metadata,
    )


def simulate(
    steps: Optional[int],
    dt: float,
    seed: int,
    csv_path: Optional[str],
    replay_path: Optional[str],
    duration_s: Optional[float] = None,
    map_preset: str = "regional",
    target_motion: str = "mixed",
    drone_mode: str = "mixed",
    terrain_preset: str = "alpine",
    weather_preset: str = "clear",
    clean_terrain: bool = False,
    platform_preset: str = "baseline",
    ground_stations: int = 7,
    target_count: int = 2,
    drone_count: int = 2,
    constants: SimulationConstants = _DEFAULT_CONSTANTS,
) -> SimulationResult:
    if steps is not None:
        simulation_config = SimulationConfig(steps=steps, dt_s=dt, seed=seed)
    else:
        active_duration_s = constants.dynamics.default_duration_s if duration_s is None else duration_s
        simulation_config = SimulationConfig.from_duration(active_duration_s, dt_s=dt, seed=seed)

    scenario = build_default_scenario(
        options=ScenarioOptions(
            map_preset=map_preset,
            target_motion_preset=target_motion,
            drone_mode_preset=drone_mode,
            terrain_preset=terrain_preset,
            weather_preset=weather_preset,
            clean_terrain=clean_terrain,
            platform_preset=platform_preset,
            ground_station_count=ground_stations,
            target_count=target_count,
            drone_count=drone_count,
        ),
        seed=seed,
        constants=constants,
    )
    result = run_simulation(
        scenario=scenario,
        simulation_config=simulation_config,
        tracker_config=DEFAULT_SIMULATION_TRACKER_CONFIG,
    )

    for line in build_simulation_report_lines(result):
        print(line)

    if csv_path:
        write_metrics_csv(csv_path, result.metrics_rows)
        print(f"CSV written to {csv_path}")

    if replay_path:
        replay_document = build_replay_document_from_result(result)
        write_replay_document(replay_path, replay_document)
        print(f"Replay written to {replay_path}")

    return result


class _TrackProvidedAction(argparse.Action):
    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: object,
        option_string: Optional[str] = None,
    ) -> None:
        provided = getattr(namespace, "_provided_sim_args", None)
        if provided is None:
            provided = set()
        else:
            provided = set(provided)
        provided.add(self.dest)
        setattr(namespace, "_provided_sim_args", provided)
        if self.nargs == 0:
            setattr(namespace, self.dest, self.const if self.const is not None else True)
        else:
            setattr(namespace, self.dest, values)


def _arg_was_provided(args: argparse.Namespace, dest: str) -> bool:
    provided = getattr(args, "_provided_sim_args", None)
    return isinstance(provided, set) and dest in provided


def _load_simulation_constants(path: str) -> SimulationConstants:
    lowered = path.lower()
    if lowered.endswith(".json"):
        return SimulationConstants.from_json(path)
    return SimulationConstants.from_yaml(path)


def add_cli_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.set_defaults(_provided_sim_args=None)
    duration_group = parser.add_mutually_exclusive_group()
    duration_group.add_argument(
        "--steps",
        type=int,
        default=None,
        action=_TrackProvidedAction,
        help="Number of simulation steps.",
    )
    duration_group.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_SIM_DURATION_S,
        action=_TrackProvidedAction,
        help="Simulation duration in seconds.",
    )
    parser.add_argument("--dt", type=float, default=DEFAULT_SIM_DT_S, action=_TrackProvidedAction, help="Simulation step in seconds.")
    parser.add_argument("--seed", type=int, default=7, action=_TrackProvidedAction, help="Random seed.")
    parser.add_argument(
        "--config-file",
        default=None,
        action=_TrackProvidedAction,
        help="Optional YAML or JSON simulation constants override file.",
    )
    parser.add_argument(
        "--map-preset",
        choices=sorted(MAP_PRESET_SCALES),
        default="regional",
        action=_TrackProvidedAction,
        help="Named map footprint preset.",
    )
    parser.add_argument(
        "--terrain-preset",
        choices=sorted(TERRAIN_PRESET_CHOICES),
        default="alpine",
        action=_TrackProvidedAction,
        help="Terrain family preset.",
    )
    parser.add_argument(
        "--weather-preset",
        choices=sorted(KNOWN_WEATHER_PRESETS),
        default="clear",
        action=_TrackProvidedAction,
        help="Weather preset affecting flight dynamics and observation quality.",
    )
    parser.add_argument(
        "--clean-terrain",
        action=_TrackProvidedAction,
        nargs=0,
        const=True,
        default=False,
        help="Keep terrain geometry but omit buildings, walls, and vegetation from the environment.",
    )
    parser.add_argument(
        "--platform-preset",
        choices=sorted(PLATFORM_PRESET_CHOICES),
        default="baseline",
        action=_TrackProvidedAction,
        help="Named kinematics and range preset for targets and sensors.",
    )
    parser.add_argument("--ground-stations", type=int, default=7, action=_TrackProvidedAction, help="Number of fixed ground stations to place.")
    parser.add_argument("--target-count", type=int, default=2, action=_TrackProvidedAction, help="Number of targets to simulate.")
    parser.add_argument("--drone-count", type=int, default=2, action=_TrackProvidedAction, help="Number of mobile drones to simulate.")
    parser.add_argument(
        "--target-motion",
        choices=sorted(TARGET_MOTION_PRESETS),
        default="mixed",
        action=_TrackProvidedAction,
        help="Target motion preset.",
    )
    parser.add_argument(
        "--drone-mode",
        choices=sorted(DRONE_MODE_PRESETS),
        default="mixed",
        action=_TrackProvidedAction,
        help="Mobile drone planner preset.",
    )
    parser.add_argument("--csv", dest="csv_path", default=None, action=_TrackProvidedAction, help="Optional per-track metrics CSV.")
    parser.add_argument("--replay", dest="replay_path", default="replay.json", action=_TrackProvidedAction, help="Replay JSON output.")
    return parser


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the 3D sensor fusion platform simulation.")
    add_cli_arguments(parser)
    return parser.parse_args(argv)


def run_from_args(args: argparse.Namespace) -> None:
    try:
        constants = (
            _load_simulation_constants(args.config_file)
            if getattr(args, "config_file", None)
            else _DEFAULT_CONSTANTS
        )
    except (OSError, ValueError, TypeError) as exc:
        raise SystemExit(f"Unable to load simulation config: {exc}") from exc

    steps = args.steps if _arg_was_provided(args, "steps") else None
    duration_s = (
        None
        if steps is not None
        else (
            args.duration_s
            if _arg_was_provided(args, "duration_s")
            else constants.dynamics.default_duration_s
        )
    )
    dt = args.dt if _arg_was_provided(args, "dt") else constants.dynamics.default_dt_s
    seed = args.seed if _arg_was_provided(args, "seed") else constants.dynamics.default_seed
    simulate(
        steps=steps,
        dt=dt,
        seed=seed,
        csv_path=args.csv_path,
        replay_path=args.replay_path,
        duration_s=duration_s,
        map_preset=args.map_preset,
        target_motion=args.target_motion,
        drone_mode=args.drone_mode,
        terrain_preset=args.terrain_preset,
        weather_preset=args.weather_preset,
        clean_terrain=args.clean_terrain,
        platform_preset=args.platform_preset,
        ground_stations=args.ground_stations,
        target_count=args.target_count,
        drone_count=args.drone_count,
        constants=constants,
    )


def main(argv: Optional[Sequence[str]] = None) -> None:
    run_from_args(parse_args(argv))


if __name__ == "__main__":
    main()
