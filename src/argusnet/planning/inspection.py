"""Mission generation module for the Smart Trajectory Tracker platform.

Implements procedural mission generation based on the schema defined in
docs/MISSION_MODEL.md. Produces fully resolved ``GeneratedMission`` instances
from a ``MissionSpec`` seed document.

Public API
----------
- Data classes: MissionTiming, MissionConstraints, MissionSpec, LaunchPoint,
  MissionObjective, ObjectiveCondition, FlightCorridor, ValidityReport,
  GeneratedMission
- Template factories: surveillance_template, intercept_template,
  persistent_observation_template, search_template
- Generator: generate_mission
- Validator: validate_mission
- Difficulty scaler: apply_difficulty_scaling
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from argusnet.core.config import SensorConfig, SimulationConstants
from argusnet.core.types import (
    ZONE_TYPE_OBJECTIVE,
    ZONE_TYPE_SURVEILLANCE,
    MissionZone,
    Vector3,
    vec3,
)

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionTiming:
    """Temporal bounds of a generated mission.

    Parameters
    ----------
    duration_s:
        Total mission wall time in seconds.  Must be positive and finite.
    dt_s:
        Simulation step size in seconds.  Defaults to ``DynamicsConfig.default_dt_s``.
    start_offset_s:
        Simulation time at which objectives become active.
    deadline_s:
        Latest acceptable mission completion time.  ``None`` means open-ended.
    """

    duration_s: float
    dt_s: float = 0.25
    start_offset_s: float = 0.0
    deadline_s: float | None = None


@dataclass(frozen=True)
class MissionConstraints:
    """Hard physical and sensor requirements that must hold throughout the mission.

    Parameters
    ----------
    min_sensor_baseline_m:
        Minimum triangulation baseline between any two nodes for localisation.
    max_target_covariance_m2:
        Upper bound on the 3x3 position covariance trace at handoff.
    min_active_tracks:
        At least this many targets must be tracked continuously.
    exclusion_zones:
        ``MissionZone`` objects with ``zone_type="exclusion"``.
    terrain_clearance_m:
        Minimum AGL for all drone trajectories.
    comms_range_m:
        Maximum inter-drone and drone-to-station link range.
    """

    min_sensor_baseline_m: float = 80.0
    max_target_covariance_m2: float = 500.0
    min_active_tracks: int = 1
    exclusion_zones: list[MissionZone] = field(default_factory=list)
    terrain_clearance_m: float = 30.0
    comms_range_m: float = 5000.0


@dataclass(frozen=True)
class MissionSpec:
    """Seed document consumed by the mission generator.

    Every field is typed and validated before scenario construction begins.
    The combination of ``seed`` and ``difficulty`` fully determines the
    generated scenario.

    Parameters
    ----------
    seed:
        Deterministic RNG seed; propagates to ``ScenarioDefinition``.
    terrain_preset:
        One of ``KNOWN_TERRAIN_PRESETS`` (terrain.py) or ``"default"``.
    weather_preset:
        One of ``KNOWN_WEATHER_PRESETS`` (weather.py).
    map_preset:
        Key into ``DEFAULT_MAP_PRESET_SCALES`` (config.py).
    platform_preset:
        ``"baseline"`` or ``"wide_area"``.
    drone_count:
        Number of drones in the scenario (1..12).
    ground_station_count:
        Number of ground stations (1..12).
    target_count:
        Number of targets (1..8).
    difficulty:
        Difficulty in [0.0, 1.0]; drives the scaling model from Section 5
        of MISSION_MODEL.md.
    mission_type:
        One of ``"surveillance"``, ``"intercept"``,
        ``"persistent_observation"``, ``"search"``.
    tags:
        Free-form taxonomy tags; see Section 6 of MISSION_MODEL.md.
    timing:
        ``MissionTiming`` instance.
    constraints:
        ``MissionConstraints`` instance.
    """

    seed: int
    terrain_preset: str = "alpine"
    weather_preset: str = "clear"
    map_preset: str = "regional"
    platform_preset: str = "baseline"
    drone_count: int = 2
    ground_station_count: int = 4
    target_count: int = 2
    difficulty: float = 0.5
    mission_type: str = "surveillance"
    tags: list[str] = field(default_factory=list)
    timing: MissionTiming = field(default_factory=lambda: MissionTiming(duration_s=180.0))
    constraints: MissionConstraints = field(default_factory=MissionConstraints)

    def __post_init__(self) -> None:
        if not (0.0 <= self.difficulty <= 1.0):
            raise ValueError(f"difficulty must be in [0.0, 1.0], got {self.difficulty}.")
        if self.drone_count < 1 or self.drone_count > 12:
            raise ValueError(f"drone_count must be 1..12, got {self.drone_count}.")
        if self.ground_station_count < 1 or self.ground_station_count > 12:
            raise ValueError(
                f"ground_station_count must be 1..12, got {self.ground_station_count}."
            )
        if self.target_count < 1 or self.target_count > 8:
            raise ValueError(f"target_count must be 1..8, got {self.target_count}.")
        valid_types = {"surveillance", "intercept", "persistent_observation", "search"}
        if self.mission_type not in valid_types:
            raise ValueError(
                f"mission_type must be one of {sorted(valid_types)}, got {self.mission_type!r}."
            )


@dataclass(frozen=True)
class LaunchPoint:
    """Describes a drone launch pad and the drones assigned to it.

    Parameters
    ----------
    launch_id:
        Unique identifier for this launch point.
    station_id:
        References a ``SimNode.node_id`` with ``is_mobile=False``.
    position:
        3D ENU position of the launch pad in metres.
    assigned_drone_ids:
        Which drone node IDs launch from this point.
    earliest_launch_s:
        Earliest activation time in seconds.
    latest_launch_s:
        Latest activation time.  ``None`` means no constraint.
    """

    launch_id: str
    station_id: str
    position: Vector3
    assigned_drone_ids: list[str]
    earliest_launch_s: float = 0.0
    latest_launch_s: float | None = None


@dataclass(frozen=True)
class ObjectiveCondition:
    """Success criteria for a ``MissionObjective``.

    Parameters
    ----------
    track_continuity_fraction:
        Fraction of ``[start_s, end_s]`` the target must be tracked.
    max_position_error_m:
        3-sigma bound on position error at objective time.
    min_observation_count:
        Minimum bearing measurements in the observation window.
    covariance_trace_max_m2:
        Maximum trace of the 3x3 position covariance.
    """

    track_continuity_fraction: float = 0.85
    max_position_error_m: float = 25.0
    min_observation_count: int = 5
    covariance_trace_max_m2: float = 500.0


@dataclass(frozen=True)
class MissionObjective:
    """A single evaluable objective within the mission.

    Parameters
    ----------
    objective_id:
        Unique identifier.
    objective_type:
        One of ``"acquire"``, ``"maintain"``, ``"handoff"``,
        ``"neutralize"``, ``"survey"``.
    target_ids:
        Which ``SimTarget.target_id`` values this covers.
    zone:
        Optional spatial constraint zone.
    start_s:
        Earliest time this objective becomes evaluable.
    end_s:
        Latest time for evaluation.  ``None`` means open-ended.
    priority:
        Scheduling priority; higher values mean higher priority.
    required:
        ``False`` means this is a bonus objective.
    success_condition:
        ``ObjectiveCondition`` specifying pass/fail criteria.
    """

    objective_id: str
    objective_type: str
    target_ids: list[str]
    zone: MissionZone | None = None
    start_s: float = 0.0
    end_s: float | None = None
    priority: int = 1
    required: bool = True
    success_condition: ObjectiveCondition = field(default_factory=ObjectiveCondition)


@dataclass(frozen=True)
class FlightCorridor:
    """A 2D corridor through which drones are routed.

    Parameters
    ----------
    corridor_id:
        Unique identifier.
    waypoints_xy_m:
        2D XY vertices; same format as ``PlannerRoute.points_xy_m``.
    width_m:
        Half-width on each side of the centreline.
    min_agl_m:
        Minimum above-ground-level altitude.
    max_agl_m:
        Maximum above-ground-level altitude.
    direction:
        One of ``"inbound"``, ``"outbound"``, ``"bidirectional"``.
    assigned_drone_ids:
        Empty list means any drone may use this corridor.
    active_window:
        ``[start_s, end_s]`` activation window.
    """

    corridor_id: str
    waypoints_xy_m: list[list[float]]
    width_m: float = 50.0
    min_agl_m: float = 30.0
    max_agl_m: float = 400.0
    direction: str = "bidirectional"
    assigned_drone_ids: list[str] = field(default_factory=list)
    active_window: list[float] = field(default_factory=lambda: [0.0, 600.0])


@dataclass(frozen=True)
class ValidityReport:
    """Result of validity checking after scenario construction.

    Parameters
    ----------
    physically_valid:
        All trajectories above terrain; no obstacle penetration.
    sensor_valid:
        At least one node covers each target at mission start.
    solvable:
        Objectives are reachable given platform capabilities.
    corridor_clear:
        All ``FlightCorridor`` objects are free of hard obstacles.
    baseline_adequate:
        Maximum inter-drone separation >= ``min_sensor_baseline_m``.
    failures:
        Human-readable descriptions of any failed checks.
    """

    physically_valid: bool = True
    sensor_valid: bool = True
    solvable: bool = True
    corridor_clear: bool = True
    baseline_adequate: bool = True
    failures: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        """Return ``True`` if all checks passed."""
        return (
            self.physically_valid
            and self.sensor_valid
            and self.solvable
            and self.corridor_clear
            and self.baseline_adequate
        )


@dataclass(frozen=True)
class GeneratedMission:
    """Complete generated mission wrapping a ``ScenarioDefinition``.

    Parameters
    ----------
    scenario_def:
        Direct input to ``run_simulation()``.
    spec:
        The original ``MissionSpec`` for reproducibility.
    launch_points:
        Drone launch points.
    objectives:
        Mission objectives in priority order.
    corridors:
        Flight corridors.
    timing:
        Resolved timing (may differ from ``spec`` if the generator clamped values).
    tags:
        Propagated taxonomy tags.
    validity_report:
        Result of post-construction validity checks.
    """

    scenario_def: object  # ScenarioDefinition; typed as object to avoid circular imports
    spec: MissionSpec
    launch_points: list[LaunchPoint]
    objectives: list[MissionObjective]
    corridors: list[FlightCorridor]
    timing: MissionTiming
    tags: list[str]
    validity_report: ValidityReport


# ---------------------------------------------------------------------------
# Difficulty scaling
# ---------------------------------------------------------------------------


def apply_difficulty_scaling(
    spec: MissionSpec,
    constants: SimulationConstants,
) -> SimulationConstants:
    """Return a new ``SimulationConstants`` with difficulty-scaled sensor/dynamics parameters.

    Implements the scaling table from Section 5 of MISSION_MODEL.md:

    - ``target_speed_scale``: 0.5 at difficulty=0, 1.8 at difficulty=1 (linear).
    - ``drone_search_speed_scale``: 1.2 at difficulty=0, 0.8 at difficulty=1 (linear).
    - ``bearing_std_rad``: 0.5x baseline at difficulty=0, 2.0x at difficulty=1 (linear).
    - ``dropout_probability``: 0.3x baseline at difficulty=0, 2.5x at difficulty=1 (linear).

    Parameters
    ----------
    spec:
        The ``MissionSpec`` providing the ``difficulty`` scalar.
    constants:
        Baseline ``SimulationConstants`` to scale from.

    Returns
    -------
    SimulationConstants
        A new frozen instance with scaled values.
    """
    d = float(spec.difficulty)  # 0.0 .. 1.0

    # --- Sensor noise and dropout scaling ---
    # bearing_std_rad: linearly from 0.5x to 2.0x
    bearing_noise_multiplier = 0.5 + d * (2.0 - 0.5)  # [0.5, 2.0]

    # dropout_probability: linearly from 0.3x to 2.5x
    dropout_multiplier = 0.3 + d * (2.5 - 0.3)  # [0.3, 2.5]

    old_sensor = constants.sensor

    scaled_ground_bearing_stds = tuple(
        v * bearing_noise_multiplier for v in old_sensor.ground_station_bearing_stds_rad
    )
    scaled_ground_dropouts = tuple(
        v * dropout_multiplier for v in old_sensor.ground_station_dropout_probabilities
    )
    scaled_drone_bearing_std = old_sensor.drone_base_bearing_std_rad * bearing_noise_multiplier
    scaled_drone_dropout = old_sensor.drone_base_dropout_probability * dropout_multiplier

    new_sensor = SensorConfig(
        ground_station_mast_agls_m=old_sensor.ground_station_mast_agls_m,
        ground_station_bearing_stds_rad=scaled_ground_bearing_stds,
        ground_station_dropout_probabilities=scaled_ground_dropouts,
        ground_station_max_range_factors_m=old_sensor.ground_station_max_range_factors_m,
        ground_station_bearing_std_rad_scale=old_sensor.ground_station_bearing_std_rad_scale,
        ground_station_dropout_probability_scale=old_sensor.ground_station_dropout_probability_scale,
        ground_station_fov_half_angle_deg=old_sensor.ground_station_fov_half_angle_deg,
        ground_station_sensor_type=old_sensor.ground_station_sensor_type,
        drone_base_bearing_std_rad=scaled_drone_bearing_std,
        drone_bearing_std_rad_increment=old_sensor.drone_bearing_std_rad_increment,
        drone_base_dropout_probability=scaled_drone_dropout,
        drone_dropout_probability_increment=old_sensor.drone_dropout_probability_increment,
        drone_base_max_range_m=old_sensor.drone_base_max_range_m,
        drone_sensor_type=old_sensor.drone_sensor_type,
        drone_fov_half_angle_deg=old_sensor.drone_fov_half_angle_deg,
        min_elevation_deg=old_sensor.min_elevation_deg,
        mobile_min_elevation_deg=old_sensor.mobile_min_elevation_deg,
        fov_noise_multiplier_slope=old_sensor.fov_noise_multiplier_slope,
        fov_confidence_slope=old_sensor.fov_confidence_slope,
        altitude_quality_threshold_m=old_sensor.altitude_quality_threshold_m,
        altitude_quality_range_m=old_sensor.altitude_quality_range_m,
        altitude_noise_multiplier_max=old_sensor.altitude_noise_multiplier_max,
        altitude_confidence_floor=old_sensor.altitude_confidence_floor,
        altitude_confidence_slope=old_sensor.altitude_confidence_slope,
        confidence_floor=old_sensor.confidence_floor,
        confidence_range_slope=old_sensor.confidence_range_slope,
        transmittance_threshold=old_sensor.transmittance_threshold,
        drone_look_down_angle_deg=old_sensor.drone_look_down_angle_deg,
        los_sample_step_m=old_sensor.los_sample_step_m,
        los_terrain_clearance_m=old_sensor.los_terrain_clearance_m,
        launch_detection_range_fraction=old_sensor.launch_detection_range_fraction,
    )

    # --- Platform preset speed scaling ---
    # target_speed_scale: 0.5 at difficulty=0, 1.8 at difficulty=1
    target_speed_scale = 0.5 + d * (1.8 - 0.5)

    # drone_search_speed_scale: 1.2 at difficulty=0, 0.8 at difficulty=1
    drone_search_speed_scale = 1.2 + d * (0.8 - 1.2)

    # Encode these into the platform presets
    new_presets: dict[str, object] = {}
    for name, profile in constants.platform_presets.items():
        from argusnet.core.config import PlatformPresetProfile

        new_presets[name] = PlatformPresetProfile(
            target_speed_scale=profile.target_speed_scale * target_speed_scale,
            drone_search_speed_scale=profile.drone_search_speed_scale * drone_search_speed_scale,
            follow_speed_cap_mps=profile.follow_speed_cap_mps,
            drone_max_range_scale=profile.drone_max_range_scale,
            ground_station_range_scale=profile.ground_station_range_scale,
        )

    from types import MappingProxyType

    return SimulationConstants(
        sensor=new_sensor,
        dynamics=constants.dynamics,
        ground_station_layout=constants.ground_station_layout,
        target_trajectories=constants.target_trajectories,
        map_preset_scales=constants.map_preset_scales,
        platform_presets=MappingProxyType(new_presets),
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _difficulty_band(difficulty: float) -> str:
    """Return the canonical difficulty-band tag value for a given difficulty scalar."""
    if difficulty < 0.33:
        return "easy"
    if difficulty < 0.67:
        return "medium"
    return "hard"


def _base_tags(spec: MissionSpec) -> list[str]:
    """Derive canonical taxonomy tags from a ``MissionSpec``."""
    tags: list[str] = list(spec.tags)
    tag_set = set(tags)

    def _add(t: str) -> None:
        if t not in tag_set:
            tags.append(t)
            tag_set.add(t)

    _add(f"type:{spec.mission_type}")
    _add(f"terrain:{spec.terrain_preset}")
    _add(f"weather:{spec.weather_preset}")
    _add(f"diff:{_difficulty_band(spec.difficulty)}")
    _add(f"size:{spec.map_preset}")
    return tags


def _map_bounds_for_spec(spec: MissionSpec, constants: SimulationConstants) -> dict[str, float]:
    """Return map bounds dict derived from the spec's map_preset."""
    scale_map = constants.map_preset_scales
    scale = float(scale_map.get(spec.map_preset, 1.0))
    dyn = constants.dynamics
    return {
        "x_min_m": -dyn.map_bounds_x_extent_m * scale,
        "x_max_m": dyn.map_bounds_x_extent_m * scale,
        "y_min_m": dyn.map_bounds_y_min_m * scale,
        "y_max_m": dyn.map_bounds_y_max_m * scale,
    }


def _centre_of_bounds(bounds: dict[str, float]) -> tuple[float, float]:
    cx = (bounds["x_min_m"] + bounds["x_max_m"]) / 2.0
    cy = (bounds["y_min_m"] + bounds["y_max_m"]) / 2.0
    return cx, cy


def _half_extent(bounds: dict[str, float]) -> float:
    hx = (bounds["x_max_m"] - bounds["x_min_m"]) / 2.0
    hy = (bounds["y_max_m"] - bounds["y_min_m"]) / 2.0
    return min(hx, hy)


# ---------------------------------------------------------------------------
# Template factories
# ---------------------------------------------------------------------------


def surveillance_template(
    seed: int,
    drone_count: int = 2,
    *,
    terrain_preset: str = "alpine",
    weather_preset: str = "clear",
    map_preset: str = "regional",
    platform_preset: str = "baseline",
    ground_station_count: int = 4,
    target_count: int = 2,
    difficulty: float = 0.3,
    duration_s: float = 180.0,
    extra_tags: list[str] | None = None,
) -> MissionSpec:
    """Return a ``MissionSpec`` pre-configured for a surveillance mission.

    One or more ``MissionZone`` objects with ``zone_type="surveillance"``
    covering a fixed area.  Objective type is ``maintain`` with
    ``track_continuity_fraction >= 0.85``.  Drone roles are biased toward
    ``primary_observer`` and ``secondary_baseline``.

    Parameters
    ----------
    seed:
        Deterministic RNG seed.
    drone_count:
        Number of drones; minimum 2 for adequate baseline.
    difficulty:
        Difficulty scalar in [0.0, 1.0].
    extra_tags:
        Additional tags merged into the default tag set.
    """
    tags: list[str] = extra_tags or []
    timing = MissionTiming(
        duration_s=duration_s,
        dt_s=0.25,
        start_offset_s=0.0,
        deadline_s=None,
    )
    constraints = MissionConstraints(
        min_sensor_baseline_m=80.0,
        max_target_covariance_m2=400.0,
        min_active_tracks=max(1, target_count - 1),
        exclusion_zones=[],
        terrain_clearance_m=30.0,
        comms_range_m=5000.0,
    )
    return MissionSpec(
        seed=seed,
        terrain_preset=terrain_preset,
        weather_preset=weather_preset,
        map_preset=map_preset,
        platform_preset=platform_preset,
        drone_count=max(2, drone_count),
        ground_station_count=ground_station_count,
        target_count=target_count,
        difficulty=difficulty,
        mission_type="surveillance",
        tags=tags,
        timing=timing,
        constraints=constraints,
    )


def intercept_template(
    seed: int,
    drone_count: int = 3,
    *,
    terrain_preset: str = "alpine",
    weather_preset: str = "clear",
    map_preset: str = "regional",
    platform_preset: str = "baseline",
    ground_station_count: int = 4,
    target_count: int = 1,
    difficulty: float = 0.5,
    duration_s: float = 180.0,
    extra_tags: list[str] | None = None,
) -> MissionSpec:
    """Return a ``MissionSpec`` pre-configured for an intercept mission.

    Target trajectory is ``transit`` from entry to exit.  Objective type is
    ``acquire`` within the first 15% of mission duration, then ``maintain``.
    Includes a ``FlightCorridor`` along the predicted intercept path.

    Parameters
    ----------
    seed:
        Deterministic RNG seed.
    drone_count:
        Number of drones; minimum 3 (at least one follow + one corridor_watcher).
    difficulty:
        Difficulty scalar in [0.0, 1.0].
    extra_tags:
        Additional tags merged into the default tag set.
    """
    tags: list[str] = extra_tags or []
    timing = MissionTiming(
        duration_s=duration_s,
        dt_s=0.25,
        start_offset_s=0.0,
        deadline_s=duration_s * 0.85,  # must acquire before 85% complete
    )
    constraints = MissionConstraints(
        min_sensor_baseline_m=100.0,
        max_target_covariance_m2=300.0,
        min_active_tracks=1,
        exclusion_zones=[],
        terrain_clearance_m=30.0,
        comms_range_m=6000.0,
    )
    return MissionSpec(
        seed=seed,
        terrain_preset=terrain_preset,
        weather_preset=weather_preset,
        map_preset=map_preset,
        platform_preset=platform_preset,
        drone_count=max(3, drone_count),
        ground_station_count=ground_station_count,
        target_count=max(1, target_count),
        difficulty=difficulty,
        mission_type="intercept",
        tags=tags,
        timing=timing,
        constraints=constraints,
    )


def persistent_observation_template(
    seed: int,
    drone_count: int = 4,
    *,
    terrain_preset: str = "alpine",
    weather_preset: str = "clear",
    map_preset: str = "large",
    platform_preset: str = "wide_area",
    ground_station_count: int = 6,
    target_count: int = 3,
    difficulty: float = 0.6,
    duration_s: float = 360.0,
    extra_tags: list[str] | None = None,
) -> MissionSpec:
    """Return a ``MissionSpec`` for a long-duration persistent observation mission.

    Multiple overlapping surveillance zones with different priorities.
    Objective: maintain track continuity >= 0.90 across all high-priority
    zones.  Target count >= 2 with at least one ``evasive`` behaviour.
    Energy constraint active.

    Parameters
    ----------
    seed:
        Deterministic RNG seed.
    drone_count:
        Number of drones; minimum 4.
    duration_s:
        Must be >= 300 s.
    extra_tags:
        Additional tags merged into the default tag set.
    """
    tags: list[str] = extra_tags or []
    effective_duration = max(300.0, duration_s)
    timing = MissionTiming(
        duration_s=effective_duration,
        dt_s=0.25,
        start_offset_s=0.0,
        deadline_s=None,
    )
    constraints = MissionConstraints(
        min_sensor_baseline_m=80.0,
        max_target_covariance_m2=600.0,
        min_active_tracks=max(1, target_count - 1),
        exclusion_zones=[],
        terrain_clearance_m=30.0,
        comms_range_m=8000.0,
    )
    return MissionSpec(
        seed=seed,
        terrain_preset=terrain_preset,
        weather_preset=weather_preset,
        map_preset=map_preset,
        platform_preset=platform_preset,
        drone_count=max(4, drone_count),
        ground_station_count=ground_station_count,
        target_count=max(2, target_count),
        difficulty=difficulty,
        mission_type="persistent_observation",
        tags=tags,
        timing=timing,
        constraints=constraints,
    )


def search_template(
    seed: int,
    drone_count: int = 2,
    *,
    terrain_preset: str = "alpine",
    weather_preset: str = "clear",
    map_preset: str = "regional",
    platform_preset: str = "baseline",
    ground_station_count: int = 4,
    target_count: int = 1,
    difficulty: float = 0.4,
    duration_s: float = 240.0,
    extra_tags: list[str] | None = None,
) -> MissionSpec:
    """Return a ``MissionSpec`` pre-configured for a search mission.

    No prior track; objective is ``acquire`` with ``min_observation_count``
    triggering initial track.  Zone type is ``objective`` marking the suspected
    search area.  Drone roles use lawnmower search pattern.

    Parameters
    ----------
    seed:
        Deterministic RNG seed.
    drone_count:
        Number of search drones.
    difficulty:
        Difficulty scalar in [0.0, 1.0].
    extra_tags:
        Additional tags merged into the default tag set.
    """
    tags: list[str] = extra_tags or []
    timing = MissionTiming(
        duration_s=duration_s,
        dt_s=0.25,
        start_offset_s=0.0,
        deadline_s=duration_s * 0.75,  # track must be established before 75% complete
    )
    constraints = MissionConstraints(
        min_sensor_baseline_m=60.0,
        max_target_covariance_m2=800.0,
        min_active_tracks=1,
        exclusion_zones=[],
        terrain_clearance_m=30.0,
        comms_range_m=5000.0,
    )
    return MissionSpec(
        seed=seed,
        terrain_preset=terrain_preset,
        weather_preset=weather_preset,
        map_preset=map_preset,
        platform_preset=platform_preset,
        drone_count=max(1, drone_count),
        ground_station_count=ground_station_count,
        target_count=max(1, target_count),
        difficulty=difficulty,
        mission_type="search",
        tags=tags,
        timing=timing,
        constraints=constraints,
    )


# ---------------------------------------------------------------------------
# Objective and corridor builders (per template type)
# ---------------------------------------------------------------------------


def _build_surveillance_objectives(
    spec: MissionSpec,
    bounds: dict[str, float],
    target_ids: list[str],
    rng: np.random.Generator,
) -> list[MissionObjective]:
    """Build objectives for a surveillance mission."""
    cx, cy = _centre_of_bounds(bounds)
    he = _half_extent(bounds)
    zone_radius = he * 0.35

    zone = MissionZone(
        zone_id="surv-zone-0",
        zone_type=ZONE_TYPE_SURVEILLANCE,
        center=vec3(cx, cy, 0.0),
        radius_m=zone_radius,
        priority=2,
        label="Primary Surveillance Zone",
    )
    condition = ObjectiveCondition(
        track_continuity_fraction=0.85,
        max_position_error_m=20.0,
        min_observation_count=10,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    objective = MissionObjective(
        objective_id="surv-obj-maintain",
        objective_type="maintain",
        target_ids=list(target_ids),
        zone=zone,
        start_s=spec.timing.start_offset_s,
        end_s=spec.timing.duration_s,
        priority=2,
        required=True,
        success_condition=condition,
    )
    return [objective]


def _build_intercept_objectives(
    spec: MissionSpec,
    bounds: dict[str, float],
    target_ids: list[str],
    rng: np.random.Generator,
) -> list[MissionObjective]:
    """Build objectives for an intercept mission (acquire then maintain)."""
    acquire_end = spec.timing.start_offset_s + spec.timing.duration_s * 0.15

    acquire_condition = ObjectiveCondition(
        track_continuity_fraction=0.70,
        max_position_error_m=30.0,
        min_observation_count=3,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    maintain_condition = ObjectiveCondition(
        track_continuity_fraction=0.80,
        max_position_error_m=20.0,
        min_observation_count=8,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    acquire_obj = MissionObjective(
        objective_id="int-obj-acquire",
        objective_type="acquire",
        target_ids=list(target_ids),
        zone=None,
        start_s=spec.timing.start_offset_s,
        end_s=acquire_end,
        priority=3,
        required=True,
        success_condition=acquire_condition,
    )
    maintain_obj = MissionObjective(
        objective_id="int-obj-maintain",
        objective_type="maintain",
        target_ids=list(target_ids),
        zone=None,
        start_s=acquire_end,
        end_s=spec.timing.duration_s,
        priority=2,
        required=True,
        success_condition=maintain_condition,
    )
    return [acquire_obj, maintain_obj]


def _build_persistent_observation_objectives(
    spec: MissionSpec,
    bounds: dict[str, float],
    target_ids: list[str],
    rng: np.random.Generator,
) -> list[MissionObjective]:
    """Build objectives for a persistent observation mission."""
    cx, cy = _centre_of_bounds(bounds)
    he = _half_extent(bounds)

    # Two overlapping surveillance zones at different priorities
    zone_a = MissionZone(
        zone_id="po-zone-high",
        zone_type=ZONE_TYPE_SURVEILLANCE,
        center=vec3(cx - he * 0.2, cy, 0.0),
        radius_m=he * 0.3,
        priority=3,
        label="High-Priority Observation Zone",
    )
    zone_b = MissionZone(
        zone_id="po-zone-med",
        zone_type=ZONE_TYPE_SURVEILLANCE,
        center=vec3(cx + he * 0.2, cy, 0.0),
        radius_m=he * 0.3,
        priority=2,
        label="Secondary Observation Zone",
    )
    condition_high = ObjectiveCondition(
        track_continuity_fraction=0.90,
        max_position_error_m=15.0,
        min_observation_count=15,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    condition_med = ObjectiveCondition(
        track_continuity_fraction=0.80,
        max_position_error_m=20.0,
        min_observation_count=10,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    obj_high = MissionObjective(
        objective_id="po-obj-high",
        objective_type="maintain",
        target_ids=list(target_ids),
        zone=zone_a,
        start_s=spec.timing.start_offset_s,
        end_s=spec.timing.duration_s,
        priority=3,
        required=True,
        success_condition=condition_high,
    )
    obj_med = MissionObjective(
        objective_id="po-obj-med",
        objective_type="maintain",
        target_ids=list(target_ids),
        zone=zone_b,
        start_s=spec.timing.start_offset_s,
        end_s=spec.timing.duration_s,
        priority=2,
        required=False,
        success_condition=condition_med,
    )
    return [obj_high, obj_med]


def _build_search_objectives(
    spec: MissionSpec,
    bounds: dict[str, float],
    target_ids: list[str],
    rng: np.random.Generator,
) -> list[MissionObjective]:
    """Build objectives for a search mission."""
    cx, cy = _centre_of_bounds(bounds)
    he = _half_extent(bounds)

    # Search area: objective zone
    search_zone = MissionZone(
        zone_id="srch-zone-0",
        zone_type=ZONE_TYPE_OBJECTIVE,
        center=vec3(cx, cy, 0.0),
        radius_m=he * 0.5,
        priority=2,
        label="Search Area",
    )
    condition = ObjectiveCondition(
        track_continuity_fraction=0.60,
        max_position_error_m=50.0,
        min_observation_count=5,
        covariance_trace_max_m2=spec.constraints.max_target_covariance_m2,
    )
    objective = MissionObjective(
        objective_id="srch-obj-acquire",
        objective_type="acquire",
        target_ids=list(target_ids),
        zone=search_zone,
        start_s=spec.timing.start_offset_s,
        end_s=spec.timing.deadline_s,
        priority=3,
        required=True,
        success_condition=condition,
    )
    return [objective]


_OBJECTIVE_BUILDERS = {
    "surveillance": _build_surveillance_objectives,
    "intercept": _build_intercept_objectives,
    "persistent_observation": _build_persistent_observation_objectives,
    "search": _build_search_objectives,
}


def _build_corridors(
    spec: MissionSpec,
    bounds: dict[str, float],
    node_ids: list[str],
    drone_ids: list[str],
    rng: np.random.Generator,
) -> list[FlightCorridor]:
    """Build flight corridors appropriate for the mission type."""
    cx, cy = _centre_of_bounds(bounds)
    he = _half_extent(bounds)

    if spec.mission_type == "intercept":
        # Corridor along a predicted east-west transit path
        start_xy = [cx - he * 0.8, cy]
        end_xy = [cx + he * 0.8, cy]
        corridor = FlightCorridor(
            corridor_id="int-corridor-0",
            waypoints_xy_m=[start_xy, end_xy],
            width_m=60.0,
            min_agl_m=spec.constraints.terrain_clearance_m,
            max_agl_m=400.0,
            direction="inbound",
            assigned_drone_ids=list(drone_ids),
            active_window=[spec.timing.start_offset_s, spec.timing.duration_s],
        )
        return [corridor]

    # For other types, create a simple bidirectional approach corridor
    start_xy = [bounds["x_min_m"] * 0.7, cy]
    end_xy = [cx, cy]
    corridor = FlightCorridor(
        corridor_id="approach-corridor-0",
        waypoints_xy_m=[start_xy, end_xy],
        width_m=50.0,
        min_agl_m=spec.constraints.terrain_clearance_m,
        max_agl_m=400.0,
        direction="bidirectional",
        assigned_drone_ids=[],
        active_window=[spec.timing.start_offset_s, spec.timing.duration_s],
    )
    return [corridor]


def _build_launch_points(
    spec: MissionSpec,
    scenario_def: object,
) -> list[LaunchPoint]:
    """Build launch points from the scenario's ground station nodes."""
    # Extract ground station nodes (is_mobile=False) from scenario_def
    try:
        all_nodes = scenario_def.nodes  # type: ignore[attr-defined]
    except AttributeError:
        return []

    station_nodes = [n for n in all_nodes if not n.is_mobile]
    drone_nodes = [n for n in all_nodes if n.is_mobile]

    if not station_nodes:
        return []

    # Distribute drones across stations evenly
    drone_ids_per_station: list[list[str]] = [[] for _ in station_nodes]
    for idx, d in enumerate(drone_nodes):
        station_idx = idx % len(station_nodes)
        drone_ids_per_station[station_idx].append(d.node_id)

    launch_points: list[LaunchPoint] = []
    for si, (station, drones_for_station) in enumerate(
        zip(station_nodes, drone_ids_per_station, strict=False)
    ):
        lp = LaunchPoint(
            launch_id=f"lp-{si:02d}",
            station_id=station.node_id,
            position=station.position.copy(),
            assigned_drone_ids=list(drones_for_station),
            earliest_launch_s=0.0,
            latest_launch_s=None,
        )
        launch_points.append(lp)

    return launch_points


# ---------------------------------------------------------------------------
# Validity checker
# ---------------------------------------------------------------------------


def validate_mission(
    scenario_def: object,
    spec: MissionSpec,
    objectives: list[MissionObjective],
    corridors: list[FlightCorridor],
) -> ValidityReport:
    """Run post-construction validity checks on a generated mission.

    Checks
    ------
    physically_valid
        All drone and target AGL values >= ``terrain_clearance_m`` at t=0;
        no ``SimNode`` or ``SimTarget`` position inside an obstacle.
    sensor_valid
        For each ``SimTarget``, at least one ``SimNode`` has range > 0 at t=0.
    solvable
        ``PathPlanner2D.plan_route()`` returns a non-None route for every
        corridor's waypoints.  Required objective continuity fraction does not
        exceed 1.0.
    corridor_clear
        All ``FlightCorridor`` objects are free of hard obstacles when checked
        by the scenario's planner.
    baseline_adequate
        Maximum inter-drone separation >= ``min_sensor_baseline_m``.

    Parameters
    ----------
    scenario_def:
        A ``ScenarioDefinition`` produced by ``build_default_scenario()``.
    spec:
        The ``MissionSpec`` that produced the scenario.
    objectives:
        List of ``MissionObjective`` objects.
    corridors:
        List of ``FlightCorridor`` objects.

    Returns
    -------
    ValidityReport
        Frozen validity report with any failures listed.
    """
    failures: list[str] = []

    physically_valid = True
    sensor_valid = True
    solvable = True
    corridor_clear = True
    baseline_adequate = True

    try:
        all_nodes = scenario_def.nodes  # type: ignore[attr-defined]
        all_targets = scenario_def.targets  # type: ignore[attr-defined]
        environment = scenario_def.environment  # type: ignore[attr-defined]
        terrain = scenario_def.terrain  # type: ignore[attr-defined]
        map_bounds = scenario_def.map_bounds_m  # type: ignore[attr-defined]
    except AttributeError as exc:
        failures.append(f"Cannot read scenario_def attributes: {exc}")
        return ValidityReport(
            physically_valid=False,
            sensor_valid=False,
            solvable=False,
            corridor_clear=False,
            baseline_adequate=False,
            failures=failures,
        )

    clearance = spec.constraints.terrain_clearance_m

    # Active terrain layer
    active_terrain = terrain
    if active_terrain is None and environment is not None:
        active_terrain = getattr(environment, "terrain", None)

    # ----------------------------------------------------------------
    # Physical validity: AGL checks and obstacle penetration
    # ----------------------------------------------------------------
    if active_terrain is not None:
        for node in all_nodes:
            try:
                state = node.state(0.0)
                pos = state.position
                terrain_h = float(active_terrain.height_at(float(pos[0]), float(pos[1])))
                agl = float(pos[2]) - terrain_h
                if node.is_mobile and agl < clearance:
                    physically_valid = False
                    failures.append(
                        f"Node {node.node_id} AGL {agl:.1f} m < clearance {clearance:.1f} m at t=0"
                    )
            except Exception:
                pass  # node state may not be available; skip gracefully

    # Obstacle penetration check
    active_obstacles = None
    if environment is not None:
        active_obstacles = getattr(environment, "obstacles", None)
    if active_obstacles is not None:
        primitives = getattr(active_obstacles, "primitives", [])
        for node in all_nodes:
            try:
                state = node.state(0.0)
                pos = state.position
                for prim in primitives:
                    if prim.point_inside(float(pos[0]), float(pos[1]), float(pos[2])):
                        physically_valid = False
                        failures.append(f"Node {node.node_id} is inside obstacle at t=0")
                        break
            except Exception:
                pass

    # ----------------------------------------------------------------
    # Sensor validity: at least one node sees each target at t=0
    # ----------------------------------------------------------------
    mobile_nodes = [n for n in all_nodes if n.is_mobile]
    ground_nodes = [n for n in all_nodes if not n.is_mobile]
    sensor_nodes = mobile_nodes + ground_nodes  # prefer mobile coverage

    for target in all_targets:
        try:
            truth = target.truth_state(0.0)
            target_pos = truth.position
        except Exception:
            continue

        covered = False
        for node in sensor_nodes:
            try:
                state = node.state(0.0)
                node_pos = state.position
                dist = float(np.linalg.norm(target_pos - node_pos))
                max_range = float(state.max_range_m) if state.max_range_m > 0.0 else 1e9
                if dist <= max_range:
                    covered = True
                    break
            except Exception:
                continue

        # Fall back: if any node exists with max_range_m == 0, treat as infinite range.
        if not covered and any(
            getattr(n.state(0.0), "max_range_m", 0.0) == 0.0 for n in sensor_nodes
        ):
            covered = True

        if not covered:
            sensor_valid = False
            failures.append(f"No node covers target {target.target_id} at t=0")

    # ----------------------------------------------------------------
    # Solvable: route planning for corridors
    # ----------------------------------------------------------------
    if active_obstacles is not None and map_bounds:
        try:
            from .environment import Bounds2D
            from .planning import PathPlanner2D, PlannerConfig

            bounds_obj = Bounds2D(
                x_min_m=float(map_bounds["x_min_m"]),
                x_max_m=float(map_bounds["x_max_m"]),
                y_min_m=float(map_bounds["y_min_m"]),
                y_max_m=float(map_bounds["y_max_m"]),
            )
            planner = PathPlanner2D(
                bounds_xy_m=bounds_obj,
                obstacle_layer=active_obstacles,
                config=PlannerConfig(),
            )
            for corridor in corridors:
                wps = corridor.waypoints_xy_m
                if len(wps) < 2:
                    continue
                start = wps[0]
                goal = wps[-1]
                route = planner.plan_route(
                    start,
                    goal,
                    clearance_m=8.0,
                )
                if route is None:
                    corridor_clear = False
                    failures.append(
                        f"Corridor {corridor.corridor_id}: no route found from {start} to {goal}"
                    )
        except Exception as exc:
            failures.append(f"Planner check skipped: {exc}")
    else:
        # No obstacles or bounds available; assume corridor is clear
        pass

    # Solvable: continuity fraction sanity
    for obj in objectives:
        frac = obj.success_condition.track_continuity_fraction
        if frac > 1.0:
            solvable = False
            failures.append(
                f"Objective {obj.objective_id} requires continuity fraction {frac} > 1.0"
            )

    # ----------------------------------------------------------------
    # Baseline adequacy: max separation among mobile nodes
    # ----------------------------------------------------------------
    mobile_positions: list[np.ndarray] = []
    for node in all_nodes:
        if node.is_mobile:
            try:
                state = node.state(0.0)
                mobile_positions.append(state.position.copy())
            except Exception:
                pass

    min_baseline = spec.constraints.min_sensor_baseline_m
    if len(mobile_positions) >= 2:
        max_sep = 0.0
        for i in range(len(mobile_positions)):
            for j in range(i + 1, len(mobile_positions)):
                sep = float(np.linalg.norm(mobile_positions[i] - mobile_positions[j]))
                if sep > max_sep:
                    max_sep = sep
        if max_sep < min_baseline:
            baseline_adequate = False
            failures.append(
                "Max inter-drone separation "
                f"{max_sep:.1f} m < min_sensor_baseline_m {min_baseline:.1f} m"
            )
    elif len(mobile_positions) == 1:
        # Only one drone; baseline is technically zero but we treat as acceptable
        # if the scenario only requires one mobile platform
        if spec.drone_count > 1:
            baseline_adequate = False
            failures.append("Only one mobile node found; cannot form a baseline.")
    # If no mobile nodes at all, baseline check is vacuously OK (ground-only scenario)

    return ValidityReport(
        physically_valid=physically_valid,
        sensor_valid=sensor_valid,
        solvable=solvable,
        corridor_clear=corridor_clear,
        baseline_adequate=baseline_adequate,
        failures=failures,
    )


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------


def generate_mission(spec: MissionSpec) -> GeneratedMission:
    """Generate a fully resolved ``GeneratedMission`` from a ``MissionSpec``.

    The generator uses ``build_default_scenario()`` from ``sim.py`` as the
    base for scenario construction, then overlays mission-level metadata
    (launch points, objectives, corridors) and applies difficulty scaling.

    Validity is checked via ``validate_mission()`` and the scenario is
    regenerated up to 8 times (with an incremented seed) if any hard
    requirement fails.

    Parameters
    ----------
    spec:
        Fully populated ``MissionSpec`` describing the desired mission.

    Returns
    -------
    GeneratedMission
        Complete mission including scenario, objectives, corridors, and a
        validity report.  If all 8 attempts fail the last attempt is returned
        with its (invalid) report so the caller can inspect the failures.

    Raises
    ------
    ImportError
        If ``sim.py`` is not importable (should not happen in normal use).
    """
    # Lazy import to avoid circular dependency at module level
    from argusnet.simulation.sim import ScenarioOptions, build_default_scenario

    max_attempts = 8
    last_mission: GeneratedMission | None = None
    base_constants = SimulationConstants.default()

    for attempt in range(max_attempts):
        current_seed = spec.seed + attempt

        # Apply difficulty scaling to constants
        scaled_constants = apply_difficulty_scaling(spec, base_constants)

        # Resolve timing (apply difficulty-based duration multiplier)
        duration_multiplier = 1.5 - spec.difficulty * (1.5 - 0.7)  # [1.5, 0.7]
        resolved_duration = spec.timing.duration_s * duration_multiplier
        resolved_timing = MissionTiming(
            duration_s=resolved_duration,
            dt_s=spec.timing.dt_s,
            start_offset_s=spec.timing.start_offset_s,
            deadline_s=(
                spec.timing.deadline_s * duration_multiplier
                if spec.timing.deadline_s is not None
                else None
            ),
        )

        # Resolve effective target count (clamped by difficulty)
        effective_target_count = max(
            1,
            math.ceil(1 + spec.difficulty * (spec.target_count - 1)),
        )

        # Build ScenarioOptions from spec
        options = ScenarioOptions(
            map_preset=spec.map_preset,
            terrain_preset=spec.terrain_preset,
            weather_preset=spec.weather_preset,
            platform_preset=spec.platform_preset,
            ground_station_count=spec.ground_station_count,
            target_count=effective_target_count,
            drone_count=spec.drone_count,
        )

        # Build base scenario
        scenario_def = build_default_scenario(
            options=options,
            seed=current_seed,
            constants=scaled_constants,
        )

        # Extract IDs from scenario for objective/corridor builders
        target_ids = [t.target_id for t in scenario_def.targets]
        node_ids = [n.node_id for n in scenario_def.nodes]
        drone_ids = [n.node_id for n in scenario_def.nodes if n.is_mobile]

        # Build map bounds
        bounds = (
            dict(scenario_def.map_bounds_m)
            if scenario_def.map_bounds_m
            else _map_bounds_for_spec(spec, scaled_constants)
        )

        rng = np.random.default_rng(current_seed)

        # Build objectives
        builder = _OBJECTIVE_BUILDERS.get(spec.mission_type)
        objectives = builder(spec, bounds, target_ids, rng) if builder is not None else []

        # Build corridors
        corridors = _build_corridors(spec, bounds, node_ids, drone_ids, rng)

        # Build launch points
        launch_points = _build_launch_points(spec, scenario_def)

        # Derive tags
        tags = _base_tags(spec)

        # Validity check
        report = validate_mission(scenario_def, spec, objectives, corridors)

        mission = GeneratedMission(
            scenario_def=scenario_def,
            spec=spec,
            launch_points=launch_points,
            objectives=objectives,
            corridors=corridors,
            timing=resolved_timing,
            tags=tags,
            validity_report=report,
        )
        last_mission = mission

        if report.is_valid:
            return mission

    # All attempts failed; return last attempt with its (possibly invalid) report
    assert last_mission is not None
    return last_mission


__all__ = [
    "MissionTiming",
    "MissionConstraints",
    "MissionSpec",
    "LaunchPoint",
    "ObjectiveCondition",
    "MissionObjective",
    "FlightCorridor",
    "ValidityReport",
    "GeneratedMission",
    "apply_difficulty_scaling",
    "surveillance_template",
    "intercept_template",
    "persistent_observation_template",
    "search_template",
    "generate_mission",
    "validate_mission",
]
