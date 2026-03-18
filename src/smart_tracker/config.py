"""Configurable simulation constants extracted from sim.py.

All magic numbers that were previously hardcoded in the simulation module are
captured here as frozen dataclasses.  Each field includes a docstring-style
comment describing what it controls and its units.

Usage::

    from smart_tracker.config import SimulationConstants

    # Use defaults (identical to the original hardcoded values)
    constants = SimulationConstants.default()

    # Override specific values
    constants = SimulationConstants.from_dict({
        "sensor": {"ground_station_bearing_std_rad_scale": 1.5},
    })
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Sensor configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SensorConfig:
    """Parameters governing sensor noise, dropout, range, and FOV behaviour."""

    # -- Ground station arrays (cycled for stations beyond list length) --

    ground_station_mast_agls_m: Tuple[float, ...] = (
        2.0, 3.0, 2.5, 2.8, 2.2, 2.6, 2.4, 2.9, 2.7, 2.3, 2.8, 2.5,
    )
    """Above-ground-level heights for ground station masts, in metres."""

    ground_station_bearing_stds_rad: Tuple[float, ...] = (
        0.008, 0.010, 0.009, 0.009, 0.011, 0.010, 0.009, 0.010, 0.009, 0.011, 0.010, 0.009,
    )
    """Base bearing measurement noise standard deviation per station, in radians."""

    ground_station_dropout_probabilities: Tuple[float, ...] = (
        0.03, 0.04, 0.05, 0.035, 0.045, 0.04, 0.038, 0.042, 0.036, 0.044, 0.039, 0.041,
    )
    """Probability of an observation being dropped per step per station."""

    ground_station_max_range_factors_m: Tuple[float, ...] = (
        520.0, 520.0, 520.0, 560.0, 540.0, 550.0, 545.0, 555.0, 550.0, 535.0, 560.0, 545.0,
    )
    """Base maximum detection range per station, in metres (before platform scale)."""

    ground_station_bearing_std_rad_scale: float = 1.3
    """Multiplier applied to ground station bearing_std_rad (radar is noisier)."""

    ground_station_dropout_probability_scale: float = 0.5
    """Multiplier applied to ground station dropout_probability (radar is more reliable)."""

    ground_station_fov_half_angle_deg: float = 180.0
    """Field-of-view half-angle for ground stations, in degrees."""

    ground_station_sensor_type: str = "radar"
    """Sensor modality string used for ground stations."""

    # -- Drone sensor defaults --

    drone_base_bearing_std_rad: float = 0.014
    """Base bearing noise for drone optical sensors, in radians."""

    drone_bearing_std_rad_increment: float = 0.001
    """Per-index increment to drone bearing noise (cycles mod 4), in radians."""

    drone_base_dropout_probability: float = 0.08
    """Base dropout probability for drone sensors."""

    drone_dropout_probability_increment: float = 0.005
    """Per-index increment to drone dropout probability (cycles mod 3)."""

    drone_base_max_range_m: float = 650.0
    """Base maximum detection range for drones, in metres (before platform scale)."""

    drone_sensor_type: str = "optical"
    """Sensor modality string used for drones."""

    drone_fov_half_angle_deg: float = 55.0
    """Field-of-view half-angle for drone sensors, in degrees."""

    # -- Observation generation --

    min_elevation_deg: float = 0.5
    """Minimum elevation angle for ground-based observation acceptance, in degrees."""

    mobile_min_elevation_deg: float = -89.0
    """Minimum elevation angle for airborne (mobile) observation acceptance, in degrees."""

    fov_noise_multiplier_slope: float = 1.5
    """Linear slope for bearing noise increase as target approaches FOV edge."""

    fov_confidence_slope: float = 0.4
    """Linear slope for confidence decrease as target approaches FOV edge."""

    altitude_quality_threshold_m: float = 200.0
    """AGL threshold above which airborne sensor quality degrades, in metres."""

    altitude_quality_range_m: float = 400.0
    """AGL range over which quality degrades (200-600 m range), in metres."""

    altitude_noise_multiplier_max: float = 1.2
    """Maximum additional noise multiplier from altitude degradation."""

    altitude_confidence_floor: float = 0.3
    """Minimum confidence factor from altitude degradation."""

    altitude_confidence_slope: float = 0.5
    """Maximum confidence reduction from altitude degradation."""

    confidence_floor: float = 0.15
    """Absolute minimum observation confidence value."""

    confidence_range_slope: float = 0.7
    """Fraction of range/max_range that reduces confidence."""

    transmittance_threshold: float = 0.999
    """Transmittance threshold below which vegetation attenuation is applied."""

    drone_look_down_angle_deg: float = 30.0
    """Default downward look angle for drones deriving sensor direction from velocity, in degrees."""

    los_sample_step_m: float = 18.0
    """Sampling step for line-of-sight terrain clearance checks, in metres."""

    los_terrain_clearance_m: float = 1.0
    """Minimum clearance above terrain for LOS checks, in metres."""

    launch_detection_range_fraction: float = 0.8
    """Fraction of ground station max_range at which drone launch is triggered."""


# ---------------------------------------------------------------------------
# Flight dynamics configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DynamicsConfig:
    """Parameters governing target and interceptor flight dynamics."""

    # -- Simulation timing --

    default_duration_s: float = 180.0
    """Default simulation duration, in seconds."""

    default_dt_s: float = 0.25
    """Default simulation time step, in seconds."""

    default_seed: int = 7
    """Default random seed for reproducibility."""

    # -- Aerial target --

    aerial_target_min_agl_m: float = 120.0
    """Minimum above-ground-level altitude for aerial targets, in metres."""

    # -- Interceptor / drone --

    interceptor_search_min_agl_m: float = 180.0
    """Minimum AGL for drones in search mode, in metres."""

    interceptor_follow_min_agl_m: float = 150.0
    """Minimum AGL for drones in follow mode, in metres."""

    interceptor_follow_radius_m: float = 55.0
    """Standoff radius for follow-mode drones, in metres."""

    interceptor_follow_altitude_offset_m: float = 35.0
    """Altitude offset above target for follow-mode drones, in metres."""

    interceptor_follow_lead_s: float = 1.5
    """Time-lead for predicting target position in follow mode, in seconds."""

    interceptor_follow_candidate_count: int = 16
    """Number of orbit candidate positions evaluated in follow mode."""

    interceptor_follow_rotation_rate_rad_s: float = 0.03
    """Orbit rotation rate for follow-mode candidate generation, in rad/s."""

    # -- Tracker drone (wide-orbit localisation role) --

    tracker_standoff_radius_m: float = 120.0
    """Standoff radius for tracker-role drones; ~2× interceptor radius so
    cooperative pairs form a wider baseline for triangulation while still
    keeping targets within reliable sensor range, in metres."""

    tracker_altitude_offset_m: float = 70.0
    """Altitude offset above target for tracker-role drones.
    Chosen so the look-down angle is ~30° at the 120 m standoff radius
    (tan(30°) × 120 m ≈ 70 m), matching the default EO sensor geometry, in metres."""

    tracker_follow_min_agl_m: float = 155.0
    """Minimum AGL for tracker drones in follow mode, in metres."""

    tracker_rotation_rate_rad_s: float = 0.018
    """Orbit rotation rate for tracker drones; slower so position geometry
    stays stable for extended observation windows, in rad/s."""

    tracker_lead_s: float = 2.5
    """Time-lead for predicting target position for tracker drones, in seconds."""

    tracker_candidate_count: int = 16
    """Orbit candidate positions evaluated by tracker drones."""

    # -- Drone flight parameters --

    drone_base_agl_m: float = 230.0
    """Base above-ground-level altitude for drones, in metres."""

    drone_agl_increment_m: float = 12.0
    """Per-index altitude increment for drones (cycles mod 4), in metres."""

    drone_vertical_amplitude_base_m: float = 16.0
    """Base vertical oscillation amplitude for drones, in metres."""

    drone_vertical_amplitude_increment_m: float = 1.0
    """Per-index vertical amplitude increment for drones (cycles mod 3), in metres."""

    drone_vertical_frequency_base_rad_s: float = 0.11
    """Base vertical oscillation frequency for drones, in rad/s."""

    drone_vertical_frequency_decrement_rad_s: float = 0.015
    """Per-index vertical frequency decrement for drones (cycles mod 3), in rad/s."""

    drone_search_speed_base_mps: float = 28.0
    """Base search-mode speed for drones, in m/s."""

    drone_search_speed_decrement_mps: float = 1.0
    """Per-index search speed decrement for drones (cycles mod 3), in m/s."""

    drone_search_lane_spacing_scale: float = 80.0
    """Lane spacing multiplier for search-mode lawnmower patterns (multiplied by scale)."""

    # -- Launch controller --

    launch_climb_duration_s: float = 8.0
    """Duration of the drone launch climb phase, in seconds."""

    launch_operational_altitude_agl_m: float = 230.0
    """Default operational altitude for launched drones, in metres."""

    # -- Follow path controller defaults --

    follow_lead_s: float = 4.0
    """Default time-lead for follow-path controller, in seconds."""

    follow_candidate_count: int = 12
    """Default candidate count for follow-path controller."""

    follow_rotation_rate_rad_s: float = 0.08
    """Default orbit rotation rate for follow-path controller, in rad/s."""

    follow_min_agl_m: float = 18.0
    """Default minimum AGL for follow-path controller, in metres."""

    # -- Collision avoidance --

    collision_push_margin_m: float = 1.0
    """Margin used when pushing positions outside obstacles, in metres."""

    collision_max_iterations: int = 8
    """Maximum iterations for collision resolution."""

    ground_contact_top_pad_m: float = 3.0
    """Vertical padding added to obstacle tops for ground contact, in metres."""

    # -- Drone dynamics --

    drone_max_accel_mps2: float = 8.0
    """Maximum acceleration for drone velocity transitions, in m/s^2."""

    terrain_following_agl_m: float = 30.0
    """AGL altitude for terrain-following (nap-of-earth) drones, in metres."""

    terrain_following_smoothing_s: float = 1.5
    """Low-pass filter time constant for terrain-following altitude, in seconds."""

    # -- Perimeter waypoints --

    perimeter_inset_fraction: float = 0.18
    """Fraction of map extent used as inset for perimeter waypoints."""

    # -- Map bounds --

    map_bounds_x_extent_m: float = 360.0
    """Half-extent of default map bounds along x-axis, in metres (before scale)."""

    map_bounds_y_min_m: float = -330.0
    """Minimum y extent of default map bounds, in metres (before scale)."""

    map_bounds_y_max_m: float = 360.0
    """Maximum y extent of default map bounds, in metres (before scale)."""

    # -- Tracker --

    default_max_stale_steps: int = 6
    """Default maximum stale steps before track is dropped."""


# ---------------------------------------------------------------------------
# Platform preset profile (moved from sim.py)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PlatformPresetProfile:
    """Speed and range scaling factors for a platform configuration."""

    target_speed_scale: float
    """Multiplier for aerial target speeds."""

    drone_search_speed_scale: float
    """Multiplier for drone search-mode speeds."""

    follow_speed_cap_mps: float
    """Maximum speed for follow-mode drones, in m/s."""

    drone_max_range_scale: float
    """Multiplier for drone maximum detection range."""

    ground_station_range_scale: float
    """Multiplier for ground station maximum detection range."""


# ---------------------------------------------------------------------------
# Ground station layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GroundStationLayoutConfig:
    """Layout parameters for procedural ground station placement."""

    normalized_offsets: Tuple[Tuple[float, float], ...] = (
        (-0.72, -0.64),
        (0.70, -0.28),
        (-0.08, 0.74),
        (0.76, 0.58),
        (-0.78, -0.78),
        (0.04, -0.82),
        (-0.74, 0.16),
        (0.82, 0.10),
        (0.18, 0.86),
        (-0.42, 0.88),
        (0.88, -0.64),
        (-0.90, -0.10),
    )
    """Normalised (x, y) offsets from map centre for ground station placement."""

    overflow_angle_offset_rad: float = 0.35
    """Starting angle offset for stations beyond the normalised offset list, in radians."""

    overflow_radius_fraction: float = 0.42
    """Fraction of map half-extent used as radius for overflow station placement."""

    bounds_margin_fraction: float = 0.06
    """Fraction of minimum map span used as placement margin."""


# ---------------------------------------------------------------------------
# Target trajectory presets
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class TargetTrajectoryConfig:
    """Hardcoded trajectory parameters for the various target motion presets."""

    # -- sinusoid preset --
    sinusoid_start_positions: Tuple[Tuple[float, float], ...] = (
        (-40.0, 40.0),
        (210.0, -160.0),
    )
    """Start XY positions (before scale) for sinusoid targets."""

    sinusoid_velocities: Tuple[Tuple[float, float], ...] = (
        (4.1, 1.7),
        (-3.5, 2.6),
    )
    """XY velocities for sinusoid targets, in m/s."""

    sinusoid_lateral_axes: Tuple[int, ...] = (1, 0)
    sinusoid_lateral_amplitudes: Tuple[float, ...] = (34.0, 26.0)
    sinusoid_lateral_frequencies: Tuple[float, ...] = (0.13, 0.11)
    sinusoid_base_agls_m: Tuple[float, ...] = (135.0, 165.0)
    sinusoid_vertical_amplitudes_m: Tuple[float, ...] = (18.0, 24.0)
    sinusoid_vertical_frequencies: Tuple[float, ...] = (0.09, 0.08)

    # -- racetrack preset --
    racetrack_centers: Tuple[Tuple[float, float], ...] = (
        (-85.0, 70.0),
        (180.0, -145.0),
    )
    racetrack_straight_lengths_m: Tuple[float, ...] = (230.0, 270.0)
    racetrack_turn_radii_m: Tuple[float, ...] = (54.0, 60.0)
    racetrack_speeds_mps: Tuple[float, ...] = (7.2, 6.6)
    racetrack_base_agls_m: Tuple[float, ...] = (145.0, 175.0)
    racetrack_vertical_amplitudes_m: Tuple[float, ...] = (16.0, 22.0)
    racetrack_vertical_frequencies: Tuple[float, ...] = (0.07, 0.06)

    # -- waypoint_patrol preset --
    patrol_centers: Tuple[Tuple[float, float], ...] = (
        (-110.0, 145.0),
        (180.0, -125.0),
    )
    patrol_widths_m: Tuple[float, ...] = (220.0, 190.0)
    patrol_heights_m: Tuple[float, ...] = (165.0, 150.0)
    patrol_speeds_mps: Tuple[float, ...] = (6.8, 6.2)
    patrol_base_agls_m: Tuple[float, ...] = (155.0, 185.0)
    patrol_vertical_amplitudes_m: Tuple[float, ...] = (18.0, 24.0)
    patrol_vertical_frequencies: Tuple[float, ...] = (0.06, 0.05)


# ---------------------------------------------------------------------------
# Map preset scales
# ---------------------------------------------------------------------------

DEFAULT_MAP_PRESET_SCALES: Mapping[str, float] = MappingProxyType({
    "small": 1.0,
    "medium": 1.75,
    "large": 2.5,
    "xlarge": 4.0,
    "regional": 6.0,
    "theater": 10.0,
    "operational": 15.0,
})


# ---------------------------------------------------------------------------
# Platform presets
# ---------------------------------------------------------------------------

DEFAULT_PLATFORM_PRESETS: Mapping[str, PlatformPresetProfile] = MappingProxyType({
    "baseline": PlatformPresetProfile(
        target_speed_scale=1.0,
        drone_search_speed_scale=1.0,
        follow_speed_cap_mps=42.0,
        drone_max_range_scale=1.0,
        ground_station_range_scale=1.5,
    ),
    "wide_area": PlatformPresetProfile(
        target_speed_scale=1.35,
        drone_search_speed_scale=1.45,
        follow_speed_cap_mps=60.0,
        drone_max_range_scale=1.8,
        ground_station_range_scale=2.4,
    ),
})


# ---------------------------------------------------------------------------
# Top-level aggregator
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SimulationConstants:
    """Top-level container for every tuneable simulation constant.

    All values default to the original hardcoded values from ``sim.py`` so
    that the simulation is fully backward-compatible when no overrides are
    supplied.
    """

    sensor: SensorConfig = field(default_factory=SensorConfig)
    """Sensor noise, dropout, range, and FOV parameters."""

    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    """Flight dynamics, timing, and physical parameters."""

    ground_station_layout: GroundStationLayoutConfig = field(default_factory=GroundStationLayoutConfig)
    """Procedural ground-station placement layout."""

    target_trajectories: TargetTrajectoryConfig = field(default_factory=TargetTrajectoryConfig)
    """Preset trajectory parameters for target motion types."""

    map_preset_scales: Mapping[str, float] = field(
        default_factory=lambda: DEFAULT_MAP_PRESET_SCALES,
    )
    """Named map footprint to scale-factor mapping."""

    platform_presets: Mapping[str, PlatformPresetProfile] = field(
        default_factory=lambda: DEFAULT_PLATFORM_PRESETS,
    )
    """Named platform kinematics / range presets."""

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def default(cls) -> "SimulationConstants":
        """Return a ``SimulationConstants`` instance with all defaults."""
        return cls()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SimulationConstants":
        """Construct from a (possibly partial) nested dictionary.

        Top-level keys correspond to the sub-config names (``sensor``,
        ``dynamics``, ``ground_station_layout``, ``target_trajectories``,
        ``map_preset_scales``, ``platform_presets``).  Any missing key
        retains its default value.
        """
        kwargs: Dict[str, Any] = {}
        if "sensor" in d:
            kwargs["sensor"] = _subconfig_from_dict(SensorConfig, d["sensor"])
        if "dynamics" in d:
            kwargs["dynamics"] = _subconfig_from_dict(DynamicsConfig, d["dynamics"])
        if "ground_station_layout" in d:
            kwargs["ground_station_layout"] = _subconfig_from_dict(
                GroundStationLayoutConfig, d["ground_station_layout"],
            )
        if "target_trajectories" in d:
            kwargs["target_trajectories"] = _subconfig_from_dict(
                TargetTrajectoryConfig, d["target_trajectories"],
            )
        if "map_preset_scales" in d:
            kwargs["map_preset_scales"] = MappingProxyType(dict(d["map_preset_scales"]))
        if "platform_presets" in d:
            presets: Dict[str, PlatformPresetProfile] = {}
            for name, values in d["platform_presets"].items():
                if isinstance(values, dict):
                    presets[name] = PlatformPresetProfile(**values)
                else:
                    presets[name] = values
            kwargs["platform_presets"] = MappingProxyType(presets)
        return cls(**kwargs)

    @classmethod
    def from_yaml(cls, path: str) -> "SimulationConstants":
        """Load from a YAML file.  Falls back to JSON if PyYAML is unavailable."""
        try:
            import yaml  # type: ignore[import-untyped]
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh)
        except ImportError:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top level of {path}.")
        return cls.from_dict(data)

    @classmethod
    def from_json(cls, path: str) -> "SimulationConstants":
        """Load from a JSON file."""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, dict):
            raise ValueError(f"Expected a mapping at the top level of {path}.")
        return cls.from_dict(data)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise to a plain nested dictionary suitable for JSON/YAML."""
        # Build manually to avoid deepcopy issues with MappingProxy on Python 3.9
        raw: Dict[str, Any] = {
            "sensor": asdict(self.sensor),
            "dynamics": asdict(self.dynamics),
            "ground_station_layout": asdict(self.ground_station_layout),
            "target_trajectories": asdict(self.target_trajectories),
            "map_preset_scales": dict(self.map_preset_scales),
            "platform_presets": {
                name: asdict(profile) for name, profile in self.platform_presets.items()
            },
        }
        return raw

    def to_json(self, indent: int = 2) -> str:
        """Serialise to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def to_yaml(self) -> str:
        """Serialise to a YAML string.  Falls back to JSON if PyYAML is unavailable."""
        try:
            import yaml  # type: ignore[import-untyped]
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:
            return self.to_json()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subconfig_from_dict(cls: type, d: Dict[str, Any]) -> Any:
    """Instantiate a frozen dataclass from a (possibly partial) dict.

    Only keys that correspond to fields on *cls* are forwarded; unknown keys
    are silently ignored to be forward-compatible.  Tuple fields are coerced
    from lists.
    """
    valid_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in d.items():
        if key not in valid_names:
            continue
        # Coerce lists to tuples for tuple-typed fields
        if isinstance(value, list):
            # Check if nested list of tuples
            if value and isinstance(value[0], list):
                value = tuple(tuple(item) for item in value)
            else:
                value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)
