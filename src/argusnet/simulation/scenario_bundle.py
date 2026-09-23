"""Export a built scenario as a versioned, data-only bundle the Rust runtime runs.

`argusnet.scenario-bundle/v1` is the "documented preprocessed world bundle" of
DELIVERY_PLAN P4.3, and the bridge that lets the Rust tick run a scenario
Python built (docs/migration/STATUS.md, P4.8e.1). Everything in it is resolved
data: terrain tiles and obstacle metadata exactly as the native ABI receives
them, each node's controller as parameters, each sensor's error model, and the
targets' true trajectories as a table sampled at every time the tick asks for.

Nothing here computes simulation state. A value is either read off the built
scenario or evaluated by calling the reference's own objects, so a bundle
cannot disagree with the scenario it came from.

Unsupported scenario features are refused by name rather than approximated;
see :class:`UnsupportedScenario`.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from argusnet._native_world import _json_default
from argusnet.world.environment import SensorVisibilityModel

BUNDLE_FORMAT = "argusnet.scenario-bundle"
BUNDLE_VERSION = 1


class UnsupportedScenario(ValueError):
    """The scenario uses something bundle v1 cannot describe yet.

    Raised instead of exporting an approximation: a bundle that silently
    dropped, say, the flight envelope would run and produce a different
    mission.
    """


def _floats(values: Any) -> list[float]:
    return [float(v) for v in np.asarray(values, dtype=float).ravel()]


def motion_weather_document(weather: Any) -> dict[str, Any] | None:
    """A `WeatherModel`'s parameters: what motion reads (wind, precipitation
    drag) and what sensing reads (atmosphere, precipitation, cloud layers)."""
    if weather is None:
        return None
    wind = weather.wind
    return {
        "wind": {
            "base_speed_mps": float(wind.base_speed_mps),
            "base_heading_rad": float(wind.base_heading_rad),
            "gust_amplitude_mps": float(wind.gust_amplitude_mps),
            "gust_period_s": float(wind.gust_period_s),
            "altitude_scaling": float(wind.altitude_scaling),
        },
        "precipitation": {
            "precip_type": str(weather.precipitation.precip_type),
            "rate_mmph": float(weather.precipitation.rate_mmph),
        },
        "atmosphere": {
            "visibility_m": float(weather.atmosphere.visibility_m),
            "relative_humidity": float(weather.atmosphere.relative_humidity),
        },
        "cloud_layers": [
            {
                "base_altitude_m": float(c.base_altitude_m),
                "top_altitude_m": float(c.top_altitude_m),
                "coverage": float(c.coverage),
            }
            for c in weather.cloud_layers
        ],
    }


def _terrain_document(terrain: Any) -> dict[str, Any]:
    """The `terrain_new` payload, tile for tile."""
    return {
        "bounds": terrain.bounds_xy_m.to_metadata(),
        "tile_size_cells": int(terrain.tile_size_cells),
        "base_resolution_m": float(terrain.base_resolution_m),
        "floor_m": float(terrain.ground_plane_m),
        "tiles": [
            {
                "tx": int(t.tx),
                "ty": int(t.ty),
                "lod": int(t.lod),
                "x_min_m": float(t.x_min_m),
                "y_min_m": float(t.y_min_m),
                "cell_size_m": float(t.cell_size_m),
                "heights_m": np.asarray(t.heights_m, dtype=float).tolist(),
            }
            for t in terrain._tiles.values()
        ],
    }


def _obstacles_document(layer: Any) -> dict[str, Any]:
    """The `obstacles_new` payload, with each primitive's `obstacle_new` metadata."""
    return {
        "bounds": layer.bounds_xy_m.to_metadata(),
        "tile_size_m": float(layer.tile_size_m),
        "primitives": [
            json.loads(json.dumps(p.to_metadata(), default=_json_default)) for p in layer.primitives
        ],
    }


def _land_cover_document(layer: Any) -> dict[str, Any] | None:
    """The `land_cover_new` payload, or `None` for an environment with none."""
    if layer is None or not hasattr(layer, "_tiles"):
        return None
    return json.loads(
        json.dumps(
            {
                "bounds": layer.bounds_xy_m.to_metadata(),
                "tile_size_cells": int(layer.tile_size_cells),
                "resolution": float(layer.base_resolution_m),
                "tiles": list(layer._tiles.values()),
            },
            default=_json_default,
        )
    )


def _sensor_document(node: Any, error_config: Any, constants: Any) -> dict[str, Any]:
    """What stage 6 needs about one node's sensor.

    `error` is the node's `SensorErrorConfig` exactly as
    `_sensor_model_for_node` builds it, so the Rust stage sees the reference's
    derived knee, falloff and drift, not recomputed ones.
    """
    sensor_cfg = constants.sensor
    return {
        "sensor_type": str(node.sensor_type),
        "max_range_m": float(node.max_range_m),
        "fov_half_angle_deg": float(node.fov_half_angle_deg),
        # `min_elevation_deg` is `ObservationRequest`'s default for ground
        # stations; airborne nodes use the mobile limit.
        "required_elevation_deg": float(
            sensor_cfg.mobile_min_elevation_deg if node.is_mobile else 0.5
        ),
        "dropout_probability": float(node.dropout_probability),
        "bearing_std_rad": float(node.bearing_std_rad),
        # Every directional sensor tilts its look direction down by this
        # (`build_observations` reads it off the sensor config for any node).
        "look_down_angle_deg": float(sensor_cfg.drone_look_down_angle_deg),
        "error": {k: float(v) for k, v in asdict(error_config).items()},
    }


def _follow_config_document(follow: Any) -> dict[str, Any]:
    if follow.planner is not None:
        raise UnsupportedScenario(
            "a follow controller with a route planner (bundle v1 has no planner adapter)"
        )
    bounds = follow.map_bounds_m
    return {
        "map_bounds_m": {k: float(bounds[k]) for k in ("x_min_m", "x_max_m", "y_min_m", "y_max_m")},
        "base_agl_m": float(follow.base_agl_m),
        "standoff_radius_m": float(follow.standoff_radius_m),
        "map_scale": float(follow.map_scale),
        "max_speed_mps": float(follow.max_speed_mps),
        "phase": float(follow.phase),
        "lead_s": float(follow.lead_s),
        "candidate_count": int(follow.candidate_count),
        "rotation_rate_rad_s": float(follow.rotation_rate_rad_s),
        "min_agl_m": float(follow.min_agl_m),
        "vertical_amplitude_m": float(follow.vertical_amplitude_m),
        "vertical_frequency_rad_s": float(follow.vertical_frequency_rad_s),
        "target_altitude_offset_m": (
            None
            if follow.target_altitude_offset_m is None
            else float(follow.target_altitude_offset_m)
        ),
        "max_accel_mps2": None if follow.max_accel_mps2 is None else float(follow.max_accel_mps2),
        "terrain_following": bool(follow.terrain_following),
        "terrain_following_agl_m": float(follow.terrain_following_agl_m),
        "terrain_following_smoothing_s": float(follow.terrain_following_smoothing_s),
        "slot_index": int(follow.slot_index),
        "slot_count": int(follow.slot_count),
        "mission_zones": [_zone_document(z) for z in follow.mission_zones],
    }


def _zone_document(zone: Any) -> dict[str, Any]:
    return {
        "zone_id": str(zone.zone_id),
        "zone_type": str(zone.zone_type),
        "center_m": _floats(zone.center),
        "radius_m": float(zone.radius_m),
        "priority": int(zone.priority),
        "label": str(zone.label),
    }


def _search_document(trajectory: Any) -> dict[str, Any]:
    spec = getattr(trajectory, "bundle_spec", None)
    if not spec or spec.get("kind") != "looped_waypoint_path":
        raise UnsupportedScenario(
            f"a search trajectory of type {type(trajectory).__name__} "
            "(bundle v1 describes looped waypoint paths only)"
        )
    return {key: value for key, value in spec.items() if key != "kind"}


def _operational_document(trajectory: Any, target_ids_by_trajectory: dict[int, str]) -> dict:
    from argusnet.simulation.sim import (
        FollowPathController,
        ObservationTriggeredFollowController,
    )

    def target_of(follow: FollowPathController) -> str:
        try:
            return target_ids_by_trajectory[id(follow.target_trajectory)]
        except KeyError as error:
            raise UnsupportedScenario(
                "a follow controller whose target is not one of the scenario's targets"
            ) from error

    if isinstance(trajectory, ObservationTriggeredFollowController):
        return {
            "kind": "search_then_follow",
            "preferred_target_id": str(trajectory.preferred_target_id),
            "search": _search_document(trajectory.search_trajectory),
            "follow": _follow_config_document(trajectory.follow_trajectory),
            "follow_target_id": target_of(trajectory.follow_trajectory),
        }
    if isinstance(trajectory, FollowPathController):
        return {
            "kind": "follow",
            "target_id": target_of(trajectory),
            "follow": _follow_config_document(trajectory),
        }
    return {"kind": "search", "search": _search_document(trajectory)}


def _controller_document(
    node: Any, scenario: Any, target_ids_by_trajectory: dict[int, str]
) -> dict[str, Any]:
    from argusnet.simulation.sim import LaunchableTrajectoryController

    trajectory = node.trajectory
    if type(trajectory).__name__ == "EnvelopeLimitedTrajectory":
        raise UnsupportedScenario("enforce_flight_envelope (EnvelopeLimitedTrajectory)")
    spec = getattr(trajectory, "bundle_spec", None)
    if spec and spec.get("kind") == "static_ground_point":
        position, _ = trajectory(0.0)
        return {"kind": "static", "position_m": _floats(position)}
    if isinstance(trajectory, LaunchableTrajectoryController):
        return {
            "kind": "launched",
            "station_position_m": _floats(trajectory.station_position),
            "climb_duration_s": float(trajectory.climb_duration_s),
            "operational_altitude_agl_m": float(trajectory.operational_altitude_agl),
            "min_operational_agl_m": float(trajectory.min_operational_agl_m),
            "wind_scale": float(trajectory.wind_scale),
            # `run_simulation` launches every launchable drone at t=0 against
            # its assigned target, after resetting runtime state.
            "auto_launch_at_s": 0.0,
            "launch_target_id": str(scenario.drone_target_assignments.get(node.node_id, "")),
            "operational": _operational_document(
                trajectory.operational_trajectory, target_ids_by_trajectory
            ),
        }
    raise UnsupportedScenario(f"node {node.node_id}'s trajectory {type(trajectory).__name__}")


def _target_sample_times(simulation_config: Any, leads: set[float]) -> list[float]:
    """Every time the tick will ask a target for its state.

    Stage 6 asks at each tick's timestamp, and each follow controller also at
    `timestamp + lead_s` -- computed here with the same single addition both
    implementations perform, so the Rust lookup keys match bit for bit.
    """
    times: set[float] = set()
    for step in range(int(simulation_config.steps)):
        timestamp_s = step * simulation_config.dt_s
        times.add(timestamp_s)
        for lead_s in leads:
            times.add(timestamp_s + lead_s)
    return sorted(times)


def _refuse_unsupported_options(options: Any) -> None:
    refusals = {
        "enforce_flight_envelope": "the flight envelope wrapper",
        "safety_blocking": "the blocking safety monitor (stage 4 assembly is P4.8e.3)",
        "frontier_exploration": "frontier redirects (P4.8e.3)",
        "occlusion_aware_mapping": "occlusion-aware mapping (P4.8e.3)",
        "adaptive_search": "adaptive cooperative search (P4.8e.3)",
    }
    for flag, what in refusals.items():
        if getattr(options, flag, False):
            raise UnsupportedScenario(f"{flag}: {what}")
    if options.mission_mode != "target_tracking":
        raise UnsupportedScenario(
            f"mission_mode {options.mission_mode!r}: bundle v1 assembles target tracking; "
            "scan_map_inspect needs the mission driver (P4.8e.3)"
        )


def export_bundle(scenario: Any, simulation_config: Any, tracker_config: Any = None) -> dict:
    """Describe `scenario` as bundle v1 data, or raise :class:`UnsupportedScenario`."""
    from argusnet.adapters.argusnet_grpc import TrackerConfig
    from argusnet.simulation.sim import (
        FollowPathController,
        LaunchableTrajectoryController,
        ObservationTriggeredFollowController,
        _build_sensor_models,
    )

    if getattr(scenario, "use_vio_position", False):
        raise UnsupportedScenario("use_vio_position (VIO position substitution)")
    _refuse_unsupported_options(scenario.options)

    target_ids_by_trajectory = {id(t.trajectory): t.target_id for t in scenario.targets}
    sensor_models = _build_sensor_models(scenario.nodes, seed=simulation_config.seed)

    leads: set[float] = set()
    for node in scenario.nodes:
        trajectory = node.trajectory
        if isinstance(trajectory, LaunchableTrajectoryController):
            trajectory = trajectory.operational_trajectory
        if isinstance(trajectory, ObservationTriggeredFollowController):
            trajectory = trajectory.follow_trajectory
        if isinstance(trajectory, FollowPathController):
            leads.add(float(trajectory.lead_s))

    times = _target_sample_times(simulation_config, leads)
    targets = []
    for target in scenario.targets:
        samples = []
        for t in times:
            position, velocity = target.trajectory(t)
            samples.append(
                {"t": float(t), "position": _floats(position), "velocity": _floats(velocity)}
            )
        targets.append({"target_id": target.target_id, "samples": samples})

    environment = scenario.environment
    return {
        "format": BUNDLE_FORMAT,
        "version": BUNDLE_VERSION,
        "scenario_name": scenario.scenario_name,
        "mission_mode": scenario.options.mission_mode,
        "seed": int(simulation_config.seed),
        "dt_s": float(simulation_config.dt_s),
        "steps": int(simulation_config.steps),
        "map_bounds_m": {k: float(v) for k, v in scenario.map_bounds_m.items()},
        "world": {
            "terrain": _terrain_document(scenario.terrain),
            "obstacles": _obstacles_document(environment.obstacles),
            "land_cover": _land_cover_document(getattr(environment, "land_cover", None)),
            "visibility_profile": json.loads(
                json.dumps(asdict(SensorVisibilityModel.optical_default()), default=_json_default)
            ),
            "los_terrain_clearance_m": float(scenario.constants.sensor.los_terrain_clearance_m),
        },
        "weather": motion_weather_document(scenario.weather),
        "tracker": asdict(tracker_config or TrackerConfig()),
        "drone_roles": dict(scenario.drone_roles),
        "mission_zones": [_zone_document(z) for z in scenario.mission_zones],
        "nodes": [
            {
                "node_id": node.node_id,
                "is_mobile": bool(node.is_mobile),
                "health": float(1.0 - node.dropout_probability * 0.5),
                "sensor": _sensor_document(
                    node, sensor_models[node.node_id].config, scenario.constants
                ),
                "controller": _controller_document(node, scenario, target_ids_by_trajectory),
            }
            for node in scenario.nodes
        ],
        "targets": targets,
    }


def write_bundle(path: str | Path, bundle: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # `allow_nan=False`: a NaN in a bundle is a bug in the scenario, and JSON
    # has no spelling for it that Rust would read back as the same value.
    path.write_text(json.dumps(bundle, allow_nan=False, separators=(",", ":")))
    return path


__all__ = [
    "BUNDLE_FORMAT",
    "BUNDLE_VERSION",
    "UnsupportedScenario",
    "export_bundle",
    "motion_weather_document",
    "write_bundle",
]
