"""ArgusNet.

World models for autonomous aerial mapping, inspection, localization, and spatial memory.
"""

from importlib import import_module

__all__ = [
    # core types (new canonical names)
    "SensorNodeState",
    "BearingMeasurement",
    "MeasurementRejection",
    "GroundTruthState",
    "TrackEstimate",
    "WorldModelFrame",
    "WorldModelMetrics",
    "ServiceHealthReport",
    # core types (backward-compat names)
    "NodeState",
    "BearingObservation",
    "ObservationRejection",
    "TruthState",
    "TrackState",
    "PlatformFrame",
    "PlatformMetrics",
    "HealthReport",
    "MissionZone",
    "LaunchEvent",
    "vec3",
    "to_jsonable",
    # core frames
    "ENUOrigin",
    "wgs84_to_enu",
    "enu_to_wgs84",
    # core config
    "SimulationConstants",
    # world
    "Bounds2D",
    "EnvironmentCRS",
    "EnvironmentModel",
    "LandCoverClass",
    "LandCoverLayer",
    "TerrainFeature",
    "TerrainModel",
    "KNOWN_TERRAIN_PRESETS",
    "MountainRange",
    "Valley",
    "Plateau",
    "RidgeLine",
    "terrain_model_from_preset",
    "SensorVisibilityModel",
    "VisibilityResult",
    "WeatherModel",
    "weather_from_preset",
    "load_environment_bundle",
    "write_environment_bundle",
    # sensing
    "SensorErrorConfig",
    "SensorModel",
    "sensor_error_config_from_preset",
    # simulation
    "ScenarioDefinition",
    "ScenarioOptions",
    "SimulationConfig",
    "SimulationResult",
    "build_default_scenario",
    "run_simulation",
    # adapters
    "TrackerConfig",
    "TrackingService",
    "WorldModelConfig",
    "WorldModelService",
    # planning
    "PathPlanner2D",
    "PlannerConfig",
    "PlannerRoute",
    # evaluation
    "ReplayDocument",
    "build_replay_document",
    "load_replay_document",
    "validate_replay_document",
    "write_replay_document",
    "export_geojson",
    "export_czml",
    "export_foxglove",
    # world / scene
    "build_scene_package",
    "build_scene_from_gis",
    "build_scene_from_replay",
    "load_scene_manifest",
    "validate_scene_manifest",
    # simulation / behaviors
    "build_target_trajectory",
    "FlightEnvelope",
    "BEHAVIOR_PRESETS",
]

_EXPORTS = {
    # core/types — new names
    "SensorNodeState": ("argusnet.core.types", "NodeState"),
    "BearingMeasurement": ("argusnet.core.types", "BearingObservation"),
    "MeasurementRejection": ("argusnet.core.types", "ObservationRejection"),
    "GroundTruthState": ("argusnet.core.types", "TruthState"),
    "TrackEstimate": ("argusnet.core.types", "TrackState"),
    "WorldModelFrame": ("argusnet.core.types", "PlatformFrame"),
    "WorldModelMetrics": ("argusnet.core.types", "PlatformMetrics"),
    "ServiceHealthReport": ("argusnet.core.types", "HealthReport"),
    # core/types — backward-compat names
    "NodeState": ("argusnet.core.types", "NodeState"),
    "BearingObservation": ("argusnet.core.types", "BearingObservation"),
    "ObservationRejection": ("argusnet.core.types", "ObservationRejection"),
    "TruthState": ("argusnet.core.types", "TruthState"),
    "TrackState": ("argusnet.core.types", "TrackState"),
    "PlatformFrame": ("argusnet.core.types", "PlatformFrame"),
    "PlatformMetrics": ("argusnet.core.types", "PlatformMetrics"),
    "HealthReport": ("argusnet.core.types", "HealthReport"),
    "MissionZone": ("argusnet.core.types", "MissionZone"),
    "LaunchEvent": ("argusnet.core.types", "LaunchEvent"),
    "vec3": ("argusnet.core.types", "vec3"),
    "to_jsonable": ("argusnet.core.types", "to_jsonable"),
    # core/frames
    "ENUOrigin": ("argusnet.core.frames", "ENUOrigin"),
    "wgs84_to_enu": ("argusnet.core.frames", "wgs84_to_enu"),
    "enu_to_wgs84": ("argusnet.core.frames", "enu_to_wgs84"),
    # core/config
    "SimulationConstants": ("argusnet.core.config", "SimulationConstants"),
    # world
    "Bounds2D": ("argusnet.world.environment", "Bounds2D"),
    "EnvironmentCRS": ("argusnet.world.environment", "EnvironmentCRS"),
    "EnvironmentModel": ("argusnet.world.environment", "EnvironmentModel"),
    "LandCoverClass": ("argusnet.world.environment", "LandCoverClass"),
    "LandCoverLayer": ("argusnet.world.environment", "LandCoverLayer"),
    "SensorVisibilityModel": ("argusnet.world.environment", "SensorVisibilityModel"),
    "VisibilityResult": ("argusnet.world.environment", "VisibilityResult"),
    "TerrainFeature": ("argusnet.world.terrain", "TerrainFeature"),
    "TerrainModel": ("argusnet.world.terrain", "TerrainModel"),
    "KNOWN_TERRAIN_PRESETS": ("argusnet.world.terrain", "KNOWN_TERRAIN_PRESETS"),
    "MountainRange": ("argusnet.world.terrain", "MountainRange"),
    "Valley": ("argusnet.world.terrain", "Valley"),
    "Plateau": ("argusnet.world.terrain", "Plateau"),
    "RidgeLine": ("argusnet.world.terrain", "RidgeLine"),
    "terrain_model_from_preset": ("argusnet.world.terrain", "terrain_model_from_preset"),
    "WeatherModel": ("argusnet.world.weather", "WeatherModel"),
    "weather_from_preset": ("argusnet.world.weather", "weather_from_preset"),
    "load_environment_bundle": ("argusnet.world.environment", "load_environment_bundle"),
    "write_environment_bundle": ("argusnet.world.environment", "write_environment_bundle"),
    # sensing
    "SensorErrorConfig": ("argusnet.sensing.models.noise", "SensorErrorConfig"),
    "SensorModel": ("argusnet.sensing.models.noise", "SensorModel"),
    "sensor_error_config_from_preset": (
        "argusnet.sensing.models.noise",
        "sensor_error_config_from_preset",
    ),
    # simulation
    "ScenarioDefinition": ("argusnet.simulation.sim", "ScenarioDefinition"),
    "ScenarioOptions": ("argusnet.simulation.sim", "ScenarioOptions"),
    "SimulationConfig": ("argusnet.simulation.sim", "SimulationConfig"),
    "SimulationResult": ("argusnet.simulation.sim", "SimulationResult"),
    "build_default_scenario": ("argusnet.simulation.sim", "build_default_scenario"),
    "run_simulation": ("argusnet.simulation.sim", "run_simulation"),
    # adapters
    "TrackerConfig": ("argusnet.adapters.argusnet_grpc", "TrackerConfig"),
    "TrackingService": ("argusnet.adapters.argusnet_grpc", "TrackingService"),
    "WorldModelConfig": ("argusnet.adapters.argusnet_grpc", "WorldModelConfig"),
    "WorldModelService": ("argusnet.adapters.argusnet_grpc", "WorldModelService"),
    # planning
    "PathPlanner2D": ("argusnet.planning.planner_base", "PathPlanner2D"),
    "PlannerConfig": ("argusnet.planning.planner_base", "PlannerConfig"),
    "PlannerRoute": ("argusnet.planning.planner_base", "PlannerRoute"),
    # evaluation
    "ReplayDocument": ("argusnet.evaluation.replay", "ReplayDocument"),
    "build_replay_document": ("argusnet.evaluation.replay", "build_replay_document"),
    "load_replay_document": ("argusnet.evaluation.replay", "load_replay_document"),
    "validate_replay_document": ("argusnet.evaluation.replay", "validate_replay_document"),
    "write_replay_document": ("argusnet.evaluation.replay", "write_replay_document"),
    "export_geojson": ("argusnet.evaluation.export", "export_geojson"),
    "export_czml": ("argusnet.evaluation.export", "export_czml"),
    "export_foxglove": ("argusnet.evaluation.export", "export_foxglove"),
    # world / scene
    "build_scene_package": ("argusnet.world.scene_loader", "build_scene_package"),
    "build_scene_from_gis": ("argusnet.world.scene_loader", "build_scene_from_gis"),
    "build_scene_from_replay": ("argusnet.world.scene_loader", "build_scene_from_replay"),
    "load_scene_manifest": ("argusnet.world.scene_loader", "load_scene_manifest"),
    "validate_scene_manifest": ("argusnet.world.scene_loader", "validate_scene_manifest"),
    # simulation / behaviors
    "build_target_trajectory": ("argusnet.simulation.behaviors", "build_target_trajectory"),
    "FlightEnvelope": ("argusnet.simulation.behaviors", "FlightEnvelope"),
    "BEHAVIOR_PRESETS": ("argusnet.simulation.behaviors", "BEHAVIOR_PRESETS"),
}


def __getattr__(name: str) -> object:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
