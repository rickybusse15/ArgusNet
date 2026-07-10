"""Mission-profile contracts for civilian domain workflows.

Profiles are declarative presets.  They compose existing simulation and
planning primitives without implementing domain-specific execution logic.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

from argusnet.core.types import (
    ZONE_TYPE_EXCLUSION,
    ZONE_TYPE_OBJECTIVE,
    InspectionPOI,
    MissionZone,
    vec3,
)

SUPPORTED_SENSOR_MODALITIES = frozenset({"optical", "thermal", "any"})
SUPPORTED_OBJECTIVE_KINDS = frozenset(
    {"search", "acquire", "confirm", "map", "inspect", "egress"}
)


class MissionDomain(StrEnum):
    SEARCH_AND_RESCUE = "search_and_rescue"
    INDUSTRIAL_SURVEY = "industrial_survey"


@dataclass(frozen=True)
class MissionObjectiveProfile:
    """Declarative completion contract for one profile objective."""

    objective_id: str
    kind: str
    priority: int = 1
    required_modality: str = "any"
    deadline_s: float | None = None
    dwell_s: float | None = None
    coverage_threshold: float | None = None
    localization_confidence: float | None = None
    required: bool = True

    def __post_init__(self) -> None:
        if not self.objective_id.strip():
            raise ValueError("objective_id must be non-empty.")
        if self.kind not in SUPPORTED_OBJECTIVE_KINDS:
            raise ValueError(f"Unsupported objective kind {self.kind!r}.")
        if self.required_modality not in SUPPORTED_SENSOR_MODALITIES:
            raise ValueError(f"Unsupported sensor modality {self.required_modality!r}.")
        if self.priority < 1:
            raise ValueError("priority must be at least 1.")
        for name in ("deadline_s", "dwell_s"):
            value = getattr(self, name)
            if value is not None and value <= 0.0:
                raise ValueError(f"{name} must be positive when provided.")
        for name in ("coverage_threshold", "localization_confidence"):
            value = getattr(self, name)
            if value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1].")


@dataclass(frozen=True)
class MissionSafetyProfile:
    terrain_clearance_m: float
    max_altitude_m: float
    battery_reserve_fraction: float
    require_safe_egress: bool = True

    def __post_init__(self) -> None:
        if self.terrain_clearance_m <= 0.0:
            raise ValueError("terrain_clearance_m must be positive.")
        if self.max_altitude_m <= self.terrain_clearance_m:
            raise ValueError("max_altitude_m must exceed terrain_clearance_m.")
        if not 0.0 <= self.battery_reserve_fraction <= 1.0:
            raise ValueError("battery_reserve_fraction must be in [0, 1].")


@dataclass(frozen=True)
class MissionProfile:
    """Immutable profile for a supported civilian mission workflow."""

    profile_id: str
    domain: MissionDomain
    workflow: str
    description: str
    mission_mode: str
    required_sensor_modalities: tuple[str, ...]
    target_semantics: str | None
    poi_semantics: str | None
    objectives: tuple[MissionObjectiveProfile, ...]
    safety: MissionSafetyProfile
    scenario_defaults: Mapping[str, object] = field(default_factory=dict)
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.profile_id.strip() or not self.workflow.strip():
            raise ValueError("profile_id and workflow must be non-empty.")
        if self.mission_mode not in {"target_tracking", "scan_map_inspect"}:
            raise ValueError(f"Unsupported mission mode {self.mission_mode!r}.")
        modalities = tuple(dict.fromkeys(self.required_sensor_modalities))
        if not modalities or any(m not in SUPPORTED_SENSOR_MODALITIES for m in modalities):
            raise ValueError("required_sensor_modalities contains an unsupported modality.")
        if not self.objectives:
            raise ValueError("A mission profile requires at least one objective.")
        if self.mission_mode == "target_tracking" and not self.target_semantics:
            raise ValueError("target_tracking profiles require target_semantics.")
        if self.mission_mode == "scan_map_inspect" and not self.poi_semantics:
            raise ValueError("scan_map_inspect profiles require poi_semantics.")
        if self.mission_mode == "target_tracking" and self.poi_semantics:
            raise ValueError("target_tracking profiles cannot require POI semantics.")
        if self.mission_mode == "scan_map_inspect" and self.target_semantics:
            raise ValueError("scan_map_inspect profiles cannot require target semantics.")
        defaults = dict(self.scenario_defaults)
        if defaults.get("mission_mode", self.mission_mode) != self.mission_mode:
            raise ValueError("scenario_defaults mission_mode contradicts the profile.")
        object.__setattr__(self, "required_sensor_modalities", modalities)
        object.__setattr__(self, "objectives", tuple(self.objectives))
        object.__setattr__(self, "scenario_defaults", MappingProxyType(defaults))
        object.__setattr__(self, "tags", tuple(dict.fromkeys(self.tags)))


SAR_PERSON_SEARCH = MissionProfile(
    profile_id="sar_person_search",
    domain=MissionDomain.SEARCH_AND_RESCUE,
    workflow="person_search_and_localization",
    description="Locate and confirm a missing person in an authorized search area.",
    mission_mode="target_tracking",
    required_sensor_modalities=("optical", "thermal"),
    target_semantics="missing_person",
    poi_semantics=None,
    objectives=(
        MissionObjectiveProfile(
            "search-area",
            "search",
            priority=3,
            required_modality="any",
            coverage_threshold=0.60,
        ),
        MissionObjectiveProfile(
            "acquire-person",
            "acquire",
            priority=4,
            required_modality="thermal",
            deadline_s=90.0,
            localization_confidence=0.70,
        ),
        MissionObjectiveProfile(
            "confirm-person",
            "confirm",
            priority=4,
            required_modality="optical",
            dwell_s=8.0,
            localization_confidence=0.80,
        ),
    ),
    safety=MissionSafetyProfile(
        terrain_clearance_m=30.0,
        max_altitude_m=400.0,
        battery_reserve_fraction=0.20,
    ),
    scenario_defaults={
        "mission_mode": "target_tracking",
        "map_preset": "medium",
        "terrain_preset": "alpine",
        "weather_preset": "clear",
        "platform_preset": "baseline",
        "target_motion_preset": "loiter",
        "drone_mode_preset": "search",
        "ground_station_count": 4,
        "target_count": 1,
        "drone_count": 3,
        "safety_blocking": True,
    },
    tags=("type:search", "domain:search_and_rescue", "workflow:person_search"),
)


INDUSTRIAL_ASSET_SURVEY = MissionProfile(
    profile_id="industrial_asset_survey",
    domain=MissionDomain.INDUSTRIAL_SURVEY,
    workflow="fixed_asset_mapping_and_inspection",
    description="Map and inspect fixed assets at an authorized industrial site.",
    mission_mode="scan_map_inspect",
    required_sensor_modalities=("optical", "thermal"),
    target_semantics=None,
    poi_semantics="fixed_industrial_asset",
    objectives=(
        MissionObjectiveProfile(
            "map-site",
            "map",
            priority=3,
            required_modality="optical",
            coverage_threshold=0.75,
        ),
        MissionObjectiveProfile(
            "inspect-assets",
            "inspect",
            priority=3,
            required_modality="any",
            dwell_s=15.0,
            localization_confidence=0.75,
        ),
        MissionObjectiveProfile("safe-egress", "egress", priority=4),
    ),
    safety=MissionSafetyProfile(
        terrain_clearance_m=35.0,
        max_altitude_m=300.0,
        battery_reserve_fraction=0.25,
    ),
    scenario_defaults={
        "mission_mode": "scan_map_inspect",
        "map_preset": "medium",
        "terrain_preset": "urban_flat",
        "weather_preset": "clear",
        "platform_preset": "baseline",
        "target_count": 0,
        "drone_mode_preset": "inspect",
        "ground_station_count": 3,
        "drone_count": 2,
        "scan_coverage_threshold": 0.75,
        "poi_count": 3,
        "safety_blocking": True,
    },
    tags=("type:inspection", "domain:industrial_survey", "workflow:asset_survey"),
)


_PROFILES = MappingProxyType(
    {
        profile.profile_id: profile
        for profile in (SAR_PERSON_SEARCH, INDUSTRIAL_ASSET_SURVEY)
    }
)

_SCENARIO_OPTION_FIELDS = frozenset(
    {
        "map_preset",
        "target_motion_preset",
        "drone_mode_preset",
        "terrain_source",
        "terrain_preset",
        "terrain_seed",
        "dem_path",
        "dem_crs",
        "detail_strength",
        "terrain_resolution_m",
        "season_month",
        "weather_preset",
        "clean_terrain",
        "platform_preset",
        "ground_station_count",
        "target_count",
        "drone_count",
        "mission_mode",
        "scan_coverage_threshold",
        "poi_count",
        "enforce_flight_envelope",
        "safety_blocking",
        "frontier_exploration",
    }
)


def list_mission_profiles(domain: MissionDomain | str | None = None) -> tuple[MissionProfile, ...]:
    active_domain = MissionDomain(domain) if domain is not None else None
    return tuple(
        profile for profile in _PROFILES.values()
        if active_domain is None or profile.domain == active_domain
    )


def get_mission_profile(profile_id: str) -> MissionProfile:
    try:
        return _PROFILES[profile_id]
    except KeyError as exc:
        known = ", ".join(_PROFILES)
        raise KeyError(f"Unknown mission profile {profile_id!r}. Known profiles: {known}") from exc


def get_profile(profile_id: str) -> MissionProfile:
    """Backward-friendly short alias for :func:`get_mission_profile`."""
    return get_mission_profile(profile_id)


def list_profiles(domain: MissionDomain | str | None = None) -> tuple[MissionProfile, ...]:
    """Backward-friendly short alias for :func:`list_mission_profiles`."""
    return list_mission_profiles(domain)


def _validated_overrides(
    profile: MissionProfile, overrides: Mapping[str, object] | None
) -> dict[str, object]:
    values = dict(profile.scenario_defaults)
    supplied = dict(overrides or {})
    unknown = sorted(set(supplied) - _SCENARIO_OPTION_FIELDS)
    if unknown:
        raise ValueError(f"Unsupported profile override(s): {', '.join(unknown)}.")
    values.update(supplied)
    if values["mission_mode"] != profile.mission_mode:
        raise ValueError(f"{profile.profile_id} requires mission_mode={profile.mission_mode!r}.")
    target_count = int(values.get("target_count", 0))
    if profile.mission_mode == "target_tracking" and target_count < 1:
        raise ValueError(f"{profile.profile_id} requires at least one target.")
    if profile.mission_mode == "scan_map_inspect" and target_count != 0:
        raise ValueError(f"{profile.profile_id} models fixed POIs and requires target_count=0.")
    if profile.mission_mode == "scan_map_inspect" and int(values.get("poi_count", 0)) < 1:
        raise ValueError(f"{profile.profile_id} requires at least one inspection POI.")
    return values


def build_profile_scenario_options(
    profile_id: str,
    overrides: Mapping[str, object] | None = None,
) -> Any:
    """Compile a profile into the simulator's existing ScenarioOptions type."""
    from argusnet.simulation.sim import ScenarioOptions

    profile = get_mission_profile(profile_id)
    values = _validated_overrides(profile, overrides)
    return ScenarioOptions(
        **values,
        mission_profile_id=profile.profile_id,
        mission_domain=profile.domain.value,
        mission_workflow=profile.workflow,
        mission_tags=profile.tags,
    )


def build_profile_mission_spec(profile_id: str, *, seed: int = 7) -> Any:
    """Compile a profile into the existing mission-generator seed contract."""
    from argusnet.planning.inspection import MissionConstraints, MissionSpec, MissionTiming

    profile = get_mission_profile(profile_id)
    defaults = profile.scenario_defaults
    deadline = next(
        (objective.deadline_s for objective in profile.objectives if objective.deadline_s),
        None,
    )
    # The current MissionSpec vocabulary predates domain profiles.  Industrial
    # survey maps to its observation template while POIs remain profile-owned.
    mission_type = "search" if profile.domain == MissionDomain.SEARCH_AND_RESCUE else "surveillance"
    return MissionSpec(
        seed=seed,
        terrain_preset=str(defaults["terrain_preset"]),
        weather_preset=str(defaults["weather_preset"]),
        map_preset=str(defaults["map_preset"]),
        platform_preset=str(defaults["platform_preset"]),
        drone_count=int(defaults["drone_count"]),
        ground_station_count=int(defaults["ground_station_count"]),
        target_count=max(1, int(defaults["target_count"])),
        difficulty=0.5,
        mission_type=mission_type,
        tags=list(profile.tags),
        timing=MissionTiming(duration_s=180.0, deadline_s=deadline),
        constraints=MissionConstraints(
            terrain_clearance_m=profile.safety.terrain_clearance_m,
            min_active_tracks=1 if profile.target_semantics else 0,
        ),
    )


def build_profile_mission_zones(
    profile_id: str, *, map_extent_m: float = 1000.0
) -> tuple[MissionZone, ...]:
    """Return deterministic domain zones in local metre coordinates."""
    profile = get_mission_profile(profile_id)
    if map_extent_m <= 0.0:
        raise ValueError("map_extent_m must be positive.")
    radius = map_extent_m * 0.28
    zones = [
        MissionZone(
            zone_id=f"{profile.profile_id}-objective",
            zone_type=ZONE_TYPE_OBJECTIVE,
            center=vec3(0.0, 0.0, 0.0),
            radius_m=radius,
            priority=3,
            label="Person Search Area" if profile.target_semantics else "Industrial Survey Area",
        )
    ]
    if profile.domain == MissionDomain.INDUSTRIAL_SURVEY:
        zones.append(
            MissionZone(
                zone_id="industrial-asset-survey-exclusion",
                zone_type=ZONE_TYPE_EXCLUSION,
                center=vec3(-map_extent_m * 0.32, map_extent_m * 0.25, 0.0),
                radius_m=map_extent_m * 0.06,
                priority=4,
                label="Authorized Exclusion Area",
            )
        )
    return tuple(zones)


def build_profile_inspection_pois(
    profile_id: str, *, map_extent_m: float = 1000.0
) -> tuple[InspectionPOI, ...]:
    profile = get_mission_profile(profile_id)
    if profile.poi_semantics is None:
        return ()
    if map_extent_m <= 0.0:
        raise ValueError("map_extent_m must be positive.")
    dwell = next(
        objective.dwell_s for objective in profile.objectives
        if objective.kind == "inspect" and objective.dwell_s is not None
    )
    specs: Sequence[tuple[str, float, float, str]] = (
        ("asset-intake", -0.20, -0.08, "optical"),
        ("asset-process", 0.02, 0.14, "thermal"),
        ("asset-storage", 0.22, -0.12, "any"),
    )
    return tuple(
        InspectionPOI(
            poi_id=poi_id,
            name=name.replace("-", " ").title(),
            position=vec3(x * map_extent_m, y * map_extent_m, 0.0),
            priority=3,
            required_dwell_s=dwell,
            sensor_modality=modality,
        )
        for name, x, y, modality in specs
        for poi_id in (f"{profile.profile_id}-{name}",)
    )


def build_profile_benchmark_config(profile_id: str, *, seed: int = 7) -> Any:
    from argusnet.evaluation.benchmarks import BenchmarkConfig

    profile = get_mission_profile(profile_id)
    defaults = profile.scenario_defaults
    return BenchmarkConfig(
        name=profile.profile_id,
        mission_type="search" if profile.target_semantics else "inspection",
        difficulty=0.5,
        seeds=[seed],
        duration_s=90.0 if profile.target_semantics else 120.0,
        tags=["mission-profile", *profile.tags],
        family="civilian-domain-profiles-v1",
        map_preset=str(defaults["map_preset"]),
        terrain_preset=str(defaults["terrain_preset"]),
        drone_count=int(defaults["drone_count"]),
        sim_args={"mission_profile": profile.profile_id},
    )
