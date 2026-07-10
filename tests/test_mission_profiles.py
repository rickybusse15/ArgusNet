from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from argusnet.mission import (
    MissionDomain,
    MissionObjectiveProfile,
    MissionProfile,
    MissionSafetyProfile,
    build_profile_benchmark_config,
    build_profile_inspection_pois,
    build_profile_mission_spec,
    build_profile_mission_zones,
    build_profile_scenario_options,
    get_profile,
    list_profiles,
)
from argusnet.simulation.sim import (
    SimulationConfig,
    build_default_scenario,
    build_replay_document_from_result,
    parse_args,
    run_from_args,
    run_simulation,
)


def test_registry_lookup_filtering_and_immutability() -> None:
    sar = get_profile("sar_person_search")
    assert sar.domain is MissionDomain.SEARCH_AND_RESCUE
    assert list_profiles(MissionDomain.INDUSTRIAL_SURVEY) == (
        get_profile("industrial_asset_survey"),
    )
    with pytest.raises(FrozenInstanceError):
        sar.workflow = "changed"  # type: ignore[misc]
    with pytest.raises(TypeError):
        sar.scenario_defaults["target_count"] = 99  # type: ignore[index]
    with pytest.raises(KeyError, match="Unknown mission profile"):
        get_profile("missing")


def test_contract_validation_rejects_bad_threshold_and_contradiction() -> None:
    with pytest.raises(ValueError, match="coverage_threshold"):
        MissionObjectiveProfile("bad", "search", coverage_threshold=1.1)
    with pytest.raises(ValueError, match="POI semantics"):
        MissionProfile(
            profile_id="bad",
            domain=MissionDomain.SEARCH_AND_RESCUE,
            workflow="bad",
            description="bad",
            mission_mode="target_tracking",
            required_sensor_modalities=("optical",),
            target_semantics="person",
            poi_semantics="asset",
            objectives=(MissionObjectiveProfile("search", "search"),),
            safety=MissionSafetyProfile(30.0, 100.0, 0.2),
        )


def test_sar_compiles_to_existing_primitives_deterministically() -> None:
    options = build_profile_scenario_options("sar_person_search")
    assert options.mission_mode == "target_tracking"
    assert options.target_count == 1
    assert options.target_motion_preset == "loiter"
    assert options.safety_blocking
    assert build_profile_inspection_pois("sar_person_search") == ()
    first_zones = build_profile_mission_zones("sar_person_search")
    second_zones = build_profile_mission_zones("sar_person_search")
    assert [zone.zone_id for zone in first_zones] == [zone.zone_id for zone in second_zones]
    assert [zone.center.tolist() for zone in first_zones] == [
        zone.center.tolist() for zone in second_zones
    ]
    spec = build_profile_mission_spec("sar_person_search", seed=17)
    assert spec.seed == 17
    assert spec.mission_type == "search"
    benchmark = build_profile_benchmark_config("sar_person_search", seed=17)
    assert benchmark.seeds == [17]
    assert benchmark.family == "civilian-domain-profiles-v1"


def test_industrial_compiles_fixed_pois_and_exclusion_zone() -> None:
    options = build_profile_scenario_options("industrial_asset_survey")
    assert options.mission_mode == "scan_map_inspect"
    assert options.target_count == 0
    assert options.scan_coverage_threshold == 0.75
    pois = build_profile_inspection_pois("industrial_asset_survey")
    assert [poi.sensor_modality for poi in pois] == ["optical", "thermal", "any"]
    assert len(build_profile_mission_zones("industrial_asset_survey")) == 2


def test_profile_overrides_and_incompatible_overrides() -> None:
    options = build_profile_scenario_options(
        "sar_person_search", {"drone_count": 4, "weather_preset": "fog"}
    )
    assert options.drone_count == 4
    assert options.weather_preset == "fog"
    with pytest.raises(ValueError, match="requires at least one target"):
        build_profile_scenario_options("sar_person_search", {"target_count": 0})
    with pytest.raises(ValueError, match="fixed POIs"):
        build_profile_scenario_options("industrial_asset_survey", {"target_count": 1})
    with pytest.raises(ValueError, match="Unsupported profile override"):
        build_profile_scenario_options("sar_person_search", {"unknown": True})


def test_cli_profile_defaults_and_explicit_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_simulate(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("argusnet.simulation.sim.simulate", fake_simulate)
    args = parse_args(["--mission-profile", "sar_person_search", "--drone-count", "4"])
    run_from_args(args)
    assert captured["mission_profile"] == "sar_person_search"
    assert captured["mission_profile_overrides"] == {"drone_count": 4}


@pytest.mark.parametrize(
    ("profile_id", "expect_scan_state"),
    [("sar_person_search", False), ("industrial_asset_survey", True)],
)
def test_profile_simulation_smoke_and_replay_metadata(
    profile_id: str, expect_scan_state: bool
) -> None:
    options = build_profile_scenario_options(
        profile_id, {"safety_blocking": False}
    )
    scenario = build_default_scenario(options=options, seed=7)
    result = run_simulation(
        scenario,
        SimulationConfig(steps=2, dt_s=0.5, seed=7),
    )
    assert any(frame.scan_mission_state is not None for frame in result.frames) is expect_scan_state
    document = build_replay_document_from_result(result)
    meta = document["meta"]
    assert isinstance(meta, dict)
    mission_profile = meta["mission_profile"]
    scenario_options = meta["scenario_options"]
    assert isinstance(mission_profile, dict)
    assert isinstance(scenario_options, dict)
    assert mission_profile["profile_id"] == profile_id
    assert scenario_options["mission_profile_id"] == profile_id
