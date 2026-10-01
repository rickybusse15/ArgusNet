#!/usr/bin/env python3
"""Record `export_bundle` over a matrix of scenario options (P4.9b.2b).

The native builder must write the bundle Python writes. The committed scenario
bundles pin three scenarios; this matrix reaches the rest of the platform
half: every target motion preset, every drone mode preset, every weather
preset, both platform presets, more ground stations than the layout table
has offsets (the overflow ring), more drones than compass names, building-
anchored objective and exclusion zones, larger maps, and scan runs with one
and several drones. Since P4.9d, both mission profiles (their zones, POIs and
name) and a blocking-safety run as well.

The `world` section and the world keys of `presentation` are dropped: P4.9b.1
covers them (`scenario_world.json`), and they are most of a bundle's bytes.

Usage:
    python3 scripts/capture_native_bundle_fixture.py --out tests/fixtures/native_bundles.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from argusnet.simulation.scenario_bundle import export_bundle  # noqa: E402
from argusnet.simulation.sim import (  # noqa: E402
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
)

# The world keys of `presentation`, which P4.9b.1 compares.
WORLD_PRESENTATION_KEYS = {
    "terrain",
    "occluding_objects",
    "environment_id",
    "crs_id",
    "environment_bounds_m",
    "terrain_summary",
    "land_cover",
    "land_cover_legend",
    "enu_origin",
}

TT = "target_tracking"
CASES = [
    dict(
        seed=3,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="loiter",
            target_count=2,
            drone_count=4,
            ground_station_count=3,
            weather_preset="light_rain",
            platform_preset="wide_area",
        ),
    ),
    dict(
        seed=4,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="transit",
            drone_mode_preset="inspect",
            target_count=3,
            drone_count=5,
            ground_station_count=4,
            weather_preset="storm",
        ),
    ),
    dict(
        seed=5,
        options=dict(
            map_preset="medium",
            mission_mode=TT,
            target_motion_preset="evasive",
            drone_mode_preset="search",
            target_count=2,
            drone_count=3,
            ground_station_count=5,
            weather_preset="fog",
        ),
    ),
    dict(
        seed=6,
        options=dict(
            map_preset="large",
            mission_mode=TT,
            target_motion_preset="racetrack",
            target_count=4,
            drone_count=6,
            ground_station_count=14,
            weather_preset="snow",
        ),
    ),
    dict(
        seed=7,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="waypoint_patrol",
            drone_mode_preset="follow",
            target_count=1,
            drone_count=3,
            ground_station_count=13,
            weather_preset="heavy_rain",
        ),
    ),
    dict(
        seed=8,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="search_pattern",
            target_count=3,
            drone_count=2,
            ground_station_count=3,
            terrain_preset="military_compound",
            weather_preset="overcast",
        ),
    ),
    dict(
        seed=9,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="sinusoid",
            target_count=2,
            drone_count=9,
            ground_station_count=3,
        ),
    ),
    dict(
        seed=10,
        options=dict(
            map_preset="regional",
            mission_mode="scan_map_inspect",
            drone_count=4,
            ground_station_count=5,
            poi_count=5,
            terrain_preset="military_compound",
            weather_preset="overcast",
            scan_coverage_threshold=0.5,
        ),
    ),
    dict(
        seed=11,
        options=dict(
            map_preset="xlarge",
            mission_mode="scan_map_inspect",
            drone_count=3,
            ground_station_count=3,
            poi_count=4,
            terrain_preset="urban_flat",
            coverage_resolution_m=25.0,
        ),
    ),
    dict(
        seed=12,
        options=dict(
            map_preset="medium",
            mission_mode="scan_map_inspect",
            drone_count=1,
            ground_station_count=2,
            poi_count=2,
            target_count=1,
        ),
    ),
    dict(
        seed=13,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_motion_preset="mixed",
            target_count=2,
            drone_count=3,
            ground_station_count=3,
            terrain_preset="alpine",
            terrain_seed=35,
            season_month=1,
            detail_strength=0.6,
        ),
    ),
]


def _profile_case(seed: int, profile_id: str, **overrides) -> dict:
    """A mission profile's resolved options, as `--mission-profile` resolves
    them (P4.9d): its zones and, for inspection, its POIs replace the
    generated ones, and it turns on the blocking safety monitor."""
    import dataclasses

    from argusnet.mission.profiles import build_profile_scenario_options

    options = dataclasses.asdict(build_profile_scenario_options(profile_id))
    return dict(seed=seed, options=json.loads(json.dumps({**options, **overrides})))


CASES += [
    _profile_case(21, "sar_person_search"),
    _profile_case(22, "industrial_asset_survey"),
    # Fewer POIs than the profile lists: the reference takes the first ones.
    _profile_case(24, "industrial_asset_survey", poi_count=2),
    # Blocking safety without a profile: only the bundle flag differs.
    dict(
        seed=23,
        options=dict(
            map_preset="small",
            mission_mode=TT,
            target_count=1,
            drone_count=2,
            ground_station_count=3,
            safety_blocking=True,
        ),
    ),
]
STEPS = 12
DT_S = 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    captured = []
    for case in CASES:
        options = ScenarioOptions(**case["options"])
        config = SimulationConfig(steps=STEPS, dt_s=DT_S, seed=case["seed"])
        scenario = build_default_scenario(options, seed=config.seed)
        bundle = json.loads(json.dumps(export_bundle(scenario, config)))
        bundle.pop("world")
        bundle["presentation"] = {
            k: v for k, v in bundle["presentation"].items() if k not in WORLD_PRESENTATION_KEYS
        }
        captured.append({"seed": case["seed"], "options": case["options"], "bundle": bundle})
        zone_types = sorted({z["zone_type"] for z in bundle["mission_zones"]})
        print(
            f"seed {case['seed']}: {len(bundle['nodes'])} nodes, "
            f"{len(bundle['targets'])} targets, zones {zone_types}"
        )
    args.out.write_text(
        json.dumps(
            {"_source": Path(__file__).name, "steps": STEPS, "dt_s": DT_S, "cases": captured},
            separators=(",", ":"),
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
