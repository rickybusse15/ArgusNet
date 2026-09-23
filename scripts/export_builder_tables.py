#!/usr/bin/env python3
"""Export the constant tables the native scenario builder reads (P4.9b.2).

The Rust builder (`argusnet_sim::builder`) embeds
`rust/argusnet-sim/src/builder/reference_tables.json`: the platform presets,
the dynamics and sensor constants, the ground-station layout, every weather
preset (as the bundle's motion-weather document), both tracker
configurations, and the sensor-error and localization defaults. Generated
from the reference rather than transcribed, and checked for drift by
`tests/test_builder_tables.py`. When P6 retires the Python runtime this file
becomes the tables' home.

Usage:
    python3 scripts/export_builder_tables.py            # rewrite the embedded tables
    python3 scripts/export_builder_tables.py --check    # fail if they drifted
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_OUT = REPO_ROOT / "rust" / "argusnet-sim" / "src" / "builder" / "reference_tables.json"


def tables() -> dict:
    from argusnet.adapters.argusnet_grpc import TrackerConfig
    from argusnet.localization.engine import LocalizationConfig
    from argusnet.planning.frontier import FrontierPlanner
    from argusnet.sensing.models.noise import SensorErrorConfig
    from argusnet.simulation import sim
    from argusnet.simulation.scenario_bundle import motion_weather_document
    from argusnet.world.weather import WEATHER_PRESETS

    constants = sim._DEFAULT_CONSTANTS
    return {
        "platform_presets": {
            name: dataclasses.asdict(profile) for name, profile in sim.PLATFORM_PRESETS.items()
        },
        "dynamics": dataclasses.asdict(constants.dynamics),
        "sensor": json.loads(json.dumps(dataclasses.asdict(constants.sensor))),
        "ground_station_layout": json.loads(
            json.dumps(dataclasses.asdict(constants.ground_station_layout))
        ),
        "weather_presets": {
            name: motion_weather_document(sim.weather_from_preset(name))
            for name in sorted(WEATHER_PRESETS)
        },
        "tracker_default": dataclasses.asdict(TrackerConfig()),
        "tracker_simulation": dataclasses.asdict(sim.DEFAULT_SIMULATION_TRACKER_CONFIG),
        "sensor_error_defaults": dataclasses.asdict(SensorErrorConfig()),
        "localization": dataclasses.asdict(LocalizationConfig()),
        "gap_fill_min_fraction": float(FrontierPlanner().cfg.gap_fill_min_fraction),
        "poi_hover_agl_m": float(sim.POI_HOVER_AGL_M),
    }


def render() -> str:
    return json.dumps(tables(), indent=1, sort_keys=True, default=str) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--check", action="store_true", help="fail if the committed file drifted")
    args = parser.parse_args()
    text = render()
    if args.check:
        if args.out.read_text() != text:
            print(f"{args.out} is stale; rerun scripts/export_builder_tables.py")
            return 1
        print(f"{args.out} is current")
        return 0
    args.out.write_text(text)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
