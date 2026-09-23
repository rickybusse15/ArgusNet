"""The scenario-bundle exporter: what it writes, and what it refuses.

The Rust side (`rust/argusnet-sim/tests/scenario_bundle.rs`) proves a bundle
runs and matches the reference. These tests pin the exporter's own contract:
a scenario it cannot describe is refused by name, never approximated.
"""

from __future__ import annotations

import json
import math

import pytest

from argusnet.simulation.scenario_bundle import (
    BUNDLE_FORMAT,
    BUNDLE_VERSION,
    UnsupportedScenario,
    export_bundle,
)
from argusnet.simulation.sim import (
    ScenarioOptions,
    SimulationConfig,
    build_default_scenario,
    looped_waypoint_path,
    static_ground_point,
)

CONFIG = SimulationConfig(steps=6, dt_s=0.5, seed=3)


def _scenario(**overrides):
    options = dict(
        map_preset="small",
        mission_mode="target_tracking",
        target_count=1,
        drone_count=2,
        ground_station_count=2,
    )
    options.update(overrides)
    return build_default_scenario(ScenarioOptions(**options), seed=3)


def test_a_target_tracking_scenario_exports_as_strict_json():
    bundle = export_bundle(_scenario(), CONFIG)
    assert bundle["format"] == BUNDLE_FORMAT
    assert bundle["version"] == BUNDLE_VERSION
    # allow_nan=False is what write_bundle uses; the document must survive it.
    json.dumps(bundle, allow_nan=False)
    kinds = {node["controller"]["kind"] for node in bundle["nodes"]}
    assert kinds == {"static", "launched"}


def test_truth_is_sampled_at_every_tick_and_every_lead():
    scenario = _scenario(drone_mode_preset="follow")
    bundle = export_bundle(scenario, CONFIG)
    times = {sample["t"] for sample in bundle["targets"][0]["samples"]}
    leads = {
        node["controller"]["operational"]["follow"]["lead_s"]
        for node in bundle["nodes"]
        if node["controller"]["kind"] == "launched"
    }
    for step in range(CONFIG.steps):
        t = step * CONFIG.dt_s
        assert t in times
        for lead in leads:
            assert t + lead in times, f"no sample at {t} + {lead}"


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        ({"mission_mode": "scan_map_inspect"}, "mission_mode"),
        ({"enforce_flight_envelope": True}, "enforce_flight_envelope"),
        ({"safety_blocking": True}, "safety_blocking"),
    ],
)
def test_unsupported_scenarios_are_refused_by_name(overrides, fragment):
    with pytest.raises(UnsupportedScenario, match=fragment):
        export_bundle(_scenario(**overrides), CONFIG)


def test_the_closure_specs_describe_how_the_closure_was_built():
    class Flat:
        def height_at(self, x_m, y_m):
            return 10.0

    station = static_ground_point(3.0, 4.0, mast_agl_m=5.0, terrain=Flat())
    assert station.bundle_spec == {
        "kind": "static_ground_point",
        "x_m": 3.0,
        "y_m": 4.0,
        "mast_agl_m": 5.0,
    }
    # The spec records the UNFLOORED speed; the Rust port floors it itself.
    lane = looped_waypoint_path(
        [[0.0, 0.0], [10.0, 0.0]],
        speed_mps=0.1,
        base_agl_m=20.0,
        vertical_amplitude=0.0,
        vertical_frequency=0.0,
        phase=math.pi,
        terrain=None,
    )
    assert lane.bundle_spec["speed_mps"] == 0.1
    assert lane.bundle_spec["avoid_obstacles"] is False
    assert lane.bundle_spec["waypoints_xy_m"] == [[0.0, 0.0], [10.0, 0.0]]
