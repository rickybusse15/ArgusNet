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
        ({"mission_mode": "scan_map_inspect", "frontier_exploration": True}, "frontier"),
        ({"mission_mode": "scan_map_inspect", "occlusion_aware_mapping": True}, "occlusion"),
        ({"enforce_flight_envelope": True}, "enforce_flight_envelope"),
    ],
)
def test_unsupported_scenarios_are_refused_by_name(overrides, fragment):
    with pytest.raises(UnsupportedScenario, match=fragment):
        export_bundle(_scenario(**overrides), CONFIG)


def test_blocking_safety_is_carried_only_when_asked_for():
    """P4.9d: stage 4 assembles from the bundle's flag. Written only when set,
    so a bundle without it is the bundle it was before stage 4 assembled."""
    assert export_bundle(_scenario(safety_blocking=True), CONFIG)["safety_blocking"] is True
    assert "safety_blocking" not in export_bundle(_scenario(), CONFIG)


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


def test_a_scan_map_inspect_bundle_carries_the_reference_pois():
    from argusnet.simulation.sim import build_run_inspection_pois

    scenario = _scenario(mission_mode="scan_map_inspect", poi_count=4)
    bundle = export_bundle(scenario, CONFIG)
    mission = bundle["mission"]
    expected = build_run_inspection_pois(scenario, CONFIG.seed, 4)
    assert [p["poi_id"] for p in mission["pois"]] == [p.poi_id for p in expected]
    assert [p["position"] for p in mission["pois"]] == [
        [float(v) for v in p.position] for p in expected
    ]
    assert mission["coverage_bounds"]["resolution_m"] == scenario.options.coverage_resolution_m
    assert mission["scan_coverage_threshold"] == scenario.options.scan_coverage_threshold


def test_a_target_tracking_bundle_has_no_mission_section():
    assert export_bundle(_scenario(), CONFIG)["mission"] is None


# --- argusnet sim --export-bundle ------------------------------------------

_CLI_ARGS = [
    "--map-preset", "small",
    "--mission-mode", "target_tracking",
    "--target-count", "1",
    "--drone-count", "2",
    "--ground-stations", "2",
    "--steps", "6",
    "--dt", "0.5",
    "--seed", "3",
]  # fmt: skip


def _argusnet_rs():
    """The built ``argusnet-rs``, or a skip -- a failure under CI's
    ``ARGUSNET_REQUIRE_RS``, which builds it first."""
    import os
    from pathlib import Path

    binary = Path(__file__).resolve().parent.parent / "target" / "debug" / "argusnet-rs"
    if not binary.exists():
        if os.environ.get("ARGUSNET_REQUIRE_RS"):
            pytest.fail(f"ARGUSNET_REQUIRE_RS is set but {binary} is not built")
        pytest.skip("argusnet-rs is not built (cargo build -p argusnet-cli)")
    return binary


def _export_via_cli(tmp_path, extra=()):
    from argusnet.simulation import sim

    out = tmp_path / "bundle.json"
    result = sim.run_from_args(sim.parse_args([*_CLI_ARGS, *extra, "--export-bundle", str(out)]))
    assert result is None, "--export-bundle ran the simulation instead of exporting"
    return out


def test_export_bundle_writes_exactly_what_the_command_would_run(tmp_path):
    from argusnet.simulation.sim import DEFAULT_SIMULATION_TRACKER_CONFIG

    written = json.loads(_export_via_cli(tmp_path).read_text())
    scenario = build_default_scenario(
        ScenarioOptions(
            map_preset="small",
            mission_mode="target_tracking",
            target_count=1,
            drone_count=2,
            ground_station_count=2,
        ),
        seed=3,
    )
    expected = export_bundle(
        scenario,
        SimulationConfig(steps=6, dt_s=0.5, seed=3),
        DEFAULT_SIMULATION_TRACKER_CONFIG,
    )
    # The command's own tracker config, not the dataclass default.
    assert (
        written["tracker"]["max_stale_steps"] == DEFAULT_SIMULATION_TRACKER_CONFIG.max_stale_steps
    )
    assert written == json.loads(json.dumps(expected))


def test_export_bundle_refuses_what_the_runtime_cannot_run(tmp_path):
    with pytest.raises(UnsupportedScenario, match="enforce_flight_envelope"):
        _export_via_cli(tmp_path, ["--enforce-flight-envelope"])


@pytest.mark.parametrize(
    "flag",
    [["--continuous"], ["--realtime"], ["--streaming"], ["--compact"], ["--replay-subsample", "2"]],
)
def test_export_bundle_refuses_run_time_options_it_would_drop(tmp_path, flag):
    with pytest.raises(UnsupportedScenario, match=flag[0]):
        _export_via_cli(tmp_path, flag)
    assert not (tmp_path / "bundle.json").exists()


def test_an_exported_bundle_runs_in_argusnet_rs_to_a_valid_replay(tmp_path):
    import subprocess

    from argusnet.evaluation.replay import load_replay_document

    binary = _argusnet_rs()
    bundle = _export_via_cli(tmp_path)
    replay = tmp_path / "replay.json"
    completed = subprocess.run(
        [str(binary), "sim", "--quiet", "--bundle", str(bundle), "--replay", str(replay)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = load_replay_document(str(replay))
    assert len(document["frames"]) == 6
    assert document["meta"]["producer"] == "argusnet-sim"


@pytest.mark.parametrize("case", ["scan", "tt_follow", "tt_safety"])
def test_an_argusnet_rs_replay_passes_the_schema_with_every_payload(tmp_path, case):
    """P4.9a.2: the enriched Rust replay is a replay the schema accepts, and its
    ``meta`` carries every key Python's does except the two it declines:
    ``performance`` (wall-clock, so never byte-stable) and ``vio_tracked_nodes``
    (stage 5 is not assembled)."""
    import subprocess
    from pathlib import Path

    from argusnet.evaluation.replay import validate_replay_with_schema

    root = Path(__file__).resolve().parent.parent
    binary = _argusnet_rs()
    bundle_path = root / "tests" / "fixtures" / f"scenario_bundle_{case}.json"
    replay = tmp_path / "replay.json"
    completed = subprocess.run(
        [str(binary), "sim", "--quiet", "--bundle", str(bundle_path), "--replay", str(replay)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    document = json.loads(replay.read_text())
    assert validate_replay_with_schema(document) == []

    reference = json.loads(
        (root / "tests" / "fixtures" / f"scenario_bundle_{case}_reference.json").read_text()
    )
    bundle = json.loads(bundle_path.read_text())
    python_meta_keys = (
        {"schema_version", "scenario_name", "frame_count", "dt_s", "seed", "node_ids", "track_ids"}
        | set(bundle["presentation"])
        | set(reference["meta_run"])
    )
    assert python_meta_keys <= set(document["meta"])
    payloads = {key for frame in document["frames"] for key in frame}
    expected = {"mapping_state", "inspection_events"}
    if case == "scan":
        # No targets: nothing to observe, so no target metadata either.
        expected |= {"scan_mission_state"}
    else:
        expected |= {
            "generation_rejections",
            "deconfliction_events",
            "localization_state",
            "target_metadata",
        }
    if bundle.get("safety_blocking"):
        expected |= {"safety_events"}
    assert expected <= payloads


def test_the_bundles_presentation_is_the_replay_writers_meta(tmp_path):
    """The scenario half of ``meta`` rides in the bundle so ``argusnet-rs`` can
    copy it. It must be byte-for-byte what ``write_replay_document`` writes --
    in particular the terrain mesh's rows stay nested, as the viewer reads
    them, which ``to_jsonable`` would flatten."""
    from argusnet.evaluation.replay import write_replay_document
    from argusnet.simulation.sim import (
        DEFAULT_SIMULATION_TRACKER_CONFIG,
        build_replay_document_from_result,
        run_simulation,
    )

    scenario = _scenario()
    config = SimulationConfig(steps=3, dt_s=0.5, seed=CONFIG.seed)
    bundle = json.loads(json.dumps(export_bundle(scenario, config)))
    result = run_simulation(scenario, config, tracker_config=DEFAULT_SIMULATION_TRACKER_CONFIG)
    replay = tmp_path / "replay.json"
    write_replay_document(str(replay), build_replay_document_from_result(result))
    meta = json.loads(replay.read_text())["meta"]
    for key, value in bundle["presentation"].items():
        assert meta[key] == value, key
    heights = bundle["presentation"]["terrain"]["viewer_mesh"]["heights_m"]
    assert isinstance(heights[0], list), "the terrain mesh was flattened"
    assert bundle["requested_duration_s"] == meta["requested_duration_s"]


def test_the_coverage_grid_carries_the_scenarios_resolution():
    """``coverage_bounds`` is ``_coverage_bounds``: the scenario's own
    resolution, in every mode, agreeing with the mission section's."""
    tracking = export_bundle(_scenario(coverage_resolution_m=25.0), CONFIG)
    assert tracking["coverage_bounds"]["resolution_m"] == 25.0
    scan = export_bundle(
        _scenario(mission_mode="scan_map_inspect", coverage_resolution_m=25.0, poi_count=1),
        CONFIG,
    )
    assert scan["coverage_bounds"] == scan["mission"]["coverage_bounds"]


def _assert_same_within(ours, theirs, path="bundle", tolerance=1e-12):
    """Exact structure, strings, booleans and integers; floats within
    `tolerance` (terrain heights and platform-fused `uniform` draws)."""
    assert type(ours) is type(theirs) or {type(ours), type(theirs)} <= {int, float}, path
    if isinstance(ours, dict):
        assert set(ours) == set(theirs), f"{path}: keys"
        for key in ours:
            _assert_same_within(ours[key], theirs[key], f"{path}.{key}", tolerance)
    elif isinstance(ours, list):
        assert len(ours) == len(theirs), f"{path}: length"
        for index, (a, b) in enumerate(zip(ours, theirs, strict=True)):
            _assert_same_within(a, b, f"{path}[{index}]", tolerance)
    elif isinstance(ours, float):
        assert abs(ours - theirs) <= tolerance, f"{path}: {ours} vs {theirs}"
    else:
        assert ours == theirs, f"{path}: {ours!r} vs {theirs!r}"


def test_argusnet_rs_builds_the_bundle_argusnet_sim_exports(tmp_path):
    """P4.9b.2b end to end: the same options through Python's
    `--export-bundle` and Rust's `argusnet-rs build` give the same bundle,
    with the tracker the command runs with."""
    import subprocess

    binary = _argusnet_rs()
    python_bundle = json.loads(_export_via_cli(tmp_path).read_text())
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "format": "argusnet.scenario-config",
                "version": 1,
                "seed": 3,
                "steps": 6,
                "dt_s": 0.5,
                "options": {
                    "map_preset": "small",
                    "mission_mode": "target_tracking",
                    "target_count": 1,
                    "drone_count": 2,
                    "ground_station_count": 2,
                },
            }
        )
    )
    rust_path = tmp_path / "rust_bundle.json"
    completed = subprocess.run(
        [str(binary), "build", "--config", str(config), "--bundle", str(rust_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    _assert_same_within(json.loads(rust_path.read_text()), python_bundle)


def test_export_config_is_the_config_the_same_flags_export_as_a_bundle(tmp_path):
    """P4.9's adapter: `--export-config` writes the versioned config these
    flags resolve to, and `argusnet-rs build` of it is the bundle
    `--export-bundle` writes for the same flags."""
    import subprocess

    from argusnet.simulation import sim

    binary = _argusnet_rs()
    config_path = tmp_path / "exported_config.json"
    result = sim.run_from_args(sim.parse_args([*_CLI_ARGS, "--export-config", str(config_path)]))
    assert result is None, "--export-config ran the simulation"
    config = json.loads(config_path.read_text())
    assert config["format"] == "argusnet.scenario-config"
    assert (config["seed"], config["steps"], config["dt_s"]) == (3, 6, 0.5)
    assert config["options"]["target_count"] == 1
    assert config["options"]["mission_mode"] == "target_tracking"

    built = tmp_path / "built.json"
    completed = subprocess.run(
        [str(binary), "build", "--config", str(config_path), "--bundle", str(built)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    exported = json.loads(_export_via_cli(tmp_path).read_text())
    _assert_same_within(json.loads(built.read_text()), exported)


@pytest.mark.parametrize("flag", [["--streaming"], ["--compact"], ["--realtime"]])
def test_export_config_refuses_output_options(tmp_path, flag):
    from argusnet.simulation import sim

    with pytest.raises(UnsupportedScenario, match=flag[0]):
        sim.run_from_args(
            sim.parse_args([*_CLI_ARGS, *flag, "--export-config", str(tmp_path / "c.json")])
        )
    assert not (tmp_path / "c.json").exists()


def test_export_config_refuses_custom_constants(tmp_path):
    """A scenario config has no constants, so a run built on custom ones is
    refused rather than exported as if it used the defaults."""
    from argusnet.core.config import SimulationConstants
    from argusnet.simulation import sim

    constants_path = tmp_path / "constants.json"
    constants_path.write_text(
        json.dumps(
            SimulationConstants.from_dict({"dynamics": {"default_duration_s": 2.0}}).to_dict()
        )
    )
    with pytest.raises(UnsupportedScenario, match="--config-file"):
        sim.run_from_args(
            sim.parse_args(
                [
                    *_CLI_ARGS,
                    "--config-file",
                    str(constants_path),
                    "--export-config",
                    str(tmp_path / "c.json"),
                ]
            )
        )
    assert not (tmp_path / "c.json").exists()
