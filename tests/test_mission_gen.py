"""Tests for the mission generation module (argusnet.planning.inspection).

Covers data class validation, defaults, frozen enforcement, difficulty scaling,
tag generation, template factories, and the generate_mission pipeline.
"""

from __future__ import annotations

import dataclasses
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from argusnet.core.config import SimulationConstants
from argusnet.planning.inspection import (
    FlightCorridor,
    GeneratedMission,
    MissionConstraints,
    MissionObjective,
    MissionSpec,
    MissionTiming,
    ObjectiveCondition,
    ValidityReport,
    _base_tags,
    _difficulty_band,
    apply_difficulty_scaling,
    generate_mission,
    intercept_template,
    persistent_observation_template,
    search_template,
    surveillance_template,
    validate_mission,
)

# ---------------------------------------------------------------------------
# MissionSpec validation & defaults
# ---------------------------------------------------------------------------


class TestMissionSpecValidation(unittest.TestCase):
    def test_difficulty_out_of_range_raises(self) -> None:
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, difficulty=-0.01)
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, difficulty=1.01)

    def test_difficulty_boundaries_ok(self) -> None:
        self.assertEqual(MissionSpec(seed=1, difficulty=0.0).difficulty, 0.0)
        self.assertEqual(MissionSpec(seed=1, difficulty=1.0).difficulty, 1.0)

    def test_drone_count_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, drone_count=0)
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, drone_count=13)

    def test_ground_station_count_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, ground_station_count=0)
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, ground_station_count=13)

    def test_target_count_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, target_count=0)
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, target_count=9)

    def test_target_count_boundaries_ok(self) -> None:
        self.assertEqual(MissionSpec(seed=1, target_count=1).target_count, 1)
        self.assertEqual(MissionSpec(seed=1, target_count=8).target_count, 8)

    def test_invalid_mission_type_raises(self) -> None:
        with self.assertRaises(ValueError):
            MissionSpec(seed=1, mission_type="dogfight")

    def test_valid_mission_types_accepted(self) -> None:
        for mt in ("surveillance", "intercept", "persistent_observation", "search"):
            self.assertEqual(MissionSpec(seed=1, mission_type=mt).mission_type, mt)


class TestMissionSpecDefaults(unittest.TestCase):
    def setUp(self) -> None:
        self.s = MissionSpec(seed=42)

    def test_defaults(self) -> None:
        self.assertEqual(self.s.terrain_preset, "alpine")
        self.assertEqual(self.s.weather_preset, "clear")
        self.assertEqual(self.s.map_preset, "regional")
        self.assertEqual(self.s.platform_preset, "baseline")
        self.assertEqual(self.s.drone_count, 2)
        self.assertEqual(self.s.ground_station_count, 4)
        self.assertEqual(self.s.target_count, 2)
        self.assertAlmostEqual(self.s.difficulty, 0.5)
        self.assertEqual(self.s.mission_type, "surveillance")
        self.assertEqual(self.s.tags, [])

    def test_timing_defaults(self) -> None:
        self.assertAlmostEqual(self.s.timing.duration_s, 180.0)
        self.assertAlmostEqual(self.s.timing.dt_s, 0.25)

    def test_constraints_defaults(self) -> None:
        self.assertAlmostEqual(self.s.constraints.min_sensor_baseline_m, 80.0)
        self.assertAlmostEqual(self.s.constraints.max_target_covariance_m2, 500.0)


# ---------------------------------------------------------------------------
# Frozen dataclass enforcement
# ---------------------------------------------------------------------------


class TestFrozenDataclasses(unittest.TestCase):
    def test_frozen_fields(self) -> None:
        cases = [
            (MissionTiming(duration_s=60.0), "duration_s", 120.0),
            (MissionConstraints(), "min_sensor_baseline_m", 999.0),
            (MissionSpec(seed=1), "difficulty", 0.9),
            (ObjectiveCondition(), "max_position_error_m", 100.0),
            (ValidityReport(), "physically_valid", False),
            (FlightCorridor(corridor_id="c0", waypoints_xy_m=[[0, 0]]), "width_m", 999.0),
        ]
        for obj, attr, val in cases:
            with (
                self.subTest(cls=type(obj).__name__),
                self.assertRaises(dataclasses.FrozenInstanceError),
            ):
                setattr(obj, attr, val)


# ---------------------------------------------------------------------------
# ValidityReport.is_valid
# ---------------------------------------------------------------------------


class TestValidityReport(unittest.TestCase):
    def test_all_true_is_valid(self) -> None:
        self.assertTrue(ValidityReport().is_valid)

    def test_single_false_makes_invalid(self) -> None:
        for flag in (
            "physically_valid",
            "sensor_valid",
            "solvable",
            "corridor_clear",
            "baseline_adequate",
        ):
            with self.subTest(flag=flag):
                self.assertFalse(ValidityReport(**{flag: False}).is_valid)

    def test_failures_preserved(self) -> None:
        vr = ValidityReport(physically_valid=False, failures=["AGL violation"])
        self.assertEqual(vr.failures, ["AGL violation"])
        self.assertFalse(vr.is_valid)


# ---------------------------------------------------------------------------
# Difficulty scaling
# ---------------------------------------------------------------------------


class TestApplyDifficultyScaling(unittest.TestCase):
    def setUp(self) -> None:
        self.c = SimulationConstants.default()

    def _sc(self, d: float) -> SimulationConstants:
        return apply_difficulty_scaling(MissionSpec(seed=1, difficulty=d), self.c)

    def test_bearing_noise_at_boundaries(self) -> None:
        base = self.c.sensor.ground_station_bearing_stds_rad[0]
        self.assertAlmostEqual(self._sc(0.0).sensor.ground_station_bearing_stds_rad[0], base * 0.5)
        self.assertAlmostEqual(self._sc(1.0).sensor.ground_station_bearing_stds_rad[0], base * 2.0)
        # midpoint: multiplier = 0.5 + 0.5*1.5 = 1.25
        self.assertAlmostEqual(self._sc(0.5).sensor.ground_station_bearing_stds_rad[0], base * 1.25)

    def test_dropout_at_boundaries(self) -> None:
        base = self.c.sensor.ground_station_dropout_probabilities[0]
        self.assertAlmostEqual(
            self._sc(0.0).sensor.ground_station_dropout_probabilities[0], base * 0.3
        )
        self.assertAlmostEqual(
            self._sc(1.0).sensor.ground_station_dropout_probabilities[0], base * 2.5
        )

    def test_target_speed_scale(self) -> None:
        bp = self.c.platform_presets["baseline"]
        self.assertAlmostEqual(
            self._sc(0.0).platform_presets["baseline"].target_speed_scale,
            bp.target_speed_scale * 0.5,
        )
        self.assertAlmostEqual(
            self._sc(1.0).platform_presets["baseline"].target_speed_scale,
            bp.target_speed_scale * 1.8,
        )

    def test_drone_search_speed_scale(self) -> None:
        bp = self.c.platform_presets["baseline"]
        self.assertAlmostEqual(
            self._sc(0.0).platform_presets["baseline"].drone_search_speed_scale,
            bp.drone_search_speed_scale * 1.2,
        )
        self.assertAlmostEqual(
            self._sc(1.0).platform_presets["baseline"].drone_search_speed_scale,
            bp.drone_search_speed_scale * 0.8,
        )

    def test_drone_bearing_std_scaled(self) -> None:
        base = self.c.sensor.drone_base_bearing_std_rad
        self.assertAlmostEqual(self._sc(1.0).sensor.drone_base_bearing_std_rad, base * 2.0)

    def test_returns_correct_type(self) -> None:
        self.assertIsInstance(self._sc(0.5), SimulationConstants)

    def test_unscaled_fields_preserved(self) -> None:
        sc = self._sc(0.7)
        self.assertEqual(sc.dynamics.default_dt_s, self.c.dynamics.default_dt_s)
        self.assertEqual(sc.sensor.drone_base_max_range_m, self.c.sensor.drone_base_max_range_m)


# ---------------------------------------------------------------------------
# _difficulty_band
# ---------------------------------------------------------------------------


class TestDifficultyBand(unittest.TestCase):
    def test_easy(self) -> None:
        self.assertEqual(_difficulty_band(0.0), "easy")
        self.assertEqual(_difficulty_band(0.32), "easy")

    def test_medium(self) -> None:
        self.assertEqual(_difficulty_band(0.33), "medium")
        self.assertEqual(_difficulty_band(0.5), "medium")
        self.assertEqual(_difficulty_band(0.66), "medium")

    def test_hard(self) -> None:
        self.assertEqual(_difficulty_band(0.67), "hard")
        self.assertEqual(_difficulty_band(1.0), "hard")


# ---------------------------------------------------------------------------
# _base_tags
# ---------------------------------------------------------------------------


class TestBaseTags(unittest.TestCase):
    def test_default_spec_tags(self) -> None:
        tags = _base_tags(MissionSpec(seed=1))
        for t in (
            "type:surveillance",
            "terrain:alpine",
            "weather:clear",
            "diff:medium",
            "size:regional",
        ):
            self.assertIn(t, tags)

    def test_difficulty_band_tags(self) -> None:
        self.assertIn("diff:easy", _base_tags(MissionSpec(seed=1, difficulty=0.1)))
        self.assertIn("diff:hard", _base_tags(MissionSpec(seed=1, difficulty=0.9)))

    def test_custom_type_tag(self) -> None:
        self.assertIn("type:intercept", _base_tags(MissionSpec(seed=1, mission_type="intercept")))

    def test_extra_tags_preserved(self) -> None:
        tags = _base_tags(MissionSpec(seed=1, tags=["custom:alpha"]))
        self.assertIn("custom:alpha", tags)
        self.assertIn("type:surveillance", tags)

    def test_no_duplicate_tags(self) -> None:
        tags = _base_tags(MissionSpec(seed=1, tags=["type:surveillance"]))
        self.assertEqual(tags.count("type:surveillance"), 1)


# ---------------------------------------------------------------------------
# Template factories
# ---------------------------------------------------------------------------


class TestTemplateFactories(unittest.TestCase):
    def test_surveillance(self) -> None:
        s = surveillance_template(seed=10)
        self.assertEqual(s.mission_type, "surveillance")
        self.assertGreaterEqual(s.drone_count, 2)

    def test_intercept(self) -> None:
        s = intercept_template(seed=10)
        self.assertEqual(s.mission_type, "intercept")
        self.assertGreaterEqual(s.drone_count, 3)

    def test_intercept_deadline(self) -> None:
        s = intercept_template(seed=10, duration_s=200.0)
        self.assertAlmostEqual(s.timing.deadline_s, 200.0 * 0.85)

    def test_persistent_observation(self) -> None:
        s = persistent_observation_template(seed=10)
        self.assertEqual(s.mission_type, "persistent_observation")
        self.assertGreaterEqual(s.drone_count, 4)
        self.assertGreaterEqual(s.target_count, 2)

    def test_persistent_observation_min_duration(self) -> None:
        s = persistent_observation_template(seed=10, duration_s=100.0)
        self.assertGreaterEqual(s.timing.duration_s, 300.0)

    def test_search(self) -> None:
        s = search_template(seed=10)
        self.assertEqual(s.mission_type, "search")
        self.assertAlmostEqual(s.timing.deadline_s, 240.0 * 0.75)

    def test_extra_tags(self) -> None:
        self.assertIn("custom:beta", surveillance_template(seed=1, extra_tags=["custom:beta"]).tags)

    def test_all_templates_valid_specs(self) -> None:
        for factory in (
            surveillance_template,
            intercept_template,
            persistent_observation_template,
            search_template,
        ):
            s = factory(seed=42)
            self.assertIsInstance(s, MissionSpec)
            self.assertTrue(1 <= s.drone_count <= 12)
            self.assertTrue(1 <= s.ground_station_count <= 12)
            self.assertTrue(1 <= s.target_count <= 8)
            self.assertTrue(0.0 <= s.difficulty <= 1.0)


# ---------------------------------------------------------------------------
# generate_mission (with mocked scenario builder)
# ---------------------------------------------------------------------------


def _make_stub_scenario(target_count=2, drone_count=2, gs_count=4):
    """Lightweight stub mimicking ScenarioDefinition attributes."""

    class _State:
        def __init__(self, pos):
            self.position, self.max_range_m = pos, 1e9

    class _Node:
        def __init__(self, nid, mobile, pos):
            self.node_id, self.is_mobile, self.position = nid, mobile, pos

        def state(self, _t):
            return _State(self.position.copy())

    class _Target:
        def __init__(self, tid, pos):
            self.target_id, self._pos = tid, pos

        def truth_state(self, _t):
            return _State(self._pos.copy())

    nodes = [_Node(f"ground-{i}", False, np.array([i * 100.0, 0.0, 5.0])) for i in range(gs_count)]
    nodes += [
        _Node(f"drone-{i}", True, np.array([i * 200.0, 100.0, 250.0])) for i in range(drone_count)
    ]
    targets = [
        _Target(f"target-{i}", np.array([i * 50.0, 50.0, 150.0])) for i in range(target_count)
    ]
    return SimpleNamespace(
        nodes=nodes,
        targets=targets,
        environment=None,
        terrain=None,
        map_bounds_m={"x_min_m": -2160.0, "x_max_m": 2160.0, "y_min_m": -1980.0, "y_max_m": 2160.0},
    )


class TestGenerateMission(unittest.TestCase):
    def _gen(self, mission_type="surveillance", **kw):
        kw.setdefault("seed", 7)
        kw.setdefault("difficulty", 0.3)
        kw["mission_type"] = mission_type
        spec = MissionSpec(**kw)
        stub = _make_stub_scenario(spec.target_count, spec.drone_count, spec.ground_station_count)
        with (
            patch("argusnet.simulation.sim.build_default_scenario", return_value=stub),
            patch("argusnet.simulation.sim.ScenarioOptions"),
        ):
            return generate_mission(spec)

    def _gen_from_template(self, factory, **kw):
        spec = factory(**kw)
        stub = _make_stub_scenario(spec.target_count, spec.drone_count, spec.ground_station_count)
        with (
            patch("argusnet.simulation.sim.build_default_scenario", return_value=stub),
            patch("argusnet.simulation.sim.ScenarioOptions"),
        ):
            return generate_mission(spec)

    def test_surveillance_structure(self) -> None:
        gm = self._gen("surveillance")
        self.assertIsInstance(gm, GeneratedMission)
        self.assertIsInstance(gm.timing, MissionTiming)
        self.assertIsInstance(gm.validity_report, ValidityReport)
        self.assertGreater(len(gm.objectives), 0)
        self.assertGreater(len(gm.corridors), 0)
        self.assertIn("type:surveillance", gm.tags)
        self.assertIsNotNone(gm.scenario_def)

    def test_intercept(self) -> None:
        gm = self._gen_from_template(intercept_template, seed=7, difficulty=0.3)
        self.assertEqual(gm.spec.mission_type, "intercept")
        self.assertGreater(len(gm.objectives), 0)

    def test_persistent_observation(self) -> None:
        gm = self._gen_from_template(persistent_observation_template, seed=7, difficulty=0.3)
        self.assertEqual(gm.spec.mission_type, "persistent_observation")

    def test_search(self) -> None:
        gm = self._gen_from_template(search_template, seed=7, difficulty=0.3)
        self.assertEqual(gm.spec.mission_type, "search")

    def test_timing_scaled_by_difficulty(self) -> None:
        gm = self._gen("surveillance")
        expected = 180.0 * (1.5 - 0.3 * 0.8)
        self.assertAlmostEqual(gm.timing.duration_s, expected, places=1)

    def test_deterministic(self) -> None:
        gm1 = self._gen("surveillance", seed=42, difficulty=0.4)
        gm2 = self._gen("surveillance", seed=42, difficulty=0.4)
        self.assertEqual(len(gm1.objectives), len(gm2.objectives))
        self.assertEqual(gm1.tags, gm2.tags)


# ---------------------------------------------------------------------------
# validate_mission with stubs
# ---------------------------------------------------------------------------


class TestValidateMission(unittest.TestCase):
    def test_missing_attributes_all_false(self) -> None:
        report = validate_mission(object(), MissionSpec(seed=1), [], [])
        self.assertFalse(report.is_valid)
        self.assertGreater(len(report.failures), 0)

    def test_unsolvable_continuity_fraction(self) -> None:
        scenario = SimpleNamespace(
            nodes=[], targets=[], environment=None, terrain=None, map_bounds_m=None
        )
        bad_obj = MissionObjective(
            objective_id="bad",
            objective_type="maintain",
            target_ids=["t-0"],
            success_condition=ObjectiveCondition(track_continuity_fraction=1.5),
        )
        report = validate_mission(scenario, MissionSpec(seed=1), [bad_obj], [])
        self.assertFalse(report.solvable)


if __name__ == "__main__":
    unittest.main()
