from __future__ import annotations

import io
import json
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

import argusnet.cli.main as cli_module
from argusnet.adapters.argusnet_grpc import TrackerConfig, TrackingService
from argusnet.core.config import SimulationConstants
from argusnet.core.types import BearingObservation, NodeState, TruthState, to_jsonable, vec3
from argusnet.evaluation.replay import load_replay_document
from argusnet.simulation.sim import (
    AERIAL_TARGET_MIN_AGL_M,
    GROUND_CONTACT_TOP_PAD_M,
    INTERCEPTOR_FOLLOW_ALTITUDE_OFFSET_M,
    INTERCEPTOR_FOLLOW_RADIUS_M,
    ObservationTriggeredFollowController,
    ScenarioDefinition,
    ScenarioOptions,
    SimNode,
    SimTarget,
    SimulationConfig,
    build_default_scenario,
    build_replay_document_from_result,
    run_simulation,
    simulate,
)
from argusnet.world.environment import LandCoverClass
from argusnet.world.terrain import TerrainModel


def bearing(
    node_id: str,
    target_id: str,
    origin: np.ndarray,
    target: np.ndarray,
    timestamp_s: float,
    confidence: float = 1.0,
    bearing_std_rad: float = 0.002,
) -> BearingObservation:
    direction = target - origin
    direction = direction / np.linalg.norm(direction)
    return BearingObservation(
        node_id=node_id,
        target_id=target_id,
        origin=origin,
        direction=direction,
        bearing_std_rad=bearing_std_rad,
        timestamp_s=timestamp_s,
        confidence=confidence,
    )


class TrackingServiceContractTest(unittest.TestCase):
    def test_tracking_service_defaults_to_stateless_history(self) -> None:
        service = TrackingService()
        frame = service.ingest_frame(
            0.0,
            node_states=self._nodes(0.0),
            observations=self._observations(0.0),
            truths=[self._truth(0.0)],
        )

        self.assertIs(frame, service.latest_frame())
        self.assertEqual([], service.history)

        retained = TrackingService(retain_history=True)
        retained.ingest_frame(
            0.0,
            node_states=self._nodes(0.0),
            observations=self._observations(0.0),
            truths=[self._truth(0.0)],
        )

        self.assertEqual(1, len(retained.history))

    def test_tracking_service_reset_clears_runtime_state(self) -> None:
        service = TrackingService(retain_history=True)
        service.ingest_frame(
            0.0,
            node_states=self._nodes(0.0),
            observations=self._observations(0.0),
            truths=[self._truth(0.0)],
        )

        service.reset()

        self.assertIsNone(service.latest_frame())
        self.assertEqual({}, service.nodes)
        self.assertEqual({}, service.tracks)
        self.assertEqual([], service.history)

    def test_tracker_config_propagates_to_runtime_behavior(self) -> None:
        service = TrackingService(config=TrackerConfig(min_confidence=0.95))
        frame = service.ingest_frame(
            0.0,
            node_states=self._nodes(0.0),
            observations=self._observations(0.0, confidence=0.7),
            truths=[self._truth(0.0)],
        )

        self.assertEqual(0, len(frame.tracks))
        self.assertEqual(2, frame.metrics.rejected_observation_count)
        self.assertEqual(2, frame.metrics.rejection_counts.get("low_confidence"))

    @staticmethod
    def _nodes(timestamp_s: float) -> list[NodeState]:
        return [
            NodeState("ground-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
            NodeState("ground-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, timestamp_s),
        ]

    @classmethod
    def _truth(cls, timestamp_s: float) -> TruthState:
        return TruthState("asset-a", vec3(50.0, 15.0, 10.0), vec3(0.0, 0.0, 0.0), timestamp_s)

    @classmethod
    def _observations(cls, timestamp_s: float, confidence: float = 1.0) -> list[BearingObservation]:
        truth = cls._truth(timestamp_s)
        nodes = cls._nodes(timestamp_s)
        return [
            bearing(
                "ground-a",
                "asset-a",
                nodes[0].position,
                truth.position,
                timestamp_s,
                confidence=confidence,
            ),
            bearing(
                "ground-b",
                "asset-a",
                nodes[1].position,
                truth.position,
                timestamp_s,
                confidence=confidence,
            ),
        ]


class SimulationCoreContractTest(unittest.TestCase):
    def test_run_simulation_is_deterministic_for_fixed_seed(self) -> None:
        scenario = build_default_scenario()
        config = SimulationConfig(steps=6, dt_s=0.5, seed=11)

        first = run_simulation(scenario, config)
        second = run_simulation(scenario, config)

        self.assertEqual(first.summary, second.summary)
        self.assertEqual(first.metrics_rows, second.metrics_rows)
        self.assertEqual(to_jsonable(first.frames), to_jsonable(second.frames))

    def test_run_simulation_has_no_stdout_side_effects(self) -> None:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            result = run_simulation(
                build_default_scenario(), SimulationConfig(steps=4, dt_s=0.5, seed=3)
            )

        self.assertEqual("", buffer.getvalue())
        self.assertEqual(4, len(result.frames))

    def test_default_scenario_returns_valid_definition(self) -> None:
        scenario = build_default_scenario()
        fixed_ground_nodes = [node for node in scenario.nodes if not node.is_mobile]

        self.assertIsInstance(scenario, ScenarioDefinition)
        self.assertTrue(scenario.scenario_name)
        self.assertEqual("regional", scenario.options.map_preset)
        self.assertEqual("mixed", scenario.options.target_motion_preset)
        self.assertEqual("mixed", scenario.options.drone_mode_preset)
        self.assertEqual("alpine", scenario.options.terrain_preset)
        self.assertFalse(scenario.options.clean_terrain)
        self.assertEqual("baseline", scenario.options.platform_preset)
        self.assertEqual(7, scenario.options.ground_station_count)
        self.assertEqual(7, len(fixed_ground_nodes))
        self.assertGreaterEqual(len(scenario.nodes), 1)
        self.assertGreaterEqual(len(scenario.targets), 1)
        terrain_metadata = scenario.terrain.to_metadata()
        self.assertGreater(
            terrain_metadata["max_height_m"] - terrain_metadata["min_height_m"], 120.0
        )
        self.assertTrue(
            any(
                getattr(obstacle, "blocker_type", None) in {"building", "wall", "vegetation"}
                for obstacle in scenario.environment.obstacles.primitives
            )
        )

    def test_ground_station_count_controls_fixed_node_count(self) -> None:
        scenario = build_default_scenario(
            ScenarioOptions(map_preset="large", ground_station_count=9), seed=7
        )
        fixed_ground_nodes = [node for node in scenario.nodes if not node.is_mobile]

        self.assertEqual(9, len(fixed_ground_nodes))

    def test_map_presets_expand_area_bounds(self) -> None:
        spans = {}
        for map_preset in ("small", "medium", "large", "xlarge", "regional"):
            scenario = build_default_scenario(ScenarioOptions(map_preset=map_preset), seed=7)
            result = run_simulation(scenario, SimulationConfig(steps=6, dt_s=0.25, seed=7))
            replay_document = build_replay_document_from_result(result)
            bounds = replay_document["meta"]["terrain"]["xy_bounds_m"]
            spans[map_preset] = max(
                bounds["x_max_m"] - bounds["x_min_m"],
                bounds["y_max_m"] - bounds["y_min_m"],
            )

        self.assertLess(spans["small"], spans["medium"])
        self.assertLess(spans["medium"], spans["large"])
        self.assertLess(spans["large"], spans["xlarge"])
        self.assertLess(spans["xlarge"], spans["regional"])

    def test_mission_zones_project_to_terrain_and_replay_metadata_preserves_height(self) -> None:
        scenario = build_default_scenario(
            ScenarioOptions(map_preset="small", terrain_preset="alpine"),
            seed=7,
        )
        self.assertTrue(scenario.mission_zones)

        for zone in scenario.mission_zones:
            expected_z = scenario.terrain.height_at(float(zone.center[0]), float(zone.center[1]))
            self.assertAlmostEqual(expected_z, float(zone.center[2]), places=4)

        replay_document = build_replay_document_from_result(
            run_simulation(scenario, SimulationConfig(steps=2, dt_s=0.5, seed=7))
        )
        for zone_meta in replay_document["meta"]["zones"]:
            expected_z = scenario.terrain.height_at(
                float(zone_meta["center"][0]), float(zone_meta["center"][1])
            )
            self.assertAlmostEqual(expected_z, float(zone_meta["center"][2]), places=4)

    def test_ground_contact_obstacles_span_local_terrain_relief(self) -> None:
        scenario = build_default_scenario(
            ScenarioOptions(map_preset="small", terrain_preset="alpine"),
            seed=7,
        )
        terrain = scenario.environment.terrain

        for obstacle in scenario.environment.obstacles.primitives:
            if getattr(obstacle, "blocker_type", "") not in {"building", "wall", "vegetation"}:
                continue
            with self.subTest(primitive_id=getattr(obstacle, "primitive_id", "unknown")):
                if hasattr(obstacle, "radius_m"):
                    angles = np.linspace(0.0, 2.0 * np.pi, num=8, endpoint=False, dtype=float)
                    sample_points = [
                        (
                            float(obstacle.center_x_m)
                            + float(obstacle.radius_m) * float(np.cos(angle)),
                            float(obstacle.center_y_m)
                            + float(obstacle.radius_m) * float(np.sin(angle)),
                        )
                        for angle in angles
                    ]
                    sample_points.append((float(obstacle.center_x_m), float(obstacle.center_y_m)))
                else:
                    footprint_source = obstacle.footprint_xy_m
                    footprint = (
                        footprint_source() if callable(footprint_source) else footprint_source
                    )
                    sample_points = [
                        (float(x_m), float(y_m)) for x_m, y_m in np.asarray(footprint, dtype=float)
                    ]

                heights = [float(terrain.height_at(x_m, y_m)) for x_m, y_m in sample_points]
                self.assertLessEqual(float(obstacle.base_z_m), min(heights) + 1.0e-6)
                self.assertGreaterEqual(
                    float(obstacle.top_z_m), max(heights) + GROUND_CONTACT_TOP_PAD_M - 1.0e-6
                )

    def test_terrain_presets_populate_expected_blockers_and_clean_terrain_strips_them(self) -> None:
        expected_classes = {
            "jungle_canopy": int(LandCoverClass.FOREST),
            "military_compound": int(LandCoverClass.URBAN),
            "river_valley": int(LandCoverClass.WATER),
        }
        for terrain_preset, expected_class in expected_classes.items():
            with self.subTest(terrain_preset=terrain_preset):
                scenario = build_default_scenario(
                    ScenarioOptions(map_preset="small", terrain_preset=terrain_preset),
                    seed=7,
                )
                blocker_types = {
                    getattr(obstacle, "blocker_type", "")
                    for obstacle in scenario.environment.obstacles.primitives
                }
                classes = {
                    int(value)
                    for tile in scenario.environment.land_cover._tiles.values()
                    for value in np.unique(tile.classes)
                }
                self.assertIn(expected_class, classes)
                if terrain_preset == "jungle_canopy":
                    self.assertIn("vegetation", blocker_types)
                elif terrain_preset == "military_compound":
                    self.assertTrue({"building", "wall"} & blocker_types)
                elif terrain_preset == "river_valley":
                    self.assertIn("vegetation", blocker_types)

                clean_scenario = build_default_scenario(
                    ScenarioOptions(
                        map_preset="small",
                        terrain_preset=terrain_preset,
                        clean_terrain=True,
                    ),
                    seed=7,
                )
                clean_blocker_types = {
                    getattr(obstacle, "blocker_type", "")
                    for obstacle in clean_scenario.environment.obstacles.primitives
                }
                self.assertFalse({"building", "wall", "vegetation"} & clean_blocker_types)
                clean_classes = {
                    int(value)
                    for tile in clean_scenario.environment.land_cover._tiles.values()
                    for value in np.unique(tile.classes)
                }
                if terrain_preset == "river_valley":
                    self.assertIn(int(LandCoverClass.WATER), clean_classes)
                else:
                    self.assertFalse(
                        {int(LandCoverClass.FOREST), int(LandCoverClass.URBAN)} & clean_classes
                    )

    def test_target_motion_presets_are_deterministic_for_fixed_seed(self) -> None:
        sample_times = (0.0, 5.0, 18.5)
        for target_motion in ("racetrack", "waypoint_patrol", "loiter", "transit", "evasive"):
            scenario_a = build_default_scenario(
                ScenarioOptions(
                    map_preset="medium",
                    target_motion_preset=target_motion,
                    drone_mode_preset="search",
                ),
                seed=13,
            )
            scenario_b = build_default_scenario(
                ScenarioOptions(
                    map_preset="medium",
                    target_motion_preset=target_motion,
                    drone_mode_preset="search",
                ),
                seed=13,
            )
            for timestamp_s in sample_times:
                truths_a = scenario_a.truths(timestamp_s)
                truths_b = scenario_b.truths(timestamp_s)
                for truth_a, truth_b in zip(truths_a, truths_b, strict=False):
                    self.assertTrue(np.allclose(truth_a.position, truth_b.position))
                    self.assertTrue(np.allclose(truth_a.velocity, truth_b.velocity))

    def test_follow_planner_stays_above_ground_and_within_bounds(self) -> None:
        options = ScenarioOptions(
            map_preset="medium", target_motion_preset="mixed", drone_mode_preset="follow"
        )
        scenario = build_default_scenario(options=options, seed=7)
        # Run a short simulation to trigger drone launches from ground stations
        result = run_simulation(scenario, SimulationConfig(steps=60, dt_s=0.25, seed=7))
        self.assertGreater(len(result.frames), 0)

        mobile_nodes = [node for node in scenario.nodes if node.is_mobile]
        altitude_histories: dict[str, list[float]] = {node.node_id: [] for node in mobile_nodes}
        airborne_start_s = result.frames[-1].timestamp_s + result.simulation_config.dt_s

        # Start after launch + climb phase (~10s) so drones are airborne
        for timestamp_s in np.linspace(airborne_start_s, airborne_start_s + 24.0, num=10):
            for node in mobile_nodes:
                state = node.state(float(timestamp_s))
                altitude_histories[node.node_id].append(float(state.position[2]))
                self.assertGreaterEqual(state.position[0], scenario.map_bounds_m["x_min_m"])
                self.assertLessEqual(state.position[0], scenario.map_bounds_m["x_max_m"])
                self.assertGreaterEqual(state.position[1], scenario.map_bounds_m["y_min_m"])
                self.assertLessEqual(state.position[1], scenario.map_bounds_m["y_max_m"])
                ground_m = scenario.terrain.height_at(
                    float(state.position[0]), float(state.position[1])
                )
                self.assertGreaterEqual(state.position[2], ground_m + 18.0)
                self.assertIsNone(
                    scenario.environment.obstacles.point_collides(
                        float(state.position[0]),
                        float(state.position[1]),
                        float(state.position[2]),
                    )
                )
        timestamps = np.linspace(airborne_start_s, airborne_start_s + 18.0, num=10)
        max_step_distance_m = 42.0 * 1.75 * ((timestamps[1] - timestamps[0]) + 1.0e-6)
        for node in mobile_nodes:
            samples = np.array(
                [node.state(float(timestamp_s)).position for timestamp_s in timestamps]
            )
            for start, end in zip(samples[:-1], samples[1:], strict=False):
                self.assertLessEqual(
                    float(np.linalg.norm(end[:2] - start[:2])), max_step_distance_m * 1.05
                )
            self.assertGreater(
                np.ptp(np.asarray(altitude_histories[node.node_id], dtype=float)), 4.0
            )

    def test_search_planner_covers_sector_and_stays_terrain_clamped(self) -> None:
        options = ScenarioOptions(
            map_preset="medium", target_motion_preset="mixed", drone_mode_preset="search"
        )
        scenario = build_default_scenario(options=options, seed=7)
        # Run a short simulation to trigger drone launches from ground stations
        result = run_simulation(scenario, SimulationConfig(steps=60, dt_s=0.25, seed=7))
        self.assertGreater(len(result.frames), 0)

        for node in [candidate for candidate in scenario.nodes if candidate.is_mobile]:
            controller = scenario.launchable_controllers[node.node_id].operational_trajectory
            if isinstance(controller, ObservationTriggeredFollowController):
                sampler = controller.search_trajectory
            else:
                sampler = controller
            # Start after launch + climb phase so drones are airborne
            samples = np.array(
                [sampler(float(timestamp_s))[0] for timestamp_s in np.linspace(24.0, 96.0, num=20)]
            )
            self.assertGreater(np.ptp(samples[:, 0]), 120.0)
            self.assertGreater(np.ptp(samples[:, 1]), 120.0)
            self.assertGreater(np.ptp(samples[:, 2]), 4.0)
            for sample in samples:
                ground_m = scenario.terrain.height_at(float(sample[0]), float(sample[1]))
                self.assertGreaterEqual(sample[2], ground_m + 18.0)
                self.assertIsNone(
                    scenario.environment.obstacles.point_collides(
                        float(sample[0]),
                        float(sample[1]),
                        float(sample[2]),
                    )
                )

    def test_search_drone_waypoints_do_not_land_inside_buildings(self) -> None:
        options = ScenarioOptions(
            map_preset="medium", target_motion_preset="mixed", drone_mode_preset="search"
        )
        scenario = build_default_scenario(options=options, seed=11)
        # Run a short simulation to trigger drone launches
        result = run_simulation(scenario, SimulationConfig(steps=60, dt_s=0.25, seed=11))
        self.assertGreater(len(result.frames), 0)
        airborne_start_s = result.frames[-1].timestamp_s + result.simulation_config.dt_s

        for node in [candidate for candidate in scenario.nodes if candidate.is_mobile]:
            # Start after launch + climb phase
            for timestamp_s in np.linspace(airborne_start_s, airborne_start_s + 20.0, num=10):
                state = node.state(float(timestamp_s))
                self.assertIsNone(
                    scenario.environment.obstacles.point_collides(
                        float(state.position[0]),
                        float(state.position[1]),
                        float(state.position[2]),
                    )
                )

    def test_search_drones_switch_into_follow_mode_after_target_detection(self) -> None:
        options = ScenarioOptions(
            map_preset="medium", target_motion_preset="mixed", drone_mode_preset="search"
        )
        scenario = build_default_scenario(options=options, seed=7)

        self.assertTrue(scenario.adaptive_drone_controllers)
        self.assertTrue(
            all(
                not controller.engaged
                for controller in scenario.adaptive_drone_controllers.values()
            )
        )

        run_simulation(scenario, SimulationConfig(steps=40, dt_s=0.5, seed=7))

        self.assertTrue(
            any(controller.engaged for controller in scenario.adaptive_drone_controllers.values())
        )

    def test_targets_traverse_as_airborne_tracks(self) -> None:
        scenario = build_default_scenario(
            ScenarioOptions(
                map_preset="medium", target_motion_preset="mixed", drone_mode_preset="search"
            ),
            seed=7,
        )

        altitude_histories: dict[str, list[float]] = {
            target.target_id: [] for target in scenario.targets
        }
        for timestamp_s in np.linspace(0.0, 36.0, num=12):
            for truth in scenario.truths(float(timestamp_s)):
                ground_m = scenario.terrain.height_at(
                    float(truth.position[0]), float(truth.position[1])
                )
                altitude_histories[truth.target_id].append(float(truth.position[2]))
                self.assertGreaterEqual(truth.position[2], ground_m + AERIAL_TARGET_MIN_AGL_M)

        for samples in altitude_histories.values():
            self.assertGreater(np.ptp(np.asarray(samples, dtype=float)), 8.0)

    def test_follow_interceptors_hold_radius_above_targets(self) -> None:
        scenario = build_default_scenario(
            ScenarioOptions(
                map_preset="medium", target_motion_preset="mixed", drone_mode_preset="follow"
            ),
            seed=7,
        )
        # Run a short simulation so ground stations trigger drone launches
        # (drones start grounded at stations and are launched on first radar detection).
        result = run_simulation(scenario, SimulationConfig(steps=80, dt_s=0.25, seed=7))
        self.assertGreater(len(result.frames), 0)

        # After launch + climb (~8s) + convergence, check late-phase positions.
        mobile_nodes = [node for node in scenario.nodes if node.is_mobile]
        for timestamp_s in np.linspace(15.0, 20.0, num=5):
            truth_by_id = {truth.target_id: truth for truth in scenario.truths(float(timestamp_s))}
            for node in mobile_nodes:
                state = node.state(float(timestamp_s))
                assigned_truth = truth_by_id[scenario.drone_target_assignments[node.node_id]]
                horizontal_distance_m = float(
                    np.linalg.norm(state.position[:2] - assigned_truth.position[:2])
                )
                self.assertGreaterEqual(horizontal_distance_m, INTERCEPTOR_FOLLOW_RADIUS_M * 0.45)
                self.assertLessEqual(horizontal_distance_m, INTERCEPTOR_FOLLOW_RADIUS_M * 1.65)
                self.assertGreaterEqual(
                    state.position[2],
                    assigned_truth.position[2] + INTERCEPTOR_FOLLOW_ALTITUDE_OFFSET_M * 0.3,
                )

    def test_new_large_terrain_presets_are_deterministic(self) -> None:
        for terrain_preset in ("rolling_highlands", "lake_district"):
            scenario_a = build_default_scenario(
                ScenarioOptions(
                    map_preset="xlarge",
                    target_motion_preset="mixed",
                    drone_mode_preset="search",
                    terrain_preset=terrain_preset,
                ),
                seed=17,
            )
            scenario_b = build_default_scenario(
                ScenarioOptions(
                    map_preset="xlarge",
                    target_motion_preset="mixed",
                    drone_mode_preset="search",
                    terrain_preset=terrain_preset,
                ),
                seed=17,
            )
            result_a = run_simulation(scenario_a, SimulationConfig(steps=6, dt_s=0.25, seed=17))
            result_b = run_simulation(scenario_b, SimulationConfig(steps=6, dt_s=0.25, seed=17))

            self.assertEqual(result_a.summary, result_b.summary)
            self.assertEqual(result_a.metrics_rows, result_b.metrics_rows)

    def test_platform_preset_decouples_speed_and_range_from_map_scale(self) -> None:
        def scenario_envelopes(
            map_preset: str, platform_preset: str
        ) -> dict[str, tuple[float, ...]]:
            scenario = build_default_scenario(
                ScenarioOptions(
                    map_preset=map_preset,
                    platform_preset=platform_preset,
                    target_motion_preset="mixed",
                    drone_mode_preset="mixed",
                ),
                seed=7,
            )
            search_speeds = []
            follow_caps = []
            for node in scenario.nodes:
                if not node.is_mobile:
                    continue
                controller = scenario.launchable_controllers[node.node_id].operational_trajectory
                if isinstance(controller, ObservationTriggeredFollowController):
                    _, velocity = controller.search_trajectory(20.0)
                    search_speeds.append(round(float(np.linalg.norm(velocity[:2])), 3))
                    follow_caps.append(round(float(controller.follow_trajectory.max_speed_mps), 3))
                else:
                    follow_caps.append(round(float(controller.max_speed_mps), 3))

            self.assertTrue(search_speeds)
            self.assertLess(len(search_speeds), len(follow_caps))

            return {
                "targets": tuple(
                    sorted(
                        round(float(np.linalg.norm(truth.velocity[:2])), 3)
                        for truth in scenario.truths(12.0)
                    )
                ),
                "search": tuple(sorted(search_speeds)),
                "follow": tuple(sorted(follow_caps)),
                "ground_ranges": tuple(
                    sorted(
                        round(float(node.max_range_m), 3)
                        for node in scenario.nodes
                        if not node.is_mobile
                    )
                ),
                "drone_ranges": tuple(
                    sorted(
                        round(float(node.max_range_m), 3)
                        for node in scenario.nodes
                        if node.is_mobile
                    )
                ),
            }

        baseline_small = scenario_envelopes("small", "baseline")
        for map_preset in ("regional", "operational"):
            with self.subTest(map_preset=map_preset):
                envelopes = scenario_envelopes(map_preset, "baseline")
                self.assertEqual(len(baseline_small["targets"]), len(envelopes["targets"]))
                for baseline_speed, candidate_speed in zip(
                    baseline_small["targets"], envelopes["targets"], strict=False
                ):
                    self.assertAlmostEqual(baseline_speed, candidate_speed, delta=0.35)
                self.assertEqual(baseline_small["search"], envelopes["search"])
                self.assertEqual(baseline_small["follow"], envelopes["follow"])
                self.assertEqual(baseline_small["ground_ranges"], envelopes["ground_ranges"])
                self.assertEqual(baseline_small["drone_ranges"], envelopes["drone_ranges"])

        wide_area_small = scenario_envelopes("small", "wide_area")

        self.assertNotEqual(
            baseline_small["targets"],
            wide_area_small["targets"],
        )
        self.assertNotEqual(
            baseline_small["search"],
            wide_area_small["search"],
        )
        self.assertNotEqual(
            baseline_small["follow"],
            wide_area_small["follow"],
        )
        self.assertNotEqual(
            baseline_small["ground_ranges"],
            wide_area_small["ground_ranges"],
        )
        self.assertNotEqual(
            baseline_small["drone_ranges"],
            wide_area_small["drone_ranges"],
        )

    def test_invalid_scenario_and_simulation_config_are_rejected(self) -> None:
        terrain = TerrainModel.default()
        node = SimNode(
            node_id="ground-a",
            is_mobile=False,
            bearing_std_rad=0.002,
            dropout_probability=0.0,
            max_range_m=500.0,
            trajectory=lambda _: (np.array([0.0, 0.0, 5.0], dtype=float), np.zeros(3, dtype=float)),
        )
        target = SimTarget(
            target_id="asset-a",
            trajectory=lambda _: (
                np.array([10.0, 0.0, 5.0], dtype=float),
                np.zeros(3, dtype=float),
            ),
        )

        with self.assertRaisesRegex(ValueError, "scenario_name"):
            ScenarioDefinition(scenario_name=" ", nodes=(node,), targets=(target,), terrain=terrain)
        with self.assertRaisesRegex(ValueError, "at least one node"):
            ScenarioDefinition(scenario_name="demo", nodes=(), targets=(target,), terrain=terrain)
        with self.assertRaisesRegex(ValueError, "at least one target"):
            ScenarioDefinition(scenario_name="demo", nodes=(node,), targets=(), terrain=terrain)
        with self.assertRaisesRegex(ValueError, "steps"):
            SimulationConfig(steps=0, dt_s=0.5, seed=1)
        with self.assertRaisesRegex(ValueError, "dt_s"):
            SimulationConfig(steps=1, dt_s=0.0, seed=1)

    def test_duration_factory_resolves_steps_and_rounds_up(self) -> None:
        config = SimulationConfig.from_duration(180.0, dt_s=0.25, seed=7)
        rounded = SimulationConfig.from_duration(180.1, dt_s=0.25, seed=7)

        self.assertEqual(721, config.steps)
        self.assertEqual(180.0, config.actual_duration_s)
        self.assertEqual(722, rounded.steps)
        self.assertAlmostEqual(180.25, rounded.actual_duration_s)

    def test_cli_defaults_and_exclusive_duration_inputs(self) -> None:
        args = cli_module.parse_args([])

        self.assertEqual("sim", args.command)
        self.assertIsNone(args.steps)
        self.assertEqual(180.0, args.duration_s)
        self.assertEqual(0.25, args.dt)
        self.assertIsNone(args.config_file)
        self.assertEqual("regional", args.map_preset)
        self.assertEqual(7, args.ground_stations)
        self.assertEqual("mixed", args.target_motion)
        self.assertEqual("mixed", args.drone_mode)
        self.assertEqual("alpine", args.terrain_preset)
        self.assertEqual("clear", args.weather_preset)
        self.assertFalse(args.clean_terrain)
        self.assertEqual("baseline", args.platform_preset)

        with (
            redirect_stdout(io.StringIO()),
            redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            cli_module.parse_args(["--steps", "10", "--duration-s", "20"])

    def test_replay_metadata_remains_viewer_compatible(self) -> None:
        result = run_simulation(
            build_default_scenario(), SimulationConfig(steps=4, dt_s=0.5, seed=9)
        )
        replay_document = build_replay_document_from_result(result)

        self.assertEqual(4, replay_document["meta"]["frame_count"])
        self.assertIn("terrain", replay_document["meta"])
        self.assertIn("terrain_summary", replay_document["meta"])
        self.assertIn("environment_id", replay_document["meta"])
        self.assertIn("land_cover_legend", replay_document["meta"])
        self.assertIn("platform", replay_document["meta"])
        self.assertIn("observation_generation", replay_document["meta"])
        self.assertIn("scenario_options", replay_document["meta"])
        self.assertIn("requested_duration_s", replay_document["meta"])
        self.assertIn("actual_duration_s", replay_document["meta"])
        self.assertIn("terrain_preset", replay_document["meta"]["scenario_options"])
        self.assertIn("weather_preset", replay_document["meta"]["scenario_options"])
        self.assertIn("clean_terrain", replay_document["meta"]["scenario_options"])
        self.assertIn("platform_preset", replay_document["meta"]["scenario_options"])
        self.assertIn("ground_station_count", replay_document["meta"]["scenario_options"])

    def test_mixed_mode_replay_metadata_tracks_follow_and_search(self) -> None:
        scenario = build_default_scenario(seed=7)
        result = run_simulation(scenario, SimulationConfig(steps=6, dt_s=0.25, seed=7))
        replay_document = build_replay_document_from_result(result)

        self.assertEqual("follow", replay_document["meta"]["drone_planner_modes"]["drone-east"])
        self.assertEqual("search", replay_document["meta"]["drone_planner_modes"]["drone-north"])
        self.assertEqual(
            "asset-lynx", replay_document["meta"]["drone_target_assignments"]["drone-east"]
        )
        self.assertEqual(
            "asset-orca", replay_document["meta"]["drone_target_assignments"]["drone-north"]
        )
        self.assertEqual(
            "mixed", replay_document["meta"]["target_motion_assignments"]["asset-lynx"]
        )
        self.assertEqual(
            "mixed", replay_document["meta"]["target_motion_assignments"]["asset-orca"]
        )

    def test_weather_preset_changes_target_motion_and_is_preserved(self) -> None:
        clear_scenario = build_default_scenario(
            ScenarioOptions(
                map_preset="medium",
                target_motion_preset="transit",
                drone_mode_preset="search",
                weather_preset="clear",
            ),
            seed=7,
        )
        storm_scenario = build_default_scenario(
            ScenarioOptions(
                map_preset="medium",
                target_motion_preset="transit",
                drone_mode_preset="search",
                weather_preset="storm",
            ),
            seed=7,
        )

        clear_truth = clear_scenario.truths(25.0)[0]
        storm_truth = storm_scenario.truths(25.0)[0]

        self.assertFalse(np.allclose(clear_truth.position[:2], storm_truth.position[:2]))
        self.assertEqual("clear", clear_scenario.options.weather_preset)
        self.assertEqual("storm", storm_scenario.options.weather_preset)

    def test_config_file_overrides_runtime_defaults(self) -> None:
        constants = SimulationConstants.from_dict(
            {
                "dynamics": {
                    "default_duration_s": 2.0,
                    "default_dt_s": 0.5,
                    "default_seed": 99,
                }
            }
        )
        with TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "sim-config.json"
            replay_path = Path(temp_dir) / "replay.json"
            config_path.write_text(json.dumps(constants.to_dict()), encoding="utf-8")
            args = cli_module.parse_args(
                [
                    "--config-file",
                    str(config_path),
                    "--replay",
                    str(replay_path),
                    "--target-motion",
                    "loiter",
                ]
            )

            with redirect_stdout(io.StringIO()):
                cli_module._import_sim_module().run_from_args(args)

            replay_document = load_replay_document(str(replay_path))
            self.assertEqual(5, replay_document["meta"]["frame_count"])
            self.assertEqual(99, replay_document["meta"]["seed"])
            self.assertEqual(
                "loiter", replay_document["meta"]["scenario_options"]["target_motion_preset"]
            )

    def test_simulation_adapter_writes_csv_and_replay(self) -> None:
        with TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "metrics.csv"
            replay_path = Path(temp_dir) / "replay.json"
            with redirect_stdout(io.StringIO()):
                simulate(
                    steps=12,
                    dt=0.5,
                    seed=5,
                    csv_path=str(csv_path),
                    replay_path=str(replay_path),
                    map_preset="small",
                    target_motion="mixed",
                    drone_mode="mixed",
                )

            self.assertTrue(csv_path.exists())
            self.assertTrue(replay_path.exists())
            self.assertIn("time_s", csv_path.read_text(encoding="utf-8"))
            replay_document = load_replay_document(str(replay_path))
            self.assertEqual(12, replay_document["meta"]["frame_count"])
            self.assertEqual("small", replay_document["meta"]["scenario_options"]["map_preset"])


class TestJPDAMode(unittest.TestCase):
    """JPDA association mode integrates end-to-end through the Rust daemon."""

    def setUp(self) -> None:
        self.service = TrackingService(config=TrackerConfig(data_association_mode="jpda"))
        self.addCleanup(self.service.close)

    def test_jpda_config_is_accepted_by_daemon(self) -> None:
        self.assertEqual("jpda", self.service.config.data_association_mode)

    def test_jpda_ingest_frame_returns_platform_frame(self) -> None:
        from argusnet.core.types import PlatformFrame

        nodes = [
            NodeState("gs-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, 0.0),
            NodeState("gs-b", vec3(200.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, 0.0),
        ]
        target_pos = vec3(100.0, 80.0, 30.0)
        obs = [
            bearing("gs-a", "asset-x", nodes[0].position, target_pos, 0.0),
            bearing("gs-b", "asset-x", nodes[1].position, target_pos, 0.0),
        ]
        frame = self.service.ingest_frame(0.0, node_states=nodes, observations=obs)
        self.assertIsInstance(frame, PlatformFrame)
        self.assertAlmostEqual(0.0, frame.timestamp_s)

    def test_jpda_produces_track_across_multiple_frames(self) -> None:
        target_pos = vec3(100.0, 80.0, 30.0)
        for step in range(6):
            t = float(step) * 0.25
            nodes = [
                NodeState("gs-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, t),
                NodeState("gs-b", vec3(200.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, t),
            ]
            obs = [
                bearing("gs-a", "asset-x", nodes[0].position, target_pos, t),
                bearing("gs-b", "asset-x", nodes[1].position, target_pos, t),
            ]
            self.service.ingest_frame(
                t,
                node_states=nodes,
                observations=obs,
                truths=[TruthState("asset-x", target_pos, vec3(0.0, 0.0, 0.0), t)],
            )
        self.assertGreater(len(self.service.tracks), 0)


class TestTrackStream(unittest.TestCase):
    """track_stream() wraps the bidirectional TrackStream gRPC RPC."""

    def setUp(self) -> None:
        self.service = TrackingService(retain_history=True)
        self.addCleanup(self.service.close)

    @staticmethod
    def _frame_inputs(steps: int = 4) -> list[tuple]:
        target_pos = vec3(50.0, 40.0, 20.0)
        result = []
        for step in range(steps):
            t = float(step) * 0.25
            nodes = [
                NodeState("gs-a", vec3(0.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, t),
                NodeState("gs-b", vec3(100.0, 0.0, 0.0), vec3(0.0, 0.0, 0.0), False, t),
            ]
            obs = [
                bearing("gs-a", "asset-a", nodes[0].position, target_pos, t),
                bearing("gs-b", "asset-a", nodes[1].position, target_pos, t),
            ]
            truths = [TruthState("asset-a", target_pos, vec3(0.0, 0.0, 0.0), t)]
            result.append((t, nodes, obs, truths))
        return result

    def test_track_stream_yields_one_frame_per_input(self) -> None:
        frames = list(self.service.track_stream(self._frame_inputs(steps=4)))
        self.assertEqual(4, len(frames))

    def test_track_stream_frames_are_platform_frames(self) -> None:
        from argusnet.core.types import PlatformFrame

        for frame in self.service.track_stream(self._frame_inputs(steps=2)):
            self.assertIsInstance(frame, PlatformFrame)

    def test_track_stream_updates_service_state(self) -> None:
        inputs = self._frame_inputs(steps=4)
        last_frame = None
        for frame in self.service.track_stream(inputs):
            last_frame = frame
        self.assertIsNotNone(last_frame)
        self.assertIs(last_frame, self.service.latest_frame())

    def test_track_stream_appends_to_history_when_retain_history(self) -> None:
        list(self.service.track_stream(self._frame_inputs(steps=3)))
        self.assertEqual(3, len(self.service.history))

    def test_track_stream_timestamps_increase(self) -> None:
        frames = list(self.service.track_stream(self._frame_inputs(steps=4)))
        timestamps = [f.timestamp_s for f in frames]
        self.assertEqual(timestamps, sorted(timestamps))


if __name__ == "__main__":
    unittest.main()
