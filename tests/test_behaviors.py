"""Test suite for the behaviors module.

Covers:
- FlightEnvelope min turn radius computation
- TurbulenceModel bounded perturbations and determinism
- LoiterBehavior orbital geometry
- TransitBehavior waypoint reachability and Hermite smoothness
- EvasiveBehavior altitude drops during evasion windows
- SearchPatternBehavior valid output shape
- CompositeTrajectory segment transitions
- build_target_trajectory factory with every preset
- Determinism: same seed produces identical trajectories
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from argusnet.simulation.behaviors import (
    BEHAVIOR_PRESETS,
    CompositeTrajectory,
    EvasiveBehavior,
    FlightEnvelope,
    LoiterBehavior,
    SearchPatternBehavior,
    TransitBehavior,
    TurbulenceModel,
    build_target_trajectory,
)

# Common bounds used by most factory tests.
_BOUNDS = {"x_min_m": -500.0, "x_max_m": 500.0, "y_min_m": -500.0, "y_max_m": 500.0}


def _valid_trajectory_output(pos: np.ndarray, vel: np.ndarray) -> None:
    """Assert basic shape/type invariants of a TrajectoryFn return value."""
    assert isinstance(pos, np.ndarray), f"position must be ndarray, got {type(pos)}"
    assert isinstance(vel, np.ndarray), f"velocity must be ndarray, got {type(vel)}"
    assert pos.shape == (3,), f"position shape must be (3,), got {pos.shape}"
    assert vel.shape == (3,), f"velocity shape must be (3,), got {vel.shape}"
    assert np.all(np.isfinite(pos)), f"position contains non-finite values: {pos}"
    assert np.all(np.isfinite(vel)), f"velocity contains non-finite values: {vel}"


class TestFlightEnvelope(unittest.TestCase):
    """FlightEnvelope configuration and derived quantities."""

    def test_defaults(self) -> None:
        env = FlightEnvelope()
        self.assertEqual(env.max_speed_mps, 50.0)
        self.assertEqual(env.min_speed_mps, 8.0)
        self.assertEqual(env.max_bank_angle_deg, 30.0)

    def test_min_turn_radius(self) -> None:
        env = FlightEnvelope(min_speed_mps=10.0, max_bank_angle_deg=30.0)
        g = 9.80665
        expected = 10.0**2 / (g * math.tan(math.radians(30.0)))
        self.assertAlmostEqual(env.min_turn_radius_m, expected, places=4)

    def test_min_turn_radius_at_speed(self) -> None:
        env = FlightEnvelope(max_bank_angle_deg=45.0)
        g = 9.80665
        speed = 25.0
        expected = speed**2 / (g * math.tan(math.radians(45.0)))
        self.assertAlmostEqual(env.min_turn_radius_at_speed(speed), expected, places=4)

    def test_turn_radius_increases_with_speed(self) -> None:
        env = FlightEnvelope()
        r_slow = env.min_turn_radius_at_speed(10.0)
        r_fast = env.min_turn_radius_at_speed(40.0)
        self.assertGreater(r_fast, r_slow)


class TestTurbulenceModel(unittest.TestCase):
    """TurbulenceModel produces bounded, deterministic perturbations."""

    def test_output_shape(self) -> None:
        tm = TurbulenceModel()
        p = tm.perturbation(0.0, np.array([0.0, 0.0, 100.0]))
        self.assertEqual(p.shape, (3,))
        self.assertTrue(np.all(np.isfinite(p)))

    def test_bounded(self) -> None:
        tm = TurbulenceModel(intensity=1.0)
        max_val = 0.0
        for t in np.linspace(0, 100, 200):
            p = tm.perturbation(t, np.array([t * 10.0, t * 5.0, 100.0]))
            max_val = max(max_val, float(np.max(np.abs(p))))
        # With intensity=1.0 the output should stay well within ~5 m/s.
        self.assertLess(max_val, 10.0)

    def test_determinism(self) -> None:
        tm = TurbulenceModel(seed=123)
        pos = np.array([50.0, 60.0, 150.0])
        p1 = tm.perturbation(5.0, pos)
        p2 = tm.perturbation(5.0, pos)
        np.testing.assert_array_equal(p1, p2)

    def test_zero_intensity(self) -> None:
        tm = TurbulenceModel(intensity=0.0)
        p = tm.perturbation(10.0, np.array([0.0, 0.0, 100.0]))
        # All components should be zero (0 * anything = 0).
        np.testing.assert_array_equal(p, np.zeros(3))


class TestLoiterBehavior(unittest.TestCase):
    """LoiterBehavior produces a circular orbit within the expected radius."""

    def setUp(self) -> None:
        self.center = np.array([100.0, 200.0, 150.0])
        self.radius = 80.0
        self.loiter = LoiterBehavior(
            center=self.center,
            radius_m=self.radius,
            speed_mps=20.0,
            clockwise=True,
        )

    def test_output_signature(self) -> None:
        pos, vel = self.loiter(0.0)
        _valid_trajectory_output(pos, vel)

    def test_stays_near_center(self) -> None:
        for t in np.linspace(0, 120, 50):
            pos, _ = self.loiter(t)
            dist_xy = math.hypot(pos[0] - self.center[0], pos[1] - self.center[1])
            self.assertAlmostEqual(
                dist_xy,
                self.radius,
                delta=1.0,
                msg=f"At t={t}, lateral distance {dist_xy} != radius {self.radius}",
            )

    def test_altitude_oscillation(self) -> None:
        altitudes = [self.loiter(t)[0][2] for t in np.linspace(0, 120, 100)]
        self.assertGreater(max(altitudes), self.center[2])
        self.assertLess(min(altitudes), self.center[2])

    def test_counterclockwise(self) -> None:
        ccw = LoiterBehavior(
            center=self.center, radius_m=self.radius, speed_mps=20.0, clockwise=False
        )
        pos0, _ = ccw(0.0)
        pos1, _ = ccw(1.0)
        _valid_trajectory_output(pos0, np.zeros(3))
        _valid_trajectory_output(pos1, np.zeros(3))
        # Just verify it produces different positions from clockwise.
        cw_pos1, _ = self.loiter(1.0)
        self.assertFalse(np.allclose(pos1, cw_pos1))


class TestTransitBehavior(unittest.TestCase):
    """TransitBehavior follows waypoints with smooth Hermite interpolation."""

    def setUp(self) -> None:
        self.waypoints = [
            np.array([0.0, 0.0, 100.0]),
            np.array([300.0, 0.0, 120.0]),
            np.array([300.0, 400.0, 110.0]),
        ]
        self.transit = TransitBehavior(
            waypoints=self.waypoints,
            speed_mps=25.0,
        )

    def test_output_signature(self) -> None:
        pos, vel = self.transit(0.0)
        _valid_trajectory_output(pos, vel)

    def test_starts_near_first_waypoint(self) -> None:
        pos, _ = self.transit(0.0)
        np.testing.assert_allclose(pos, self.waypoints[0], atol=1.0)

    def test_reaches_final_waypoint(self) -> None:
        # Total distance ~ 300 + 400 = 700 m at 25 m/s => ~28 s.
        # Hermite may deviate, so check at a time well past expected arrival.
        total_dist = sum(
            float(np.linalg.norm(self.waypoints[i + 1] - self.waypoints[i]))
            for i in range(len(self.waypoints) - 1)
        )
        arrival_t = total_dist / 25.0
        # Just past arrival, before loiter moves too far.
        pos, _ = self.transit(arrival_t + 0.01)
        dist = float(np.linalg.norm(pos - self.waypoints[-1]))
        self.assertLess(dist, 25.0, f"Expected to be near final waypoint, but distance is {dist}")

    def test_loiter_after_completion(self) -> None:
        t_late = 200.0
        pos, vel = self.transit(t_late)
        _valid_trajectory_output(pos, vel)
        # Should be loitering within ~20 m of the final waypoint.
        dist = float(np.linalg.norm(pos[:2] - self.waypoints[-1][:2]))
        self.assertLess(dist, 25.0)

    def test_smooth_trajectory(self) -> None:
        """Position should change smoothly (no large jumps between time steps)."""
        dt = 0.1
        prev_pos, _ = self.transit(0.0)
        max_jump = 0.0
        for t in np.arange(dt, 60.0, dt):
            pos, _ = self.transit(t)
            jump = float(np.linalg.norm(pos - prev_pos))
            max_jump = max(max_jump, jump)
            prev_pos = pos
        # At 25 m/s with dt=0.1, max displacement should be ~2.5 m per step.
        # Hermite overshoot at waypoints can push this higher, so use a
        # generous threshold.
        self.assertLess(
            max_jump, 25.0, f"Max position jump {max_jump} exceeds smoothness threshold."
        )

    def test_requires_at_least_two_waypoints(self) -> None:
        with self.assertRaises(ValueError):
            TransitBehavior(waypoints=[np.array([0, 0, 0])], speed_mps=10.0)


class TestEvasiveBehavior(unittest.TestCase):
    """EvasiveBehavior drops altitude during evasion windows."""

    def setUp(self) -> None:
        def base_traj(t: float):
            return (
                np.array([t * 10.0, 0.0, 200.0], dtype=float),
                np.array([10.0, 0.0, 0.0], dtype=float),
            )

        self.base = base_traj
        self.terrain_fn = lambda x, y: 50.0  # flat at 50 m
        self.evasive = EvasiveBehavior(
            base_trajectory=self.base,
            terrain_height_fn=self.terrain_fn,
            nap_of_earth_agl_m=40.0,
            evasion_probability=0.8,  # high so we almost certainly get evasion
            evasion_duration_s=15.0,
            seed=42,
        )

    def test_output_signature(self) -> None:
        pos, vel = self.evasive(0.0)
        _valid_trajectory_output(pos, vel)

    def test_altitude_drops_during_evasion(self) -> None:
        """At least one time sample should show altitude below the base 200 m."""
        found_drop = False
        for t in np.linspace(0, 100, 500):
            pos, _ = self.evasive(t)
            if pos[2] < 195.0:
                found_drop = True
                break
        self.assertTrue(found_drop, "Expected altitude drop during evasion but none found.")

    def test_noe_altitude(self) -> None:
        """During a fully-blended evasion the altitude should approach nap-of-earth."""
        # Walk through time and record the minimum altitude seen.
        min_alt = 999.0
        for t in np.linspace(0, 100, 1000):
            pos, _ = self.evasive(t)
            min_alt = min(min_alt, pos[2])
        # noe = terrain (50) + agl (40) = 90.  Allow some margin.
        self.assertLess(min_alt, 150.0, f"Minimum altitude {min_alt} never approached NOE.")

    def test_xy_unchanged(self) -> None:
        """Evasion should not change the XY track."""
        for t in [0.0, 5.0, 20.0, 50.0]:
            pos_evasive, _ = self.evasive(t)
            pos_base, _ = self.base(t)
            np.testing.assert_allclose(
                pos_evasive[:2], pos_base[:2], atol=1e-9, err_msg=f"XY mismatch at t={t}"
            )


class TestSearchPatternBehavior(unittest.TestCase):
    """SearchPatternBehavior produces valid trajectory outputs."""

    def test_expanding_square_output(self) -> None:
        sp = SearchPatternBehavior(
            center=np.array([0.0, 0.0, 150.0]),
            pattern="expanding_square",
            leg_length_m=100.0,
            speed_mps=20.0,
            altitude_m=150.0,
        )
        for t in np.linspace(0, 120, 30):
            pos, vel = sp(t)
            _valid_trajectory_output(pos, vel)

    def test_sector_output(self) -> None:
        sp = SearchPatternBehavior(
            center=np.array([0.0, 0.0, 150.0]),
            pattern="sector",
            leg_length_m=150.0,
            speed_mps=25.0,
            altitude_m=150.0,
        )
        for t in np.linspace(0, 120, 30):
            pos, vel = sp(t)
            _valid_trajectory_output(pos, vel)

    def test_invalid_pattern(self) -> None:
        with self.assertRaises(ValueError):
            SearchPatternBehavior(
                center=np.array([0.0, 0.0, 100.0]),
                pattern="spiral",
            )

    def test_starts_at_center(self) -> None:
        center = np.array([100.0, 200.0, 150.0])
        sp = SearchPatternBehavior(
            center=center,
            pattern="expanding_square",
            leg_length_m=100.0,
            speed_mps=20.0,
            altitude_m=150.0,
        )
        pos, _ = sp(0.0)
        # Should be at or very near the centre at t=0.
        np.testing.assert_allclose(pos, center, atol=1.0)


class TestCompositeTrajectory(unittest.TestCase):
    """CompositeTrajectory blends between segments smoothly."""

    def setUp(self) -> None:
        self.loiter1 = LoiterBehavior(
            center=np.array([0.0, 0.0, 100.0]),
            radius_m=50.0,
            speed_mps=15.0,
        )
        self.loiter2 = LoiterBehavior(
            center=np.array([500.0, 500.0, 120.0]),
            radius_m=60.0,
            speed_mps=18.0,
        )
        self.composite = CompositeTrajectory(
            segments=[
                (0.0, self.loiter1),
                (30.0, self.loiter2),
            ]
        )

    def test_output_signature(self) -> None:
        pos, vel = self.composite(0.0)
        _valid_trajectory_output(pos, vel)

    def test_uses_first_segment_initially(self) -> None:
        pos_comp, _ = self.composite(5.0)
        pos_seg1, _ = self.loiter1(5.0)
        np.testing.assert_allclose(pos_comp, pos_seg1, atol=1e-9)

    def test_uses_second_segment_later(self) -> None:
        # Well after the blend zone (30 + 2 = 32 s).
        pos_comp, _ = self.composite(50.0)
        pos_seg2, _ = self.loiter2(50.0)
        np.testing.assert_allclose(pos_comp, pos_seg2, atol=1e-9)

    def test_blend_zone_intermediate(self) -> None:
        """In the blend zone, the position should lie between the two segments."""
        blend_t = 31.0  # 1 s into the 2 s blend zone
        pos_comp, _ = self.composite(blend_t)
        pos_seg1, _ = self.loiter1(blend_t)
        pos_seg2, _ = self.loiter2(blend_t)
        # The composite position should not exactly match either segment.
        dist_to_1 = float(np.linalg.norm(pos_comp - pos_seg1))
        dist_to_2 = float(np.linalg.norm(pos_comp - pos_seg2))
        self.assertGreater(dist_to_1, 0.1, "Should differ from segment 1 in blend")
        self.assertGreater(dist_to_2, 0.1, "Should differ from segment 2 in blend")

    def test_requires_at_least_one_segment(self) -> None:
        with self.assertRaises(ValueError):
            CompositeTrajectory(segments=[])


class TestBuildTargetTrajectory(unittest.TestCase):
    """Factory function produces valid trajectories for every preset."""

    def test_all_presets_produce_valid_output(self) -> None:
        for preset in sorted(BEHAVIOR_PRESETS):
            with self.subTest(preset=preset):
                traj = build_target_trajectory(
                    preset=preset,
                    bounds=_BOUNDS,
                    altitude_m=150.0,
                    speed_mps=25.0,
                    seed=42,
                )
                for t in [0.0, 5.0, 30.0, 100.0]:
                    pos, vel = traj(t)
                    _valid_trajectory_output(pos, vel)

    def test_unknown_preset_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_target_trajectory(
                preset="nonexistent",
                bounds=_BOUNDS,
                altitude_m=150.0,
                speed_mps=25.0,
            )

    def test_evasive_with_terrain_fn(self) -> None:
        terrain_fn = lambda x, y: 20.0  # noqa: E731
        traj = build_target_trajectory(
            preset="evasive",
            bounds=_BOUNDS,
            altitude_m=200.0,
            speed_mps=30.0,
            seed=99,
            terrain_height_fn=terrain_fn,
        )
        pos, vel = traj(10.0)
        _valid_trajectory_output(pos, vel)


class TestDeterminism(unittest.TestCase):
    """Same seed produces identical trajectories."""

    def test_same_seed_same_output(self) -> None:
        for preset in sorted(BEHAVIOR_PRESETS):
            with self.subTest(preset=preset):
                traj_a = build_target_trajectory(
                    preset=preset,
                    bounds=_BOUNDS,
                    altitude_m=150.0,
                    speed_mps=25.0,
                    seed=77,
                )
                traj_b = build_target_trajectory(
                    preset=preset,
                    bounds=_BOUNDS,
                    altitude_m=150.0,
                    speed_mps=25.0,
                    seed=77,
                )
                for t in [0.0, 10.0, 50.0]:
                    pos_a, vel_a = traj_a(t)
                    pos_b, vel_b = traj_b(t)
                    np.testing.assert_array_equal(
                        pos_a,
                        pos_b,
                        err_msg=f"Position mismatch for {preset} at t={t}",
                    )
                    np.testing.assert_array_equal(
                        vel_a,
                        vel_b,
                        err_msg=f"Velocity mismatch for {preset} at t={t}",
                    )

    def test_different_seed_different_output(self) -> None:
        traj_a = build_target_trajectory(
            preset="loiter",
            bounds=_BOUNDS,
            altitude_m=150.0,
            speed_mps=25.0,
            seed=1,
        )
        traj_b = build_target_trajectory(
            preset="loiter",
            bounds=_BOUNDS,
            altitude_m=150.0,
            speed_mps=25.0,
            seed=999,
        )
        pos_a, _ = traj_a(10.0)
        pos_b, _ = traj_b(10.0)
        self.assertFalse(
            np.allclose(pos_a, pos_b), "Different seeds should produce different trajectories."
        )


if __name__ == "__main__":
    unittest.main()
