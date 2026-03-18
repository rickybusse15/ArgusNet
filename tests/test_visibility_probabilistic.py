"""Tests for probabilistic line-of-sight detection in visibility.py.

Tests cover:
- DetectionResult dataclass construction and field access
- free_space_path_loss helper
- compute_weather_factor helper
- compute_effective_noise helper
- identify_dominant_loss helper
- compute_detection_probability via a mocked EnvironmentQuery
"""
from __future__ import annotations

import math
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from smart_tracker.environment import (
    DetectionResult,
    EnvironmentQuery,
    SensorVisibilityModel,
    VisibilityResult,
    compute_effective_noise,
    compute_weather_factor,
    free_space_path_loss,
    identify_dominant_loss,
)
from smart_tracker.weather import (
    AtmosphericConditions,
    PrecipitationModel,
    WeatherModel,
    WindModel,
)


# -----------------------------------------------------------------------
# DetectionResult dataclass
# -----------------------------------------------------------------------

class TestDetectionResult(unittest.TestCase):
    """DetectionResult is a frozen dataclass with the expected fields."""

    def test_fields_populated(self):
        vis = VisibilityResult(visible=True, transmittance=0.9, noise_multiplier=1.2)
        dr = DetectionResult(
            p_d=0.75,
            vis_result=vis,
            effective_noise_multiplier=1.4,
            dominant_loss_factor="range",
        )
        self.assertAlmostEqual(dr.p_d, 0.75)
        self.assertIs(dr.vis_result, vis)
        self.assertAlmostEqual(dr.effective_noise_multiplier, 1.4)
        self.assertEqual(dr.dominant_loss_factor, "range")

    def test_frozen(self):
        dr = DetectionResult(
            p_d=0.5,
            vis_result=VisibilityResult(visible=True),
            effective_noise_multiplier=1.0,
            dominant_loss_factor="range",
        )
        with self.assertRaises(AttributeError):
            dr.p_d = 0.9  # type: ignore[misc]

    def test_blocked_result(self):
        vis = VisibilityResult(visible=False, blocker_type="terrain", transmittance=0.0)
        dr = DetectionResult(
            p_d=0.0,
            vis_result=vis,
            effective_noise_multiplier=2.5,
            dominant_loss_factor="blocked",
        )
        self.assertAlmostEqual(dr.p_d, 0.0)
        self.assertEqual(dr.dominant_loss_factor, "blocked")
        self.assertFalse(dr.vis_result.visible)


# -----------------------------------------------------------------------
# free_space_path_loss
# -----------------------------------------------------------------------

class TestFreeSpacePathLoss(unittest.TestCase):

    def test_zero_range_gives_one(self):
        self.assertAlmostEqual(free_space_path_loss(0.0), 1.0)

    def test_negative_range_gives_one(self):
        self.assertAlmostEqual(free_space_path_loss(-5.0), 1.0)

    def test_close_range_high(self):
        """A target at 10 m should have path loss near 1.0."""
        loss = free_space_path_loss(10.0, max_range_m=2000.0)
        self.assertGreater(loss, 0.8)

    def test_far_range_lower(self):
        """A target at 1800 m should have substantially lower path-loss factor."""
        close = free_space_path_loss(50.0, max_range_m=2000.0)
        far = free_space_path_loss(1800.0, max_range_m=2000.0)
        self.assertGreater(close, far)

    def test_monotonically_decreasing(self):
        prev = 1.0
        for r in [10, 100, 500, 1000, 2000, 5000]:
            val = free_space_path_loss(float(r))
            self.assertLessEqual(val, prev + 1e-12)
            prev = val

    def test_always_positive(self):
        for r in [0, 1, 100, 10000, 1e6]:
            self.assertGreaterEqual(free_space_path_loss(float(r)), 0.0)


# -----------------------------------------------------------------------
# compute_weather_factor
# -----------------------------------------------------------------------

class TestComputeWeatherFactor(unittest.TestCase):

    def test_no_weather_gives_one(self):
        self.assertAlmostEqual(compute_weather_factor(500.0, None), 1.0)

    def test_clear_weather_near_one_at_short_range(self):
        weather = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=15000.0),
            precipitation=PrecipitationModel(precip_type="none"),
        )
        factor = compute_weather_factor(50.0, weather)
        self.assertGreater(factor, 0.95)

    def test_fog_reduces_factor(self):
        clear = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=15000.0),
            precipitation=PrecipitationModel(precip_type="none"),
        )
        fog = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=200.0),
            precipitation=PrecipitationModel(precip_type="fog"),
        )
        range_m = 500.0
        clear_f = compute_weather_factor(range_m, clear)
        fog_f = compute_weather_factor(range_m, fog)
        self.assertGreater(clear_f, fog_f)

    def test_heavy_rain_reduces_factor(self):
        clear = WeatherModel()
        rain = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=2000.0),
            precipitation=PrecipitationModel(precip_type="rain", rate_mmph=25.0),
        )
        range_m = 800.0
        self.assertGreater(
            compute_weather_factor(range_m, clear),
            compute_weather_factor(range_m, rain),
        )

    def test_factor_non_negative(self):
        fog = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=50.0),
            precipitation=PrecipitationModel(precip_type="fog"),
        )
        self.assertGreaterEqual(compute_weather_factor(10000.0, fog), 0.0)


# -----------------------------------------------------------------------
# compute_effective_noise
# -----------------------------------------------------------------------

class TestComputeEffectiveNoise(unittest.TestCase):

    def test_at_zero_range_minimum(self):
        noise = compute_effective_noise(0.0, 1.0, None)
        self.assertGreaterEqual(noise, 1.0)

    def test_increases_with_range(self):
        near = compute_effective_noise(100.0, 1.0, None)
        far = compute_effective_noise(1500.0, 1.0, None)
        self.assertGreater(far, near)

    def test_weather_increases_noise(self):
        rain = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=2000.0),
            precipitation=PrecipitationModel(precip_type="rain", rate_mmph=25.0),
        )
        no_weather = compute_effective_noise(500.0, 1.0, None)
        with_weather = compute_effective_noise(500.0, 1.0, rain)
        self.assertGreater(with_weather, no_weather)

    def test_land_cover_noise_multiplied(self):
        low = compute_effective_noise(500.0, 1.0, None)
        high = compute_effective_noise(500.0, 1.45, None)
        self.assertGreater(high, low)

    def test_always_at_least_one(self):
        self.assertGreaterEqual(compute_effective_noise(0.0, 0.5, None), 1.0)


# -----------------------------------------------------------------------
# identify_dominant_loss
# -----------------------------------------------------------------------

class TestIdentifyDominantLoss(unittest.TestCase):

    def test_blocked(self):
        self.assertEqual(
            identify_dominant_loss(0.8, 0.9, 0.95, is_blocked=True),
            "blocked",
        )

    def test_range_dominant(self):
        self.assertEqual(
            identify_dominant_loss(0.3, 0.95, 0.98, is_blocked=False),
            "range",
        )

    def test_transmittance_dominant(self):
        self.assertEqual(
            identify_dominant_loss(0.95, 0.2, 0.98, is_blocked=False),
            "transmittance",
        )

    def test_weather_dominant(self):
        self.assertEqual(
            identify_dominant_loss(0.95, 0.95, 0.1, is_blocked=False),
            "weather",
        )


# -----------------------------------------------------------------------
# compute_detection_probability via mocked EnvironmentQuery
# -----------------------------------------------------------------------

def _make_mock_env_query(vis_result: VisibilityResult) -> EnvironmentQuery:
    """Build a minimal EnvironmentQuery mock that returns *vis_result* from los()."""
    mock_env = MagicMock()
    query = EnvironmentQuery.__new__(EnvironmentQuery)
    query.environment = mock_env
    # Patch the los method to return the desired VisibilityResult
    query.los = MagicMock(return_value=vis_result)  # type: ignore[method-assign]
    return query


class TestComputeDetectionProbability(unittest.TestCase):
    """Integration tests using a mocked EnvironmentQuery."""

    def test_close_range_clear_gives_high_pd(self):
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([20.0, 0.0, 10.0])

        result = query.compute_detection_probability(origin, target)

        self.assertGreater(result.p_d, 0.8)
        self.assertIsInstance(result, DetectionResult)
        self.assertTrue(result.vis_result.visible)

    def test_far_range_gives_lower_pd(self):
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        close_target = np.array([20.0, 0.0, 10.0])
        far_target = np.array([1800.0, 0.0, 10.0])

        close_result = query.compute_detection_probability(origin, close_target)
        far_result = query.compute_detection_probability(origin, far_target)

        self.assertGreater(close_result.p_d, far_result.p_d)

    def test_weather_reduces_pd(self):
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([800.0, 0.0, 10.0])

        result_clear = query.compute_detection_probability(origin, target, weather=None)
        fog = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=200.0),
            precipitation=PrecipitationModel(precip_type="fog"),
        )
        result_fog = query.compute_detection_probability(origin, target, weather=fog)

        self.assertGreater(result_clear.p_d, result_fog.p_d)

    def test_blocked_los_gives_near_zero_pd(self):
        vis = VisibilityResult(
            visible=False,
            blocker_type="terrain",
            transmittance=0.0,
            detection_multiplier=0.0,
            noise_multiplier=2.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([500.0, 0.0, 10.0])

        result = query.compute_detection_probability(origin, target)

        self.assertAlmostEqual(result.p_d, 0.0)
        self.assertEqual(result.dominant_loss_factor, "blocked")

    def test_vegetation_partial_transmittance(self):
        """Partial vegetation transmittance lowers P_d but is not zero."""
        vis = VisibilityResult(
            visible=True,
            transmittance=0.4,
            detection_multiplier=0.35,
            noise_multiplier=1.8,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([100.0, 0.0, 10.0])

        result = query.compute_detection_probability(origin, target)

        self.assertGreater(result.p_d, 0.0)
        self.assertLess(result.p_d, 1.0)

    def test_noise_multiplier_increases_with_range(self):
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        close_target = np.array([50.0, 0.0, 10.0])
        far_target = np.array([1500.0, 0.0, 10.0])

        close_result = query.compute_detection_probability(origin, close_target)
        far_result = query.compute_detection_probability(origin, far_target)

        self.assertGreater(
            far_result.effective_noise_multiplier,
            close_result.effective_noise_multiplier,
        )

    def test_detection_result_fields_populated(self):
        vis = VisibilityResult(
            visible=True,
            transmittance=0.85,
            detection_multiplier=0.78,
            noise_multiplier=1.3,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([300.0, 0.0, 10.0])

        result = query.compute_detection_probability(origin, target)

        self.assertIsInstance(result.p_d, float)
        self.assertIsInstance(result.vis_result, VisibilityResult)
        self.assertIsInstance(result.effective_noise_multiplier, float)
        self.assertIsInstance(result.dominant_loss_factor, str)
        self.assertIn(
            result.dominant_loss_factor,
            {"range", "transmittance", "weather", "blocked"},
        )
        self.assertGreaterEqual(result.p_d, 0.0)
        self.assertLessEqual(result.p_d, 1.0)
        self.assertGreaterEqual(result.effective_noise_multiplier, 1.0)

    def test_sensor_profile_passed_to_los(self):
        """Verify that the sensor_profile parameter is forwarded to los()."""
        vis = VisibilityResult(visible=True)
        query = _make_mock_env_query(vis)
        profile = SensorVisibilityModel.optical_default()
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([100.0, 0.0, 10.0])

        query.compute_detection_probability(origin, target, sensor_profile=profile)

        query.los.assert_called_once()
        call_kwargs = query.los.call_args
        self.assertIs(call_kwargs.kwargs.get("sensor_profile") or call_kwargs[1].get("sensor_profile", None) or
                      (call_kwargs[0][2] if len(call_kwargs[0]) > 2 else None), profile)

    def test_heavy_rain_and_long_range(self):
        """Combined weather + range gives very low P_d."""
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([1800.0, 0.0, 10.0])

        storm = WeatherModel(
            atmosphere=AtmosphericConditions(visibility_m=1000.0),
            precipitation=PrecipitationModel(precip_type="rain", rate_mmph=50.0),
        )
        result = query.compute_detection_probability(origin, target, weather=storm)

        self.assertLess(result.p_d, 0.3)

    def test_max_range_parameter(self):
        """A shorter max_range_m makes path loss steeper."""
        vis = VisibilityResult(
            visible=True,
            transmittance=1.0,
            detection_multiplier=1.0,
            noise_multiplier=1.0,
        )
        query = _make_mock_env_query(vis)
        origin = np.array([0.0, 0.0, 10.0])
        target = np.array([500.0, 0.0, 10.0])

        short = query.compute_detection_probability(origin, target, max_range_m=600.0)
        long_ = query.compute_detection_probability(origin, target, max_range_m=5000.0)

        # With a shorter max-range normalisation, path loss is harsher at 500 m
        self.assertLess(short.p_d, long_.p_d)


if __name__ == "__main__":
    unittest.main()
