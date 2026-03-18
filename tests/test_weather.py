"""Test suite for the weather effects module.

Covers:
- Weather preset loading (all known presets)
- Wind vector computation at varying altitudes and times
- Atmospheric attenuation increases with range
- Precipitation reduces visibility
- Cloud layer LOS blocking logic
- Bearing-noise multiplicative factors are >= 1.0
- ``weather_from_preset`` raises ``ValueError`` for unknown names
- Flight speed penalty is within (0, 1]
- Edge cases (zero range, zero rate, extreme values)
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from smart_tracker.weather import (
    AtmosphericConditions,
    CloudLayer,
    KNOWN_WEATHER_PRESETS,
    PrecipitationModel,
    WEATHER_PRESETS,
    WeatherModel,
    WindModel,
    weather_from_preset,
)


class TestWeatherPresets(unittest.TestCase):
    """Every preset must load without error and have sensible defaults."""

    def test_all_presets_load(self) -> None:
        for name in KNOWN_WEATHER_PRESETS:
            with self.subTest(preset=name):
                model = weather_from_preset(name)
                self.assertIsInstance(model, WeatherModel)

    def test_preset_keys_match_known_set(self) -> None:
        self.assertEqual(set(WEATHER_PRESETS.keys()), set(KNOWN_WEATHER_PRESETS))

    def test_clear_preset_has_no_precipitation(self) -> None:
        clear = weather_from_preset("clear")
        self.assertEqual(clear.precipitation.precip_type, "none")
        self.assertAlmostEqual(clear.precipitation.rate_mmph, 0.0)

    def test_heavy_rain_preset_has_rain(self) -> None:
        heavy = weather_from_preset("heavy_rain")
        self.assertEqual(heavy.precipitation.precip_type, "rain")
        self.assertGreater(heavy.precipitation.rate_mmph, 10.0)

    def test_fog_preset_low_visibility(self) -> None:
        fog = weather_from_preset("fog")
        self.assertLess(fog.atmosphere.visibility_m, 500.0)

    def test_storm_preset_high_wind(self) -> None:
        storm = weather_from_preset("storm")
        self.assertGreater(storm.wind.base_speed_mps, 15.0)

    def test_snow_preset_negative_temperature(self) -> None:
        snow = weather_from_preset("snow")
        self.assertLess(snow.atmosphere.temperature_c, 0.0)

    def test_invalid_preset_raises_value_error(self) -> None:
        with self.assertRaises(ValueError):
            weather_from_preset("nonexistent_weather")


class TestWindModel(unittest.TestCase):
    """Wind vector computation at various altitudes and times."""

    def test_zero_wind(self) -> None:
        wind = WindModel()
        vec = wind.wind_at(0.0, 0.0)
        np.testing.assert_allclose(vec, [0.0, 0.0, 0.0], atol=1e-12)

    def test_wind_has_no_vertical_component(self) -> None:
        wind = WindModel(base_speed_mps=10.0, base_heading_rad=math.pi / 4)
        vec = wind.wind_at(500.0, 10.0)
        self.assertAlmostEqual(vec[2], 0.0)

    def test_wind_speed_increases_with_altitude(self) -> None:
        wind = WindModel(
            base_speed_mps=5.0,
            base_heading_rad=0.0,
            gust_amplitude_mps=0.0,
            altitude_scaling=0.5,
        )
        low = wind.wind_at(0.0, 0.0)
        high = wind.wind_at(200.0, 0.0)
        speed_low = float(np.linalg.norm(low))
        speed_high = float(np.linalg.norm(high))
        self.assertGreater(speed_high, speed_low)

    def test_gust_oscillation(self) -> None:
        wind = WindModel(
            base_speed_mps=5.0,
            base_heading_rad=0.0,
            gust_amplitude_mps=3.0,
            gust_period_s=10.0,
            altitude_scaling=0.0,
        )
        # At t=0 gust is sin(0)=0, at t=T/4 gust is sin(pi/2)=1 (peak)
        speed_t0 = float(np.linalg.norm(wind.wind_at(0.0, 0.0)))
        speed_peak = float(np.linalg.norm(wind.wind_at(0.0, 2.5)))  # T/4
        self.assertAlmostEqual(speed_t0, 5.0, places=4)
        self.assertAlmostEqual(speed_peak, 8.0, places=4)

    def test_from_north_wind_pushes_south(self) -> None:
        wind = WindModel(
            base_speed_mps=10.0,
            base_heading_rad=0.0,  # from north
            gust_amplitude_mps=0.0,
            altitude_scaling=0.0,
        )
        vec = wind.wind_at(0.0, 0.0)
        # "from north" means air moves south: north component should be negative
        self.assertAlmostEqual(vec[0], 0.0, places=6)
        self.assertLess(vec[1], 0.0)

    def test_gust_period_zero_disables_gust(self) -> None:
        wind = WindModel(
            base_speed_mps=5.0,
            base_heading_rad=0.0,
            gust_amplitude_mps=10.0,
            gust_period_s=0.0,
            altitude_scaling=0.0,
        )
        speed = float(np.linalg.norm(wind.wind_at(0.0, 100.0)))
        self.assertAlmostEqual(speed, 5.0, places=4)

    def test_negative_altitude_treated_as_zero(self) -> None:
        wind = WindModel(
            base_speed_mps=5.0,
            altitude_scaling=0.5,
            gust_amplitude_mps=0.0,
        )
        vec_neg = wind.wind_at(-100.0, 0.0)
        vec_zero = wind.wind_at(0.0, 0.0)
        np.testing.assert_allclose(vec_neg, vec_zero, atol=1e-12)


class TestAtmosphericConditions(unittest.TestCase):
    """Atmospheric attenuation and refraction."""

    def test_zero_range_no_attenuation(self) -> None:
        atm = AtmosphericConditions()
        self.assertAlmostEqual(atm.attenuation_factor(0.0), 1.0)

    def test_attenuation_increases_with_range(self) -> None:
        atm = AtmosphericConditions(visibility_m=5000.0)
        near = atm.attenuation_factor(500.0)
        far = atm.attenuation_factor(5000.0)
        self.assertGreater(near, far)
        self.assertGreater(near, 0.0)
        self.assertLessEqual(near, 1.0)

    def test_attenuation_monotonically_decreasing(self) -> None:
        atm = AtmosphericConditions(visibility_m=8000.0)
        prev = 1.0
        for r in [100, 500, 1000, 3000, 8000, 15000]:
            val = atm.attenuation_factor(float(r))
            self.assertLessEqual(val, prev)
            prev = val

    def test_rf_attenuation_worse_than_optical_only(self) -> None:
        atm = AtmosphericConditions(visibility_m=5000.0, relative_humidity=0.9)
        optical = atm.attenuation_factor(3000.0, frequency_ghz=0.0)
        rf = atm.attenuation_factor(3000.0, frequency_ghz=24.0)
        self.assertLessEqual(rf, optical)

    def test_refraction_zero_for_zenith(self) -> None:
        atm = AtmosphericConditions()
        # elevation > pi/2 should return 0
        self.assertAlmostEqual(
            atm.refraction_offset_rad(math.pi, 1000.0), 0.0
        )

    def test_refraction_positive_near_horizon(self) -> None:
        atm = AtmosphericConditions()
        offset = atm.refraction_offset_rad(math.radians(2.0), 5000.0)
        self.assertGreater(offset, 0.0)

    def test_refraction_zero_at_zero_range(self) -> None:
        atm = AtmosphericConditions()
        self.assertAlmostEqual(
            atm.refraction_offset_rad(math.radians(10.0), 0.0), 0.0
        )

    def test_negative_range_no_attenuation(self) -> None:
        atm = AtmosphericConditions()
        self.assertAlmostEqual(atm.attenuation_factor(-100.0), 1.0)


class TestPrecipitationModel(unittest.TestCase):
    """Precipitation visibility reduction and sensor noise."""

    def test_none_no_reduction(self) -> None:
        p = PrecipitationModel(precip_type="none")
        self.assertAlmostEqual(p.visibility_reduction_factor(), 1.0)
        self.assertAlmostEqual(p.sensor_noise_multiplier(), 1.0)

    def test_rain_reduces_visibility(self) -> None:
        p = PrecipitationModel(precip_type="rain", rate_mmph=10.0)
        factor = p.visibility_reduction_factor()
        self.assertGreater(factor, 0.0)
        self.assertLess(factor, 1.0)

    def test_heavier_rain_reduces_more(self) -> None:
        light = PrecipitationModel(precip_type="rain", rate_mmph=2.0)
        heavy = PrecipitationModel(precip_type="rain", rate_mmph=50.0)
        self.assertGreater(
            light.visibility_reduction_factor(),
            heavy.visibility_reduction_factor(),
        )

    def test_snow_reduces_visibility(self) -> None:
        p = PrecipitationModel(precip_type="snow", rate_mmph=5.0)
        self.assertLess(p.visibility_reduction_factor(), 1.0)

    def test_sleet_reduces_visibility(self) -> None:
        p = PrecipitationModel(precip_type="sleet", rate_mmph=5.0)
        self.assertLess(p.visibility_reduction_factor(), 1.0)

    def test_fog_reduces_visibility(self) -> None:
        p = PrecipitationModel(precip_type="fog")
        self.assertLess(p.visibility_reduction_factor(), 1.0)

    def test_sensor_noise_ge_one(self) -> None:
        for ptype in ("none", "rain", "snow", "sleet", "fog"):
            with self.subTest(precip_type=ptype):
                p = PrecipitationModel(precip_type=ptype, rate_mmph=10.0)
                self.assertGreaterEqual(p.sensor_noise_multiplier(), 1.0)

    def test_rain_noise_increases_with_rate(self) -> None:
        light = PrecipitationModel(precip_type="rain", rate_mmph=1.0)
        heavy = PrecipitationModel(precip_type="rain", rate_mmph=40.0)
        self.assertGreater(
            heavy.sensor_noise_multiplier(),
            light.sensor_noise_multiplier(),
        )

    def test_invalid_precip_type(self) -> None:
        with self.assertRaises(ValueError):
            PrecipitationModel(precip_type="hail")

    def test_zero_rate_rain_no_reduction(self) -> None:
        p = PrecipitationModel(precip_type="rain", rate_mmph=0.0)
        self.assertAlmostEqual(p.visibility_reduction_factor(), 1.0)

    def test_zero_rate_noise_is_one(self) -> None:
        p = PrecipitationModel(precip_type="rain", rate_mmph=0.0)
        self.assertAlmostEqual(p.sensor_noise_multiplier(), 1.0)


class TestCloudLayer(unittest.TestCase):
    """Cloud layer LOS blocking logic."""

    def test_full_coverage_blocks_los(self) -> None:
        cloud = CloudLayer(base_altitude_m=1000.0, top_altitude_m=2000.0, coverage=1.0)
        # Observer below, target above
        self.assertTrue(cloud.obscures_los(500.0, 3000.0))
        # Observer above, target below
        self.assertTrue(cloud.obscures_los(3000.0, 500.0))

    def test_partial_coverage_does_not_block(self) -> None:
        cloud = CloudLayer(base_altitude_m=1000.0, top_altitude_m=2000.0, coverage=0.9)
        self.assertFalse(cloud.obscures_los(500.0, 3000.0))

    def test_both_below_not_blocked(self) -> None:
        cloud = CloudLayer(base_altitude_m=1000.0, top_altitude_m=2000.0, coverage=1.0)
        self.assertFalse(cloud.obscures_los(200.0, 800.0))

    def test_both_above_not_blocked(self) -> None:
        cloud = CloudLayer(base_altitude_m=1000.0, top_altitude_m=2000.0, coverage=1.0)
        self.assertFalse(cloud.obscures_los(2500.0, 3000.0))

    def test_both_inside_cloud_not_blocked(self) -> None:
        cloud = CloudLayer(base_altitude_m=1000.0, top_altitude_m=2000.0, coverage=1.0)
        # Both inside the cloud slab: neither below base nor above top
        # so obscures_los should be False (low >= base)
        self.assertFalse(cloud.obscures_los(1200.0, 1800.0))

    def test_zero_coverage_never_blocks(self) -> None:
        cloud = CloudLayer(base_altitude_m=0.0, top_altitude_m=10000.0, coverage=0.0)
        self.assertFalse(cloud.obscures_los(0.0, 20000.0))


class TestWeatherModel(unittest.TestCase):
    """Composite weather model behaviour."""

    def test_visibility_at_range_decreases(self) -> None:
        model = weather_from_preset("clear")
        near = model.visibility_at_range(100.0)
        far = model.visibility_at_range(5000.0)
        self.assertGreater(near, far)

    def test_visibility_at_zero_range(self) -> None:
        model = weather_from_preset("clear")
        self.assertAlmostEqual(model.visibility_at_range(0.0), 1.0)

    def test_fog_visibility_much_lower(self) -> None:
        clear = weather_from_preset("clear")
        fog = weather_from_preset("fog")
        range_m = 500.0
        self.assertGreater(
            clear.visibility_at_range(range_m),
            fog.visibility_at_range(range_m),
        )

    def test_bearing_noise_scale_ge_one(self) -> None:
        for name in KNOWN_WEATHER_PRESETS:
            with self.subTest(preset=name):
                model = weather_from_preset(name)
                self.assertGreaterEqual(model.bearing_noise_scale(), 1.0)

    def test_clear_noise_lower_than_storm(self) -> None:
        clear = weather_from_preset("clear")
        storm = weather_from_preset("storm")
        self.assertLess(
            clear.bearing_noise_scale(),
            storm.bearing_noise_scale(),
        )

    def test_flight_speed_penalty_in_range(self) -> None:
        for name in KNOWN_WEATHER_PRESETS:
            with self.subTest(preset=name):
                model = weather_from_preset(name)
                penalty = model.flight_speed_penalty(100.0)
                self.assertGreater(penalty, 0.0)
                self.assertLessEqual(penalty, 1.0)

    def test_clear_speed_penalty_near_one(self) -> None:
        clear = weather_from_preset("clear")
        penalty = clear.flight_speed_penalty(0.0)
        self.assertGreater(penalty, 0.8)

    def test_storm_speed_penalty_low(self) -> None:
        storm = weather_from_preset("storm")
        penalty = storm.flight_speed_penalty(0.0)
        self.assertLess(penalty, 0.6)

    def test_default_weather_model(self) -> None:
        """Default WeatherModel should represent benign conditions."""
        model = WeatherModel()
        self.assertAlmostEqual(model.visibility_at_range(0.0), 1.0)
        self.assertAlmostEqual(model.bearing_noise_scale(), 1.0)
        self.assertAlmostEqual(model.flight_speed_penalty(0.0), 1.0)


if __name__ == "__main__":
    unittest.main()
