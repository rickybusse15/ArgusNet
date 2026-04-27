"""Tests for sensor error models."""

from __future__ import annotations

import math
import unittest

import numpy as np

from argusnet.sensing.models.noise import (
    SensorBiasDrift,
    SensorErrorConfig,
    SensorModel,
    apply_bias_to_direction,
    atmospheric_attenuation,
    detection_probability,
    generate_false_alarms,
    range_dependent_bearing_noise,
    sensor_error_config_from_preset,
    snr_at_range,
)


class TestRangeDependentNoise(unittest.TestCase):
    def test_zero_range_returns_base(self):
        cfg = SensorErrorConfig()
        result = range_dependent_bearing_noise(0.015, 0.0, cfg)
        self.assertAlmostEqual(result, 0.015)

    def test_noise_increases_with_range(self):
        cfg = SensorErrorConfig(
            base_bearing_std_rad=0.01,
            range_noise_exponent=2.0,
            range_noise_reference_m=500.0,
        )
        noise_near = range_dependent_bearing_noise(0.01, 200.0, cfg)
        noise_far = range_dependent_bearing_noise(0.01, 1000.0, cfg)
        self.assertGreater(noise_far, noise_near)

    def test_noise_capped_at_max(self):
        cfg = SensorErrorConfig(max_bearing_std_rad=0.05)
        result = range_dependent_bearing_noise(0.01, 1e6, cfg)
        self.assertLessEqual(result, 0.05)

    def test_at_reference_range_equals_base(self):
        cfg = SensorErrorConfig(
            base_bearing_std_rad=0.02,
            range_noise_exponent=1.0,
            range_noise_reference_m=1000.0,
        )
        result = range_dependent_bearing_noise(0.02, 1000.0, cfg)
        self.assertAlmostEqual(result, 0.02, places=6)


class TestAtmosphericAttenuation(unittest.TestCase):
    def test_zero_range_no_attenuation(self):
        cfg = SensorErrorConfig(atmospheric_attenuation_coeff=0.001)
        self.assertAlmostEqual(atmospheric_attenuation(0.0, cfg), 1.0)

    def test_attenuation_decreases_with_range(self):
        cfg = SensorErrorConfig(atmospheric_attenuation_coeff=0.001)
        t_near = atmospheric_attenuation(100.0, cfg)
        t_far = atmospheric_attenuation(1000.0, cfg)
        self.assertGreater(t_near, t_far)
        self.assertGreater(t_near, 0.0)
        self.assertGreater(t_far, 0.0)

    def test_zero_coeff_no_attenuation(self):
        cfg = SensorErrorConfig(atmospheric_attenuation_coeff=0.0)
        self.assertAlmostEqual(atmospheric_attenuation(5000.0, cfg), 1.0)

    def test_known_value(self):
        cfg = SensorErrorConfig(atmospheric_attenuation_coeff=0.001)
        # e^(-0.001 * 1000) = e^(-1) ≈ 0.3679
        result = atmospheric_attenuation(1000.0, cfg)
        self.assertAlmostEqual(result, math.exp(-1.0), places=6)


class TestSNR(unittest.TestCase):
    def test_snr_at_reference_range(self):
        cfg = SensorErrorConfig(snr_reference_db=30.0, range_noise_reference_m=1000.0)
        result = snr_at_range(1000.0, cfg)
        self.assertAlmostEqual(result, 30.0, places=6)

    def test_snr_decreases_with_range(self):
        cfg = SensorErrorConfig()
        snr_near = snr_at_range(500.0, cfg)
        snr_far = snr_at_range(2000.0, cfg)
        self.assertGreater(snr_near, snr_far)


class TestDetectionProbability(unittest.TestCase):
    def test_close_range_high_pd(self):
        cfg = SensorErrorConfig(detection_range_knee_m=800.0)
        pd = detection_probability(100.0, cfg)
        self.assertAlmostEqual(pd, 1.0)

    def test_pd_drops_past_knee(self):
        cfg = SensorErrorConfig(
            detection_range_knee_m=500.0,
            detection_range_falloff_m=200.0,
        )
        pd_at_knee = detection_probability(500.0, cfg)
        pd_past_knee = detection_probability(900.0, cfg)
        self.assertAlmostEqual(pd_at_knee, 1.0)
        self.assertLess(pd_past_knee, 1.0)

    def test_pd_never_below_minimum(self):
        cfg = SensorErrorConfig(min_detection_probability=0.1)
        pd = detection_probability(1e6, cfg)
        self.assertGreaterEqual(pd, 0.1)

    def test_atmosphere_reduces_pd(self):
        cfg = SensorErrorConfig()
        pd_clear = detection_probability(500.0, cfg, atmospheric_transmittance=1.0)
        pd_foggy = detection_probability(500.0, cfg, atmospheric_transmittance=0.5)
        self.assertGreaterEqual(pd_clear, pd_foggy)


class TestFalseAlarms(unittest.TestCase):
    def test_zero_rate_no_alarms(self):
        rng = np.random.default_rng(42)
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=0.0)
        pos = np.array([0.0, 0.0, 100.0])
        alarms = generate_false_alarms(rng, pos, cfg, 0.0, "node-1")
        self.assertEqual(len(alarms), 0)

    def test_high_rate_produces_alarms(self):
        rng = np.random.default_rng(42)
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=5.0)
        pos = np.array([0.0, 0.0, 100.0])
        # Run many scans to ensure at least some alarms
        total = 0
        for _ in range(100):
            alarms = generate_false_alarms(rng, pos, cfg, 0.0, "node-1")
            total += len(alarms)
        self.assertGreater(total, 0)

    def test_alarm_has_clutter_target_id(self):
        rng = np.random.default_rng(42)
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=10.0)
        pos = np.array([0.0, 0.0, 100.0])
        alarms = generate_false_alarms(rng, pos, cfg, 1.0, "node-1")
        for alarm in alarms:
            self.assertEqual(alarm.target_id, "clutter")
            self.assertEqual(alarm.node_id, "node-1")

    def test_alarm_direction_is_unit_vector(self):
        rng = np.random.default_rng(42)
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=10.0)
        pos = np.array([0.0, 0.0, 100.0])
        alarms = generate_false_alarms(rng, pos, cfg, 1.0, "node-1")
        for alarm in alarms:
            norm = float(np.linalg.norm(alarm.direction))
            self.assertAlmostEqual(norm, 1.0, places=4)


class TestBiasDrift(unittest.TestCase):
    def test_zero_drift_rate(self):
        drift = SensorBiasDrift(drift_rate_rad_per_s=0.0, seed=42)
        for t in range(100):
            bias = drift.step(float(t))
            self.assertAlmostEqual(bias, 0.0)

    def test_drift_stays_bounded(self):
        drift = SensorBiasDrift(
            drift_rate_rad_per_s=0.01,
            max_bias_rad=0.05,
            seed=42,
        )
        for t in range(10000):
            bias = drift.step(float(t) * 0.01)
            self.assertLessEqual(abs(bias), 0.05 + 1e-10)

    def test_drift_changes_over_time(self):
        drift = SensorBiasDrift(
            drift_rate_rad_per_s=0.001,
            max_bias_rad=0.1,
            seed=42,
        )
        biases = [drift.step(float(t)) for t in range(100)]
        # Not all the same
        self.assertGreater(len(set(round(b, 10) for b in biases)), 1)

    def test_reset_clears_state(self):
        drift = SensorBiasDrift(drift_rate_rad_per_s=0.01, max_bias_rad=0.1, seed=42)
        for t in range(50):
            drift.step(float(t))
        drift.reset()
        self.assertAlmostEqual(drift.current_bias_rad, 0.0)


class TestApplyBias(unittest.TestCase):
    def test_zero_bias_unchanged(self):
        rng = np.random.default_rng(42)
        direction = np.array([1.0, 0.0, 0.0])
        result = apply_bias_to_direction(direction, 0.0, rng)
        np.testing.assert_array_almost_equal(result, direction)

    def test_bias_changes_direction(self):
        rng = np.random.default_rng(42)
        direction = np.array([1.0, 0.0, 0.0])
        result = apply_bias_to_direction(direction, 0.1, rng)
        angle = np.arccos(np.clip(np.dot(result, direction), -1.0, 1.0))
        self.assertAlmostEqual(angle, 0.1, places=3)

    def test_preserves_magnitude(self):
        rng = np.random.default_rng(42)
        direction = np.array([2.0, 1.0, 0.5])
        orig_norm = np.linalg.norm(direction)
        result = apply_bias_to_direction(direction, 0.05, rng)
        self.assertAlmostEqual(float(np.linalg.norm(result)), float(orig_norm), places=6)


class TestSensorModel(unittest.TestCase):
    def test_effective_bearing_std_increases_with_range(self):
        model = SensorModel(config=SensorErrorConfig())
        std_near = model.effective_bearing_std(0.01, 100.0)
        std_far = model.effective_bearing_std(0.01, 2000.0)
        self.assertGreater(std_far, std_near)

    def test_should_detect_deterministic_close_range(self):
        model = SensorModel(
            config=SensorErrorConfig(
                min_detection_probability=0.95,
                detection_range_knee_m=5000.0,
            )
        )
        rng = np.random.default_rng(42)
        detections = sum(model.should_detect(rng, 100.0) for _ in range(100))
        self.assertGreaterEqual(detections, 90)

    def test_clutter_generation(self):
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=3.0)
        model = SensorModel(config=cfg)
        rng = np.random.default_rng(42)
        pos = np.array([0.0, 0.0, 50.0])
        total = 0
        for _ in range(50):
            clutter = model.generate_clutter(rng, pos, 0.0, "s1")
            total += len(clutter)
        self.assertGreater(total, 0)

    def test_initialize_creates_bias_drift(self):
        cfg = SensorErrorConfig(bias_drift_rate_rad_per_s=1e-4)
        model = SensorModel(config=cfg)
        model.initialize(seed=42)
        self.assertIsNotNone(model.bias_drift)

    def test_initialize_no_drift_when_zero(self):
        cfg = SensorErrorConfig(bias_drift_rate_rad_per_s=0.0)
        model = SensorModel(config=cfg)
        model.initialize(seed=42)
        self.assertIsNone(model.bias_drift)


class TestPresets(unittest.TestCase):
    def test_all_presets_exist(self):
        for name in ("ideal", "baseline", "degraded", "noisy"):
            cfg = sensor_error_config_from_preset(name)
            self.assertIsInstance(cfg, SensorErrorConfig)

    def test_unknown_preset_raises(self):
        with self.assertRaises(ValueError):
            sensor_error_config_from_preset("nonexistent")

    def test_ideal_has_low_noise(self):
        cfg = sensor_error_config_from_preset("ideal")
        self.assertLess(cfg.base_bearing_std_rad, 0.01)

    def test_noisy_has_high_false_alarm_rate(self):
        cfg = sensor_error_config_from_preset("noisy")
        self.assertGreater(cfg.false_alarm_rate_per_scan, 0.0)


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_clutter(self):
        cfg = SensorErrorConfig(false_alarm_rate_per_scan=2.0)
        pos = np.array([100.0, 200.0, 50.0])

        rng1 = np.random.default_rng(123)
        alarms1 = generate_false_alarms(rng1, pos, cfg, 0.0, "s1")

        rng2 = np.random.default_rng(123)
        alarms2 = generate_false_alarms(rng2, pos, cfg, 0.0, "s1")

        self.assertEqual(len(alarms1), len(alarms2))
        for a1, a2 in zip(alarms1, alarms2, strict=False):
            np.testing.assert_array_equal(a1.direction, a2.direction)

    def test_same_seed_same_bias_drift(self):
        drift1 = SensorBiasDrift(drift_rate_rad_per_s=0.01, seed=99)
        drift2 = SensorBiasDrift(drift_rate_rad_per_s=0.01, seed=99)
        for t in range(50):
            b1 = drift1.step(float(t) * 0.1)
            b2 = drift2.step(float(t) * 0.1)
            self.assertAlmostEqual(b1, b2)


if __name__ == "__main__":
    unittest.main()
