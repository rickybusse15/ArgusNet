"""Tests for advanced fusion features: CoordinatedTurnTrack3D, IMMTrack3D, ManagedTrack."""

from __future__ import annotations

import math
import unittest

import numpy as np

from argusnet.localization.state import (
    TRACK_STATE_COASTING,
    TRACK_STATE_CONFIRMED,
    TRACK_STATE_DELETED,
    TRACK_STATE_TENTATIVE,
    AdaptiveFilterConfig,
    CoordinatedTurnTrack3D,
    IMMTrack3D,
    KalmanTrack3D,
    ManagedTrack,
    TrackLifecycleConfig,
    _gaussian_likelihood,
)

# ---------------------------------------------------------------------------
# CoordinatedTurnTrack3D tests
# ---------------------------------------------------------------------------


class TestCoordinatedTurnTrack3D(unittest.TestCase):
    def test_initialize_state_shape(self):
        ct = CoordinatedTurnTrack3D.initialize(0.0, np.array([1.0, 2.0, 3.0]))
        self.assertEqual(ct.state.shape, (7,))
        self.assertEqual(ct.covariance.shape, (7, 7))
        np.testing.assert_array_equal(ct.state[:3], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(ct.state[6], 0.0)  # omega = 0

    def test_initialize_with_velocity(self):
        ct = CoordinatedTurnTrack3D.initialize(
            0.0,
            np.array([0.0, 0.0, 100.0]),
            velocity=np.array([10.0, 5.0, 0.0]),
        )
        np.testing.assert_array_almost_equal(ct.state[3:6], [10.0, 5.0, 0.0])

    def test_predict_straight_line(self):
        """With zero turn rate, CT should behave like CV."""
        ct = CoordinatedTurnTrack3D.initialize(
            0.0,
            np.array([0.0, 0.0, 100.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
        )
        ct.predict(1.0)
        # After 1s at 10 m/s east: x ≈ 10
        self.assertAlmostEqual(ct.state[0], 10.0, places=2)
        self.assertAlmostEqual(ct.state[1], 0.0, places=2)
        self.assertAlmostEqual(ct.timestamp_s, 1.0)

    def test_predict_turning(self):
        """With nonzero omega, the track should curve."""
        ct = CoordinatedTurnTrack3D.initialize(
            0.0,
            np.array([0.0, 0.0, 100.0]),
            velocity=np.array([10.0, 0.0, 0.0]),
        )
        ct.state[6] = 0.5  # 0.5 rad/s turn rate
        ct.predict(1.0)
        # After turning, y should be nonzero
        self.assertNotAlmostEqual(ct.state[1], 0.0, places=1)

    def test_predict_no_time_advance(self):
        ct = CoordinatedTurnTrack3D.initialize(0.0, np.array([1.0, 2.0, 3.0]))
        state_before = ct.state.copy()
        ct.predict(0.0)  # dt=0
        np.testing.assert_array_equal(ct.state, state_before)

    def test_predict_z_is_cv(self):
        """Z dimension should use constant-velocity regardless of omega."""
        ct = CoordinatedTurnTrack3D.initialize(
            0.0,
            np.array([0.0, 0.0, 100.0]),
            velocity=np.array([10.0, 0.0, 5.0]),
        )
        ct.state[6] = 1.0  # high turn rate
        ct.predict(2.0)
        self.assertAlmostEqual(ct.state[2], 110.0, places=1)

    def test_update_reduces_covariance(self):
        ct = CoordinatedTurnTrack3D.initialize(
            0.0,
            np.array([0.0, 0.0, 100.0]),
            position_std_m=50.0,
        )
        ct.predict(1.0)
        cov_before = np.trace(ct.covariance[:3, :3])
        ct.update_position(np.array([1.0, 0.0, 100.0]), measurement_std_m=5.0)
        cov_after = np.trace(ct.covariance[:3, :3])
        self.assertLess(cov_after, cov_before)

    def test_update_returns_nis(self):
        ct = CoordinatedTurnTrack3D.initialize(0.0, np.array([0.0, 0.0, 0.0]))
        ct.predict(1.0)
        nis = ct.update_position(np.array([5.0, 0.0, 0.0]), measurement_std_m=10.0)
        self.assertIsInstance(nis, float)
        self.assertGreaterEqual(nis, 0.0)

    def test_from_cv_track(self):
        cv = KalmanTrack3D.initialize(0.0, np.array([10.0, 20.0, 30.0]))
        ct = CoordinatedTurnTrack3D.from_cv_track(cv)
        np.testing.assert_array_almost_equal(ct.state[:6], cv.state)
        self.assertAlmostEqual(ct.state[6], 0.0)

    def test_to_cv_state(self):
        ct = CoordinatedTurnTrack3D.initialize(0.0, np.array([1.0, 2.0, 3.0]))
        state_6, cov_6x6 = ct.to_cv_state()
        self.assertEqual(state_6.shape, (6,))
        self.assertEqual(cov_6x6.shape, (6, 6))

    def test_snapshot(self):
        ct = CoordinatedTurnTrack3D.initialize(0.0, np.array([1.0, 2.0, 3.0]))
        snap = ct.snapshot("t1", 10.0, 5, 0)
        self.assertEqual(snap.track_id, "t1")
        np.testing.assert_array_almost_equal(snap.position, [1.0, 2.0, 3.0])


# ---------------------------------------------------------------------------
# IMMTrack3D tests
# ---------------------------------------------------------------------------


class TestIMMTrack3D(unittest.TestCase):
    def test_initialize(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        self.assertIsNotNone(imm.cv_track)
        self.assertIsNotNone(imm.ct_track)
        self.assertAlmostEqual(imm.mode_probabilities.sum(), 1.0)

    def test_state_is_weighted_combination(self):
        imm = IMMTrack3D.initialize(0.0, np.array([10.0, 20.0, 30.0]))
        expected = (
            imm.mode_probabilities[0] * imm.cv_track.state
            + imm.mode_probabilities[1] * imm.ct_track.state[:6]
        )
        np.testing.assert_array_almost_equal(imm.state, expected)

    def test_predict_advances_time(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        imm.predict(1.0)
        self.assertAlmostEqual(imm.timestamp_s, 1.0)

    def test_predict_changes_mode_probs(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        probs_before = imm.mode_probabilities.copy()
        imm.predict(1.0)
        # Mode probs should shift due to transition matrix
        self.assertFalse(np.allclose(imm.mode_probabilities, probs_before))

    def test_update_changes_state(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        imm.predict(1.0)
        state_before = imm.state.copy()
        imm.update_position(np.array([20.0, 0.0, 100.0]), measurement_std_m=5.0)
        self.assertFalse(np.allclose(imm.state, state_before))

    def test_update_adjusts_mode_probabilities(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        imm.predict(1.0)
        probs_before = imm.mode_probabilities.copy()
        imm.update_position(np.array([0.5, 0.0, 100.0]), measurement_std_m=10.0)
        # Mode probs should change after update
        self.assertFalse(np.allclose(imm.mode_probabilities, probs_before))

    def test_mode_probabilities_sum_to_one(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        for t in range(1, 20):
            imm.predict(float(t))
            imm.update_position(
                np.array([float(t) * 5.0, 0.0, 100.0]),
                measurement_std_m=10.0,
            )
            self.assertAlmostEqual(imm.mode_probabilities.sum(), 1.0, places=6)

    def test_covariance_is_6x6(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        self.assertEqual(imm.covariance.shape, (6, 6))

    def test_covariance_positive_semidefinite(self):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        imm.predict(1.0)
        imm.update_position(np.array([5.0, 0.0, 100.0]), 10.0)
        eigenvalues = np.linalg.eigvalsh(imm.covariance)
        self.assertTrue(np.all(eigenvalues >= -1e-10))

    def test_mode_probs_stay_valid_for_straight_flight(self):
        """Mode probabilities should remain valid (sum to 1, both > 0) throughout."""
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        for t in range(1, 30):
            imm.predict(float(t))
            imm.update_position(
                np.array([float(t) * 10.0, 0.0, 100.0]),
                measurement_std_m=5.0,
            )
            self.assertAlmostEqual(imm.mode_probabilities.sum(), 1.0, places=6)
            self.assertTrue(np.all(imm.mode_probabilities > 0))

    def test_adaptive_q_scaling_increases_for_maneuvering(self):
        """When measurements are inconsistent, Q scaling should increase."""
        config = AdaptiveFilterConfig(innovation_window=5, innovation_scale_factor=1.0)
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]), config=config)
        for t in range(1, 10):
            imm.predict(float(t))
            # Large, varying offsets to trigger high NIS
            offset = 100.0 * math.sin(t)
            imm.update_position(
                np.array([offset, offset, 100.0]),
                measurement_std_m=5.0,
            )
        self.assertGreater(imm._q_scale, 1.0)

    def test_snapshot(self):
        imm = IMMTrack3D.initialize(0.0, np.array([1.0, 2.0, 3.0]))
        snap = imm.snapshot("track-1", 10.0, 3, 0)
        self.assertEqual(snap.track_id, "track-1")

    def test_position_and_velocity_properties(self):
        imm = IMMTrack3D.initialize(0.0, np.array([5.0, 10.0, 15.0]))
        self.assertEqual(imm.position.shape, (3,))
        self.assertEqual(imm.velocity.shape, (3,))
        np.testing.assert_array_almost_equal(imm.position, [5.0, 10.0, 15.0])


# ---------------------------------------------------------------------------
# Gaussian likelihood helper
# ---------------------------------------------------------------------------


class TestGaussianLikelihood(unittest.TestCase):
    def test_zero_innovation_high_likelihood(self):
        S = np.eye(3) * 10.0
        innovation = np.array([0.0, 0.0, 0.0])
        lk = _gaussian_likelihood(innovation, S)
        self.assertGreater(lk, 1e-10)

    def test_large_innovation_low_likelihood(self):
        S = np.eye(3) * 1.0
        zero_innov = np.array([0.0, 0.0, 0.0])
        large_innov = np.array([100.0, 100.0, 100.0])
        lk_zero = _gaussian_likelihood(zero_innov, S)
        lk_large = _gaussian_likelihood(large_innov, S)
        self.assertGreater(lk_zero, lk_large)

    def test_returns_positive(self):
        S = np.eye(3) * 5.0
        lk = _gaussian_likelihood(np.array([1.0, 2.0, 3.0]), S)
        self.assertGreater(lk, 0.0)


# ---------------------------------------------------------------------------
# ManagedTrack lifecycle tests
# ---------------------------------------------------------------------------


class TestManagedTrack(unittest.TestCase):
    def _make_track(self, config=None):
        imm = IMMTrack3D.initialize(0.0, np.array([0.0, 0.0, 100.0]))
        return ManagedTrack(
            track_id="test-1",
            filter=imm,
            config=config or TrackLifecycleConfig(),
        )

    def test_initial_state_is_tentative(self):
        track = self._make_track()
        self.assertEqual(track.lifecycle_state, TRACK_STATE_TENTATIVE)
        self.assertTrue(track.is_alive)

    def test_confirmed_after_m_updates(self):
        config = TrackLifecycleConfig(confirmation_m=3, confirmation_n=5)
        track = self._make_track(config)
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        self.assertEqual(track.lifecycle_state, TRACK_STATE_CONFIRMED)

    def test_stays_tentative_before_m_updates(self):
        config = TrackLifecycleConfig(confirmation_m=3, confirmation_n=5)
        track = self._make_track(config)
        for t in range(1, 3):  # Only 2 updates
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        self.assertEqual(track.lifecycle_state, TRACK_STATE_TENTATIVE)

    def test_coasting_on_missed_observation(self):
        config = TrackLifecycleConfig(confirmation_m=2, confirmation_n=5)
        track = self._make_track(config)
        # Confirm the track
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        self.assertEqual(track.lifecycle_state, TRACK_STATE_CONFIRMED)
        # Miss an observation
        track.predict(4.0)
        track.mark_missed(4.0)
        self.assertEqual(track.lifecycle_state, TRACK_STATE_COASTING)

    def test_reconfirmed_after_coasting(self):
        config = TrackLifecycleConfig(confirmation_m=2, confirmation_n=5)
        track = self._make_track(config)
        # Confirm
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        # Coast
        track.predict(4.0)
        track.mark_missed(4.0)
        self.assertEqual(track.lifecycle_state, TRACK_STATE_COASTING)
        # Re-acquire
        track.predict(5.0)
        track.update(np.array([5.0, 0.0, 100.0]), 10.0, 5.0)
        self.assertEqual(track.lifecycle_state, TRACK_STATE_CONFIRMED)

    def test_deleted_after_max_coast_frames(self):
        config = TrackLifecycleConfig(
            confirmation_m=2,
            confirmation_n=5,
            max_coast_frames=3,
            max_coast_seconds=100.0,
        )
        track = self._make_track(config)
        # Confirm
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        # Miss enough frames
        for t in range(4, 7):
            track.predict(float(t))
            track.mark_missed(float(t))
        self.assertEqual(track.lifecycle_state, TRACK_STATE_DELETED)
        self.assertFalse(track.is_alive)

    def test_deleted_after_max_coast_seconds(self):
        config = TrackLifecycleConfig(
            confirmation_m=2,
            confirmation_n=5,
            max_coast_frames=100,
            max_coast_seconds=2.0,
        )
        track = self._make_track(config)
        # Confirm at t=1
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        # Miss at t=6 (3 seconds after last update at t=3)
        track.predict(6.0)
        track.mark_missed(6.0)
        self.assertEqual(track.lifecycle_state, TRACK_STATE_DELETED)

    def test_tentative_deleted_if_too_few_updates(self):
        config = TrackLifecycleConfig(confirmation_m=3, confirmation_n=5)
        track = self._make_track(config)
        # One update, then many misses
        track.predict(1.0)
        track.update(np.array([1.0, 0.0, 100.0]), 10.0, 1.0)
        for t in range(2, 7):
            track.predict(float(t))
            track.mark_missed(float(t))
        self.assertEqual(track.lifecycle_state, TRACK_STATE_DELETED)

    def test_quality_score_high_with_consistent_updates(self):
        config = TrackLifecycleConfig(confirmation_m=2, confirmation_n=5)
        track = self._make_track(config)
        for t in range(1, 6):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        self.assertGreater(track.quality_score, 0.8)

    def test_quality_score_low_with_many_misses(self):
        config = TrackLifecycleConfig(
            confirmation_m=2,
            confirmation_n=5,
            max_coast_frames=20,
            max_coast_seconds=100.0,
            min_quality_score=0.01,
        )
        track = self._make_track(config)
        # Confirm
        for t in range(1, 4):
            track.predict(float(t))
            track.update(np.array([float(t), 0.0, 100.0]), 10.0, float(t))
        # Many misses
        for t in range(4, 9):
            track.predict(float(t))
            track.mark_missed(float(t))
        self.assertLess(track.quality_score, 0.6)

    def test_update_count_increments(self):
        track = self._make_track()
        self.assertEqual(track.update_count, 0)
        track.predict(1.0)
        track.update(np.array([1.0, 0.0, 100.0]), 10.0, 1.0)
        self.assertEqual(track.update_count, 1)

    def test_stale_steps_resets_on_update(self):
        track = self._make_track()
        track.predict(1.0)
        track.mark_missed(1.0)
        self.assertEqual(track.stale_steps, 1)
        track.predict(2.0)
        track.update(np.array([2.0, 0.0, 100.0]), 10.0, 2.0)
        self.assertEqual(track.stale_steps, 0)

    def test_snapshot_returns_track_state(self):
        track = self._make_track()
        track.predict(1.0)
        track.update(np.array([1.0, 0.0, 100.0]), 10.0, 1.0)
        snap = track.snapshot()
        self.assertEqual(snap.track_id, "test-1")
        self.assertEqual(snap.update_count, 1)


# ---------------------------------------------------------------------------
# AdaptiveFilterConfig tests
# ---------------------------------------------------------------------------


class TestAdaptiveFilterConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = AdaptiveFilterConfig()
        self.assertAlmostEqual(cfg.cv_accel_std, 3.0)
        self.assertAlmostEqual(cfg.ct_accel_std, 8.0)
        self.assertAlmostEqual(cfg.cv_to_ct_prob, 0.05)
        self.assertAlmostEqual(cfg.ct_to_cv_prob, 0.10)
        self.assertEqual(cfg.innovation_window, 5)

    def test_custom_config(self):
        cfg = AdaptiveFilterConfig(
            cv_accel_std=5.0,
            ct_accel_std=10.0,
            innovation_window=10,
        )
        self.assertAlmostEqual(cfg.cv_accel_std, 5.0)
        self.assertEqual(cfg.innovation_window, 10)


# ---------------------------------------------------------------------------
# TrackLifecycleConfig tests
# ---------------------------------------------------------------------------


class TestTrackLifecycleConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = TrackLifecycleConfig()
        self.assertEqual(cfg.confirmation_m, 3)
        self.assertEqual(cfg.confirmation_n, 5)
        self.assertEqual(cfg.max_coast_frames, 10)
        self.assertAlmostEqual(cfg.max_coast_seconds, 5.0)

    def test_custom(self):
        cfg = TrackLifecycleConfig(confirmation_m=4, confirmation_n=8)
        self.assertEqual(cfg.confirmation_m, 4)
        self.assertEqual(cfg.confirmation_n, 8)


if __name__ == "__main__":
    unittest.main()
