"""Tests for the GNSS noise models: white, Gauss-Markov colored, and outages."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.sensing.gnss import GNSSModel, GNSSSimulator, OutageSchedule, sample_gnss_position


class TestGNSSModel(unittest.TestCase):
    def test_sigma_scales_with_dop(self):
        low_dop = GNSSModel(hdop=1.0, vdop=1.0)
        high_dop = GNSSModel(hdop=3.0, vdop=4.0)
        self.assertGreater(high_dop.horizontal_sigma_m, low_dop.horizontal_sigma_m)
        self.assertGreater(high_dop.vertical_sigma_m, low_dop.vertical_sigma_m)

    def test_sample_gnss_position_is_seed_deterministic(self):
        model = GNSSModel()
        truth = np.array([100.0, 200.0, 50.0])
        fix_a = sample_gnss_position(truth, model, 1.0, np.random.default_rng(42))
        fix_b = sample_gnss_position(truth, model, 1.0, np.random.default_rng(42))
        np.testing.assert_array_equal(fix_a.position, fix_b.position)


class TestGNSSSimulator(unittest.TestCase):
    TRUTH = np.array([0.0, 0.0, 100.0])

    def _positions(self, sim: GNSSSimulator, n: int = 400, dt: float = 1.0) -> np.ndarray:
        out = []
        for k in range(n):
            fix = sim.sample(self.TRUTH, k * dt)
            if fix is not None:
                out.append(np.asarray(fix.position, dtype=float))
        return np.array(out)

    def test_same_seed_reproduces_fix_sequence(self):
        sim_a = GNSSSimulator(gm_sigma_m=2.0, seed=7)
        sim_b = GNSSSimulator(gm_sigma_m=2.0, seed=7)
        np.testing.assert_array_equal(self._positions(sim_a), self._positions(sim_b))

    def test_reset_state_reproduces_fix_sequence(self):
        sim = GNSSSimulator(gm_sigma_m=2.0, seed=7)
        first = self._positions(sim)
        sim.reset_state()
        second = self._positions(sim)
        np.testing.assert_array_equal(first, second)

    def test_colored_noise_is_temporally_correlated(self):
        """GM-biased error at 1 s spacing correlates; white error does not."""

        def lag1_correlation(sim: GNSSSimulator) -> float:
            errors = self._positions(sim, n=2000)[:, 0] - self.TRUTH[0]
            return float(np.corrcoef(errors[:-1], errors[1:])[0, 1])

        colored = lag1_correlation(GNSSSimulator(gm_sigma_m=8.0, gm_time_constant_s=120.0, seed=3))
        white = lag1_correlation(GNSSSimulator(gm_sigma_m=0.0, seed=3))
        self.assertGreater(colored, 0.5)
        self.assertLess(abs(white), 0.15)

    def test_outage_returns_none_and_is_deterministic(self):
        schedule = OutageSchedule(rate_per_hour=30.0, horizon_s=600.0, seed=11)
        windows = schedule.windows()
        repeat = OutageSchedule(rate_per_hour=30.0, horizon_s=600.0, seed=11).windows()
        self.assertEqual(windows, repeat)
        self.assertGreater(len(windows), 0)

        sim = GNSSSimulator(outages=schedule, seed=5)
        start, end = windows[0]
        inside = 0.5 * (start + end)
        self.assertIsNone(sim.sample(self.TRUTH, inside))
        self.assertTrue(sim.in_outage(inside))

    def test_fixes_resume_after_outage(self):
        schedule = OutageSchedule(rate_per_hour=30.0, horizon_s=600.0, seed=11)
        _, end = schedule.windows()[0]
        sim = GNSSSimulator(outages=schedule, seed=5)
        self.assertIsNotNone(sim.sample(self.TRUTH, end + 1.0))


if __name__ == "__main__":
    unittest.main()
