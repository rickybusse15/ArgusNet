"""Tests for the battery energy model, including wind-aware return reserve."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.planning.battery import BatteryModel


class TestReturnHomeCost(unittest.TestCase):
    def setUp(self):
        self.model = BatteryModel(capacity_wh=500.0, cruise_speed_m_per_s=20.0)
        self.home = np.array([0.0, 0.0, 100.0])

    def test_headwind_costs_more_than_tailwind(self):
        position = np.array([2000.0, 0.0, 100.0])  # homeward track is -x
        headwind = np.array([10.0, 0.0, 0.0])  # blowing +x, against the return leg
        tailwind = np.array([-10.0, 0.0, 0.0])  # blowing -x, with the return leg
        calm = self.model.return_home_cost_wh(position, self.home, None)
        into_wind = self.model.return_home_cost_wh(position, self.home, headwind)
        with_wind = self.model.return_home_cost_wh(position, self.home, tailwind)
        self.assertGreater(into_wind, calm)
        self.assertLess(with_wind, calm)

    def test_extreme_headwind_cost_is_bounded(self):
        position = np.array([2000.0, 0.0, 100.0])
        hurricane = np.array([100.0, 0.0, 0.0])
        cost = self.model.return_home_cost_wh(position, self.home, hurricane)
        # Groundspeed floors at 25% of cruise, so cost is at most 4x the calm cost.
        calm = self.model.return_home_cost_wh(position, self.home, None)
        self.assertLessEqual(cost, calm * 4.0 + 1e-9)

    def test_descending_home_costs_less_than_climbing_home(self):
        above_home = np.array([1000.0, 0.0, 300.0])  # must descend 200 m
        below_home = np.array([1000.0, 0.0, -100.0])  # must climb 200 m
        descend = self.model.return_home_cost_wh(above_home, self.home, None)
        climb = self.model.return_home_cost_wh(below_home, self.home, None)
        self.assertLess(descend, climb)


class TestDynamicReserve(unittest.TestCase):
    def setUp(self):
        self.model = BatteryModel(capacity_wh=500.0, cruise_speed_m_per_s=20.0)
        self.home = np.array([0.0, 0.0, 100.0])

    def test_static_reserve_is_a_floor_near_home(self):
        near_home = np.array([10.0, 0.0, 100.0])
        reserve = self.model.dynamic_reserve_wh(near_home, self.home, None)
        self.assertEqual(reserve, self.model.capacity_wh * self.model.reserve_fraction)

    def test_reserve_grows_with_distance_from_home(self):
        near = self.model.dynamic_reserve_wh(np.array([1000.0, 0.0, 100.0]), self.home, None)
        far = self.model.dynamic_reserve_wh(np.array([50_000.0, 0.0, 100.0]), self.home, None)
        self.assertGreaterEqual(far, near)

    def test_far_reserve_exceeds_static_floor(self):
        far_position = np.array([100_000.0, 0.0, 100.0])
        reserve = self.model.dynamic_reserve_wh(far_position, self.home, None)
        self.assertGreater(reserve, self.model.capacity_wh * self.model.reserve_fraction)


if __name__ == "__main__":
    unittest.main()
