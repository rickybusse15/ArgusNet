from __future__ import annotations

import unittest

import numpy as np

from argusnet.core.types import BeliefCellStatus
from argusnet.mapping.belief import WorldBeliefQuery
from argusnet.mapping.occupancy import GridBounds, OccupancyGrid
from argusnet.mapping.world_map import WorldMap


class WorldBeliefQueryTest(unittest.TestCase):
    def setUp(self) -> None:
        self.bounds = GridBounds(0.0, 100.0, 0.0, 100.0, 10.0)
        self.world_map = WorldMap(self.bounds)
        self.world_map.add_scan_observation(np.array([25.0, 25.0, 30.0]), 4.0, 12.0)

    def test_unknown_observed_and_obstacle_states_are_distinct(self) -> None:
        occupancy = OccupancyGrid(self.bounds)
        occupancy.mark_occupied(25.0, 25.0, height_m=5.0)
        query = WorldBeliefQuery(self.world_map, occupancy_grid=occupancy)

        unknown = query.cell_at(85.0, 85.0)
        obstacle = query.cell_at(25.0, 25.0)

        self.assertEqual(unknown.status, BeliefCellStatus.UNKNOWN.value)
        self.assertEqual(obstacle.status, BeliefCellStatus.KNOWN_OBSTACLE.value)
        self.assertFalse(query.is_known_safe(85.0, 85.0))

    def test_known_safe_requires_observation_and_low_obstacle_probability(self) -> None:
        query = WorldBeliefQuery(self.world_map)
        cell = query.cell_at(25.0, 25.0)

        self.assertEqual(cell.status, BeliefCellStatus.KNOWN_SAFE.value)
        self.assertTrue(query.is_known_safe(25.0, 25.0))
        self.assertGreater(cell.terrain_confidence, 0.0)
        self.assertIsNotNone(cell.height_estimate_m)

    def test_geofence_excludes_cells(self) -> None:
        query = WorldBeliefQuery(self.world_map, geofence=lambda x, y: x < 50.0 and y < 50.0)
        cell = query.cell_at(85.0, 85.0)

        self.assertEqual(cell.status, BeliefCellStatus.OUTSIDE_GEOFENCE.value)
        self.assertFalse(query.is_inside_geofence(85.0, 85.0))

    def test_score_frontiers_returns_deterministic_candidates(self) -> None:
        query = WorldBeliefQuery(self.world_map)
        first = query.score_frontiers((25.0, 25.0), max_candidates=3)
        second = query.score_frontiers((25.0, 25.0), max_candidates=3)

        self.assertGreater(len(first), 0)
        self.assertEqual([c.cell.ij for c in first], [c.cell.ij for c in second])


if __name__ == "__main__":
    unittest.main()
