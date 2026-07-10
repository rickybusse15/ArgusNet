"""Tests for CoverageMap marking, including vectorized/legacy equivalence."""

from __future__ import annotations

import unittest

import numpy as np

from argusnet.mapping.coverage import CoverageMap, circular_footprint, rectangular_footprint
from argusnet.mapping.occupancy import GridBounds


def _bounds() -> GridBounds:
    return GridBounds(
        x_min_m=-500.0, x_max_m=500.0, y_min_m=-400.0, y_max_m=400.0, resolution_m=50.0
    )


class TestCoverageMarkEquivalence(unittest.TestCase):
    """mark_circular/mark_rectangular must match the legacy footprint+mark path."""

    def test_mark_circular_matches_legacy_footprint_path(self):
        bounds = _bounds()
        centers = [(0.0, 0.0), (-480.0, 390.0), (510.0, -410.0), (123.4, -56.7)]
        radii = [75.0, 120.0, 30.0, 260.0]

        fast = CoverageMap(bounds)
        legacy = CoverageMap(bounds)
        for center, radius in zip(centers, radii, strict=True):
            fast.mark_circular(center_xy=center, radius_m=radius)
            legacy.mark(circular_footprint(center, radius, bounds.resolution_m))

        np.testing.assert_array_equal(fast.count_grid, legacy.count_grid)

    def test_mark_rectangular_matches_legacy_footprint_path(self):
        bounds = _bounds()
        fast = CoverageMap(bounds)
        legacy = CoverageMap(bounds)
        for center, w, h, yaw in [
            ((0.0, 0.0), 200.0, 100.0, 0.0),
            ((100.0, -50.0), 150.0, 300.0, 0.7),
            ((-490.0, 380.0), 90.0, 90.0, -1.2),
        ]:
            fast.mark_rectangular(center_xy=center, width_m=w, height_m=h, yaw_rad=yaw)
            legacy.mark(rectangular_footprint(center, w, h, bounds.resolution_m, yaw))

        np.testing.assert_array_equal(fast.count_grid, legacy.count_grid)

    def test_footprint_entirely_outside_bounds_is_noop(self):
        bounds = _bounds()
        cov = CoverageMap(bounds)
        cov.mark_circular(center_xy=(10_000.0, 10_000.0), radius_m=50.0)
        self.assertEqual(cov.stats.covered_cells, 0)

    def test_overlapping_marks_accumulate_counts(self):
        bounds = _bounds()
        cov = CoverageMap(bounds)
        cov.mark_circular(center_xy=(0.0, 0.0), radius_m=100.0)
        cov.mark_circular(center_xy=(0.0, 0.0), radius_m=100.0)
        self.assertEqual(cov.count_at(0.0, 0.0), 2)


if __name__ == "__main__":
    unittest.main()
