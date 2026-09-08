"""Parity at the migrated localization boundary, including seeded update state."""

from unittest.mock import patch

import numpy as np
import pytest

from argusnet._native_world import score_footprints
from argusnet.localization.engine import GridLocalizer, LocalizationConfig
from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds


def reference_scores(grid, bounds, radius, positions):
    scores = []
    for xy in positions:
        i, j = bounds.xy_to_ij(*xy)
        r = max(1, int(radius / bounds.resolution_m))
        i0, i1 = max(0, i - r), min(bounds.nx - 1, i + r)
        j0, j1 = max(0, j - r), min(bounds.ny - 1, j + r)
        scores.append(-1.0 if i0 >= i1 or j0 >= j1 else np.log1p(grid[i0:i1, j0:j1].mean()))
    return np.array(scores)


@pytest.mark.parametrize("shape", [(1, 7), (8, 1), (3, 11), (31, 23)])
@pytest.mark.parametrize("radius", [0.0, 0.3, 5.0, 10000.0])
def test_scores_match_legacy_edges_and_rectangular_grids(shape, radius):
    nx, ny = shape
    bounds = GridBounds(-10, -10 + nx * 0.5, 7, 7 + ny * 0.5, 0.5)
    rng = np.random.default_rng(184)
    grid = rng.integers(0, 100, size=shape, dtype=np.int32)
    points = rng.uniform(-30, 40, size=(200, 2))
    actual = score_footprints(grid, bounds, radius, points)
    np.testing.assert_allclose(actual, reference_scores(grid, bounds, radius, points), atol=1e-12)


def test_seeded_updates_confidence_timeout_and_reset_match_reference():
    coverage = CoverageMap(GridBounds(-100, 100, -100, 100, 5))
    coverage.mark_circular((0, 0), 95)
    config = LocalizationConfig(localization_timeout_steps=4)
    native, reference = GridLocalizer(config), GridLocalizer(config)
    for step in range(6):
        for drone in ["a", "b"]:
            args = (drone, np.array([step * 2, 3, 40]), 0.2, coverage, 30, step * 0.25)
            actual = native.update(*args)
            with patch("argusnet.localization.engine.score_footprints", reference_scores):
                expected = reference.update(*args)
            np.testing.assert_allclose(
                actual.position_estimate, expected.position_estimate, atol=1e-12
            )
            assert actual.position_std_m == pytest.approx(expected.position_std_m, abs=1e-12)
            assert actual.confidence == pytest.approx(expected.confidence, abs=1e-12)
    assert native.any_timed_out and reference.any_timed_out
    native.reset("a")
    assert [e.drone_id for e in native.all_estimates()] == ["b"]
    native.reset()
    assert not native.any_timed_out


def test_localizer_reads_coverage_once_per_particle_batch():
    class CountingCoverage(CoverageMap):
        reads = 0

        @property
        def count_grid(self):
            self.reads += 1
            return super().count_grid

    coverage = CountingCoverage(GridBounds(0, 100, 0, 100, 5))
    localizer = GridLocalizer()
    localizer.update("a", np.array([50, 50, 20]), 0, coverage, 10, 0)
    assert coverage.reads == 0  # Below minimum coverage: no scoring or RNG advance.
    coverage.mark_circular((50, 50), 100)
    localizer.update("a", np.array([50, 50, 20]), 0, coverage, 10, 1)
    assert coverage.reads == 1


def test_invalid_native_query_is_rejected():
    bounds = GridBounds(0, 2, 0, 2, 1)
    with pytest.raises(ValueError):
        score_footprints(np.zeros((2, 2)), bounds, 1, [[float("nan"), 0]])
    with pytest.raises(ValueError):
        score_footprints(np.zeros((2, 2)), bounds, -1, [[0, 0]])
