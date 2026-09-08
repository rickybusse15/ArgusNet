"""Compare legacy per-particle coverage copies with the native scoring batch."""

import json
import statistics
import time

import numpy as np

from argusnet._native_world import score_footprints
from argusnet.mapping.coverage import CoverageMap
from argusnet.mapping.occupancy import GridBounds


def benchmark():
    bounds = GridBounds(-500, 500, -500, 500, 5)
    coverage = CoverageMap(bounds)
    coverage.mark_circular((0, 0), 400)
    positions = np.random.default_rng(42).normal(0, 80, (200, 2))
    radius = 30.0

    def legacy():
        scores = []
        for xy in positions:
            i, j = bounds.xy_to_ij(*xy)
            r = max(1, int(radius / bounds.resolution_m))
            i0, i1 = max(0, i - r), min(bounds.nx - 1, i + r)
            j0, j1 = max(0, j - r), min(bounds.ny - 1, j + r)
            scores.append(
                -1.0
                if i0 >= i1 or j0 >= j1
                else float(np.log1p(coverage.count_grid[i0:i1, j0:j1].mean()))
            )
        return np.array(scores)

    def native():
        return score_footprints(coverage.count_grid, bounds, radius, positions)

    np.testing.assert_allclose(native(), legacy(), atol=1e-12)
    result = {
        "grid_shape": [bounds.nx, bounds.ny],
        "particles": len(positions),
        "radius_m": radius,
        "repetitions": 50,
    }
    for name, fn in [("legacy", legacy), ("native", native)]:
        samples = []
        for _ in range(50):
            start = time.perf_counter()
            fn()
            samples.append((time.perf_counter() - start) * 1000)
        result[name + "_median_ms"] = statistics.median(samples)
    result["speedup"] = result["legacy_median_ms"] / result["native_median_ms"]
    return result


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
