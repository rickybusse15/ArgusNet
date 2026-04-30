from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pytest_benchmark")

from argusnet.world.environment import Bounds2D, EnvironmentModel
from argusnet.world.terrain import TerrainModel


def _require_benchmark_only(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--benchmark-only", default=False):
        pytest.skip("benchmark tests run only with --benchmark-only")


@pytest.mark.benchmark_fast
def test_height_at_grid(benchmark, request: pytest.FixtureRequest) -> None:
    _require_benchmark_only(request)
    terrain = TerrainModel.default()
    points = np.linspace(-512.0, 512.0, 1024)

    def run() -> float:
        total = 0.0
        for index, x_m in enumerate(points):
            total += terrain.height_at(float(x_m), float(points[index % len(points)]))
        return total

    benchmark(run)


@pytest.mark.benchmark_fast
def test_gradient_at_grid(benchmark, request: pytest.FixtureRequest) -> None:
    _require_benchmark_only(request)
    terrain = TerrainModel.default()
    points = np.linspace(-512.0, 512.0, 1024)

    def run() -> float:
        total = 0.0
        for index, x_m in enumerate(points):
            gradient = terrain.gradient_at(float(x_m), float(points[index % len(points)]))
            total += float(gradient[0] + gradient[1])
        return total

    benchmark(run)


@pytest.mark.benchmark_fast
def test_los_grid(benchmark, request: pytest.FixtureRequest) -> None:
    _require_benchmark_only(request)
    terrain = TerrainModel.default()
    env = EnvironmentModel.from_legacy(
        environment_id="bench-los",
        bounds_xy_m=Bounds2D(-1024.0, 1024.0, -1024.0, 1024.0),
        terrain_model=terrain,
        terrain_resolution_m=8.0,
    )

    def run() -> int:
        visible = 0
        for i in range(512):
            origin = np.array([-800.0 + i, -700.0, 180.0], dtype=float)
            target = np.array([700.0, 800.0 - i, 40.0], dtype=float)
            visible += int(env.query.los(origin, target).visible)
        return visible

    benchmark(run)
