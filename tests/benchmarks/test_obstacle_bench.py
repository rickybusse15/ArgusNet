from __future__ import annotations

import pytest

pytest.importorskip("pytest_benchmark")

from argusnet.world.environment import Bounds2D, ObstacleLayer
from argusnet.world.obstacles import (
    BuildingPrism,
    CylinderObstacle,
    ForestStand,
    OrientedBox,
    WallSegment,
)


def _require_benchmark_only(request: pytest.FixtureRequest) -> None:
    if not request.config.getoption("--benchmark-only", default=False):
        pytest.skip("benchmark tests run only with --benchmark-only")


@pytest.mark.benchmark_fast
def test_point_inside_obstacles(benchmark, request: pytest.FixtureRequest) -> None:
    _require_benchmark_only(request)
    layer = ObstacleLayer(
        bounds_xy_m=Bounds2D(-500.0, 500.0, -500.0, 500.0),
        tile_size_m=100.0,
        primitives=(
            BuildingPrism("building", [(0, 0), (60, 0), (60, 60), (0, 60)], 0.0, 80.0),
            ForestStand("forest", [(100, 0), (180, 0), (180, 80), (100, 80)], 0.0, 45.0),
            OrientedBox("box", "building", -120.0, 40.0, 75.0, 45.0, 0.4, 0.0, 65.0),
            WallSegment("wall", "wall", (-50.0, -100.0), (160.0, -95.0), 3.0, 0.0, 40.0),
            CylinderObstacle("cylinder", "building", -160.0, -140.0, 35.0, 0.0, 60.0),
        ),
    )

    def run() -> int:
        collisions = 0
        for i in range(4096):
            x_m = -220.0 + (i % 128) * 3.5
            y_m = -180.0 + (i // 128) * 10.0
            collisions += int(layer.point_collides(x_m, y_m, 25.0) is not None)
        return collisions

    benchmark(run)
