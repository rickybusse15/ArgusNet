"""Tests for the terrain/obstacle-aware 3D altitude profiler."""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from argusnet.planning.planner_3d import (
    VERTICAL_ROUTE_CONTRACT_VERSION,
    AltitudeProfiler,
    Route3D,
    Route3DConfig,
    TerrainHeightField,
)
from argusnet.planning.planner_base import PathPlanner2D, PlannerRoute
from argusnet.world.environment import Bounds2D, ObstacleLayer
from argusnet.world.obstacles import BuildingPrism, ForestStand


class _FlatTerrain:
    def __init__(self, height_m: float) -> None:
        self.height_m = float(height_m)

    def height_at_many(self, xy_m: np.ndarray) -> np.ndarray:
        points = np.asarray(xy_m, dtype=float).reshape(-1, 2)
        return np.full(points.shape[0], self.height_m, dtype=float)


class _FunctionTerrain:
    def __init__(self, fn: Callable[[float, float], float]) -> None:
        self.fn = fn

    def height_at_many(self, xy_m: np.ndarray) -> np.ndarray:
        points = np.asarray(xy_m, dtype=float).reshape(-1, 2)
        return np.array([self.fn(float(x), float(y)) for x, y in points], dtype=float)


def _rect(x0: float, x1: float, y0: float, y1: float) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=float)


def _layer(primitives: list, *, size_m: float = 1000.0) -> ObstacleLayer:
    bounds = Bounds2D(x_min_m=-size_m, x_max_m=size_m, y_min_m=-size_m, y_max_m=size_m)
    return ObstacleLayer(bounds_xy_m=bounds, tile_size_m=64.0, primitives=primitives)


def _straight(x_end_m: float, n: int = 2) -> np.ndarray:
    xs = np.linspace(0.0, x_end_m, n)
    return np.column_stack([xs, np.zeros_like(xs)])


# --- terrain clearance ------------------------------------------------------


def test_flat_terrain_no_obstacles_cruises_at_agl() -> None:
    config = Route3DConfig(cruise_agl_m=60.0, min_terrain_clearance_m=25.0)
    profiler = AltitudeProfiler(terrain=_FlatTerrain(100.0), config=config)
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert np.allclose(route.altitudes_m, 160.0)
    assert route.ascent_m == pytest.approx(0.0, abs=1e-6)
    assert route.descent_m == pytest.approx(0.0, abs=1e-6)
    assert route.horizontal_length_m == pytest.approx(200.0, abs=1e-6)
    assert route.length_m == pytest.approx(200.0, abs=1e-6)
    assert route.max_gradient == pytest.approx(0.0, abs=1e-9)
    assert route.contract_version == VERTICAL_ROUTE_CONTRACT_VERSION


def test_terrain_clearance_maintained_over_bumpy_ground() -> None:
    terrain = _FunctionTerrain(lambda x, y: 50.0 + 30.0 * math.sin(x / 40.0))
    config = Route3DConfig(
        cruise_agl_m=40.0,
        min_terrain_clearance_m=25.0,
        sample_spacing_m=2.0,
        decimate_tolerance_m=0.0,  # keep every sample for an exact clearance check
        max_climb_gradient=1.0,
    )
    profiler = AltitudeProfiler(terrain=terrain, config=config)
    route = profiler.profile_route(_straight(400.0))
    assert route is not None
    ground = terrain.height_at_many(route.points_xy_m)
    clearance = route.altitudes_m - ground
    assert np.all(clearance >= 25.0 - 1e-6)
    assert route.max_gradient <= 1.0 + 1e-6


# --- obstacle clearance -----------------------------------------------------


def test_climbs_over_vegetation_canopy() -> None:
    forest = ForestStand("forest", _rect(80.0, 120.0, -50.0, 50.0), 0.0, 40.0, 0.5)
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_climb_gradient=1.0,
        sample_spacing_m=2.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([forest]), config=config
    )
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    needed_m = 40.0 + 12.0
    assert route.max_altitude_m >= needed_m - 1e-6
    for x_m, y_m, z_m in route.points_xyz_m:
        if forest.point_inside(float(x_m), float(y_m), 20.0):
            assert z_m >= needed_m - 1e-6
    assert route.ascent_m > 0.0
    assert route.descent_m > 0.0
    # Far from the canopy the drone returns toward cruise altitude.
    assert route.altitudes_m[0] == pytest.approx(20.0, abs=1e-6)


def test_no_climb_when_cruise_already_clears_obstacle() -> None:
    forest = ForestStand("forest", _rect(80.0, 120.0, -50.0, 50.0), 0.0, 20.0, 0.5)
    # cruise 60 -> alt 60 already exceeds canopy top 20 + clearance 12 = 32.
    config = Route3DConfig(
        cruise_agl_m=60.0, min_terrain_clearance_m=25.0, obstacle_clearance_m=12.0
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([forest]), config=config
    )
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert np.allclose(route.altitudes_m, 60.0)


def test_obstacle_beside_path_is_ignored() -> None:
    # Footprint sits off the y=0 flight line; the profiler must not climb for it.
    forest = ForestStand("forest", _rect(80.0, 120.0, 100.0, 200.0), 0.0, 40.0, 0.5)
    config = Route3DConfig(
        cruise_agl_m=20.0, min_terrain_clearance_m=10.0, obstacle_clearance_m=12.0
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([forest]), config=config
    )
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert np.allclose(route.altitudes_m, 20.0)


def test_narrow_obstacle_between_samples_is_cleared() -> None:
    # A strip 1 m wide, sitting at x=55 between the 10 m-spaced samples at
    # x=50/60. Point-only sampling would miss it; segment testing must not.
    strip = BuildingPrism("strip", _rect(54.5, 55.5, -30.0, 30.0), 0.0, 40.0)
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_climb_gradient=5.0,
        sample_spacing_m=10.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([strip]), config=config
    )
    route = profiler.profile_route(_straight(100.0))
    assert route is not None
    needed_m = 40.0 + 12.0
    assert route.max_altitude_m >= needed_m - 1e-6
    # Densely re-sample the flown polyline; altitude over the strip must clear it.
    crossed = False
    for x_m, y_m, z_m in _resample_xyz(route.points_xyz_m, spacing_m=0.25):
        if strip.point_inside(float(x_m), float(y_m), 20.0):
            crossed = True
            assert z_m >= needed_m - 1e-6
    assert crossed  # the resampling really does pass over the strip


# --- gradient feasibility ---------------------------------------------------


def test_gradient_never_exceeds_limit_and_start_stays_at_cruise() -> None:
    building = BuildingPrism("b", _rect(295.0, 305.0, -50.0, 50.0), 0.0, 80.0)
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_climb_gradient=0.5,
        sample_spacing_m=2.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([building]), config=config
    )
    route = profiler.profile_route(_straight(600.0))
    assert route is not None
    assert route.max_gradient <= 0.5 + 1e-6
    assert route.max_altitude_m >= 80.0 + 12.0 - 1e-6
    # 92 m climb from 20 m needs 144 m at gradient 0.5; the 300 m lead-in fits.
    assert route.altitudes_m[0] == pytest.approx(20.0, abs=1e-6)


def test_gradient_limited_even_when_peak_forces_early_climb() -> None:
    building = BuildingPrism("b", _rect(98.0, 102.0, -50.0, 50.0), 0.0, 200.0)
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_climb_gradient=0.5,
        sample_spacing_m=1.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([building]), config=config
    )
    route = profiler.profile_route(_straight(100.0))
    assert route is not None
    # A 212 m peak reachable only by starting well above cruise, yet still
    # gradient-limited throughout.
    assert route.max_gradient <= 0.5 + 1e-6
    assert route.altitudes_m[0] > 21.0


# --- ceiling handling -------------------------------------------------------


def test_ceiling_conflict_flagged_and_safety_wins() -> None:
    building = BuildingPrism("b", _rect(90.0, 110.0, -50.0, 50.0), 0.0, 300.0)
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_altitude_m=200.0,
        max_climb_gradient=2.0,
        sample_spacing_m=2.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([building]), config=config
    )
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert route.ceiling_conflicts > 0
    assert route.max_altitude_m >= 300.0 + 12.0 - 1e-6  # clearance overrides the ceiling


def test_ceiling_respected_when_feasible() -> None:
    building = BuildingPrism("b", _rect(90.0, 110.0, -50.0, 50.0), 0.0, 80.0)
    config = Route3DConfig(
        cruise_agl_m=60.0,
        min_terrain_clearance_m=25.0,
        obstacle_clearance_m=12.0,
        max_altitude_m=150.0,
        max_climb_gradient=1.0,
        sample_spacing_m=2.0,
        decimate_tolerance_m=0.0,
    )
    profiler = AltitudeProfiler(
        terrain=_FlatTerrain(0.0), obstacle_layer=_layer([building]), config=config
    )
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert route.ceiling_conflicts == 0
    assert route.max_altitude_m <= 150.0 + 1e-6


def test_min_altitude_floor_applied() -> None:
    config = Route3DConfig(cruise_agl_m=25.0, min_terrain_clearance_m=25.0, min_altitude_m=200.0)
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0), config=config)
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    assert np.all(route.altitudes_m >= 200.0 - 1e-6)


# --- composition with the 2D planner ---------------------------------------


def test_plan_route_3d_routes_around_building_and_clears_terrain() -> None:
    building = BuildingPrism("b", _rect(-20.0, 20.0, -20.0, 20.0), 0.0, 50.0)
    layer = _layer([building])
    planner = PathPlanner2D(
        bounds_xy_m=Bounds2D(x_min_m=-500.0, x_max_m=500.0, y_min_m=-500.0, y_max_m=500.0),
        obstacle_layer=layer,
    )
    config = Route3DConfig(cruise_agl_m=60.0, min_terrain_clearance_m=25.0, sample_spacing_m=5.0)
    profiler = AltitudeProfiler(terrain=_FlatTerrain(100.0), obstacle_layer=layer, config=config)
    route = profiler.plan_route_3d((-200.0, 0.0), (200.0, 0.0), planner_2d=planner, clearance_m=8.0)
    assert route is not None
    for x_m, y_m, z_m in route.points_xyz_m:
        assert not building.point_inside(float(x_m), float(y_m), 25.0)
        assert z_m - 100.0 >= 25.0 - 1e-6


def test_plan_route_3d_returns_none_when_2d_unplannable() -> None:
    planner = PathPlanner2D(
        bounds_xy_m=Bounds2D(x_min_m=0.0, x_max_m=100.0, y_min_m=0.0, y_max_m=100.0),
        obstacle_layer=_layer([], size_m=1000.0),
    )
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0))
    # Goal is outside the planner bounds -> no 2D route -> no 3D route.
    route = profiler.plan_route_3d(
        (10.0, 10.0), (900.0, 900.0), planner_2d=planner, clearance_m=8.0
    )
    assert route is None


# --- decimation ------------------------------------------------------------


def test_decimation_collapses_straight_profile() -> None:
    config = Route3DConfig(cruise_agl_m=60.0, min_terrain_clearance_m=25.0, sample_spacing_m=5.0)
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0), config=config)
    route = profiler.profile_route(_straight(500.0))
    assert route is not None
    assert len(route.points_xyz_m) == 2  # a flat straight run needs only endpoints


def test_decimation_preserves_horizontal_bends() -> None:
    # An L-shaped route: the corner waypoint must survive decimation.
    points_xy = np.array([[0.0, 0.0], [200.0, 0.0], [200.0, 200.0]])
    config = Route3DConfig(cruise_agl_m=60.0, min_terrain_clearance_m=25.0, sample_spacing_m=5.0)
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0), config=config)
    route = profiler.profile_route(points_xy)
    assert route is not None
    corner_present = np.any(np.all(np.isclose(route.points_xy_m, [200.0, 0.0]), axis=1))
    assert corner_present
    assert route.horizontal_length_m == pytest.approx(400.0, abs=1e-6)


def test_decimation_never_violates_obstacle_clearance() -> None:
    forest = ForestStand("forest", _rect(80.0, 120.0, -50.0, 50.0), 0.0, 40.0, 0.5)
    layer = _layer([forest])
    config = Route3DConfig(
        cruise_agl_m=20.0,
        min_terrain_clearance_m=10.0,
        obstacle_clearance_m=12.0,
        max_climb_gradient=1.0,
        sample_spacing_m=2.0,
        decimate_tolerance_m=1.0,  # aggressive decimation
    )
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0), obstacle_layer=layer, config=config)
    route = profiler.profile_route(_straight(200.0))
    assert route is not None
    # Re-sample the decimated polyline densely and confirm clearance holds.
    dense = _resample_xyz(route.points_xyz_m, spacing_m=1.0)
    for x_m, y_m, z_m in dense:
        if forest.point_inside(float(x_m), float(y_m), 20.0):
            assert z_m >= 40.0 + 12.0 - 1e-6


def _resample_xyz(points_xyz: np.ndarray, *, spacing_m: float) -> np.ndarray:
    out: list[np.ndarray] = [points_xyz[0]]
    for start, end in zip(points_xyz[:-1], points_xyz[1:], strict=True):
        length = float(np.linalg.norm(end[:2] - start[:2]))
        steps = max(int(np.ceil(length / spacing_m)), 1)
        for step in range(1, steps + 1):
            out.append(start + (end - start) * (step / steps))
    return np.vstack(out)


# --- determinism and validation --------------------------------------------


def test_determinism() -> None:
    terrain = _FunctionTerrain(lambda x, y: 10.0 + 0.05 * x)
    forest = ForestStand("forest", _rect(120.0, 160.0, -40.0, 40.0), 0.0, 30.0, 0.4)
    config = Route3DConfig(sample_spacing_m=3.0)
    profiler = AltitudeProfiler(terrain=terrain, obstacle_layer=_layer([forest]), config=config)
    first = profiler.profile_route(_straight(300.0))
    second = profiler.profile_route(_straight(300.0))
    assert first is not None and second is not None
    assert np.array_equal(first.points_xyz_m, second.points_xyz_m)


def test_degenerate_route_returns_none() -> None:
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0))
    assert profiler.profile_route(np.array([[10.0, 10.0], [10.0, 10.0]])) is None


def test_accepts_planner_route_input() -> None:
    planner_route = PlannerRoute(
        points_xy_m=np.array([[0.0, 0.0], [150.0, 0.0]]), length_m=150.0, vertex_count=2
    )
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0))
    route = profiler.profile_route(planner_route)
    assert route is not None
    assert route.horizontal_length_m == pytest.approx(150.0, abs=1e-6)


def test_invalid_route_shape_raises() -> None:
    profiler = AltitudeProfiler(terrain=_FlatTerrain(0.0))
    with pytest.raises(ValueError):
        profiler.profile_route(np.array([[0.0, 0.0]]))  # single point
    with pytest.raises(ValueError):
        profiler.profile_route(np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))  # 3 columns


def test_route3d_rejects_bad_shape() -> None:
    with pytest.raises(ValueError):
        Route3D(
            points_xyz_m=np.array([[0.0, 0.0]]),
            length_m=0.0,
            horizontal_length_m=0.0,
            ascent_m=0.0,
            descent_m=0.0,
            max_altitude_m=0.0,
            min_altitude_m=0.0,
            max_gradient=0.0,
            ceiling_conflicts=0,
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"cruise_agl_m": 10.0, "min_terrain_clearance_m": 25.0},  # cruise below floor
        {"max_climb_gradient": 0.0},
        {"sample_spacing_m": 0.0},
        {"min_terrain_clearance_m": -1.0},
        {"max_samples": 1},
        {"min_altitude_m": 300.0, "max_altitude_m": 100.0},
    ],
)
def test_config_validation(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        Route3DConfig(**kwargs)


def test_terrain_protocol_runtime_checkable() -> None:
    assert isinstance(_FlatTerrain(0.0), TerrainHeightField)
