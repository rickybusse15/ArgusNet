"""Contract tests for the native world boundary, independent of the tracking daemon."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from argusnet._native_world import call
from argusnet.world.environment import Bounds2D, TerrainLayer
from argusnet.world.terrain import terrain_model_from_preset
from argusnet.world.visibility import _grid_dda_intervals


def test_all_analytic_presets_match_pre_migration_fixture():
    fixture = json.loads((Path(__file__).parent / "fixtures/world_authority.json").read_text())
    for case in fixture["terrains"]:
        terrain = terrain_model_from_preset(case["preset"], 1.0)
        points = np.asarray(case["points"])
        np.testing.assert_allclose(
            terrain.analytic_height_at_many(points), case["analytic_heights"], atol=1e-9, rtol=0
        )
        np.testing.assert_allclose(
            terrain.height_at_many(points), case["heights"], atol=1e-9, rtol=0
        )


def test_native_errors_do_not_poison_subsequent_queries():
    with pytest.raises(ValueError, match="expired"):
        call("heights", handle=0, points=[[0.0, 0.0]])
    with pytest.raises(ValueError, match="height grid"):
        call("pyramid", grid=[[0.0], [1.0]], maximum=True)
    assert call("version") == 1


def test_batch_queries_preserve_shape_and_empty_arrays():
    terrain = terrain_model_from_preset("alpine", 1.0)
    points = np.zeros((2, 3, 2))
    assert terrain.height_at_many(points).shape == (2, 3)
    assert terrain.height_at_many(np.empty((0, 2))).shape == (0,)
    assert isinstance(terrain.height_at_many([0.0, 0.0]), float)
    with pytest.raises(ValueError):
        terrain.height_at_many([float("nan"), 0.0])


def test_lod_tiles_are_materialized_and_base_queries_match():
    grid = np.add.outer(np.arange(5.0) * 2, np.arange(5.0))
    terrain = TerrainLayer.from_height_grid(
        environment_id="lod",
        bounds_xy_m=Bounds2D(0, 4, 0, 4),
        heights_m=grid,
        resolution_m=1,
        tile_size_cells=2,
        lod_resolutions_m=(1, 2),
    )
    assert terrain.tile_for_xy(1, 1, lod=1) is not None
    assert terrain.height_at(2.5, 1.5) == 5.5
    assert terrain.height_at(4, 4) == 12


def test_dda_negative_directions_visit_same_cells():
    def cells(a, b):
        return [
            (x, y)
            for x, y, start, end in _grid_dda_intervals(
                origin_xy=np.asarray(a),
                target_xy=np.asarray(b),
                grid_bounds=Bounds2D(0, 40, 0, 40),
                cell_size_m=10,
            )
            if end - start > 1e-10
        ]

    assert cells([1, 5], [39, 5]) == list(reversed(cells([39, 5], [1, 5])))
    assert len(cells([39, 5], [1, 5])) == 4


def test_copy_owns_independent_native_handle():
    import copy
    import gc

    terrain = terrain_model_from_preset("alpine", 1.0)
    expected = terrain.height_at(5.0, 6.0)
    copied = copy.deepcopy(terrain)
    assert copied._native.id != terrain._native.id
    del terrain
    gc.collect()
    assert copied.height_at(5.0, 6.0) == expected
    assert copied.height_at(7.0, 8.0) > 0


def test_blocking_simulation_reuses_native_monitors_and_stays_above_terrain():
    from unittest.mock import patch

    from argusnet.safety.native import SafetyMonitor
    from argusnet.simulation.sim import (
        ScenarioOptions,
        SimulationConfig,
        build_default_scenario,
        run_simulation,
    )

    scenario = build_default_scenario(
        options=ScenarioOptions(map_preset="small", drone_count=2, safety_blocking=True), seed=7
    )
    with patch("argusnet.simulation.sim.SafetyMonitor", wraps=SafetyMonitor) as constructor:
        result = run_simulation(scenario, SimulationConfig(steps=4, dt_s=0.25, seed=7))
    mobile_ids = {node.node_id for node in result.frames[0].nodes if node.is_mobile}
    assert constructor.call_count == len(mobile_ids)
    for frame in result.frames:
        for node in frame.nodes:
            if node.is_mobile:
                assert node.position[2] >= scenario.terrain.height_at(*node.position[:2])
