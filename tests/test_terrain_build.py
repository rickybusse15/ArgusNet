from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile

from argusnet.world.environment import Bounds2D, LandCoverClass
from argusnet.world.procedural import (
    LandscapeBuildConfig,
    TerrainBuildConfig,
    build_land_cover_layer,
    build_landscape,
    build_terrain_layer,
    procedural_terrain_grid,
)


def test_procedural_terrain_is_deterministic_by_seed() -> None:
    bounds = Bounds2D(-100.0, 100.0, -80.0, 80.0)
    config = TerrainBuildConfig(
        terrain_preset="mountain_pass",
        terrain_seed=42,
        terrain_resolution_m=10.0,
    )

    first = build_terrain_layer(config, bounds, environment_id="first")
    second = build_terrain_layer(config, bounds, environment_id="second")
    points = np.array([[-60.0, -20.0], [0.0, 0.0], [50.0, 35.0]], dtype=float)

    assert np.allclose(first.height_at_many(points), second.height_at_many(points))
    assert first.source_metadata["source"] == "procedural"
    assert first.source_metadata["seed"] == 42


def test_height_at_many_matches_scalar_height_at() -> None:
    bounds = Bounds2D(0.0, 20.0, 0.0, 20.0)
    terrain = build_terrain_layer(
        TerrainBuildConfig(terrain_preset="rolling_highlands", terrain_resolution_m=5.0),
        bounds,
        environment_id="batch",
    )
    points = np.array([[0.0, 0.0], [2.5, 7.5], [20.0, 20.0], [25.0, 25.0]], dtype=float)

    batch = terrain.height_at_many(points)
    scalar = np.array([terrain.height_at(float(x), float(y)) for x, y in points], dtype=float)

    assert np.allclose(batch, scalar)


def test_dem_and_hybrid_terrain_construction() -> None:
    heights = np.array(
        [
            [100.0, 102.0, 104.0],
            [101.0, 103.0, 105.0],
            [102.0, 104.0, 106.0],
        ],
        dtype=np.float32,
    )
    with TemporaryDirectory() as tmp:
        dem_path = Path(tmp) / "terrain.tif"
        tifffile.imwrite(
            dem_path,
            heights,
            extratags=[
                (33550, "d", 3, (10.0, 10.0, 0.0), False),
                (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 20.0, 0.0), False),
            ],
        )
        bounds = Bounds2D(0.0, 20.0, 0.0, 20.0)
        dem = build_terrain_layer(
            TerrainBuildConfig(
                terrain_source="dem",
                dem_path=str(dem_path),
                dem_crs="EPSG:32611",
                terrain_resolution_m=10.0,
            ),
            bounds,
            environment_id="dem",
        )
        hybrid = build_terrain_layer(
            TerrainBuildConfig(
                terrain_source="hybrid",
                terrain_preset="alpine",
                terrain_seed=9,
                dem_path=str(dem_path),
                dem_crs="EPSG:32611",
                terrain_resolution_m=10.0,
                detail_strength=0.25,
            ),
            bounds,
            environment_id="hybrid",
        )

    assert dem.source_metadata["source"] == "dem"
    assert hybrid.source_metadata["source"] == "hybrid"
    assert (
        np.max(
            np.abs(
                hybrid.height_at_many([[10.0, 10.0]])
                - dem.height_at_many([[10.0, 10.0]])
            )
        )
        > 0.0
    )


def test_seasonal_land_cover_adds_snow() -> None:
    bounds = Bounds2D(-150.0, 150.0, -150.0, 150.0)
    terrain = build_terrain_layer(
        TerrainBuildConfig(
            terrain_preset="alpine",
            terrain_seed=11,
            terrain_resolution_m=15.0,
        ),
        bounds,
        environment_id="snow",
    )

    land_cover = build_land_cover_layer(
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=(),
        resolution_m=20.0,
        season=TerrainBuildConfig(season_month=1).season,
        terrain_preset="alpine",
        seed=11,
    )
    sampled_classes = [
        land_cover.land_cover_at(float(x), float(y))
        for x in np.linspace(bounds.x_min_m, bounds.x_max_m, 8)
        for y in np.linspace(bounds.y_min_m, bounds.y_max_m, 8)
    ]

    assert LandCoverClass.SNOW in sampled_classes


def test_procedural_terrain_grid_exposes_semantic_masks() -> None:
    bounds = Bounds2D(-250.0, 250.0, -200.0, 200.0)
    grid = procedural_terrain_grid(
        bounds_xy_m=bounds,
        preset_name="river_valley",
        seed=21,
        resolution_m=10.0,
    )

    assert grid.metadata["generator_version"] == "procedural-landscape-v1"
    for name in ("water", "ridge", "valley", "steep_slope", "lowland", "road"):
        assert name in grid.masks
        assert grid.masks[name].shape == grid.heights_m.shape
    assert np.any(grid.masks["water"])
    assert np.any(grid.masks["valley"])


def test_terrain_layer_metadata_preserves_mask_summary() -> None:
    bounds = Bounds2D(-150.0, 150.0, -150.0, 150.0)
    terrain = build_terrain_layer(
        TerrainBuildConfig(
            terrain_preset="lake_district",
            terrain_seed=17,
            terrain_resolution_m=10.0,
        ),
        bounds,
        environment_id="masked-terrain",
    )
    summary = terrain.terrain_summary()

    assert "water" in summary["semantic_masks"]
    assert summary["source"]["generator_version"] == "procedural-landscape-v1"
    assert summary["mask_coverage_fraction"]["water"] > 0.0


def test_mask_driven_land_cover_adds_roads_for_urban_flat() -> None:
    bounds = Bounds2D(-180.0, 180.0, -180.0, 180.0)
    terrain = build_terrain_layer(
        TerrainBuildConfig(
            terrain_preset="urban_flat",
            terrain_seed=5,
            terrain_resolution_m=10.0,
        ),
        bounds,
        environment_id="urban-masks",
    )
    land_cover = build_land_cover_layer(
        bounds_xy_m=bounds,
        terrain=terrain,
        obstacles=(),
        resolution_m=20.0,
        terrain_preset="urban_flat",
        seed=5,
    )
    sampled_classes = [
        land_cover.land_cover_at(float(x), float(y))
        for x in np.linspace(bounds.x_min_m, bounds.x_max_m, 12)
        for y in np.linspace(bounds.y_min_m, bounds.y_max_m, 12)
    ]

    assert LandCoverClass.ROAD in sampled_classes


def test_build_landscape_returns_coherent_layers() -> None:
    bounds = Bounds2D(-120.0, 120.0, -120.0, 120.0)
    result = build_landscape(
        LandscapeBuildConfig(
            terrain=TerrainBuildConfig(
                terrain_preset="coastal",
                terrain_seed=3,
                terrain_resolution_m=10.0,
            ),
            land_cover_resolution_m=20.0,
        ),
        bounds,
        environment_id="coastal-landscape",
    )

    assert result.terrain.environment_id == "coastal-landscape"
    assert result.land_cover.bounds_xy_m == bounds
    assert "water" in result.masks
    assert result.metadata["generator_version"] == "landscape-build-v1"
