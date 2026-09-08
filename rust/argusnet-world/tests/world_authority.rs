use argusnet_world::{
    analytic::AnalyticTerrain,
    index::ObstacleLayer,
    obstacles::{BlockerType, Bounds2d, Obstacle},
    raster::TerrainRaster,
    tiled::{self, TiledTerrain},
    visibility::{self, VisibilityProfile},
    TerrainQuery,
};
fn bounds() -> Bounds2d {
    Bounds2d::new(0.0, 40.0, 0.0, 40.0)
}
fn terrain(grid: Vec<Vec<f64>>) -> TiledTerrain {
    TiledTerrain::from_grid(bounds(), &grid, 10.0, 2, 0.0, &[10.0, 20.0]).unwrap()
}
fn empty() -> ObstacleLayer {
    ObstacleLayer::new(bounds(), 10.0, vec![]).unwrap()
}
#[test]
fn python_analytic_presets_match_frozen_reference() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/fixtures/world_authority.json")).unwrap();
    for case in fixture["terrains"].as_array().unwrap() {
        let model: AnalyticTerrain = serde_json::from_value(case["parameters"].clone()).unwrap();
        let points: Vec<[f64; 2]> = serde_json::from_value(case["points"].clone()).unwrap();
        for (i, p) in points.iter().enumerate() {
            let expected = case["analytic_heights"][i].as_f64().unwrap();
            assert!(
                (model.analytic_height(*p) - expected).abs() < 1e-9,
                "{} {p:?}",
                case["preset"]
            );
            let expected = case["heights"][i].as_f64().unwrap();
            assert!((model.height(*p) - expected).abs() < 1e-9);
            let g = tiled::gradient(|x, y| model.height([x, y]), p[0], p[1], 0.5);
            for (j, v) in g.iter().enumerate() {
                assert!((v - case["gradients"][i][j].as_f64().unwrap()).abs() < 1e-9);
            }
        }
    }
}
#[test]
fn tiled_bilinear_lods_and_boundaries_share_surface() {
    let grid = (0..5)
        .map(|y| (0..5).map(|x| (x * 10 + y * 20) as f64).collect())
        .collect();
    let t = terrain(grid);
    for (x, y) in [
        (0.0, 0.0),
        (40.0, 40.0),
        (20.0, 19.9),
        (20.001, 20.001),
        (3.25, 13.75),
    ] {
        assert!((t.height_at(x, y) - (x + 2.0 * y)).abs() < 1e-10);
        assert!((t.height_lod(x, y, 1) - (x + 2.0 * y)).abs() < 1e-10);
    }
    assert_eq!(t.height_at(-1.0, 1.0), 0.0);
    assert_eq!(t.tiles.len(), 8);
}
#[test]
fn pyramids_preserve_extrema_for_odd_rectangles() {
    let mut grid = vec![vec![2.0; 8]; 5];
    grid[4][7] = 100.0;
    grid[0][0] = -10.0;
    let max = tiled::pyramid(&grid, true).unwrap();
    let min = tiled::pyramid(&grid, false).unwrap();
    assert!(max.last().unwrap().iter().flatten().any(|v| *v == 100.0));
    assert!(min.last().unwrap().iter().flatten().any(|v| *v == -10.0));
}
#[test]
fn malformed_grids_are_errors() {
    for grid in [
        vec![],
        vec![vec![0.0]],
        vec![vec![0.0; 2], vec![1.0]],
        vec![vec![f64::NAN; 2]; 2],
    ] {
        assert!(TiledTerrain::from_grid(bounds(), &grid, 10.0, 2, 0.0, &[]).is_err());
    }
    assert!(TiledTerrain::from_grid(bounds(), &vec![vec![0.0; 2]; 2], 0.0, 2, 0.0, &[]).is_err());
}
#[test]
fn dda_visits_every_cell_in_both_directions() {
    for (a, b) in [
        ([1.0, 5.0], [39.0, 5.0]),
        ([5.0, 1.0], [5.0, 39.0]),
        ([1.0, 1.0], [39.0, 39.0]),
    ] {
        let cells = |a, b| {
            visibility::dda(a, b, bounds(), 10.0)
                .into_iter()
                .filter(|v| v.3 - v.2 > 1e-10)
                .map(|v| (v.0, v.1))
                .collect::<Vec<_>>()
        };
        let forward = cells(a, b);
        let mut reverse = cells(b, a);
        reverse.reverse();
        assert_eq!(forward, reverse);
    }
}
#[test]
fn negative_direction_does_not_skip_a_terrain_ridge() {
    let mut grid = vec![vec![0.0; 5]; 5];
    for row in &mut grid {
        row[2] = 100.0;
    }
    let t = terrain(grid);
    for (a, b) in [
        ([1.0, 5.0, 40.0], [39.0, 5.0, 40.0]),
        ([39.0, 5.0, 40.0], [1.0, 5.0, 40.0]),
    ] {
        let vis = visibility::los(&t, &empty(), a, b, &VisibilityProfile::default(), 0, 1.0);
        assert!(!vis.visible);
        assert_eq!(vis.blocker_type, "terrain");
    }
}
#[test]
fn missing_tile_is_explicit_coverage_loss() {
    let mut t = terrain(vec![vec![0.0; 5]; 5]);
    t.tiles.remove(&(0, 1, 0));
    let v = visibility::los(
        &t,
        &empty(),
        [1.0, 5.0, 100.0],
        [39.0, 5.0, 100.0],
        &VisibilityProfile::default(),
        0,
        1.0,
    );
    assert!(!v.visible);
    assert_eq!(v.blocker_type, "out_of_coverage");
}
#[test]
fn canopy_attenuation_and_weather_compose() {
    let forest = Obstacle::forest_stand(
        "trees",
        vec![[10.0, 0.0], [30.0, 0.0], [30.0, 40.0], [10.0, 40.0]],
        0.0,
        150.0,
        0.35,
    )
    .unwrap();
    let layer = ObstacleLayer::new(bounds(), 10.0, vec![forest]).unwrap();
    let vis = visibility::los(
        &terrain(vec![vec![0.0; 5]; 5]),
        &layer,
        [0.0, 5.0, 100.0],
        [40.0, 5.0, 100.0],
        &VisibilityProfile::default(),
        2,
        1.0,
    );
    assert!(vis.visible);
    assert!((vis.transmittance - (-0.35_f64 * 20.0 / 45.0).exp()).abs() < 1e-12);
    let clear = visibility::detection(&vis, 40.0, 2000.0, 1.0, 1.0);
    let weather = visibility::detection(&vis, 40.0, 2000.0, 0.5, 2.0);
    assert!((weather.p_d - clear.p_d * 0.5).abs() < 1e-12);
    assert!(weather.effective_noise_multiplier > clear.effective_noise_multiplier);
}
#[test]
fn index_deduplicates_and_orders_ids() {
    let box_ = |id: &str| {
        Obstacle::oriented_box(
            id,
            BlockerType::Building,
            20.0,
            20.0,
            30.0,
            30.0,
            0.0,
            0.0,
            100.0,
        )
        .unwrap()
    };
    let layer = ObstacleLayer::new(bounds(), 10.0, vec![box_("z"), box_("a")]).unwrap();
    assert_eq!(
        layer
            .query(bounds())
            .iter()
            .map(|o| o.id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "z"]
    );
    assert_eq!(layer.point_collides([20.0, 20.0, 50.0]).unwrap().id, "a");
}
#[test]
fn replay_raster_supports_submetre_and_rectangular_cells() {
    let r = TerrainRaster {
        x_min_m: 0.0,
        x_max_m: 0.5,
        y_min_m: 0.0,
        y_max_m: 2.0,
        rows: 2,
        cols: 2,
        heights_m: vec![vec![0.0, 10.0], vec![20.0, 30.0]],
    };
    assert_eq!(r.height_at(0.25, 1.0), Some(15.0));
    assert_eq!(r.height_at(0.5, 2.0), Some(30.0));
    assert_eq!(r.height_at(-0.1, 0.0), None);
    let mut bad = r;
    bad.heights_m[0].pop();
    assert_eq!(bad.height_at(0.0, 0.0), None);
}

#[test]
fn clipped_dda_parameters_stay_in_original_ray_space() {
    let cells = visibility::dda([-10.0, 5.0], [50.0, 5.0], bounds(), 10.0);
    assert_eq!(cells.len(), 4);
    assert!((cells[0].2 - 1.0 / 6.0).abs() < 1e-12);
    assert!((cells[3].3 - 5.0 / 6.0).abs() < 1e-12);
    for (i, c) in cells.iter().enumerate() {
        assert_eq!(c.0, i as i64);
        assert!((c.3 - c.2 - 1.0 / 6.0).abs() < 1e-12);
    }
}
