//! Parity for the composed builders: `build_terrain_layer` and
//! `build_land_cover_layer` end to end.
//!
//! P4.3a–f each verified one piece in isolation. This checks the thing none of
//! them could: that the pieces are wired together the way the reference wires
//! them. Every failure mode here is a wiring failure — a seed derived from the
//! wrong parts, a resolution shared where it should be doubled, a height read
//! from the source grid instead of the tiled field.

use argusnet_world::land_cover_build::SeasonalVariation;
use argusnet_world::landscape::{
    build_land_cover_layer, build_terrain_layer, default_land_cover_resolution_m,
    season_from_month, LandCoverRequest, TerrainBuildConfig, TerrainSource,
};
use argusnet_world::noise::Bounds2D;
use serde::Deserialize;
use std::collections::BTreeMap;

/// Heights reach this test through `exp` (the terrain's Gaussian envelopes) and
/// a bilinear tile resample; slopes add an `atan`. ADR-006 bounds the first at
/// numpy's `exp`, measured at 5.7e-14 m in P4.3e. This is four orders of
/// magnitude above that and still six below the coarsest threshold the
/// classifier tests against, so it cannot hide a wrong port while absorbing
/// the transcendental.
const SAMPLE_TOLERANCE: f64 = 1.0e-9;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    preset: String,
    season_month: u32,
    terrain_seed: Option<i64>,
    effective_seed: i64,
    detail_strength: String,
    terrain_resolution_m: String,
    cover_resolution_m: String,
    tile_size_cells: usize,
    ground_plane_m: String,
    suppress_vegetation: bool,
    season: Season,
    bounds: Bounds,
    rows: usize,
    cols: usize,
    sampled_heights_m: Vec<String>,
    sampled_slopes_rad: Vec<String>,
    mask_coverage: BTreeMap<String, String>,
    mask_shape: BTreeMap<String, [usize; 2]>,
    classes: Vec<u8>,
    density: Vec<u8>,
}

#[derive(Deserialize)]
struct Season {
    foliage_density_factor: String,
    snow_cover: bool,
}

#[derive(Deserialize)]
struct Bounds {
    x_min_m: String,
    x_max_m: String,
    y_min_m: String,
    y_max_m: String,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/landscape.json"
    );
    let text = std::fs::read_to_string(path).expect("landscape.json");
    serde_json::from_str(&text).expect("landscape.json parses")
}

fn config_for(case: &Case) -> TerrainBuildConfig {
    TerrainBuildConfig {
        terrain_preset: case.preset.clone(),
        terrain_seed: case.terrain_seed,
        detail_strength: f(&case.detail_strength),
        terrain_resolution_m: Some(f(&case.terrain_resolution_m)),
        season_month: case.season_month,
        tile_size_cells: case.tile_size_cells,
        ground_plane_m: f(&case.ground_plane_m),
    }
}

fn bounds_for(case: &Case) -> Bounds2D {
    Bounds2D {
        x_min_m: f(&case.bounds.x_min_m),
        x_max_m: f(&case.bounds.x_max_m),
        y_min_m: f(&case.bounds.y_min_m),
        y_max_m: f(&case.bounds.y_max_m),
    }
}

#[test]
fn the_effective_seed_derivation_matches() {
    // Seam 1. When `terrain_seed` is None the seed is
    // `stable_seed(["terrain", preset])` — NOT the seed the grid builder then
    // derives from it, and not the one the moisture field uses. A port that
    // collapses the three produces a plausible, wrong world.
    let fixture = load();
    let mut derived = 0;
    for case in &fixture.cases {
        let config = config_for(case);
        assert_eq!(
            config.effective_seed(),
            case.effective_seed,
            "{}: effective seed",
            case.name
        );
        if case.terrain_seed.is_none() {
            derived += 1;
        }
    }
    assert!(derived > 0, "no scene exercises the derived branch");
}

#[test]
fn the_land_cover_resolution_rule_matches() {
    // Seam 2, both halves: the *2 multiplier and the 10 m floor.
    let fixture = load();
    let (mut multiplier, mut floor) = (0, 0);
    for case in &fixture.cases {
        let terrain_res = f(&case.terrain_resolution_m);
        let expected = f(&case.cover_resolution_m);
        assert_eq!(
            default_land_cover_resolution_m(terrain_res),
            expected,
            "{}: derived cover resolution",
            case.name
        );
        if terrain_res * 2.0 > 10.0 {
            multiplier += 1;
        } else {
            floor += 1;
        }
    }
    assert!(
        multiplier > 0 && floor > 0,
        "the rule has two halves and only one is covered: \
         {multiplier} multiplier, {floor} floor"
    );
}

#[test]
fn the_season_table_matches() {
    let fixture = load();
    let mut factors = std::collections::BTreeSet::new();
    for case in &fixture.cases {
        let season = season_from_month(case.season_month).expect("a real month");
        assert_eq!(
            season.foliage_density_factor.to_bits(),
            f(&case.season.foliage_density_factor).to_bits(),
            "{}: foliage factor",
            case.name
        );
        assert_eq!(season.snow_cover, case.season.snow_cover, "{}", case.name);
        factors.insert(season.foliage_density_factor.to_bits());
    }
    assert!(factors.len() >= 3, "seasons barely vary");
    assert!(season_from_month(0).is_err());
    assert!(season_from_month(13).is_err());
}

#[test]
fn every_scene_reproduces_the_reference_terrain_and_land_cover() {
    let fixture = load();
    assert!(fixture.cases.len() >= 9, "fixture shrank unexpectedly");
    let mut worst_height = 0.0f64;
    let mut worst_slope = 0.0f64;

    for case in &fixture.cases {
        let config = config_for(case);
        let bounds = bounds_for(case);
        let terrain = build_terrain_layer(&config, bounds, TerrainSource::Procedural)
            .unwrap_or_else(|e| panic!("{}: build_terrain_layer: {e}", case.name));

        // The masks the classifier will see, cross-checked before it runs so a
        // failure below localises to the classifier rather than the generator.
        let named: [(&str, &argusnet_world::grid_ops::Mask); 6] = [
            ("water", &terrain.masks.water),
            ("ridge", &terrain.masks.ridge),
            ("valley", &terrain.masks.valley),
            ("steep_slope", &terrain.masks.steep_slope),
            ("lowland", &terrain.masks.lowland),
            ("road", &terrain.masks.road),
        ];
        for (name, mask) in named {
            let shape = case.mask_shape.get(name).expect("recorded mask shape");
            assert_eq!([mask.nx, mask.ny], *shape, "{}: {name} shape", case.name);
            let covered =
                mask.values.iter().filter(|v| **v).count() as f64 / mask.values.len() as f64;
            let expected = f(case.mask_coverage.get(name).expect("recorded coverage"));
            assert!(
                (covered - expected).abs() <= 1.0e-12,
                "{}: {name} coverage {covered} != {expected}",
                case.name
            );
        }

        // Seam 3 and 4: sampled through the TILED field, with the gradient step
        // `max(resolution * 0.5, 0.25)`.
        let cover_res = f(&case.cover_resolution_m);
        for row in 0..case.rows {
            for col in 0..case.cols {
                let k = row * case.cols + col;
                let x = bounds.x_min_m + ((col as f64 + 0.5) * cover_res);
                let y = bounds.y_min_m + ((row as f64 + 0.5) * cover_res);

                let dh = (terrain.height_at(x, y) - f(&case.sampled_heights_m[k])).abs();
                worst_height = worst_height.max(dh);
                assert!(
                    dh <= SAMPLE_TOLERANCE,
                    "{}: height at row {row}, col {col} out by {dh} m",
                    case.name
                );

                let ds = (terrain.slope_rad_at(x, y) - f(&case.sampled_slopes_rad[k])).abs();
                worst_slope = worst_slope.max(ds);
                assert!(
                    ds <= SAMPLE_TOLERANCE,
                    "{}: slope at row {row}, col {col} out by {ds} rad",
                    case.name
                );
            }
        }

        let grid = build_land_cover_layer(LandCoverRequest {
            terrain: &terrain,
            obstacles: &[],
            patches: &[],
            resolution_m: cover_res,
            season: SeasonalVariation {
                foliage_density_factor: f(&case.season.foliage_density_factor),
                snow_cover: case.season.snow_cover,
            },
            terrain_preset: &case.preset,
            seed: case.effective_seed,
            suppress_vegetation: case.suppress_vegetation,
        })
        .unwrap_or_else(|e| panic!("{}: build_land_cover_layer: {e}", case.name));

        assert_eq!(
            (grid.rows, grid.cols),
            (case.rows, case.cols),
            "{}",
            case.name
        );
        for k in 0..case.rows * case.cols {
            let (row, col) = (k / case.cols, k % case.cols);
            assert_eq!(
                grid.classes[k], case.classes[k],
                "{}: class at row {row}, col {col}",
                case.name
            );
            assert_eq!(
                grid.density[k], case.density[k],
                "{}: density at row {row}, col {col}",
                case.name
            );
        }
    }
    println!("worst height {worst_height:e} m, worst slope {worst_slope:e} rad");
}

#[test]
fn a_rejected_config_does_not_build_a_degenerate_world() {
    // The reference validates in `__post_init__` and raises. Building anyway
    // with a zero tile size or a twelfth-month-plus season would produce a
    // world rather than an error, which is worse than failing.
    let bounds = Bounds2D {
        x_min_m: 0.0,
        x_max_m: 100.0,
        y_min_m: 0.0,
        y_max_m: 100.0,
    };
    let base = TerrainBuildConfig::default();
    for (label, config) in [
        (
            "tile_size_cells below two",
            TerrainBuildConfig {
                tile_size_cells: 1,
                ..base.clone()
            },
        ),
        (
            "month out of range",
            TerrainBuildConfig {
                season_month: 13,
                ..base.clone()
            },
        ),
        (
            "negative detail",
            TerrainBuildConfig {
                detail_strength: -0.5,
                ..base.clone()
            },
        ),
        (
            "zero resolution",
            TerrainBuildConfig {
                terrain_resolution_m: Some(0.0),
                ..base.clone()
            },
        ),
        (
            "non-finite detail",
            TerrainBuildConfig {
                detail_strength: f64::NAN,
                ..base.clone()
            },
        ),
    ] {
        assert!(
            build_terrain_layer(&config, bounds, TerrainSource::Procedural).is_err(),
            "{label} was accepted"
        );
    }
}
