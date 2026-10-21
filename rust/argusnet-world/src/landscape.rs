//! Compose the ported terrain parts into the builders the runtime calls.
//!
//! P4.3a–f ported the pieces: the noise primitives, the grid operations, the
//! semantic pass, the land-cover classification, `procedural_terrain_grid` and
//! the polygon overlay. Each was verified against its own fixture in isolation.
//! This module is what makes them load-bearing — the three functions the rest
//! of the system actually calls:
//!
//! * [`build_terrain_layer`] — preset and seed to a tiled height field plus the
//!   six semantic masks.
//! * [`build_land_cover_layer`] — that terrain to a class and density raster.
//! * [`build_landscape`] — both, with the resolution rule that couples them.
//!
//! There is no new mathematics here. Everything this does is call something
//! already under test, which is why the fixture for this slice compares whole
//! builder outputs rather than re-checking any individual formula.
//!
//! # The DEM and hybrid sources stay in Python
//!
//! `TerrainBuildConfig.terrain_source` accepts `procedural`, `dem` and
//! `hybrid` in the reference. Only `procedural` is built here, deliberately:
//! P4.3's own scope note says "DEM/GIS decoding may stay in offline import
//! tools, but the Rust runtime accepts a documented preprocessed world bundle
//! directly", and the DEM path pulls in GeoTIFF reading and CRS projection —
//! a large dependency surface serving an import step, not the tick loop. The
//! runtime's requirement is the one P4.3 states: *"Pure synthetic defaults MUST
//! need neither Python nor GLB baking."* That is what this delivers.
//!
//! [`build_terrain_layer`] therefore rejects the other two sources by name
//! rather than silently treating them as procedural, which would generate a
//! world that ignores the DEM the caller asked for.
//!
//! # Two resolutions, and the one that is derived
//!
//! Terrain and land cover are rasterised at *different* resolutions. The
//! terrain's comes from the config (default 5 m); the land cover's defaults to
//! `max(10, terrain_resolution * 2)` and can be overridden. So the mask
//! resampler in [`crate::land_cover_build`] does real work rather than an
//! identity map, and a port that shared one resolution would pass every
//! isolated fixture and still build the wrong world.

use crate::grid_ops::{Grid, Mask};
use crate::land_cover_build::{
    apply_land_cover_overlay, classify_land_cover, sample_mask_to_grid, CoverGrid, CoverMask,
    LandCoverInputs, LandCoverPatch, SampledMasks, SeasonalVariation,
};
use crate::noise::{fbm, stable_seed, Bounds2D, FbmConfig};
use crate::obstacles::Obstacle;
use crate::semantic::{terrain_profile_for_preset, SemanticMasks, TerrainProfile};
use crate::terrain_grid::{procedural_terrain_grid, TerrainGridRequest};
use crate::tiled::{self, TiledTerrain};
use crate::TerrainQuery;

/// `terrain_resolution_m or 5.0` in the reference.
pub const DEFAULT_TERRAIN_RESOLUTION_M: f64 = 5.0;

/// The floor on the land-cover resolution, and the terrain multiplier under it.
const LAND_COVER_MIN_RESOLUTION_M: f64 = 10.0;
const LAND_COVER_TERRAIN_MULTIPLE: f64 = 2.0;

/// The reference's `_KNOWN_TERRAIN_SOURCES`, minus the two this crate declines.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TerrainSource {
    Procedural,
}

/// Mirrors `TerrainBuildConfig`, restricted to the procedural source.
#[derive(Debug, Clone, PartialEq)]
pub struct TerrainBuildConfig {
    pub terrain_preset: String,
    /// `None` derives the seed from the preset, exactly as `effective_seed` does.
    pub terrain_seed: Option<i64>,
    pub detail_strength: f64,
    pub terrain_resolution_m: Option<f64>,
    pub season_month: u32,
    pub tile_size_cells: usize,
    pub ground_plane_m: f64,
}

impl Default for TerrainBuildConfig {
    fn default() -> Self {
        Self {
            terrain_preset: "alpine".to_string(),
            terrain_seed: None,
            detail_strength: 1.0,
            terrain_resolution_m: None,
            season_month: 7,
            tile_size_cells: 256,
            ground_plane_m: 0.0,
        }
    }
}

impl TerrainBuildConfig {
    /// `terrain_seed` when set, otherwise `_stable_seed("terrain", preset)`.
    ///
    /// Note the parts: `["terrain", preset]` and **not** the seed — this is a
    /// different derivation from the one `procedural_terrain_grid` uses
    /// internally, which hashes `["profile", preset, seed]`. Two stable seeds,
    /// one feeding the other.
    pub fn effective_seed(&self) -> i64 {
        self.terrain_seed
            .unwrap_or_else(|| stable_seed(&["terrain", &self.terrain_preset]))
    }

    pub fn resolution_m(&self) -> f64 {
        self.terrain_resolution_m
            .unwrap_or(DEFAULT_TERRAIN_RESOLUTION_M)
    }

    /// The reference's `__post_init__`, which rejects before building rather
    /// than producing a degenerate world.
    pub fn validate(&self) -> Result<(), String> {
        if let Some(resolution) = self.terrain_resolution_m {
            if !resolution.is_finite() || resolution <= 0.0 {
                return Err(
                    "terrain_resolution_m must be finite and positive when provided".into(),
                );
            }
        }
        if self.tile_size_cells < 2 {
            return Err("tile_size_cells must be at least 2".into());
        }
        if !(1..=12).contains(&self.season_month) {
            return Err("season_month must be in 1..12".into());
        }
        if !self.detail_strength.is_finite() || self.detail_strength < 0.0 {
            return Err("detail_strength must be finite and non-negative".into());
        }
        Ok(())
    }
}

/// A built terrain: the tiled field the runtime samples, plus the masks the
/// land-cover pass needs.
///
/// The source grid is kept alongside the tiles. `TiledTerrain` resamples into
/// padded tiles on construction, so reading a height back out of it is not
/// bit-identical to the grid it was built from — and the land-cover pass
/// samples the *tiled* field, exactly as the reference does through
/// `TerrainLayer.height_at_many`. Keeping both makes that distinction explicit
/// instead of leaving a reader to wonder which one a caller meant.
pub struct BuiltTerrain {
    pub tiled: TiledTerrain,
    pub masks: SemanticMasks,
    pub source_heights_m: Grid,
    pub profile: TerrainProfile,
    pub bounds_xy_m: Bounds2D,
    pub resolution_m: f64,
    pub ground_plane_m: f64,
}

impl BuiltTerrain {
    /// The finite-difference step the reference's gradient sampler uses:
    /// `max(base_resolution * 0.5, 0.25)`.
    ///
    /// Worth stating because `TerrainModel.gradient_at` defaults to `0.5`
    /// while `slope_rad_at` defaults to `1.0` and passes it down — two
    /// different steps for "the gradient" in the same file. This is the one
    /// `TerrainLayer.slope_rad_at_many` uses, which is the path land cover
    /// takes.
    pub fn gradient_delta_m(&self) -> f64 {
        (self.resolution_m * 0.5).max(0.25)
    }

    pub fn height_at(&self, x_m: f64, y_m: f64) -> f64 {
        self.tiled.height_at(x_m, y_m)
    }

    /// `arctan(|gradient|)`, matching `slope_rad_at_many`.
    pub fn slope_rad_at(&self, x_m: f64, y_m: f64) -> f64 {
        let delta = self.gradient_delta_m();
        let gradient = tiled::gradient(|x, y| self.tiled.height_at(x, y), x_m, y_m, delta);
        gradient[0].hypot(gradient[1]).atan()
    }
}

/// Build the terrain from a preset and a seed.
pub fn build_terrain_layer(
    config: &TerrainBuildConfig,
    bounds_xy_m: Bounds2D,
    source: TerrainSource,
) -> Result<BuiltTerrain, String> {
    config.validate()?;
    let TerrainSource::Procedural = source;

    let resolution_m = config.resolution_m();
    let grid = procedural_terrain_grid(TerrainGridRequest {
        bounds_xy_m,
        preset_name: &config.terrain_preset,
        seed: config.effective_seed(),
        resolution_m,
        detail_strength: config.detail_strength,
        ground_plane_m: config.ground_plane_m,
    })?;
    tile_terrain(
        config,
        bounds_xy_m,
        grid.heights_m,
        grid.masks,
        grid.profile,
    )
}

/// A terrain from heights made elsewhere: an imported DEM (R5,
/// `argusnet.terrain`), already on the grid [`grid_values`] gives
/// `bounds_xy_m` at the config's resolution, rows walking y.
///
/// It carries no semantic masks: water, roads and the rest come from the
/// data, not from the shape of the ground, so every mask is empty, as the
/// reference's DEM source leaves them. The preset still picks the profile
/// the land cover is classified with.
///
/// [`grid_values`]: crate::noise::grid_values
pub fn terrain_from_heights(
    config: &TerrainBuildConfig,
    bounds_xy_m: Bounds2D,
    heights_m: Grid,
) -> Result<BuiltTerrain, String> {
    config.validate()?;
    let (x_values, y_values) = crate::noise::grid_values(bounds_xy_m, config.resolution_m());
    if heights_m.nx != y_values.len() || heights_m.ny != x_values.len() {
        return Err(format!(
            "imported heights are {} by {}; the map's grid at {} m is {} by {}",
            heights_m.nx,
            heights_m.ny,
            config.resolution_m(),
            y_values.len(),
            x_values.len()
        ));
    }
    if let Some(low) = heights_m
        .values
        .iter()
        .find(|h| !h.is_finite() || **h < config.ground_plane_m)
    {
        return Err(format!(
            "imported heights must be finite and at or above the ground plane ({} m), got {low}",
            config.ground_plane_m
        ));
    }
    let empty = || Mask::filled(heights_m.nx, heights_m.ny, false);
    let masks = SemanticMasks {
        water: empty(),
        ridge: empty(),
        valley: empty(),
        steep_slope: empty(),
        lowland: empty(),
        road: empty(),
    };
    let profile = terrain_profile_for_preset(&config.terrain_preset);
    tile_terrain(config, bounds_xy_m, heights_m, masks, profile)
}

/// Tile a height grid into the runtime's terrain, keeping the grid.
fn tile_terrain(
    config: &TerrainBuildConfig,
    bounds_xy_m: Bounds2D,
    heights_m: Grid,
    masks: SemanticMasks,
    profile: TerrainProfile,
) -> Result<BuiltTerrain, String> {
    let resolution_m = config.resolution_m();
    // `heights_m` is axis-0-major with axis 0 = y, so `nx` counts ROWS and
    // `ny` counts COLUMNS despite the names. `TiledTerrain::from_grid` wants
    // rows of columns, so the reshape is a row split and moves no element.
    let rows = heights_m.nx;
    let cols = heights_m.ny;
    let row_major: Vec<Vec<f64>> = heights_m.values.chunks(cols).map(<[f64]>::to_vec).collect();
    debug_assert_eq!(row_major.len(), rows);

    let tiled = TiledTerrain::from_grid(
        argusnet_types::Bounds2d {
            x_min_m: bounds_xy_m.x_min_m,
            x_max_m: bounds_xy_m.x_max_m,
            y_min_m: bounds_xy_m.y_min_m,
            y_max_m: bounds_xy_m.y_max_m,
        },
        &row_major,
        resolution_m,
        config.tile_size_cells,
        config.ground_plane_m,
        &[resolution_m],
    )?;

    Ok(BuiltTerrain {
        tiled,
        masks,
        source_heights_m: heights_m,
        profile,
        bounds_xy_m,
        resolution_m,
        ground_plane_m: config.ground_plane_m,
    })
}

/// Everything `build_land_cover_layer` reads beyond the terrain itself.
pub struct LandCoverRequest<'a> {
    pub terrain: &'a BuiltTerrain,
    pub obstacles: &'a [Obstacle],
    pub patches: &'a [LandCoverPatch],
    pub resolution_m: f64,
    pub season: SeasonalVariation,
    pub terrain_preset: &'a str,
    pub seed: i64,
    pub suppress_vegetation: bool,
}

/// The cover grid's cell centres: `min + (i + 0.5) * resolution`, flattened
/// row-major with **row from y**.
fn cover_cell_centres(bounds: Bounds2D, resolution_m: f64) -> (usize, usize, Vec<f64>, Vec<f64>) {
    let cols = ((bounds.width_m() / resolution_m).ceil() as usize).max(1);
    let rows = ((bounds.height_m() / resolution_m).ceil() as usize).max(1);
    let mut xx = Vec::with_capacity(rows * cols);
    let mut yy = Vec::with_capacity(rows * cols);
    for row in 0..rows {
        for col in 0..cols {
            xx.push(bounds.x_min_m + ((col as f64 + 0.5) * resolution_m));
            yy.push(bounds.y_min_m + ((row as f64 + 0.5) * resolution_m));
        }
    }
    (rows, cols, xx, yy)
}

/// Classify land cover over the terrain, then apply the polygon overlay.
pub fn build_land_cover_layer(request: LandCoverRequest<'_>) -> Result<CoverGrid, String> {
    let LandCoverRequest {
        terrain,
        obstacles,
        patches,
        resolution_m,
        season,
        terrain_preset,
        seed,
        suppress_vegetation,
    } = request;
    if !resolution_m.is_finite() || resolution_m <= 0.0 {
        return Err("land-cover resolution must be finite and positive".into());
    }

    let bounds = terrain.bounds_xy_m;
    let (rows, cols, xx, yy) = cover_cell_centres(bounds, resolution_m);

    let heights_m: Vec<f64> = xx
        .iter()
        .zip(&yy)
        .map(|(x, y)| terrain.height_at(*x, *y))
        .collect();
    let slopes_rad: Vec<f64> = xx
        .iter()
        .zip(&yy)
        .map(|(x, y)| terrain.slope_rad_at(*x, *y))
        .collect();

    // The moisture field. Its seed hashes ["land-cover", preset, seed] --
    // a third stable-seed derivation, distinct from the terrain's and from
    // the profile's.
    let noise_seed = stable_seed(&["land-cover", terrain_preset, &seed.to_string()]);
    let base_wavelength_m = bounds.width_m().max(bounds.height_m()) * 0.22;
    let moisture_noise: Vec<f64> = xx
        .iter()
        .zip(&yy)
        .map(|(x, y)| fbm(*x, *y, FbmConfig::new(noise_seed, base_wavelength_m, 4)))
        .collect();

    // The masks live on the TERRAIN grid and have to be resampled onto the
    // cover grid, which is coarser. This is the only caller that supplies the
    // source bounds, so it is the only place the two rasters are related.
    let source = terrain.bounds_xy_m;
    let resample = |mask: &Mask| -> Vec<bool> {
        sample_mask_to_grid(
            &CoverMask::from_axis0_y(mask),
            source.x_min_m,
            source.width_m(),
            source.y_min_m,
            source.height_m(),
            &xx,
            &yy,
        )
    };
    let masks = SampledMasks {
        water: Some(resample(&terrain.masks.water)),
        lowland: Some(resample(&terrain.masks.lowland)),
        valley: Some(resample(&terrain.masks.valley)),
        steep_slope: Some(resample(&terrain.masks.steep_slope)),
        ridge: Some(resample(&terrain.masks.ridge)),
        road: Some(resample(&terrain.masks.road)),
    };

    let mut grid = classify_land_cover(LandCoverInputs {
        rows,
        cols,
        profile: terrain_profile_for_preset(terrain_preset),
        terrain_preset,
        season,
        suppress_vegetation,
        ground_plane_m: terrain.ground_plane_m,
        heights_m: &heights_m,
        slopes_rad: &slopes_rad,
        moisture_noise: &moisture_noise,
        masks: &masks,
    })?;

    apply_land_cover_overlay(&mut grid, &xx, &yy, obstacles, patches, suppress_vegetation)?;
    Ok(grid)
}

/// The land-cover resolution the reference derives when none is given.
pub fn default_land_cover_resolution_m(terrain_resolution_m: f64) -> f64 {
    (terrain_resolution_m * LAND_COVER_TERRAIN_MULTIPLE).max(LAND_COVER_MIN_RESOLUTION_M)
}

/// `SeasonalVariation.from_month`. The four seasons and their two parameters
/// live in `environment.py`, not in the land-cover module, which is why this
/// table is here rather than beside [`SeasonalVariation`].
///
/// Winter is the only season whose foliage factor is not a whole multiple of
/// 0.05 times 190 — `190 * 0.15 = 28.5`, which numpy's `uint8` cast truncates
/// to 28 where a round would give 29. `land_cover.json`'s `winter_forest`
/// scene exists for that one cast.
pub fn season_from_month(month: u32) -> Result<SeasonalVariation, String> {
    let (foliage_density_factor, snow_cover) = match month {
        12 | 1 | 2 => (0.15, true),
        3..=5 => (0.6, false),
        6..=8 => (1.0, false),
        9..=11 => (0.4, false),
        _ => return Err(format!("month must be 1-12, got {month}")),
    };
    Ok(SeasonalVariation {
        foliage_density_factor,
        snow_cover,
    })
}

/// Terrain and land cover together.
pub struct Landscape {
    pub terrain: BuiltTerrain,
    pub land_cover: CoverGrid,
    pub land_cover_resolution_m: f64,
}

/// Build both, with the coupling rule between their resolutions.
pub fn build_landscape(
    config: &TerrainBuildConfig,
    bounds_xy_m: Bounds2D,
    obstacles: &[Obstacle],
    patches: &[LandCoverPatch],
    land_cover_resolution_m: Option<f64>,
    suppress_vegetation: bool,
) -> Result<Landscape, String> {
    let terrain = build_terrain_layer(config, bounds_xy_m, TerrainSource::Procedural)?;
    let cover_resolution = land_cover_resolution_m
        .unwrap_or_else(|| default_land_cover_resolution_m(terrain.resolution_m));
    let land_cover = build_land_cover_layer(LandCoverRequest {
        terrain: &terrain,
        obstacles,
        patches,
        resolution_m: cover_resolution,
        season: season_from_month(config.season_month)?,
        terrain_preset: &config.terrain_preset,
        seed: config.effective_seed(),
        suppress_vegetation,
    })?;
    Ok(Landscape {
        terrain,
        land_cover,
        land_cover_resolution_m: cover_resolution,
    })
}
