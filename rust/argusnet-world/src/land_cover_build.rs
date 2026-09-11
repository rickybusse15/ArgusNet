//! Land-cover classification: turn heights, slopes and the semantic masks into
//! a class and density raster.
//!
//! # This grid is `(row = y, col = x)` — the transpose of everything else here
//!
//! `build_land_cover_layer` builds its coordinates with
//! `np.meshgrid(x_values, y_values)` and **no `indexing="ij"`**, so numpy's
//! default applies and axis 0 is **y**. `grid_ops::Grid` and
//! `semantic::semantic_masks` are documented as `(i = x, j = y)`. The two
//! conventions sit in the same file in the reference and are transposes of
//! each other.
//!
//! [`CoverGrid`] is therefore spelled `rows`/`cols`, not `nx`/`ny`, so a
//! reader cannot mistake one for the other. Getting this wrong produces a
//! land-cover map that is plausible, self-consistent and rotated — the same
//! class of bug the ellipse fix in P4.3c cost us.
//!
//! # `np.rint` is banker's rounding
//!
//! `sample_mask_to_grid` resamples with `np.rint`, which rounds **ties to
//! even**, not half away from zero. Rust's `f64::round()` is half-away, so
//! this uses `round_ties_even()`. That is the **third** place this trap has
//! appeared in the migration — the route-cache key and the POI boost were the
//! first two.
//!
//! # The water-bias clamp is not the one next door
//!
//! `_semantic_masks` clamps the bias to `[0.01, 0.35]`. Here it is
//! `[0.03, 0.30]`, and it only applies to four presets. Two different clamps
//! on the same profile field, forty lines apart in the reference.

use crate::grid_ops::Mask;
use crate::semantic::{quantile, TerrainProfile};

/// The land-cover classes, with the reference's integer values. These reach
/// the viewer and the scene package, so the numbers are the contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum LandCoverClass {
    Open = 0,
    Urban = 1,
    Forest = 2,
    Water = 3,
    Scrub = 4,
    Wetland = 5,
    Rocky = 6,
    Snow = 7,
    Road = 8,
}

/// Season-dependent parameters. `foliage_density_factor` is constrained to
/// `[0, 1]` by the reference's own validation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SeasonalVariation {
    pub foliage_density_factor: f64,
    pub snow_cover: bool,
}

/// A `rows x cols` raster in **row-major (y, x)** order. See the module docs:
/// this is deliberately *not* [`crate::grid_ops::Grid`]'s convention.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverGrid {
    pub rows: usize,
    pub cols: usize,
    pub classes: Vec<u8>,
    pub density: Vec<u8>,
}

/// The semantic masks this pass reads, already resampled onto the cover grid.
#[derive(Debug, Clone, Default)]
pub struct SampledMasks {
    pub water: Option<Vec<bool>>,
    pub lowland: Option<Vec<bool>>,
    pub valley: Option<Vec<bool>>,
    pub steep_slope: Option<Vec<bool>>,
    pub ridge: Option<Vec<bool>>,
    pub road: Option<Vec<bool>>,
}

impl SampledMasks {
    /// A missing mask reads as all-false, exactly as
    /// `sampled_masks.get(name, zeros(...))` does.
    fn get(&self, which: &Option<Vec<bool>>, index: usize) -> bool {
        which.as_ref().is_some_and(|mask| mask[index])
    }
}

/// A source mask in this module's `(row = y, col = x)` convention.
///
/// [`crate::grid_ops::Mask`] names its axes `nx`/`ny`, which reads as
/// `(i = x, j = y)`. The masks that reach this pass do **not** use that layout:
/// `procedural_terrain_grid` builds its coordinate grids with numpy's default
/// `meshgrid`, so their axis 0 is y. Passing such a mask around in a struct
/// whose fields say otherwise is a lie that happens to work, and it is the same
/// unstated-layout mistake that put the lake ellipse 0.013 m out in P4.3c.
///
/// So the conversion is named and happens in exactly one place, rather than
/// being an assumption inside a function body.
#[derive(Debug, Clone, PartialEq)]
pub struct CoverMask {
    pub rows: usize,
    pub cols: usize,
    pub values: Vec<bool>,
}

impl CoverMask {
    pub fn new(rows: usize, cols: usize, values: Vec<bool>) -> Result<Self, &'static str> {
        if rows.checked_mul(cols) != Some(values.len()) {
            return Err("mask length does not match its shape");
        }
        Ok(Self { rows, cols, values })
    }

    /// Reinterpret a [`Mask`] whose **axis 0 is y** — the layout
    /// `procedural_terrain_grid` produces. No elements move: only the names of
    /// the two axes change, because `Mask` stores axis 0 major just as this
    /// does.
    pub fn from_axis0_y(mask: &Mask) -> Self {
        Self {
            rows: mask.nx,
            cols: mask.ny,
            values: mask.values.clone(),
        }
    }
}

/// Resample a source mask onto target coordinates.
///
/// The source is indexed `[row, col]` with **row from y and col from x**, and
/// the scale factors are `max(cols - 1, 1)` and `max(rows - 1, 1)` — note the
/// floor at **one**, not zero, so a single-column source still divides by 1.
///
/// Rounding is `np.rint`: ties to even. See the module docs.
pub fn sample_mask_to_grid(
    mask: &CoverMask,
    source_x_min_m: f64,
    source_width_m: f64,
    source_y_min_m: f64,
    source_height_m: f64,
    target_xx: &[f64],
    target_yy: &[f64],
) -> Vec<bool> {
    // `mask_array.ndim != 2 or mask_array.size == 0` -> all false.
    if mask.values.is_empty() {
        return vec![false; target_xx.len()];
    }
    let rows = mask.rows;
    let cols = mask.cols;
    let width = source_width_m.max(1.0e-9);
    let height = source_height_m.max(1.0e-9);
    let col_scale = (cols.saturating_sub(1)).max(1) as f64;
    let row_scale = (rows.saturating_sub(1)).max(1) as f64;

    target_xx
        .iter()
        .zip(target_yy)
        .map(|(x, y)| {
            let col = (((x - source_x_min_m) / width) * col_scale)
                .round_ties_even()
                .clamp(0.0, (cols - 1) as f64) as usize;
            let row = (((y - source_y_min_m) / height) * row_scale)
                .round_ties_even()
                .clamp(0.0, (rows - 1) as f64) as usize;
            mask.values[row * cols + col]
        })
        .collect()
}

/// Everything the classification reads.
pub struct LandCoverInputs<'a> {
    pub rows: usize,
    pub cols: usize,
    pub profile: TerrainProfile,
    pub terrain_preset: &'a str,
    pub season: SeasonalVariation,
    pub suppress_vegetation: bool,
    pub ground_plane_m: f64,
    /// Sampled at each cell centre, row-major `(y, x)`.
    pub heights_m: &'a [f64],
    pub slopes_rad: &'a [f64],
    /// `_fbm(...)` evaluated at each cell centre, before the moisture blend.
    pub moisture_noise: &'a [f64],
    pub masks: &'a SampledMasks,
}

/// Classify every cell.
///
/// The order is the contract: **water, wetland, snow, rocky, forest, scrub,
/// road**, each assignment overwriting the last. The masks are *not* mutually
/// exclusive by construction — `snow_mask` excludes only water, so snow
/// overwrites wetland wherever both hold. Reordering these blocks changes the
/// map even though every individual predicate is unchanged.
///
/// Density follows two different rules, also deliberately: wetland, snow,
/// rocky and road **assign**, while forest and scrub take `max(existing, new)`.
pub fn classify_land_cover(inputs: LandCoverInputs<'_>) -> Result<CoverGrid, &'static str> {
    let LandCoverInputs {
        rows,
        cols,
        profile,
        terrain_preset,
        season,
        suppress_vegetation,
        ground_plane_m,
        heights_m,
        slopes_rad,
        moisture_noise,
        masks,
    } = inputs;

    let cells = rows.checked_mul(cols).ok_or("grid too large")?;
    if heights_m.len() != cells || slopes_rad.len() != cells || moisture_noise.len() != cells {
        return Err("input grids must match the requested shape");
    }

    let h_min = heights_m.iter().copied().fold(f64::INFINITY, f64::min);
    let h_max = heights_m.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (h_max - h_min).max(1.0);
    let elev_norm: Vec<f64> = heights_m.iter().map(|h| (h - h_min) / span).collect();
    let moisture: Vec<f64> = moisture_noise
        .iter()
        .map(|n| (profile.moisture + (0.28 * n)).clamp(0.0, 1.0))
        .collect();

    let mut classes = vec![LandCoverClass::Open as u8; cells];
    let mut density = vec![0u8; cells];

    // --- water -------------------------------------------------------
    // A SECOND water-level quantile, clamped to [0.03, 0.30] and applied only
    // to four presets. `_semantic_masks` clamps the same field to
    // [0.01, 0.35]. See the module docs.
    let extra_water_level = matches!(
        terrain_preset,
        "coastal" | "lake_district" | "river_valley" | "arctic_tundra"
    )
    .then(|| quantile(heights_m, profile.water_bias.clamp(0.03, 0.30)))
    .flatten();

    let mut water_mask = vec![false; cells];
    for k in 0..cells {
        let mut wet = heights_m[k] <= ground_plane_m + 1.0 || masks.get(&masks.water, k);
        if let Some(level) = extra_water_level {
            wet |= heights_m[k] <= level && slopes_rad[k] < 0.16;
        }
        water_mask[k] = wet;
        if wet {
            classes[k] = LandCoverClass::Water as u8;
        }
    }

    // --- wetland (vegetation-gated) ----------------------------------
    let mut wetland_mask = vec![false; cells];
    for k in 0..cells {
        wetland_mask[k] = !water_mask[k]
            && (masks.get(&masks.lowland, k)
                || masks.get(&masks.valley, k)
                || (moisture[k] > 0.68 && elev_norm[k] < 0.30))
            && slopes_rad[k] < 0.14;
        if wetland_mask[k] && !suppress_vegetation {
            classes[k] = LandCoverClass::Wetland as u8;
            density[k] = 120;
        }
    }

    // --- snow ---------------------------------------------------------
    // Excludes water only, so it overwrites wetland where both hold.
    let mut snow_mask = vec![false; cells];
    for k in 0..cells {
        snow_mask[k] = !water_mask[k]
            && ((season.snow_cover && elev_norm[k] > profile.snow_bias)
                || terrain_preset == "arctic_tundra"
                || (matches!(terrain_preset, "alpine" | "mountain_pass") && elev_norm[k] > 0.82));
        if snow_mask[k] {
            classes[k] = LandCoverClass::Snow as u8;
            density[k] = 40;
        }
    }

    // --- rocky ---------------------------------------------------------
    let mut rocky_mask = vec![false; cells];
    for k in 0..cells {
        rocky_mask[k] = !water_mask[k]
            && !snow_mask[k]
            && (masks.get(&masks.steep_slope, k)
                || masks.get(&masks.ridge, k)
                || slopes_rad[k] > 0.52
                || elev_norm[k] > 0.78);
        if rocky_mask[k] {
            classes[k] = LandCoverClass::Rocky as u8;
            density[k] = 60;
        }
    }

    // --- forest and scrub (vegetation-gated) -----------------------------
    if !suppress_vegetation {
        let forest_moisture = (0.55 - (profile.vegetation_bias * 0.18)).max(0.35);
        let forest_density = (190.0 * season.foliage_density_factor).clamp(0.0, 255.0) as u8;
        let scrub_density = (105.0 * season.foliage_density_factor).clamp(0.0, 255.0) as u8;

        let mut forest_mask = vec![false; cells];
        for k in 0..cells {
            forest_mask[k] = !water_mask[k]
                && !snow_mask[k]
                && !rocky_mask[k]
                && moisture[k] > forest_moisture
                && elev_norm[k] > 0.18
                && slopes_rad[k] < 0.45;
            if forest_mask[k] {
                classes[k] = LandCoverClass::Forest as u8;
                density[k] = density[k].max(forest_density);
            }
        }
        for k in 0..cells {
            let scrub = !water_mask[k]
                && !snow_mask[k]
                && !rocky_mask[k]
                && !forest_mask[k]
                && (moisture[k] < 0.42 || slopes_rad[k] > 0.24);
            if scrub {
                classes[k] = LandCoverClass::Scrub as u8;
                density[k] = density[k].max(scrub_density);
            }
        }
    }

    // --- road ------------------------------------------------------------
    // Assigns density ZERO, so a road over forest clears the foliage rather
    // than keeping the larger value. The only place density goes down.
    for k in 0..cells {
        if masks.get(&masks.road, k) && !water_mask[k] {
            classes[k] = LandCoverClass::Road as u8;
            density[k] = 0;
        }
    }

    Ok(CoverGrid {
        rows,
        cols,
        classes,
        density,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::semantic::terrain_profile_for_preset;

    fn summer() -> SeasonalVariation {
        SeasonalVariation {
            foliage_density_factor: 1.0,
            snow_cover: false,
        }
    }

    fn flat_inputs<'a>(
        preset: &'a str,
        heights: &'a [f64],
        slopes: &'a [f64],
        noise: &'a [f64],
        masks: &'a SampledMasks,
    ) -> LandCoverInputs<'a> {
        LandCoverInputs {
            rows: 1,
            cols: heights.len(),
            profile: terrain_profile_for_preset(preset),
            terrain_preset: preset,
            season: summer(),
            suppress_vegetation: false,
            ground_plane_m: 0.0,
            heights_m: heights,
            slopes_rad: slopes,
            moisture_noise: noise,
            masks,
        }
    }

    #[test]
    fn mismatched_input_lengths_are_rejected() {
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("default", &[1.0, 2.0], &[0.0, 0.0], &[0.0, 0.0], &masks);
        inputs.rows = 2;
        assert!(classify_land_cover(inputs).is_err());
    }

    #[test]
    fn snow_overwrites_wetland_because_it_only_excludes_water() {
        // A cell that is both wetland (lowland, flat, low) and snow (snow
        // cover on, elevation above the bias) must come out SNOW: the snow
        // block runs later and its mask excludes only water.
        let heights = [0.0, 100.0];
        let slopes = [0.0, 0.0];
        let noise = [0.0, 0.0];
        let masks = SampledMasks {
            lowland: Some(vec![true, true]),
            ..SampledMasks::default()
        };
        let mut inputs = flat_inputs("default", &heights, &slopes, &noise, &masks);
        inputs.ground_plane_m = -50.0; // keep the low cell out of the water class
        inputs.season = SeasonalVariation {
            foliage_density_factor: 1.0,
            snow_cover: true,
        };
        let grid = classify_land_cover(inputs).unwrap();
        // The high cell is above snow_bias, so snow wins over wetland there.
        assert_eq!(grid.classes[1], LandCoverClass::Snow as u8);
        assert_eq!(grid.density[1], 40, "snow assigns, it does not take a max");
    }

    #[test]
    fn a_road_clears_density_rather_than_keeping_the_larger_value() {
        let heights = [50.0, 50.0];
        let slopes = [0.0, 0.0];
        let noise = [1.0, 1.0]; // wet enough for forest
        let masks = SampledMasks {
            road: Some(vec![false, true]),
            ..SampledMasks::default()
        };
        let mut inputs = flat_inputs("jungle_canopy", &heights, &slopes, &noise, &masks);
        inputs.ground_plane_m = -50.0;
        let grid = classify_land_cover(inputs).unwrap();
        assert_eq!(grid.classes[1], LandCoverClass::Road as u8);
        assert_eq!(
            grid.density[1], 0,
            "road is the only rule that lowers density"
        );
        assert!(grid.density[0] > 0, "its neighbour keeps its foliage");
    }

    #[test]
    fn suppressing_vegetation_leaves_wetland_forest_and_scrub_unassigned() {
        let heights = [50.0; 4];
        let slopes = [0.0; 4];
        let noise = [1.0; 4];
        let masks = SampledMasks {
            lowland: Some(vec![true; 4]),
            ..SampledMasks::default()
        };
        let mut inputs = flat_inputs("jungle_canopy", &heights, &slopes, &noise, &masks);
        inputs.ground_plane_m = -50.0;
        inputs.suppress_vegetation = true;
        let grid = classify_land_cover(inputs).unwrap();
        for class in &grid.classes {
            assert_ne!(*class, LandCoverClass::Wetland as u8);
            assert_ne!(*class, LandCoverClass::Forest as u8);
            assert_ne!(*class, LandCoverClass::Scrub as u8);
        }
    }

    #[test]
    fn only_four_presets_take_the_second_water_quantile() {
        let heights: Vec<f64> = (0..20).map(|i| 10.0 + i as f64).collect();
        let slopes = [0.0; 20];
        let noise = [0.0; 20];
        let masks = SampledMasks::default();
        let water_cells = |preset: &str| {
            let mut inputs = flat_inputs(preset, &heights, &slopes, &noise, &masks);
            inputs.ground_plane_m = -50.0;
            classify_land_cover(inputs)
                .unwrap()
                .classes
                .iter()
                .filter(|c| **c == LandCoverClass::Water as u8)
                .count()
        };
        // river_valley is on the list and has a high enough bias to select some.
        assert!(water_cells("river_valley") > 0);
        // rolling_highlands is not on the list, so no second quantile applies.
        assert_eq!(water_cells("rolling_highlands"), 0);
    }

    #[test]
    fn the_resampler_rounds_ties_to_even() {
        // A 4-column source across 3 m: scale is 3, so x = 0.5 maps to
        // 0.5/3*3 = 0.5, a tie. Banker's rounding gives 0, half-away gives 1.
        let mask = CoverMask::new(1, 4, vec![false, true, true, true]).unwrap();
        let sampled = sample_mask_to_grid(&mask, 0.0, 3.0, 0.0, 1.0, &[0.5], &[0.0]);
        assert!(!sampled[0], "0.5 must round to 0 (ties to even), not to 1");
        // 1.5 is also a tie, and rounds to 2 — the even neighbour.
        let mask2 = CoverMask::new(1, 4, vec![false, false, true, false]).unwrap();
        let sampled2 = sample_mask_to_grid(&mask2, 0.0, 3.0, 0.0, 1.0, &[1.5], &[0.0]);
        assert!(sampled2[0], "1.5 must round to 2 (ties to even), not to 1");
    }

    #[test]
    fn an_empty_source_mask_samples_as_all_false() {
        let mask = CoverMask::new(0, 0, vec![]).unwrap();
        let sampled = sample_mask_to_grid(&mask, 0.0, 1.0, 0.0, 1.0, &[0.0, 1.0], &[0.0, 1.0]);
        assert_eq!(sampled, vec![false, false]);
    }

    // ---------------------------------------------------------------------
    // Boundaries the fixture cannot reach.
    //
    // The parity fixture drives the reference, so it can only exercise inputs
    // the terrain generator actually produces. These do not occur there:
    // `<=` versus `<` needs a value landing bit-exactly on a threshold, and
    // the degenerate shapes need a one-row mask or a zero-width source, which
    // `Bounds2D` forbids. The Rust function takes all of it as data, so a unit
    // test can put the input exactly on the boundary.
    // ---------------------------------------------------------------------

    /// The index expression, with the two floors made switchable, so the
    /// equivalences below can be *computed* rather than argued.
    fn index_with(ratio: f64, extent_count: usize, floor_scale: bool) -> usize {
        let scale = if floor_scale {
            (extent_count.saturating_sub(1)).max(1) as f64
        } else {
            extent_count.saturating_sub(1) as f64
        };
        (ratio * scale)
            .round_ties_even()
            .clamp(0.0, (extent_count - 1) as f64) as usize
    }

    #[test]
    fn the_scale_floor_is_defensive_and_provably_cannot_change_an_index() {
        // `max(cols - 1, 1)` differs from `cols - 1` only when the source has
        // ONE column, where the floor makes the scale 1 instead of 0. It
        // cannot change the answer there, because the only valid index is 0
        // and `clamp(_, 0.0, 0.0)` returns 0 for every input -- including the
        // NaN that `inf * 0.0` produces, which `as usize` saturates to 0.
        //
        // An earlier version of this test asserted that a 1x1 source "answers
        // true everywhere". That is true of both spellings, so it caught
        // nothing: a property that holds either way is not an assertion. This
        // one runs both spellings and requires them to agree.
        for ratio in [
            0.0,
            0.5,
            1.0,
            -1.0,
            1.0e9,
            -1.0e9,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::NAN,
        ] {
            assert_eq!(
                index_with(ratio, 1, true),
                index_with(ratio, 1, false),
                "the scale floor changed the index of a one-cell axis at ratio {ratio}"
            );
        }
        // And for any larger axis the two spellings are literally the same
        // number, so there is nothing to check beyond that they are.
        for count in 2..64usize {
            assert_eq!((count - 1).max(1), count - 1);
        }
    }

    #[test]
    fn the_extent_floor_is_unreachable_through_the_reference() {
        // `max(width, 1e-9)` differs from `width` only for an extent below a
        // nanometre. `Bounds2D` rejects a non-positive extent outright, and
        // the terrain layers this pass reads are hundreds of metres across, so
        // the reference cannot produce one.
        //
        // At exactly zero the two agree anyway: the ratio becomes +/-inf or
        // NaN, and the clamp sends all three to a valid index. Stated as
        // UNREACHABLE rather than equivalent, because for an extent strictly
        // between 0 and 1e-9 the floor really would change the answer -- no
        // caller can construct one, and claiming equivalence would be wrong.
        let mask = CoverMask::new(1, 3, vec![true, false, true]).unwrap();
        for x in [-1.0, 0.0, 1.0] {
            let floored = sample_mask_to_grid(&mask, 0.0, 1.0e-9, 0.0, 1.0, &[x], &[0.0]);
            let bare = {
                // The same computation with the floor removed.
                let ratio = (x - 0.0) / 0.0;
                let col = (ratio * 2.0).round_ties_even().clamp(0.0, 2.0) as usize;
                vec![mask.values[col]]
            };
            assert_eq!(
                floored, bare,
                "the extent floor changed the answer at x={x}"
            );
        }
    }

    #[test]
    fn the_x_and_y_origins_are_not_interchangeable() {
        // With both origins at zero a port that reads one where it means the
        // other is invisible. The fixture's bounds start at (-120, 75) for the
        // same reason; this pins it without a fixture.
        let mask = CoverMask::new(2, 2, vec![true, false, false, false]).unwrap();
        let with_x = sample_mask_to_grid(&mask, 10.0, 4.0, 0.0, 4.0, &[10.0], &[0.0]);
        let with_y = sample_mask_to_grid(&mask, 0.0, 4.0, 10.0, 4.0, &[10.0], &[0.0]);
        assert_ne!(
            with_x, with_y,
            "swapping which origin applies to which axis must change the answer"
        );
    }

    #[test]
    fn the_water_tests_include_their_boundaries() {
        // `heights <= ground_plane + 1.0`, not `<`. A height landing exactly on
        // the margin is water. The terrain generator never produces one, so
        // this is constructed.
        let heights = [1.0, 1.0 + f64::EPSILON];
        let masks = SampledMasks::default();
        let inputs = flat_inputs("default", &heights, &[0.0; 2], &[0.0; 2], &masks);
        let grid = classify_land_cover(inputs).unwrap();
        assert_eq!(
            grid.classes[0],
            LandCoverClass::Water as u8,
            "exactly ground_plane + 1.0 is water"
        );
        assert_ne!(
            grid.classes[1],
            LandCoverClass::Water as u8,
            "one ULP above it is not"
        );
    }

    #[test]
    fn the_second_water_quantile_includes_its_level() {
        // `heights <= water_level`, not `<`. With an odd count the quantile
        // lands exactly on a sample, so the boundary is a real height.
        let heights: Vec<f64> = (0..21).map(|i| 100.0 + i as f64).collect();
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("river_valley", &heights, &[0.0; 21], &[0.0; 21], &masks);
        inputs.ground_plane_m = -500.0; // nothing is water by the ground plane
                                        // water_bias 0.16 clamps to 0.16; 0.16 * 20 = 3.2, so the level sits
                                        // between samples 3 and 4. Drive the profile so it lands ON a sample.
        inputs.profile.water_bias = 0.25; // 0.25 * 20 = 5 exactly -> heights[5]
        let grid = classify_land_cover(inputs).unwrap();
        assert_eq!(
            grid.classes[5],
            LandCoverClass::Water as u8,
            "a cell exactly at the water level is water"
        );
        assert_ne!(
            grid.classes[6],
            LandCoverClass::Water as u8,
            "the next one up is not"
        );
    }

    #[test]
    fn the_rocky_slope_test_excludes_its_threshold() {
        // `slopes > 0.52`, strictly. A slope landing bit-exactly on 0.52 is
        // NOT rocky -- the opposite of the water tests above, in the same
        // function.
        let heights = [50.0; 2];
        let slopes = [0.52, 0.52 + f64::EPSILON];
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("default", &heights, &slopes, &[0.0; 2], &masks);
        inputs.ground_plane_m = -50.0;
        inputs.suppress_vegetation = true; // keep scrub from overwriting
        let grid = classify_land_cover(inputs).unwrap();
        assert_ne!(
            grid.classes[0],
            LandCoverClass::Rocky as u8,
            "exactly 0.52 is not steep enough"
        );
        assert_eq!(
            grid.classes[1],
            LandCoverClass::Rocky as u8,
            "one ULP above it is"
        );
    }

    #[test]
    fn the_snow_bias_test_excludes_its_threshold() {
        // `elev_norm > snow_bias`, strictly. The minimum-height cell has
        // elev_norm exactly 0.0, so a bias of 0.0 puts the boundary on it.
        // No preset has a snow_bias below 0.2, so only a driven profile
        // reaches this.
        let heights = [10.0, 30.0];
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("default", &heights, &[0.0; 2], &[0.0; 2], &masks);
        inputs.ground_plane_m = -50.0;
        inputs.profile.snow_bias = 0.0;
        inputs.season = SeasonalVariation {
            foliage_density_factor: 1.0,
            snow_cover: true,
        };
        let grid = classify_land_cover(inputs).unwrap();
        assert_ne!(
            grid.classes[0],
            LandCoverClass::Snow as u8,
            "elev_norm exactly 0.0 does not exceed a bias of 0.0"
        );
        assert_eq!(grid.classes[1], LandCoverClass::Snow as u8);
    }

    #[test]
    fn the_elevation_span_floor_is_one_metre() {
        // `max(h_max - h_min, 1.0)`. On a field with less than a metre of
        // relief the floor changes every normalised elevation, and no preset
        // produces terrain that flat -- a search over ten scene geometries
        // found none below 1 m, so this cannot come from the fixture.
        let heights = [100.0, 100.1, 100.2];
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("default", &heights, &[0.0; 3], &[0.0; 3], &masks);
        inputs.ground_plane_m = -50.0;
        inputs.suppress_vegetation = true;
        inputs.profile.snow_bias = 0.15;
        inputs.season = SeasonalVariation {
            foliage_density_factor: 1.0,
            snow_cover: true,
        };
        // With the floor, elev_norm spans 0.0..0.2 and only the top cell clears
        // 0.15. Without it the span is 0.2, elev_norm spans 0.0..1.0, and two
        // cells clear it.
        let grid = classify_land_cover(inputs).unwrap();
        let snowy = grid
            .classes
            .iter()
            .filter(|c| **c == LandCoverClass::Snow as u8)
            .count();
        assert_eq!(
            snowy, 1,
            "the 1 m floor must compress elev_norm; without it two cells snow"
        );
    }

    #[test]
    fn foliage_density_truncates_rather_than_rounding() {
        // numpy's uint8 cast truncates. Winter is the only season whose
        // scaling is fractional: 190 * 0.15 = 28.5, which truncates to 28 and
        // would round to 29.
        // Getting a forest cell at all takes a three-cell field. Forest needs
        // `elev_norm > 0.18`, so equal heights -- which normalise to zero
        // everywhere -- cannot grow a tree; and the HIGHEST cell normalises to
        // 1.0, which trips `elev_norm > 0.78` and comes out rocky instead.
        // Only a middle cell is forest.
        let heights = [10.0, 40.0, 60.0]; // elev_norm 0.0, 0.6, 1.0
        let masks = SampledMasks::default();
        let mut inputs = flat_inputs("jungle_canopy", &heights, &[0.0; 3], &[1.0; 3], &masks);
        inputs.ground_plane_m = -50.0;
        inputs.season = SeasonalVariation {
            foliage_density_factor: 0.15,
            snow_cover: false,
        };
        let grid = classify_land_cover(inputs).unwrap();
        assert_eq!(grid.classes[1], LandCoverClass::Forest as u8);
        assert_eq!(grid.density[1], 28, "190 * 0.15 = 28.5 truncates to 28");
    }

    #[test]
    fn the_valley_term_can_make_a_cell_wetland_on_its_own() {
        // `lowland | valley | (moist & low)`. The valley term is redundant on
        // most real terrain -- valley_mask is a snapshot of lowland_mask, and
        // where it is not, the road rule usually overwrites the difference.
        // The fixture does find two cases; this pins the requirement directly
        // so a future fixture change cannot quietly lose it.
        let heights = [50.0, 50.0];
        let masks = SampledMasks {
            valley: Some(vec![true, false]),
            ..SampledMasks::default()
        };
        let mut inputs = flat_inputs("default", &heights, &[0.0; 2], &[0.0; 2], &masks);
        inputs.ground_plane_m = -50.0;
        // Moisture has to sit in a narrow window for this to be observable at
        // all, which is most of why the fixture struggles to find such a cell:
        // above 0.42 or scrub overwrites it, below 0.68 or the moisture path
        // qualifies it anyway and the valley term proves nothing. Equal heights
        // keep elev_norm at zero, which rules out forest and rocky.
        inputs.profile.moisture = 0.5;
        let grid = classify_land_cover(inputs).unwrap();
        assert_eq!(
            grid.classes[0],
            LandCoverClass::Wetland as u8,
            "valley alone makes a flat, non-water cell wetland"
        );
        assert_ne!(grid.classes[1], LandCoverClass::Wetland as u8);
    }

    #[test]
    fn the_moisture_clamp_cannot_change_any_decision() {
        // Recorded as a PROVEN EQUIVALENCE, not a gap. `moisture` is only ever
        // compared against 0.68 (wetland), 0.42 (scrub) and the forest
        // threshold, which is `max(0.35, 0.55 - veg * 0.18)` and so lies in
        // [0.35, 0.55]. Every one of those is strictly inside (0, 1), so
        // clamping a value that was already above 1 or below 0 cannot move it
        // across any of them. The clamp is defensive, not load-bearing.
        for veg in [0.0, 0.5, 0.95, 1.5, 10.0] {
            let threshold = (0.55 - (veg * 0.18f64)).max(0.35);
            assert!(
                threshold > 0.0 && threshold < 1.0,
                "forest threshold {threshold} left (0, 1) at veg={veg}"
            );
        }
        for threshold in [0.68f64, 0.42] {
            assert!(threshold > 0.0 && threshold < 1.0);
        }
    }

    #[test]
    fn the_resampler_clamps_outside_the_source_bounds() {
        let mask = CoverMask::new(1, 3, vec![true, false, false]).unwrap();
        let sampled = sample_mask_to_grid(&mask, 0.0, 2.0, 0.0, 1.0, &[-99.0, 99.0], &[0.0, 0.0]);
        assert!(sampled[0], "far left clamps to column 0");
        assert!(!sampled[1], "far right clamps to the last column");
    }
}
