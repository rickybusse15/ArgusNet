//! `procedural_terrain_grid`: the top of the terrain builder.
//!
//! Composes the pieces already ported — the noise primitives (P4.3a), the grid
//! operations (P4.3b) and the semantic pass (P4.3c) — over the numpy RNG
//! stream (`numpy_rng`), and produces the height field and masks that
//! `build_terrain_layer` hands to everything downstream.
//!
//! # Axis 0 is Y here
//!
//! The reference builds its coordinates with `np.meshgrid(x_values, y_values)`
//! and **no `indexing="ij"`**, so numpy's default applies and axis 0 is y. That
//! makes every grid in this module `(row = y, col = x)` — the transpose of what
//! `grid_ops` and `semantic` name their axes.
//!
//! `semantic_masks` is layout-agnostic for the masks it returns but **not** for
//! the heights it rewrites: `smooth_heights` sums its four neighbours in a
//! fixed positional order, and floating-point addition is not associative, so
//! the two layouts differ by up to 7e-15. P4.3c pinned both layouts in its
//! fixture precisely so this caller could rely on the xy one.
//!
//! # The RNG draw ORDER is the contract
//!
//! One generator serves the whole function, so every draw shifts the ones after
//! it. Reordering two draws that are used in the same expression still changes
//! the terrain. The valley loop is the sharpest case: its first iteration draws
//! **one** uniform for the river and pass presets and **two** for everything
//! else, so the preset decides the stream alignment of every later draw.

use crate::grid_ops::{Grid, Mask};
use crate::noise::{fbm, grid_values, line_gaussian, stable_seed, FbmConfig, LineEnvelope};
use crate::numpy_rng::NumpyRng;
use crate::semantic::{
    quantile, semantic_masks, terrain_profile_for_preset, Bounds2D, SemanticInputs, SemanticMasks,
    TerrainProfile,
};

/// Everything `procedural_terrain_grid` returns.
pub struct ProceduralTerrainGrid {
    pub x_values_m: Vec<f64>,
    pub y_values_m: Vec<f64>,
    /// `(row = y, col = x)`. See the module docs.
    pub heights_m: Grid,
    pub masks: SemanticMasks,
    pub profile: TerrainProfile,
}

pub struct TerrainGridRequest<'a> {
    pub bounds_xy_m: Bounds2D,
    pub preset_name: &'a str,
    pub seed: i64,
    pub resolution_m: f64,
    pub detail_strength: f64,
    pub ground_plane_m: f64,
}

/// A ridge or valley's randomly drawn geometry.
///
/// Held as a struct so the draw order is stated once, in one place, rather than
/// implied by the order of arguments at a call site.
struct LineDraw {
    angle_rad: f64,
    offset_m: f64,
    width_m: f64,
    length_m: f64,
    amplitude_m: f64,
}

pub fn procedural_terrain_grid(
    request: TerrainGridRequest<'_>,
) -> Result<ProceduralTerrainGrid, &'static str> {
    let TerrainGridRequest {
        bounds_xy_m,
        preset_name,
        seed,
        resolution_m,
        detail_strength,
        ground_plane_m,
    } = request;

    let profile = terrain_profile_for_preset(preset_name);
    // `_stable_seed` is masked to 31 bits, so this cannot truncate.
    let mut rng = NumpyRng::new(stable_seed(&["profile", preset_name, &seed.to_string()]) as u32);

    let (x_values, y_values) = grid_values(bounds_xy_m, resolution_m);
    let (rows, cols) = (y_values.len(), x_values.len());
    let cells = rows * cols;

    // meshgrid's DEFAULT layout: row index walks y, column index walks x.
    let mut xx = Grid::filled(rows, cols, 0.0);
    let mut yy = Grid::filled(rows, cols, 0.0);
    for (row, &y) in y_values.iter().enumerate() {
        for (col, &x) in x_values.iter().enumerate() {
            xx.set(row, col, x);
            yy.set(row, col, y);
        }
    }

    let scene_scale_m = bounds_xy_m.width_m().max(bounds_xy_m.height_m()).max(1.0);
    let center_xy = (
        (bounds_xy_m.x_min_m + bounds_xy_m.x_max_m) * 0.5,
        (bounds_xy_m.y_min_m + bounds_xy_m.y_max_m) * 0.5,
    );

    // --- domain warp -------------------------------------------------------
    // The warp is applied ONCE and every later field reads the warped
    // coordinates, not the original ones. `slope` uses them too, which is what
    // stops the base gradient being a perfect plane.
    let warp_wavelength_m = scene_scale_m * 0.42;
    let warp_strength_m = scene_scale_m * 0.035 * profile.roughness;
    let warp_x_config = FbmConfig::new(
        stable_seed(&["warp-x", &seed.to_string(), preset_name]),
        warp_wavelength_m,
        3,
    );
    let warp_y_config = FbmConfig::new(
        stable_seed(&["warp-y", &seed.to_string(), preset_name]),
        warp_wavelength_m,
        3,
    );
    let mut wx = vec![0.0f64; cells];
    let mut wy = vec![0.0f64; cells];
    for k in 0..cells {
        let (x, y) = (xx.values[k], yy.values[k]);
        wx[k] = x + fbm(x, y, warp_x_config) * warp_strength_m;
        wy[k] = y + fbm(x, y, warp_y_config) * warp_strength_m;
    }

    // --- base gradient -----------------------------------------------------
    // Every term below is accumulated into its OWN array and the five are
    // combined once at the end, in the reference's order. Folding them into a
    // running total as they are computed is the obvious way to write this and
    // is wrong by 1 ULP: floating-point addition is not associative, and the
    // reference's `base + slope + macro + ridges - valleys + detail` fixes both
    // the grouping and the position of `base_elevation_m` at the FRONT.
    let mut slope = vec![0.0f64; cells];
    for k in 0..cells {
        slope[k] =
            (wx[k] - center_xy.0) * profile.slope_x + (wy[k] - center_xy.1) * profile.slope_y;
    }

    // --- macro undulation --------------------------------------------------
    // TWO draws, and the sine's phase is drawn BEFORE the cosine's. Swapping
    // them exchanges the two phases and shifts nothing else, which is a wrong
    // terrain that still looks entirely plausible.
    let macro_phase_x = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
    let macro_phase_y = rng.uniform(-std::f64::consts::PI, std::f64::consts::PI);
    let macro_x_scale = (scene_scale_m * 0.42f64).max(1.0);
    let macro_y_scale = (scene_scale_m * 0.37f64).max(1.0);
    let mut macro_term = vec![0.0f64; cells];
    for k in 0..cells {
        macro_term[k] = ((wx[k] / macro_x_scale) + macro_phase_x).sin()
            * ((wy[k] / macro_y_scale) + macro_phase_y).cos()
            * profile.relief_m
            * 0.10;
    }

    // --- ridges ------------------------------------------------------------
    // Five draws per ridge, in this order. Each ridge is modulated by its own
    // fbm "crenulation" keyed on the ridge index, so the index is part of the
    // seed and not just a loop counter.
    let mut ridges = vec![0.0f64; cells];
    for index in 0..profile.ridge_count {
        let draw = LineDraw {
            angle_rad: rng.uniform(-std::f64::consts::PI, std::f64::consts::PI),
            offset_m: rng.uniform(-0.36, 0.36) * scene_scale_m,
            width_m: scene_scale_m * rng.uniform(0.045, 0.13),
            length_m: scene_scale_m * rng.uniform(0.36, 0.85),
            amplitude_m: profile.relief_m * rng.uniform(0.20, 0.58),
        };
        let envelope = LineEnvelope {
            angle_rad: draw.angle_rad,
            offset_m: draw.offset_m,
            width_m: draw.width_m,
            length_m: draw.length_m,
            center_xy,
        };
        let crenulation_config = FbmConfig::new(
            stable_seed(&["ridge", &seed.to_string(), preset_name, &index.to_string()]),
            (draw.width_m * 1.8).max(10.0),
            3,
        );
        for k in 0..cells {
            let ridge = line_gaussian(wx[k], wy[k], envelope);
            let crenulation = 1.0 + 0.18 * fbm(wx[k], wy[k], crenulation_config);
            ridges[k] += draw.amplitude_m * ridge * crenulation;
        }
    }

    // --- valleys -----------------------------------------------------------
    // SUBTRACTED, not added, and the draw count of the first iteration depends
    // on the preset: the river and pass presets take ONE draw for a
    // near-axis-aligned angle and hardcode the offset to zero, where every
    // other case takes TWO. That one missing draw realigns the entire rest of
    // the stream.
    // Accumulated POSITIVELY and subtracted once at the end -- the reference
    // builds `valleys` as a sum and then writes `- valleys`. Subtracting each
    // valley as it is computed groups the arithmetic differently.
    let mut valleys = vec![0.0f64; cells];
    let axis_aligned_first = matches!(preset_name, "river_valley" | "mountain_pass");
    for index in 0..profile.valley_count {
        let (angle_rad, offset_m) = if axis_aligned_first && index == 0 {
            (rng.uniform(-0.10, 0.10), 0.0)
        } else {
            (
                rng.uniform(-std::f64::consts::PI, std::f64::consts::PI),
                rng.uniform(-0.30, 0.30) * scene_scale_m,
            )
        };
        let draw = LineDraw {
            angle_rad,
            offset_m,
            width_m: scene_scale_m * rng.uniform(0.035, 0.10),
            length_m: scene_scale_m * rng.uniform(0.45, 1.05),
            amplitude_m: profile.relief_m * rng.uniform(0.12, 0.36),
        };
        let envelope = LineEnvelope {
            angle_rad: draw.angle_rad,
            offset_m: draw.offset_m,
            width_m: draw.width_m,
            length_m: draw.length_m,
            center_xy,
        };
        for k in 0..cells {
            valleys[k] += draw.amplitude_m * line_gaussian(wx[k], wy[k], envelope);
        }
    }

    // --- detail ------------------------------------------------------------
    // Only the FINE octave is scaled by roughness; the broad one is not.
    let broad_config = FbmConfig::new(
        stable_seed(&["broad", &seed.to_string(), preset_name]),
        scene_scale_m * 0.30,
        5,
    );
    let fine_config = FbmConfig::new(
        stable_seed(&["fine", &seed.to_string(), preset_name]),
        scene_scale_m * 0.075,
        4,
    );
    let mut detail = vec![0.0f64; cells];
    for k in 0..cells {
        let broad = fbm(wx[k], wy[k], broad_config);
        let fine = fbm(wx[k], wy[k], fine_config);
        detail[k] = (broad * profile.relief_m * 0.12
            + fine * profile.relief_m * 0.045 * profile.roughness)
            * detail_strength;
    }

    // `base + slope + macro + ridges - valleys + detail`, left to right. The
    // order and the grouping are both load-bearing; see the note above.
    let mut heights = vec![0.0f64; cells];
    for k in 0..cells {
        heights[k] = profile.base_elevation_m + slope[k] + macro_term[k] + ridges[k] - valleys[k]
            + detail[k];
    }

    // --- lowland deepening -------------------------------------------------
    // A THIRD clamp on water_bias, `[0.01, 0.35]` — the same range
    // `_semantic_masks` uses and a different one from the `[0.03, 0.30]` in
    // `build_land_cover_layer`. Skipped entirely when the bias is zero, which
    // is not the same as clamping zero up to 0.01.
    if profile.water_bias > 0.0 {
        let water_level = quantile(&heights, profile.water_bias.clamp(0.01, 0.35))
            .ok_or("an empty height grid has no water level")?;
        let depth_scale = (profile.relief_m * 0.18f64).max(1.0);
        for h in heights.iter_mut() {
            let lowland = ((water_level - *h) / depth_scale).clamp(0.0, 1.0);
            *h -= lowland * profile.relief_m * 0.08;
        }
    }

    for h in heights.iter_mut() {
        *h = h.max(ground_plane_m);
    }

    let (heights_m, masks) = semantic_masks(SemanticInputs {
        preset_name,
        profile,
        heights_m: &Grid::new(rows, cols, heights)?,
        xx: &xx,
        yy: &yy,
        bounds_xy_m,
        resolution_m,
        seed,
        ground_plane_m,
    })?;

    Ok(ProceduralTerrainGrid {
        x_values_m: x_values,
        y_values_m: y_values,
        heights_m,
        masks,
        profile,
    })
}

/// `mask_coverage_fraction` from the reference's metadata: the mean of each
/// mask as a float. Recorded in scene metadata, so it is part of the contract.
pub fn mask_coverage_fraction(mask: &Mask) -> f64 {
    if mask.values.is_empty() {
        return f64::NAN;
    }
    let set = mask.values.iter().filter(|v| **v).count();
    set as f64 / mask.values.len() as f64
}

#[cfg(test)]
mod tests {
    #[test]
    fn the_final_sum_is_not_reassociable() {
        // `base + slope + macro + ridges - valleys + detail`, left to right.
        //
        // This is pinned HERE rather than in the parity fixture because the
        // parity test cannot see it. Heights are compared to 1e-9 m -- numpy's
        // `exp` makes anything tighter impossible (see the parity test) -- and
        // a reassociation moves them by about one ULP, roughly 1e-14 m. Two
        // mutation-test variants, "base added last" and "detail before
        // valleys", survive the parity suite for exactly that reason. They are
        // NOT equivalents; they are below the floor the tolerance imposes, and
        // this test is what keeps the requirement stated.
        //
        // These magnitudes were SEARCHED FOR, not chosen. The first attempt
        // used a plausible-looking alpine cell -- large base, small detail --
        // on the reasoning that small terms lose bits against large ones. All
        // three orderings agreed exactly, so the test asserted something false.
        // Reassociation shows up where the terms very nearly CANCEL: here the
        // six sum to 4.47 from parts as large as 28, and the orderings differ
        // in the last bit.
        let base = 28.0f64;
        let slope = -12.623442820099424;
        let macro_term = -14.229796665702498;
        let ridges = 10.601301427053153;
        let valleys = 12.339272964077374;
        let detail = 5.058021745920502;

        let reference = base + slope + macro_term + ridges - valleys + detail;
        let base_last = slope + macro_term + ridges - valleys + detail + base;
        let detail_before_valleys = base + slope + macro_term + ridges + detail - valleys;

        assert_ne!(
            reference.to_bits(),
            base_last.to_bits(),
            "moving base_elevation_m to the end of the sum must change the result"
        );
        assert_ne!(
            reference.to_bits(),
            detail_before_valleys.to_bits(),
            "swapping the detail and valley terms must change the result"
        );
        // ... and by about one ULP, which is what puts it under the parity
        // tolerance. If this ever grew, the tolerance would start catching it
        // and this test could go.
        assert!((reference - base_last).abs() < 1.0e-9);
    }

    #[test]
    fn the_water_bias_clamp_bounds_are_unreachable_from_the_preset_table() {
        // `water_bias.clamp(0.01, 0.35)`. Recorded as PROVEN UNREACHABLE rather
        // than left as a mutation survivor.
        //
        // `procedural_terrain_grid` takes a preset NAME and looks the profile
        // up itself, so unlike `build_land_cover_layer` there is no way for a
        // caller to drive the bias directly. Across the whole table the biases
        // are 0.0, 0.01, 0.02, 0.03, 0.06, 0.08, 0.16, 0.2 and 0.28: none
        // exceeds 0.30, let alone 0.35, and the two zeroes skip the block
        // entirely rather than clamping up to 0.01. Both bounds are therefore
        // no-ops on every reachable input.
        use crate::semantic::terrain_profile_for_preset;
        for preset in [
            "alpine",
            "arctic_tundra",
            "coastal",
            "default",
            "desert_canyon",
            "jungle_canopy",
            "lake_district",
            "military_compound",
            "mountain_pass",
            "river_valley",
            "rolling_highlands",
            "urban_flat",
        ] {
            let bias = terrain_profile_for_preset(preset).water_bias;
            assert!(
                bias == 0.0 || (0.01..=0.30).contains(&bias),
                "{preset} has water_bias {bias}, which would make the clamp observable"
            );
        }
    }

    #[test]
    fn the_lowland_depth_scale_floor_is_unreachable_from_the_preset_table() {
        // `max(relief_m * 0.18, 1.0)`. The floor bites below a relief of
        // 5.56 m. The two presets with the smallest relief -- urban_flat at 12
        // and military_compound at 24 -- both have `water_bias == 0` and so
        // never reach this line at all; every preset that does reach it has a
        // relief of at least 28 m, giving 5.04. Another proven no-op.
        use crate::semantic::terrain_profile_for_preset;
        for preset in [
            "alpine",
            "arctic_tundra",
            "coastal",
            "default",
            "desert_canyon",
            "jungle_canopy",
            "lake_district",
            "military_compound",
            "mountain_pass",
            "river_valley",
            "rolling_highlands",
            "urban_flat",
        ] {
            let profile = terrain_profile_for_preset(preset);
            if profile.water_bias > 0.0 {
                assert!(
                    profile.relief_m * 0.18 > 1.0,
                    "{preset} reaches the depth scale with relief {} m, where the \
                     1.0 floor would bite",
                    profile.relief_m
                );
            }
        }
    }
}
