//! Preset terrain profiles and the semantic mask pass.
//!
//! `_semantic_masks` is where the terrain stops being noise and becomes a
//! place: it carves rivers, fills lakes, floods a coast, flattens a compound
//! pad, then labels water, ridge, valley, steep-slope, lowland and road.
//!
//! It also **rewrites the height field while it runs**, and the order in which
//! it does so is the contract. Four things below are stale-by-design — they
//! read a value computed before an edit that came after — and reproducing that
//! staleness is the whole job.
//!
//! # `elev_norm` is computed once, from the ORIGINAL heights
//!
//! It is normalised from the height field as it arrives, and then used at the
//! very end for `ridge_mask` — after rivers have been cut, lakes filled, the
//! coast flooded, water levelled and the whole field smoothed and floored.
//! Recomputing it from the final heights is the obvious "cleanup" and it
//! changes every ridge on every preset that edits heights.
//!
//! # `lowland_mask` is built from the ORIGINAL slopes
//!
//! `slopes` is computed at the top, used for `lowland_mask`, and then
//! **recomputed** after smoothing for `steep_slope_mask`. The two are
//! different fields with the same name at different points in the function.
//!
//! # `valley_mask` is a snapshot of `lowland_mask`, not an alias
//!
//! `valley_mask = lowland_mask.copy()` happens *before* the preset branches.
//! `lake_district` and `coastal` then widen `lowland_mask` and not
//! `valley_mask`. A port that aliased them would widen both.
//!
//! # `_dilate_mask(water_mask, 0)` is a no-op that is not dead
//!
//! Zero iterations dilate nothing; the call survives only because
//! `np.asarray(..., dtype=bool)` copies. It is kept as a comment rather than
//! as code, because the copy has no observable effect here — but deleting the
//! line from the *reference* would be a behaviour change if the mask were ever
//! aliased upstream, so it is recorded rather than silently dropped.

use crate::grid_ops::{dilate_mask, ellipse_mask_grid, slope_grid, smooth_heights, Grid, Mask};
use crate::noise::stable_seed;

// `Bounds2D` lives in `noise` and is re-exported here, so callers that reach
// for `semantic::Bounds2D` keep working. It used to be defined twice, which
// made the two spellings incompatible types; see the definition for why that
// went unnoticed.
pub use crate::noise::Bounds2D;

/// A named terrain preset.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TerrainProfile {
    pub name: &'static str,
    pub base_elevation_m: f64,
    pub relief_m: f64,
    pub slope_x: f64,
    pub slope_y: f64,
    pub ridge_count: i64,
    pub valley_count: i64,
    pub moisture: f64,
    pub roughness: f64,
    pub water_bias: f64,
    pub snow_bias: f64,
    pub vegetation_bias: f64,
    pub urban_bias: f64,
    pub hydrology_strength: f64,
    pub smoothing_passes: i64,
}

include!("presets.rs");

/// The profile for a preset name, falling back to `"default"`.
///
/// The reference uses `dict.get(name, _PRESET_PROFILES["default"])`, so an
/// unknown name is not an error — it silently yields the default terrain.
pub fn terrain_profile_for_preset(preset_name: &str) -> TerrainProfile {
    PRESETS
        .iter()
        .find(|profile| profile.name == preset_name)
        .copied()
        .unwrap_or_else(|| {
            *PRESETS
                .iter()
                .find(|profile| profile.name == "default")
                .expect("the default preset exists")
        })
}

/// Every preset name, for tests and tooling.
pub fn preset_names() -> Vec<&'static str> {
    PRESETS.iter().map(|profile| profile.name).collect()
}

/// `np.median` over a slice: the middle value, or the mean of the two middle
/// values for an even count.
pub fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// `np.quantile(values, q)` with the default `method="linear"`.
///
/// The interpolation is **not** a single lerp. numpy switches form at the
/// halfway point to keep the result monotonic in `q`:
///
/// ```text
/// t <  0.5 : a + (b - a) * t
/// t >= 0.5 : b - (b - a) * (1 - t)
/// ```
///
/// Using only the first branch disagrees with numpy on about 3% of random
/// inputs. Note this is the *opposite* of `noise::value_noise`, where the
/// plain `a + t * (b - a)` form is the correct one — the two are different
/// functions in different libraries and must not be unified.
pub fn quantile(values: &[f64], q: f64) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n == 1 {
        return Some(sorted[0]);
    }
    let h = (n - 1) as f64 * q;
    let lo = h.floor();
    if lo >= (n - 1) as f64 {
        return Some(sorted[n - 1]);
    }
    let index = lo as usize;
    let t = h - lo;
    let (a, b) = (sorted[index], sorted[index + 1]);
    Some(if t < 0.5 {
        a + (b - a) * t
    } else {
        b - (b - a) * (1.0 - t)
    })
}

/// The six masks the pass produces, alongside the rewritten heights.
#[derive(Debug, Clone, PartialEq)]
pub struct SemanticMasks {
    pub water: Mask,
    pub ridge: Mask,
    pub valley: Mask,
    pub steep_slope: Mask,
    pub lowland: Mask,
    pub road: Mask,
}

/// Inputs to [`semantic_masks`].
pub struct SemanticInputs<'a> {
    pub preset_name: &'a str,
    pub profile: TerrainProfile,
    pub heights_m: &'a Grid,
    /// The x and y coordinate of every cell, i-major, as `meshgrid(indexing="ij")`
    /// produces them.
    pub xx: &'a Grid,
    pub yy: &'a Grid,
    pub bounds_xy_m: Bounds2D,
    pub resolution_m: f64,
    pub seed: i64,
    pub ground_plane_m: f64,
}

fn masked_values(grid: &Grid, mask: &Mask) -> Vec<f64> {
    grid.values
        .iter()
        .zip(&mask.values)
        .filter_map(|(value, keep)| keep.then_some(*value))
        .collect()
}

fn or_into(target: &mut Mask, other: &Mask) {
    for (slot, value) in target.values.iter_mut().zip(&other.values) {
        *slot |= *value;
    }
}

/// Carve the preset's water and terrain features, then label the result.
///
/// Returns the **rewritten** heights and the six masks. See the module docs
/// for the four stale-by-design reads this reproduces.
pub fn semantic_masks(inputs: SemanticInputs<'_>) -> Result<(Grid, SemanticMasks), &'static str> {
    let SemanticInputs {
        preset_name,
        profile,
        heights_m,
        xx,
        yy,
        bounds_xy_m,
        resolution_m,
        seed,
        ground_plane_m,
    } = inputs;

    let (nx, ny) = (heights_m.nx, heights_m.ny);
    if xx.nx != nx || xx.ny != ny || yy.nx != nx || yy.ny != ny {
        return Err("coordinate grids must match the height grid");
    }

    let scene_scale_m = bounds_xy_m.width_m().max(bounds_xy_m.height_m()).max(1.0);
    let center_x = (bounds_xy_m.x_min_m + bounds_xy_m.x_max_m) * 0.5;
    let center_y = (bounds_xy_m.y_min_m + bounds_xy_m.y_max_m) * 0.5;

    // The FIRST slope field. Used only for `lowland_mask`; a second one is
    // computed after smoothing for `steep_slope_mask`.
    let slopes = slope_grid(heights_m, resolution_m)?;

    let h_min = heights_m
        .values
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let h_max = heights_m
        .values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    // Computed from the ORIGINAL heights and never recomputed. See module docs.
    let elev_norm: Vec<f64> = heights_m
        .values
        .iter()
        .map(|h| (h - h_min) / (h_max - h_min).max(1.0))
        .collect();

    let water_level = if profile.water_bias > 0.0 {
        quantile(&heights_m.values, profile.water_bias.clamp(0.01, 0.35))
            .expect("a non-empty height grid")
    } else {
        ground_plane_m
    };

    let mut heights = heights_m.clone();

    let mut lowland_mask = Mask::new(
        nx,
        ny,
        (0..nx * ny)
            .map(|k| {
                heights.values[k] <= water_level + (profile.relief_m * 0.05)
                    && slopes.values[k] < 0.22
            })
            .collect(),
    )?;
    let mut water_mask = Mask::new(
        nx,
        ny,
        (0..nx * ny)
            .map(|k| {
                heights.values[k] <= ground_plane_m + 1.0
                    || (profile.water_bias > 0.0
                        && lowland_mask.values[k]
                        && heights.values[k] <= water_level)
            })
            .collect(),
    )?;
    // A SNAPSHOT, taken before the preset branches widen `lowland_mask`.
    let mut valley_mask = lowland_mask.clone();

    if preset_name == "river_valley" || preset_name == "mountain_pass" {
        let phase =
            (stable_seed(&["river-phase", preset_name, &seed.to_string()]) % 6283) as f64 / 1000.0;
        let amplitude_m = scene_scale_m
            * if preset_name == "mountain_pass" {
                0.015
            } else {
                0.035
            };
        let river_width = (resolution_m * 1.5).max(scene_scale_m * 0.018);
        let wavelength = (scene_scale_m * 0.23).max(1.0);

        let mut river_mask = Mask::filled(nx, ny, false);
        let mut river_bank_mask = Mask::filled(nx, ny, false);
        for k in 0..nx * ny {
            let river_y =
                center_y + amplitude_m * (((xx.values[k] - center_x) / wavelength) + phase).sin();
            let distance = (yy.values[k] - river_y).abs();
            river_mask.values[k] = distance <= river_width;
            river_bank_mask.values[k] = distance <= river_width * 3.0;
        }

        let river_level =
            quantile(&masked_values(&heights, &river_mask), 0.18).unwrap_or(water_level);
        for k in 0..nx * ny {
            if river_bank_mask.values[k] {
                heights.values[k] -= profile.relief_m * 0.035;
            }
        }
        for k in 0..nx * ny {
            if river_mask.values[k] {
                heights.values[k] = river_level;
            }
        }
        or_into(&mut water_mask, &river_mask);
        or_into(&mut valley_mask, &river_bank_mask);
    }

    if preset_name == "lake_district" {
        // Elementwise over the coordinate GRIDS. Rebuilding 1-D axes from
        // them would assume axis 0 is x, which the real caller's meshgrid
        // convention does not guarantee.
        let first = ellipse_mask_grid(
            xx,
            yy,
            (
                center_x - scene_scale_m * 0.22,
                center_y + scene_scale_m * 0.16,
            ),
            scene_scale_m * 0.09,
            scene_scale_m * 0.06,
        )?;
        let second = ellipse_mask_grid(
            xx,
            yy,
            (
                center_x + scene_scale_m * 0.24,
                center_y - scene_scale_m * 0.17,
            ),
            scene_scale_m * 0.08,
            scene_scale_m * 0.055,
        )?;
        let mut lakes = first;
        or_into(&mut lakes, &second);

        let lake_level = quantile(&masked_values(&heights, &lakes), 0.28).unwrap_or(water_level);
        // The shore clamp runs BEFORE the lake fill, matching the reference.
        // The order happens not to matter -- `dilate_mask(lakes, 1)` includes
        // the lake cells themselves, and `min(lake_level, lake_level + 1.5)`
        // is `lake_level`, so a lake cell ends at the same height either way.
        // Kept in the reference's order anyway: the equivalence depends on the
        // clearance being positive, and would break if it ever went negative.
        let shore = dilate_mask(&lakes, 1);
        for k in 0..nx * ny {
            if shore.values[k] {
                heights.values[k] = heights.values[k].min(lake_level + 1.5);
            }
        }
        for k in 0..nx * ny {
            if lakes.values[k] {
                heights.values[k] = lake_level;
            }
        }
        or_into(&mut water_mask, &lakes);
        // Widens `lowland_mask` only; `valley_mask` was snapshotted earlier.
        let wide = dilate_mask(&lakes, 2);
        or_into(&mut lowland_mask, &wide);
    }

    if preset_name == "coastal" {
        let coast_limit = bounds_xy_m.y_min_m + bounds_xy_m.height_m() * 0.26;
        let shelf_limit = bounds_xy_m.y_min_m + bounds_xy_m.height_m() * 0.34;
        let coast_level = ground_plane_m.max(water_level);

        let mut coast_mask = Mask::filled(nx, ny, false);
        let mut shelf = Mask::filled(nx, ny, false);
        for k in 0..nx * ny {
            coast_mask.values[k] = yy.values[k] <= coast_limit;
            shelf.values[k] = yy.values[k] <= shelf_limit;
        }
        for k in 0..nx * ny {
            if shelf.values[k] {
                heights.values[k] = heights.values[k].min(coast_level + 2.0);
            }
        }
        for k in 0..nx * ny {
            if coast_mask.values[k] {
                heights.values[k] = coast_level;
            }
        }
        or_into(&mut water_mask, &coast_mask);
        or_into(&mut lowland_mask, &shelf);
    }

    if preset_name == "urban_flat" || preset_name == "military_compound" {
        let (rx, ry) = if preset_name == "urban_flat" {
            (0.18, 0.15)
        } else {
            (0.20, 0.18)
        };
        let pad = ellipse_mask_grid(
            xx,
            yy,
            (center_x, center_y),
            scene_scale_m * rx,
            scene_scale_m * ry,
        )?;
        let inside = masked_values(&heights, &pad);
        // `np.median(heights[pad]) if np.any(pad) else np.median(heights)` --
        // an empty pad falls back to the median of the WHOLE field, not to
        // zero or to the water level.
        let pad_level = if inside.is_empty() {
            median(&heights.values)
        } else {
            median(&inside)
        };
        let apron = dilate_mask(&pad, 2);
        for k in 0..nx * ny {
            if apron.values[k] {
                heights.values[k] = (heights.values[k] * 0.35) + (pad_level * 0.65);
            }
        }
    }

    // `_dilate_mask(water_mask, 0)` in the reference: zero iterations, so a
    // no-op beyond the bool copy. Left out as code and recorded here.
    if water_mask.values.iter().any(|v| *v) {
        for k in 0..nx * ny {
            if water_mask.values[k] {
                heights.values[k] = heights.values[k].min(water_level);
            }
        }
    }
    heights = smooth_heights(&heights, &water_mask, profile.smoothing_passes, 0.28)?;
    for value in heights.values.iter_mut() {
        *value = value.max(ground_plane_m);
    }

    // The SECOND slope field, from the finished heights.
    let slopes = slope_grid(&heights, resolution_m)?;
    let steep_threshold = if profile.roughness < 0.7 { 0.52 } else { 0.62 };

    let mut steep_slope_mask = Mask::filled(nx, ny, false);
    let mut ridge_mask = Mask::filled(nx, ny, false);
    for (k, norm) in elev_norm.iter().enumerate() {
        let steep = slopes.values[k] > steep_threshold;
        steep_slope_mask.values[k] = steep;
        // `norm` here is the ORIGINAL-height normalisation. See module docs.
        ridge_mask.values[k] = *norm > 0.72 || (steep && *norm > 0.55);
    }

    let mut road_mask = Mask::filled(nx, ny, false);
    if preset_name == "urban_flat" || preset_name == "military_compound" {
        let road_width = (resolution_m * 1.2).max(scene_scale_m * 0.012);
        for k in 0..nx * ny {
            road_mask.values[k] = (xx.values[k] - center_x).abs() <= road_width
                || (yy.values[k] - center_y).abs() <= road_width;
        }
    } else if preset_name == "river_valley" || preset_name == "lake_district" {
        for k in 0..nx * ny {
            road_mask.values[k] =
                valley_mask.values[k] && !water_mask.values[k] && slopes.values[k] < 0.16;
        }
    }

    Ok((
        heights,
        SemanticMasks {
            water: water_mask,
            ridge: ridge_mask,
            valley: valley_mask,
            steep_slope: steep_slope_mask,
            lowland: lowland_mask,
            road: road_mask,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_preset_is_reachable_by_name() {
        for name in preset_names() {
            assert_eq!(terrain_profile_for_preset(name).name, name);
        }
        assert_eq!(preset_names().len(), 12);
    }

    #[test]
    fn an_unknown_preset_falls_back_to_default_rather_than_failing() {
        assert_eq!(terrain_profile_for_preset("no_such_preset").name, "default");
        assert_eq!(terrain_profile_for_preset("").name, "default");
    }

    #[test]
    fn median_handles_both_parities() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), 2.0);
        assert_eq!(median(&[4.0, 1.0, 3.0, 2.0]), 2.5);
        assert_eq!(median(&[7.0]), 7.0);
    }

    #[test]
    fn quantile_switches_lerp_form_at_the_halfway_point() {
        // Two samples, so t == q exactly. Below 0.5 numpy interpolates from
        // the low end; at or above it, back from the high end.
        let values = [0.0, 1.0];
        assert_eq!(quantile(&values, 0.25), Some(0.25));
        assert_eq!(quantile(&values, 0.75), Some(0.75));
        assert_eq!(quantile(&values, 0.0), Some(0.0));
        assert_eq!(quantile(&values, 1.0), Some(1.0));
    }

    #[test]
    fn quantile_of_an_empty_slice_is_none() {
        assert_eq!(quantile(&[], 0.5), None);
    }

    #[test]
    fn quantile_of_one_value_is_that_value() {
        assert_eq!(quantile(&[42.0], 0.9), Some(42.0));
    }

    fn ramp_inputs(nx: usize, ny: usize) -> (Grid, Grid, Grid, Bounds2D) {
        let mut heights = Grid::filled(nx, ny, 0.0);
        let mut xx = Grid::filled(nx, ny, 0.0);
        let mut yy = Grid::filled(nx, ny, 0.0);
        for i in 0..nx {
            for j in 0..ny {
                xx.set(i, j, i as f64 * 10.0);
                yy.set(i, j, j as f64 * 10.0);
                heights.set(i, j, 20.0 + (i as f64) * 2.0 + (j as f64) * 0.5);
            }
        }
        let bounds = Bounds2D {
            x_min_m: 0.0,
            x_max_m: (nx - 1) as f64 * 10.0,
            y_min_m: 0.0,
            y_max_m: (ny - 1) as f64 * 10.0,
        };
        (heights, xx, yy, bounds)
    }

    fn run(preset: &str) -> (Grid, SemanticMasks) {
        let (heights, xx, yy, bounds) = ramp_inputs(12, 12);
        semantic_masks(SemanticInputs {
            preset_name: preset,
            profile: terrain_profile_for_preset(preset),
            heights_m: &heights,
            xx: &xx,
            yy: &yy,
            bounds_xy_m: bounds,
            resolution_m: 10.0,
            seed: 1234,
            ground_plane_m: 0.0,
        })
        .unwrap()
    }

    #[test]
    fn a_mismatched_coordinate_grid_is_rejected() {
        let (heights, xx, _, bounds) = ramp_inputs(6, 6);
        let wrong = Grid::filled(5, 6, 0.0);
        assert!(semantic_masks(SemanticInputs {
            preset_name: "default",
            profile: terrain_profile_for_preset("default"),
            heights_m: &heights,
            xx: &xx,
            yy: &wrong,
            bounds_xy_m: bounds,
            resolution_m: 10.0,
            seed: 1,
            ground_plane_m: 0.0,
        })
        .is_err());
    }

    #[test]
    fn heights_never_fall_below_the_ground_plane() {
        let (heights, xx, yy, bounds) = ramp_inputs(10, 10);
        let (result, _) = semantic_masks(SemanticInputs {
            preset_name: "coastal",
            profile: terrain_profile_for_preset("coastal"),
            heights_m: &heights,
            xx: &xx,
            yy: &yy,
            bounds_xy_m: bounds,
            resolution_m: 10.0,
            seed: 7,
            ground_plane_m: 5.0,
        })
        .unwrap();
        assert!(result.values.iter().all(|v| *v >= 5.0));
    }

    #[test]
    fn lake_district_widens_lowland_but_not_valley() {
        // `valley_mask` is snapshotted before the lake branch runs, so the
        // dilated lake shore reaches lowland only.
        let (_, masks) = run("lake_district");
        let lowland_only = masks
            .lowland
            .values
            .iter()
            .zip(&masks.valley.values)
            .filter(|(low, valley)| **low && !**valley)
            .count();
        assert!(
            lowland_only > 0,
            "the lake branch must widen lowland beyond the valley snapshot"
        );
    }

    #[test]
    fn only_the_named_presets_produce_roads() {
        for preset in ["urban_flat", "military_compound"] {
            assert!(
                run(preset).1.road.count() > 0,
                "{preset} draws a road cross"
            );
        }
        for preset in ["alpine", "arctic_tundra", "default"] {
            assert_eq!(run(preset).1.road.count(), 0, "{preset} has no roads");
        }
    }

    #[test]
    fn the_compound_pad_flattens_the_centre() {
        let (heights, _) = run("military_compound");
        let (original, _, _, _) = ramp_inputs(12, 12);
        // The apron is blended 35/65 toward the pad median, so the centre
        // spread shrinks.
        let centre_spread = |grid: &Grid| {
            let values: Vec<f64> = (4..8)
                .flat_map(|i| (4..8).map(move |j| (i, j)))
                .map(|(i, j)| grid.at(i, j))
                .collect();
            values.iter().copied().fold(f64::NEG_INFINITY, f64::max)
                - values.iter().copied().fold(f64::INFINITY, f64::min)
        };
        assert!(centre_spread(&heights) < centre_spread(&original));
    }

    #[test]
    fn an_unknown_preset_takes_no_feature_branch() {
        let (_, masks) = run("no_such_preset");
        assert_eq!(masks.road.count(), 0);
        // Valley and lowland stay identical: nothing widened either.
        assert_eq!(masks.valley.values, masks.lowland.values);
    }
}
