//! Parity for `_semantic_masks` and the preset table.
//!
//! Every preset is run, so every feature branch is exercised, and the whole
//! rewritten height field plus all six masks are compared bit-for-bit.
//!
//! The function edits the height field as it goes, and four reads are stale by
//! design — they use a value computed before an edit that came after. The
//! guards at the bottom implement each one the "clean" way and require it to
//! disagree, so the staleness cannot quietly stop being tested.

use argusnet_world::grid_ops::{slope_grid, Grid};
use argusnet_world::semantic::{
    is_steep_slope, median, quantile, semantic_masks, steep_slope_threshold,
    terrain_profile_for_preset, Bounds2D, SemanticInputs, TerrainProfile,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    presets: Vec<PresetCase>,
    fallbacks: Vec<FallbackCase>,
    quantiles: Vec<QuantileCase>,
    medians: Vec<MedianCase>,
    runs: Vec<RunCase>,
}

#[derive(Deserialize)]
struct QuantileCase {
    values: Vec<String>,
    q: String,
    result: String,
}

#[derive(Deserialize)]
struct MedianCase {
    values: Vec<String>,
    result: String,
}

#[derive(Deserialize)]
struct PresetCase {
    #[serde(default)]
    lookup_name: String,
    name: String,
    base_elevation_m: String,
    relief_m: String,
    slope_x: String,
    slope_y: String,
    ridge_count: i64,
    valley_count: i64,
    moisture: String,
    roughness: String,
    water_bias: String,
    snow_bias: String,
    vegetation_bias: String,
    urban_bias: String,
    hydrology_strength: String,
    smoothing_passes: i64,
}

#[derive(Deserialize)]
struct FallbackCase {
    lookup_name: String,
    resolved_name: String,
}

#[derive(Deserialize)]
struct RunCase {
    label: String,
    scene: String,
    /// `"ij"` (axis 0 is x) or `"xy"` (axis 0 is y — numpy's default, and what
    /// `procedural_terrain_grid` actually passes).
    layout: String,
    preset_name: String,
    profile: PresetCase,
    shape: [usize; 2],
    resolution_m: String,
    seed: i64,
    ground_plane_m: String,
    bounds: BoundsSpec,
    heights_in_m: Vec<String>,
    /// `xx` and `yy` are NOT stored: they are exactly `i * resolution` and
    /// `j * resolution`, and storing them doubled the fixture. The capture
    /// script asserts that identity before omitting them.
    heights_out_m: Vec<String>,
    masks: MaskSpec,
}

#[derive(Deserialize)]
struct BoundsSpec {
    x_min_m: String,
    x_max_m: String,
    y_min_m: String,
    y_max_m: String,
}

#[derive(Deserialize)]
struct MaskSpec {
    water: Vec<bool>,
    ridge: Vec<bool>,
    valley: Vec<bool>,
    steep_slope: Vec<bool>,
    lowland: Vec<bool>,
    road: Vec<bool>,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn fv(values: &[String]) -> Vec<f64> {
    values.iter().map(|v| f(v)).collect()
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/semantic.json"
    );
    let text = std::fs::read_to_string(path).expect("semantic fixture");
    serde_json::from_str(&text).expect("semantic fixture parses")
}

/// The 12x15 preset table, field by field. A transcription typo in 180 numbers
/// survives review; it does not survive this.
#[test]
fn the_preset_table_matches_the_reference() {
    let fixture = load();
    assert_eq!(fixture.presets.len(), 12, "preset count");
    for case in &fixture.presets {
        let profile = terrain_profile_for_preset(&case.lookup_name);
        let at = &case.lookup_name;
        assert_eq!(profile.name, case.name, "{at}: name");
        assert_eq!(
            profile.base_elevation_m.to_bits(),
            f(&case.base_elevation_m).to_bits(),
            "{at}: base_elevation_m"
        );
        assert_eq!(
            profile.relief_m.to_bits(),
            f(&case.relief_m).to_bits(),
            "{at}: relief_m"
        );
        assert_eq!(
            profile.slope_x.to_bits(),
            f(&case.slope_x).to_bits(),
            "{at}: slope_x"
        );
        assert_eq!(
            profile.slope_y.to_bits(),
            f(&case.slope_y).to_bits(),
            "{at}: slope_y"
        );
        assert_eq!(profile.ridge_count, case.ridge_count, "{at}: ridge_count");
        assert_eq!(
            profile.valley_count, case.valley_count,
            "{at}: valley_count"
        );
        assert_eq!(
            profile.moisture.to_bits(),
            f(&case.moisture).to_bits(),
            "{at}: moisture"
        );
        assert_eq!(
            profile.roughness.to_bits(),
            f(&case.roughness).to_bits(),
            "{at}: roughness"
        );
        assert_eq!(
            profile.water_bias.to_bits(),
            f(&case.water_bias).to_bits(),
            "{at}: water_bias"
        );
        assert_eq!(
            profile.snow_bias.to_bits(),
            f(&case.snow_bias).to_bits(),
            "{at}: snow_bias"
        );
        assert_eq!(
            profile.vegetation_bias.to_bits(),
            f(&case.vegetation_bias).to_bits(),
            "{at}: vegetation_bias"
        );
        assert_eq!(
            profile.urban_bias.to_bits(),
            f(&case.urban_bias).to_bits(),
            "{at}: urban_bias"
        );
        assert_eq!(
            profile.hydrology_strength.to_bits(),
            f(&case.hydrology_strength).to_bits(),
            "{at}: hydrology_strength"
        );
        assert_eq!(
            profile.smoothing_passes, case.smoothing_passes,
            "{at}: smoothing_passes"
        );
    }
}

#[test]
fn unknown_preset_names_fall_back_the_way_the_reference_does() {
    let fixture = load();
    for case in &fixture.fallbacks {
        assert_eq!(
            terrain_profile_for_preset(&case.lookup_name).name,
            case.resolved_name,
            "lookup {:?}",
            case.lookup_name
        );
    }
}

/// Real values, not a finiteness check.
///
/// An earlier version of this test asserted only that the results were finite,
/// which is not an assertion: all three `quantile` mutants — dropping either
/// branch of numpy's two-form lerp, and moving its switch point — survived it.
#[test]
fn quantile_matches_the_reference() {
    let fixture = load();
    assert!(fixture.quantiles.len() >= 80, "quantile case count");
    for case in &fixture.quantiles {
        let values = fv(&case.values);
        let actual = quantile(&values, f(&case.q)).expect("non-empty");
        assert_eq!(
            actual.to_bits(),
            f(&case.result).to_bits(),
            "quantile(n={}, q={}): got {actual}, expected {}",
            values.len(),
            case.q,
            case.result
        );
    }
}

#[test]
fn median_matches_the_reference() {
    let fixture = load();
    assert!(!fixture.medians.is_empty());
    for case in &fixture.medians {
        let values = fv(&case.values);
        let actual = median(&values);
        assert_eq!(
            actual.to_bits(),
            f(&case.result).to_bits(),
            "median(n={}): got {actual}, expected {}",
            values.len(),
            case.result
        );
    }
}

/// numpy switches lerp form at `t = 0.5`. Without a case on each side of that
/// switch, either single-branch spelling passes.
#[test]
fn the_quantile_fixture_still_straddles_the_lerp_switch() {
    let fixture = load();
    let mut below = 0usize;
    let mut at_or_above = 0usize;
    for case in &fixture.quantiles {
        let n = case.values.len();
        if n < 2 {
            continue;
        }
        let h = (n - 1) as f64 * f(&case.q);
        let t = h - h.floor();
        if h.floor() >= (n - 1) as f64 {
            continue;
        }
        if t > 0.0 && t < 0.5 {
            below += 1;
        } else if t >= 0.5 {
            at_or_above += 1;
        }
    }
    assert!(
        below > 0 && at_or_above > 0,
        "quantile cases must straddle numpy's t = 0.5 lerp switch \
         (below: {below}, at or above: {at_or_above})"
    );
}

/// Rebuild a profile from the fixture rather than looking it up, so runs with
/// overridden fields — a water bias past either clamp, a roughness on the
/// steep-slope branch boundary — drive the port with exactly what the
/// reference was given.
fn profile_of(spec: &PresetCase) -> TerrainProfile {
    // `name` is a `&'static str` on the Rust side; the runs only ever use
    // names from the preset table, so resolve through it and override the
    // numeric fields.
    let base = terrain_profile_for_preset(&spec.name);
    TerrainProfile {
        name: base.name,
        base_elevation_m: f(&spec.base_elevation_m),
        relief_m: f(&spec.relief_m),
        slope_x: f(&spec.slope_x),
        slope_y: f(&spec.slope_y),
        ridge_count: spec.ridge_count,
        valley_count: spec.valley_count,
        moisture: f(&spec.moisture),
        roughness: f(&spec.roughness),
        water_bias: f(&spec.water_bias),
        snow_bias: f(&spec.snow_bias),
        vegetation_bias: f(&spec.vegetation_bias),
        urban_bias: f(&spec.urban_bias),
        hydrology_strength: f(&spec.hydrology_strength),
        smoothing_passes: spec.smoothing_passes,
    }
}

fn run_case(case: &RunCase) -> (Grid, argusnet_world::semantic::SemanticMasks) {
    let (nx, ny) = (case.shape[0], case.shape[1]);
    let heights = Grid::new(nx, ny, fv(&case.heights_in_m)).unwrap();
    let resolution = f(&case.resolution_m);
    let mut xx = Grid::filled(nx, ny, 0.0);
    let mut yy = Grid::filled(nx, ny, 0.0);
    let along_x_is_axis0 = case.layout == "ij";
    for i in 0..nx {
        for j in 0..ny {
            let (x, y) = if along_x_is_axis0 {
                (i as f64, j as f64)
            } else {
                (j as f64, i as f64)
            };
            xx.set(i, j, x * resolution);
            yy.set(i, j, y * resolution);
        }
    }
    semantic_masks(SemanticInputs {
        preset_name: &case.preset_name,
        profile: profile_of(&case.profile),
        heights_m: &heights,
        xx: &xx,
        yy: &yy,
        bounds_xy_m: Bounds2D {
            x_min_m: f(&case.bounds.x_min_m),
            x_max_m: f(&case.bounds.x_max_m),
            y_min_m: f(&case.bounds.y_min_m),
            y_max_m: f(&case.bounds.y_max_m),
        },
        resolution_m: resolution,
        seed: case.seed,
        ground_plane_m: f(&case.ground_plane_m),
    })
    .unwrap_or_else(|e| panic!("{}: {e}", case.label))
}

#[test]
fn semantic_masks_match_the_reference_for_every_preset() {
    let fixture = load();
    assert!(
        fixture.runs.len() >= 40,
        "every preset across several scenes, plus overrides"
    );

    let mut total_unresolvable = 0usize;
    for case in &fixture.runs {
        let (heights, masks) = run_case(case);
        let ny = case.shape[1];
        let at = &case.label;

        assert_eq!(
            heights.values.len(),
            case.heights_out_m.len(),
            "{at}: height count"
        );
        for (index, (value, text)) in heights.values.iter().zip(&case.heights_out_m).enumerate() {
            assert_eq!(
                value.to_bits(),
                f(text).to_bits(),
                "{at}: height at ({}, {}) got {value}, expected {text}",
                index / ny,
                index % ny
            );
        }

        // Cells whose slope sits within one ULP of the steep-slope threshold
        // cannot be classified the same way by two different `atan`
        // implementations, because the value is *on* the boundary. They are
        // excluded here and pinned exactly, without any libm, by
        // `the_steep_slope_comparison_is_strict` below.
        let unresolvable = unresolvable_slope_cells(case, &heights);
        // The only scenes allowed to contain a boundary cell are the ones
        // built to contain nothing else. Anywhere else, a cell on the
        // threshold would be a silent hole in an otherwise exact comparison.
        assert!(
            unresolvable.is_empty() || case.label.starts_with("exact_slope/"),
            "{at}: {} cell(s) sit on the steep-slope threshold outside the \
             scenes constructed for it, so the mask comparison there is no \
             longer exact",
            unresolvable.len()
        );
        total_unresolvable += unresolvable.len();

        for (name, actual, expected) in [
            ("water", &masks.water.values, &case.masks.water),
            ("ridge", &masks.ridge.values, &case.masks.ridge),
            ("valley", &masks.valley.values, &case.masks.valley),
            (
                "steep_slope",
                &masks.steep_slope.values,
                &case.masks.steep_slope,
            ),
            ("lowland", &masks.lowland.values, &case.masks.lowland),
            ("road", &masks.road.values, &case.masks.road),
        ] {
            // Only the two masks derived from the slope threshold can be moved
            // by a boundary cell; the other four are exact everywhere.
            let slope_derived = name == "steep_slope" || name == "ridge";
            let first_bad = actual
                .iter()
                .zip(expected)
                .enumerate()
                .filter(|(index, _)| !(slope_derived && unresolvable.contains(index)))
                .find(|(_, (a, b))| a != b)
                .map(|(index, _)| (index / ny, index % ny));
            assert!(
                first_bad.is_none(),
                "{at}: {name} mask differs first at {:?}",
                first_bad.unwrap()
            );
        }
    }

    // Both `exact_slope` scenes are pure ramps, so every one of their 6x5 = 30
    // cells has the same derivative and therefore sits on the threshold. That
    // is the whole content of those scenes; the assertion inside the loop is
    // what keeps the exclusion from reaching any other.
    //
    // Measured against the `libm` crate -- a third implementation, differing
    // from glibc on 4.3% of `atan` samples -- 5 of those 60 flip, and no cell
    // of any other mask, preset or scene does.
    assert_eq!(
        total_unresolvable, 60,
        "the two exact_slope ramps should contribute 30 boundary cells each"
    );
}

/// `elev_norm` is normalised from the **original** heights and used at the end
/// for `ridge_mask`. Recomputing it from the finished heights is the obvious
/// cleanup, and it changes the ridges.
#[test]
fn the_fixture_still_pins_the_stale_elevation_normalisation() {
    let fixture = load();
    let discriminating = fixture
        .runs
        .iter()
        .filter(|case| {
            let final_heights = fv(&case.heights_out_m);
            let h_min = final_heights.iter().copied().fold(f64::INFINITY, f64::min);
            let h_max = final_heights
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let span = (h_max - h_min).max(1.0);
            final_heights
                .iter()
                .zip(&case.masks.steep_slope)
                .zip(&case.masks.ridge)
                .any(|((height, steep), expected)| {
                    let fresh = (height - h_min) / span;
                    let wrong = fresh > 0.72 || (*steep && fresh > 0.55);
                    wrong != *expected
                })
        })
        .count();
    assert!(
        discriminating > 0,
        "no run distinguishes elev_norm taken from the original heights from \
         one recomputed at the end; the staleness is untested"
    );
}

/// `valley_mask` is a snapshot of `lowland_mask` taken before the preset
/// branches widen the latter. If they were equal everywhere, an alias would
/// pass.
#[test]
fn the_fixture_still_pins_the_valley_snapshot() {
    let fixture = load();
    let differing = fixture
        .runs
        .iter()
        .filter(|case| case.masks.valley != case.masks.lowland)
        .count();
    assert!(
        differing > 0,
        "valley_mask equals lowland_mask in every run; the pre-branch snapshot \
         is indistinguishable from an alias"
    );
}

/// `procedural_terrain_grid` builds its coordinate grids with numpy's DEFAULT
/// meshgrid, so axis 0 is **y** — the transpose of the `ij` runs.
///
/// The masks are layout-agnostic, but the heights are not: `smooth_heights`
/// sums its four neighbours in a fixed positional order, transposition
/// permutes that order, and floating-point addition is not associative. The
/// two layouts differ by ~7e-15 — nothing physically, everything to a fixture
/// that compares bit patterns. Without `xy` runs the port is only pinned
/// against a layout production never uses.
#[test]
fn the_fixture_covers_the_layout_the_real_caller_uses() {
    let fixture = load();
    let by_label: std::collections::HashMap<&str, &RunCase> = fixture
        .runs
        .iter()
        .map(|run| (run.label.as_str(), run))
        .collect();

    let xy_runs: Vec<&&RunCase> = by_label.values().filter(|run| run.layout == "xy").collect();
    assert!(
        !xy_runs.is_empty(),
        "no run uses the default (axis 0 = y) meshgrid layout that \
         procedural_terrain_grid actually passes"
    );

    // Each must genuinely differ from its ij twin, or it is redundant.
    let mut differing = 0usize;
    for run in &xy_runs {
        let twin_label = format!("base/{}", &run.label["xy/".len()..]);
        let Some(twin) = by_label.get(twin_label.as_str()) else {
            continue;
        };
        let (rows, cols) = (run.shape[0], run.shape[1]);
        let mine = fv(&run.heights_out_m);
        let theirs = fv(&twin.heights_out_m);
        let transposed_differs = (0..rows).any(|a| {
            (0..cols).any(|b| mine[a * cols + b].to_bits() != theirs[b * rows + a].to_bits())
        });
        if transposed_differs {
            differing += 1;
        }
    }
    assert!(
        differing > 0,
        "every xy-layout run is a bit-exact transpose of its ij twin, so the \
         layout sensitivity of smooth_heights is untested"
    );
}

/// Every feature branch must actually change the terrain, or running that
/// preset proves nothing beyond the default path.
#[test]
fn every_preset_branch_changes_the_terrain() {
    let fixture = load();
    let fallback = fixture
        .runs
        .iter()
        .find(|case| case.label == "base/no_such_preset")
        .expect("a fallback run");
    for preset in [
        "river_valley",
        "mountain_pass",
        "lake_district",
        "coastal",
        "urban_flat",
        "military_compound",
    ] {
        let case = fixture
            .runs
            .iter()
            .find(|case| case.scene == "base" && case.preset_name == preset)
            .unwrap_or_else(|| panic!("a run for {preset}"));
        assert_ne!(
            case.heights_out_m, fallback.heights_out_m,
            "{preset}'s branch changed nothing versus the fallback, so it is untested"
        );
    }
}

/// Flat indices whose finished slope is within one ULP of the profile's
/// steep-slope threshold.
///
/// One ULP is the whole budget: `atan` is not required by IEEE-754 to be
/// correctly rounded, and the measured divergence between three separate libm
/// implementations is exactly one ULP. A wider window would start excusing real
/// disagreement.
fn unresolvable_slope_cells(case: &RunCase, heights: &Grid) -> std::collections::HashSet<usize> {
    let profile = profile_of(&case.profile);
    let threshold = steep_slope_threshold(&profile);
    let resolution = f(&case.resolution_m);
    let Ok(slopes) = slope_grid(heights, resolution) else {
        return std::collections::HashSet::new();
    };
    let above = f64::from_bits(threshold.to_bits() + 1);
    let below = f64::from_bits(threshold.to_bits() - 1);
    slopes
        .values
        .iter()
        .enumerate()
        .filter(|(_, slope)| **slope >= below && **slope <= above)
        .map(|(index, _)| index)
        .collect()
}

/// The property the `exact_slope` scenes were constructed to prove, asserted
/// where no libm can reach it.
///
/// `semantic_masks` classifies with a strict `>`, so a slope landing exactly on
/// the threshold is **not** steep. The fixture proves this by building a ramp
/// whose `atan` lands bit-exactly on 0.52 -- which works only on the platform
/// the fixture was captured on. Here the boundary value is supplied directly.
#[test]
fn the_steep_slope_comparison_is_strict() {
    for threshold in [0.52_f64, 0.62_f64] {
        assert!(
            !is_steep_slope(threshold, threshold),
            "a slope exactly on {threshold} must not be steep"
        );
        assert!(
            is_steep_slope(f64::from_bits(threshold.to_bits() + 1), threshold),
            "one ULP above {threshold} must be steep"
        );
        assert!(!is_steep_slope(
            f64::from_bits(threshold.to_bits() - 1),
            threshold
        ));
    }
}

/// And that the roughness split picking the threshold is strict the same way.
#[test]
fn the_roughness_split_is_strict() {
    let mut profile = terrain_profile_for_preset("default");
    profile.roughness = 0.699_999_999;
    assert_eq!(steep_slope_threshold(&profile), 0.52);
    profile.roughness = 0.7;
    assert_eq!(
        steep_slope_threshold(&profile),
        0.62,
        "`< 0.7`, not `<= 0.7`"
    );
}
