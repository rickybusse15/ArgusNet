//! Parity for `build_land_cover_layer`'s classification pass.
//!
//! Covers everything the reference does before the obstacle and patch polygon
//! overlay, which is P4.3e. The fixture was captured with `obstacles=()` and
//! `patches=()`, so the layer the reference returned IS this pass's output and
//! the comparison is direct.
//!
//! # The test recomputes the seams, it does not take them as given
//!
//! `moisture_noise` and the sampled masks are recorded, but they are also
//! **recomputed here** — the noise through `noise::fbm` and `noise::stable_seed`
//! from P4.3a, the masks through `sample_mask_to_grid`. Comparing those against
//! what was recorded before classifying means a failure says which stage broke,
//! and it means the P4.3a integration is exercised rather than assumed. P4.3c's
//! real bug lived exactly at a seam like this: a fixture that drove a function
//! correctly while never exercising how production calls it.
//!
//! # Bit-for-bit, and whole grids
//!
//! Classes and densities are `u8`, so "close" is not a thing; every cell of
//! every case is compared. The heights, slopes and profile are compared by bit
//! pattern after parsing.

use argusnet_world::land_cover_build::{
    classify_land_cover, sample_mask_to_grid, CoverGrid, CoverMask, LandCoverInputs, SampledMasks,
    SeasonalVariation,
};
use argusnet_world::noise::{fbm, stable_seed, FbmConfig};
use argusnet_world::semantic::{terrain_profile_for_preset, TerrainProfile};
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Deserialize)]
struct Fixture {
    mask_names: Vec<String>,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    preset: String,
    seed: i64,
    profile: Profile,
    rows: usize,
    cols: usize,
    suppress_vegetation: bool,
    ground_plane_m: String,
    season: Season,
    cover_bounds: Bounds,
    source_bounds: Bounds,
    cover_resolution_m: String,
    heights_m: Vec<String>,
    slopes_rad: Vec<String>,
    moisture_noise: Vec<String>,
    source_masks: BTreeMap<String, SourceMask>,
    sampled_masks: BTreeMap<String, String>,
    classes: Vec<u8>,
    density: Vec<u8>,
}

#[derive(Deserialize)]
struct Profile {
    name: String,
    moisture: String,
    water_bias: String,
    snow_bias: String,
    vegetation_bias: String,
}

#[derive(Deserialize)]
struct Season {
    foliage_density_factor: String,
    snow_cover: bool,
}

#[derive(Deserialize)]
struct Bounds {
    x_min_m: String,
    y_min_m: String,
    width_m: String,
    height_m: String,
}

#[derive(Deserialize)]
struct SourceMask {
    rows: usize,
    cols: usize,
    values: String,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn fv(texts: &[String]) -> Vec<f64> {
    texts.iter().map(|t| f(t)).collect()
}

/// Masks are stored a character a cell rather than as JSON booleans: as arrays
/// this fixture was 469 KB, of which 320 KB was the words `true` and `false`.
fn unbits(text: &str) -> Vec<bool> {
    text.bytes().map(|b| b == b'1').collect()
}

fn load() -> Fixture {
    let path = format!(
        "{}/../../tests/fixtures/land_cover.json",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{path}: {e}"));
    serde_json::from_str(&text).expect("land cover fixture parses")
}

/// The cover grid's cell centres: `x_min + (i + 0.5) * resolution`.
///
/// Not stored in the fixture — they are exactly reconstructible, and P4.3c's
/// `xx`/`yy` arrays were most of a 2 MB file before they were dropped for the
/// same reason.
fn cover_coordinates(case: &Case) -> (Vec<f64>, Vec<f64>) {
    let res = f(&case.cover_resolution_m);
    let x_min = f(&case.cover_bounds.x_min_m);
    let y_min = f(&case.cover_bounds.y_min_m);
    let mut xx = Vec::with_capacity(case.rows * case.cols);
    let mut yy = Vec::with_capacity(case.rows * case.cols);
    // Row-major over (y, x): axis 0 is Y here. See `land_cover_build`'s module
    // docs — this is the transpose of `grid_ops` and `semantic`.
    for row in 0..case.rows {
        for col in 0..case.cols {
            xx.push(x_min + ((col as f64 + 0.5) * res));
            yy.push(y_min + ((row as f64 + 0.5) * res));
        }
    }
    (xx, yy)
}

fn profile_for(case: &Case) -> TerrainProfile {
    // Start from the table so the `&'static str` name and the fields this pass
    // does not read are right, then take the recorded values — some cases
    // OVERRIDE the profile, because no preset can reach both water-bias clamps.
    let mut profile = terrain_profile_for_preset(&case.preset);
    assert_eq!(
        profile.name, case.profile.name,
        "{}: preset table disagrees with the fixture on the profile name",
        case.name
    );
    profile.moisture = f(&case.profile.moisture);
    profile.water_bias = f(&case.profile.water_bias);
    profile.snow_bias = f(&case.profile.snow_bias);
    profile.vegetation_bias = f(&case.profile.vegetation_bias);
    profile
}

fn recompute_moisture_noise(case: &Case, xx: &[f64], yy: &[f64]) -> Vec<f64> {
    let seed = stable_seed(&["land-cover", &case.preset, &case.seed.to_string()]);
    let width = f(&case.cover_bounds.width_m);
    let height = f(&case.cover_bounds.height_m);
    let config = FbmConfig::new(seed, width.max(height) * 0.22, 4);
    xx.iter()
        .zip(yy)
        .map(|(x, y)| fbm(*x, *y, config))
        .collect()
}

fn recompute_sampled(case: &Case, xx: &[f64], yy: &[f64]) -> BTreeMap<String, Vec<bool>> {
    case.source_masks
        .iter()
        .map(|(name, source)| {
            let mask = CoverMask::new(source.rows, source.cols, unbits(&source.values))
                .expect("source mask shape");
            let sampled = sample_mask_to_grid(
                &mask,
                f(&case.source_bounds.x_min_m),
                f(&case.source_bounds.width_m),
                f(&case.source_bounds.y_min_m),
                f(&case.source_bounds.height_m),
                xx,
                yy,
            );
            (name.clone(), sampled)
        })
        .collect()
}

fn classify(case: &Case, sampled: &BTreeMap<String, Vec<bool>>, noise: &[f64]) -> CoverGrid {
    let pick = |name: &str| sampled.get(name).cloned();
    let masks = SampledMasks {
        water: pick("water"),
        lowland: pick("lowland"),
        valley: pick("valley"),
        steep_slope: pick("steep_slope"),
        ridge: pick("ridge"),
        road: pick("road"),
    };
    let heights = fv(&case.heights_m);
    let slopes = fv(&case.slopes_rad);
    classify_land_cover(LandCoverInputs {
        rows: case.rows,
        cols: case.cols,
        profile: profile_for(case),
        terrain_preset: &case.preset,
        season: SeasonalVariation {
            foliage_density_factor: f(&case.season.foliage_density_factor),
            snow_cover: case.season.snow_cover,
        },
        suppress_vegetation: case.suppress_vegetation,
        ground_plane_m: f(&case.ground_plane_m),
        heights_m: &heights,
        slopes_rad: &slopes,
        moisture_noise: noise,
        masks: &masks,
    })
    .expect("classification succeeds")
}

#[test]
fn the_fbm_seam_reproduces_the_recorded_moisture_noise() {
    let fixture = load();
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        let actual = recompute_moisture_noise(case, &xx, &yy);
        let expected = fv(&case.moisture_noise);
        assert_eq!(actual.len(), expected.len(), "{}: length", case.name);
        for (index, (got, want)) in actual.iter().zip(&expected).enumerate() {
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "{}: moisture noise cell {index}: got {got}, expected {want}",
                case.name
            );
        }
    }
}

#[test]
fn the_resampler_reproduces_the_recorded_sampled_masks() {
    let fixture = load();
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        let actual = recompute_sampled(case, &xx, &yy);
        assert_eq!(
            actual.len(),
            case.sampled_masks.len(),
            "{}: mask count",
            case.name
        );
        for (name, sampled) in &actual {
            let expected = unbits(&case.sampled_masks[name]);
            assert_eq!(
                *sampled, expected,
                "{}: sampled mask {name} differs from the reference",
                case.name
            );
        }
    }
}

#[test]
fn every_case_reproduces_the_reference_classification() {
    let fixture = load();
    assert!(fixture.cases.len() >= 11, "case count");
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        let noise = recompute_moisture_noise(case, &xx, &yy);
        let sampled = recompute_sampled(case, &xx, &yy);
        let grid = classify(case, &sampled, &noise);
        assert_eq!(
            (grid.rows, grid.cols),
            (case.rows, case.cols),
            "{}",
            case.name
        );
        for (index, (got, want)) in grid.classes.iter().zip(&case.classes).enumerate() {
            assert_eq!(
                got,
                want,
                "{}: class at cell {index} (row {}, col {})",
                case.name,
                index / case.cols,
                index % case.cols
            );
        }
        for (index, (got, want)) in grid.density.iter().zip(&case.density).enumerate() {
            assert_eq!(
                got,
                want,
                "{}: density at cell {index} (row {}, col {})",
                case.name,
                index / case.cols,
                index % case.cols
            );
        }
    }
}

#[test]
fn the_fixture_names_every_mask_the_classifier_reads() {
    // A mask the reference emits but `SampledMasks` has no field for would be
    // silently dropped, and every cell it governs would just fall through to a
    // later rule — a plausible map, quietly missing a layer.
    let fixture = load();
    for name in &fixture.mask_names {
        assert!(
            matches!(
                name.as_str(),
                "water" | "lowland" | "valley" | "steep_slope" | "ridge" | "road"
            ),
            "fixture carries mask {name:?}, which SampledMasks has no field for"
        );
    }
    for case in &fixture.cases {
        for name in case.source_masks.keys() {
            assert!(
                fixture.mask_names.contains(name),
                "{}: source mask {name:?} is not in mask_names",
                case.name
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Discrimination guards.
//
// Each of these implements a wrong version and requires it to disagree with
// what was recorded. Asserting a proxy instead — "is a tie present", "are the
// masks non-uniform" — has passed while catching nothing three times in this
// migration, so none of them stops at a property.
// ---------------------------------------------------------------------------

#[test]
fn reading_the_source_mask_transposed_changes_the_answer() {
    // `grid_ops` and `semantic` are documented (i = x, j = y). The masks that
    // reach this pass are the transpose, because `procedural_terrain_grid` uses
    // numpy's default meshgrid. Reading them the other way round produces a
    // self-consistent, rotated map — the bug that cost 0.013 m in P4.3c.
    let fixture = load();
    let mut disagreed = 0;
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        for (name, source) in &case.source_masks {
            let values = unbits(&source.values);
            // Transpose the source, then read it with the same code.
            let mut flipped = vec![false; values.len()];
            for row in 0..source.rows {
                for col in 0..source.cols {
                    flipped[col * source.rows + row] = values[row * source.cols + col];
                }
            }
            let wrong = CoverMask::new(source.cols, source.rows, flipped).expect("shape");
            let sampled = sample_mask_to_grid(
                &wrong,
                f(&case.source_bounds.x_min_m),
                f(&case.source_bounds.width_m),
                f(&case.source_bounds.y_min_m),
                f(&case.source_bounds.height_m),
                &xx,
                &yy,
            );
            if sampled != unbits(&case.sampled_masks[name]) {
                disagreed += 1;
            }
        }
    }
    assert!(
        disagreed > 0,
        "no mask in any case distinguishes the two layouts — the fixture cannot \
         catch a transposed port"
    );
}

#[test]
fn the_class_assignment_order_is_observable() {
    // Running snow before wetland leaves both predicates correct and changes
    // the map, because `snow_mask` excludes only water. If no case shows it,
    // the order is not actually pinned by this fixture.
    let fixture = load();
    let mut disagreed = 0;
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        let noise = recompute_moisture_noise(case, &xx, &yy);
        let sampled = recompute_sampled(case, &xx, &yy);
        let grid = classify(case, &sampled, &noise);
        // Wetland is 5 and snow is 7. A cell that came out snow while the
        // wetland predicate also held is a cell the order decided.
        let snow_over_wetland = grid
            .classes
            .iter()
            .zip(&grid.density)
            .any(|(class, density)| *class == 7 && *density == 40);
        if snow_over_wetland {
            disagreed += 1;
        }
    }
    assert!(
        disagreed > 0,
        "no case puts snow on a cell at all — the wetland/snow order is untested"
    );
}

#[test]
fn a_road_lowers_density_and_nothing_else_does() {
    // Road is the only rule in the pass that reduces a density. Every other
    // one either assigns a positive value or takes a max. A case where a road
    // lands on foliage is what makes `density = 0` distinguishable from
    // `max(density, 0)`.
    let fixture = load();
    let mut roads_over_foliage = 0;
    for case in &fixture.cases {
        let (xx, yy) = cover_coordinates(case);
        let noise = recompute_moisture_noise(case, &xx, &yy);
        let sampled = recompute_sampled(case, &xx, &yy);
        let grid = classify(case, &sampled, &noise);
        // Rebuild what the cell would have been with the road rule removed.
        let road = sampled.get("road").cloned().unwrap_or_default();
        for (index, is_road) in road.iter().enumerate() {
            if *is_road && grid.classes[index] == 8 {
                // Density must be zero wherever a road won.
                assert_eq!(
                    grid.density[index], 0,
                    "{}: road at cell {index} kept density {}",
                    case.name, grid.density[index]
                );
                roads_over_foliage += 1;
            }
        }
    }
    assert!(
        roads_over_foliage > 0,
        "no case has a road at all — the one density-lowering rule is untested"
    );
}

#[test]
fn the_resampler_is_not_an_identity_map() {
    // If every source mask were the same shape as the cover grid, the resampler
    // would be an index-for-index copy and every rounding question in it would
    // be moot. At least one case must actually resample.
    let fixture = load();
    let resampling = fixture.cases.iter().any(|case| {
        case.source_masks
            .values()
            .any(|m| m.rows != case.rows || m.cols != case.cols)
    });
    assert!(
        resampling,
        "every source mask matches its cover grid — the resampler never resamples"
    );
}
