//! Parity for `procedural.py`'s grid operations, bit-for-bit over whole grids.
//!
//! Whole grids, not samples: the divergences these functions are prone to are
//! edge-handling divergences, which show up on the border and nowhere else. A
//! fixture that probed interior cells would pass a port that got every
//! boundary wrong.

use argusnet_world::grid_ops::{dilate_mask, ellipse_mask, slope_grid, smooth_heights, Grid, Mask};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    slope: Vec<SlopeCase>,
    dilate: Vec<DilateCase>,
    smooth: Vec<SmoothCase>,
    ellipse: Vec<EllipseCase>,
}

#[derive(Deserialize)]
struct SlopeCase {
    name: String,
    shape: [usize; 2],
    resolution_m: String,
    heights_m: Vec<String>,
    slope_rad: Vec<String>,
}

#[derive(Deserialize)]
struct DilateCase {
    name: String,
    shape: [usize; 2],
    iterations: i64,
    mask: Vec<bool>,
    dilated: Vec<bool>,
}

#[derive(Deserialize)]
struct SmoothCase {
    name: String,
    shape: [usize; 2],
    passes: i64,
    blend: String,
    heights_m: Vec<String>,
    preserve_mask: Vec<bool>,
    smoothed_m: Vec<String>,
}

#[derive(Deserialize)]
struct EllipseCase {
    name: String,
    xs: Vec<String>,
    ys: Vec<String>,
    center_xy: [String; 2],
    radius_x_m: String,
    radius_y_m: String,
    mask: Vec<bool>,
}

/// How far a slope may sit from the reference's.
///
/// `slope_grid` ends in `atan`, which IEEE-754 does not require to be
/// correctly rounded, so its last bit is a property of the platform's libm
/// rather than of the port. ADR-006 anticipated exactly this: *"the parity
/// fixtures for `argusnet-world` are valid for one platform at a time. If those
/// fixtures are ever run on a second platform, expect ULP-level differences and
/// widen the comparison rather than treating it as a regression."* CI moved to
/// an arm64 Mac on 2026-09-16 and `ramp_along_axis0` cell (0, 0) came back
/// 1 ULP high.
///
/// **Measured, not guessed.** Running the whole fixture against the `libm`
/// crate -- a third implementation, differing from glibc on 4.3% of `atan`
/// samples -- moves 5 cells, every one by exactly **1 ULP, 2.3e-16 rad**. This
/// bound is ~4,000x that, and eleven orders of magnitude below the ~0.2 rad a
/// port that clamped the edge index instead of taking numpy's one-sided
/// difference would produce. `the_slope_fixture_still_pins_the_edge_difference`
/// asserts that margin rather than leaving it to this comment.
const SLOPE_TOLERANCE_RAD: f64 = 1.0e-12;

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn fv(values: &[String]) -> Vec<f64> {
    values.iter().map(|v| f(v)).collect()
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/grid_ops.json"
    );
    let text = std::fs::read_to_string(path).expect("grid_ops fixture");
    serde_json::from_str(&text).expect("grid_ops fixture parses")
}

/// Compare a whole grid within `tolerance`, naming the first cell that misses.
///
/// Used only where the value passed through a transcendental. Everything else
/// -- smoothing, dilation, the ellipse masks -- stays bit-exact, and was
/// confirmed bit-identical under the third-libm probe described on
/// [`SLOPE_TOLERANCE_RAD`].
fn near_grid(actual: &Grid, expected: &[String], tolerance: f64, label: &str) {
    assert_eq!(actual.values.len(), expected.len(), "{label}: cell count");
    for (index, (value, text)) in actual.values.iter().zip(expected).enumerate() {
        let (i, j) = (index / actual.ny, index % actual.ny);
        let want = f(text);
        assert!(
            (value - want).abs() <= tolerance,
            "{label}: cell ({i}, {j}) got {value}, expected {text} \
             (off by {:e}, tolerance {tolerance:e})",
            (value - want).abs()
        );
    }
}

/// Compare a whole grid bit-for-bit, naming the first cell that diverges.
fn same_grid(actual: &Grid, expected: &[String], label: &str) {
    assert_eq!(actual.values.len(), expected.len(), "{label}: cell count");
    for (index, (value, text)) in actual.values.iter().zip(expected).enumerate() {
        let (i, j) = (index / actual.ny, index % actual.ny);
        assert_eq!(
            value.to_bits(),
            f(text).to_bits(),
            "{label}: cell ({i}, {j}) got {value}, expected {text}"
        );
    }
}

#[test]
fn slope_matches_the_reference() {
    let fixture = load();
    assert!(!fixture.slope.is_empty());
    for case in &fixture.slope {
        let heights = Grid::new(case.shape[0], case.shape[1], fv(&case.heights_m)).unwrap();
        let slope = slope_grid(&heights, f(&case.resolution_m))
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        near_grid(&slope, &case.slope_rad, SLOPE_TOLERANCE_RAD, &case.name);
    }
}

/// `np.gradient` takes a **one-sided** difference at each edge. A port that
/// clamped the index instead — reading `f[0]` twice — halves the derivative on
/// the border and is correct everywhere else, so it passes any interior-only
/// check.
#[test]
fn the_slope_fixture_still_pins_the_edge_difference() {
    let fixture = load();
    let discriminating = fixture
        .slope
        .iter()
        .filter(|case| {
            let (nx, ny) = (case.shape[0], case.shape[1]);
            let heights = fv(&case.heights_m);
            let spacing = f(&case.resolution_m).max(1.0);
            let at = |i: usize, j: usize| heights[i * ny + j];
            let expected = fv(&case.slope_rad);
            (0..nx).any(|i| {
                (0..ny).any(|j| {
                    let lo0 = at(i.saturating_sub(1), j);
                    let hi0 = at((i + 1).min(nx - 1), j);
                    let lo1 = at(i, j.saturating_sub(1));
                    let hi1 = at(i, (j + 1).min(ny - 1));
                    let d0 = (hi0 - lo0) / (2.0 * spacing);
                    let d1 = (hi1 - lo1) / (2.0 * spacing);
                    let wrong = ((d0 * d0) + (d1 * d1)).sqrt().atan();
                    // Must miss by more than the tolerance, or widening the
                    // comparison would have swallowed the boundary bug it
                    // exists to catch.
                    (wrong - expected[i * ny + j]).abs() > SLOPE_TOLERANCE_RAD
                })
            })
        })
        .count();
    assert!(
        discriminating > 0,
        "no slope case distinguishes numpy's one-sided edge difference from \
         an index-clamping port by more than SLOPE_TOLERANCE_RAD; the boundary \
         handling is untested, or the tolerance has been widened over it"
    );

    // And by how much: ADR-006 asks a guard to require a wrong port to miss by
    // a wide margin, not merely to miss.
    let worst = fixture
        .slope
        .iter()
        .flat_map(|case| {
            let (nx, ny) = (case.shape[0], case.shape[1]);
            let heights = fv(&case.heights_m);
            let spacing = f(&case.resolution_m).max(1.0);
            let expected = fv(&case.slope_rad);
            (0..nx)
                .flat_map(move |i| (0..ny).map(move |j| (i, j)))
                .map(move |(i, j)| {
                    let at = |a: usize, b: usize| heights[a * ny + b];
                    let d0 =
                        (at((i + 1).min(nx - 1), j) - at(i.saturating_sub(1), j)) / (2.0 * spacing);
                    let d1 =
                        (at(i, (j + 1).min(ny - 1)) - at(i, j.saturating_sub(1))) / (2.0 * spacing);
                    let wrong = ((d0 * d0) + (d1 * d1)).sqrt().atan();
                    (wrong - expected[i * ny + j]).abs()
                })
        })
        .fold(0.0f64, f64::max);
    assert!(
        worst > SLOPE_TOLERANCE_RAD * 1.0e6,
        "an index-clamping port misses by at most {worst:e} rad, only \
         {:.0}x the tolerance -- too close for the tolerance to be safe",
        worst / SLOPE_TOLERANCE_RAD
    );
}

#[test]
fn dilation_matches_the_reference() {
    let fixture = load();
    assert!(!fixture.dilate.is_empty());
    for case in &fixture.dilate {
        let mask = Mask::new(case.shape[0], case.shape[1], case.mask.clone()).unwrap();
        let dilated = dilate_mask(&mask, case.iterations);
        assert_eq!(dilated.values, case.dilated, "{}: dilated mask", case.name);
    }
}

#[test]
fn smoothing_matches_the_reference() {
    let fixture = load();
    assert!(!fixture.smooth.is_empty());
    for case in &fixture.smooth {
        let heights = Grid::new(case.shape[0], case.shape[1], fv(&case.heights_m)).unwrap();
        let preserve = Mask::new(case.shape[0], case.shape[1], case.preserve_mask.clone()).unwrap();
        let smoothed = smooth_heights(&heights, &preserve, case.passes, f(&case.blend))
            .unwrap_or_else(|e| panic!("{}: {e}", case.name));
        same_grid(&smoothed, &case.smoothed_m, &case.name);
    }
}

/// The kernel is `(4*centre + 4 neighbours) / 8` — a weight sum of 8 over 5
/// cells, which is **not** an average. On a constant field a true 5-cell mean
/// gives the identical answer, so this also proves the cases carry structure.
#[test]
fn the_smoothing_fixture_still_pins_the_kernel_weights() {
    let fixture = load();
    let discriminating = fixture
        .smooth
        .iter()
        .filter(|case| {
            let (nx, ny) = (case.shape[0], case.shape[1]);
            let mut smoothed = fv(&case.heights_m);
            let blend = f(&case.blend);
            for _ in 0..case.passes.max(0) {
                let source = smoothed.clone();
                let sample = |i: i64, j: i64| -> f64 {
                    let ci = i.clamp(0, nx as i64 - 1) as usize;
                    let cj = j.clamp(0, ny as i64 - 1) as usize;
                    source[ci * ny + cj]
                };
                for i in 0..nx {
                    for j in 0..ny {
                        let (si, sj) = (i as i64, j as i64);
                        // The wrong version: a true five-cell mean.
                        let mean = (source[i * ny + j]
                            + sample(si - 1, sj)
                            + sample(si + 1, sj)
                            + sample(si, sj - 1)
                            + sample(si, sj + 1))
                            / 5.0;
                        let candidate = (source[i * ny + j] * (1.0 - blend)) + (mean * blend);
                        smoothed[i * ny + j] = if case.preserve_mask[i * ny + j] {
                            source[i * ny + j]
                        } else {
                            candidate
                        };
                    }
                }
            }
            smoothed
                .iter()
                .zip(&case.smoothed_m)
                .any(|(value, text)| value.to_bits() != f(text).to_bits())
        })
        .count();
    assert!(
        discriminating > 0,
        "no smoothing case distinguishes the /8 kernel from a true 5-cell \
         mean; the weighting is untested"
    );
}

#[test]
fn ellipse_masks_match_the_reference() {
    let fixture = load();
    assert!(!fixture.ellipse.is_empty());
    for case in &fixture.ellipse {
        let mask = ellipse_mask(
            &fv(&case.xs),
            &fv(&case.ys),
            (f(&case.center_xy[0]), f(&case.center_xy[1])),
            f(&case.radius_x_m),
            f(&case.radius_y_m),
        );
        assert_eq!(mask.values, case.mask, "{}: ellipse mask", case.name);
    }
}

/// The test is `<= 1.0`. Without a sample landing exactly on the boundary,
/// `<=` and `<` are indistinguishable.
#[test]
fn the_ellipse_fixture_still_lands_on_the_boundary() {
    let fixture = load();
    let on_boundary: usize = fixture
        .ellipse
        .iter()
        .map(|case| {
            let rx = f(&case.radius_x_m).max(1.0);
            let ry = f(&case.radius_y_m).max(1.0);
            let (cx, cy) = (f(&case.center_xy[0]), f(&case.center_xy[1]));
            fv(&case.xs)
                .iter()
                .flat_map(|x| fv(&case.ys).into_iter().map(move |y| (*x, y)))
                .filter(|(x, y)| {
                    let dx = (x - cx) / rx;
                    let dy = (y - cy) / ry;
                    (dx * dx) + (dy * dy) == 1.0
                })
                .count()
        })
        .sum();
    assert!(
        on_boundary > 0,
        "no ellipse sample lands exactly on the boundary; `<= 1.0` and \
         `< 1.0` are indistinguishable"
    );
}
