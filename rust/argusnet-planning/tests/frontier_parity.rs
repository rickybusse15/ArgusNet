//! Parity for the frontier planner.
//!
//! Fixture from `scripts/capture_frontier_fixture.py`, which drives the
//! reference and reads its results back.
//!
//! Two comparisons are exact and one is not, deliberately:
//!
//! * **Chosen cells and gap cells are compared exactly.** They are integers,
//!   and the choice is what the caller acts on.
//! * **The coverage gradient is compared at `1e-12`.** The reference computes
//!   it with `uniform_filter(mask, size=3) * 9`, two separable `1/3` passes,
//!   which lands near the neighbour count rather than on it — the fixture
//!   holds `0.9999999999999991` where the count is 1 and `-8.3e-16` where it
//!   is 0. The port counts directly. That can only change an answer by
//!   flipping a tie, so `the_recorded_margins_clear_the_filter_noise` checks
//!   every case's winning margin against the noise rather than assuming it is
//!   harmless.

use argusnet_planning::frontier::{
    find_gap_cells, select_frontier_cell, ClaimedCells, CoverageView, FrontierConfig, GridBounds,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    config: ConfigSpec,
    scenes: Vec<SceneSpec>,
    gradients: Vec<GradientSpec>,
    selections: Vec<SelectionCase>,
}

#[derive(Deserialize)]
struct ConfigSpec {
    distance_weight: String,
    coverage_gradient_weight: String,
    exclusion_radius_cells: i64,
    gap_fill_min_fraction: String,
}

#[derive(Deserialize)]
struct SceneSpec {
    name: String,
    nx: usize,
    ny: usize,
    grid: Vec<Vec<u32>>,
    coverage_fraction: String,
    covered_cells: usize,
    total_cells: usize,
    gap_cells: Vec<Vec<usize>>,
}

#[derive(Deserialize)]
struct GradientSpec {
    scene: String,
    gradient_weight: String,
    map: Vec<Vec<String>>,
}

#[derive(Deserialize)]
struct SelectionCase {
    scene: String,
    note: String,
    drone_xy: [String; 2],
    drone_id: String,
    claims: Vec<(String, Vec<usize>)>,
    expected: Option<Vec<usize>>,
    score_margin: String,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/frontier.json"
    );
    serde_json::from_str(&std::fs::read_to_string(path).expect("fixture readable"))
        .expect("fixture parses")
}

fn config_of(spec: &ConfigSpec) -> FrontierConfig {
    FrontierConfig {
        distance_weight: f(&spec.distance_weight),
        coverage_gradient_weight: f(&spec.coverage_gradient_weight),
        exclusion_radius_cells: spec.exclusion_radius_cells,
        gap_fill_min_fraction: f(&spec.gap_fill_min_fraction),
    }
}

/// Flattened row-major, matching the reference's `(nx, ny)` array.
fn counts_of(spec: &SceneSpec) -> Vec<u32> {
    spec.grid.iter().flatten().copied().collect()
}

fn bounds_of(spec: &SceneSpec) -> GridBounds {
    GridBounds {
        x_min_m: 0.0,
        y_min_m: 0.0,
        resolution_m: 1.0,
        nx: spec.nx,
        ny: spec.ny,
    }
}

fn scene_named<'a>(fixture: &'a Fixture, name: &str) -> &'a SceneSpec {
    fixture
        .scenes
        .iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("scene {name} missing"))
}

#[test]
fn the_grids_round_trip_and_agree_on_coverage() {
    for spec in load().scenes.iter() {
        let counts = counts_of(spec);
        assert_eq!(counts.len(), spec.nx * spec.ny, "{}: shape", spec.name);
        let view = CoverageView {
            counts: &counts,
            bounds: bounds_of(spec),
        };
        assert_eq!(
            counts.iter().filter(|c| **c > 0).count(),
            spec.covered_cells,
            "{}: covered cells",
            spec.name
        );
        assert_eq!(counts.len(), spec.total_cells, "{}: total cells", spec.name);
        let expected = f(&spec.coverage_fraction);
        assert!(
            (view.coverage_fraction() - expected).abs() <= 1.0e-12,
            "{}: coverage fraction {} != {expected}",
            spec.name,
            view.coverage_fraction()
        );
    }
}

#[test]
fn gap_cells_match_python() {
    for spec in load().scenes.iter() {
        let counts = counts_of(spec);
        let view = CoverageView {
            counts: &counts,
            bounds: bounds_of(spec),
        };
        let actual: Vec<Vec<usize>> = find_gap_cells(&view)
            .into_iter()
            .map(|(i, j)| vec![i, j])
            .collect();
        assert_eq!(actual, spec.gap_cells, "gap cells in {}", spec.name);
    }
}

#[test]
fn selected_cells_match_python() {
    let fixture = load();
    let config = config_of(&fixture.config);
    for case in &fixture.selections {
        let spec = scene_named(&fixture, &case.scene);
        let counts = counts_of(spec);
        let view = CoverageView {
            counts: &counts,
            bounds: bounds_of(spec),
        };
        let mut claimed = ClaimedCells::new();
        for (id, cell) in &case.claims {
            claimed.claim(id, (cell[0], cell[1]));
        }
        let actual = select_frontier_cell(
            &view,
            &config,
            [f(&case.drone_xy[0]), f(&case.drone_xy[1])],
            &claimed,
            &case.drone_id,
        );
        let actual = actual.map(|(i, j)| vec![i, j]);
        assert_eq!(actual, case.expected, "selection[{}]", case.note);
    }
}

/// The gradient the port computes, against what `uniform_filter` produced.
///
/// Compared at `1e-12`, not exactly: the reference's separable `1/3` passes
/// do not land on the integer they represent. Checked only where the ramp is
/// on, since the reference skips the filter entirely below 25% coverage.
#[test]
fn the_coverage_gradient_matches_python_within_the_filter_noise() {
    let fixture = load();
    for gradient in &fixture.gradients {
        if f(&gradient.gradient_weight) <= 0.0 {
            continue;
        }
        let spec = scene_named(&fixture, &gradient.scene);
        let counts = counts_of(spec);
        let view = CoverageView {
            counts: &counts,
            bounds: bounds_of(spec),
        };
        for i in 0..spec.nx {
            for j in 0..spec.ny {
                let expected = f(&gradient.map[i][j]);
                let actual = argusnet_planning::frontier::neighbour_count_for_test(&view, i, j);
                assert!(
                    (actual - expected).abs() <= 1.0e-12,
                    "{} gradient at ({i}, {j}): {actual} != {expected}",
                    gradient.scene
                );
            }
        }
    }
}

/// The filter's noise cannot flip any recorded decision.
///
/// The port's integer neighbour count differs from the reference's filtered
/// one by at most a few ULPs. That matters only if it changes which candidate
/// wins, so every case's winning margin is checked against a bound far above
/// the observed noise. A margin of exactly zero is a genuine tie, which both
/// implementations resolve the same way — by candidate order.
#[test]
fn the_recorded_margins_clear_the_filter_noise() {
    let fixture = load();
    let weight = f(&fixture.config.coverage_gradient_weight);
    // The largest deviation seen in the fixture is ~8.3e-16 per cell; allow
    // three orders of headroom, doubled for a difference between two scores.
    let noise_bound = 2.0 * weight * 1.0e-12;
    let mut ties = 0;
    let mut clear = 0;
    for case in &fixture.selections {
        let margin = f(&case.score_margin);
        if margin == 0.0 {
            ties += 1;
            continue;
        }
        assert!(
            margin > noise_bound,
            "selection[{}] wins by only {margin}, within the filter noise \\
             ({noise_bound}); an integer neighbour count could flip it",
            case.note
        );
        clear += 1;
    }
    assert!(clear >= 8, "only {clear} cases have a measurable margin");
    assert!(
        ties >= 2,
        "only {ties} tie(s); the first-index tie-break is barely exercised"
    );
}

/// Ties go to the lowest `(i, j)`, which is what `np.argmax` does over
/// row-major candidates. Every other scene has a clear winner, so without a
/// genuine tie a port keeping the LAST maximum would pass everything.
#[test]
fn a_genuine_tie_goes_to_the_first_candidate_in_row_major_order() {
    let fixture = load();
    let case = fixture
        .selections
        .iter()
        .find(|c| c.note.contains("genuine four-way tie"))
        .expect("the tie case must exist");
    assert_eq!(f(&case.score_margin), 0.0, "the case must be an exact tie");
    let chosen = case.expected.as_ref().expect("a tie still selects a cell");
    // The four orthogonal neighbours of the covered centre tie; row-major
    // order visits (0, 1) before (1, 0), (1, 2) and (2, 1).
    assert_eq!(
        chosen,
        &vec![0, 1],
        "the first tied candidate in row-major order must win"
    );
}

/// A platform's own claim must not exclude it, and another's must.
#[test]
fn the_fixture_covers_both_sides_of_the_claim_exclusion() {
    let fixture = load();
    let own = fixture
        .selections
        .iter()
        .find(|c| c.note.contains("own claim"))
        .expect("the self-claim case must exist");
    assert!(
        own.claims.iter().any(|(id, _)| id == &own.drone_id),
        "the case must claim a cell for the selecting drone"
    );
    let fallback = fixture
        .selections
        .iter()
        .find(|c| c.note.contains("fallback"))
        .expect("the all-excluded case must exist");
    assert!(
        fallback.claims.len() >= 3,
        "the fallback needs enough claims to exclude every candidate"
    );
    assert!(
        fallback.expected.is_some(),
        "the fallback must still return a cell rather than giving up"
    );
}

/// An uncovered region touching the border is open frontier, not a hole.
/// Without a scene like that, a port that reported every uncovered cell as a
/// gap would pass — and the gap-fill gate would never let the mission advance.
#[test]
fn the_fixture_distinguishes_an_edge_notch_from_an_enclosed_hole() {
    let fixture = load();
    let notch = scene_named(&fixture, "edge_notch_5x5");
    let hole = scene_named(&fixture, "single_hole_5x5");
    assert_eq!(notch.covered_cells, hole.covered_cells, "same coverage");
    assert!(notch.gap_cells.is_empty(), "an edge notch is not a hole");
    assert_eq!(hole.gap_cells.len(), 1, "an enclosed cell is");
}

/// The gradient ramp itself, compared against the reference's own weight.
///
/// The selection tests only see the ramp through which cell wins, and a
/// wrongly-scaled weight often picks the same cell anyway — widening the ramp
/// from 0.50 to 1.0, or dropping the clamp at the top, changed no selection.
/// This compares the weight directly, so any of those is a failure.
#[test]
fn the_gradient_ramp_matches_python() {
    let fixture = load();
    let config = config_of(&fixture.config);
    let mut saturated = 0;
    let mut partial = 0;
    let mut off = 0;
    for gradient in &fixture.gradients {
        let spec = scene_named(&fixture, &gradient.scene);
        let fraction = f(&spec.coverage_fraction);
        // Calls the source, rather than recomputing the formula here: a test
        // that reimplements what it checks cannot fail when the source
        // changes, and this one did not until it was rewritten this way.
        let actual = argusnet_planning::frontier::gradient_weight(&config, fraction);
        let expected = f(&gradient.gradient_weight);
        assert!(
            (actual - expected).abs() <= 1.0e-12,
            "{}: gradient weight {actual} != {expected} at coverage {fraction}",
            gradient.scene
        );
        if expected == 0.0 {
            off += 1;
        } else if expected >= config.coverage_gradient_weight {
            saturated += 1;
        } else {
            partial += 1;
        }
    }
    // All three parts of the ramp must be represented, or its shape is only
    // half pinned: the floor at 25%, the slope, and the clamp at 75%.
    assert!(off > 0, "no scene below the 25% floor");
    assert!(partial > 0, "no scene on the ramp itself");
    assert!(saturated > 0, "no scene above the 75% clamp");
}

/// **Two equivalent mutants, recorded rather than tested around.**
///
/// 1. Skipping the neighbour count when the gradient weight is zero. The
///    weight multiplies the count, and `0.0 * n` is `0.0` for every finite
///    `n`, so computing it and discarding it costs time and changes nothing.
///    The branch is an optimisation, not a behaviour.
/// 2. The `max(1.0)` floor on the distance normaliser. It is
///    `hypot(nx, ny)`, and a grid has at least one cell on each axis, so the
///    value is at least `hypot(1, 1)` — already above 1. The floor can never
///    bind.
#[test]
fn two_recorded_equivalent_mutants_in_the_frontier() {
    // 1. A zero weight annihilates any finite count.
    for count in 0..=9u32 {
        assert_eq!(0.0_f64 * count as f64, 0.0, "0 * {count} must be 0");
    }

    // 2. The normaliser is above its own floor for every possible grid.
    for nx in 1..64usize {
        for ny in 1..64usize {
            let normaliser = (nx as f64).hypot(ny as f64);
            assert!(
                normaliser > 1.0,
                "a {nx}x{ny} grid gives a normaliser of {normaliser}, so the \
                 floor of 1.0 would bind"
            );
        }
    }
}

/// The gap scenes must separate 4- from 8-connectivity, and must not let a
/// fill seeded only from the top and left edges pass.
#[test]
fn the_gap_scenes_pin_the_flood_fill() {
    let fixture = load();
    let leak = scene_named(&fixture, "diagonal_leak_5x5");
    assert_eq!(
        leak.gap_cells.len(),
        2,
        "the diagonally-linked cells are enclosed under 4-connectivity; an \
         8-connected fill would reach them from the border and find no holes"
    );
    let interior = scene_named(&fixture, "bottom_row_interior_5x5");
    assert!(
        interior.gap_cells.is_empty(),
        "this strip reaches the border only through the INTERIOR of the bottom \
         row, so a fill that seeds only the top row plus the side edges would \
         report holes here"
    );
    assert!(interior.covered_cells < interior.total_cells);

    let far_edge = scene_named(&fixture, "bottom_right_opening_5x5");
    assert!(
        far_edge.gap_cells.is_empty(),
        "this region opens onto the bottom and right edges only, so a fill \
         seeded from the top and left alone would wrongly call it enclosed"
    );
    assert!(
        far_edge.covered_cells < far_edge.total_cells,
        "the scene must actually contain uncovered cells"
    );
}

/// Distance is normalised by the grid diagonal, which is what lets the
/// gradient term matter on a large grid at all.
#[test]
fn the_fixture_makes_the_distance_normalisation_observable() {
    let fixture = load();
    let case = fixture
        .selections
        .iter()
        .find(|c| c.note.contains("normalised distance"))
        .expect("the scaling case must exist");
    let chosen = case.expected.as_ref().expect("a cell is chosen");
    assert_eq!(
        chosen,
        &vec![9, 9],
        "the FAR, better-surrounded cell must win; the near block winning \
         means the distance term was left un-normalised"
    );
}
