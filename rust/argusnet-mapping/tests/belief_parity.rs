//! Parity for `WorldBeliefQuery` against the captured reference.
//!
//! Every scenario compares the **whole cell table** — every cell, all twelve
//! fields — rather than a handful of scalar probes. The divergences this
//! module is prone to are not wrong numbers, they are wrong *cells*: one cell
//! of a ragged grid reporting itself outside the geofence, or an edge patch
//! divided by the wrong denominator. Neither shows up in an aggregate.
//!
//! Floats arrive as `repr()` strings and are compared bit-for-bit.

use argusnet_mapping::belief::{BeliefCellStatus, BeliefGrids, WorldBeliefQuery};
use argusnet_mapping::footprint::GridBounds;
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    scenarios: Vec<Scenario>,
}

#[derive(Deserialize)]
struct Scenario {
    name: String,
    bounds: BoundsSpec,
    dimensions: [usize; 2],
    geofence: String,
    semantics: String,
    coverage: Vec<u32>,
    mean_height: Option<Vec<String>>,
    occupancy: Option<Vec<String>>,
    last_observed_s: Option<Vec<String>>,
    cells: Vec<CellSpec>,
    frontier_ij: Vec<[usize; 2]>,
    expected_gain: Vec<String>,
    rankings: Vec<RankingSpec>,
    corridors: Vec<CorridorSpec>,
    probes: Vec<ProbeSpec>,
}

#[derive(Deserialize)]
struct ProbeSpec {
    xy_m: [String; 2],
    coverage_at: u32,
    confidence_at: String,
    obstacle_probability_at: String,
    height_estimate_at: Option<String>,
    height_uncertainty_at: Option<String>,
    is_observed: bool,
    is_inside_geofence: bool,
    is_known_safe: bool,
    cell_at_ij: [usize; 2],
}

#[derive(Deserialize)]
struct BoundsSpec {
    x_min_m: String,
    x_max_m: String,
    y_min_m: String,
    y_max_m: String,
    resolution_m: String,
}

#[derive(Deserialize)]
struct CellSpec {
    cell_id: String,
    ij: [usize; 2],
    center_xy_m: [String; 2],
    height_estimate_m: Option<String>,
    height_uncertainty_m: Option<String>,
    obstacle_probability: String,
    terrain_confidence: String,
    coverage_count: u32,
    last_observed_s: Option<String>,
    inside_geofence: bool,
    status: String,
    source_ids: Vec<String>,
}

#[derive(Deserialize)]
struct RankingSpec {
    origin_xy_m: [String; 2],
    max_candidates: usize,
    candidates: Vec<CandidateSpec>,
}

#[derive(Deserialize)]
struct CandidateSpec {
    ij: [usize; 2],
    score: String,
    expected_coverage_gain: String,
    uncertainty_reduction: String,
    travel_cost_m: String,
    risk_cost: String,
}

#[derive(Deserialize)]
struct CorridorSpec {
    start_xy_m: [String; 2],
    end_xy_m: [String; 2],
    sample_count: usize,
    allow_unknown: bool,
    safe: bool,
    /// The answer the naive `k/(n-1)` spelling would give. Where this differs
    /// from `safe`, the probe is the one that separates the two.
    safe_under_naive_alphas: bool,
    /// The answer with numpy's forced final `1.0` dropped.
    safe_under_unsnapped_alphas: bool,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn fv(values: &[String]) -> Vec<f64> {
    values.iter().map(|v| f(v)).collect()
}

/// Bit-for-bit, with the context needed to find the cell that diverged.
fn same(actual: f64, expected: &str, what: &str) {
    assert_eq!(
        actual.to_bits(),
        f(expected).to_bits(),
        "{what}: got {actual}, expected {expected}"
    );
}

fn same_opt(actual: Option<f64>, expected: &Option<String>, what: &str) {
    match (actual, expected) {
        (None, None) => {}
        (Some(a), Some(e)) => same(a, e, what),
        (a, e) => panic!("{what}: got {a:?}, expected {e:?}"),
    }
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/belief.json"
    );
    let text = std::fs::read_to_string(path).expect("belief fixture");
    serde_json::from_str(&text).expect("belief fixture parses")
}

/// A semantic labeller, owned so the query can borrow it.
type SemanticLabeller = Box<dyn Fn(f64, f64) -> Option<String>>;

/// Rebuild the named geofence. These mirror `GEOFENCES` in the capture script;
/// the name travels in the fixture so the two cannot silently drift apart.
fn geofence(name: &str) -> Option<Box<dyn Fn(f64, f64) -> bool>> {
    match name {
        "none" => None,
        "x_below_5" => Some(Box::new(|x: f64, _y: f64| x < 5.0)),
        "always" => Some(Box::new(|_x: f64, _y: f64| true)),
        "never" => Some(Box::new(|_x: f64, _y: f64| false)),
        other => panic!("unknown geofence {other:?}"),
    }
}

fn semantics(name: &str) -> Option<SemanticLabeller> {
    match name {
        "none" => None,
        "band" => Some(Box::new(|_x: f64, y: f64| {
            (y < 3.0).then(|| "scrub".to_string())
        })),
        "empty" => Some(Box::new(|_x: f64, _y: f64| Some(String::new()))),
        other => panic!("unknown semantics {other:?}"),
    }
}

fn status_of(name: &str) -> BeliefCellStatus {
    match name {
        "outside_geofence" => BeliefCellStatus::OutsideGeofence,
        "unknown" => BeliefCellStatus::Unknown,
        "known_obstacle" => BeliefCellStatus::KnownObstacle,
        "uncertain" => BeliefCellStatus::Uncertain,
        "known_safe" => BeliefCellStatus::KnownSafe,
        other => panic!("unknown status {other:?}"),
    }
}

fn check(scenario: &Scenario) {
    let bounds = GridBounds {
        x_min_m: f(&scenario.bounds.x_min_m),
        x_max_m: f(&scenario.bounds.x_max_m),
        y_min_m: f(&scenario.bounds.y_min_m),
        y_max_m: f(&scenario.bounds.y_max_m),
        resolution_m: f(&scenario.bounds.resolution_m),
    };
    let heights = scenario.mean_height.as_ref().map(|v| fv(v));
    let occupancy = scenario.occupancy.as_ref().map(|v| fv(v));
    let last_observed = scenario.last_observed_s.as_ref().map(|v| fv(v));

    let fence = geofence(&scenario.geofence);
    let label = semantics(&scenario.semantics);

    let mut query = WorldBeliefQuery::new(
        bounds,
        BeliefGrids {
            coverage: &scenario.coverage,
            mean_height: heights.as_deref(),
            occupancy: occupancy.as_deref(),
            last_observed_s: last_observed.as_deref(),
        },
    )
    .unwrap_or_else(|e| panic!("{}: {e}", scenario.name));
    if let Some(ref predicate) = fence {
        query = query.with_geofence(predicate.as_ref());
    }
    if let Some(ref predicate) = label {
        query = query.with_semantic_label(predicate.as_ref());
    }

    let name = &scenario.name;
    assert_eq!(
        query.dimensions(),
        (scenario.dimensions[0], scenario.dimensions[1]),
        "{name}: dimensions"
    );

    // --- the whole cell table ---
    let actual: Vec<_> = query.cells().collect();
    assert_eq!(actual.len(), scenario.cells.len(), "{name}: cell count");
    for (cell, spec) in actual.iter().zip(&scenario.cells) {
        let at = format!("{name} {}", spec.cell_id);
        assert_eq!(cell.cell_id, spec.cell_id, "{at}: cell_id");
        assert_eq!(cell.ij, (spec.ij[0], spec.ij[1]), "{at}: ij");
        same(
            cell.center_xy_m.0,
            &spec.center_xy_m[0],
            &format!("{at}: x"),
        );
        same(
            cell.center_xy_m.1,
            &spec.center_xy_m[1],
            &format!("{at}: y"),
        );
        same_opt(
            cell.height_estimate_m,
            &spec.height_estimate_m,
            &format!("{at}: height"),
        );
        same_opt(
            cell.height_uncertainty_m,
            &spec.height_uncertainty_m,
            &format!("{at}: height uncertainty"),
        );
        same(
            cell.obstacle_probability,
            &spec.obstacle_probability,
            &format!("{at}: obstacle probability"),
        );
        same(
            cell.terrain_confidence,
            &spec.terrain_confidence,
            &format!("{at}: terrain confidence"),
        );
        assert_eq!(
            cell.coverage_count, spec.coverage_count,
            "{at}: coverage count"
        );
        same_opt(
            cell.last_observed_s,
            &spec.last_observed_s,
            &format!("{at}: last observed"),
        );
        assert_eq!(
            cell.inside_geofence, spec.inside_geofence,
            "{at}: inside geofence"
        );
        assert_eq!(cell.status, status_of(&spec.status), "{at}: status");
        assert_eq!(cell.source_ids, spec.source_ids, "{at}: source ids");
    }

    // --- frontiers ---
    let frontier: Vec<[usize; 2]> = query
        .frontier_cells()
        .iter()
        .map(|c| [c.ij.0, c.ij.1])
        .collect();
    assert_eq!(frontier, scenario.frontier_ij, "{name}: frontier cells");

    // --- expected gain, cell by cell ---
    let (nx, ny) = query.dimensions();
    for i in 0..nx {
        for j in 0..ny {
            same(
                query.expected_coverage_gain(i, j),
                &scenario.expected_gain[i * ny + j],
                &format!("{name} expected gain at ({i},{j})"),
            );
        }
    }

    // --- rankings ---
    for ranking in &scenario.rankings {
        let origin = (f(&ranking.origin_xy_m[0]), f(&ranking.origin_xy_m[1]));
        let ranked = query.score_frontiers(origin, ranking.max_candidates);
        assert_eq!(
            ranked.len(),
            ranking.candidates.len(),
            "{name}: candidate count from {origin:?}"
        );
        for (index, (candidate, spec)) in ranked.iter().zip(&ranking.candidates).enumerate() {
            let at = format!("{name} rank {index} from {origin:?}");
            assert_eq!(
                candidate.cell.ij,
                (spec.ij[0], spec.ij[1]),
                "{at}: ij (ordering)"
            );
            same(candidate.score, &spec.score, &format!("{at}: score"));
            same(
                candidate.expected_coverage_gain,
                &spec.expected_coverage_gain,
                &format!("{at}: gain"),
            );
            same(
                candidate.uncertainty_reduction,
                &spec.uncertainty_reduction,
                &format!("{at}: uncertainty reduction"),
            );
            same(
                candidate.travel_cost_m,
                &spec.travel_cost_m,
                &format!("{at}: travel cost"),
            );
            same(candidate.risk_cost, &spec.risk_cost, &format!("{at}: risk"));
        }
    }

    // --- scalar probes, including points outside the raw bounds ---
    for probe in &scenario.probes {
        let (x, y) = (f(&probe.xy_m[0]), f(&probe.xy_m[1]));
        let at = format!("{name} probe ({x}, {y})");
        assert_eq!(query.coverage_at(x, y), probe.coverage_at, "{at}: coverage");
        same(
            query.confidence_at(x, y),
            &probe.confidence_at,
            &format!("{at}: confidence"),
        );
        same(
            query.obstacle_probability_at(x, y),
            &probe.obstacle_probability_at,
            &format!("{at}: obstacle probability"),
        );
        same_opt(
            query.height_estimate_at(x, y),
            &probe.height_estimate_at,
            &format!("{at}: height"),
        );
        same_opt(
            query.height_uncertainty_at(x, y),
            &probe.height_uncertainty_at,
            &format!("{at}: height uncertainty"),
        );
        assert_eq!(query.is_observed(x, y), probe.is_observed, "{at}: observed");
        assert_eq!(
            query.is_inside_geofence(x, y),
            probe.is_inside_geofence,
            "{at}: inside geofence"
        );
        assert_eq!(
            query.is_known_safe(x, y),
            probe.is_known_safe,
            "{at}: known safe"
        );
        assert_eq!(
            query.cell_at(x, y).ij,
            (probe.cell_at_ij[0], probe.cell_at_ij[1]),
            "{at}: cell_at"
        );
    }

    // --- corridors ---
    for probe in &scenario.corridors {
        let start = (f(&probe.start_xy_m[0]), f(&probe.start_xy_m[1]));
        let end = (f(&probe.end_xy_m[0]), f(&probe.end_xy_m[1]));
        assert_eq!(
            query.safe_corridor_between(start, end, probe.sample_count, probe.allow_unknown),
            probe.safe,
            "{name}: corridor {start:?} -> {end:?} n={} unknown={}",
            probe.sample_count,
            probe.allow_unknown
        );
    }
}

#[test]
fn belief_queries_match_the_reference() {
    let fixture = load();
    assert!(!fixture.scenarios.is_empty(), "no scenarios");
    for scenario in &fixture.scenarios {
        check(scenario);
    }
}

/// The fixture is only worth what its hardest cases pin. Assert from the Rust
/// side that the three known divergences are still reachable, so the set
/// cannot silently lose its teeth — the same guard the capture script applies
/// on the way in.
#[test]
fn the_fixture_still_covers_the_three_known_divergences() {
    let fixture = load();

    // (a) A cell outside the geofence while the grid holds a count for it.
    let ragged = fixture
        .scenarios
        .iter()
        .flat_map(|s| {
            let ny = s.dimensions[1];
            s.cells
                .iter()
                .map(move |c| !c.inside_geofence && s.coverage[c.ij[0] * ny + c.ij[1]] > 0)
        })
        .filter(|hit| *hit)
        .count();
    assert!(
        ragged > 0,
        "no cell reports itself outside the geofence while the grid holds a \
         count for it; the ragged-grid divergence is untested"
    );

    // (b) An expected_gain that differs from the always-divide-by-9 spelling.
    let patchy = fixture
        .scenarios
        .iter()
        .map(|s| {
            let (nx, ny) = (s.dimensions[0], s.dimensions[1]);
            (0..nx)
                .flat_map(|i| (0..ny).map(move |j| (i, j)))
                .filter(|(i, j)| {
                    let zeros = (i.saturating_sub(1)..(i + 2).min(nx))
                        .flat_map(|pi| {
                            (j.saturating_sub(1)..(j + 2).min(ny)).map(move |pj| (pi, pj))
                        })
                        .filter(|(pi, pj)| s.coverage[pi * ny + pj] == 0)
                        .count();
                    f(&s.expected_gain[i * ny + j]) != zeros as f64 / 9.0
                })
                .count()
        })
        .sum::<usize>();
    assert!(
        patchy > 0,
        "every expected_coverage_gain equals the divide-by-9 spelling; the \
         shrinking-patch denominator is untested"
    );

    // (c) A corridor whose ANSWER flips under the naive `k/(n-1)` spelling.
    //     Requiring only that a divergent sample count is present is the
    //     weaker property, and the first version of this guard checked exactly
    //     that — it passed while a port with the naive spelling survived the
    //     whole mutation set.
    for (label, wrong) in [
        (
            "the naive k/(n-1) spelling",
            (|p: &CorridorSpec| p.safe_under_naive_alphas) as fn(&CorridorSpec) -> bool,
        ),
        ("dropping numpy's forced final 1.0", |p: &CorridorSpec| {
            p.safe_under_unsnapped_alphas
        }),
    ] {
        let flips = fixture
            .scenarios
            .iter()
            .flat_map(|s| &s.corridors)
            .any(|p| p.safe != wrong(p));
        assert!(
            flips,
            "no corridor returns a different answer under {label}; a divergent \
             sample count alone is not enough — the samples must land in cells \
             that differ in safety"
        );
    }

    // (d) A probe outside the raw bounds under a permissive geofence. The cell
    //     table cannot reach this: a cell is always evaluated at its own
    //     centre, so the inclusive bound tests and the lower clamp on
    //     `xy_to_ij` are only observable from a scalar probe.
    let outside = fixture
        .scenarios
        .iter()
        .filter(|s| s.geofence == "none" || s.geofence == "always")
        .flat_map(|s| &s.probes)
        .filter(|p| !p.is_inside_geofence)
        .count();
    assert!(
        outside > 0,
        "no probe falls outside the raw bounds under a permissive geofence; \
         the inclusive bound tests and the lower clamp are untested"
    );
}
