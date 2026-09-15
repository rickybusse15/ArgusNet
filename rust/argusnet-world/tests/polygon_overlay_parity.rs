//! Parity for `build_land_cover_layer`'s obstacle and patch polygon overlay.
//!
//! The classification pass this runs after is covered by `land_cover_parity`,
//! whose fixture was captured with `obstacles=()` and `patches=()` precisely so
//! these two loops were no-ops there. Everything they do is checked here.
//!
//! # The fixture is a difference, not a snapshot
//!
//! Each case records the reference run **twice** on identical inputs: `base_*`
//! with no obstacles and no patches, `final_*` with them. Everything upstream
//! of the overlay is deterministic, so the two differ by the overlay and
//! nothing else. This test loads `base_*`, applies the port, and must land on
//! `final_*` — so it cannot pass by reproducing the classification, only by
//! reproducing the overlay.
//!
//! # The primitives are rebuilt, not handed over
//!
//! Obstacles are recorded as **constructor arguments**, and this test builds
//! each one through [`Obstacle`]'s real constructors before the overlay sees
//! it. Two of the five primitives derive their footprint rather than storing
//! one — in the reference that shows up as a *callable* attribute, a
//! distinction that does not survive into Rust — and both of those cases here
//! are **rotated**, which is the geometry the old clamped-denominator polygon
//! test got wrong while every axis-aligned fixture passed.
//!
//! # Bit-for-bit, and whole grids
//!
//! Classes and densities are `u8`. Every cell of every case is compared.

use argusnet_world::land_cover_build::{
    apply_land_cover_overlay, obstacle_cover_polygon, CoverGrid, LandCoverClass, LandCoverPatch,
};
use argusnet_world::obstacles::{BlockerType, Obstacle, ObstacleShape};
use serde::Deserialize;

#[derive(Deserialize)]
struct Fixture {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    rows: usize,
    cols: usize,
    suppress_vegetation: bool,
    cover_bounds: Bounds,
    cover_resolution_m: String,
    obstacles: Vec<ObstacleSpec>,
    patches: Vec<PatchSpec>,
    base_classes: Vec<u8>,
    base_density: Vec<u8>,
    final_classes: Vec<u8>,
    final_density: Vec<u8>,
}

#[derive(Deserialize)]
struct Bounds {
    x_min_m: String,
    y_min_m: String,
}

/// Every constructor argument any of the five primitives takes. The fixture
/// normalises the per-kind defaults before recording, so nothing here has to
/// know what the reference's default canopy height is.
#[derive(Deserialize)]
struct ObstacleSpec {
    kind: String,
    id: String,
    blocker_type: String,
    base_z_m: String,
    top_z_m: String,
    footprint_xy_m: Option<Vec<[String; 2]>>,
    density: Option<String>,
    center_x_m: Option<String>,
    center_y_m: Option<String>,
    radius_m: Option<String>,
    length_m: Option<String>,
    width_m: Option<String>,
    yaw_rad: Option<String>,
    start_xy_m: Option<[String; 2]>,
    end_xy_m: Option<[String; 2]>,
    thickness_m: Option<String>,
    /// What the reference's `footprint_xy_m` attribute resolved to — `None`
    /// for the primitives that have no such attribute at all.
    derived_footprint_xy_m: Option<Vec<[String; 2]>>,
}

#[derive(Deserialize)]
struct PatchSpec {
    polygon_xy_m: Vec<[String; 2]>,
    land_cover_class: u8,
    density: i64,
}

fn f(text: &str) -> f64 {
    text.parse().expect("fixture float")
}

fn polygon(points: &[[String; 2]]) -> Vec<[f64; 2]> {
    points.iter().map(|p| [f(&p[0]), f(&p[1])]).collect()
}

fn blocker(name: &str) -> BlockerType {
    match name {
        "building" => BlockerType::Building,
        "wall" => BlockerType::Wall,
        "vegetation" => BlockerType::Vegetation,
        other => BlockerType::Other(other.to_string()),
    }
}

fn load() -> Fixture {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../tests/fixtures/polygon_overlay.json"
    );
    let text = std::fs::read_to_string(path).expect("polygon_overlay.json");
    serde_json::from_str(&text).expect("polygon_overlay.json parses")
}

fn build_obstacle(spec: &ObstacleSpec) -> Obstacle {
    let (base, top) = (f(&spec.base_z_m), f(&spec.top_z_m));
    let footprint = || polygon(spec.footprint_xy_m.as_ref().expect("footprint"));
    match spec.kind.as_str() {
        "forest-stand-v1" => Obstacle::forest_stand(
            &spec.id,
            footprint(),
            base,
            top,
            f(spec.density.as_ref().expect("stand density")),
        ),
        "building-v1" | "polygon-prism-v1" => Obstacle::polygon_prism(
            &spec.id,
            blocker(&spec.blocker_type),
            footprint(),
            base,
            top,
        ),
        "box-v1" => Obstacle::oriented_box(
            &spec.id,
            blocker(&spec.blocker_type),
            f(spec.center_x_m.as_ref().expect("center_x_m")),
            f(spec.center_y_m.as_ref().expect("center_y_m")),
            f(spec.length_m.as_ref().expect("length_m")),
            f(spec.width_m.as_ref().expect("width_m")),
            f(spec.yaw_rad.as_ref().expect("yaw_rad")),
            base,
            top,
        ),
        "wall-v1" => {
            let start = spec.start_xy_m.as_ref().expect("start_xy_m");
            let end = spec.end_xy_m.as_ref().expect("end_xy_m");
            Obstacle::wall_segment(
                &spec.id,
                blocker(&spec.blocker_type),
                [f(&start[0]), f(&start[1])],
                [f(&end[0]), f(&end[1])],
                f(spec.thickness_m.as_ref().expect("thickness_m")),
                base,
                top,
            )
        }
        "cylinder-v1" => Obstacle::cylinder(
            &spec.id,
            blocker(&spec.blocker_type),
            f(spec.center_x_m.as_ref().expect("center_x_m")),
            f(spec.center_y_m.as_ref().expect("center_y_m")),
            f(spec.radius_m.as_ref().expect("radius_m")),
            base,
            top,
        ),
        other => panic!("unknown obstacle kind {other}"),
    }
    .expect("the fixture's primitives are all valid")
}

/// Cell centres, `x_min + (i + 0.5) * resolution`. Derived rather than
/// recorded, exactly as `land_cover_parity` derives them from the same two
/// fields.
fn cell_centres(case: &Case) -> (Vec<f64>, Vec<f64>) {
    let res = f(&case.cover_resolution_m);
    let (x_min, y_min) = (f(&case.cover_bounds.x_min_m), f(&case.cover_bounds.y_min_m));
    let mut xx = Vec::with_capacity(case.rows * case.cols);
    let mut yy = Vec::with_capacity(case.rows * case.cols);
    for row in 0..case.rows {
        for col in 0..case.cols {
            xx.push(x_min + ((col as f64 + 0.5) * res));
            yy.push(y_min + ((row as f64 + 0.5) * res));
        }
    }
    (xx, yy)
}

fn overlay(case: &Case) -> CoverGrid {
    let (xx, yy) = cell_centres(case);
    let mut grid = CoverGrid {
        rows: case.rows,
        cols: case.cols,
        classes: case.base_classes.clone(),
        density: case.base_density.clone(),
    };
    let obstacles: Vec<Obstacle> = case.obstacles.iter().map(build_obstacle).collect();
    let patches: Vec<LandCoverPatch> = case
        .patches
        .iter()
        .map(|patch| LandCoverPatch {
            polygon_xy_m: polygon(&patch.polygon_xy_m),
            land_cover_class: LandCoverClass::from_u8(patch.land_cover_class)
                .expect("the fixture only records real classes"),
            density: patch.density,
        })
        .collect();
    apply_land_cover_overlay(
        &mut grid,
        &xx,
        &yy,
        &obstacles,
        &patches,
        case.suppress_vegetation,
    )
    .expect("the overlay accepts a well-formed grid");
    grid
}

#[test]
fn every_case_reproduces_the_reference_overlay() {
    let fixture = load();
    assert!(fixture.cases.len() >= 14, "fixture shrank unexpectedly");
    for case in &fixture.cases {
        let grid = overlay(case);
        for k in 0..case.rows * case.cols {
            let (row, col) = (k / case.cols, k % case.cols);
            assert_eq!(
                grid.classes[k], case.final_classes[k],
                "{}: class at row {row}, col {col}",
                case.name
            );
            assert_eq!(
                grid.density[k], case.final_density[k],
                "{}: density at row {row}, col {col}",
                case.name
            );
        }
    }
}

/// How far a derived footprint vertex may sit from the reference's, metres.
///
/// Not zero, and the reason is worth reading before anyone tightens it.
/// `OrientedBox.footprint_xy_m()` rotates its corners with `corners @ rot.T`,
/// which numpy hands to BLAS — and BLAS accumulates the two-term dot product
/// with a **fused multiply-add**: `fma(y, -sin, x * cos)`, one rounding where
/// a plain `x * cos - y * sin` has two. Reproducing that was checked and it
/// does reproduce, exactly; it is deliberately *not* done.
///
/// ADR-006 Decision 1 is the reason. Whether numpy's matmul fuses depends on
/// the BLAS vendor, the matrix size and the CPU's feature set — a `mul_add` in
/// this crate would match one of those builds and silently mismatch the next.
/// That is the fragile bit-parity the ADR declines, and it is a different
/// case from the PRNG in P4.3e, where the stream is defined by an algorithm
/// rather than by a vendor.
///
/// **This cannot change a classification**, which is why accepting it is not
/// merely convenient. The observed divergence is ~1e-14 m, and
/// `point_in_polygon` treats anything within `SEGMENT_EPSILON_M` — 1e-6 m — of
/// an edge as on the boundary and inside. A vertex shift eight orders of
/// magnitude inside that band cannot move a cell across it: a point close
/// enough to be at risk was already answered by the boundary short-circuit,
/// identically, under both spellings. The overlay comparison above passes
/// bit-for-bit on the same fixture, which is the observation rather than the
/// argument.
const FOOTPRINT_TOLERANCE_M: f64 = 1.0e-12;

/// `obstacles::SEGMENT_EPSILON_M`, the band `point_in_polygon` answers with its
/// boundary short-circuit. Private there, restated here because the tolerance
/// above is only defensible relative to it.
const BOUNDARY_BAND_M: f64 = 1.0e-6;

/// The argument above, as a compile-time check. Widening the tolerance to
/// within a thousandth of the boundary band would put a vertex shift close
/// enough to change a cell's class, and the reasoning would no longer hold.
const _: () = assert!(FOOTPRINT_TOLERANCE_M * 1.0e3 < BOUNDARY_BAND_M);

#[test]
fn the_constructors_derive_the_same_footprints_the_reference_does() {
    // `OrientedBox` and `WallSegment` compute their footprint instead of
    // storing one, and the reference reaches it through a callable. Checking
    // the derived polygon separately means a rotation error shows up here
    // rather than as a mysterious handful of mispainted cells above.
    let fixture = load();
    let mut derived = 0;
    let mut worst = 0.0f64;
    for case in &fixture.cases {
        for spec in &case.obstacles {
            let obstacle = build_obstacle(spec);
            match (&obstacle.shape, &spec.derived_footprint_xy_m) {
                (ObstacleShape::Polygon { footprint_xy_m }, Some(expected)) => {
                    let expected = polygon(expected);
                    assert_eq!(footprint_xy_m.len(), expected.len());
                    for (got, want) in footprint_xy_m.iter().zip(&expected) {
                        for axis in 0..2 {
                            let delta = (got[axis] - want[axis]).abs();
                            worst = worst.max(delta);
                            assert!(
                                delta <= FOOTPRINT_TOLERANCE_M,
                                "{}: {} vertex out by {delta} m",
                                case.name,
                                spec.id
                            );
                        }
                    }
                    // A stored footprint is copied, not computed, so it must
                    // be exact. Only the derived ones may drift at all.
                    if spec.footprint_xy_m.is_some() {
                        assert_eq!(
                            footprint_xy_m, &expected,
                            "{}: {} stores its footprint and must copy it verbatim",
                            case.name, spec.id
                        );
                    } else {
                        derived += 1;
                    }
                }
                (ObstacleShape::Cylinder { .. }, None) => {}
                (shape, expected) => panic!(
                    "{}: {} has shape {shape:?} but the reference recorded {expected:?}",
                    case.name, spec.id
                ),
            }
        }
    }
    assert!(
        derived >= 2,
        "no primitive in the fixture derives its footprint, so the rotated \
         cases this test exists for are not present"
    );
    println!("worst derived-footprint divergence: {worst:e} m");
}

#[test]
fn a_cylinder_contributes_no_polygon_at_all() {
    // The reference's `else: continue`. A cylinder occludes sight and blocks
    // movement and contributes nothing to land cover, which is an asymmetry in
    // the reference rather than an omission here.
    let fixture = load();
    let cylinders: Vec<&ObstacleSpec> = fixture
        .cases
        .iter()
        .flat_map(|case| case.obstacles.iter())
        .filter(|spec| spec.kind == "cylinder-v1")
        .collect();
    assert!(!cylinders.is_empty(), "no cylinder in the fixture");
    for spec in cylinders {
        let obstacle = build_obstacle(spec);
        assert_eq!(obstacle_cover_polygon(&obstacle, false), None);
        assert_eq!(obstacle_cover_polygon(&obstacle, true), None);
    }
}

#[test]
fn suppression_drops_the_stands_and_keeps_the_buildings() {
    // Both halves, on the fixture's own primitives: a wrong port that skips
    // every obstacle under the flag passes the first assertion and fails the
    // second.
    let fixture = load();
    let (mut stands, mut others) = (0, 0);
    for case in &fixture.cases {
        for spec in &case.obstacles {
            let obstacle = build_obstacle(spec);
            if spec.kind == "forest-stand-v1" {
                stands += 1;
                assert_eq!(obstacle_cover_polygon(&obstacle, true), None, "{}", spec.id);
                assert_eq!(
                    obstacle_cover_polygon(&obstacle, false)
                        .expect("a stand contributes when vegetation is not suppressed")
                        .cover_class,
                    LandCoverClass::Forest
                );
            } else if !matches!(obstacle.shape, ObstacleShape::Cylinder { .. }) {
                others += 1;
                assert_eq!(
                    obstacle_cover_polygon(&obstacle, true)
                        .expect("only stands are suppressed")
                        .cover_class,
                    LandCoverClass::Urban,
                    "{}",
                    spec.id
                );
            }
        }
    }
    assert!(stands > 0 && others > 0, "{stands} stands, {others} others");
}

#[test]
fn the_fixture_would_notice_a_port_that_did_nothing() {
    // `no_polygons_is_identity` is the one case where doing nothing is the
    // right answer. Every other case must move at least one cell, or the
    // comparison above proves only that the base grid parses.
    let fixture = load();
    let mut moved = 0;
    for case in &fixture.cases {
        let changed =
            case.base_classes != case.final_classes || case.base_density != case.final_density;
        if case.name == "no_polygons_is_identity" {
            assert!(!changed, "the identity case is not an identity");
        } else {
            assert!(changed, "{}: the overlay changed nothing", case.name);
            moved += 1;
        }
    }
    assert!(moved >= 13, "only {moved} cases exercise the overlay");
}
