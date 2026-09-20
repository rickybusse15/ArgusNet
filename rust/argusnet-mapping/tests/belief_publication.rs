//! The join between P4.6d's write side and P4.6c's read side.
//!
//! `BeliefStore` accumulates and publishes; `WorldBeliefQuery` is the
//! planning-facing facade ported in P4.6c. They have to agree, and the way
//! they are wired is what makes mapping load-bearing rather than two parallel
//! implementations of belief. Every assertion here crosses that boundary: a
//! patch and a query over the same store must say the same thing about the
//! same cell.

use argusnet_mapping::belief::BeliefGrids;
use argusnet_mapping::{
    BeliefCellStatus, BeliefStore, CellObservation, GridBounds, Observation, OccupancyEvidence,
    Revision, Sampling, TileIndex, WorldBeliefQuery,
};

fn bounds() -> GridBounds {
    GridBounds {
        x_min_m: 0.0,
        x_max_m: 200.0,
        y_min_m: 0.0,
        y_max_m: 200.0,
        resolution_m: 10.0,
    }
}

fn observation(source: &str, t: f64, cells: Vec<CellObservation>) -> Observation {
    Observation {
        source_id: source.to_string(),
        timestamp_s: t,
        cells,
    }
}

/// Everything the query needs comes off the store; nothing is recomputed.
fn query_over<'a>(store: &'a BeliefStore, scratch: &'a Scratch) -> WorldBeliefQuery<'a> {
    WorldBeliefQuery::new(
        store.bounds(),
        BeliefGrids {
            coverage: store.coverage_counts(),
            mean_height: Some(&scratch.mean_height),
            occupancy: Some(&scratch.occupancy),
            last_observed_s: Some(store.last_observed_grid()),
        },
    )
    .unwrap()
}

/// The two grids the store materialises on demand, kept alive for the query.
struct Scratch {
    mean_height: Vec<f64>,
    occupancy: Vec<f64>,
}

impl Scratch {
    fn of(store: &BeliefStore) -> Self {
        Self {
            mean_height: store.mean_height_grid(),
            occupancy: store.obstacle_probability_grid(),
        }
    }
}

#[test]
fn a_published_patch_and_the_query_agree_cell_for_cell() {
    let b = bounds();
    let mut store = BeliefStore::with_tile_size(b, 8).unwrap();

    // A sweep that covers ground, and one cell with a strong obstacle return.
    store.apply(
        Observation::from_footprint(
            &b,
            "drone-1",
            4.0,
            [60.0, 60.0],
            35.0,
            Some(118.5),
            Sampling::WorldAnchored,
        )
        .unwrap(),
    );
    store.apply(observation(
        "drone-2",
        5.0,
        (0..4)
            .map(|_| CellObservation {
                ij: (6, 6),
                height_m: Some(121.0),
                occupancy: Some(OccupancyEvidence::Occupied { height_m: 9.0 }),
            })
            .collect(),
    ));
    let revision = store.commit();
    assert_eq!(revision.revision, Revision(1));
    assert!(!revision.patches.is_empty());

    let scratch = Scratch::of(&store);
    let query = query_over(&store, &scratch);
    let (_, ny) = store.dimensions();

    for patch in &revision.patches {
        for li in 0..patch.dims.0 {
            for lj in 0..patch.dims.1 {
                let i = patch.origin_ij.0 + li;
                let j = patch.origin_ij.1 + lj;
                let slot = li * patch.dims.1 + lj;
                let cell = query.cell_ij(i, j);

                assert_eq!(cell.coverage_count, patch.coverage_count[slot], "({i},{j})");
                assert_eq!(
                    cell.obstacle_probability, patch.obstacle_probability[slot],
                    "({i},{j})"
                );
                assert_eq!(
                    cell.height_estimate_m.map(f64::to_bits),
                    patch.mean_height_m[slot]
                        .is_finite()
                        .then(|| patch.mean_height_m[slot].to_bits()),
                    "({i},{j})"
                );
                // The query's uncertainty is the geometric proxy, which is the
                // field the patch names `height_uncertainty_m`.
                assert_eq!(
                    cell.height_uncertainty_m.map(f64::to_bits),
                    patch.height_uncertainty_m[slot]
                        .is_finite()
                        .then(|| patch.height_uncertainty_m[slot].to_bits()),
                    "({i},{j})"
                );
                // `observed` is exactly the query's Unknown boundary.
                let unknown = cell.status == BeliefCellStatus::Unknown;
                assert_eq!(!patch.observed[slot], unknown, "({i},{j})");
                assert_eq!(
                    store.coverage_counts()[i * ny + j],
                    patch.coverage_count[slot]
                );
            }
        }
    }

    // The obstacle cell reads as one on both sides.
    let obstacle = query.cell_ij(6, 6);
    assert_eq!(obstacle.status, BeliefCellStatus::KnownObstacle);
    assert!(obstacle.obstacle_probability > 0.9);
}

#[test]
fn an_unpublished_grid_reads_as_unknown_through_the_query_too() {
    let store = BeliefStore::with_tile_size(bounds(), 8).unwrap();
    let scratch = Scratch::of(&store);
    let query = query_over(&store, &scratch);

    for patch in store.all_patches() {
        assert!(!patch.any_observed());
    }
    // Every cell, not a sample: this is the path that runs on every replay the
    // repo ships, all of which carry `belief_cells: []`.
    let (nx, ny) = store.dimensions();
    for i in 0..nx {
        for j in 0..ny {
            let cell = query.cell_ij(i, j);
            assert_eq!(cell.status, BeliefCellStatus::Unknown, "({i},{j})");
            assert_eq!(cell.coverage_count, 0);
            assert!(cell.height_estimate_m.is_none());
            assert!(cell.last_observed_s.is_none());
        }
    }
}

/// A consumer that resyncs from scratch and one that follows revisions must
/// converge on the same belief. Anything else means `patches_since` is lossy.
#[test]
fn following_revisions_converges_on_the_same_belief_as_a_full_resync() {
    let b = bounds();
    let mut store = BeliefStore::with_tile_size(b, 8).unwrap();
    let (nx, ny) = store.dimensions();

    // A follower's mirror of the grid, filled only from patches.
    let mut mirror_coverage = vec![0u32; nx * ny];
    let mut mirror_observed = vec![false; nx * ny];
    let mut seen = Revision::INITIAL;

    for tick in 0..6u32 {
        let x = 30.0 + f64::from(tick) * 22.0;
        store.apply(
            Observation::from_footprint(
                &b,
                format!("drone-{}", tick % 2),
                f64::from(tick),
                [x, 90.0],
                28.0,
                Some(100.0 + f64::from(tick)),
                Sampling::CellDisc,
            )
            .unwrap(),
        );
        store.commit();

        for patch in store.patches_since(seen) {
            for li in 0..patch.dims.0 {
                for lj in 0..patch.dims.1 {
                    let i = patch.origin_ij.0 + li;
                    let j = patch.origin_ij.1 + lj;
                    let slot = li * patch.dims.1 + lj;
                    mirror_coverage[i * ny + j] = patch.coverage_count[slot];
                    mirror_observed[i * ny + j] = patch.observed[slot];
                }
            }
        }
        seen = store.revision();
    }

    assert_eq!(mirror_coverage, store.coverage_counts());
    let expected_observed: Vec<bool> = store.coverage_counts().iter().map(|&c| c > 0).collect();
    assert_eq!(mirror_observed, expected_observed);
    assert!(
        mirror_observed.iter().any(|&o| o) && mirror_observed.iter().any(|&o| !o),
        "the run must leave some ground unobserved for this to mean anything"
    );
}

/// A tile that was never dirtied is absent from `patches_since` but still
/// answerable by name — the missing-tile contract P4.6 asks to define.
#[test]
fn a_tile_outside_every_footprint_is_absent_from_patches_but_still_answerable() {
    let b = bounds();
    let mut store = BeliefStore::with_tile_size(b, 8).unwrap();
    store.apply(
        Observation::from_footprint(
            &b,
            "drone-1",
            1.0,
            [20.0, 20.0],
            20.0,
            Some(100.0),
            Sampling::CellDisc,
        )
        .unwrap(),
    );
    let revision = store.commit();

    let dirty: Vec<TileIndex> = revision.patches.iter().map(|p| p.tile).collect();
    assert_eq!(dirty, vec![TileIndex { ti: 0, tj: 0 }]);

    // A full-size tile nobody looked at.
    let far = TileIndex { ti: 1, tj: 1 };
    assert!(!dirty.contains(&far));
    let patch = store.tile_patch(far);
    assert!(!patch.any_observed());
    assert_eq!(patch.revision, Revision::INITIAL);
    assert_eq!(patch.dims, (8, 8));
    assert_eq!(patch.len(), 64);

    // And the ragged corner: 20 cells at tile size 8 is 8 + 8 + 4, so the last
    // tile is a 4-cell strip and answers at that size rather than padding.
    let corner = store.tile_patch(TileIndex { ti: 2, tj: 2 });
    assert_eq!(corner.dims, (4, 4));
    assert_eq!(corner.origin_ij, (16, 16));
    assert!(!corner.any_observed());
}
