//! What a run's map was asked for against what it achieved.
//!
//! A count of cells met says how much of the map is done. It does not say
//! whether the flight did what its sensor model promised, whether the map's
//! own statement of its error can be believed, or what is left and why. This
//! does, at the end of a run, from the finished map:
//!
//! - **predicted**: what one pass at the survey height and speed should
//!   leave on the ground, from the declared sensors alone;
//! - **achieved**: what the cells hold;
//! - **calibration**: the map's claimed sigma against its true error;
//! - **missed**: every cell not met, by the map's reason and by what the true
//!   world has there;
//! - **areas**: where the cells still worth flying for are.
//!
//! Calibration and causes read the true world. They are evaluation, computed
//! after the run; nothing a platform does depends on them.

use std::collections::BTreeMap;

use argusnet_mapping::quality::{
    assess, centre_height_m, vertical_sigma_m, CellEvidence, DeficitCounts, DeficitReason,
    QualityRequirement,
};
use serde::{Deserialize, Serialize};

pub const MAP_REPORT_FORMAT: &str = "argusnet.map-report";
pub const MAP_REPORT_VERSION: u32 = 1;

/// Areas listed, largest first.
const MAX_AREAS: usize = 10;
/// The fewest cells worth listing as an area.
const MIN_AREA_CELLS: usize = 4;
/// A cell this near the map's edge, in cells, is an edge cell.
const EDGE_CELLS: usize = 2;

/// What one pass should leave, from the declared sensors.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PassPrediction {
    /// The height over the ground and the speed it is for.
    pub agl_m: f64,
    pub speed_mps: f64,
    /// The width of ground whose returns the requirement accepts, metres.
    pub swath_m: f64,
    /// Accepted returns per square metre of that swath.
    pub density_per_m2: f64,
    /// A cell's vertical sigma at that density, metres.
    pub vertical_sigma_m: f64,
    /// The visible image's sample distance straight down; absent without
    /// the imaging model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_gsd_m: Option<f64>,
}

/// The 10th, 50th and 90th percentiles of something over cells.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Spread {
    pub cells: u64,
    pub p10: f64,
    pub p50: f64,
    pub p90: f64,
}

impl Spread {
    fn of(mut values: Vec<f64>) -> Option<Self> {
        if values.is_empty() {
            return None;
        }
        values.sort_by(f64::total_cmp);
        let at = |share: f64| values[((values.len() - 1) as f64 * share).round() as usize];
        Some(Self {
            cells: values.len() as u64,
            p10: at(0.1),
            p50: at(0.5),
            p90: at(0.9),
        })
    }
}

/// What the cells hold.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Achieved {
    pub met_fraction: f64,
    pub counts: DeficitCounts,
    /// Over cells with accepted returns.
    pub density_per_m2: Option<Spread>,
    pub vertical_sigma_m: Option<Spread>,
    /// Over cells placed by an estimated pose; absent without one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizontal_sigma_m: Option<Spread>,
    /// Over imaged cells; absent without an image requirement.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub image_gsd_m: Option<Spread>,
    /// The median density over the prediction for one pass: how many
    /// passes' worth the median cell got.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passes_at_median: Option<f64>,
}

/// The map's claimed error against its true one, over cells with accepted
/// returns. A cell's error is its height (its accepted returns' plane at its
/// centre, or their mean where none is fitted) less the true ground at its
/// centre, so a mean's includes where in the cell the returns fell.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Calibration {
    pub cells: u64,
    pub rms_error_m: f64,
    /// The mean error: what no averaging within a run removes.
    pub bias_m: f64,
    pub rms_claimed_sigma_m: f64,
    /// RMS of error over claimed sigma: 1 is honest, above it the map
    /// claims more than it knows.
    pub normalized_rms: f64,
    pub within_1_sigma: f64,
    pub within_2_sigma: f64,
}

/// What the true world has in a cell.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MissCause {
    /// A building or wall is in the cell or beside it.
    Structure,
    /// A tree's crown reaches over the cell.
    Canopy,
    /// The cell is at the map's edge.
    Edge,
    /// Open ground: nothing but flying was in the way.
    Open,
}

/// How many cells missed for one reason and one cause.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Missed {
    pub reason: DeficitReason,
    pub cause: MissCause,
    pub cells: u64,
}

/// A connected area of cells flying could still bring to the requirement.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MissedArea {
    pub cells: u64,
    pub centre_m: [f64; 2],
    /// `[x_min, y_min, x_max, y_max]`.
    pub bounds_m: [f64; 4],
    /// The commonest reason and cause in it.
    pub reason: DeficitReason,
    pub cause: MissCause,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MapReport {
    pub format: String,
    pub version: u32,
    pub tick: u64,
    pub timestamp_s: f64,
    pub resolution_m: f64,
    pub cells: u64,
    pub asked: QualityRequirement,
    /// Absent when no rig says what a pass should leave.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub predicted: Option<PassPrediction>,
    pub achieved: Achieved,
    /// Absent when nothing was accepted, or there is no true ground to
    /// judge against.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<Calibration>,
    pub missed: Vec<Missed>,
    pub areas: Vec<MissedArea>,
}

/// A finished map and the true world, as the report reads them. Cells are
/// `j`-major: `j * nx + i`.
pub struct ReportInput<'a> {
    pub nx: usize,
    pub ny: usize,
    pub origin_m: [f64; 2],
    pub resolution_m: f64,
    pub tick: u64,
    pub timestamp_s: f64,
    pub requirement: &'a QualityRequirement,
    pub per_return_sigma_m: f64,
    pub finest_image_gsd_m: Option<f64>,
    pub evidence: Vec<CellEvidence>,
    pub predicted: Option<PassPrediction>,
    /// True ground at each cell's centre; empty without it.
    pub true_ground_m: Vec<f64>,
    /// True structures and crowns, by cell; empty for none.
    pub structures: &'a [bool],
    pub canopy: &'a [bool],
}

pub fn build(input: ReportInput<'_>) -> MapReport {
    let ReportInput { nx, ny, .. } = input;
    let area = input.resolution_m * input.resolution_m;
    let quality: Vec<_> = input
        .evidence
        .iter()
        .map(|evidence| {
            assess(
                evidence,
                area,
                input.per_return_sigma_m,
                input.finest_image_gsd_m,
                input.requirement,
            )
        })
        .collect();
    let mut counts = DeficitCounts::default();
    for cell in &quality {
        counts.add(cell.reason);
    }
    let accepted: Vec<usize> = (0..quality.len())
        .filter(|&k| input.evidence[k].accepted_returns > 0)
        .collect();
    let density = Spread::of(
        accepted
            .iter()
            .map(|&k| quality[k].density_per_m2)
            .collect(),
    );
    let sigma = |k: usize| {
        vertical_sigma_m(
            &input.evidence[k],
            input.per_return_sigma_m,
            input.requirement.bias_floor_m,
        )
        .expect("accepted cells have a sigma")
    };
    let achieved = Achieved {
        met_fraction: counts.met_fraction(),
        density_per_m2: density,
        vertical_sigma_m: Spread::of(accepted.iter().map(|&k| sigma(k)).collect()),
        horizontal_sigma_m: Spread::of(
            quality
                .iter()
                .filter_map(|cell| cell.horizontal_sigma_m)
                .collect(),
        ),
        image_gsd_m: Spread::of(
            input
                .evidence
                .iter()
                .filter_map(|evidence| evidence.image_gsd_m)
                .collect(),
        ),
        passes_at_median: density
            .zip(input.predicted.as_ref())
            .and_then(|(density, predicted)| {
                (predicted.density_per_m2 > 0.0).then(|| density.p50 / predicted.density_per_m2)
            }),
        counts,
    };

    let calibration = (!accepted.is_empty() && !input.true_ground_m.is_empty()).then(|| {
        let n = accepted.len() as f64;
        let (mut sq, mut sum, mut sigma_sq, mut z_sq) = (0.0, 0.0, 0.0, 0.0);
        let (mut one, mut two) = (0u64, 0u64);
        for &k in &accepted {
            let evidence = &input.evidence[k];
            let error = centre_height_m(evidence).expect("accepted cells have a height")
                - input.true_ground_m[k];
            let claimed = sigma(k);
            let z = error / claimed.max(1e-9);
            sq += error * error;
            sum += error;
            sigma_sq += claimed * claimed;
            z_sq += z * z;
            one += u64::from(z.abs() <= 1.0);
            two += u64::from(z.abs() <= 2.0);
        }
        Calibration {
            cells: accepted.len() as u64,
            rms_error_m: (sq / n).sqrt(),
            bias_m: sum / n,
            rms_claimed_sigma_m: (sigma_sq / n).sqrt(),
            normalized_rms: (z_sq / n).sqrt(),
            within_1_sigma: one as f64 / n,
            within_2_sigma: two as f64 / n,
        }
    });

    let holds = |layer: &[bool], k: usize| layer.get(k).copied().unwrap_or(false);
    let cause = |k: usize| -> MissCause {
        let (i, j) = (k % nx, k / nx);
        let beside = (j.saturating_sub(1)..=(j + 1).min(ny - 1)).any(|jj| {
            (i.saturating_sub(1)..=(i + 1).min(nx - 1))
                .any(|ii| holds(input.structures, jj * nx + ii))
        });
        if beside {
            MissCause::Structure
        } else if holds(input.canopy, k) {
            MissCause::Canopy
        } else if i < EDGE_CELLS || j < EDGE_CELLS || i + EDGE_CELLS >= nx || j + EDGE_CELLS >= ny {
            MissCause::Edge
        } else {
            MissCause::Open
        }
    };
    let mut missed: BTreeMap<(DeficitReason, MissCause), u64> = BTreeMap::new();
    for (k, cell) in quality.iter().enumerate() {
        if cell.reason != DeficitReason::Met {
            *missed.entry((cell.reason, cause(k))).or_default() += 1;
        }
    }

    // Connected areas (sides and corners) of cells flying could still fix,
    // in cell order, so equal sizes list the same way every time.
    let mut seen: Vec<bool> = quality
        .iter()
        .map(|cell| !cell.reason.is_improvable())
        .collect();
    let mut areas = Vec::new();
    for start in 0..quality.len() {
        if seen[start] {
            continue;
        }
        seen[start] = true;
        let mut cells = Vec::new();
        let mut open = vec![start];
        while let Some(k) = open.pop() {
            cells.push(k);
            let (i, j) = (k % nx, k / nx);
            for jj in j.saturating_sub(1)..=(j + 1).min(ny - 1) {
                for ii in i.saturating_sub(1)..=(i + 1).min(nx - 1) {
                    let next = jj * nx + ii;
                    if !seen[next] {
                        seen[next] = true;
                        open.push(next);
                    }
                }
            }
        }
        if cells.len() >= MIN_AREA_CELLS {
            areas.push(describe(&input, &cells, |k| quality[k].reason, cause));
        }
    }
    areas.sort_by_key(|area| std::cmp::Reverse(area.cells));
    areas.truncate(MAX_AREAS);

    MapReport {
        format: MAP_REPORT_FORMAT.into(),
        version: MAP_REPORT_VERSION,
        tick: input.tick,
        timestamp_s: input.timestamp_s,
        resolution_m: input.resolution_m,
        cells: quality.len() as u64,
        asked: input.requirement.clone(),
        predicted: input.predicted,
        achieved,
        calibration,
        missed: missed
            .into_iter()
            .map(|((reason, cause), cells)| Missed {
                reason,
                cause,
                cells,
            })
            .collect(),
        areas,
    }
}

fn describe(
    input: &ReportInput<'_>,
    cells: &[usize],
    reason: impl Fn(usize) -> DeficitReason,
    cause: impl Fn(usize) -> MissCause,
) -> MissedArea {
    let n = cells.len() as f64;
    let (mut low, mut high) = ([usize::MAX; 2], [0usize; 2]);
    let mut sum = [0.0; 2];
    let mut reasons: BTreeMap<DeficitReason, u64> = BTreeMap::new();
    let mut causes: BTreeMap<MissCause, u64> = BTreeMap::new();
    for &k in cells {
        let at = [k % input.nx, k / input.nx];
        for axis in 0..2 {
            low[axis] = low[axis].min(at[axis]);
            high[axis] = high[axis].max(at[axis]);
            sum[axis] += (at[axis] as f64 + 0.5) * input.resolution_m;
        }
        *reasons.entry(reason(k)).or_default() += 1;
        *causes.entry(cause(k)).or_default() += 1;
    }
    let edge = |axis: usize, index: usize| input.origin_m[axis] + index as f64 * input.resolution_m;
    MissedArea {
        cells: cells.len() as u64,
        centre_m: std::array::from_fn(|axis| input.origin_m[axis] + sum[axis] / n),
        bounds_m: [
            edge(0, low[0]),
            edge(1, low[1]),
            edge(0, high[0] + 1),
            edge(1, high[1] + 1),
        ],
        reason: commonest(&reasons),
        cause: commonest(&causes),
    }
}

/// The key counted most often; the first in order of equals.
fn commonest<K: Copy + Ord>(counts: &BTreeMap<K, u64>) -> K {
    counts
        .iter()
        .max_by(|a, b| a.1.cmp(b.1).then(b.0.cmp(a.0)))
        .map(|(key, _)| *key)
        .expect("an area has cells")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn requirement() -> QualityRequirement {
        QualityRequirement {
            min_ground_density_per_m2: Some(0.02),
            max_vertical_sigma_m: None,
            max_age_s: None,
            max_image_gsd_m: None,
            max_horizontal_sigma_m: None,
            max_incidence_deg: 40.0,
            bias_floor_m: 0.0,
        }
    }

    /// `n` accepted returns at `height`.
    fn cell(n: u32, height: f64) -> CellEvidence {
        CellEvidence {
            ground_returns: n,
            accepted_returns: n,
            accepted_height_sum_m: n as f64 * height,
            accepted_height_sq_sum_m2: n as f64 * height * height,
            accepted_age_s: (n > 0).then_some(0.0),
            image_gsd_m: None,
            plane: None,
            relief_m2: 0.0,
            pose: None,
        }
    }

    /// A 10 by 10 map of 10 m cells at height 5, each met, with `change`
    /// applied.
    fn report(
        change: impl Fn(&mut Vec<CellEvidence>),
        structures: &[bool],
        canopy: &[bool],
    ) -> MapReport {
        let requirement = requirement();
        let mut evidence = vec![cell(4, 5.0); 100];
        change(&mut evidence);
        build(ReportInput {
            nx: 10,
            ny: 10,
            origin_m: [1000.0, 2000.0],
            resolution_m: 10.0,
            tick: 7,
            timestamp_s: 1.75,
            requirement: &requirement,
            per_return_sigma_m: 0.2,
            finest_image_gsd_m: None,
            evidence,
            predicted: Some(PassPrediction {
                agl_m: 100.0,
                speed_mps: 20.0,
                swath_m: 120.0,
                density_per_m2: 0.02,
                vertical_sigma_m: 0.1,
                image_gsd_m: None,
            }),
            true_ground_m: vec![5.1; 100],
            structures,
            canopy,
        })
    }

    #[test]
    fn a_met_map_misses_nothing_and_says_how_honest_it_is() {
        let report = report(|_| {}, &[], &[]);
        assert_eq!(report.achieved.met_fraction, 1.0);
        assert!(report.missed.is_empty() && report.areas.is_empty());
        assert_eq!(report.achieved.passes_at_median, Some(2.0));
        // Four returns of 0.2 m claim 0.1 m; every cell is 0.1 m low.
        let c = report.calibration.unwrap();
        assert!((c.rms_error_m - 0.1).abs() < 1e-9);
        assert!((c.bias_m + 0.1).abs() < 1e-9);
        assert!((c.normalized_rms - 1.0).abs() < 1e-9);
        assert_eq!(c.within_2_sigma, 1.0);
    }

    #[test]
    fn what_is_missed_is_named_by_reason_and_by_what_is_there() {
        let mut structures = vec![false; 100];
        structures[44] = true;
        let mut canopy = vec![false; 100];
        canopy[77] = true;
        let report = report(
            |cells| {
                cells[45] = cell(0, 0.0); // beside the structure
                cells[77] = cell(0, 0.0); // under the crown
                cells[0] = cell(1, 5.0); // sparse, in the corner
                cells[72] = cell(1, 5.0); // sparse, in the open
            },
            &structures,
            &canopy,
        );
        let found: Vec<_> = report
            .missed
            .iter()
            .map(|m| (m.reason, m.cause, m.cells))
            .collect();
        assert_eq!(
            found,
            vec![
                (DeficitReason::Unseen, MissCause::Structure, 1),
                (DeficitReason::Unseen, MissCause::Canopy, 1),
                (DeficitReason::Sparse, MissCause::Edge, 1),
                (DeficitReason::Sparse, MissCause::Open, 1),
            ]
        );
        let missed: u64 = report.missed.iter().map(|m| m.cells).sum();
        assert_eq!(missed + report.achieved.counts.met, 100);
        assert!(report.areas.is_empty(), "single cells are not areas");
    }

    #[test]
    fn areas_are_connected_largest_first_and_placed_in_the_world() {
        let report = report(
            |cells| {
                for k in [22, 23, 33, 34, 44] {
                    cells[k] = cell(0, 0.0);
                }
                for k in [60, 61, 70, 71] {
                    cells[k] = cell(1, 5.0);
                }
            },
            &[],
            &[],
        );
        assert_eq!(report.areas.len(), 2);
        let first = &report.areas[0];
        assert_eq!((first.cells, first.reason), (5, DeficitReason::Unseen));
        assert_eq!(first.bounds_m, [1020.0, 2020.0, 1050.0, 2050.0]);
        let second = &report.areas[1];
        assert_eq!((second.cells, second.cause), (4, MissCause::Edge));
        assert_eq!(second.centre_m, [1010.0, 2070.0]);
    }

    #[test]
    fn an_overconfident_map_is_called_one() {
        let report = report(|cells| cells.fill(cell(400, 4.0)), &[], &[]);
        let c = report.calibration.unwrap();
        assert!(c.normalized_rms > 50.0, "{c:?}");
        assert_eq!(c.within_2_sigma, 0.0);
    }

    #[test]
    fn nothing_accepted_has_no_calibration_to_claim() {
        let report = report(|cells| cells.fill(cell(0, 0.0)), &[], &[]);
        assert_eq!(report.calibration, None);
        assert_eq!(report.achieved.density_per_m2, None);
        assert_eq!(report.achieved.met_fraction, 0.0);
        assert_eq!(report.areas[0].cells, 100);
    }
}
