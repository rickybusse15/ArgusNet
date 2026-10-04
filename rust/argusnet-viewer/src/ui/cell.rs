//! The map-cell inspector (VIEWER_UI §7.3, C3.4): what the belief world says
//! about one ground location, as a pure function of the belief snapshot.
//! What the snapshot does not carry -- uncertainty, provenance, obstacle
//! probability, semantic hypotheses, the planner's and safety rationale --
//! reads as unavailable; an unobserved cell never reads as a confident zero.

use crate::belief_render::{BeliefClass, BeliefLayer};
use crate::theme::Field;

use super::platform::Row;

const NOT_IN_SNAPSHOT: &str = "not in the belief snapshot";

fn class_label(class: BeliefClass) -> Option<&'static str> {
    match class {
        BeliefClass::NoData => None,
        BeliefClass::OutsideGeofence => Some("outside geofence"),
        BeliefClass::Unknown => Some("unknown"),
        BeliefClass::KnownObstacle => Some("known obstacle"),
        BeliefClass::Uncertain => Some("uncertain"),
        BeliefClass::KnownSafe => Some("known safe"),
    }
}

/// §7.3's fields for the ground point `(x_m, y_m)`.
pub fn inspect_cell(belief: &BeliefLayer, x_m: f64, y_m: f64) -> Vec<Row> {
    let mut rows = vec![Row::new(
        "Location",
        Field::Value(format!("{x_m:.1}, {y_m:.1} m")),
    )];
    let snapshot = match belief {
        BeliefLayer::Ready(snapshot) => snapshot,
        BeliefLayer::Unavailable(_) => {
            for label in [
                "Classification",
                "Coverage",
                "Last observation",
                "Confidence",
            ] {
                rows.push(Row::unavailable(label, "no belief data (see Layers)"));
            }
            return rows;
        }
    };
    let Some((i, j)) = snapshot.cell_at(x_m, y_m) else {
        for label in [
            "Classification",
            "Coverage",
            "Last observation",
            "Confidence",
        ] {
            rows.push(Row::unavailable(label, "outside the belief grid"));
        }
        return rows;
    };
    let observed = snapshot.observed_ij(i, j);
    let sample = snapshot.sample(x_m, y_m);
    rows.push(Row::new("Cell", Field::Value(format!("{i}, {j}"))));
    rows.push(Row::new(
        "Classification",
        class_label(snapshot.class_ij(i, j)).map_or(Field::Unavailable("no class"), |label| {
            Field::Value(label.into())
        }),
    ));
    rows.push(Row::new(
        "Coverage",
        Field::Value(
            if observed {
                "observed"
            } else {
                "never observed"
            }
            .into(),
        ),
    ));
    rows.push(Row::new(
        "Last observation",
        sample
            .last_observed_s
            .map_or(Field::Unavailable("never observed"), |t| {
                Field::Value(format!("t = {t:.1} s"))
            }),
    ));
    rows.push(Row::new(
        "Confidence",
        if observed {
            // As the terrain draws it: interpolated among observed cells.
            Field::Value(format!("{:.2}", sample.confidence))
        } else {
            Field::Unavailable("never observed")
        },
    ));
    rows.push(Row::new(
        "Belief revision",
        Field::Value(snapshot.revision().0.to_string()),
    ));
    rows.push(Row::unavailable("Uncertainty", NOT_IN_SNAPSHOT));
    rows.push(Row::unavailable("Provenance", NOT_IN_SNAPSHOT));
    rows.push(Row::unavailable("Obstacle probability", NOT_IN_SNAPSHOT));
    rows.push(Row::unavailable("Semantic hypotheses", NOT_IN_SNAPSHOT));
    rows.push(Row::unavailable(
        "Classification rationale",
        "planner and safety rationale are not recorded",
    ));
    rows
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_render::{BeliefRenderSnapshot, BeliefUnavailable};

    fn value<'a>(rows: &'a [Row], label: &str) -> &'a Field<String> {
        &rows
            .iter()
            .find(|row| row.label == label)
            .unwrap_or_else(|| panic!("no row {label}"))
            .value
    }

    fn ready() -> BeliefLayer {
        BeliefLayer::Ready(Box::new(
            BeliefRenderSnapshot::from_store(&crate::belief_render::synthetic_belief_store())
                .unwrap(),
        ))
    }

    #[test]
    fn without_belief_every_belief_field_is_unavailable() {
        let rows = inspect_cell(
            &BeliefLayer::Unavailable(BeliefUnavailable::NoBeliefData),
            10.0,
            20.0,
        );
        assert_eq!(
            value(&rows, "Location"),
            &Field::Value("10.0, 20.0 m".into())
        );
        for label in [
            "Classification",
            "Coverage",
            "Last observation",
            "Confidence",
        ] {
            assert!(
                matches!(value(&rows, label), Field::Unavailable(_)),
                "{label}"
            );
        }
    }

    #[test]
    fn outside_the_grid_says_so() {
        let rows = inspect_cell(&ready(), -1.0e7, -1.0e7);
        assert_eq!(
            value(&rows, "Coverage"),
            &Field::Unavailable("outside the belief grid")
        );
    }

    /// Every cell of the synthetic store: an observed cell shows its class,
    /// time and confidence; an unobserved one never shows a confidence.
    #[test]
    fn observed_and_unobserved_cells_read_differently() {
        let layer = ready();
        let BeliefLayer::Ready(snapshot) = &layer else {
            unreachable!()
        };
        let b = snapshot.bounds();
        let (mut seen_observed, mut seen_unobserved) = (false, false);
        let mut y = b.y_min_m + b.resolution_m / 2.0;
        while y < b.y_max_m {
            let mut x = b.x_min_m + b.resolution_m / 2.0;
            while x < b.x_max_m {
                let rows = inspect_cell(&layer, x, y);
                let (i, j) = snapshot.cell_at(x, y).expect("inside");
                if snapshot.observed_ij(i, j) {
                    seen_observed = true;
                    assert_eq!(value(&rows, "Coverage"), &Field::Value("observed".into()));
                    assert!(matches!(value(&rows, "Confidence"), Field::Value(_)));
                } else {
                    seen_unobserved = true;
                    assert_eq!(
                        value(&rows, "Confidence"),
                        &Field::Unavailable("never observed"),
                        "an unobserved cell is not a confident zero"
                    );
                }
                for label in ["Uncertainty", "Provenance", "Obstacle probability"] {
                    assert!(matches!(value(&rows, label), Field::Unavailable(_)));
                }
                x += b.resolution_m;
            }
            y += b.resolution_m;
        }
        assert!(seen_observed && seen_unobserved, "the fixture covers both");
    }
}
