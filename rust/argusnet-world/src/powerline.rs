//! Power lines (`powerline-v1`, R3 in `docs/WORLD_REALISM_PLAN.md`).
//!
//! A line is a row of supports (poles or towers) with conductors strung
//! between each neighbouring pair. A conductor hangs under its own weight:
//! a catenary, which for the sag of a real span (a few per cent of its
//! length) is a parabola to within a millimetre (0.9 mm on a 100 m span
//! sagging 3 m), so it is modelled as one. Each span's conductors are cut into straight pieces no
//! longer than [`MAX_PIECE_M`], and a piece with the conductor's radius is a
//! capsule.
//!
//! Wires are thin. A platform moving metres per tick passes a wire between
//! two ticks without either position being near it, so the clearance test is
//! **swept**: [`PowerLine::clearance_along`] takes the segment flown between
//! two ticks, not a point.
//!
//! This module is the model alone. Nothing in the simulation reads it yet:
//! collision, clearance, planning, sensing and drawing come in later slices.

use serde::{Deserialize, Serialize};

use crate::obstacles::Bounds2d;

/// The kind tag a power line carries in a world bundle.
pub const POWERLINE_KIND: &str = "powerline-v1";

/// The longest straight piece a conductor is cut into, metres. A piece of
/// length `h` departs from the parabola by at most `sag_ratio * h² / span`:
/// 7.5 mm for a 3 % sag on a 100 m span, less on longer ones.
pub const MAX_PIECE_M: f64 = 5.0;

/// A pole or tower, standing on the ground at `xy_m`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Support {
    pub xy_m: [f64; 2],
    /// Ground level at the support, metres above datum.
    pub base_z_m: f64,
    /// Height of the top cross-arm above `base_z_m`, where the conductors
    /// are attached.
    pub height_m: f64,
}

impl Support {
    pub fn top_z_m(&self) -> f64 {
        self.base_z_m + self.height_m
    }
}

/// A power line: supports in order, and the conductors strung between them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PowerLine {
    pub id: String,
    pub supports: Vec<Support>,
    /// Where each conductor hangs across the cross-arm, metres from the
    /// line's centre, to the left of the direction of travel (negative is
    /// to the right). One entry per conductor; `[0.0]` is a single wire.
    pub conductor_offsets_m: Vec<f64>,
    /// Sag at mid-span as a fraction of the span's horizontal length.
    pub sag_ratio: f64,
    /// Conductor radius, metres.
    pub conductor_radius_m: f64,
    /// How far a platform must stay from a conductor's surface, metres. Set
    /// from the line's voltage by whoever builds the line.
    pub clearance_m: f64,
}

/// Why a [`PowerLine`] cannot be used.
#[derive(Debug, Clone, PartialEq)]
pub enum PowerLineError {
    TooFewSupports {
        id: String,
        supports: usize,
    },
    NoConductors {
        id: String,
    },
    /// A value that must be finite and not negative (or, for a support's
    /// height, positive) is not.
    BadValue {
        id: String,
        field: &'static str,
    },
    /// Two neighbouring supports at the same place: no span between them.
    ZeroSpan {
        id: String,
        span: usize,
    },
}

impl std::fmt::Display for PowerLineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooFewSupports { id, supports } => {
                write!(f, "{id}: a power line needs two supports, not {supports}")
            }
            Self::NoConductors { id } => write!(f, "{id}: a power line needs a conductor"),
            Self::BadValue { id, field } => write!(f, "{id}: {field} is out of range"),
            Self::ZeroSpan { id, span } => {
                write!(f, "{id}: span {span} joins two supports at the same place")
            }
        }
    }
}

impl std::error::Error for PowerLineError {}

/// One straight piece of one conductor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductorPiece {
    pub start_m: [f64; 3],
    pub end_m: [f64; 3],
}

impl PowerLine {
    /// The line, checked: at least two supports and one conductor, every
    /// value finite, heights positive, sag, radius and clearance not
    /// negative, and no span of zero length.
    pub fn validated(self) -> Result<Self, PowerLineError> {
        let id = self.id.clone();
        let bad = |field| PowerLineError::BadValue {
            id: id.clone(),
            field,
        };
        if self.supports.len() < 2 {
            return Err(PowerLineError::TooFewSupports {
                id,
                supports: self.supports.len(),
            });
        }
        if self.conductor_offsets_m.is_empty() {
            return Err(PowerLineError::NoConductors { id });
        }
        let non_negative = |v: f64| v.is_finite() && v >= 0.0;
        if !non_negative(self.sag_ratio) {
            return Err(bad("sag_ratio"));
        }
        if !non_negative(self.conductor_radius_m) {
            return Err(bad("conductor_radius_m"));
        }
        if !non_negative(self.clearance_m) {
            return Err(bad("clearance_m"));
        }
        if self.conductor_offsets_m.iter().any(|o| !o.is_finite()) {
            return Err(bad("conductor_offsets_m"));
        }
        for support in &self.supports {
            if !(support.xy_m.iter().all(|v| v.is_finite()) && support.base_z_m.is_finite()) {
                return Err(bad("supports"));
            }
            if !(support.height_m.is_finite() && support.height_m > 0.0) {
                return Err(bad("height_m"));
            }
        }
        for (span, pair) in self.supports.windows(2).enumerate() {
            if horizontal(pair[0].xy_m, pair[1].xy_m) <= 0.0 {
                return Err(PowerLineError::ZeroSpan { id, span });
            }
        }
        Ok(self)
    }

    /// Where conductor `offset_m` hangs at fraction `t` (0 at `from`, 1 at
    /// `to`) of the span between two supports.
    fn conductor_at(&self, from: &Support, to: &Support, offset_m: f64, t: f64) -> [f64; 3] {
        let length = horizontal(from.xy_m, to.xy_m);
        let along = [
            (to.xy_m[0] - from.xy_m[0]) / length,
            (to.xy_m[1] - from.xy_m[1]) / length,
        ];
        // Left of the direction of travel.
        let left = [-along[1], along[0]];
        let sag_m = self.sag_ratio * length;
        let chord_z = from.top_z_m() + (to.top_z_m() - from.top_z_m()) * t;
        [
            from.xy_m[0] + (to.xy_m[0] - from.xy_m[0]) * t + left[0] * offset_m,
            from.xy_m[1] + (to.xy_m[1] - from.xy_m[1]) * t + left[1] * offset_m,
            chord_z - 4.0 * sag_m * t * (1.0 - t),
        ]
    }

    /// Every straight piece of every conductor, span by span.
    pub fn pieces(&self) -> Vec<ConductorPiece> {
        let mut pieces = Vec::new();
        for pair in self.supports.windows(2) {
            let (from, to) = (&pair[0], &pair[1]);
            let count = (horizontal(from.xy_m, to.xy_m) / MAX_PIECE_M)
                .ceil()
                .max(1.0) as usize;
            for &offset_m in &self.conductor_offsets_m {
                let mut start = self.conductor_at(from, to, offset_m, 0.0);
                for k in 1..=count {
                    let end = self.conductor_at(from, to, offset_m, k as f64 / count as f64);
                    pieces.push(ConductorPiece {
                        start_m: start,
                        end_m: end,
                    });
                    start = end;
                }
            }
        }
        pieces
    }

    /// The lowest point of any conductor, metres above datum.
    pub fn lowest_conductor_z_m(&self) -> f64 {
        self.pieces()
            .iter()
            .flat_map(|piece| [piece.start_m[2], piece.end_m[2]])
            .fold(f64::INFINITY, f64::min)
    }

    /// The ground the line and its clearance cover, from above.
    pub fn bounds_xy_m(&self) -> Bounds2d {
        let reach = self
            .conductor_offsets_m
            .iter()
            .fold(0.0_f64, |m, o| m.max(o.abs()))
            + self.conductor_radius_m
            + self.clearance_m;
        let (mut x_min, mut x_max) = (f64::INFINITY, f64::NEG_INFINITY);
        let (mut y_min, mut y_max) = (f64::INFINITY, f64::NEG_INFINITY);
        for support in &self.supports {
            x_min = x_min.min(support.xy_m[0]);
            x_max = x_max.max(support.xy_m[0]);
            y_min = y_min.min(support.xy_m[1]);
            y_max = y_max.max(support.xy_m[1]);
        }
        Bounds2d::new(x_min - reach, x_max + reach, y_min - reach, y_max + reach)
    }

    /// How far the segment from `from` to `to` (the flight between two
    /// ticks) comes to any conductor's surface, metres: negative when it
    /// passes through a wire. Swept, so a wire crossed between the ends
    /// counts.
    pub fn clearance_along(&self, from: [f64; 3], to: [f64; 3]) -> f64 {
        self.pieces()
            .iter()
            .map(|piece| segment_distance(from, to, piece.start_m, piece.end_m))
            .fold(f64::INFINITY, f64::min)
            - self.conductor_radius_m
    }

    /// Whether the flight from `from` to `to` comes closer to a conductor
    /// than the line's clearance.
    pub fn violated_by(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        self.clearance_along(from, to) < self.clearance_m
    }
}

fn horizontal(a: [f64; 2], b: [f64; 2]) -> f64 {
    (b[0] - a[0]).hypot(b[1] - a[1])
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// The shortest distance between segments `p0`–`p1` and `q0`–`q1`, either
/// of which may be a point (Ericson, *Real-Time Collision Detection* 5.1.9).
pub fn segment_distance(p0: [f64; 3], p1: [f64; 3], q0: [f64; 3], q1: [f64; 3]) -> f64 {
    const EPS: f64 = 1e-12;
    let d1 = sub(p1, p0);
    let d2 = sub(q1, q0);
    let r = sub(p0, q0);
    let a = dot(d1, d1);
    let e = dot(d2, d2);
    let f = dot(d2, r);
    let (s, t) = if a <= EPS && e <= EPS {
        (0.0, 0.0)
    } else if a <= EPS {
        (0.0, (f / e).clamp(0.0, 1.0))
    } else {
        let c = dot(d1, r);
        if e <= EPS {
            ((-c / a).clamp(0.0, 1.0), 0.0)
        } else {
            let b = dot(d1, d2);
            let denom = a * e - b * b;
            // Parallel segments: any s will do; take the start.
            let mut s = if denom > EPS {
                ((b * f - c * e) / denom).clamp(0.0, 1.0)
            } else {
                0.0
            };
            let mut t = (b * s + f) / e;
            if t < 0.0 {
                t = 0.0;
                s = (-c / a).clamp(0.0, 1.0);
            } else if t > 1.0 {
                t = 1.0;
                s = ((b - c) / a).clamp(0.0, 1.0);
            }
            (s, t)
        }
    };
    let on_p = [p0[0] + d1[0] * s, p0[1] + d1[1] * s, p0[2] + d1[2] * s];
    let on_q = [q0[0] + d2[0] * t, q0[1] + d2[1] * t, q0[2] + d2[2] * t];
    let gap = sub(on_p, on_q);
    dot(gap, gap).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two 20 m poles 100 m apart along x, one wire, 3 % sag (3 m), 1 cm
    /// radius, 5 m clearance.
    fn line() -> PowerLine {
        PowerLine {
            id: "line-a".into(),
            supports: vec![
                Support {
                    xy_m: [0.0, 0.0],
                    base_z_m: 0.0,
                    height_m: 20.0,
                },
                Support {
                    xy_m: [100.0, 0.0],
                    base_z_m: 0.0,
                    height_m: 20.0,
                },
            ],
            conductor_offsets_m: vec![0.0],
            sag_ratio: 0.03,
            conductor_radius_m: 0.01,
            clearance_m: 5.0,
        }
        .validated()
        .expect("valid")
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn a_span_sags_by_its_ratio_at_mid_span_and_meets_its_supports() {
        let line = line();
        let pieces = line.pieces();
        assert_eq!(pieces.len(), 20, "100 m in 5 m pieces");
        assert_eq!(pieces[0].start_m, [0.0, 0.0, 20.0]);
        assert_eq!(pieces[19].end_m, [100.0, 0.0, 20.0]);
        // Mid-span is the end of the tenth piece: 3 m below the attachments.
        assert!(close(pieces[9].end_m[2], 17.0), "{:?}", pieces[9]);
        assert!(close(line.lowest_conductor_z_m(), 17.0));
        // Pieces join up.
        for pair in pieces.windows(2) {
            assert_eq!(pair[0].end_m, pair[1].start_m);
        }
    }

    #[test]
    fn conductors_hang_to_the_left_and_right_of_the_line() {
        let mut line = line();
        line.conductor_offsets_m = vec![-2.0, 2.0];
        let pieces = line.pieces();
        assert_eq!(pieces.len(), 40);
        // Travelling +x, left is +y.
        assert_eq!(pieces[0].start_m[1], -2.0);
        assert_eq!(pieces[20].start_m[1], 2.0);
        let bounds = line.bounds_xy_m();
        assert!(close(bounds.y_max_m, 2.0 + 0.01 + 5.0));
        assert!(close(bounds.x_min_m, -(2.0 + 0.01 + 5.0)));
    }

    /// The reason the test is swept: a drone moving 28 m in a tick crosses
    /// the wire between two positions that are each 14 m from it.
    #[test]
    fn a_wire_crossed_between_two_ticks_is_caught() {
        let line = line();
        let (from, to) = ([50.0, -14.0, 17.0], [50.0, 14.0, 17.0]);
        // Each end alone is clear by far.
        assert!(!line.violated_by(from, from));
        assert!(!line.violated_by(to, to));
        // The flight between them goes through the wire.
        assert!(line.clearance_along(from, to) < 0.0);
        assert!(line.violated_by(from, to));
    }

    #[test]
    fn a_flight_over_the_line_is_clear_only_above_the_clearance() {
        let line = line();
        // Over the attachment height, the sag gives more room at mid-span.
        let over = |z: f64| line.clearance_along([50.0, -30.0, z], [50.0, 30.0, z]);
        // 8 m over the lowest point, less the radius; the straight pieces
        // either side of mid-span tilt toward the flight by a fraction of a
        // millimetre, which is the pieces' own approximation.
        assert!(
            (over(25.0) - (25.0 - 17.0 - 0.01)).abs() < 1e-3,
            "{}",
            over(25.0)
        );
        assert!(!line.violated_by([50.0, -30.0, 25.0], [50.0, 30.0, 25.0]));
        assert!(line.violated_by([50.0, -30.0, 21.0], [50.0, 30.0, 21.0]));
        // Near a support the wire is higher.
        assert!(line.violated_by([1.0, -30.0, 24.0], [1.0, 30.0, 24.0]));
    }

    #[test]
    fn segment_distance_covers_crossing_parallel_and_point_cases() {
        // Crossing at right angles, 2 m apart vertically.
        let d = segment_distance(
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 2.0],
            [0.0, 1.0, 2.0],
        );
        assert!(close(d, 2.0));
        // Parallel, overlapping, 3 m apart.
        let d = segment_distance(
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [5.0, 3.0, 0.0],
            [15.0, 3.0, 0.0],
        );
        assert!(close(d, 3.0));
        // Collinear and apart: end to end.
        let d = segment_distance(
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [4.0, 0.0, 0.0],
            [6.0, 0.0, 0.0],
        );
        assert!(close(d, 3.0));
        // A point and a segment, and two points.
        let p = [5.0, 4.0, 0.0];
        assert!(close(
            segment_distance(p, p, [0.0, 0.0, 0.0], [10.0, 0.0, 0.0]),
            4.0
        ));
        assert!(close(
            segment_distance(p, p, [5.0, 0.0, 0.0], [5.0, 0.0, 0.0]),
            4.0
        ));
        // Past the end of the other segment: to its end point.
        let d = segment_distance(
            [20.0, -1.0, 0.0],
            [20.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
        );
        assert!(close(d, 10.0));
    }

    #[test]
    fn a_bad_line_is_refused_with_its_reason() {
        let mut one = line();
        one.supports.truncate(1);
        assert_eq!(
            one.validated(),
            Err(PowerLineError::TooFewSupports {
                id: "line-a".into(),
                supports: 1
            })
        );
        let mut bare = line();
        bare.conductor_offsets_m.clear();
        assert!(matches!(
            bare.validated(),
            Err(PowerLineError::NoConductors { .. })
        ));
        let mut stacked = line();
        stacked.supports[1].xy_m = [0.0, 0.0];
        assert!(matches!(
            stacked.validated(),
            Err(PowerLineError::ZeroSpan { span: 0, .. })
        ));
        for (field, breaks) in [
            (
                "sag_ratio",
                (|l: &mut PowerLine| l.sag_ratio = -0.1) as fn(&mut PowerLine),
            ),
            ("conductor_radius_m", |l| l.conductor_radius_m = f64::NAN),
            ("clearance_m", |l| l.clearance_m = f64::INFINITY),
            ("height_m", |l| l.supports[0].height_m = 0.0),
            ("conductor_offsets_m", |l| {
                l.conductor_offsets_m = vec![f64::NAN]
            }),
        ] {
            let mut broken = line();
            breaks(&mut broken);
            assert_eq!(
                broken.validated(),
                Err(PowerLineError::BadValue {
                    id: "line-a".into(),
                    field
                }),
                "{field}"
            );
        }
    }

    #[test]
    fn a_line_round_trips_through_json() {
        let line = line();
        let text = serde_json::to_string(&line).expect("serialises");
        let back: PowerLine = serde_json::from_str(&text).expect("parses");
        assert_eq!(back, line);
    }
}
