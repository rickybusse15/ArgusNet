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
//! This module is the model. The safety engine judges flights against it
//! (`structure_clearance`), the mission planner's routes keep its clearance
//! (efficient_v1, over the wires or round a line's ends), the physical guard
//! keeps platforms off the wires ([`PowerLine::envelope`], each piece a
//! capsule met by [`capsule_entry`]) and the viewer draws it (supports at
//! [`PowerLine::cross_arm`], conductors from [`PowerLine::pieces`]); the
//! range scanner reports a span ([`PowerLine::span_line`]) when it is near.

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
/// Read only through [`PowerLine::validated`], so a line in a bundle that
/// the model refuses is an error, never a line that silently clears nothing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(try_from = "UncheckedPowerLine")]
pub struct PowerLine {
    pub id: String,
    pub supports: Vec<Support>,
    /// Where each conductor hangs across the cross-arm, metres from the
    /// line's centre, to the left of the direction of travel (negative is
    /// to the right), along a cross-arm that bisects the turn at a support
    /// between two spans. One entry per conductor; `[0.0]` is a single wire.
    pub conductor_offsets_m: Vec<f64>,
    /// Sag at mid-span as a fraction of the span's horizontal length.
    pub sag_ratio: f64,
    /// Conductor radius, metres.
    pub conductor_radius_m: f64,
    /// How far a platform must stay from a conductor's surface, metres. Set
    /// from the line's voltage by whoever builds the line.
    pub clearance_m: f64,
}

/// A [`PowerLine`] as written, before it is checked.
#[derive(Deserialize)]
struct UncheckedPowerLine {
    id: String,
    supports: Vec<Support>,
    conductor_offsets_m: Vec<f64>,
    sag_ratio: f64,
    conductor_radius_m: f64,
    clearance_m: f64,
}

impl TryFrom<UncheckedPowerLine> for PowerLine {
    type Error = PowerLineError;

    fn try_from(line: UncheckedPowerLine) -> Result<Self, Self::Error> {
        PowerLine {
            id: line.id,
            supports: line.supports,
            conductor_offsets_m: line.conductor_offsets_m,
            sag_ratio: line.sag_ratio,
            conductor_radius_m: line.conductor_radius_m,
            clearance_m: line.clearance_m,
        }
        .validated()
    }
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
    /// How far the conductor can hang below this straight piece between its
    /// ends, metres: the parabola sags under its chord by at most
    /// `sag_ratio * h² / span` for a piece of length `h`.
    pub sag_error_m: f64,
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

    /// Which way the cross-arm at support `index` points: square to its
    /// span at an end support, and along the bisector of its two spans at a
    /// turn, so the conductors of both spans meet on it. Left of the
    /// direction of travel: a conductor at offset `o` is attached at the
    /// support's `xy_m + o * cross_arm`.
    pub fn cross_arm(&self, index: usize) -> [f64; 2] {
        let left_of = |span: usize| {
            let (from, to) = (&self.supports[span], &self.supports[span + 1]);
            let length = horizontal(from.xy_m, to.xy_m);
            [
                -(to.xy_m[1] - from.xy_m[1]) / length,
                (to.xy_m[0] - from.xy_m[0]) / length,
            ]
        };
        let last = self.supports.len() - 1;
        if index == 0 {
            return left_of(0);
        }
        if index == last {
            return left_of(last - 1);
        }
        let (a, b) = (left_of(index - 1), left_of(index));
        let sum = [a[0] + b[0], a[1] + b[1]];
        let norm = sum[0].hypot(sum[1]);
        // A line that doubles straight back has no bisector: keep the
        // incoming span's arm.
        if norm < 1e-9 {
            a
        } else {
            [sum[0] / norm, sum[1] / norm]
        }
    }

    /// How far the lowest conductor of span `span` (from support `span` to
    /// the next) hangs above `terrain`, metres: solved, not sampled. Along a
    /// conductor the ground is a quadratic in each terrain cell it crosses
    /// ([`TiledTerrain::height_along`]) and the sagging wire a quadratic over
    /// the whole span, so their difference is solved for its lowest point in
    /// each cell. Negative when a wire is under the ground.
    pub fn lowest_over_ground(&self, span: usize, terrain: &crate::tiled::TiledTerrain) -> f64 {
        let mut lowest = f64::INFINITY;
        for &offset_m in &self.conductor_offsets_m {
            let (a, b) = (
                self.conductor_at(span, offset_m, 0.0),
                self.conductor_at(span, offset_m, 1.0),
            );
            for piece in terrain.height_along([a[0], a[1]], [b[0], b[1]]) {
                // The wire over the piece, as `c0 + c1·s + c2·s²` in the
                // piece's own 0..1, from three points of it.
                let wire = |s: f64| {
                    self.conductor_at(span, offset_m, piece.t0 + (piece.t1 - piece.t0) * s)[2]
                };
                let (w0, wm, w1) = (wire(0.0), wire(0.5), wire(1.0));
                let w2 = 2.0 * (w0 - 2.0 * wm + w1);
                let wire = [w0, w1 - w0 - w2, w2];
                let [g0, g1, g2] = piece.coefficients;
                // Above the piece's quadratic, and above the floor it is
                // raised to.
                for ground in [[g0, g1, g2], [terrain.floor_m, 0.0, 0.0]] {
                    let q = [
                        wire[0] - ground[0],
                        wire[1] - ground[1],
                        wire[2] - ground[2],
                    ];
                    let at = |s: f64| q[0] + (q[1] + q[2] * s) * s;
                    let mut low = at(0.0).min(at(1.0));
                    if q[2] > 0.0 {
                        let vertex = -q[1] / (2.0 * q[2]);
                        if vertex > 0.0 && vertex < 1.0 {
                            low = low.min(at(vertex));
                        }
                    }
                    lowest = lowest.min(low);
                }
            }
        }
        lowest
    }

    /// Where conductor `offset_m` hangs at fraction `t` (0 at support
    /// `span`, 1 at the next) of that span: between its attachments on the
    /// two cross-arms, sagging below their chord.
    fn conductor_at(&self, span: usize, offset_m: f64, t: f64) -> [f64; 3] {
        let (from, to) = (&self.supports[span], &self.supports[span + 1]);
        let (arm_from, arm_to) = (self.cross_arm(span), self.cross_arm(span + 1));
        let start = [
            from.xy_m[0] + arm_from[0] * offset_m,
            from.xy_m[1] + arm_from[1] * offset_m,
        ];
        let end = [
            to.xy_m[0] + arm_to[0] * offset_m,
            to.xy_m[1] + arm_to[1] * offset_m,
        ];
        let sag_m = self.sag_ratio * horizontal(from.xy_m, to.xy_m);
        let chord_z = from.top_z_m() + (to.top_z_m() - from.top_z_m()) * t;
        [
            start[0] + (end[0] - start[0]) * t,
            start[1] + (end[1] - start[1]) * t,
            chord_z - 4.0 * sag_m * t * (1.0 - t),
        ]
    }

    /// Every straight piece of every conductor, span by span.
    pub fn pieces(&self) -> Vec<ConductorPiece> {
        let mut pieces = Vec::new();
        for (span, pair) in self.supports.windows(2).enumerate() {
            let (from, to) = (&pair[0], &pair[1]);
            let span_m = horizontal(from.xy_m, to.xy_m);
            let count = (span_m / MAX_PIECE_M).ceil().max(1.0) as usize;
            // The sag is a parabola in the span's fraction, so a piece over
            // 1/count of it departs from its chord by at most sag/count²,
            // whatever the conductor's own length.
            let sag_error_m = self.sag_ratio * span_m / (count * count) as f64;
            for &offset_m in &self.conductor_offsets_m {
                let mut start = self.conductor_at(span, offset_m, 0.0);
                for k in 1..=count {
                    let end = self.conductor_at(span, offset_m, k as f64 / count as f64);
                    pieces.push(ConductorPiece {
                        start_m: start,
                        end_m: end,
                        sag_error_m,
                    });
                    start = end;
                }
            }
        }
        pieces
    }

    /// [`Self::pieces`], span by span: every conductor's pieces of span
    /// `k` at index `k`.
    pub fn pieces_by_span(&self) -> Vec<Vec<ConductorPiece>> {
        let mut spans = vec![Vec::new(); self.supports.len() - 1];
        let mut pieces = self.pieces().into_iter();
        for (span, pair) in self.supports.windows(2).enumerate() {
            let count = (horizontal(pair[0].xy_m, pair[1].xy_m) / MAX_PIECE_M)
                .ceil()
                .max(1.0) as usize;
            spans[span].extend(pieces.by_ref().take(count * self.conductor_offsets_m.len()));
        }
        spans
    }

    /// The span from support `span` to the next, alone: a line of those two
    /// supports, with this line's conductors, sag, radius and clearance.
    /// Its cross-arms are square to the span, so at a support where this
    /// line turns its outer conductors sit a little off this line's (the
    /// offset times one minus the cosine of half the turn).
    pub fn span_line(&self, span: usize) -> PowerLine {
        PowerLine {
            id: format!("{}#{span}", self.id),
            supports: self.supports[span..=span + 1].to_vec(),
            conductor_offsets_m: self.conductor_offsets_m.clone(),
            sag_ratio: self.sag_ratio,
            conductor_radius_m: self.conductor_radius_m,
            clearance_m: self.clearance_m,
        }
    }

    /// The lowest point of any conductor, metres above datum: each span's
    /// parabola at its vertex (or at a support, when the vertex lies beyond
    /// one), not just where the pieces meet.
    pub fn lowest_conductor_z_m(&self) -> f64 {
        self.supports
            .windows(2)
            .map(|pair| {
                let (from, to) = (&pair[0], &pair[1]);
                let sag_m = self.sag_ratio * horizontal(from.xy_m, to.xy_m);
                let rise_m = to.top_z_m() - from.top_z_m();
                // z(t) = z0 + rise·t − 4·sag·t(1 − t), lowest where
                // dz/dt = rise − 4·sag + 8·sag·t is zero.
                let t = if sag_m > 0.0 {
                    ((4.0 * sag_m - rise_m) / (8.0 * sag_m)).clamp(0.0, 1.0)
                } else if rise_m >= 0.0 {
                    0.0
                } else {
                    1.0
                };
                from.top_z_m() + rise_m * t - 4.0 * sag_m * t * (1.0 - t)
            })
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
    /// counts. Never more than the true clearance: each piece's distance is
    /// cut by how far the wire can sag below it, so the straight pieces never
    /// pass off a flight just under the wire as clear.
    pub fn clearance_along(&self, from: [f64; 3], to: [f64; 3]) -> f64 {
        self.pieces()
            .iter()
            .map(|piece| segment_distance(from, to, piece.start_m, piece.end_m) - piece.sag_error_m)
            .fold(f64::INFINITY, f64::min)
            - self.conductor_radius_m
    }

    /// Whether the flight from `from` to `to` comes closer to a conductor
    /// than the line's clearance.
    pub fn violated_by(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        self.clearance_along(from, to) < self.clearance_m
    }

    /// The space within `margin_m` of a conductor's surface, as capsules
    /// cut once for repeated queries (the physical guard asks every tick).
    pub fn envelope(&self, margin_m: f64) -> ConductorEnvelope {
        ConductorEnvelope::new(self.pieces(), self.conductor_radius_m + margin_m)
    }
}

/// How far past an envelope's surface a push puts a point, metres, so that
/// rounding never leaves it a hair inside.
const PUSH_PAST_M: f64 = 1e-6;

/// The space within a margin of a line's conductors ([`PowerLine::envelope`]):
/// a capsule round each straight piece, of the conductor's radius plus the
/// margin plus the piece's sag error. The true wire hangs within its sag error
/// of the piece, so the capsules hold everything within the margin of it:
/// conservative by at most the sag error (millimetres), never short.
#[derive(Debug, Clone, PartialEq)]
pub struct ConductorEnvelope {
    pieces: Vec<ConductorPiece>,
    /// The conductor's radius plus the margin; each capsule adds its own
    /// piece's sag error.
    radius_m: f64,
    /// Corners of a box holding every capsule.
    low_m: [f64; 3],
    high_m: [f64; 3],
}

impl ConductorEnvelope {
    fn new(pieces: Vec<ConductorPiece>, radius_m: f64) -> Self {
        let mut low_m = [f64::INFINITY; 3];
        let mut high_m = [f64::NEG_INFINITY; 3];
        for piece in &pieces {
            let reach = radius_m + piece.sag_error_m;
            for axis in 0..3 {
                low_m[axis] = low_m[axis].min(piece.start_m[axis].min(piece.end_m[axis]) - reach);
                high_m[axis] = high_m[axis].max(piece.start_m[axis].max(piece.end_m[axis]) + reach);
            }
        }
        Self {
            pieces,
            radius_m,
            low_m,
            high_m,
        }
    }

    fn capsule_radius(&self, piece: &ConductorPiece) -> f64 {
        self.radius_m + piece.sag_error_m
    }

    /// Whether the box round `from`–`to` meets the envelope's box.
    fn near(&self, from: [f64; 3], to: [f64; 3]) -> bool {
        (0..3).all(|axis| {
            from[axis].max(to[axis]) >= self.low_m[axis]
                && from[axis].min(to[axis]) <= self.high_m[axis]
        })
    }

    /// Whether `point` is inside the envelope.
    pub fn contains(&self, point: [f64; 3]) -> bool {
        self.near(point, point)
            && self.pieces.iter().any(|piece| {
                segment_distance(point, point, piece.start_m, piece.end_m)
                    < self.capsule_radius(piece)
            })
    }

    /// Where the segment from `from` to `to` first enters the envelope, as a
    /// fraction of its length: solved against each piece's capsule
    /// ([`capsule_entry`]), so a wire passed between two ends far from it is
    /// still met. `Some(0.0)` when `from` is already inside; `None` when the
    /// segment never enters.
    pub fn first_entry(&self, from: [f64; 3], to: [f64; 3]) -> Option<f64> {
        if !self.near(from, to) {
            return None;
        }
        self.pieces
            .iter()
            .filter_map(|piece| {
                capsule_entry(
                    from,
                    to,
                    piece.start_m,
                    piece.end_m,
                    self.capsule_radius(piece),
                )
            })
            .min_by(f64::total_cmp)
    }

    /// `point` pushed straight out of the capsule it is deepest in, away
    /// from the nearest point of that piece, to just past its surface; up
    /// (square to the piece) when it is on the piece itself. `None` when it
    /// is in no capsule. The push can land in a neighbouring capsule, where
    /// two pieces meet at an angle: push again.
    pub fn push_out(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        if !self.near(point, point) {
            return None;
        }
        let (piece, nearest, _) = self
            .pieces
            .iter()
            .map(|piece| {
                let nearest = closest_on_segment(point, piece.start_m, piece.end_m);
                let gap = sub(point, nearest);
                (
                    piece,
                    nearest,
                    self.capsule_radius(piece) - dot(gap, gap).sqrt(),
                )
            })
            .filter(|&(_, _, depth)| depth > 0.0)
            .max_by(|a, b| a.2.total_cmp(&b.2))?;
        let mut away = sub(point, nearest);
        let mut length = dot(away, away).sqrt();
        if length < 1e-9 {
            // On the wire: up, less whatever of up lies along the piece.
            let along = sub(piece.end_m, piece.start_m);
            let along_sq = dot(along, along);
            let k = if along_sq > 0.0 {
                along[2] / along_sq
            } else {
                0.0
            };
            away = [-along[0] * k, -along[1] * k, 1.0 - along[2] * k];
            length = dot(away, away).sqrt();
            if length < 1e-9 {
                // A vertical piece: any square direction will do.
                away = [1.0, 0.0, 0.0];
                length = 1.0;
            }
        }
        let reach = self.capsule_radius(piece) + PUSH_PAST_M;
        Some([
            nearest[0] + away[0] / length * reach,
            nearest[1] + away[1] / length * reach,
            nearest[2] + away[2] / length * reach,
        ])
    }

    /// `point` raised straight up to the lowest height above it that is
    /// outside the envelope: out of every capsule it is in, and out of any
    /// that raises it into, solved on each capsule. `None` when it is in
    /// none.
    pub fn lift_out(&self, point: [f64; 3]) -> Option<[f64; 3]> {
        let mut lifted = point;
        let mut moved = false;
        // Each pass leaves every capsule it was in; at most every piece.
        for _ in 0..=self.pieces.len() {
            let mut top = lifted[2];
            for piece in &self.pieces {
                let radius = self.capsule_radius(piece);
                if segment_distance(lifted, lifted, piece.start_m, piece.end_m) >= radius {
                    continue;
                }
                // Down from above the capsule, its first surface is the
                // top of it over this point.
                let above = piece.start_m[2].max(piece.end_m[2]) + radius + 1.0;
                let from = [lifted[0], lifted[1], above];
                if let Some(t) = capsule_entry(from, lifted, piece.start_m, piece.end_m, radius) {
                    top = top.max(above - t * (above - lifted[2]));
                }
            }
            if top <= lifted[2] {
                break;
            }
            lifted[2] = top + PUSH_PAST_M;
            moved = true;
        }
        moved.then_some(lifted)
    }
}

/// The point of segment `a`–`b` nearest `p`.
pub fn closest_on_segment(p: [f64; 3], a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let length_sq = dot(ab, ab);
    let t = if length_sq > 0.0 {
        (dot(sub(p, a), ab) / length_sq).clamp(0.0, 1.0)
    } else {
        0.0
    };
    [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t]
}

/// Where the segment from `from` to `to` first comes within `radius` of the
/// segment `a`–`b`, as a fraction of its length: its entry into the capsule
/// round `a`–`b`, solved, not sampled. `Some(0.0)` when `from` is already
/// within; `None` when it never comes within.
///
/// A capsule is a cylinder round `a`–`b` capped by a sphere at each end, and
/// the first entry into it is the earliest of: entering the cylinder's side
/// between the ends, or entering either sphere (a flat cap lies inside its
/// sphere, so the sphere is entered first). Each is a quadratic in the
/// distance flown. A flight within 1e-12 of parallel to the piece is taken
/// to enter only through the ends: it misses the side by less than the
/// piece's length times that angle (micrometres).
pub fn capsule_entry(
    from: [f64; 3],
    to: [f64; 3],
    a: [f64; 3],
    b: [f64; 3],
    radius: f64,
) -> Option<f64> {
    if segment_distance(from, from, a, b) < radius {
        return Some(0.0);
    }
    let flown = sub(to, from);
    let length = dot(flown, flown).sqrt();
    if length == 0.0 {
        return None;
    }
    let unit = [flown[0] / length, flown[1] / length, flown[2] / length];
    // Starting outside the capsule, it starts outside each part of it.
    let mut first = f64::INFINITY;
    for centre in [a, b] {
        let offset = sub(from, centre);
        let half_b = dot(unit, offset);
        let c = dot(offset, offset) - radius * radius;
        let h = half_b * half_b - c;
        if h >= 0.0 {
            let s = -half_b - h.sqrt();
            if (0.0..=length).contains(&s) {
                first = first.min(s);
            }
        }
    }
    let axis = sub(b, a);
    let axis_sq = dot(axis, axis);
    if axis_sq > 0.0 {
        let offset = sub(from, a);
        let along_unit = dot(axis, unit);
        let along_offset = dot(axis, offset);
        let qa = axis_sq - along_unit * along_unit;
        if qa > 1e-12 * axis_sq {
            let qb = axis_sq * dot(unit, offset) - along_offset * along_unit;
            let qc = axis_sq * dot(offset, offset)
                - along_offset * along_offset
                - radius * radius * axis_sq;
            let h = qb * qb - qa * qc;
            if h >= 0.0 {
                let s = (-qb - h.sqrt()) / qa;
                let y = along_offset + s * along_unit;
                if (0.0..=length).contains(&s) && (0.0..=axis_sq).contains(&y) {
                    first = first.min(s);
                }
            }
        }
    }
    first.is_finite().then(|| first / length)
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

    /// The lowest point is the parabola's, wherever it falls: inside the one
    /// piece of a short span, or off-centre on a span between supports of
    /// different heights, or at the lower support when the vertex lies
    /// beyond it.
    #[test]
    fn the_lowest_point_is_the_parabolas_not_a_piece_ends() {
        let mut short = line();
        short.supports[1].xy_m = [4.0, 0.0];
        assert_eq!(short.pieces().len(), 1);
        assert!(close(short.lowest_conductor_z_m(), 20.0 - 0.03 * 4.0));

        // 100 m span, 3 m sag, the far support 4 m higher: lowest where
        // 4 - 12 + 24 t = 0, t = 1/3, z = 20 + 4/3 - 12·(1/3)(2/3) = 18.666...
        let mut uneven = line();
        uneven.supports[1].height_m = 24.0;
        assert!(close(
            uneven.lowest_conductor_z_m(),
            20.0 + 4.0 / 3.0 - 8.0 / 3.0
        ));

        // A rise steeper than the sag pulls the vertex past the near support.
        let mut steep = line();
        steep.supports[1].height_m = 40.0;
        assert!(close(steep.lowest_conductor_z_m(), 20.0));

        // No sag at all: the lower support.
        let mut taut = line();
        taut.sag_ratio = 0.0;
        taut.supports[0].height_m = 30.0;
        assert!(close(taut.lowest_conductor_z_m(), 20.0));
    }

    /// At a turn both spans' conductors meet on one cross-arm, along the
    /// bisector, so the wire is continuous through the support: a flight
    /// between the two spans' ends is not passed off as clear.
    #[test]
    fn offset_conductors_are_continuous_through_a_turn() {
        let mut turning = line();
        turning.supports.push(Support {
            xy_m: [100.0, 100.0],
            base_z_m: 0.0,
            height_m: 20.0,
        });
        turning.conductor_offsets_m = vec![-3.0, 3.0];
        let pieces = turning.pieces();
        // Per span: 20 pieces for each of two conductors.
        assert_eq!(pieces.len(), 80);
        for conductor in 0..2 {
            let into_turn = pieces[conductor * 20 + 19].end_m;
            let out_of_turn = pieces[40 + conductor * 20].start_m;
            assert!(
                segment_distance(into_turn, into_turn, out_of_turn, out_of_turn) < 1e-9,
                "conductor {conductor}: {into_turn:?} vs {out_of_turn:?}"
            );
        }
        // The arm at the turn bisects the 90° corner: 3 m along it.
        let r = 3.0 / 2f64.sqrt();
        // The -3 m conductor leaves the turn on the second span's first
        // piece; travelling +x then +y, left at the turn is (-1, 1)/√2.
        let at_turn = pieces[40].start_m;
        assert!((at_turn[0] - (100.0 + r)).abs() < 1e-9, "{at_turn:?}");
        assert!((at_turn[1] - (-r)).abs() < 1e-9, "{at_turn:?}");
        // Where the viewer hangs it: the support plus the offset along the
        // cross-arm, square to the span at the ends.
        let arm = turning.cross_arm(1);
        assert!((arm[0] + 0.5f64.sqrt()).abs() < 1e-12 && (arm[1] - 0.5f64.sqrt()).abs() < 1e-12);
        assert!((at_turn[0] - (100.0 - 3.0 * arm[0])).abs() < 1e-9);
        assert!((at_turn[1] - (0.0 - 3.0 * arm[1])).abs() < 1e-9);
        assert_eq!(turning.cross_arm(0), [-0.0, 1.0]);
        assert_eq!(turning.cross_arm(2), [-1.0, 0.0]);
        // A flight straight down through the support's attachment point is
        // caught on both conductors' joins.
        let joint = pieces[19].end_m;
        assert!(turning.violated_by(
            [joint[0], joint[1], joint[2] + 1.0],
            [joint[0], joint[1], joint[2] - 1.0]
        ));
    }

    /// A line is read only if the model accepts it: one with no conductors
    /// would otherwise clear every flight.
    #[test]
    fn a_line_is_checked_as_it_is_read() {
        let good = serde_json::to_value(line()).unwrap();
        let read: PowerLine = serde_json::from_value(good.clone()).unwrap();
        assert_eq!(read, line());
        let mut no_conductors = good.clone();
        no_conductors["conductor_offsets_m"] = serde_json::json!([]);
        let error = serde_json::from_value::<PowerLine>(no_conductors).unwrap_err();
        assert!(error.to_string().contains("conductor"), "{error}");
        let mut one_support = good;
        one_support["supports"].as_array_mut().unwrap().pop();
        assert!(serde_json::from_value::<PowerLine>(one_support).is_err());
    }

    /// The lowest a wire hangs over the ground is solved on the terrain's
    /// cells: over a 30 m hump at mid-span (the wire there is at 17 m) it is
    /// 13 m under, as dense sampling finds, and never more than sampling
    /// says.
    #[test]
    fn the_lowest_wire_over_the_ground_is_solved() {
        use crate::tiled::{Tile, TiledTerrain};
        let heights_m = (0..5)
            .map(|_| {
                (0..21)
                    .map(|col| if col == 10 { 30.0 } else { 0.0 })
                    .collect()
            })
            .collect();
        let tile = Tile {
            tx: 0,
            ty: 0,
            lod: 0,
            x_min_m: 0.0,
            y_min_m: -10.0,
            cell_size_m: 5.0,
            heights_m,
            min_pyramid: Vec::new(),
            max_pyramid: Vec::new(),
        };
        let terrain = TiledTerrain::new(
            Bounds2d::new(0.0, 100.0, -10.0, 10.0),
            20,
            5.0,
            0.0,
            vec![tile],
        )
        .unwrap();
        let line = line();
        let solved = line.lowest_over_ground(0, &terrain);
        let sampled = (0..=100_000)
            .map(|k| {
                let t = k as f64 / 100_000.0;
                let p = line.conductor_at(0, 0.0, t);
                p[2] - terrain.height_lod(p[0], p[1], 0)
            })
            .fold(f64::INFINITY, f64::min);
        assert!((solved - sampled).abs() < 1e-3, "{solved} vs {sampled}");
        assert!(solved <= sampled + 1e-9);
        assert!((solved + 13.0).abs() < 1e-6, "{solved}");
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
        // 8 m over the lowest point, less the radius. Reported short of that
        // by the pieces' sag error (7.5 mm), counted against it, and by the
        // pieces either side of mid-span tilting toward the flight (a
        // fraction of a millimetre), never over it.
        let truth = 25.0 - 17.0 - 0.01;
        assert!(over(25.0) <= truth + 1e-9, "{}", over(25.0));
        assert!(over(25.0) >= truth - 0.0075 - 1e-3, "{}", over(25.0));
        assert!(!line.violated_by([50.0, -30.0, 25.0], [50.0, 30.0, 25.0]));
        assert!(line.violated_by([50.0, -30.0, 21.0], [50.0, 30.0, 21.0]));
        // Near a support the wire is higher.
        assert!(line.violated_by([1.0, -30.0, 24.0], [1.0, 30.0, 24.0]));
    }

    /// Under the wire between two piece ends the chord sits above the
    /// parabola, so the clearance must not be read off the chord: it is
    /// never more than the clearance from the true curve, and short of it by
    /// no more than the piece's sag error.
    #[test]
    fn clearance_under_the_wire_is_never_more_than_the_true_curves() {
        let line = line();
        let piece_error = 0.03 * 5.0 * 5.0 / 100.0;
        assert!(line
            .pieces()
            .iter()
            .all(|p| close(p.sag_error_m, piece_error)));
        // The true conductor, sampled densely.
        let true_distance = |p: [f64; 3]| {
            (0..=100_000)
                .map(|k| {
                    let t = k as f64 / 100_000.0;
                    let wire = [100.0 * t, 0.0, 20.0 - 12.0 * t * (1.0 - t)];
                    segment_distance(p, p, wire, wire)
                })
                .fold(f64::INFINITY, f64::min)
        };
        // Between the piece ends at 50 m and 55 m, 5 m under the curve.
        for x in [51.25, 52.5, 53.75] {
            let t = x / 100.0;
            let p = [x, 0.0, 20.0 - 12.0 * t * (1.0 - t) - 5.0];
            let truth = true_distance(p) - 0.01;
            let reported = line.clearance_along(p, p);
            assert!(reported <= truth + 1e-6, "{x}: {reported} > {truth}");
            assert!(reported >= truth - piece_error - 1e-6, "{x}: {reported}");
        }
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

    /// The builder's single wire with no sag: a straight capsule of radius
    /// 0.01 + `margin`, along x from 0 to 100 at 20 m.
    fn taut() -> PowerLine {
        let mut taut = line();
        taut.sag_ratio = 0.0;
        taut
    }

    /// Entering the envelope is solved: across the wire, straight down onto
    /// it, along it into its end, and at an angle into the end's sphere, each
    /// where the geometry says; from inside it is at once; over it, never.
    #[test]
    fn a_flight_enters_a_wires_envelope_where_it_is_solved() {
        let envelope = taut().envelope(1.0);
        let r = 1.01;
        let entry = |from, to| envelope.first_entry(from, to);
        let close = |a: Option<f64>, b: f64| a.is_some_and(|a| (a - b).abs() < 1e-12);
        // Across, level: in at y = -r.
        assert!(close(
            entry([50.0, -10.0, 20.0], [50.0, 10.0, 20.0]),
            (10.0 - r) / 20.0
        ));
        // Down onto it: in at z = 20 + r.
        assert!(close(
            entry([50.0, 0.0, 30.0], [50.0, 0.0, 10.0]),
            (10.0 - r) / 20.0
        ));
        // Along its axis from beyond its end: in at x = -r.
        assert!(close(
            entry([-10.0, 0.0, 20.0], [10.0, 0.0, 20.0]),
            (10.0 - r) / 20.0
        ));
        // At 45° up into its end: in r short of the end point.
        let length = 5.0 * 2f64.sqrt();
        assert!(close(
            entry([-5.0, 0.0, 15.0], [0.0, 0.0, 20.0]),
            1.0 - r / length
        ));
        // Starting inside.
        assert_eq!(entry([50.0, 0.5, 20.0], [50.0, 10.0, 20.0]), Some(0.0));
        // Over it with 0.5 m to spare, and stopping short of it.
        assert_eq!(entry([50.0, -10.0, 21.51], [50.0, 10.0, 21.51]), None);
        assert_eq!(entry([50.0, -10.0, 20.0], [50.0, -1.02, 20.0]), None);
        // Nearly along it, closing on it at 1 in 40, into its side.
        assert!(close(
            capsule_entry(
                [10.0, 2.0, 20.0],
                [90.0, 0.0, 20.0],
                [0.0, 0.0, 20.0],
                [100.0, 0.0, 20.0],
                r
            ),
            (2.0 - r) / 2.0
        ));
        // Far away: not even near its box.
        assert_eq!(entry([500.0, 0.0, 20.0], [600.0, 0.0, 20.0]), None);
    }

    /// Against dense sampling, on segments thrown at a tilted capsule from
    /// every side: the solved entry is on the surface, nothing before it is
    /// inside, and a segment that samples inside is never solved as a miss.
    #[test]
    fn capsule_entry_agrees_with_dense_sampling() {
        let (a, b, radius) = ([0.0, 0.0, 0.0], [4.0, 1.0, 2.0], 1.3);
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 * 12.0 - 4.0
        };
        let at = |from: [f64; 3], to: [f64; 3], t: f64| {
            [
                from[0] + (to[0] - from[0]) * t,
                from[1] + (to[1] - from[1]) * t,
                from[2] + (to[2] - from[2]) * t,
            ]
        };
        let mut hits = 0;
        for _ in 0..2000 {
            let from = [next(), next(), next()];
            let to = [next(), next(), next()];
            let gap = |t: f64| {
                let p = at(from, to, t);
                segment_distance(p, p, a, b)
            };
            let solved = capsule_entry(from, to, a, b, radius);
            let samples = 4000;
            let first_inside = (0..=samples)
                .map(|k| k as f64 / samples as f64)
                .find(|&t| gap(t) < radius);
            match solved {
                Some(t) => {
                    hits += 1;
                    assert!(t > 0.0 || gap(0.0) < radius);
                    if t > 0.0 {
                        assert!((gap(t) - radius).abs() < 1e-9, "{from:?} {to:?} {t}");
                    }
                    let inside = first_inside.expect("sampling finds it too");
                    // Sampling finds it no earlier than the solve, and within
                    // a sample of it unless it only grazes.
                    assert!(inside >= t - 1e-12, "{from:?} {to:?} {t} vs {inside}");
                }
                None => assert!(
                    first_inside.is_none(),
                    "{from:?} {to:?} missed at {first_inside:?}"
                ),
            }
        }
        assert!(hits > 200, "the fixture hits the capsule often: {hits}");
    }

    /// The pieces are chords of a sagging wire: the envelope holds everything
    /// within the margin of the true curve, and is wider by about twice a
    /// piece's sag error at most.
    #[test]
    fn the_envelope_holds_the_true_sagging_wire() {
        let line = line();
        let (margin, r) = (1.0, 0.01);
        let envelope = line.envelope(margin);
        let error = 0.03 * 100.0 / 400.0;
        // The true wire over x, and its surface plus margin under and over it.
        let wire_z = |x: f64| {
            let t = x / 100.0;
            20.0 - 12.0 * t * (1.0 - t)
        };
        for x in [2.5, 31.25, 50.0, 52.5, 97.5] {
            let reach = r + margin - 1e-6;
            assert!(envelope.contains([x, 0.0, wire_z(x) - reach]), "{x} under");
            assert!(envelope.contains([x, 0.0, wire_z(x) + reach]), "{x} over");
            // Straight down onto it, met no lower than its true top.
            let t = envelope
                .first_entry([x, 0.0, 40.0], [x, 0.0, 0.0])
                .expect("met");
            let met = 40.0 - 40.0 * t;
            // The true top over x: the highest point within the reach of
            // any point of the true wire, sampled densely along it.
            let reach = r + margin;
            let truth = (0..=200_000)
                .map(|k| x - reach + 2.0 * reach * k as f64 / 200_000.0)
                .map(|xs| wire_z(xs) + (reach * reach - (x - xs).powi(2)).max(0.0).sqrt())
                .fold(f64::NEG_INFINITY, f64::max);
            assert!(met >= truth - 1e-7, "{x}: {met} < {truth}");
            // Wider by the chord's height over the curve and the capsule's
            // extra radius, each at most the sag error, the radius read
            // vertically through a wire sloping at most 0.12 (1/cos < 1.01).
            assert!(
                met <= truth + 2.0 * error * 1.01 + 1e-7,
                "{x}: {met} vs {truth}"
            );
        }
    }

    /// A point inside is pushed straight away from the wire to just past the
    /// surface; one on the wire itself is pushed up. Raised, it goes to the
    /// lowest height clear above it.
    #[test]
    fn a_point_is_pushed_out_of_the_envelope_and_lifted_over_it() {
        let envelope = taut().envelope(1.0);
        let r = 1.01;
        let under = envelope.push_out([50.0, 0.0, 19.5]).expect("inside");
        assert!(
            (under[2] - (20.0 - r)).abs() < 1e-5 && under[0] == 50.0,
            "{under:?}"
        );
        assert!(!envelope.contains(under));
        let side = envelope.push_out([30.0, 0.6, 20.8]).expect("inside");
        let d = segment_distance(side, side, [0.0, 0.0, 20.0], [100.0, 0.0, 20.0]);
        assert!((d - r).abs() < 1e-5, "{d}");
        // The push is along the line from the wire: same direction.
        assert!(
            (side[1] / (side[2] - 20.0) - 0.6 / 0.8).abs() < 1e-9,
            "{side:?}"
        );
        let on = envelope.push_out([50.0, 0.0, 20.0]).expect("inside");
        assert!(
            (on[2] - (20.0 + r)).abs() < 1e-5 && on[1].abs() < 1e-12,
            "{on:?}"
        );
        assert_eq!(envelope.push_out([50.0, 0.0, 22.0]), None);

        let lifted = envelope.lift_out([50.0, 0.5, 19.5]).expect("inside");
        let top = 20.0 + (r * r - 0.25f64).sqrt();
        assert!((lifted[2] - top).abs() < 1e-5, "{lifted:?} vs {top}");
        assert_eq!([lifted[0], lifted[1]], [50.0, 0.5]);
        assert!(!envelope.contains(lifted));
        assert_eq!(envelope.lift_out([50.0, 1.5, 19.5]), None);
    }

    /// Where two pieces meet at an angle, a push out of one can land in the
    /// next; pushing again settles it within a few pushes.
    #[test]
    fn pushes_settle_where_pieces_meet() {
        let envelope = line().envelope(1.0);
        for x in [5.0, 50.0, 95.0] {
            let wire = line().pieces()[(x / 5.0) as usize - 1].end_m;
            let mut p = [x, 0.0, wire[2] - 0.2];
            let mut pushes = 0;
            while let Some(out) = envelope.push_out(p) {
                p = out;
                pushes += 1;
                assert!(pushes < 4, "{x}: {p:?}");
            }
            assert!(!envelope.contains(p), "{x}");
        }
    }

    /// The pieces span by span are the pieces in order, and a span alone is
    /// that span: on a straight line its pieces are exactly the line's.
    #[test]
    fn a_span_alone_hangs_as_it_does_in_its_line() {
        let mut long = line();
        long.supports.push(Support {
            xy_m: [250.0, 0.0],
            base_z_m: 5.0,
            height_m: 25.0,
        });
        long.conductor_offsets_m = vec![-2.0, 2.0];
        let by_span = long.pieces_by_span();
        assert_eq!(by_span.len(), 2);
        assert_eq!(by_span.concat(), long.pieces());
        for (k, pieces) in by_span.iter().enumerate() {
            let alone = long.span_line(k);
            assert_eq!(alone.id, format!("line-a#{k}"));
            assert_eq!(&alone.pieces(), pieces, "span {k}");
            assert!(alone.clone().validated().is_ok());
        }
    }
}
